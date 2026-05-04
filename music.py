"""Voice + music playback for BandiBot.

Per-guild state is held in VoiceManager (singleton via module-level instance).
Tracks are resolved with yt-dlp (URL or search query), played via FFmpeg.
After 60s of idle/empty VC the bot auto-disconnects.

All public methods return human-readable strings — these are surfaced back
to the LLM as tool results, which the LLM then uses to compose its chat reply.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import discord
import yt_dlp

logger = logging.getLogger(__name__)

# yt-dlp config — extract audio stream URL only, no download to disk
_YDL_OPTS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
    "cookiefile": "youtube_cookies.txt",
}

# FFmpeg options — reconnect on transient network blips, no video
_FFMPEG_BEFORE = (
    "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
)
_FFMPEG_OPTS = "-vn"

# Auto-disconnect after this many seconds of nothing playing or empty VC
_IDLE_TIMEOUT = 60


@dataclass
class Track:
    """A queued or playing track."""
    title: str
    stream_url: str
    requested_by: str       # display name of whoever asked for it
    webpage_url: str        # the original YouTube page (for display only)


class GuildPlayer:
    """Per-guild voice state. Owns the queue and voice client lifecycle."""

    def __init__(self, guild: discord.Guild):
        self.guild = guild
        self.queue: deque[Track] = deque()
        self.current: Optional[Track] = None
        self.voice_client: Optional[discord.VoiceClient] = None
        self._idle_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        return self.voice_client is not None and self.voice_client.is_connected()

    @property
    def is_playing(self) -> bool:
        return self.is_connected and self.voice_client.is_playing()

    async def connect(self, voice_channel: discord.VoiceChannel):
        """Connect to (or move to) the given voice channel."""
        if self.is_connected:
            if self.voice_client.channel.id != voice_channel.id:
                await self.voice_client.move_to(voice_channel)
        else:
            self.voice_client = await voice_channel.connect()

    async def disconnect(self):
        """Tear down voice connection and clear state."""
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
        if self.is_connected:
            await self.voice_client.disconnect()
        self.voice_client = None
        self.current = None
        self.queue.clear()

    def play_next(self):
        """Pop the next track and start playing it. Called by the after-callback."""
        if not self.queue or not self.is_connected:
            self.current = None
            self._schedule_idle_check()
            return

        track = self.queue.popleft()
        self.current = track

        source = discord.FFmpegPCMAudio(
            track.stream_url,
            before_options=_FFMPEG_BEFORE,
            options=_FFMPEG_OPTS,
        )

        # The after callback runs in a thread, not the event loop. We schedule
        # play_next back onto the loop using the bot's loop reference.
        loop = self.voice_client.client.loop

        def _after(error):
            if error:
                logger.error(f"Playback error in {self.guild.name}: {error}")
            loop.call_soon_threadsafe(self.play_next)

        self.voice_client.play(source, after=_after)
        logger.info(f"[music] now playing in {self.guild.name}: {track.title}")

    def _schedule_idle_check(self):
        """Start a timer; if still idle after _IDLE_TIMEOUT, disconnect."""
        if self._idle_task:
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_disconnect())

    async def _idle_disconnect(self):
        """Disconnect after _IDLE_TIMEOUT seconds if still idle or VC empty."""
        try:
            await asyncio.sleep(_IDLE_TIMEOUT)
        except asyncio.CancelledError:
            return

        if not self.is_connected:
            return
        if self.is_playing:
            return  # something started playing; cancel disconnect

        logger.info(f"[music] idle timeout reached in {self.guild.name}, leaving VC")
        await self.disconnect()


class VoiceManager:
    """One per bot — holds GuildPlayer instances keyed by guild ID."""

    def __init__(self):
        self._players: dict[int, GuildPlayer] = {}

    def get_player(self, guild: discord.Guild) -> GuildPlayer:
        if guild.id not in self._players:
            self._players[guild.id] = GuildPlayer(guild)
        return self._players[guild.id]

    # ---- Tool-callable methods ----
    # Each returns a string suitable for feeding back to the LLM as the
    # tool-call result. The LLM uses these to compose its chat reply.

    async def play(self, guild, requester_member, query: str) -> str:
        """Resolve query/URL, queue the track, start playback if idle."""
        if not requester_member.voice or not requester_member.voice.channel:
            return "User is not in a voice channel; cannot play music."

        voice_channel = requester_member.voice.channel
        player = self.get_player(guild)

        try:
            track = await asyncio.to_thread(_resolve_track, query, requester_member.display_name)
        except Exception as e:
            logger.error(f"[music] resolve failed for {query!r}: {e}")
            return f"Could not resolve track: {e}"

        await player.connect(voice_channel)
        player.queue.append(track)

        if not player.is_playing:
            player.play_next()
            return f"Now playing: {track.title}"
        else:
            position = len(player.queue)
            return f"Queued at position {position}: {track.title}"

    async def skip(self, guild) -> str:
        player = self.get_player(guild)
        if not player.is_playing:
            return "Nothing is playing."
        skipped = player.current.title if player.current else "current track"
        player.voice_client.stop()  # triggers the after-callback → play_next
        return f"Skipped: {skipped}"

    async def pause(self, guild) -> str:
        player = self.get_player(guild)
        if not player.is_playing:
            return "Nothing is playing."
        player.voice_client.pause()
        return "Paused."

    async def resume(self, guild) -> str:
        player = self.get_player(guild)
        if not player.is_connected or not player.voice_client.is_paused():
            return "Nothing is paused."
        player.voice_client.resume()
        return "Resumed."

    async def stop(self, guild) -> str:
        player = self.get_player(guild)
        if not player.is_connected:
            return "Bot is not in a voice channel."
        player.queue.clear()
        if player.is_playing:
            player.voice_client.stop()
        return "Stopped and cleared the queue."

    async def leave(self, guild) -> str:
        player = self.get_player(guild)
        if not player.is_connected:
            return "Bot is not in a voice channel."
        await player.disconnect()
        return "Left the voice channel."

    async def now_playing(self, guild) -> str:
        player = self.get_player(guild)
        if not player.current:
            return "Nothing is playing."
        return f"Now playing: {player.current.title} (requested by {player.current.requested_by})"

    async def get_queue(self, guild) -> str:
        player = self.get_player(guild)
        if not player.queue and not player.current:
            return "Queue is empty."
        lines = []
        if player.current:
            lines.append(f"Now: {player.current.title}")
        for i, track in enumerate(player.queue, start=1):
            lines.append(f"{i}. {track.title}")
        return "\n".join(lines)


def _resolve_track(query: str, requested_by: str) -> Track:
    """Run yt-dlp synchronously (called via asyncio.to_thread).

    Accepts either a URL or a free-text search query.
    """
    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        info = ydl.extract_info(query, download=False)

    # If it was a search query, yt-dlp returns a 'playlist' (list of results)
    if "entries" in info:
        info = info["entries"][0]

    return Track(
        title=info.get("title", "Unknown title"),
        stream_url=info["url"],
        requested_by=requested_by,
        webpage_url=info.get("webpage_url", ""),
    )


# Module-level singleton. Imported by handlers.py.
voice_manager = VoiceManager()