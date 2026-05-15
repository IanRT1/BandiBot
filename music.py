"""Voice + music playback for BandiBot.

Per-guild state is held in VoiceManager (singleton via module-level instance).
Tracks are resolved with yt-dlp (URL or search query), played via FFmpeg.
After 60s of idle/empty VC the bot auto-disconnects.

Uses MixerSource from tts.py so TTS can be injected over music without
pausing or restarting.

Per-guild play lock prevents duplicate queuing from concurrent requests.
"""

import asyncio
import logging
import time
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from discord.ext import voice_recv

import discord
import yt_dlp

logger = logging.getLogger(__name__)

_YDL_OPTS = {
    "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best[acodec!=none]/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
    "ignoreerrors": True,
}

DEFAULT_VOLUME = 0.30

_FFMPEG_BEFORE = (
    "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
)
_FFMPEG_OPTS = (
    "-vn -af loudnorm=I=-16:TP=-1.5:LRA=11"
)

_IDLE_TIMEOUT = 300


@dataclass
class Track:
    title: str
    stream_url: str
    requested_by: str
    webpage_url: str
    duration: Optional[int] = None
    thumbnail: Optional[str] = None
    artist: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    paused_at: Optional[float] = None
    total_paused: float = 0.0


class GuildPlayer:

    def __init__(self, guild: discord.Guild):
        self.guild = guild
        self.queue: deque[Track] = deque()
        self.current: Optional[Track] = None
        self.voice_client: Optional[discord.VoiceClient] = None
        self._idle_task: Optional[asyncio.Task] = None
        self.now_playing_message = None
        self._now_playing_view = None
        self._natural_transition = False
        self._manual_stop = False
        self._loop = False
        self.text_channel: Optional[discord.TextChannel] = None
        self.queue_empty_message = None

    @property
    def is_connected(self) -> bool:
        return self.voice_client is not None and self.voice_client.is_connected()

    @property
    def is_playing(self) -> bool:
        return self.is_connected and self.voice_client.is_playing()

    @property
    def elapsed_seconds(self) -> float:
        if not self.current:
            return 0.0
        elapsed = time.time() - self.current.started_at - self.current.total_paused
        if self.current.paused_at is not None:
            elapsed -= (time.time() - self.current.paused_at)
        return max(0.0, elapsed)

    async def connect(self, voice_channel: discord.VoiceChannel):
        existing = voice_channel.guild.voice_client
        if existing and existing.is_connected():
            self.voice_client = existing
            if existing.channel.id != voice_channel.id:
                await existing.move_to(voice_channel)
        elif self.is_connected:
            if self.voice_client.channel.id != voice_channel.id:
                await self.voice_client.move_to(voice_channel)
        else:
            self.voice_client = await voice_channel.connect(
                cls=voice_recv.VoiceRecvClient
            )

    async def disconnect(self):
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
        if self._now_playing_view:
            await self._now_playing_view.stop_updates()
            self._now_playing_view = None
        if self.is_connected:
            await self.voice_client.disconnect()
        self.voice_client = None
        self.current = None
        self.queue.clear()
        self.now_playing_message = None

    def play_next(self):
        """Pop the next track and start playing it via MixerSource."""
        from tts import MixerSource

        if not self.queue or not self.is_connected:
            self.current = None
            self._schedule_idle_check()
            if self._now_playing_view:
                loop = self.voice_client.client.loop if self.voice_client else asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self._now_playing_view.on_queue_empty(),
                    loop,
                )
            return

        track = self.queue.popleft()
        track.started_at = time.time()
        track.paused_at = None
        track.total_paused = 0.0
        self.current = track

        ffmpeg_source = discord.FFmpegPCMAudio(
            track.stream_url,
            before_options=_FFMPEG_BEFORE,
            options=_FFMPEG_OPTS,
        )
        volume_source = discord.PCMVolumeTransformer(ffmpeg_source, volume=DEFAULT_VOLUME)
        mixer_source  = MixerSource(volume_source)

        loop = self.voice_client.client.loop

        def _after(error):
            if error:
                logger.error(f"Playback error in {self.guild.name}: {error}")

            if self._loop and self.current:
                self.queue.appendleft(self.current)

            self._natural_transition = (
                not self._manual_stop and not self._loop
            )
            self._manual_stop = False
            loop.call_soon_threadsafe(self.play_next)

        self.voice_client.play(mixer_source, after=_after)
        logger.info(
            f"[music] play() called | "
            f"is_playing={self.voice_client.is_playing()} "
            f"is_connected={self.voice_client.is_connected()}"
        )
        logger.info(f"[music] now playing in {self.guild.name}: {track.title}")

        # Notify now playing view to update banner
        if self._now_playing_view:
            asyncio.run_coroutine_threadsafe(
                self._now_playing_view.on_track_changed(track, len(self.queue)),
                loop,
            )

    def _schedule_idle_check(self):
        if self._idle_task:
            self._idle_task.cancel()
        logger.info(f"[music] idle check scheduled in {_IDLE_TIMEOUT}s")
        self._idle_task = asyncio.create_task(self._idle_disconnect())

    async def _idle_disconnect(self):
        try:
            await asyncio.sleep(_IDLE_TIMEOUT)
        except asyncio.CancelledError:
            return
        if not self.is_connected:
            return
        if self.is_playing:
            return
        logger.info(f"[music] idle timeout reached in {self.guild.name}, leaving VC")
        await self.disconnect()


class VoiceManager:

    def __init__(self):
        self._players: dict[int, GuildPlayer] = {}
        self._play_locks: dict[int, asyncio.Lock] = {}

    def get_player(self, guild: discord.Guild) -> GuildPlayer:
        if guild.id not in self._players:
            self._players[guild.id] = GuildPlayer(guild)
        return self._players[guild.id]

    def _get_play_lock(self, guild_id: int) -> asyncio.Lock:
        if guild_id not in self._play_locks:
            self._play_locks[guild_id] = asyncio.Lock()
        return self._play_locks[guild_id]

    async def play(self, guild, requester_member, query: str) -> str:
        if not requester_member.voice or not requester_member.voice.channel:
            return "User is not in a voice channel; cannot play music."

        voice_channel = requester_member.voice.channel
        player = self.get_player(guild)

        try:
            track = await asyncio.to_thread(_resolve_track, query, requester_member.display_name)
        except Exception as e:
            logger.error(f"[music] resolve failed for {query!r}: {e}")
            return f"Could not resolve track: {e}"

        # Per-guild lock prevents duplicate queuing from concurrent requests
        async with self._get_play_lock(guild.id):
            await player.connect(voice_channel)
            player.queue.append(track)

            is_busy = player.is_playing or (player.is_connected and player.voice_client.is_paused())

            if not is_busy:
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
        player._manual_stop = True
        player.voice_client.stop_playing()
        return f"Skipped: {skipped}"

    async def restart(self, guild) -> str:
        player = self.get_player(guild)
        if not player.current:
            return "Nothing is playing."
        player._manual_stop = True
        player.queue.appendleft(player.current)
        player.voice_client.stop_playing()
        return f"Restarting: {player.current.title}"

    async def pause(self, guild) -> str:
        player = self.get_player(guild)
        if not player.is_playing:
            return "Nothing is playing."
        if player.current and player.current.paused_at is None:
            player.current.paused_at = time.time()
        player.voice_client.pause()
        return "Paused."

    async def resume(self, guild) -> str:
        player = self.get_player(guild)
        if not player.is_connected or not player.voice_client.is_paused():
            return "Nothing is paused."
        if player.current and player.current.paused_at is not None:
            player.current.total_paused += time.time() - player.current.paused_at
            player.current.paused_at = None
        player.voice_client.resume()
        return "Resumed."

    async def stop(self, guild) -> str:
        player = self.get_player(guild)
        if not player.is_connected:
            return "Bot is not in a voice channel."
        if player._now_playing_view:
            await player._now_playing_view.on_queue_empty()
        player._manual_stop = True
        player.queue.clear()
        if player.is_playing:
            player.voice_client.stop_playing()
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

    async def toggle_loop(self, guild) -> bool:
        player = self.get_player(guild)
        player._loop = not player._loop
        return player._loop

    async def shuffle(self, guild) -> str:
        player = self.get_player(guild)
        if len(player.queue) < 2:
            return "Not enough songs in queue to shuffle."
        queue_list = list(player.queue)
        random.shuffle(queue_list)
        player.queue = deque(queue_list)
        return f"Shuffled {len(queue_list)} songs."

    async def move_track(self, guild, from_pos: int, to_pos: int) -> str:
        player = self.get_player(guild)
        if not player.queue:
            return "Queue is empty."
        queue_list = list(player.queue)
        max_pos = len(queue_list)
        if from_pos < 1 or from_pos > max_pos:
            return f"Invalid position {from_pos}. Queue has {max_pos} songs."
        if to_pos < 1 or to_pos > max_pos:
            return f"Invalid position {to_pos}. Queue has {max_pos} songs."
        track = queue_list.pop(from_pos - 1)
        queue_list.insert(to_pos - 1, track)
        player.queue = deque(queue_list)
        return f"Moved '{track.title}' from position {from_pos} to {to_pos}."

    async def delete_track(self, guild, position: int) -> str:
        player = self.get_player(guild)
        if not player.queue:
            return "Queue is empty."
        queue_list = list(player.queue)
        max_pos = len(queue_list)
        if position < 1 or position > max_pos:
            return f"Invalid position {position}. Queue has {max_pos} songs."
        track = queue_list.pop(position - 1)
        player.queue = deque(queue_list)
        return f"Removed '{track.title}' from the queue."


def _resolve_track(query: str, requested_by: str) -> Track:
    if not query.startswith("http://") and not query.startswith("https://"):
        query = f"ytsearch5:{query}"

    logger.info(f"  [yt-dlp] starting resolution for {query!r}")
    t = time.time()

    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        info = ydl.extract_info(query, download=False)

    logger.info(f"  [yt-dlp] resolved in {time.time() - t:.2f}s")

    if "entries" in info:
        entry = next((e for e in info["entries"] if e), None)
        if not entry:
            raise Exception("No playable results found.")
        info = entry

    logger.info(f"  [yt-dlp] using: {info.get('title', '?')!r}")

    return Track(
        title=info.get("title", "Unknown title"),
        stream_url=info["url"],
        requested_by=requested_by,
        webpage_url=info.get("webpage_url", ""),
        duration=info.get("duration"),
        thumbnail=info.get("thumbnail"),
        artist=info.get("uploader") or info.get("channel"),
    )


voice_manager = VoiceManager()