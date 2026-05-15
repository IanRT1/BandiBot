"""
music.py

Voice + music playback for BandiBot.

Per-guild state is held in VoiceManager (singleton via module-level instance).
Tracks are resolved with yt-dlp (URL or search query), played via FFmpeg,
and mixed via MixerSource so TTS can be injected over music without
pausing or restarting.

Track resolution:
  Single tracks  → resolved immediately before queuing
  Bulk / playlist → queued instantly as placeholders, resolved one ahead
                    in the background as songs play

State machine (per guild):
  idle      → no voice connection, idle timer running
  connected → voice client active, queue may be empty or populated
  playing   → FFmpeg streaming via MixerSource, resolver pre-loading next

Bulk queue flow:
  Placeholder tracks enter the queue with resolved=False. The background
  resolver picks up the next unresolved track when a song starts playing,
  resolving it while the current song is audible. Errors are stored on the
  track and announced at playback time, never during resolution.

Per-guild play lock prevents duplicate queuing from concurrent requests.
Idle timeout disconnects the bot after 5 minutes of silence.
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

_YDL_OPTS_FLAT = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": True,
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
    resolved: bool = True
    query: Optional[str] = None
    error: Optional[str] = None


class GuildPlayer:

    def __init__(self, guild: discord.Guild):
        self.guild = guild
        self.queue: deque[Track] = deque()
        self.current: Optional[Track] = None
        self.voice_client: Optional[discord.VoiceClient] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._resolver_task: Optional[asyncio.Task] = None
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
        if self._resolver_task:
            self._resolver_task.cancel()
            self._resolver_task = None
        if self._now_playing_view:
            await self._now_playing_view.stop_updates()
            self._now_playing_view = None
        if self.is_connected:
            await self.voice_client.disconnect()
        self.voice_client = None
        self.current = None
        self.queue.clear()
        self.now_playing_message = None

    def start_resolver(self):
        """Start the background resolver loop if not already running."""
        if self._resolver_task and not self._resolver_task.done():
            return
        self._resolver_task = asyncio.create_task(self._resolver_loop())

    async def _resolver_loop(self):
        """Resolves only the next unresolved track in the queue, then stops."""
        try:
            next_unresolved = None
            for track in self.queue:
                if not track.resolved:
                    next_unresolved = track
                    break

            if not next_unresolved:
                logger.info("[music] resolver: nothing to resolve, stopping")
                return

            logger.info(f"[music] resolver: pre-resolving {next_unresolved.title!r}")
            try:
                resolved = await asyncio.to_thread(
                    _resolve_track, next_unresolved.query, next_unresolved.requested_by
                )
                next_unresolved.title = resolved.title
                next_unresolved.stream_url = resolved.stream_url
                next_unresolved.webpage_url = resolved.webpage_url
                next_unresolved.duration = resolved.duration
                next_unresolved.thumbnail = resolved.thumbnail
                next_unresolved.artist = resolved.artist
                next_unresolved.resolved = True
                next_unresolved.error = None
                logger.info(f"[music] resolver: resolved → {resolved.title!r}")
            except Exception as e:
                next_unresolved.resolved = True
                next_unresolved.error = str(e)
                logger.error(f"[music] resolver: failed to resolve {next_unresolved.title!r}: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[music] resolver loop error: {e}")

    def play_next(self):
            """Pop the next track and start playing it via MixerSource."""
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

            # If track has an error, notify and skip to next
            if track.error:
                logger.warning(f"[music] skipping errored track {track.title!r}: {track.error}")
                loop = self.voice_client.client.loop
                if self.text_channel:
                    asyncio.run_coroutine_threadsafe(
                        self.text_channel.send(f"⚠️ Could not load **{track.title}** — skipping."),
                        loop,
                    )
                loop.call_soon_threadsafe(self.play_next)
                return

            # If track is not yet resolved, resolve it directly
            if not track.resolved:
                logger.info(f"[music] waiting for resolution of {track.title!r}")
                loop = self.voice_client.client.loop

                async def _wait_and_play():
                    try:
                        resolved = await asyncio.to_thread(
                            _resolve_track, track.query, track.requested_by
                        )
                        track.title = resolved.title
                        track.stream_url = resolved.stream_url
                        track.webpage_url = resolved.webpage_url
                        track.duration = resolved.duration
                        track.thumbnail = resolved.thumbnail
                        track.artist = resolved.artist
                        track.resolved = True
                        track.error = None
                    except Exception as e:
                        track.resolved = True
                        track.error = str(e)
                    loop.call_soon_threadsafe(lambda: self._play_resolved(track))

                asyncio.run_coroutine_threadsafe(_wait_and_play(), loop)
                return

            self._play_resolved(track)

    def _play_resolved(self, track: "Track"):
        """Actually start playing a resolved track."""
        from tts import MixerSource

        if not self.is_connected:
            return

        if track.error:
            loop = self.voice_client.client.loop
            if self.text_channel:
                asyncio.run_coroutine_threadsafe(
                    self.text_channel.send(f"⚠️ Could not load **{track.title}** — skipping."),
                    loop,
                )
            loop.call_soon_threadsafe(self.play_next)
            return

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

        # Start pre-resolving next track
        self.start_resolver()

        if self._now_playing_view:
            asyncio.run_coroutine_threadsafe(
                self._now_playing_view.on_track_changed(track, len(self.queue)),
                loop,
            )
        elif self.text_channel:
            asyncio.run_coroutine_threadsafe(
                _post_now_playing_for_track(self, track),
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

    async def queue_bulk(self, guild, requester_member, queries: list[str]) -> str:
        """Queue multiple songs as placeholders and resolve them in background."""
        if not requester_member.voice or not requester_member.voice.channel:
            return "User is not in a voice channel; cannot play music."

        voice_channel = requester_member.voice.channel
        player = self.get_player(guild)

        async with self._get_play_lock(guild.id):
            await player.connect(voice_channel)

            for query in queries:
                track = Track(
                    title=query,
                    stream_url="",
                    requested_by=requester_member.display_name,
                    webpage_url="",
                    resolved=False,
                    query=query,
                )
                player.queue.append(track)

            is_busy = player.is_playing or (player.is_connected and player.voice_client.is_paused())

            player.start_resolver()

            if not is_busy:
                player.play_next()

            return f"Added {len(queries)} songs to queue."

    async def queue_playlist(self, guild, requester_member, url: str) -> str:
        """Extract playlist entries and queue them as placeholders."""
        if not requester_member.voice or not requester_member.voice.channel:
            return "User is not in a voice channel; cannot play music."

        voice_channel = requester_member.voice.channel
        player = self.get_player(guild)

        try:
            entries = await asyncio.to_thread(_extract_playlist, url, requester_member.display_name)
        except Exception as e:
            logger.error(f"[music] playlist extract failed for {url!r}: {e}")
            return f"Could not load playlist: {e}"

        if not entries:
            return "No playable tracks found in playlist."

        async with self._get_play_lock(guild.id):
            await player.connect(voice_channel)

            for track in entries:
                player.queue.append(track)

            is_busy = player.is_playing or (player.is_connected and player.voice_client.is_paused())

            player.start_resolver()

            if not is_busy:
                player.play_next()

            return f"Added {len(entries)} songs from playlist to queue."

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
        if player._resolver_task:
            player._resolver_task.cancel()
            player._resolver_task = None
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
        player.start_resolver()
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


def _extract_playlist(url: str, requested_by: str) -> list[Track]:
    """Extract playlist entries as unresolved placeholder tracks."""
    # Normalize to pure playlist URL
    if "list=" in url:
        import re
        match = re.search(r"list=([A-Za-z0-9_-]+)", url)
        if match:
            url = f"https://www.youtube.com/playlist?list={match.group(1)}"

    logger.info(f"  [yt-dlp] extracting playlist {url!r}")
    t = time.time()

    with yt_dlp.YoutubeDL(_YDL_OPTS_FLAT) as ydl:
        info = ydl.extract_info(url, download=False)

    logger.info(f"  [yt-dlp] playlist extracted in {time.time() - t:.2f}s")

    entries = info.get("entries", [])
    tracks = []
    for entry in entries:
        if not entry:
            continue
        title = entry.get("title") or entry.get("id") or "Unknown"
        entry_url = entry.get("url") or entry.get("webpage_url") or url
        tracks.append(Track(
            title=title,
            stream_url="",
            requested_by=requested_by,
            webpage_url=entry_url,
            resolved=False,
            query=entry_url,
        ))

    logger.info(f"  [yt-dlp] found {len(tracks)} playlist entries")
    return tracks

async def _post_now_playing_for_track(player, track):
    """Post a now playing banner for a track when none exists yet."""
    if not player.text_channel:
        return
    from now_playing_view import post_now_playing
    await post_now_playing(
        player.text_channel,
        player,
        title=track.title,
        artist=track.artist,
        duration_seconds=track.duration,
        queue_size=len(player.queue),
        requested_by=track.requested_by,
        thumbnail_url=track.thumbnail,
    )


voice_manager = VoiceManager()