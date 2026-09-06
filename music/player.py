"""
music/player.py

Voice + music playback for BandiBot.

Per-guild state is held in VoiceManager (singleton via module-level instance).
Tracks are resolved by music.resolver, played via FFmpeg, and mixed via
MixerSource so TTS can be injected over music without pausing or restarting.

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
import subprocess
from collections import deque
from typing import Optional
from discord.ext import voice_recv

import discord

from music.resolver import extract_playlist, resolve_track
from music.tracks import Track

logger = logging.getLogger(__name__)

DEFAULT_VOLUME = 0.30

_FFMPEG_BEFORE = (
    "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
)
_FFMPEG_OPTS = (
    "-vn -af loudnorm=I=-16:TP=-1.5:LRA=11"
)
_FFMPEG_BLOCKED_HEADER_KEYS = {"authorization"}

_IDLE_TIMEOUT = 300
_RESOLVE_TIMEOUT = 35
_STREAM_REFRESH_AFTER = 90


async def _resolve_track_async(
    query: str,
    requested_by: str,
    exclude_webpage_urls: set[str] | None = None,
) -> Track:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(resolve_track, query, requested_by, exclude_webpage_urls),
            timeout=_RESOLVE_TIMEOUT,
        )
    except asyncio.TimeoutError as exc:
        logger.error(f"[music] resolve timed out after {_RESOLVE_TIMEOUT}s for {query!r}")
        raise TimeoutError("YouTube search timed out.") from exc


def _copy_resolved_track(target: Track, resolved: Track):
    target.title = resolved.title
    target.stream_url = resolved.stream_url
    target.webpage_url = resolved.webpage_url
    target.duration = resolved.duration
    target.thumbnail = resolved.thumbnail
    target.thumbnail_bytes = resolved.thumbnail_bytes
    target.artist = resolved.artist
    target.http_headers = dict(resolved.http_headers)
    target.resolved = True
    target.resolved_at = time.time()
    target.query = resolved.query or target.query
    target.error = None


def _ffmpeg_before_options(track: Track) -> str:
    headers = getattr(track, "http_headers", None) or {}
    header_lines = [
        f"{key}: {value}"
        for key, value in headers.items()
        if value and key.lower() not in _FFMPEG_BLOCKED_HEADER_KEYS
    ]
    if not header_lines:
        return _FFMPEG_BEFORE

    header_blob = "\r\n".join(header_lines) + "\r\n"
    header_blob = header_blob.replace('"', r'\"')
    return f'{_FFMPEG_BEFORE} -headers "{header_blob}"'


def _read_ffmpeg_stderr(ffmpeg_source) -> str:
    process = getattr(ffmpeg_source, "_process", None)
    stderr = getattr(process, "stderr", None)
    if not stderr:
        return ""

    try:
        raw = stderr.read()
    except Exception as exc:
        return f"<failed to read ffmpeg stderr: {exc}>"

    if not raw:
        return ""

    text = raw.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 12:
        lines = lines[-12:]
    return " | ".join(lines)


class GuildPlayer:

    def __init__(self, guild: discord.Guild):
        self.guild = guild
        self.queue: deque[Track] = deque()
        self.current: Optional[Track] = None
        self.voice_client: Optional[discord.VoiceClient] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._resolver_task: Optional[asyncio.Task] = None
        self._start_when_free_task: Optional[asyncio.Task] = None
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
        return (
            self.is_connected
            and self.current is not None
            and self.voice_client.is_playing()
        )

    @property
    def has_active_track(self) -> bool:
        """Whether a track is owned by the player, including end transitions.

        Discord's voice client can briefly report ``is_playing()`` as false
        between FFmpeg ending and the ``after`` callback advancing the queue.
        ``current`` remains authoritative during that handoff.
        """
        return self.current is not None

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
        await self._ensure_voice_listener(voice_channel)

    async def disconnect(self):
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
        if self._resolver_task:
            self._resolver_task.cancel()
            self._resolver_task = None
        if self._start_when_free_task:
            self._start_when_free_task.cancel()
            self._start_when_free_task = None
        await self._clear_now_playing_messages()
        if self.is_connected:
            await self.voice_client.disconnect()
        self.voice_client = None
        self.current = None
        self.queue.clear()
        self.now_playing_message = None

    async def _clear_now_playing_messages(self):
        """Delete playback UI when the player leaves voice or is torn down."""
        view = self._now_playing_view
        if view:
            await view.stop_updates()

        messages = []
        if view and view.message:
            messages.append(view.message)
        if self.now_playing_message and self.now_playing_message not in messages:
            messages.append(self.now_playing_message)
        if self.queue_empty_message and self.queue_empty_message not in messages:
            messages.append(self.queue_empty_message)

        for msg in messages:
            try:
                await msg.delete()
            except discord.NotFound:
                pass
            except Exception as e:
                logger.error(f"[now_playing] failed to delete playback UI during disconnect: {e}")

        if messages:
            logger.info("[now_playing] cleared playback UI during disconnect")

        if view:
            view.message = None
        self._now_playing_view = None
        self.now_playing_message = None
        self.queue_empty_message = None

    async def _ensure_voice_listener(self, voice_channel: discord.VoiceChannel):
        """Start wake-word listening after a music-driven voice connection."""
        if not self.voice_client or not self.voice_client.is_connected():
            return
        try:
            from voice.listener import voice_listener_manager

            client = self.voice_client.client
            await voice_listener_manager.start_listening(
                self.guild,
                voice_channel,
                client,
                client.loop,
            )
        except Exception as e:
            logger.error(f"[voice] failed to start listener after music connect: {e}")

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
                logger.debug("[music] resolver: nothing to resolve, stopping")
                return

            logger.debug(f"[music] resolver: pre-resolving {next_unresolved.title!r}")
            try:
                resolved = await _resolve_track_async(
                    next_unresolved.query, next_unresolved.requested_by
                )
                _copy_resolved_track(next_unresolved, resolved)
                logger.debug(f"[music] resolver: resolved → {resolved.title!r}")
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

        if self._voice_busy_without_music():
            self._schedule_start_when_free()
            return

        track = self.queue.popleft()

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

        if not track.resolved or self._needs_stream_refresh(track):
            if track.resolved:
                age = time.time() - track.resolved_at
                logger.info(f"[music] refreshing stale stream for {track.title!r} age={age:.0f}s")
            logger.info(f"[music] waiting for resolution of {track.title!r}")
            loop = self.voice_client.client.loop

            async def _wait_and_play():
                try:
                    refresh_query = track.query or track.webpage_url or track.title
                    resolved = await _resolve_track_async(
                        refresh_query, track.requested_by
                    )
                    _copy_resolved_track(track, resolved)
                except Exception as e:
                    track.resolved = True
                    track.error = str(e)
                loop.call_soon_threadsafe(lambda: self._play_resolved(track))

            asyncio.run_coroutine_threadsafe(_wait_and_play(), loop)
            return

        self._play_resolved(track)

    def _voice_busy_without_music(self) -> bool:
        return (
            self.is_connected
            and self.current is None
            and self.voice_client.is_playing()
        )

    def _schedule_start_when_free(self):
        if self._start_when_free_task and not self._start_when_free_task.done():
            return
        logger.info("[music] voice client busy with standalone audio; music will start when free")
        loop = self.voice_client.client.loop if self.voice_client else asyncio.get_event_loop()
        self._start_when_free_task = loop.create_task(self._start_when_free())

    async def _start_when_free(self):
        try:
            deadline = time.time() + 30
            while (
                self.queue
                and self.is_connected
                and self.current is None
                and self.voice_client.is_playing()
                and time.time() < deadline
            ):
                await asyncio.sleep(0.2)

            if self.queue and self.is_connected and self.current is None:
                logger.info("[music] standalone audio finished; starting queued music")
                self.play_next()
        except asyncio.CancelledError:
            pass
        finally:
            if self._start_when_free_task is asyncio.current_task():
                self._start_when_free_task = None

    def _needs_stream_refresh(self, track: Track) -> bool:
        if not track.resolved or not track.stream_url:
            return False
        # Uploaded attachments have a webpage_url (their Discord CDN URL),
        # but they are already fully resolved and must never be sent through
        # yt-dlp. Only resolver-created tracks have a query to refresh.
        if not track.query:
            return False
        return time.time() - track.resolved_at >= _STREAM_REFRESH_AFTER

    def _play_resolved(self, track: "Track"):
        """Actually start playing a resolved track."""
        if not self.is_connected:
            return

        # A track may finish resolving while a standalone activation sound or
        # TTS response has started. Resolver callbacks enter here directly and
        # otherwise bypass play_next()'s busy check, causing Discord's
        # ``Already playing audio`` exception. Put the track back and let the
        # normal free-voice scheduler start it after the standalone source ends.
        if self.current is None and self.voice_client.is_playing():
            logger.info(
                "[music] resolved track deferred; voice client is busy with standalone audio"
            )
            self.queue.appendleft(track)
            self._schedule_start_when_free()
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

        from voice.tts import MixerSource
        from voice.listener import voice_listener_manager
        session = voice_listener_manager.get_session(self.guild)
        clip_buffer = session.clip_buffer if session else None

        ffmpeg_source = discord.FFmpegPCMAudio(
            track.stream_url,
            before_options=_ffmpeg_before_options(track),
            options=_FFMPEG_OPTS,
            stderr=subprocess.PIPE,
        )
        volume_source = discord.PCMVolumeTransformer(ffmpeg_source, volume=DEFAULT_VOLUME)
        mixer_source  = MixerSource(volume_source, clip_buffer=clip_buffer)

        loop = self.voice_client.client.loop

        def _after(error):
            finished_track = self.current
            elapsed = self.elapsed_seconds if finished_track else 0.0
            stderr_text = _read_ffmpeg_stderr(ffmpeg_source)
            if error:
                logger.error(f"Playback error in {self.guild.name}: {error}")
            if stderr_text:
                logger.warning(
                    "[music] ffmpeg stderr for %r: %s",
                    finished_track.title if finished_track else "?",
                    stderr_text,
                )
            if finished_track and elapsed < 5 and (finished_track.duration or 0) > 30:
                logger.warning(
                    "[music] track ended after %.1fs despite duration=%s: %r",
                    elapsed,
                    finished_track.duration,
                    finished_track.title,
                )
                if finished_track.playback_failures < 1 and finished_track.query:
                    finished_track.playback_failures += 1
                    logger.info(f"[music] refreshing and retrying early-ended track: {finished_track.title!r}")

                    async def _refresh_and_retry():
                        try:
                            refresh_query = finished_track.query
                            excluded_urls = {finished_track.webpage_url} if finished_track.webpage_url else set()
                            resolved = await _resolve_track_async(
                                refresh_query,
                                finished_track.requested_by,
                                exclude_webpage_urls=excluded_urls,
                            )
                            _copy_resolved_track(finished_track, resolved)
                        except Exception as exc:
                            finished_track.resolved = True
                            finished_track.error = str(exc)
                            logger.error(f"[music] retry refresh failed for {finished_track.title!r}: {exc}")

                        if self.current is finished_track:
                            self.current = None
                            self.queue.appendleft(finished_track)
                            self.play_next()
                            return

                        if self.current or self.voice_client.is_playing():
                            logger.info(
                                "[music] retry ready for %r, but another track is active; re-queueing",
                                finished_track.title,
                            )
                            self.queue.appendleft(finished_track)
                            return

                        self.queue.appendleft(finished_track)
                        self.play_next()

                    asyncio.run_coroutine_threadsafe(_refresh_and_retry(), loop)
                    return

            if self._loop and self.current:
                self.queue.appendleft(self.current)

            self._natural_transition = (
                not self._manual_stop and not self._loop
            )
            self._manual_stop = False
            loop.call_soon_threadsafe(self.play_next)

        try:
            self.voice_client.play(mixer_source, after=_after)
        except Exception as exc:
            # Discord can reject play() when another standalone source starts
            # between our busy check and this call. Do not lose the track or
            # leave a phantom current track behind in that race.
            if self.current is track:
                self.current = None
            self.queue.appendleft(track)
            logger.error(
                "[music] failed to start %r; restored it to the queue: %s",
                track.title,
                exc,
            )
            if self.voice_client.is_playing():
                self._schedule_start_when_free()
            return

        logger.debug(
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

    async def shutdown(self):
        """Disconnect every guild player and clear runtime playback state."""
        players = list(self._players.values())
        for player in players:
            try:
                await player.disconnect()
            except Exception as exc:
                logger.error("[music] shutdown cleanup failed for %s: %s", player.guild.name, exc)
        self._players.clear()
        self._play_locks.clear()

    def _get_play_lock(self, guild_id: int) -> asyncio.Lock:
        if guild_id not in self._play_locks:
            self._play_locks[guild_id] = asyncio.Lock()
        return self._play_locks[guild_id]

    async def play(self, guild, requester_member, query: str) -> str:
        if not requester_member.voice or not requester_member.voice.channel:
            return "User is not in a voice channel; cannot play music."

        voice_channel = requester_member.voice.channel
        player = self.get_player(guild)

        async with self._get_play_lock(guild.id):
            try:
                track = await _resolve_track_async(query, requester_member.display_name)
            except Exception as e:
                logger.error(f"[music] resolve failed for {query!r}: {e}")
                return f"Could not resolve track: {e}"

            await player.connect(voice_channel)
            player.queue.append(track)

            is_busy = player.has_active_track or (player.is_connected and player.voice_client.is_paused())

            if not is_busy:
                player.play_next()
                return f"Now playing: {track.title}"
            else:
                position = len(player.queue)
                logger.info("[music] queued position=%d: %s", position, track.title)
                return f"Queued at position {position}: {track.title}"

    async def queue_bulk(self, guild, requester_member, queries: list[str], text_channel=None) -> str:
        """Queue multiple songs as placeholders and resolve them in background."""
        if not requester_member.voice or not requester_member.voice.channel:
            return "User is not in a voice channel; cannot play music."

        voice_channel = requester_member.voice.channel
        player = self.get_player(guild)

        async with self._get_play_lock(guild.id):
            await player.connect(voice_channel)

            if text_channel and not player.text_channel:
                player.text_channel = text_channel

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

            is_busy = player.has_active_track or (player.is_connected and player.voice_client.is_paused())

            player.start_resolver()

            if not is_busy:
                player.play_next()

            return f"Added {len(queries)} songs to queue."

    async def queue_playlist(self, guild, requester_member, url: str, text_channel=None) -> str:
        """Extract playlist entries and queue them as placeholders."""
        if not requester_member.voice or not requester_member.voice.channel:
            return "User is not in a voice channel; cannot play music."

        voice_channel = requester_member.voice.channel
        player = self.get_player(guild)

        try:
            entries = await asyncio.to_thread(extract_playlist, url, requester_member.display_name)
        except Exception as e:
            logger.error(f"[music] playlist extract failed for {url!r}: {e}")
            return f"Could not load playlist: {e}"

        if not entries:
            return "No playable tracks found in playlist."

        async with self._get_play_lock(guild.id):
            await player.connect(voice_channel)

            if text_channel and not player.text_channel:
                player.text_channel = text_channel

            for track in entries:
                player.queue.append(track)

            is_busy = player.has_active_track or (player.is_connected and player.voice_client.is_paused())

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
            player._now_playing_view = None
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


async def _post_now_playing_for_track(player, track):
    logger.debug(f"[music] posting now playing for {track.title!r}")
    if not player.text_channel:
        logger.warning("[music] text_channel is None — cannot post now playing")
        return
    from music.now_playing import post_now_playing
    await post_now_playing(
        player.text_channel,
        player,
        current_track=track,
        title=track.title,
        artist=track.artist,
        duration_seconds=track.duration,
        queue_size=len(player.queue),
        requested_by=track.requested_by,
        thumbnail_url=track.thumbnail,
        thumbnail_bytes=track.thumbnail_bytes,
    )


voice_manager = VoiceManager()
