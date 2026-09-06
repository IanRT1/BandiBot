"""
voice/listener.py

Handles per-guild voice channel listening for BandiBot.

Wake word detection uses openwakeword.model.Model which handles
the full mel → embedding → classifier pipeline internally and correctly.

State machine (per user):
  idle       → audio thread feeds wake word pipeline
  waiting    → audio thread discards all audio (activation sound playing)
  listening  → audio thread feeds Silero VAD, asyncio task monitors silence
  processing → audio thread discards all audio (STT/LLM/TTS running)

Interruption support:
  Any user can trigger wake word at any time, even if the bot is speaking.
  When a new wake word fires mid-TTS, the current TTS is cancelled immediately
  and the new user takes over. The same user can also re-trigger while their
  previous TTS response is playing.

Crypto error recovery:
  Discord DAVE encryption key rotation can corrupt the receive pipeline.
  When detected, the sink is restarted automatically.

Music plays continuously via MixerSource — TTS is injected on top,
no pausing or restarting needed.
"""


import os
import re
import time
import asyncio
import logging
import threading
import warnings

from typing import Optional
from collections import deque
from dataclasses import dataclass, field

import torch
import discord
import numpy as np
from discord.ext import voice_recv
from openwakeword.model import Model

from silero_vad import load_silero_vad
from core.interaction_logging import log_message, log_done, track_usage
from core.paths import assets_root
from bot.utils import clean_username
from voice.audio import (
    mono48k_to_16k,
    mono_to_stereo,
    samples_to_wav_bytes,
    stereo_to_mono,
    to_float32,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*tflite runtime.*")


class _CryptoPacketLogFilter(logging.Filter):
    """Collapse Discord voice crypto packet errors into one burst warning."""

    MESSAGE = "CryptoError decoding packet data"
    WINDOW_SECONDS = 10.0
    BURST_THRESHOLD = 5

    def __init__(self, clock=time.monotonic):
        super().__init__()
        self._clock = clock
        self._events: deque[float] = deque()
        self._reported = False

    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != self.MESSAGE:
            return True

        now = self._clock()
        had_events = bool(self._events)
        while self._events and self._events[0] <= now - self.WINDOW_SECONDS:
            self._events.popleft()
        if had_events and not self._events:
            self._reported = False
        self._events.append(now)

        if len(self._events) >= self.BURST_THRESHOLD and not self._reported:
            logger.warning(
                "[voice] %d Discord voice packet decryption errors in %.0fs; "
                "possible voice connection instability",
                len(self._events),
                self.WINDOW_SECONDS,
            )
            self._reported = True
        return False


logging.getLogger("discord.ext.voice_recv.reader").addFilter(_CryptoPacketLogFilter())


def _is_crypto_error(error: Exception) -> bool:
    message = str(error).lower()
    return "crypto" in message or "decrypt" in message or "cryptoerror" in message

# ── Toggle ────────────────────────────────────────────────────────────────────

VOICE_ENABLED = True
STT_TIMEOUT_SECONDS = 30
VOICE_COMMAND_TIMEOUT_SECONDS = 120
TTS_TIMEOUT_SECONDS = 120

# ── Constants ─────────────────────────────────────────────────────────────────

WAKEWORD_MODEL_PATH = str(assets_root() / "BandiBot.onnx")
WAKEWORD_THRESHOLD    = 0.05
WAKEWORD_COOLDOWN     = 2
HITS_REQUIRED         = 2
SMOOTHING_WINDOW      = 3

SOURCE_SAMPLE_RATE    = 48000
OWW_SAMPLE_RATE       = 16000
OWW_CHUNK_SIZE        = 1280

VAD_CHUNK_SIZE        = 512
VAD_SPEECH_THRESHOLD  = 0.6
VAD_MIN_SPEECH_CHUNKS = 4
VAD_GRACE_PERIOD      = 0.6

SPEECH_START_TIMEOUT  = 10.0
SPEECH_SILENCE_TIME   = 1.3
SPEECH_MAX_DURATION   = 30.0
MONITOR_INTERVAL      = 0.1

SESSION_HISTORY_SIZE  = 10
IDLE_TIMEOUT          = 600
VOICE_WATCHDOG_INTERVAL = 5.0
VOICE_DISCONNECT_GRACE = 15.0
VOICE_RECOVERY_ATTEMPTS = 3

CLIP_BUFFER_SECONDS = 30
CLIP_FRAME_SECONDS = 0.02
CLIP_FRAME_SAMPLES = int(SOURCE_SAMPLE_RATE * CLIP_FRAME_SECONDS) * 2
CLIP_BUFFER_FRAMES = int(CLIP_BUFFER_SECONDS / CLIP_FRAME_SECONDS)
CLIP_PLAYBACK_SOURCE_ID = -1
CLIP_TARGET_RMS = 0.075
CLIP_SPEECH_RMS_FLOOR = 0.008
CLIP_MAX_GAIN = 3.2
CLIP_MIN_GAIN = 0.5
CLIP_RMS_SMOOTHING = 0.12

_PLAYBACK_COMMAND_RE = re.compile(
    r"\b("
    r"play|queue|add|put\s+on|"
    r"pon|ponme|reproduce|toca|"
    r"skip|next|pause|resume|stop"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_playback_command(text: str) -> bool:
    return bool(_PLAYBACK_COMMAND_RE.search(text or ""))


class RollingClipBuffer:
    """Timestamped 48kHz stereo clip timeline.

    Discord receive callbacks arrive per user, and playback callbacks arrive
    from a separate audio thread. This buffer writes each frame into the 20ms
    wall-clock slot where it arrived, so export means "the last 30 seconds" by
    time instead of "the last N packets" or a backlog being drained later.
    """

    def __init__(self):
        self._frames: dict[int, np.ndarray] = {}
        self._last_source_slot: dict[int, int] = {}
        self._source_rms: dict[int, float] = {}
        self._lock = threading.Lock()

    def start(self):
        pass

    def stop(self):
        pass

    def add_voice_frame(self, source_id: int, frame: np.ndarray):
        self._write_frame(source_id, frame)

    def append(self, frame: np.ndarray):
        self._write_frame(CLIP_PLAYBACK_SOURCE_ID, frame)

    def _write_frame(self, source_id: int, frame: np.ndarray):
        normalized = self._normalize_frame(frame)
        now_slot = self._current_slot()

        with self._lock:
            if source_id != CLIP_PLAYBACK_SOURCE_ID:
                normalized = self._level_voice_frame_locked(source_id, normalized)

            last_slot = self._last_source_slot.get(source_id)
            slot = now_slot
            if last_slot is not None and last_slot >= now_slot:
                slot = last_slot + 1

            # If a source bursts far ahead, collapse back to real time instead
            # of dragging old audio into future clips.
            if slot > now_slot + 2:
                slot = now_slot

            existing = self._frames.get(slot)
            if existing is None:
                self._frames[slot] = normalized
            else:
                mixed = existing.astype(np.int32) + normalized.astype(np.int32)
                self._frames[slot] = np.clip(mixed, -32768, 32767).astype(np.int16)

            self._last_source_slot[source_id] = slot
            self._prune_locked(now_slot)

    def _level_voice_frame_locked(self, source_id: int, frame: np.ndarray) -> np.ndarray:
        samples = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples)))

        if rms < CLIP_SPEECH_RMS_FLOOR:
            return frame

        previous = self._source_rms.get(source_id, rms)
        smoothed = (
            previous * (1.0 - CLIP_RMS_SMOOTHING)
            + rms * CLIP_RMS_SMOOTHING
        )
        self._source_rms[source_id] = smoothed

        gain = CLIP_TARGET_RMS / max(smoothed, 1e-6)
        gain = max(CLIP_MIN_GAIN, min(CLIP_MAX_GAIN, gain))

        leveled = np.clip(samples * gain, -1.0, 1.0)
        return (leveled * 32767).astype(np.int16)

    def to_pcm(self, seconds: int = CLIP_BUFFER_SECONDS) -> bytes:
        frame_count = int(seconds / CLIP_FRAME_SECONDS)
        current_slot = self._current_slot()
        start_slot = current_slot - frame_count + 1

        with self._lock:
            frames = [
                self._frames.get(slot, np.zeros(CLIP_FRAME_SAMPLES, dtype=np.int16))
                for slot in range(start_slot, current_slot + 1)
            ]

        return np.concatenate(frames).astype(np.int16).tobytes()

    @staticmethod
    def _current_slot() -> int:
        return int(time.monotonic() / CLIP_FRAME_SECONDS)

    def _prune_locked(self, current_slot: int):
        oldest_slot = current_slot - CLIP_BUFFER_FRAMES - 5
        stale_slots = [slot for slot in self._frames if slot < oldest_slot]
        for slot in stale_slots:
            self._frames.pop(slot, None)

        stale_sources = [
            source_id
            for source_id, slot in self._last_source_slot.items()
            if slot < oldest_slot
        ]
        for source_id in stale_sources:
            self._last_source_slot.pop(source_id, None)
            self._source_rms.pop(source_id, None)

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        samples = np.asarray(frame, dtype=np.int16).reshape(-1)
        if len(samples) >= CLIP_FRAME_SAMPLES:
            return samples[:CLIP_FRAME_SAMPLES].copy()
        padded = np.zeros(CLIP_FRAME_SAMPLES, dtype=np.int16)
        padded[:len(samples)] = samples
        return padded

# ── Per-user session history ──────────────────────────────────────────────────

@dataclass
class VoiceSession:
    user_id: int
    display_name: str
    history: deque = field(default_factory=lambda: deque(maxlen=SESSION_HISTORY_SIZE))

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_history(self) -> list:
        return list(self.history)


# ── Per-user capture state ────────────────────────────────────────────────────

class UserCaptureState:
    def __init__(self):
        self.state: str = "idle"
        self.audio_buf: list = []
        self.vad_buf: np.ndarray = np.array([], dtype=np.float32)
        self.speech_chunks: int = 0
        self.total_speech_chunks: int = 0
        self.heard_speech: bool = False
        self.start_time: float = 0.0
        self.last_speech_time: float = 0.0
        self.last_vad_score: float = 0.0
        self.vad_grace_until: float = 0.0
        self.interrupted: bool = False

    def reset(self):
        self.state = "idle"
        self.audio_buf = []
        self.vad_buf = np.array([], dtype=np.float32)
        self.speech_chunks = 0
        self.total_speech_chunks = 0
        self.heard_speech = False
        self.start_time = 0.0
        self.last_speech_time = 0.0
        self.last_vad_score = 0.0
        self.vad_grace_until = 0.0
        self.interrupted = False


# ── Per-guild audio sink ──────────────────────────────────────────────────────

class BandiBotSink(voice_recv.AudioSink):

    def __init__(self, guild_session: "GuildVoiceSession"):
        super().__init__()
        self.gs = guild_session
        self._users: dict[int, UserCaptureState] = {}
        self._active_uid: Optional[int] = None
        self._oww_buf: dict[int, np.ndarray] = {}
        self._score_buf: dict[int, deque] = {}
        self._oww_models: dict[int, Model] = {}
        self._crypto_error_scheduled = False

    def wants_opus(self) -> bool:
        return False

    def _get_user(self, uid: int) -> UserCaptureState:
        if uid not in self._users:
            self._users[uid] = UserCaptureState()
        return self._users[uid]

    def _get_oww_model(self, uid: int) -> Model:
        if uid not in self._oww_models:
            self._oww_models[uid] = Model(wakeword_models=[WAKEWORD_MODEL_PATH], inference_framework="onnx")
        return self._oww_models[uid]

    def write(self, user: discord.User, data: voice_recv.VoiceData):
        if user is None:
            return
        if user.id == self.gs.guild.me.id:
            return
        try:
            pcm = data.pcm
            if not pcm:
                return
            raw    = np.frombuffer(pcm, dtype=np.int16).copy()
            mono48 = stereo_to_mono(raw)
            uid    = user.id
            self.gs.clip_buffer.add_voice_frame(uid, mono_to_stereo(mono48))
            u      = self._get_user(uid)

            if u.state in ("idle", "processing"):
                mono16 = mono48k_to_16k(mono48)
                self._feed_wakeword(uid, mono16, user, u)
            if u.state == "listening":
                if uid == self._active_uid:
                    self._feed_vad(uid, mono48, u)

        except Exception as e:
            if _is_crypto_error(e):
                # Discord may deliver a burst of undecodable packets while
                # the receive encryption state is rotating. Log one useful
                # connection warning and restart the sink once; do not flood
                # the log with one error per packet.
                if not self._crypto_error_scheduled:
                    self._crypto_error_scheduled = True
                    logger.warning(
                        "[voice] receive connection packet errors detected; restarting audio sink"
                    )
                    asyncio.run_coroutine_threadsafe(
                        self.gs._restart_sink(), self.gs.loop
                    )
                return

            logger.error(f"[voice] write error: {e}")

    def _feed_wakeword(self, uid: int, samples16k: np.ndarray, user: discord.User, u: UserCaptureState):
        if uid not in self._oww_buf:
            self._oww_buf[uid]   = np.array([], dtype=np.int16)
            self._score_buf[uid] = deque(maxlen=SMOOTHING_WINDOW)

        self._oww_buf[uid] = np.concatenate([self._oww_buf[uid], samples16k])

        oww        = self._get_oww_model(uid)
        model_name = self.gs._oww_model_name

        while len(self._oww_buf[uid]) >= OWW_CHUNK_SIZE:
            chunk = self._oww_buf[uid][:OWW_CHUNK_SIZE]
            self._oww_buf[uid] = self._oww_buf[uid][OWW_CHUNK_SIZE:]

            prediction = oww.predict(chunk)
            raw_score  = float(prediction[model_name])
            self._score_buf[uid].append(raw_score)

            hits = sum(s >= WAKEWORD_THRESHOLD for s in self._score_buf[uid])
            avg  = sum(self._score_buf[uid]) / len(self._score_buf[uid])
            #logger.debug(f"[ww] uid={uid} raw={raw_score:.5f} avg={avg:.5f} hits={hits}/{SMOOTHING_WINDOW}")

            if hits >= HITS_REQUIRED and avg >= WAKEWORD_THRESHOLD:
                processing_interrupt = False
                if u.state == "processing":
                    if not self.gs.can_interrupt_processing(uid):
                        self._oww_buf[uid] = np.array([], dtype=np.int16)
                        self._score_buf[uid].clear()
                        oww.reset()
                        logger.debug(
                            f"[voice] ── ignored duplicate wake word for {user.display_name}; command already processing ──"
                        )
                        break

                    processing_interrupt = True
                    u.interrupted = True
                    self.gs._interrupted_pipeline_uids.add(uid)
                    logger.debug(
                        "[voice] interrupting current TTS for a repeated wake word"
                    )
                    asyncio.run_coroutine_threadsafe(
                        self.gs._interrupt_current(uid), self.gs.loop
                    )
                    u.reset()

                elapsed = time.time() - self.gs._last_wake_word
                if not processing_interrupt and elapsed < WAKEWORD_COOLDOWN:
                    break

                self.gs._last_wake_word = time.time()
                self._oww_buf[uid]   = np.array([], dtype=np.int16)
                self._score_buf[uid].clear()
                oww.reset()

                prev_uid = self._active_uid
                if prev_uid is not None and prev_uid != uid:
                    prev_u = self._get_user(prev_uid)
                    if prev_u.state not in ("idle",):
                        prev_u.interrupted = True
                        self.gs._interrupted_pipeline_uids.add(prev_uid)
                        logger.debug("[voice] interrupting current response for a new wake word")
                        asyncio.run_coroutine_threadsafe(
                            self.gs._interrupt_current(uid), self.gs.loop
                        )
                        prev_u.reset()

                logger.info("[voice] wake word detected")
                u.state = "waiting"
                self._active_uid = uid
                asyncio.run_coroutine_threadsafe(
                    self.gs.on_wake_word(user), self.gs.loop
                )
                break

    def _feed_vad(self, uid: int, mono48: np.ndarray, u: UserCaptureState):
        u.audio_buf.append(mono48)

        if time.time() < u.vad_grace_until:
            return

        mono16  = mono48k_to_16k(mono48)
        mono16f = to_float32(mono16)
        u.vad_buf = np.concatenate([u.vad_buf, mono16f])

        while len(u.vad_buf) >= VAD_CHUNK_SIZE:
            chunk  = u.vad_buf[:VAD_CHUNK_SIZE]
            u.vad_buf = u.vad_buf[VAD_CHUNK_SIZE:]
            vad_score = float(self.gs._vad_model(
                torch.from_numpy(chunk), OWW_SAMPLE_RATE
            ).item())
            u.last_vad_score = vad_score

            if vad_score >= VAD_SPEECH_THRESHOLD:
                if not u.heard_speech:
                    elapsed = time.time() - u.start_time
                    logger.debug(f"[vad]  → first speech detected (score={vad_score:.2f}, {elapsed:.1f}s after activation)")
                u.heard_speech = True
                u.speech_chunks += 1
                u.total_speech_chunks += 1
                u.last_speech_time = time.time()

    def _end_capture(self, uid: int, u: UserCaptureState):
        if u.state != "listening":
            return
        u.state = "processing"
        chunks  = list(u.audio_buf)
        u.audio_buf = []

        if not chunks:
            logger.info("[vad]  ✗ empty buffer — resetting")
            self._reset_user(uid)
            return

        all_samples = np.concatenate(chunks)
        duration    = len(all_samples) / SOURCE_SAMPLE_RATE

        if u.total_speech_chunks < VAD_MIN_SPEECH_CHUNKS:
            logger.info(f"[vad]  ✗ too little speech ({u.total_speech_chunks} chunks, {duration:.2f}s) — resetting")
            self._reset_user(uid)
            return

        logger.debug(f"[voice] ← captured {duration:.2f}s | {u.total_speech_chunks} speech chunks | sending to STT")
        wav = samples_to_wav_bytes(all_samples, SOURCE_SAMPLE_RATE)
        #with open("debug_capture.wav", "wb") as f:
        #    f.write(wav)
        asyncio.run_coroutine_threadsafe(
            self.gs.on_speech_captured(uid, wav), self.gs.loop
        )

    def _reset_user(self, uid: int):
        u = self._get_user(uid)
        was_state = u.state
        u.reset()
        if self._active_uid == uid:
            self._active_uid = None
        self.gs._vad_model.reset_states()
        if was_state != "idle":
            logger.debug(f"[voice] → listening for wake word | sink_active={self.gs._voice_client is not None and self.gs._voice_client.is_connected()}")

    def set_listening(self, uid: int):
        u   = self._get_user(uid)
        now = time.time()
        u.state               = "listening"
        u.audio_buf           = []
        u.vad_buf             = np.array([], dtype=np.float32)
        u.speech_chunks       = 0
        u.total_speech_chunks = 0
        u.heard_speech        = False
        u.start_time          = now
        u.last_speech_time    = now
        u.last_vad_score      = 0.0
        u.vad_grace_until     = now + VAD_GRACE_PERIOD
        u.interrupted         = False
        self.gs._vad_model.reset_states()
        logger.debug(f"[vad]  → grace period {VAD_GRACE_PERIOD:.1f}s, then waiting for speech (timeout {SPEECH_START_TIMEOUT:.0f}s)")

    def cleanup(self):
        pass

# ── Per-guild voice session ───────────────────────────────────────────────────

class GuildVoiceSession:

    def __init__(self, guild: discord.Guild, client: discord.Client, loop: asyncio.AbstractEventLoop):
        self.guild  = guild
        self.client = client
        self.loop   = loop
        self._pipeline_is_music = False
        self.clip_buffer = RollingClipBuffer()

        self._sessions: dict[int, VoiceSession] = {}

        if not os.path.exists(WAKEWORD_MODEL_PATH):
            raise FileNotFoundError(f"Wake word model not found at {WAKEWORD_MODEL_PATH}")

        self._oww_model      = Model(wakeword_models=[WAKEWORD_MODEL_PATH], inference_framework="onnx")
        self._oww_model_name = list(self._oww_model.models.keys())[0]
        self._vad_model      = load_silero_vad()

        self.sink: Optional[BandiBotSink] = None
        self._voice_client: Optional[voice_recv.VoiceRecvClient] = None
        self._last_activity: float = time.time()
        self._idle_task: Optional[asyncio.Task] = None
        self._connection_watchdog_task: Optional[asyncio.Task] = None
        self._last_wake_word: float = 0.0
        self._pipeline_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._interrupted_pipeline_uids: set[int] = set()
        self._speech_interrupted_for: set[int] = set()
        self._protected_pipeline_tasks: set[asyncio.Task] = set()

        logger.info(f"[voice] ready in {guild.name} | model: {self._oww_model_name}")

    def bump_activity(self):
        self._last_activity = time.time()

    def get_clip_pcm(self) -> bytes:
        return self.clip_buffer.to_pcm(CLIP_BUFFER_SECONDS)

    def get_session(self, user: discord.User) -> VoiceSession:
        if user.id not in self._sessions:
            self._sessions[user.id] = VoiceSession(
                user_id=user.id,
                display_name=getattr(user, 'display_name', user.name)
            )
        return self._sessions[user.id]

    def can_interrupt_processing(self, uid: int) -> bool:
        """Allow same-user wake words to interrupt only once TTS is active."""
        if self.sink and self.sink._active_uid != uid:
            return False
        return self._tts_is_active()

    def _tts_is_active(self) -> bool:
        if not self._voice_client or not self._voice_client.is_connected():
            return False

        from voice.tts_sources import MixerSource, StandaloneSource

        standalone = getattr(self._voice_client, "_standalone_tts", None)
        if isinstance(standalone, StandaloneSource):
            return not standalone._finished_evt.is_set() and not standalone.is_cancelled()

        source = getattr(self._voice_client, "source", None)
        source = getattr(source, "mixer", None) or source
        if isinstance(source, MixerSource):
            with source._lock:
                return source._tts_active or bool(source._tts_buf)

        return False

    async def _interrupt_current(self, next_uid: int | None = None):
        """Cancel any in-progress TTS immediately."""
        from voice.tts import cancel_tts
        if next_uid is not None and self._tts_is_active():
            self._speech_interrupted_for.add(next_uid)
        cancel_tts(self._voice_client)
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None
        pipeline_task = self._pipeline_task
        if pipeline_task and not pipeline_task.done():
            if pipeline_task not in self._protected_pipeline_tasks:
                pipeline_task.cancel()
                self._pipeline_task = None
        self._pipeline_is_music = False
        logger.info("[voice] ← interrupted current pipeline")

    def _stop_receiving(self):
        if not self._voice_client:
            return
        try:
            self._voice_client.stop_listening()
        except discord.ClientException:
            pass
        except Exception as e:
            logger.error(f"[voice] failed to stop receiver: {e}")

    async def _start_receiving(self, voice_channel: discord.VoiceChannel):
        await asyncio.sleep(1.0)
        self._oww_model.reset()
        self._stop_receiving()
        try:
            self._voice_client.listen(self.sink)
        except discord.ClientException as e:
            if "Already receiving audio" not in str(e):
                raise
            logger.warning("[voice] receiver was already active — restarting receive sink")
            self._stop_receiving()
            self._voice_client.listen(self.sink)
        self._last_activity = time.time()
        if not self._idle_task or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_loop())
        logger.info(f"[voice] → listening for wake word in {voice_channel.name}")

    async def _restart_sink(self):
        if not self._voice_client or not self._voice_client.is_connected():
            return
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None
        self.sink = BandiBotSink(self)
        self._stop_receiving()
        try:
            self._voice_client.listen(self.sink)
            logger.info("[voice] receive sink restarted")
        except Exception as e:
            logger.error(f"[voice] receive sink restart failed: {e}")

    async def start(self, voice_channel: discord.VoiceChannel):
        self.sink = BandiBotSink(self)
        existing  = voice_channel.guild.voice_client
        if existing:
            self._voice_client = existing
        else:
            self._voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        self.clip_buffer.start()
        await self._start_receiving(voice_channel)
        self._connection_watchdog_task = asyncio.create_task(
            self._connection_watchdog()
        )

    async def stop(self):
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        if self._connection_watchdog_task:
            self._connection_watchdog_task.cancel()
            self._connection_watchdog_task = None
        if self._pipeline_task:
            self._pipeline_task.cancel()
            self._pipeline_task = None
        self.clip_buffer.stop()
        if self._voice_client:
            self._stop_receiving()
            try:
                await self._voice_client.disconnect(force=True)
            except TypeError:
                await self._voice_client.disconnect()
            except Exception as exc:
                logger.warning("[voice] stale voice disconnect failed: %s", exc)
        self._voice_client = None
        self.sink = None
        logger.info(f"[voice] disconnected from {self.guild.name}")

    async def _connection_watchdog(self):
        """Replace a voice client that remains disconnected after grace time."""
        disconnected_at = None
        try:
            while True:
                await asyncio.sleep(VOICE_WATCHDOG_INTERVAL)
                voice_client = self._voice_client
                if voice_client is None:
                    return
                if voice_client.is_connected():
                    disconnected_at = None
                    continue

                now = time.monotonic()
                if disconnected_at is None:
                    disconnected_at = now
                    logger.warning(
                        "[voice] voice connection lost in %s; waiting %.0fs before recovery",
                        self.guild.name,
                        VOICE_DISCONNECT_GRACE,
                    )
                    continue
                if now - disconnected_at < VOICE_DISCONNECT_GRACE:
                    continue

                logger.error(
                    "[voice] voice connection stayed disconnected in %s; starting recovery",
                    self.guild.name,
                )
                asyncio.create_task(voice_listener_manager.recover(self.guild, self))
                return
        except asyncio.CancelledError:
            pass

    async def _idle_loop(self):
        from music.player import voice_manager
        last_reset = time.time()
        try:
            while True:
                await asyncio.sleep(60)

                if time.time() - last_reset > 600:
                    self._oww_model.reset()
                    last_reset = time.time()
                    logger.info("[voice] reset wake word model state")

                player = voice_manager.get_player(self.guild)
                music_active = player.is_playing or bool(player.queue) or (
                    player.is_connected and player.voice_client and player.voice_client.is_paused()
                )
                if music_active:
                    self._last_activity = time.time()
                if not music_active and time.time() - self._last_activity >= IDLE_TIMEOUT:
                    logger.info(f"[voice] idle timeout — leaving {self.guild.name}")
                    await self.stop()
                    voice_listener_manager._sessions.pop(self.guild.id, None)
                    break
        except asyncio.CancelledError:
            pass

    async def on_wake_word(self, user: discord.User):
        from voice.tts import play_activation, cancel_tts
        self.bump_activity()
        cancel_tts(self._voice_client)
        logger.debug("[voice] listening for command")
        try:
            asyncio.create_task(play_activation(self._voice_client))
        except Exception as e:
            logger.error(f"[voice] activation failed: {e}")
        if self.sink:
            self.sink.set_listening(user.id)
        self._monitor_task = asyncio.create_task(self._speech_monitor(user.id, user.display_name))

    async def _speech_monitor(self, uid: int, display_name: str):
        start = time.time()
        logger.debug("[vad] monitor started")

        try:
            while True:
                await asyncio.sleep(MONITOR_INTERVAL)

                if not self.sink:
                    return

                u = self.sink._get_user(uid)

                if u.interrupted:
                    logger.info(f"[vad]  monitor exiting — interrupted")
                    return

                if u.state != "listening":
                    logger.info(f"[vad]  monitor exiting — state={u.state}")
                    return

                elapsed           = time.time() - start
                since_last_speech = time.time() - u.last_speech_time
                in_grace          = time.time() < u.vad_grace_until

                if elapsed >= SPEECH_MAX_DURATION:
                    logger.info(f"[vad]  ✗ max duration {SPEECH_MAX_DURATION:.0f}s — forcing STT")
                    self.sink._end_capture(uid, u)
                    return

                if in_grace:
                    continue

                if not u.heard_speech:
                    grace_elapsed = elapsed - VAD_GRACE_PERIOD
                    if grace_elapsed >= SPEECH_START_TIMEOUT:
                        logger.info(f"[vad]  ✗ no speech in {grace_elapsed:.1f}s — resetting")
                        self.sink._reset_user(uid)
                        return
                    continue

                if since_last_speech >= SPEECH_SILENCE_TIME:
                    total_duration = sum(len(s) for s in u.audio_buf) / SOURCE_SAMPLE_RATE
                    logger.debug(
                        f"[vad]  ← {since_last_speech:.1f}s silence | "
                        f"{u.total_speech_chunks} speech chunks | "
                        f"{total_duration:.2f}s captured"
                    )
                    self.sink._end_capture(uid, u)
                    return

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[vad]  monitor error: {e}")

    async def on_speech_captured(self, uid: int, wav_bytes: bytes):
        from voice.stt import transcribe
        from voice.tts import speak, play_activation, cancel_tts, MixerSource
        from voice.handler import handle_voice_command

        member = self.guild.get_member(uid)
        if not member:
            if self.sink:
                self.sink._reset_user(uid)
            return

        if self.sink:
            u = self.sink._get_user(uid)
            if u.interrupted:
                logger.debug(f"[voice] ✗ pipeline cancelled before STT — user was interrupted")
                u.reset()
                return

        session = self.get_session(member)
        pending_interruptions = getattr(self, "_speech_interrupted_for", set())
        speech_was_interrupted = uid in pending_interruptions
        pending_interruptions.discard(uid)

        @track_usage
        async def _pipeline():
            current_task = asyncio.current_task()
            interaction_start = time.perf_counter()
            completion_status = "failed"
            try:
                try:
                    text = await asyncio.wait_for(
                        transcribe(wav_bytes), timeout=STT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[stt] request timed out after %ds; resetting voice state",
                        STT_TIMEOUT_SECONDS,
                    )
                    return
                if not text or len(text.strip()) < 2:
                    logger.info("[voice] ✗ empty transcription — resetting")
                    if self.sink:
                        self.sink._reset_user(uid)
                    return

                if self.sink and self.sink._get_user(uid).interrupted:
                    completion_status = "interrupted"
                    logger.debug("[voice] ✗ interrupted after STT — dropping response")
                    self.sink._get_user(uid).reset()
                    return

                log_message(logger, "voice", "user", clean_username(getattr(member, "nick", None), member.name), text)
                session.add("user", text)
                t = time.perf_counter()
                likely_playback = _looks_like_playback_command(text)
                if likely_playback:
                    self._pipeline_is_music = True
                    if current_task:
                        self._protected_pipeline_tasks.add(current_task)
                try:
                    response_text, should_leave = await asyncio.wait_for(
                        handle_voice_command(
                            text=text, member=member, guild=self.guild,
                            client=self.client, history=session.get_history(),
                            speech_was_interrupted=speech_was_interrupted,
                        ),
                        timeout=VOICE_COMMAND_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[voice] command timed out after %ds; resetting voice state",
                        VOICE_COMMAND_TIMEOUT_SECONDS,
                    )
                    return
                elapsed = (time.perf_counter() - t) * 1000
                self._pipeline_is_music = not bool(response_text)

                was_interrupted = uid in self._interrupted_pipeline_uids
                if was_interrupted:
                    self._interrupted_pipeline_uids.discard(uid)

                if self.sink and (self.sink._get_user(uid).interrupted or was_interrupted):
                    completion_status = "interrupted"
                    logger.debug("[voice] ✗ interrupted after LLM — dropping TTS")
                    self.sink._get_user(uid).reset()
                    if response_text:
                        return

                if response_text:
                    log_message(logger, "voice", "bot", self.client.user.display_name, response_text)
                    session.add("assistant", response_text)
                    try:
                        await asyncio.wait_for(
                            speak(
                                self._voice_client,
                                response_text,
                                guild=self.guild,
                                clip_buffer=self.clip_buffer,
                            ),
                            timeout=TTS_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "[tts] speech timed out after %ds; cancelling and resetting voice state",
                            TTS_TIMEOUT_SECONDS,
                        )
                        cancel_tts(self._voice_client)
                        return
                    if should_leave:
                        await asyncio.sleep(1)
                        logger.info("[voice] → leaving voice channel")
                        if self._idle_task:
                            self._idle_task.cancel()
                            self._idle_task = None
                        voice_listener_manager._sessions.pop(self.guild.id, None)
                        try:
                            if self._voice_client and self._voice_client.is_connected():
                                self._stop_receiving()
                                await self._voice_client.disconnect()
                                logger.info("[voice] → disconnected")
                        except Exception as e:
                            logger.error(f"[voice] → disconnect failed: {e}")
                        self._voice_client = None
                        self.sink = None
                else:
                    logger.debug(f"[voice] music command pipeline completed in {elapsed:.0f}ms | final_reply=no")

                if completion_status != "interrupted":
                    completion_status = "done"

            except asyncio.CancelledError:
                completion_status = "interrupted"
                logger.debug("[voice] pipeline task cancelled")
            except Exception as e:
                logger.error(f"[voice] pipeline error: {e}")
            finally:
                log_done(logger, "voice", (time.perf_counter() - interaction_start) * 1000, completion_status)
                self._interrupted_pipeline_uids.discard(uid)
                if current_task:
                    self._protected_pipeline_tasks.discard(current_task)
                if self._pipeline_task is current_task:
                    self._pipeline_task = None
                    self._pipeline_is_music = False
                if self.sink:
                    u = self.sink._get_user(uid)
                    if u.state not in ("listening", "waiting"):
                        self.sink._reset_user(uid)

        self._pipeline_task = asyncio.create_task(_pipeline())


# ── Module-level manager ──────────────────────────────────────────────────────

class VoiceListenerManager:

    def __init__(self):
        self._sessions: dict[int, GuildVoiceSession] = {}
        self._lifecycle_lock = asyncio.Lock()

    def get_session(self, guild: discord.Guild) -> Optional[GuildVoiceSession]:
        return self._sessions.get(guild.id)

    async def start_listening(self, guild, voice_channel, client, loop):
        async with self._lifecycle_lock:
            if not VOICE_ENABLED:
                logger.info("[voice] disabled — skipping")
                return None
            existing = self._sessions.get(guild.id)
            if existing:
                voice_client = existing._voice_client
                if (
                    existing.sink is not None
                    and voice_client is not None
                    and voice_client.is_connected()
                ):
                    return existing
                await existing.stop()
                self._sessions.pop(guild.id, None)
            session = GuildVoiceSession(guild, client, loop)
            self._sessions[guild.id] = session
            await session.start(voice_channel)
            return session

    async def stop_listening(self, guild):
        async with self._lifecycle_lock:
            if not VOICE_ENABLED:
                return
            session = self._sessions.pop(guild.id, None)
            if session:
                await session.stop()

    async def recover(self, guild, failed_session: GuildVoiceSession):
        """Replace a voice session after a sustained connection failure."""
        async with self._lifecycle_lock:
            if self._sessions.get(guild.id) is not failed_session:
                return

            from music.player import voice_manager

            player = voice_manager.get_player(guild)
            await player.prepare_for_voice_recovery()

            voice_channel = getattr(failed_session._voice_client, "channel", None)
            if voice_channel is None:
                bot_member = getattr(guild, "me", None)
                voice_state = getattr(bot_member, "voice", None)
                voice_channel = getattr(voice_state, "channel", None)
            if voice_channel is None:
                logger.error("[voice] recovery aborted in %s: channel unavailable", guild.name)
                self._sessions.pop(guild.id, None)
                await failed_session.stop()
                return

            self._sessions.pop(guild.id, None)
            await failed_session.stop()

            for attempt in range(1, VOICE_RECOVERY_ATTEMPTS + 1):
                session = None
                try:
                    logger.warning(
                        "[voice] reconnecting to %s (attempt %d/%d)",
                        guild.name,
                        attempt,
                        VOICE_RECOVERY_ATTEMPTS,
                    )
                    session = GuildVoiceSession(
                        guild, failed_session.client, failed_session.loop
                    )
                    self._sessions[guild.id] = session
                    await session.start(voice_channel)
                    player.restore_after_voice_recovery(session._voice_client)
                    logger.info("[voice] voice recovery succeeded in %s", guild.name)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        "[voice] voice recovery attempt %d failed in %s: %s",
                        attempt,
                        guild.name,
                        exc,
                    )
                    if self._sessions.get(guild.id) is session:
                        self._sessions.pop(guild.id, None)
                    if session:
                        await session.stop()
                    if attempt < VOICE_RECOVERY_ATTEMPTS:
                        await asyncio.sleep(min(30, 2 ** attempt))

            logger.error("[voice] voice recovery exhausted in %s", guild.name)

    async def shutdown(self):
        """Stop all active voice listener sessions during process shutdown."""
        async with self._lifecycle_lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()
            for guild_id, session in sessions:
                try:
                    await session.stop()
                except Exception as exc:
                    logger.error("[voice] shutdown cleanup failed for guild %s: %s", guild_id, exc)


def notify_music_activity(guild: discord.Guild):
    session = voice_listener_manager.get_session(guild)
    if session:
        session.bump_activity()


voice_listener_manager = VoiceListenerManager()
