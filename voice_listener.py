"""
voice_listener.py

Handles per-guild voice channel listening for BandiBot.

Wake word detection uses openwakeword.model.Model which handles
the full mel → embedding → classifier pipeline internally and correctly.

State machine (per user):
  idle       → audio thread feeds wake word pipeline
  waiting    → audio thread discards all audio (activation sound playing)
  listening  → audio thread feeds Silero VAD, asyncio task monitors silence
  processing → audio thread discards all audio (STT/LLM/TTS running)
"""

import asyncio
import logging
import os
import time
import wave
import io
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import discord
import numpy as np
import torch
from silero_vad import load_silero_vad
from openwakeword.model import Model
from discord.ext import voice_recv

logger = logging.getLogger(__name__)

# ── Toggle ────────────────────────────────────────────────────────────────────

VOICE_ENABLED = True  # Set to False to disable voice commands

# ── Constants ─────────────────────────────────────────────────────────────────

WAKEWORD_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "BandiBot.onnx")
WAKEWORD_THRESHOLD    = 0.40
WAKEWORD_COOLDOWN     = 5
HITS_REQUIRED         = 3
SMOOTHING_WINDOW      = 5

SOURCE_SAMPLE_RATE    = 48000
OWW_SAMPLE_RATE       = 16000
OWW_CHUNK_SIZE        = 1280   # required by openWakeWord

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


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _stereo_to_mono(samples: np.ndarray) -> np.ndarray:
    if len(samples) % 2 == 0:
        left  = samples[0::2].astype(np.int32)
        right = samples[1::2].astype(np.int32)
        return ((left + right) >> 1).astype(np.int16)
    return samples


def _mono48k_to_16k(samples: np.ndarray) -> np.ndarray:
    n = (len(samples) // 3) * 3
    return samples[:n].reshape(-1, 3).mean(axis=1).astype(np.int16)


def _to_float32(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / 32768.0


def _samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = SOURCE_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


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


# ── Per-guild audio sink ──────────────────────────────────────────────────────

class BandiBotSink(voice_recv.AudioSink):

    def __init__(self, guild_session: "GuildVoiceSession"):
        super().__init__()
        self.gs = guild_session
        self._users: dict[int, UserCaptureState] = {}
        self._active_uid: Optional[int] = None
        self._oww_buf: dict[int, np.ndarray] = {}
        self._score_buf: dict[int, deque] = {}

    def wants_opus(self) -> bool:
        return False

    def _get_user(self, uid: int) -> UserCaptureState:
        if uid not in self._users:
            self._users[uid] = UserCaptureState()
        return self._users[uid]

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
            mono48 = _stereo_to_mono(raw)
            uid    = user.id
            u      = self._get_user(uid)

            if u.state == "idle":
                if self._active_uid is None:
                    mono16 = _mono48k_to_16k(mono48)
                    self._feed_wakeword(uid, mono16, user, u)
            elif u.state == "listening":
                if uid == self._active_uid:
                    self._feed_vad(uid, mono48, u)

        except Exception:
            pass

    def _feed_wakeword(self, uid: int, samples16k: np.ndarray, user: discord.User, u: UserCaptureState):
        if uid not in self._oww_buf:
            self._oww_buf[uid]   = np.array([], dtype=np.int16)
            self._score_buf[uid] = deque(maxlen=SMOOTHING_WINDOW)

        self._oww_buf[uid] = np.concatenate([self._oww_buf[uid], samples16k])

        while len(self._oww_buf[uid]) >= OWW_CHUNK_SIZE:
            chunk = self._oww_buf[uid][:OWW_CHUNK_SIZE]
            self._oww_buf[uid] = self._oww_buf[uid][OWW_CHUNK_SIZE:]

            prediction  = self.gs._oww_model.predict(chunk)
            model_name  = self.gs._oww_model_name
            raw_score   = float(prediction[model_name])
            self._score_buf[uid].append(raw_score)

            hits = sum(s >= WAKEWORD_THRESHOLD for s in self._score_buf[uid])
            avg  = sum(self._score_buf[uid]) / len(self._score_buf[uid])

            if hits >= HITS_REQUIRED and avg >= WAKEWORD_THRESHOLD:
                elapsed = time.time() - self.gs._last_wake_word
                if elapsed < WAKEWORD_COOLDOWN:
                    break
                self.gs._last_wake_word = time.time()
                # Clear buffers and reset model state after detection
                self._oww_buf[uid]   = np.array([], dtype=np.int16)
                self._score_buf[uid].clear()
                self.gs._oww_model.reset()
                logger.info(f"[voice] ── wake word ({user.display_name}, hits={hits}/{SMOOTHING_WINDOW} avg={avg:.3f}) ──")
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

        mono16  = _mono48k_to_16k(mono48)
        mono16f = _to_float32(mono16)
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
                    logger.info(f"[vad]  → first speech detected (score={vad_score:.2f}, {elapsed:.1f}s after activation)")
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

        logger.info(f"[voice] ← captured {duration:.2f}s | {u.total_speech_chunks} speech chunks | sending to STT")
        wav = _samples_to_wav_bytes(all_samples)
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
            # Verify sink is still listening
            vc = self.gs._voice_client
            logger.info(f"[voice] → listening for wake word | sink_active={vc is not None and vc.is_connected()}")

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
        self.gs._vad_model.reset_states()
        logger.info(f"[vad]  → grace period {VAD_GRACE_PERIOD:.1f}s, then waiting for speech (timeout {SPEECH_START_TIMEOUT:.0f}s)")

    def cleanup(self):
        pass


# ── Per-guild voice session ───────────────────────────────────────────────────

class GuildVoiceSession:

    def __init__(self, guild: discord.Guild, client: discord.Client, loop: asyncio.AbstractEventLoop):
        self.guild  = guild
        self.client = client
        self.loop   = loop

        self._sessions: dict[int, VoiceSession] = {}

        if not os.path.exists(WAKEWORD_MODEL_PATH):
            raise FileNotFoundError(f"Wake word model not found at {WAKEWORD_MODEL_PATH}")

        # Load openWakeWord model using their full pipeline
        self._oww_model      = Model(wakeword_models=[WAKEWORD_MODEL_PATH])
        self._oww_model_name = list(self._oww_model.models.keys())[0]

        # Silero VAD
        self._vad_model = load_silero_vad()

        self.sink: Optional[BandiBotSink] = None
        self._voice_client: Optional[voice_recv.VoiceRecvClient] = None
        self._last_activity: float = time.time()
        self._idle_task: Optional[asyncio.Task] = None
        self._last_wake_word: float = 0.0

        logger.info(f"[voice] ready in {guild.name} | model: {self._oww_model_name}")

    def bump_activity(self):
        self._last_activity = time.time()

    def get_session(self, user: discord.User) -> VoiceSession:
        if user.id not in self._sessions:
            self._sessions[user.id] = VoiceSession(
                user_id=user.id,
                display_name=getattr(user, 'display_name', user.name)
            )
        return self._sessions[user.id]

    async def start(self, voice_channel: discord.VoiceChannel):
        self.sink = BandiBotSink(self)
        existing  = voice_channel.guild.voice_client
        if existing:
            self._voice_client = existing
        else:
            self._voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        await asyncio.sleep(1.0)
        self._oww_model.reset()  # ← add this
        self._voice_client.listen(self.sink)
        self._last_activity = time.time()
        self._idle_task = asyncio.create_task(self._idle_loop())
        logger.info(f"[voice] → listening for wake word in {voice_channel.name}")

    async def stop(self):
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
        if self._voice_client and self._voice_client.is_connected():
            self._voice_client.stop_listening()
            await self._voice_client.disconnect()
        self._voice_client = None
        self.sink = None
        logger.info(f"[voice] disconnected from {self.guild.name}")

    async def _idle_loop(self):
        from music import voice_manager
        last_reset = time.time()
        try:
            while True:
                await asyncio.sleep(60)

                # Reset OWW model state every 10 minutes to prevent drift
                if time.time() - last_reset > 600:
                    self._oww_model.reset()
                    last_reset = time.time()
                    logger.info("[voice] reset wake word model state")

                player = voice_manager.get_player(self.guild)
                music_active = player.is_playing or bool(player.queue) or (
                    player.is_connected and player.voice_client and player.voice_client.is_paused()
                )
                if not music_active and time.time() - self._last_activity >= IDLE_TIMEOUT:
                    logger.info(f"[voice] idle timeout — leaving {self.guild.name}")
                    await self.stop()
                    voice_listener_manager._sessions.pop(self.guild.id, None)
                    break
        except asyncio.CancelledError:
            pass

    async def on_wake_word(self, user: discord.User):
        from tts import play_activation
        from music import voice_manager
        self.bump_activity()
        player = voice_manager.get_player(self.guild)

        if player.is_playing:
            logger.info(f"[voice] → listening for command [{user.display_name}] (music playing)")
            if hasattr(player.voice_client, 'source') and player.voice_client.source:
                if hasattr(player.voice_client.source, 'volume'):
                    player.voice_client.source.volume = 0.08
        else:
            logger.info(f"[voice] → listening for command [{user.display_name}]")
            asyncio.create_task(play_activation(self._voice_client))

        if self.sink:
            self.sink.set_listening(user.id)

        asyncio.create_task(self._speech_monitor(user.id, user.display_name))

    async def _speech_monitor(self, uid: int, display_name: str):
        start = time.time()
        logger.info(f"[vad]  monitor started for {display_name}")

        try:
            while True:
                await asyncio.sleep(MONITOR_INTERVAL)

                if not self.sink:
                    return

                u = self.sink._get_user(uid)

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
                        await self._unduck()
                        return
                    continue

                if since_last_speech >= SPEECH_SILENCE_TIME:
                    total_duration = sum(len(s) for s in u.audio_buf) / SOURCE_SAMPLE_RATE
                    logger.info(
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
    
    async def _unduck(self):
        from music import voice_manager
        player = voice_manager.get_player(self.guild)
        if player.is_connected:
            if hasattr(player.voice_client, 'source') and player.voice_client.source:
                if hasattr(player.voice_client.source, 'volume'):
                    player.voice_client.source.volume = 0.30

    async def on_speech_captured(self, uid: int, wav_bytes: bytes):
        from stt import transcribe
        from tts import speak
        from voice_handler import handle_voice_command

        member = self.guild.get_member(uid)
        if not member:
            if self.sink:
                self.sink._reset_user(uid)
            return

        session = self.get_session(member)
        try:
            text = await transcribe(wav_bytes)
            if not text or len(text.strip()) < 2:
                logger.info("[voice] ✗ empty transcription — resetting")
                if self.sink:
                    self.sink._reset_user(uid)
                return

            logger.info(f"[llm]  → processing: {text!r}")
            session.add("user", text)
            t = time.perf_counter()
            response_text = await handle_voice_command(
                text=text, member=member, guild=self.guild,
                client=self.client, history=session.get_history(),
            )
            elapsed = (time.perf_counter() - t) * 1000

            if response_text:
                session.add("assistant", response_text)
                logger.info(f"[llm]  ← {elapsed:.0f}ms | speaking response")
                await speak(self._voice_client, response_text, guild=self.guild)
            else:
                logger.info(f"[llm]  ← {elapsed:.0f}ms | no TTS (music command)")

        except Exception as e:
            logger.error(f"[voice] pipeline error: {e}")
        finally:
            if self.sink:
                self.sink._reset_user(uid)
            await self._unduck()


# ── Module-level manager ──────────────────────────────────────────────────────

class VoiceListenerManager:

    def __init__(self):
        self._sessions: dict[int, GuildVoiceSession] = {}

    def get_session(self, guild: discord.Guild) -> Optional[GuildVoiceSession]:
        return self._sessions.get(guild.id)

    async def start_listening(self, guild, voice_channel, client, loop):
        if not VOICE_ENABLED:
            logger.info("[voice] disabled — skipping")
            return None
        existing = self._sessions.get(guild.id)
        if existing:
            if existing.sink is not None:
                return existing
            await existing.stop()
            self._sessions.pop(guild.id, None)
        session = GuildVoiceSession(guild, client, loop)
        self._sessions[guild.id] = session
        await session.start(voice_channel)
        return session

    async def stop_listening(self, guild):
        if not VOICE_ENABLED:
            return
        session = self._sessions.pop(guild.id, None)
        if session:
            await session.stop()


def notify_music_activity(guild: discord.Guild):
    session = voice_listener_manager.get_session(guild)
    if session:
        session.bump_activity()


voice_listener_manager = VoiceListenerManager()