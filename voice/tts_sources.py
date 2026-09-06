"""
voice/tts_sources.py

Discord AudioSource implementations for BandiBot text-to-speech playback.

This module owns the low-level audio buffering that Discord pulls from in its
voice thread. It does not call TTS providers or decide what text to say; it
only accepts 48kHz int16 mono PCM, expands it to Discord stereo frames, and
serves cancellable audio frames to the voice client.

Playback modes:
  MixerSource      -> wraps active music, ducks it, and injects TTS on top
  StandaloneSource -> feeds TTS directly when no music source is playing

Cancellation:
  Both sources are thread-safe. The asyncio voice pipeline can cancel from the
  event loop while Discord continues reading frames from its audio thread.
"""

import logging
import threading

import discord
import numpy as np

from voice.audio import mono_to_stereo

logger = logging.getLogger(__name__)

DISCORD_FRAME_SIZE = 3840
MUSIC_DUCK_VOLUME = 0.3


class MixerSource(discord.AudioSource):
    """
    Audio source that mixes a primary source (music) with optional TTS.

    Music plays continuously. TTS is injected on top when speak() feeds PCM
    into the source. Music is ducked during speech, and TTS can be cancelled
    mid-stream without stopping the music source.
    """

    def __init__(self, primary: discord.AudioSource, clip_buffer=None):
        self._primary = primary
        self._tts_buf = bytearray()
        self._tts_done = False
        self._tts_active = False
        self._cancelled = False
        self._lock = threading.Lock()
        self._tts_finished = threading.Event()
        self._clip_buffer = clip_buffer

    def read(self) -> bytes:
        music_frame = self._primary.read()
        if not music_frame:
            return b""

        music = np.frombuffer(music_frame, dtype=np.int16).astype(np.int32)

        with self._lock:
            has_tts = len(self._tts_buf) >= DISCORD_FRAME_SIZE
            tts_done = self._tts_done and len(self._tts_buf) < DISCORD_FRAME_SIZE

        if has_tts:
            with self._lock:
                tts_frame = bytes(self._tts_buf[:DISCORD_FRAME_SIZE])
                del self._tts_buf[:DISCORD_FRAME_SIZE]
            tts = np.frombuffer(tts_frame, dtype=np.int16).astype(np.int32)
            mixed = (music * MUSIC_DUCK_VOLUME).astype(np.int32) + tts
            mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
            frame = mixed.tobytes()
        elif tts_done and self._tts_active:
            with self._lock:
                self._tts_active = False
                self._tts_done = False
                self._cancelled = False
            self._tts_finished.set()
            frame = music_frame
        else:
            frame = music_frame

        if self._clip_buffer is not None:
            self._clip_buffer.append(np.frombuffer(frame, dtype=np.int16).copy())

        return frame

    def feed_tts(self, pcm_mono: bytes):
        if not pcm_mono:
            return
        with self._lock:
            if self._cancelled:
                return
        samples = np.frombuffer(pcm_mono, dtype=np.int16)
        stereo = mono_to_stereo(samples)
        with self._lock:
            self._tts_active = True
            self._tts_buf.extend(stereo.tobytes())

    def finish_tts(self):
        with self._lock:
            self._tts_done = True

    def cancel(self):
        with self._lock:
            self._tts_buf.clear()
            self._tts_done = True
            self._cancelled = True
            self._tts_active = False
        self._tts_finished.set()
        logger.debug("[tts]  <- cancelled (wake word interrupt)")

    def reset_cancel(self):
        with self._lock:
            self._cancelled = False

    def wait_tts_done(self, timeout: float = 30.0) -> bool:
        return self._tts_finished.wait(timeout=timeout)


class StandaloneSource(discord.AudioSource):
    """Standalone TTS source for when no music is playing."""

    def __init__(self, clip_buffer=None):
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._done = False
        self._cancelled = False
        self._finished_evt = threading.Event()
        self._clip_buffer = clip_buffer

    def feed(self, pcm_mono: bytes):
        with self._lock:
            if self._cancelled:
                return
        samples = np.frombuffer(pcm_mono, dtype=np.int16)
        stereo = mono_to_stereo(samples)
        with self._lock:
            self._buf.extend(stereo.tobytes())

    def cancel(self):
        with self._lock:
            self._buf.clear()
            self._done = True
            self._cancelled = True
        self._finished_evt.set()
        logger.debug("[tts]  <- standalone cancelled (wake word interrupt)")

    def set_done(self):
        with self._lock:
            self._done = True

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def read(self) -> bytes:
        with self._lock:
            if len(self._buf) >= DISCORD_FRAME_SIZE:
                data = bytes(self._buf[:DISCORD_FRAME_SIZE])
                del self._buf[:DISCORD_FRAME_SIZE]
            elif self._done:
                if self._buf:
                    data = bytes(self._buf).ljust(DISCORD_FRAME_SIZE, b"\x00")
                    self._buf.clear()
                else:
                    logger.debug("[tts]  -> standalone source exhausted, returning empty")
                    return b""
            else:
                return bytes(DISCORD_FRAME_SIZE)

        if self._clip_buffer is not None:
            self._clip_buffer.append(np.frombuffer(data, dtype=np.int16).copy())
        return data

    def after_playback(self, error):
        if error:
            logger.error(f"[tts]  x standalone playback error: {error}")
        self._finished_evt.set()

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        pass
