"""
voice/tts_sources.py

Discord AudioSource implementations for BandiBot text-to-speech playback.

This module owns the low-level audio buffering that Discord pulls from in its
voice thread. It does not call TTS providers or decide what text to say; it
only accepts 48kHz int16 mono PCM, expands it to Discord stereo frames, and
serves cancellable audio frames to the voice client.

Playback modes:
  MixerSource      -> wraps active music, ducks it, and injects TTS on top
  StandaloneSource -> feeds TTS directly when idle; can attach a music mixer
                      in place without restarting Discord playback or speech

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
DUCK_ATTACK_SECONDS = 0.100
DUCK_RELEASE_SECONDS = 0.300
PCM_SAMPLE_RATE = 48000


class MixerSource(discord.AudioSource):
    """
    Audio source that mixes a primary source (music) with optional TTS.

    Music plays continuously. TTS is injected on top when speak() feeds PCM
    into the source. Music is ducked during speech, and TTS can be cancelled
    mid-stream without stopping the music source. Music gain ramps down over
    100ms and recovers over 300ms, staying ducked through streaming gaps.
    """

    def __init__(self, primary: discord.AudioSource, clip_buffer=None):
        self._primary = primary
        self._duck_gain = 1.0
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
            tts_frame = None
            if has_tts:
                tts_frame = bytes(self._tts_buf[:DISCORD_FRAME_SIZE])
                del self._tts_buf[:DISCORD_FRAME_SIZE]
            elif self._tts_done and self._tts_active:
                self._tts_active = False
                self._tts_done = False
                self._tts_finished.set()
            # Keep ducking between provider chunks until speech ends or is cancelled.
            ducking = has_tts or self._tts_active

        target = MUSIC_DUCK_VOLUME if ducking else 1.0
        duration = DUCK_ATTACK_SECONDS if target < self._duck_gain else DUCK_RELEASE_SECONDS
        step = (1.0 - MUSIC_DUCK_VOLUME) / (PCM_SAMPLE_RATE * duration)
        sample_count = len(music) // 2
        offsets = np.arange(1, sample_count + 1) * step
        if target < self._duck_gain:
            gains = np.maximum(target, self._duck_gain - offsets)
        else:
            gains = np.minimum(target, self._duck_gain + offsets)
        self._duck_gain = float(gains[-1])
        mixed = music * np.repeat(gains, 2)
        if tts_frame is not None:
            mixed += np.frombuffer(tts_frame, dtype=np.int16).astype(np.int32)
        frame = np.clip(np.rint(mixed), -32768, 32767).astype(np.int16).tobytes()

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
            remainder = len(self._tts_buf) % DISCORD_FRAME_SIZE
            if remainder:
                self._tts_buf.extend(bytes(DISCORD_FRAME_SIZE - remainder))
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
        self.mixer = None
        self._music_after = None
        self._exhausted = False

    def attach_music(self, mixer, after) -> bool:
        """Promote in place so Discord keeps reading speech while music starts."""
        with self._lock:
            if self._exhausted or self.mixer is not None:
                return False
            mixer._tts_buf = self._buf
            mixer._tts_done = self._done
            mixer._tts_active = True
            mixer._cancelled = self._cancelled
            mixer._tts_finished = self._finished_evt
            if self._done:
                mixer.finish_tts()
            self._music_after = after
            self.mixer = mixer
            return True

    def feed(self, pcm_mono: bytes):
        with self._lock:
            if self.mixer is not None:
                self.mixer.feed_tts(pcm_mono)
                return
            if self._cancelled:
                return
        samples = np.frombuffer(pcm_mono, dtype=np.int16)
        stereo = mono_to_stereo(samples)
        with self._lock:
            if self.mixer is not None:
                self.mixer.feed_tts(pcm_mono)
                return
            self._buf.extend(stereo.tobytes())

    def cancel(self):
        with self._lock:
            if self.mixer is not None:
                self.mixer.cancel()
            self._buf.clear()
            self._done = True
            self._cancelled = True
        self._finished_evt.set()
        logger.debug("[tts]  <- standalone cancelled (wake word interrupt)")

    def set_done(self):
        with self._lock:
            if self.mixer is not None:
                self.mixer.finish_tts()
            self._done = True

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def read(self) -> bytes:
        with self._lock:
            if self.mixer is not None:
                return self.mixer.read()
            if len(self._buf) >= DISCORD_FRAME_SIZE:
                data = bytes(self._buf[:DISCORD_FRAME_SIZE])
                del self._buf[:DISCORD_FRAME_SIZE]
            elif self._done:
                if self._buf:
                    data = bytes(self._buf).ljust(DISCORD_FRAME_SIZE, b"\x00")
                    self._buf.clear()
                else:
                    logger.debug("[tts]  -> standalone source exhausted, returning empty")
                    self._exhausted = True
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
        if self._music_after is not None:
            self._music_after(error)

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        if self.mixer is not None:
            self.mixer._primary.cleanup()
