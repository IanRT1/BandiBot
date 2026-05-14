"""
tts.py — Text-to-speech using Deepgram Aura-2 with PCM mixing.

MixerSource holds the music stream and injects TTS on top.
Music never pauses or restarts — both streams play simultaneously.
TTS can be cancelled mid-stream via cancel_tts() for wake word interruptions.

Deepgram linear16 format: 48kHz, 16-bit signed, mono.
Discord needs:            48kHz, 16-bit signed, stereo.
"""

import asyncio
import logging
import os
import time
import threading
from typing import Optional

import aiohttp
import discord
import numpy as np

logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY")
TTS_MODEL          = "aura-2-javier-es"
TTS_SAMPLE_RATE    = 48000
DISCORD_FRAME_SIZE = 3840
MUSIC_DUCK_VOLUME  = 0.3


class MixerSource(discord.AudioSource):
    """
    Audio source that mixes a primary source (music) with an
    optional secondary source (TTS).

    Music plays continuously. TTS is injected on top when speak() is called.
    Both are audible simultaneously — no pausing or restarting.
    TTS can be cancelled mid-stream via cancel().
    """

    def __init__(self, primary: discord.AudioSource):
        self._primary      = primary
        self._tts_buf      = bytearray()
        self._tts_done     = False
        self._tts_active   = False
        self._cancelled    = False
        self._lock         = threading.Lock()
        self._tts_finished = threading.Event()

    def feed_tts(self, pcm_mono: bytes):
        if not pcm_mono:
            return
        with self._lock:
            if self._cancelled:
                return
        samples = np.frombuffer(pcm_mono, dtype=np.int16)
        stereo  = np.empty(len(samples) * 2, dtype=np.int16)
        stereo[0::2] = samples
        stereo[1::2] = samples
        with self._lock:
            self._tts_active = True
            self._tts_buf.extend(stereo.tobytes())

    def finish_tts(self):
        with self._lock:
            self._tts_done = True

    def cancel(self):
        with self._lock:
            self._tts_buf.clear()
            self._tts_done   = True
            self._cancelled  = True
            self._tts_active = False
        self._tts_finished.set()
        logger.info("[tts]  ← cancelled (wake word interrupt)")

    def reset_cancel(self):
        with self._lock:
            self._cancelled = False

    def wait_tts_done(self, timeout: float = 30.0) -> bool:
        return self._tts_finished.wait(timeout=timeout)

    def read(self) -> bytes:
        music_frame = self._primary.read()
        if not music_frame:
            return b""

        music = np.frombuffer(music_frame, dtype=np.int16).astype(np.int32)

        with self._lock:
            has_tts  = len(self._tts_buf) >= DISCORD_FRAME_SIZE
            tts_done = self._tts_done and len(self._tts_buf) < DISCORD_FRAME_SIZE

        if has_tts:
            with self._lock:
                tts_frame = bytes(self._tts_buf[:DISCORD_FRAME_SIZE])
                del self._tts_buf[:DISCORD_FRAME_SIZE]
            tts   = np.frombuffer(tts_frame, dtype=np.int16).astype(np.int32)
            mixed = (music * MUSIC_DUCK_VOLUME).astype(np.int32) + tts
            mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
            return mixed.tobytes()

        elif tts_done and self._tts_active:
            with self._lock:
                self._tts_active = False
                self._tts_done   = False
                self._cancelled  = False
            self._tts_finished.set()
            return music_frame

        else:
            return music_frame

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        self._primary.cleanup()


class _StandaloneSource(discord.AudioSource):
    """Standalone TTS source for when no music is playing. Cancellable."""

    def __init__(self):
        self._buf          = bytearray()
        self._lock         = threading.Lock()
        self._done         = False
        self._cancelled    = False
        self._finished_evt = threading.Event()

    def feed(self, pcm_mono: bytes):
        with self._lock:
            if self._cancelled:
                return
        samples = np.frombuffer(pcm_mono, dtype=np.int16)
        stereo  = np.empty(len(samples) * 2, dtype=np.int16)
        stereo[0::2] = samples
        stereo[1::2] = samples
        with self._lock:
            self._buf.extend(stereo.tobytes())

    def cancel(self):
        with self._lock:
            self._buf.clear()
            self._done      = True
            self._cancelled = True
        self._finished_evt.set()
        logger.info("[tts]  ← standalone cancelled (wake word interrupt)")

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
                return data
            elif self._done:
                logger.info("[tts]  → standalone source exhausted, returning empty")
                return b""
            else:
                return bytes(DISCORD_FRAME_SIZE)

    def _after_playback(self, error):
        """Called by discord audio thread when playback ends."""
        if error:
            logger.error(f"[tts]  ✗ standalone playback error: {error}")
        self._finished_evt.set()

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        pass

def cancel_tts(voice_client: discord.VoiceClient):
    """
    Cancel any in-progress TTS immediately.
    Works for both MixerSource (music+TTS) and standalone TTS.
    """
    if not voice_client or not voice_client.is_connected():
        return

    source = getattr(voice_client, 'source', None)
    if isinstance(source, MixerSource):
        source.cancel()
        return

    # Standalone TTS — cancel via stored reference
    standalone = getattr(voice_client, '_standalone_tts', None)
    if standalone is not None and isinstance(standalone, _StandaloneSource):
        standalone.cancel()
        voice_client._standalone_tts = None
        try:
            voice_client.stop_playing()
        except Exception:
            pass


async def speak(
    voice_client: discord.VoiceClient,
    text: str,
    guild=None,
):
    """
    Stream TTS from Deepgram and mix it into the currently playing music.
    Music continues playing throughout — no pausing or restarting.
    If no music is playing, plays TTS standalone.
    Cancellable via cancel_tts().
    """
    if not voice_client or not voice_client.is_connected():
        return
    if not text or not text.strip():
        return

    t_start = time.perf_counter()
    logger.info(f"[tts]  → speaking ({len(text)} chars)")

    source   = getattr(voice_client, 'source', None)
    is_mixer = isinstance(source, MixerSource) and voice_client.is_playing()

    if not is_mixer:
        await _speak_standalone(voice_client, text, t_start)
        return

    mixer: MixerSource = source
    mixer.reset_cancel()
    mixer._tts_finished.clear()

    url = (
        f"https://api.deepgram.com/v1/speak"
        f"?model={TTS_MODEL}&encoding=linear16"
        f"&sample_rate={TTS_SAMPLE_RATE}&container=none&speed=1.3"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"text": text}) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"[tts]  ✗ Deepgram {resp.status}: {body}")
                    return

                first_chunk = True
                async for chunk in resp.content.iter_chunked(4096):
                    with mixer._lock:
                        cancelled = mixer._cancelled
                    if cancelled:
                        logger.info("[tts]  → stream cancelled mid-flight")
                        return
                    if chunk:
                        if first_chunk:
                            t_first = (time.perf_counter() - t_start) * 1000
                            logger.info(f"[tts]  → first chunk in {t_first:.0f}ms")
                            first_chunk = False
                        mixer.feed_tts(chunk)

        mixer.finish_tts()
        logger.info("[tts]  → all chunks streamed, waiting for playback")

    except Exception as e:
        logger.error(f"[tts]  ✗ streaming error: {e}")
        mixer.finish_tts()
        return

    await asyncio.get_event_loop().run_in_executor(
        None, lambda: mixer.wait_tts_done(timeout=30.0)
    )

    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(f"[tts]  ← done ({elapsed:.0f}ms total)")


async def _speak_standalone(voice_client: discord.VoiceClient, text: str, t_start: float):
    """
    Play TTS standalone when no music is playing.
    Uses threading.Event for reliable cancellation from any thread.
    """

    url = (
        f"https://api.deepgram.com/v1/speak"
        f"?model={TTS_MODEL}&encoding=linear16"
        f"&sample_rate={TTS_SAMPLE_RATE}&container=none&speed=1.3"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }

    source = _StandaloneSource()

    voice_client._standalone_tts = source

    logger.info(f"[tts]  → standalone play | is_playing={voice_client.is_playing()} is_connected={voice_client.is_connected()} source={voice_client.source}")
    try:
        voice_client.play(source, after=source._after_playback)
        logger.info(f"[tts]  → standalone play succeeded")
    except Exception as e:
        logger.error(f"[tts]  ✗ standalone play failed: {e}")
        source.set_done()
        return

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"text": text}) as resp:
                if resp.status != 200:
                    source.set_done()
                    return
                first_chunk = True
                async for chunk in resp.content.iter_chunked(4096):
                    if source.is_cancelled():
                        logger.info("[tts]  → standalone stream cancelled mid-flight")
                        return
                    if chunk:
                        if first_chunk:
                            t_first = (time.perf_counter() - t_start) * 1000
                            logger.info(f"[tts]  → first chunk in {t_first:.0f}ms")
                            first_chunk = False
                        source.feed(chunk)
        source.set_done()
        logger.info("[tts]  → all chunks streamed, waiting for playback")
    except asyncio.CancelledError:
        source.cancel()
        return
    except Exception as e:
        logger.error(f"[tts]  ✗ {e}")
        source.set_done()

    await asyncio.get_event_loop().run_in_executor(
        None, lambda: source._finished_evt.wait(timeout=60.0)
    )

    voice_client._standalone_tts = None
    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(f"[tts]  ← done ({elapsed:.0f}ms total)")


async def play_activation(voice_client: discord.VoiceClient):
    """Play wake activation sound. Mixes with music if playing."""
    if not voice_client or not voice_client.is_connected():
        return
    wav_path = os.path.join(os.path.dirname(__file__), "wake_activation.wav")
    if not os.path.exists(wav_path):
        logger.warning("[tts]  ✗ wake_activation.wav not found")
        return

    source = getattr(voice_client, 'source', None)
    if isinstance(source, MixerSource) and voice_client.is_playing():
        import wave as wave_mod
        try:
            with wave_mod.open(wav_path, 'rb') as wf:
                pcm = wf.readframes(wf.getnframes())
            source.reset_cancel()
            source._tts_finished.clear()
            chunk_size = 4096
            for i in range(0, len(pcm), chunk_size):
                source.feed_tts(pcm[i:i + chunk_size])
            source.finish_tts()
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: source.wait_tts_done(timeout=10.0)
            )
        except Exception as e:
            logger.error(f"[tts]  ✗ activation error: {e}")
    else:
        if voice_client.is_playing():
            voice_client.stop_playing()
            await asyncio.sleep(0.2)
        done = asyncio.Event()
        def _after(error):
            done.set()
        try:
            voice_client.play(discord.FFmpegPCMAudio(wav_path), after=_after)
            await done.wait()
        except Exception as e:
            logger.error(f"[tts]  ✗ activation play failed: {e}")
            done.set()

    logger.info("[tts]  ← activation sound done")