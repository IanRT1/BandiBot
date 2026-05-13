"""
tts.py — Text-to-speech using Deepgram Aura-2 with streaming PCM playback.

Streams raw linear16 PCM from Deepgram directly into Discord's audio pipeline.
Sub-200ms TTFB vs OpenAI's 1-3s — much faster for voice agents.

Deepgram linear16 format: 48kHz, 16-bit signed, mono.
Discord needs:            48kHz, 16-bit signed, stereo.

We just duplicate mono → stereo, no resampling needed.
"""

import asyncio
import logging
import os
import time
import threading

import aiohttp
import discord
import numpy as np

logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
TTS_MODEL        = "aura-2-javier-es"
TTS_SAMPLE_RATE  = 48000   # request 48kHz so no resampling needed
DISCORD_FRAME_SIZE = 3840  # 20ms of 48kHz stereo int16


class _StreamingPCMAudio(discord.AudioSource):
    """
    Discord AudioSource fed by streaming PCM chunks from Deepgram.
    discord.py calls read() every 20ms from the audio thread.
    We feed chunks from the asyncio thread via a thread-safe buffer.
    """

    def __init__(self):
        self._buf   = bytearray()
        self._lock  = threading.Lock()
        self._done  = False

    def feed(self, pcm_mono: bytes):
        """Feed mono int16 PCM chunk. Called from asyncio thread."""
        if not pcm_mono:
            return
        samples   = np.frombuffer(pcm_mono, dtype=np.int16)
        stereo    = np.empty(len(samples) * 2, dtype=np.int16)
        stereo[0::2] = samples
        stereo[1::2] = samples
        with self._lock:
            self._buf.extend(stereo.tobytes())

    def set_done(self):
        with self._lock:
            self._done = True

    def read(self) -> bytes:
        with self._lock:
            if len(self._buf) >= DISCORD_FRAME_SIZE:
                data = bytes(self._buf[:DISCORD_FRAME_SIZE])
                del self._buf[:DISCORD_FRAME_SIZE]
                return data
            elif self._done:
                return b""
            else:
                return bytes(DISCORD_FRAME_SIZE)  # silence during buffer underrun

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        pass


async def speak(
    voice_client: discord.VoiceClient,
    text: str,
    guild: discord.Guild = None,
):
    """
    Stream TTS audio from Deepgram Aura-2 directly into Discord.
    Sub-200ms TTFB — playback starts almost immediately.
    If music was playing, restarts it after TTS finishes.
    """
    if not voice_client or not voice_client.is_connected():
        return
    if not text or not text.strip():
        return

    t_start = time.perf_counter()
    logger.info(f"[tts]  → speaking ({len(text)} chars)")

    source     = _StreamingPCMAudio()
    done_event = asyncio.Event()

    was_playing = voice_client.is_playing()
    if was_playing:
        voice_client.pause()

    def _after(error):
        if error:
            logger.error(f"[tts]  ✗ playback error: {error}")
        done_event.set()

    voice_client.play(source, after=_after)

    url = f"https://api.deepgram.com/v1/speak?model={TTS_MODEL}&encoding=linear16&sample_rate={TTS_SAMPLE_RATE}&container=none&speed=1.3"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"[tts]  ✗ Deepgram error {resp.status}: {body}")
                    source.set_done()
                else:
                    first_chunk = True
                    async for chunk in resp.content.iter_chunked(4096):
                        if chunk:
                            if first_chunk:
                                t_first = (time.perf_counter() - t_start) * 1000
                                logger.info(f"[tts]  → first chunk in {t_first:.0f}ms")
                                first_chunk = False
                            source.feed(chunk)
                    source.set_done()
                    logger.info("[tts]  → all chunks streamed, waiting for playback")

    except Exception as e:
        logger.error(f"[tts]  ✗ streaming error: {e}")
        source.set_done()

    await done_event.wait()

    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(f"[tts]  ← done ({elapsed:.0f}ms total)")

    # Restart music if it was playing
    if was_playing and voice_client.is_connected() and guild:
        from music import voice_manager
        player = voice_manager.get_player(guild)
        if player.current and not player.is_playing:
            logger.info("[music] restarting after TTS")
            player.queue.appendleft(player.current)
            player.current = None
            player.play_next()


async def play_activation(voice_client: discord.VoiceClient):
    """Play wake activation sound. Non-blocking — caller does not await."""
    if not voice_client or not voice_client.is_connected():
        return
    wav_path = os.path.join(os.path.dirname(__file__), "wake_activation.wav")
    if not os.path.exists(wav_path):
        logger.warning("[tts]  ✗ wake_activation.wav not found")
        return
    done = asyncio.Event()
    def _after(error):
        done.set()
    voice_client.play(discord.FFmpegPCMAudio(wav_path), after=_after)
    await done.wait()
    logger.info("[tts]  ← activation sound done")