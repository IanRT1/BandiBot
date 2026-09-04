"""
voice/tts.py

Text-to-speech orchestration for BandiBot.

This module chooses the configured TTS provider, routes generated PCM into the
right Discord audio source, and handles cancellation. Provider-specific code
lives in voice.tts_providers; low-level AudioSource buffering and music mixing
live in voice.tts_sources.

Playback modes:
  Mixer mode      -> inject TTS into MixerSource while music continues playing
  Standalone mode -> play a dedicated StandaloneSource when no music is active

Switching providers:
  Change TTS_PROVIDER in .env to "kokoro", "deepgram", or "elevenlabs" and
  restart the bot. Provider-specific code is hidden behind one PCM stream API.

Public API:
  speak()           -> generate and play TTS for a connected voice client
  cancel_tts()      -> cancel in-progress mixer or standalone TTS
  play_activation() -> play the wake activation sound
  MixerSource       -> re-exported for music.player compatibility
"""

import asyncio
import logging
import os
import time
import wave

import discord

from core.config import TTS_PROVIDER
from voice.tts_providers import (
    KOKORO_CHUNK_SIZE,
    create_tts_provider,
)
from voice.tts_sources import MixerSource, StandaloneSource

logger = logging.getLogger(__name__)

_provider = create_tts_provider(TTS_PROVIDER)


def cancel_tts(voice_client: discord.VoiceClient):
    """Cancel any in-progress TTS immediately."""
    if not voice_client or not voice_client.is_connected():
        return

    source = getattr(voice_client, "source", None)
    if isinstance(source, MixerSource):
        source.cancel()
        return

    standalone = getattr(voice_client, "_standalone_tts", None)
    if isinstance(standalone, StandaloneSource):
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
    clip_buffer=None,
):
    if not voice_client or not voice_client.is_connected():
        return
    if not text or not text.strip():
        return

    t_start = time.perf_counter()
    logger.info(f"[tts]  -> speaking ({len(text)} chars) via {TTS_PROVIDER}")

    source = getattr(voice_client, "source", None)
    is_mixer = isinstance(source, MixerSource) and voice_client.is_playing()

    if is_mixer:
        await _speak_mixer(source, text, t_start)
    else:
        await _speak_standalone(voice_client, text, t_start, clip_buffer=clip_buffer)


async def _speak_mixer(mixer: MixerSource, text: str, t_start: float):
    mixer.reset_cancel()
    mixer._tts_finished.clear()

    try:
        first_chunk = True
        async for chunk in _iter_provider_pcm(text):
            with mixer._lock:
                cancelled = mixer._cancelled
            if cancelled:
                logger.info("[tts]  -> stream cancelled mid-flight")
                return

            if first_chunk:
                _log_first_chunk(t_start)
                first_chunk = False

            for feed_chunk in _iter_feed_chunks(chunk):
                mixer.feed_tts(feed_chunk)

        mixer.finish_tts()
        logger.debug("[tts]  -> all chunks streamed, waiting for playback")

    except Exception as e:
        logger.error(f"[tts]  x {TTS_PROVIDER} error: {e}")
        mixer.finish_tts()
        return

    await asyncio.get_event_loop().run_in_executor(
        None, lambda: mixer.wait_tts_done(timeout=30.0)
    )
    _log_done(t_start)


async def _speak_standalone(
    voice_client: discord.VoiceClient,
    text: str,
    t_start: float,
    clip_buffer=None,
):
    source = StandaloneSource(clip_buffer=clip_buffer)
    voice_client._standalone_tts = source

    logger.debug(
        "[tts]  -> standalone play | "
        f"is_playing={voice_client.is_playing()} "
        f"is_connected={voice_client.is_connected()} "
        f"source={voice_client.source}"
    )
    try:
        voice_client.play(source, after=source.after_playback)
        logger.debug("[tts]  -> standalone play succeeded")
    except Exception as e:
        logger.error(f"[tts]  x standalone play failed: {e}")
        source.set_done()
        voice_client._standalone_tts = None
        return

    try:
        try:
            first_chunk = True
            async for chunk in _iter_provider_pcm(text):
                if source.is_cancelled():
                    logger.info("[tts]  -> standalone stream cancelled mid-flight")
                    return

                if first_chunk:
                    _log_first_chunk(t_start)
                    first_chunk = False

                for feed_chunk in _iter_feed_chunks(chunk):
                    source.feed(feed_chunk)

            source.set_done()
            logger.info("[tts]  -> all chunks streamed, waiting for playback")

        except asyncio.CancelledError:
            source.cancel()
            return
        except Exception as e:
            logger.error(f"[tts]  x {TTS_PROVIDER} standalone error: {e}")
            source.set_done()

        await asyncio.get_event_loop().run_in_executor(
            None, lambda: source._finished_evt.wait(timeout=60.0)
        )
        _log_done(t_start)
    finally:
        if getattr(voice_client, "_standalone_tts", None) is source:
            voice_client._standalone_tts = None


async def _iter_provider_pcm(text: str):
    """Stream the selected provider, falling back to local Kokoro on failure."""
    yielded = False
    try:
        async for chunk in _provider.stream_pcm(text):
            yielded = True
            yield chunk
    except Exception as exc:
        # Restarting from Kokoro after partial audio would repeat speech, so
        # only fail over when the provider failed before producing audio.
        if yielded or TTS_PROVIDER == "kokoro":
            raise

        logger.warning(
            "[tts] %s failed before audio (%s); falling back to kokoro",
            TTS_PROVIDER,
            exc,
        )
        fallback = create_tts_provider("kokoro")
        async for chunk in fallback.stream_pcm(text):
            yield chunk


def _iter_feed_chunks(chunk: bytes):
    chunk_size = KOKORO_CHUNK_SIZE * 2 if TTS_PROVIDER == "kokoro" else len(chunk)
    for i in range(0, len(chunk), chunk_size):
        yield chunk[i:i + chunk_size]


async def play_activation(voice_client: discord.VoiceClient):
    """Play wake activation sound, mixing with music when possible."""
    if not voice_client or not voice_client.is_connected():
        return

    wav_path = os.path.join(os.path.dirname(__file__), "..", "assets", "wake_activation.wav")
    if not os.path.exists(wav_path):
        logger.warning("[tts]  x wake_activation.wav not found")
        return

    source = getattr(voice_client, "source", None)
    if isinstance(source, MixerSource) and voice_client.is_playing():
        await _play_activation_mixed(source, wav_path)
    else:
        await _play_activation_standalone(voice_client, wav_path)

    logger.info("[tts]  <- activation sound done")


async def _play_activation_mixed(source: MixerSource, wav_path: str):
    try:
        with wave.open(wav_path, "rb") as wf:
            pcm = wf.readframes(wf.getnframes())

        source.reset_cancel()
        source._tts_finished.clear()
        for i in range(0, len(pcm), 4096):
            source.feed_tts(pcm[i:i + 4096])
        source.finish_tts()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: source.wait_tts_done(timeout=10.0)
        )
    except Exception as e:
        logger.error(f"[tts]  x activation error: {e}")


async def _play_activation_standalone(voice_client: discord.VoiceClient, wav_path: str):
    if voice_client.is_playing():
        voice_client.stop_playing()
        await asyncio.sleep(0.2)

    done = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _after(error):
        loop.call_soon_threadsafe(done.set)

    try:
        voice_client.play(discord.FFmpegPCMAudio(wav_path), after=_after)
        await done.wait()
    except Exception as e:
        logger.error(f"[tts]  x activation play failed: {e}")
        done.set()


def _log_first_chunk(t_start: float):
    t_first = (time.perf_counter() - t_start) * 1000
    logger.info(f"[tts]  -> first chunk in {t_first:.0f}ms")


def _log_done(t_start: float):
    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(f"[tts]  <- done ({elapsed:.0f}ms total)")
