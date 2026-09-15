"""
voice/tts.py

Text-to-speech orchestration for BandiBot.

This module chooses the configured TTS provider, routes generated PCM into the
right Discord audio source, and handles cancellation. Provider-specific code
lives in voice.tts_providers; low-level AudioSource buffering and music mixing
live in voice.output.

Playback:
  One output owner mixes music, speech, and activation independently.

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

import discord

from core.config import TTS_PROVIDER
from core.paths import assets_root
from voice.tts_providers import (
    create_tts_provider,
)
from voice.tts_sources import MixerSource, StandaloneSource

logger = logging.getLogger(__name__)

_provider = create_tts_provider(TTS_PROVIDER)


def cancel_tts(voice_client: discord.VoiceClient):
    from voice.output import existing_output
    output = existing_output(voice_client)
    if output:
        output.source.cancel_speech()


async def speak(voice_client: discord.VoiceClient, text: str, guild=None, clip_buffer=None):
    from voice.output import get_output
    from voice.results import SpeechResult
    if not voice_client or not voice_client.is_connected() or not text.strip():
        return SpeechResult.FAILED
    output = get_output(voice_client, clip_buffer)
    started = time.perf_counter()
    result = await output.speak(text, _iter_provider_pcm)
    logger.debug("[tts] delivery=%s total=%.0fms", result.value, (time.perf_counter() - started) * 1000)
    return result


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


async def play_activation(voice_client: discord.VoiceClient):
    from voice.output import get_output
    if not voice_client or not voice_client.is_connected():
        return
    wav_path = str(assets_root() / "wake_activation.wav")
    if not os.path.isfile(wav_path):
        return
    output = get_output(voice_client)
    try:
        output.ensure_started()
        output.source.set_activation(discord.FFmpegPCMAudio(wav_path))
        while not output.source.activation_done.is_set():
            await asyncio.sleep(.02)
    except Exception as exc:
        logger.error("[tts] activation playback failed: %s", exc)
