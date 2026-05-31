"""
voice/tts_providers.py

Provider adapters for BandiBot text-to-speech synthesis.

This module owns external and model-specific TTS work. It converts text into
48kHz int16 mono PCM chunks that voice.tts can feed into Discord audio sources.
It does not know whether playback is mixed with music or standalone.

Providers:
  deepgram -> streams Aura-2 linear16 PCM from the Deepgram HTTP API
  kokoro   -> generates local 24kHz float32 PCM and resamples it to 48kHz int16

Switching:
  The active provider is selected by core.config.TTS_PROVIDER. That value is a
  code-level operational choice, not a secret; API keys remain in the env.
"""

import logging

import aiohttp

from core.config import DEEPGRAM_API_KEY
from voice.audio import float32_24k_to_int16_48k

logger = logging.getLogger(__name__)

# Deepgram settings
TTS_MODEL = "aura-2-javier-es"
TTS_SAMPLE_RATE = 48000
DEEPGRAM_SPEED = 1.3

# Kokoro settings
KOKORO_VOICE = "ef_dora"
KOKORO_LANG = "e"
KOKORO_SPEED = 1.1
KOKORO_SAMPLE_RATE = 24000
KOKORO_CHUNK_SIZE = 4096

_kokoro_pipeline = None


def load_tts_provider(provider: str) -> None:
    """Initialize provider resources that should be ready before first speech."""
    if provider != "kokoro":
        return
    _get_kokoro_pipeline()


async def stream_deepgram_pcm(text: str):
    """Yield Deepgram Aura-2 48kHz int16 mono PCM chunks."""
    url = (
        "https://api.deepgram.com/v1/speak"
        f"?model={TTS_MODEL}&encoding=linear16"
        f"&sample_rate={TTS_SAMPLE_RATE}&container=none&speed={DEEPGRAM_SPEED}"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json={"text": text}) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"[tts]  x Deepgram {resp.status}: {body}")
                return

            async for chunk in resp.content.iter_chunked(4096):
                if chunk:
                    yield chunk


def generate_kokoro_pcm(text: str) -> list[bytes]:
    """Return Kokoro-generated 48kHz int16 mono PCM chunks."""
    pipeline = _get_kokoro_pipeline()
    chunks = []
    for _, _, audio in pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
        resampled = float32_24k_to_int16_48k(audio)
        chunks.append(resampled.tobytes())
    return chunks


def _get_kokoro_pipeline():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline

        logger.info("[tts]  -> loading Kokoro pipeline (this may take a moment on first run)...")
        _kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)
        logger.info("[tts]  -> Kokoro pipeline ready")
    return _kokoro_pipeline
