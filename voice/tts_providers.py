"""Provider adapters and registry for BandiBot text-to-speech.

Every provider yields 48 kHz, 16-bit, mono PCM. Playback and music mixing do
not need to know which service generated the audio.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from typing import Protocol

import aiohttp
import numpy as np

from core.config import (
    DEEPGRAM_API_KEY,
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL,
    ELEVENLABS_VOICE_ID,
)
from voice.audio import float32_24k_to_int16_48k, resample_int16_mono
from core.interaction_logging import record_usage

logger = logging.getLogger(__name__)

TTS_SAMPLE_RATE = 48000
DEEPGRAM_MODEL = "aura-2-javier-es"
DEEPGRAM_SPEED = 1.3
KOKORO_VOICE = "ef_dora"
KOKORO_LANG = "e"
KOKORO_SPEED = 1.1
KOKORO_CHUNK_SIZE = 4096
ELEVENLABS_PCM_RATE = 24000


class TTSProviderError(RuntimeError):
    """A provider could not synthesize speech."""


class TTSProvider(Protocol):
    name: str

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Yield 48 kHz, 16-bit, mono PCM chunks."""


class DeepgramProvider:
    name = "deepgram"

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        url = (
            "https://api.deepgram.com/v1/speak"
            f"?model={DEEPGRAM_MODEL}&encoding=linear16"
            f"&sample_rate={TTS_SAMPLE_RATE}&container=none&speed={DEEPGRAM_SPEED}"
        )
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"text": text}) as resp:
                if resp.status != 200:
                    try:
                        data = await resp.json(content_type=None)
                    except (aiohttp.ContentTypeError, ValueError):
                        data = await resp.text()
                    raise TTSProviderError(
                        f"Deepgram HTTP {resp.status}: {_format_api_error(data)}"
                    )
                record_usage(self.name, "chars", len(text))
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        yield chunk


class KokoroProvider:
    name = "kokoro"
    _pipeline = None

    async def initialize(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._get_pipeline)

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        chunks = await asyncio.get_running_loop().run_in_executor(None, self._generate, text)
        for chunk in chunks:
            yield chunk

    def _generate(self, text: str) -> list[bytes]:
        chunks = []
        for _, _, audio in self._get_pipeline()(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
            chunks.append(float32_24k_to_int16_48k(audio).tobytes())
        return chunks

    @classmethod
    def _get_pipeline(cls):
        if cls._pipeline is None:
            from kokoro import KPipeline

            logger.info("[tts] -> loading Kokoro pipeline")
            cls._pipeline = KPipeline(lang_code=KOKORO_LANG)
            logger.info("[tts] -> Kokoro pipeline ready")
        return cls._pipeline


class ElevenLabsProvider:
    name = "elevenlabs"

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        url = (
            "https://api.elevenlabs.io/v1/text-to-speech/"
            f"{ELEVENLABS_VOICE_ID}/stream?output_format=pcm_24000"
        )
        headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
        payload = {"text": text, "model_id": ELEVENLABS_MODEL}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    try:
                        data = await resp.json(content_type=None)
                    except (aiohttp.ContentTypeError, ValueError):
                        data = await resp.text()
                    raise TTSProviderError(
                        f"ElevenLabs HTTP {resp.status}: {_format_api_error(data)}"
                    )
                _record_elevenlabs_usage(resp.headers, text)
                pending = bytearray()
                async for chunk in resp.content.iter_chunked(4096):
                    if not chunk:
                        continue
                    pending.extend(chunk)
                    even_length = len(pending) - len(pending) % 2
                    if even_length:
                        samples = np.frombuffer(pending[:even_length], dtype=np.int16).copy()
                        del pending[:even_length]
                        yield resample_int16_mono(
                            samples, ELEVENLABS_PCM_RATE, TTS_SAMPLE_RATE
                        ).tobytes()


TTS_PROVIDERS: dict[str, type] = {
    "deepgram": DeepgramProvider,
    "kokoro": KokoroProvider,
    "elevenlabs": ElevenLabsProvider,
}


def _record_elevenlabs_usage(headers, text: str):
    """Use reported generation cost; retain input chars if cost is unavailable."""
    try:
        credits = float(headers.get("character-cost"))
    except (TypeError, ValueError):
        credits = None
    if credits is not None and math.isfinite(credits) and credits >= 0:
        record_usage("elevenlabs", "credits", credits)
        logger.debug("[tts] elevenlabs usage | chars=%d | credits=%g", len(text), credits)
    else:
        record_usage("elevenlabs", "chars", len(text))
        logger.debug("[tts] elevenlabs usage | chars=%d | credits=unavailable", len(text))


def create_tts_provider(name: str) -> TTSProvider:
    try:
        return TTS_PROVIDERS[name]()
    except KeyError as exc:
        raise RuntimeError(f"Unsupported TTS provider: {name!r}") from exc


async def load_tts_provider(provider: TTSProvider) -> None:
    initialize = getattr(provider, "initialize", None)
    if initialize:
        await initialize()


def _format_api_error(data) -> str:
    """Expose provider diagnostics without logging headers or request payloads."""
    error = data.get("detail", data.get("error", data)) if isinstance(data, dict) else data
    if isinstance(error, dict):
        parts = [str(error[key]) for key in ("type", "code", "message", "status") if error.get(key)]
        if parts:
            return " | ".join(parts)[:500]
    return str(error)[:500]
