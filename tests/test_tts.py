import asyncio

import pytest

from voice.audio import resample_int16_mono
from voice.tts_providers import TTS_PROVIDERS, _format_api_error


def test_all_tts_providers_are_registered():
    assert {"kokoro", "deepgram", "elevenlabs"} <= TTS_PROVIDERS.keys()


def test_provider_error_details_are_readable_without_secrets():
    error = {
        "detail": {
            "type": "authentication_error",
            "code": "invalid_api_key",
            "message": "The API key is invalid",
            "status": "error",
        }
    }
    assert _format_api_error(error) == (
        "authentication_error | invalid_api_key | "
        "The API key is invalid | error"
    )


def test_pcm_resampling_doubles_24khz_sample_count():
    import numpy as np

    samples = np.array([0, 1000, -1000, 2000], dtype=np.int16)
    converted = resample_int16_mono(samples, 24000, 48000)
    assert len(converted) == 8
    assert converted.dtype == np.int16


def test_failed_remote_provider_falls_back_to_kokoro(monkeypatch):
    import voice.tts as tts

    class BrokenProvider:
        async def stream_pcm(self, text):
            raise RuntimeError("plan limit")
            yield  # Keep this an async generator.

    class FallbackProvider:
        async def stream_pcm(self, text):
            yield b"kokoro-audio"

    monkeypatch.setattr(tts, "TTS_PROVIDER", "elevenlabs")
    monkeypatch.setattr(tts, "_provider", BrokenProvider())
    monkeypatch.setattr(tts, "create_tts_provider", lambda name: FallbackProvider())

    async def collect():
        return [chunk async for chunk in tts._iter_provider_pcm("hello")]

    chunks = asyncio.run(collect())

    assert chunks == [b"kokoro-audio"]


def test_partial_remote_audio_does_not_repeat_with_fallback(monkeypatch):
    import voice.tts as tts

    class PartialProvider:
        async def stream_pcm(self, text):
            yield b"partial-audio"
            raise RuntimeError("connection lost")

    monkeypatch.setattr(tts, "TTS_PROVIDER", "elevenlabs")
    monkeypatch.setattr(tts, "_provider", PartialProvider())

    async def collect():
        return [chunk async for chunk in tts._iter_provider_pcm("hello")]

    with pytest.raises(RuntimeError, match="connection lost"):
        asyncio.run(collect())
