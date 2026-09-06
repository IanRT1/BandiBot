import asyncio

import pytest

from voice.audio import resample_int16_mono
from voice.tts_providers import TTS_PROVIDERS, _format_api_error


def test_idle_speech_promotes_to_ducked_music_without_losing_audio():
    import numpy as np
    from voice.tts_sources import MixerSource, StandaloneSource, DISCORD_FRAME_SIZE

    class Music:
        def cleanup(self):
            pass

        def read(self):
            return np.full(DISCORD_FRAME_SIZE // 2, 1000, dtype=np.int16).tobytes()

    source = StandaloneSource()
    pcm = np.full(DISCORD_FRAME_SIZE // 4, 2000, dtype=np.int16).tobytes()
    source.feed(pcm)
    mixer = MixerSource(Music())
    assert source.attach_music(mixer, lambda error: None)
    frame = np.frombuffer(source.read(), dtype=np.int16)
    assert 2300 <= frame[-1] < frame[0] <= 3000
    source.feed(pcm)  # Provider chunks continue arriving after handoff.
    source.set_done()
    frame = np.frombuffer(source.read(), dtype=np.int16)
    assert 2300 <= frame[-1] < frame[0] <= 3000
    for _ in range(15):
        frame = np.frombuffer(source.read(), dtype=np.int16)
    assert frame[-1] == 1000
    assert source._finished_evt.is_set()


def test_promoted_speech_cancellation_keeps_music_and_completion_callback():
    import numpy as np
    from voice.tts_sources import MixerSource, StandaloneSource, DISCORD_FRAME_SIZE

    class Music:
        def cleanup(self):
            pass

        def read(self):
            return np.full(DISCORD_FRAME_SIZE // 2, 1000, dtype=np.int16).tobytes()

    source = StandaloneSource()
    completed = []
    assert source.attach_music(MixerSource(Music()), completed.append)
    source.cancel()
    assert np.all(np.frombuffer(source.read(), dtype=np.int16) == 1000)
    assert completed == []
    source.after_playback(None)
    assert completed == [None]


def test_exhausted_standalone_cannot_accept_music():
    from voice.tts_sources import StandaloneSource

    source = StandaloneSource()
    source.set_done()
    assert source.read() == b""
    assert not source.attach_music(object(), None)


@pytest.mark.parametrize("cancelled", [False, True])
def test_ducking_ramps_holds_streaming_gaps_and_recovers(cancelled):
    import numpy as np
    from voice.tts_sources import MixerSource, DISCORD_FRAME_SIZE

    class Music:
        def read(self):
            return np.full(DISCORD_FRAME_SIZE // 2, 10000, dtype=np.int16).tobytes()

    mixer = MixerSource(Music())
    assert np.all(np.frombuffer(mixer.read(), dtype=np.int16) == 10000)
    mixer.feed_tts(bytes(DISCORD_FRAME_SIZE // 2 * 5))
    attack = np.concatenate([np.frombuffer(mixer.read(), dtype=np.int16) for _ in range(5)])
    assert attack[0] > 9900
    assert attack[-1] == 3000
    assert np.all(np.diff(attack.astype(int)) <= 0)
    # No chunks available yet, but provider has not finished.
    for _ in range(3):
        assert np.all(np.frombuffer(mixer.read(), dtype=np.int16) == 3000)
    if cancelled:
        mixer.cancel()
    else:
        mixer.finish_tts()
    release = np.concatenate([np.frombuffer(mixer.read(), dtype=np.int16) for _ in range(15)])
    assert release[0] < 3100
    assert release[-1] == 10000
    assert np.all(np.diff(release.astype(int)) >= 0)
    assert np.array_equal(release[::2], release[1::2])


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
