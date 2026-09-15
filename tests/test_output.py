import asyncio
from types import SimpleNamespace

import numpy as np

from voice.output import AudioOutput, OutputSource, Speech, SILENCE
from voice.results import SpeechResult


class PCM:
    def __init__(self, value=1000, frames=10):
        self.value, self.frames, self.cleaned = value, frames, False

    def read(self):
        if self.frames == 0:
            return b""
        self.frames -= 1
        return np.full(1920, self.value, dtype=np.int16).tobytes()

    def cleanup(self):
        self.cleaned = True


def test_pause_preserves_music_position_while_speech_plays_and_clip_records_once():
    clips = []
    output = OutputSource(clips)
    output.set_music(PCM(frames=3), lambda error: None)
    output.read()
    output.music_paused = True
    speech = Speech(output.generation, bytearray(np.full(1920, 2000, dtype=np.int16).tobytes()), True)
    output.speech = speech
    frame = output.read()
    assert np.all(np.frombuffer(frame, dtype=np.int16) == 2000)
    assert output.music_frames == 1
    assert speech.finished.is_set()
    assert len(clips) == 2
    output.music_paused = False
    output.read()
    assert output.music_frames == 2


def test_music_eof_does_not_end_speech_or_outer_source():
    output = OutputSource()
    ended = []
    output.set_music(PCM(frames=0), lambda error: ended.append(error))
    speech = Speech(output.generation, bytearray(SILENCE * 2), True)
    output.speech = speech
    assert len(output.read()) == len(SILENCE)
    assert ended == [None]
    assert not speech.finished.is_set()
    assert len(output.read()) == len(SILENCE)
    assert speech.finished.is_set()
    assert ended == [None]
    assert output.read() == SILENCE


def test_wake_discards_buffered_speech_and_preserves_music():
    output = OutputSource()
    music = PCM()
    output.set_music(music, lambda error: None)
    speech = Speech(output.generation, bytearray(SILENCE))
    output.speech = speech
    output.set_activation(PCM(value=2000, frames=1))
    assert speech.result == SpeechResult.INTERRUPTED
    assert speech.finished.is_set()
    assert not speech.buffer
    assert output.music is music
    output.read()
    output.read()
    assert output.activation_done.is_set()


def test_delayed_provider_audio_cannot_survive_interruption():
    async def run():
        client = SimpleNamespace(source=None, is_playing=lambda: client.source is not None,
                                 is_paused=lambda: False)
        client.play = lambda source, after: setattr(client, "source", source)
        output = AudioOutput(client)
        started, release = asyncio.Event(), asyncio.Event()

        async def provider(text):
            started.set()
            await release.wait()
            yield np.ones(960, dtype=np.int16).tobytes()

        task = asyncio.create_task(output.speak("hello", provider))
        await started.wait()
        output.source.cancel_speech()
        release.set()
        assert await task == SpeechResult.INTERRUPTED
        assert output.source.read() == SILENCE
    asyncio.run(run())


def test_ducking_holds_provider_gaps_and_ramps_back_after_cancellation():
    output = OutputSource()
    output.set_music(PCM(value=10000, frames=100), lambda error: None)
    output.speech = Speech(output.generation)  # Provider hasn't delivered a chunk yet.
    for _ in range(6):
        output.read()
    quiet = np.frombuffer(output.read(), dtype=np.int16)
    assert np.all(quiet == 3000)
    output.cancel_speech()
    first = np.frombuffer(output.read(), dtype=np.int16)
    assert first[0] == 3000 and first[-1] < 10000
    for _ in range(20):
        output.read()
    assert np.all(np.frombuffer(output.read(), dtype=np.int16) == 10000)
