"""Exercise real FFmpeg PCM through the output engine without Discord/network."""
import shutil
import wave

import discord
import numpy as np
import pytest

from voice.output import OutputSource, SILENCE


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_real_ffmpeg_eof_and_clip_frames(tmp_path):
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        t = np.arange(4800) / 48000
        tone = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)
        audio.writeframes(np.repeat(tone, 2).tobytes())
    source = discord.FFmpegPCMAudio(str(path))
    clips, completed = [], []
    output = OutputSource(clips)
    output.set_music(source, lambda error: completed.append(error))
    try:
        frames = [output.read() for _ in range(8)]
        assert all(len(frame) == len(SILENCE) for frame in frames)
        assert any(frame != SILENCE for frame in frames)
        assert completed == [None]
        assert output.music_frames == 5
        assert len(clips) == len(frames)
        assert output.read() == SILENCE
    finally:
        output.cleanup()
