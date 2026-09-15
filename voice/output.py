"""The sole owner of Discord output, with independent music, speech and cues."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

import discord
import numpy as np

from voice.audio import mono_to_stereo
from voice.results import SpeechResult
from voice.tts_sources import (
    DISCORD_FRAME_SIZE, MUSIC_DUCK_VOLUME, DUCK_ATTACK_SECONDS, DUCK_RELEASE_SECONDS,
)

SILENCE = bytes(DISCORD_FRAME_SIZE)
MAX_SPEECH_BYTES = 48000 * 2 * 2 * 10


@dataclass
class Speech:
    generation: int
    buffer: bytearray = field(default_factory=bytearray)
    ended: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    result: SpeechResult = SpeechResult.COMPLETED


class OutputSource(discord.AudioSource):
    """Stable PCM source. Callbacks are delivered outside its short state lock."""
    def __init__(self, clip_buffer=None):
        self.lock = threading.RLock()
        self.music = None
        self.music_after = None
        self.music_paused = False
        self.music_frames = 0
        self.music_exhausted = False
        self.speech: Speech | None = None
        self.generation = 0
        self.activation = None
        self.activation_done = threading.Event()
        self.activation_done.set()
        self.closed = False
        self.clip_buffer = clip_buffer
        self.gain = 1.0

    def is_opus(self):
        return False

    def cancel_speech(self):
        with self.lock:
            self.generation += 1
            if self.speech:
                self.speech.buffer.clear()
                self.speech.result = SpeechResult.INTERRUPTED
                self.speech.finished.set()
                self.speech = None

    def set_music(self, source, after):
        self.stop_music()
        with self.lock:
            self.music, self.music_after = source, after
            self.music_paused = False
            self.music_frames = 0
            self.music_exhausted = False

    def stop_music(self):
        with self.lock:
            source, after = self.music, self.music_after
            self.music = self.music_after = None
            self.music_paused = False
        if source:
            source.cleanup()
        if after:
            after(None)

    def set_activation(self, source):
        self.cancel_speech()
        with self.lock:
            old = self.activation
            self.activation = source
            self.activation_done.clear()
        if old:
            old.cleanup()

    def read(self):
        # Discord calls this from its audio thread. Do not mutate guild state.
        with self.lock:
            if self.closed:
                return b""
            primary = self.music if not self.music_paused else None
            cue = self.activation
        error = None
        try:
            music = primary.read() if primary else SILENCE
        except Exception as exc:
            music, error = b"", exc
        try:
            cue_pcm = cue.read() if cue else b""
        except Exception:
            cue_pcm = b""
        after = None
        cleanup = []
        with self.lock:
            if primary is not self.music:
                music = SILENCE
            elif primary and not music:
                self.music_exhausted = True
                after = self.music_after
                self.music = self.music_after = None
                cleanup.append(primary)
            elif primary:
                self.music_frames += 1
            if cue is not self.activation:
                cue_pcm = b""
            elif cue and not cue_pcm:
                self.activation = None
                self.activation_done.set()
                cleanup.append(cue)
            speech_pcm = b""
            speech = self.speech
            if speech and speech.generation == self.generation and not self.activation:
                speech_pcm = bytes(speech.buffer[:DISCORD_FRAME_SIZE])
                del speech.buffer[:DISCORD_FRAME_SIZE]
                if speech.ended and not speech.buffer:
                    speech.finished.set()
                    self.speech = None
            speaking = bool(speech_pcm or (speech and not speech.finished.is_set()) or cue_pcm)
            target = MUSIC_DUCK_VOLUME if speaking else 1.0
            seconds = DUCK_ATTACK_SECONDS if target < self.gain else DUCK_RELEASE_SECONDS
            step = (1.0 - MUSIC_DUCK_VOLUME) * .02 / seconds
            new_gain = max(target, self.gain - step) if target < self.gain else min(target, self.gain + step)
            gains = np.repeat(np.linspace(self.gain, new_gain, 960, endpoint=False), 2)
            self.gain = new_gain

            def samples(pcm):
                return np.frombuffer((pcm or b"")[:DISCORD_FRAME_SIZE].ljust(DISCORD_FRAME_SIZE, b"\0"), dtype=np.int16).astype(np.int32)

            mixed = samples(music) * gains + samples(speech_pcm) + samples(cue_pcm)
            frame = np.clip(np.rint(mixed), -32768, 32767).astype(np.int16)
            if self.clip_buffer is not None:
                self.clip_buffer.append(frame.copy())
        for source in cleanup:
            source.cleanup()
        if after:
            after(error)
        return frame.tobytes()

    def cleanup(self):
        self.cancel_speech()
        with self.lock:
            if self.closed:
                return
            self.closed = True
            sources = [self.music, self.activation]
            self.music = self.music_after = self.activation = None
            self.activation_done.set()
        for source in sources:
            if source:
                source.cleanup()


class AudioOutput:
    def __init__(self, voice_client, clip_buffer=None):
        self.client = voice_client
        self.source = OutputSource(clip_buffer)
        self.speech_lock = asyncio.Lock()

    def ensure_started(self):
        if getattr(self.client, "source", None) is self.source and self.client.is_playing():
            return
        # This also makes ownership handover explicit: never replace a live owner.
        if self.client.is_playing() or self.client.is_paused():
            raise RuntimeError("Discord audio is owned by another output source")
        self.client.play(self.source, after=self._after)

    def _after(self, error):
        with self.source.lock:
            after = self.source.music_after
        self.source.cleanup()
        if after:
            after(error)

    async def speak(self, text, pcm_stream, *, generation=None):
        generation = self.source.generation if generation is None else generation
        async with self.speech_lock:
            try:
                self.ensure_started()
                while not self.source.activation_done.is_set():
                    if generation != self.source.generation or self.source.closed:
                        return SpeechResult.STALE
                    await asyncio.sleep(.02)
                with self.source.lock:
                    if generation != self.source.generation or self.source.closed:
                        return SpeechResult.STALE
                    speech = Speech(generation)
                    self.source.speech = speech
                async for mono in pcm_stream(text):
                    stereo = mono_to_stereo(np.frombuffer(mono, dtype=np.int16)).tobytes()
                    for start in range(0, len(stereo), DISCORD_FRAME_SIZE):
                        while len(speech.buffer) >= MAX_SPEECH_BYTES:
                            if speech.finished.is_set():
                                return speech.result
                            await asyncio.sleep(.02)
                        with self.source.lock:
                            if generation != self.source.generation or speech.finished.is_set():
                                return SpeechResult.INTERRUPTED
                            speech.buffer.extend(stereo[start:start + DISCORD_FRAME_SIZE])
                with self.source.lock:
                    speech.ended = True
                while not speech.finished.is_set():
                    await asyncio.sleep(.02)
                return speech.result
            except asyncio.CancelledError:
                self.source.cancel_speech()
                raise
            except Exception:
                self.source.cancel_speech()
                return SpeechResult.FAILED


def get_output(voice_client, clip_buffer=None) -> AudioOutput:
    output = getattr(voice_client, "_bandibot_output", None)
    if output is None or output.source.closed:
        output = AudioOutput(voice_client, clip_buffer)
        voice_client._bandibot_output = output
    elif clip_buffer is not None:
        output.source.clip_buffer = clip_buffer
    return output


def existing_output(voice_client):
    return getattr(voice_client, "_bandibot_output", None) if voice_client else None


def music_active(voice_client):
    output = existing_output(voice_client)
    return bool(output and output.source.music is not None)


def music_paused(voice_client):
    output = existing_output(voice_client)
    return bool(output and output.source.music_paused)


def stop_music(voice_client):
    output = existing_output(voice_client)
    if output:
        output.source.stop_music()


def pause_music(voice_client, paused=True):
    output = existing_output(voice_client)
    if output:
        with output.source.lock:
            output.source.music_paused = paused
