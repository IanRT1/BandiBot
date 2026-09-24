"""Opt-in local recording of accepted wake-word examples.

Only the short in-memory pre-roll is retained. A WAV is queued after an
accepted wake-word detection, and file I/O runs away from Discord's audio
callback thread.
"""

from __future__ import annotations

import queue
import threading
import uuid
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
PRE_ROLL_SECONDS = 4.0
SAVED_SAMPLE_SECONDS = 1.5
FRAME_SAMPLES = int(SAMPLE_RATE * 0.02)
SPEECH_RMS_THRESHOLD = 0.01
PRESERVE_BEFORE_SECONDS = 0.4
TARGET_RMS = 0.075
MAX_GAIN = 4.0


class WakewordSampleRecorder:
    """Collect recent audio and write accepted detections as mono WAV files."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self._recent: dict[int, deque[np.ndarray]] = {}
        self._jobs: queue.Queue[tuple[Path, np.ndarray] | None] = queue.Queue()
        self._worker = threading.Thread(target=self._write_loop, daemon=True)
        self._worker.start()

    def observe(self, user_id: int, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.int16).reshape(-1).copy()
        if not len(samples):
            return
        frames = self._recent.setdefault(user_id, deque())
        frames.append(samples)
        total = sum(len(frame) for frame in frames)
        limit = int(PRE_ROLL_SECONDS * SAMPLE_RATE)
        while total > limit and frames:
            total -= len(frames.popleft())

    def record_detection(self, user_id: int, guild_id: int, score: float) -> None:
        frames = self._recent.get(user_id)
        if not frames:
            return
        samples = np.concatenate(tuple(frames))
        samples = samples[-int(SAVED_SAMPLE_SECONDS * SAMPLE_RATE):]
        samples = self._trim_leading_silence(samples)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        if rms > 1.0:
            gain = min(MAX_GAIN, TARGET_RMS * 32768.0 / rms)
            samples = np.clip(samples.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        filename = (
            f"{timestamp}_guild-{guild_id}_user-{user_id}_score-{score:.3f}_"
            f"{uuid.uuid4().hex[:8]}.wav"
        )
        self._jobs.put((self.output_dir / filename, samples))

    def close(self) -> None:
        self._jobs.put(None)

    @staticmethod
    def _trim_leading_silence(samples: np.ndarray) -> np.ndarray:
        """Keep a short lead-in, removing silence before the utterance."""
        frame_count = len(samples) // FRAME_SAMPLES
        if not frame_count:
            return samples
        frames = samples[:frame_count * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
        rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1)) / 32768.0
        active = np.flatnonzero(rms >= SPEECH_RMS_THRESHOLD)
        if not len(active):
            return samples
        keep_before = int(PRESERVE_BEFORE_SECONDS * SAMPLE_RATE)
        start = max(0, active[0] * FRAME_SAMPLES - keep_before)
        return samples[start:]

    def _write_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            path, samples = job
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(path), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(SAMPLE_RATE)
                    audio.writeframes(samples.astype(np.int16).tobytes())
            except OSError:
                # Dataset capture must never affect voice recognition.
                continue
