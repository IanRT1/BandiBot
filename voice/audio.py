"""
voice/audio.py

Pure PCM audio helpers for BandiBot's voice pipeline.

Discord voice receive, wake word detection, VAD, STT, TTS mixing, and clip
export all use slightly different audio shapes. This module owns the small,
stateless conversions between those shapes so listener/session logic can stay
focused on state transitions.

Formats:
  Discord receive/playback → 48kHz, 16-bit signed, stereo PCM
  Wake word / VAD input    → 16kHz mono PCM or float32 mono samples
  STT capture              → WAV bytes encoded from 48kHz mono PCM
"""

import io
import wave

import numpy as np


def stereo_to_mono(samples: np.ndarray) -> np.ndarray:
    """Downmix interleaved int16 stereo PCM into int16 mono PCM."""
    if len(samples) % 2 == 0:
        left = samples[0::2].astype(np.int32)
        right = samples[1::2].astype(np.int32)
        return ((left + right) >> 1).astype(np.int16)
    return samples


def mono_to_stereo(samples: np.ndarray) -> np.ndarray:
    """Duplicate int16 mono samples into interleaved int16 stereo PCM."""
    stereo = np.empty(len(samples) * 2, dtype=np.int16)
    stereo[0::2] = samples
    stereo[1::2] = samples
    return stereo


def mono48k_to_16k(samples: np.ndarray) -> np.ndarray:
    """Downsample int16 48kHz mono PCM to int16 16kHz mono PCM."""
    n = (len(samples) // 3) * 3
    return samples[:n].reshape(-1, 3).mean(axis=1).astype(np.int16)


def to_float32(samples: np.ndarray) -> np.ndarray:
    """Normalize int16 PCM samples into float32 samples in the -1.0..1.0 range."""
    return samples.astype(np.float32) / 32768.0


def float32_24k_to_int16_48k(samples: np.ndarray) -> np.ndarray:
    """Resample float32 24kHz mono audio to int16 48kHz mono PCM."""
    original_len = len(samples)
    target_len = original_len * 2
    indices = np.linspace(0, original_len - 1, target_len)
    resampled = np.interp(indices, np.arange(original_len), samples)
    return (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16)


def resample_int16_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono int16 PCM between rates used by TTS providers."""
    if source_rate == target_rate or not len(samples):
        return samples.astype(np.int16, copy=False)
    target_len = max(1, round(len(samples) * target_rate / source_rate))
    indices = np.linspace(0, len(samples) - 1, target_len)
    resampled = np.interp(indices, np.arange(len(samples)), samples.astype(np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode int16 mono PCM samples as an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()
