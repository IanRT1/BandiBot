"""
stt.py — Speech-to-text using Deepgram Nova-3.
Accepts WAV bytes, returns transcribed text string.
"""

import logging
import os
import time

from deepgram import DeepgramClient, PrerecordedOptions

logger = logging.getLogger(__name__)

_DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not _DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")

_client = DeepgramClient(_DEEPGRAM_API_KEY)

_OPTIONS = PrerecordedOptions(
    model="nova-3",
    language="es",
    smart_format=True,
    punctuate=True,
    utterances=False,
)


async def transcribe(wav_bytes: bytes) -> str:
    try:
        t = time.perf_counter()
        logger.info(f"[stt]  → transcribing {len(wav_bytes) // 1000}KB")
        payload = {"buffer": wav_bytes, "mimetype": "audio/wav"}
        response = await _client.listen.asyncprerecorded.v("1").transcribe_file(payload, _OPTIONS)
        alt = response.results.channels[0].alternatives[0]
        transcript = alt.transcript
        confidence = getattr(alt, "confidence", 0.0)
        elapsed = (time.perf_counter() - t) * 1000
        if transcript:
            logger.info(f"[stt]  ← {transcript!r} (conf={confidence:.2f}, {elapsed:.0f}ms)")
        else:
            logger.info(f"[stt]  ✗ empty (conf={confidence:.2f}, {elapsed:.0f}ms)")
        return transcript.strip()
    except Exception as e:
        logger.error(f"[stt]  ✗ error: {e}")
        return ""