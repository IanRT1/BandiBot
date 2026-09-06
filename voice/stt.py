"""
voice/stt.py

Speech-to-text for BandiBot using Deepgram Nova-3.

Accepts raw WAV bytes captured from the Discord voice pipeline and returns a
transcribed text string. Configured for multilingual code-switching so users can
speak Spanish, English, or mixed Spanglish without the recognizer rejecting
speech because it was constrained to a single language.

Pipeline:
  WAV bytes -> Deepgram Nova-3 prerecorded API -> transcript string

The client is initialized once at module load using DEEPGRAM_API_KEY from the
environment. All errors are caught and logged; an empty string is returned on
failure so the voice pipeline can reset gracefully.

Language: multi — supports code-switching across Nova-3 multilingual languages.
Model: nova-3 — Deepgram's general-purpose speech recognition model.
"""

import logging
import time

from deepgram import DeepgramClient, PrerecordedOptions

from core.config import DEEPGRAM_API_KEY

logger = logging.getLogger(__name__)

_client = DeepgramClient(DEEPGRAM_API_KEY)
STT_LANGUAGE = "multi"

_OPTIONS = PrerecordedOptions(
    model="nova-3",
    language=STT_LANGUAGE,
    smart_format=True,
    punctuate=True,
    utterances=False,
)


async def transcribe(wav_bytes: bytes) -> str:
    try:
        t = time.perf_counter()
        logger.debug(f"[stt]  → transcribing {len(wav_bytes) // 1000}KB")
        payload = {"buffer": wav_bytes, "mimetype": "audio/wav"}
        response = await _client.listen.asyncprerecorded.v("1").transcribe_file(payload, _OPTIONS)
        alt = response.results.channels[0].alternatives[0]
        transcript = alt.transcript
        confidence = getattr(alt, "confidence", 0.0)
        elapsed = (time.perf_counter() - t) * 1000
        if transcript:
            logger.info(
                f"[stt]  ← {transcript!r} "
                f"(lang={STT_LANGUAGE}, conf={confidence:.2f}, {elapsed:.0f}ms)"
            )
        else:
            logger.info(
                f"[stt]  ✗ empty "
                f"(lang={STT_LANGUAGE}, conf={confidence:.2f}, {elapsed:.0f}ms)"
            )
        return transcript.strip()
    except Exception as e:
        logger.error(f"[stt]  ✗ error: {e}")
        return ""
