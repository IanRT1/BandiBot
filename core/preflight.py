"""
core/preflight.py

Compact startup validation for BandiBot's local runtime and configuration.

The preflight intentionally emits one summary instead of logging every check.
It distinguishes required failures, which prevent a Discord connection, from
optional warnings, which disable only the affected feature.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("DISCORD_TOKEN", "OPENAI_API_KEY", "DEEPGRAM_API_KEY")
_VALID_TTS_PROVIDERS = {"kokoro", "deepgram", "elevenlabs"}


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of the startup checks used by the client entry point."""

    ok: bool
    summary: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def run_preflight(
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    warmup: Callable[[], str] | None = None,
) -> PreflightResult:
    """Validate startup prerequisites and emit a compact operational summary."""
    root = project_root or Path(__file__).resolve().parents[1]
    env = os.environ if environ is None else environ
    summary: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    missing_keys = [key for key in _REQUIRED_KEYS if not env.get(key)]
    if missing_keys:
        errors.append(f"missing required keys: {', '.join(missing_keys)}")
    else:
        summary.append("apis=ok")

    tts_provider = env.get("TTS_PROVIDER", "kokoro").lower()
    if tts_provider not in _VALID_TTS_PROVIDERS:
        errors.append("invalid TTS_PROVIDER")
    elif tts_provider == "elevenlabs" and not env.get("ELEVENLABS_API_KEY"):
        warnings.append("ElevenLabs key missing; Kokoro fallback will be used")
        summary.append("tts=fallback")
    else:
        summary.append(f"tts={tts_provider}")

    if shutil.which("ffmpeg"):
        summary.append("ffmpeg=ok")
    else:
        warnings.append("ffmpeg unavailable; music and voice playback may fail")
        summary.append("ffmpeg=missing")

    if shutil.which("node"):
        summary.append("node=ok")
    else:
        warnings.append("node unavailable; YouTube challenge solving may fail")
        summary.append("node=missing")

    assets = root / "assets"
    missing_assets = [
        name for name in ("BandiBot.onnx", "wake_activation.wav")
        if not (assets / name).is_file()
    ]
    if missing_assets:
        warnings.append(f"missing voice assets: {', '.join(missing_assets)}")
        summary.append("voice-assets=missing")
    else:
        summary.append("voice-assets=ok")

    data = root / "data"
    missing_context = [
        name for name in ("instructions.txt", "server_info.txt")
        if not ((data / name).is_file() or (data / name.replace(".txt", ".example.txt")).is_file())
    ]
    if missing_context:
        errors.append(f"missing context files: {', '.join(missing_context)}")
    else:
        summary.append("context=ok")

    if not errors and warmup is not None:
        started = time.perf_counter()
        try:
            retrieval_status = warmup()
        except Exception:
            retrieval_status = "fallback"
            logger.debug("[startup] retrieval warm-up failed", exc_info=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        summary.append(f"retrieval={retrieval_status} ({elapsed_ms:.0f}ms)")

    result = PreflightResult(
        ok=not errors,
        summary=tuple(summary),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
    if result.ok:
        logger.info("[startup] preflight passed | %s", " | ".join(result.summary))
    else:
        logger.error("[startup] preflight failed | %s", "; ".join(result.errors))
    if result.warnings:
        logger.warning("[startup] preflight warnings | %s", "; ".join(result.warnings))
    return result
