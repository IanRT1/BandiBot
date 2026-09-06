"""
core/interaction_logging.py

Shared interaction logging for BandiBot's text chat and voice pipelines.

Keeps INFO output consistent across both modes with labeled user messages,
bot replies, and a separate completion line showing total elapsed time.

Usage:
  Collects provider-reported token totals and ElevenLabs credit charges
  (falling back to input characters when unavailable), other TTS input chars,
  and submitted WAV duration for successful STT requests. These are usage
  measurements, not billing credits or costs. Includes calls
  in awaited child tasks. Completion lines list only sources with usage.
  Context-local tracking keeps concurrent interactions independent and is
  reset on success, failure, or cancellation.

Log levels:
  INFO  -> compact, single-line message previews and interaction completion
  DEBUG -> speaker names, character counts, and full message content

Privacy:
  LOG_SENSITIVE_CONTENT controls speaker names and message content at both
  levels. When disabled, only message roles and character counts are logged
  for message events; completion timing remains available.
"""

import logging
import math
from contextvars import ContextVar
from functools import wraps

from core.config import LOG_SENSITIVE_CONTENT

_usage: ContextVar[dict[tuple[str, str], int | float] | None] = ContextVar("interaction_usage", default=None)


def track_usage(func):
    """Isolate each interaction while sharing totals with its awaited tasks."""
    @wraps(func)
    async def wrapped(*args, **kwargs):
        token = _usage.set({})
        try:
            return await func(*args, **kwargs)
        finally:
            _usage.reset(token)
    return wrapped


def record_token_usage(source: str, total: int | None):
    """Accumulate provider-reported totals; never estimate missing usage."""
    if isinstance(total, int) and not isinstance(total, bool):
        record_usage(source, "tokens", total)


def record_usage(source: str, unit: str, amount: int | float | None):
    """Sum observed usage by provider and unit without mixing billing units."""
    usage = _usage.get()
    if (usage is not None and isinstance(amount, (int, float))
            and not isinstance(amount, bool) and math.isfinite(amount)
            and (amount > 0 or (unit == "credits" and amount == 0))):
        key = (source, unit)
        usage[key] = usage.get(key, 0) + amount


def log_message(logger: logging.Logger, mode: str, role: str, name: str, text: str):
    arrow = "→" if role == "user" else "←"
    if LOG_SENSITIVE_CONTENT:
        preview = " ".join(text.split())
        if len(preview) > 160:
            preview = preview[:159] + "…"
        logger.info("[%s]  %s %s (%s): %s", mode, arrow, role, " ".join(name.split()), preview)
        logger.debug("[%s] %s message | speaker=%r | chars=%d | content=%r", mode, role, name, len(text), text)
    else:
        logger.info("[%s]  %s %s message", mode, arrow, role)
        logger.debug("[%s] %s message | chars=%d", mode, role, len(text))


def log_done(logger: logging.Logger, mode: str, total_ms: float):
    usage = _usage.get()
    suffix = ""
    if usage:
        source_order = {"openai": 0, "gemini": 1, "elevenlabs": 2, "deepgram": 3}
        suffix = " | " + " | ".join(
            f"{source}={amount:.1f}s audio" if unit == "audio_seconds"
            else f"{source}={amount:g} {unit}"
            for (source, unit), amount in sorted(
                usage.items(),
                key=lambda item: (source_order.get(item[0][0], 4), item[0]),
            )
        )
    logger.info("[%s]  <- done (%.0fms total)%s", mode, total_ms, suffix)
