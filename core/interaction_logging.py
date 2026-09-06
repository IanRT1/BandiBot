"""
core/interaction_logging.py

Shared interaction logging for BandiBot's text chat and voice pipelines.

Keeps INFO output consistent across both modes with labeled user messages,
bot replies, and a separate completion line showing total elapsed time.

Token usage:
  Each interaction collects provider-reported token totals, including calls
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
from contextvars import ContextVar
from functools import wraps

from core.config import LOG_SENSITIVE_CONTENT

_token_usage: ContextVar[dict[str, int] | None] = ContextVar("interaction_token_usage", default=None)


def track_token_usage(func):
    """Isolate each interaction while sharing totals with its awaited tasks."""
    @wraps(func)
    async def wrapped(*args, **kwargs):
        token = _token_usage.set({})
        try:
            return await func(*args, **kwargs)
        finally:
            _token_usage.reset(token)
    return wrapped


def record_token_usage(source: str, total: int | None):
    """Accumulate provider-reported totals; never estimate missing usage."""
    usage = _token_usage.get()
    if usage is not None and isinstance(total, int) and not isinstance(total, bool) and total > 0:
        usage[source] = usage.get(source, 0) + total


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
    usage = _token_usage.get()
    suffix = ""
    if usage:
        suffix = " | tokens: " + ", ".join(
            f"{source}={total}" for source, total in sorted(usage.items())
        )
    logger.info("[%s]  <- done (%.0fms total)%s", mode, total_ms, suffix)
