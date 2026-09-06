"""
tests/test_interaction_logging.py

Offline checks for interaction token accounting and completion logs.

Coverage:
  Repeated provider calls, awaited child tasks, concurrent interactions,
  missing usage, and cleanup after failure or cancellation.
"""

import asyncio
import logging

import pytest

from core.interaction_logging import log_done, record_token_usage, record_usage, track_usage


logger = logging.getLogger(__name__)


def test_token_totals_include_child_tasks_and_keep_interactions_isolated(caplog):
    async def child():
        record_token_usage("gemini", 30)
        record_usage("deepgram", "audio_seconds", 1.2)
        record_usage("deepgram", "audio_seconds", 3.0)
        record_usage("elevenlabs", "chars", 85)

    @track_usage
    async def interaction(mode, amount):
        record_token_usage("openai", amount)
        await asyncio.sleep(0)
        record_token_usage("openai", 5)
        if mode == "chat":
            await asyncio.create_task(child())
        log_done(logger, mode, 100)

    async def run():
        await asyncio.gather(interaction("chat", 10), interaction("voice", 20))
        log_done(logger, "outside", 1)

    with caplog.at_level(logging.DEBUG):
        asyncio.run(run())
    assert "[chat]  <- done (100ms total) | openai=15 tokens | gemini=30 tokens | elevenlabs=85 chars | deepgram=4.2s audio" in caplog.messages
    assert "[voice]  <- done (100ms total) | openai=25 tokens" in caplog.messages
    assert "[outside]  <- done (1ms total)" in caplog.messages


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_token_tracking_resets_after_failure(caplog, error):
    @track_usage
    async def failed():
        record_token_usage("openai", 123)
        raise error()

    async def run():
        with pytest.raises(error):
            await failed()
        log_done(logger, "chat", 1)

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    assert caplog.messages == ["[chat]  <- done (1ms total)"]


def test_missing_or_invalid_usage_is_omitted(caplog):
    @track_usage
    async def run():
        for value in (None, 0, -1, "12", True):
            record_token_usage("unknown", value)
        log_done(logger, "voice", 2)

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    assert caplog.messages == ["[voice]  <- done (2ms total)"]


@pytest.mark.parametrize("header, expected", [
    ("42.5", "42.5 credits"),
    ("0", "0 credits"),
    (None, "5 chars"),
    ("invalid", "5 chars"),
    ("nan", "5 chars"),
    ("inf", "5 chars"),
    ("-2", "5 chars"),
])
def test_elevenlabs_reported_cost_and_missing_cost_fallback(caplog, header, expected):
    from voice.tts_providers import _record_elevenlabs_usage

    @track_usage
    async def run():
        _record_elevenlabs_usage({"character-cost": header}, "hello")
        log_done(logger, "voice", 10)

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    assert caplog.messages == [f"[voice]  <- done (10ms total) | elevenlabs={expected}"]
