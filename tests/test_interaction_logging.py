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

from core.interaction_logging import log_done, record_token_usage, track_token_usage


logger = logging.getLogger(__name__)


def test_token_totals_include_child_tasks_and_keep_interactions_isolated(caplog):
    async def child():
        record_token_usage("gemini", 30)

    @track_token_usage
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
    assert "[chat]  <- done (100ms total) | tokens: gemini=30, openai=15" in caplog.messages
    assert "[voice]  <- done (100ms total) | tokens: openai=25" in caplog.messages
    assert "[outside]  <- done (1ms total)" in caplog.messages


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_token_tracking_resets_after_failure(caplog, error):
    @track_token_usage
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
    @track_token_usage
    async def run():
        for value in (None, 0, -1, "12", True):
            record_token_usage("unknown", value)
        log_done(logger, "voice", 2)

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    assert caplog.messages == ["[voice]  <- done (2ms total)"]
