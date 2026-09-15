"""Test-only environment defaults loaded before importing application config."""

import os
import pytest


@pytest.fixture(autouse=True)
def validated_optional_capabilities(monkeypatch):
    """Most unit tests model a healthy startup without contacting providers."""
    monkeypatch.setattr("core.capabilities._enabled_optional_tools", frozenset({"web_search"}))


def pytest_configure():
    os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
    os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
    os.environ.setdefault("DEEPGRAM_API_KEY", "test-deepgram-key")
    os.environ.setdefault("BANDIBOT_DISABLE_SEMANTIC_RAG", "1")
