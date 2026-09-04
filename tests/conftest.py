"""Test-only environment defaults loaded before importing application config."""

import os


def pytest_configure():
    os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
    os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
    os.environ.setdefault("DEEPGRAM_API_KEY", "test-deepgram-key")
