"""Offline tests for cleanup of runtime voice and music resources."""

import asyncio
from types import SimpleNamespace

from music.player import VoiceManager
from voice.listener import VoiceListenerManager


def test_music_shutdown_disconnects_players_and_clears_state():
    manager = VoiceManager()
    first = SimpleNamespace(guild=SimpleNamespace(name="first"))
    second = SimpleNamespace(guild=SimpleNamespace(name="second"))
    disconnected = []

    async def disconnect(player):
        disconnected.append(player.guild.name)

    first.disconnect = lambda: disconnect(first)
    second.disconnect = lambda: disconnect(second)
    manager._players = {1: first, 2: second}
    manager._play_locks = {1: asyncio.Lock(), 2: asyncio.Lock()}

    asyncio.run(manager.shutdown())

    assert disconnected == ["first", "second"]
    assert manager._players == {}
    assert manager._play_locks == {}


def test_music_shutdown_continues_after_one_player_cleanup_fails():
    manager = VoiceManager()
    failed = SimpleNamespace(guild=SimpleNamespace(name="failed"))
    healthy = SimpleNamespace(guild=SimpleNamespace(name="healthy"))
    disconnected = []

    async def failed_disconnect():
        raise RuntimeError("cleanup failure")

    async def healthy_disconnect():
        disconnected.append("healthy")

    failed.disconnect = failed_disconnect
    healthy.disconnect = healthy_disconnect
    manager._players = {1: failed, 2: healthy}

    asyncio.run(manager.shutdown())

    assert disconnected == ["healthy"]
    assert manager._players == {}


def test_voice_shutdown_stops_sessions_and_clears_state():
    manager = VoiceListenerManager()
    stopped = []

    class Session:
        async def stop(self):
            stopped.append(True)

    manager._sessions = {1: Session(), 2: Session()}

    asyncio.run(manager.shutdown())

    assert stopped == [True, True]
    assert manager._sessions == {}


def test_voice_shutdown_continues_after_one_session_fails():
    manager = VoiceListenerManager()
    stopped = []

    class FailedSession:
        async def stop(self):
            raise RuntimeError("cleanup failure")

    class HealthySession:
        async def stop(self):
            stopped.append(True)

    manager._sessions = {1: FailedSession(), 2: HealthySession()}

    asyncio.run(manager.shutdown())

    assert stopped == [True]
    assert manager._sessions == {}
