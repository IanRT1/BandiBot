"""Exercise playback races with explicit barriers instead of timing sleeps."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

import music.player as music
from music.tracks import Track


@pytest.mark.parametrize("paused", [False, True])
def test_stop_clears_looping_playback_and_ignores_late_audio_callback(monkeypatch, paused):
    async def run():
        import voice.listener as listener
        import voice.tts as tts

        guild = SimpleNamespace(id=1, name="Test")
        manager = music.VoiceManager()
        player = manager.get_player(guild)
        callbacks = []
        voice = SimpleNamespace(
            client=SimpleNamespace(loop=asyncio.get_running_loop()),
            is_connected=lambda: True, is_playing=lambda: False,
            is_paused=lambda: paused, stop_playing=Mock(),
            play=lambda source, after: callbacks.append(after),
        )
        player.voice_client = voice
        player._loop = True
        monkeypatch.setattr(listener.voice_listener_manager, "get_session", lambda guild: None)
        monkeypatch.setattr(music.discord, "FFmpegPCMAudio", lambda *a, **kw: object())
        monkeypatch.setattr(music.discord, "PCMVolumeTransformer", lambda source, volume: source)
        monkeypatch.setattr(tts, "MixerSource", lambda source, clip_buffer: source)
        monkeypatch.setattr(player, "start_resolver", lambda: None)
        player._play_resolved(Track("song", "stream", "user", "video"))
        voice.is_playing = lambda: not paused
        await manager.stop(guild)
        callbacks[0](None)  # Discord's audio thread may finish after stop returns.
        await asyncio.sleep(0)
        voice.stop_playing.assert_called_once()
        assert player.current is None
        assert not player.queue
        assert len(callbacks) == 1

    asyncio.run(run())


@pytest.mark.parametrize("action", ["stop", "disconnect", "recover"])
def test_pending_resolution_cannot_restart_invalidated_playback(monkeypatch, action):
    async def run():
        guild = SimpleNamespace(id=1, name="Test")
        manager = music.VoiceManager()
        player = manager.get_player(guild)
        voice = SimpleNamespace(
            client=SimpleNamespace(loop=asyncio.get_running_loop()),
            is_connected=lambda: True, is_playing=lambda: False,
            is_paused=lambda: False, disconnect=AsyncMock(),
        )
        player.voice_client = voice
        player.queue.append(Track("pending", "", "user", "", resolved=False, query="song"))
        entered, release = asyncio.Event(), asyncio.Event()
        callbacks = []
        tasks = []

        async def resolve(*args):
            entered.set()
            await release.wait()
            return Track("resolved", "stream", "user", "video")

        def submit(coro, loop):
            task = loop.create_task(coro)
            tasks.append(task)
            return task

        monkeypatch.setattr(music, "_resolve_track_async", resolve)
        monkeypatch.setattr(music.asyncio, "run_coroutine_threadsafe", submit)
        monkeypatch.setattr(player, "_play_resolved", callbacks.append)
        player.play_next()
        await asyncio.wait_for(entered.wait(), 1)
        if action == "stop":
            await manager.stop(guild)
        elif action == "disconnect":
            await player.disconnect()
            player.voice_client = voice  # A later reconnect must not revive the old request.
        else:
            await player.prepare_for_voice_recovery()
            player.restore_after_voice_recovery(voice)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        await asyncio.sleep(0)  # Drain playback callbacks queued by resolution.
        assert callbacks == []
        assert player.current is None
        assert not player.queue

    asyncio.run(run())


def test_expired_stream_refreshes_existing_video_without_search(monkeypatch):
    async def run():
        player = music.GuildPlayer(SimpleNamespace(name="Test"))
        player.voice_client = SimpleNamespace(
            client=SimpleNamespace(loop=asyncio.get_running_loop()),
            is_connected=lambda: True, is_playing=lambda: False,
        )
        track = Track("song", "expired", "user", "https://example.test/video",
                      query="original search", resolved_at=0)
        player.queue.append(track)
        resolve = AsyncMock(return_value=Track("song", "fresh", "user", track.webpage_url))
        played = asyncio.Event()
        monkeypatch.setattr(music, "_resolve_track_async", resolve)
        monkeypatch.setattr(player, "_play_resolved", lambda result: played.set())
        player.play_next()
        await asyncio.wait_for(played.wait(), 1)
        resolve.assert_awaited_once_with(track.webpage_url, "user")
        assert track.stream_url == "fresh"
        assert track.error is None

    asyncio.run(run())


def test_stop_cancels_next_track_preload(monkeypatch):
    async def run():
        manager = music.VoiceManager()
        guild = SimpleNamespace(id=1, name="Test")
        player = manager.get_player(guild)
        player.voice_client = SimpleNamespace(
            is_connected=lambda: True, is_playing=lambda: False, is_paused=lambda: False,
        )
        track = Track("pending", "", "user", "", resolved=False, query="song")
        player.queue.append(track)
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def resolve(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(music, "_resolve_track_async", resolve)
        player.start_resolver()
        task = player._resolver_task
        await asyncio.wait_for(entered.wait(), 1)
        await manager.stop(guild)
        await asyncio.wait_for(task, 1)
        assert cancelled.is_set()
        assert not player.queue
        assert not track.resolved
        assert track.error is None

    asyncio.run(run())
