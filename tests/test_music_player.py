from collections import deque
from types import SimpleNamespace
import asyncio
from unittest.mock import AsyncMock
import pytest

from music.player import GuildPlayer
from music.tracks import Track


def _track(title):
    return Track(
        title=title,
        stream_url="https://example.test/stream",
        requested_by="Ian",
        webpage_url="https://example.test/video",
    )


@pytest.mark.parametrize("busy", [False, True])
def test_single_song_returns_structured_track_and_position(monkeypatch, busy):
    import music.player as music_player

    manager = music_player.VoiceManager()
    track = _track("Actual resolved title")
    track.artist = "Actual artist"
    player = SimpleNamespace(
        queue=deque([_track("existing")] if busy else []), current=None,
        has_active_track=busy, is_connected=True,
        voice_client=SimpleNamespace(is_paused=lambda: False), connect=AsyncMock(),
    )
    def play_next():
        player.current = player.queue.popleft()
    player.play_next = play_next
    monkeypatch.setattr(manager, "get_player", lambda guild: player)
    monkeypatch.setattr(music_player, "_resolve_track_async", AsyncMock(return_value=track))
    member = SimpleNamespace(voice=SimpleNamespace(channel=object()), display_name="Ian")
    result = asyncio.run(manager.play(SimpleNamespace(id=1), member, "requested song"))
    assert result.status == ("queued" if busy else "playing")
    assert result.title == "Actual resolved title"
    assert result.artist == "Actual artist"
    assert result.queue_position == (2 if busy else None)


def test_single_song_resolution_failure_is_structured(monkeypatch):
    import music.player as music_player

    manager = music_player.VoiceManager()
    monkeypatch.setattr(manager, "get_player", lambda guild: object())
    monkeypatch.setattr(music_player, "_resolve_track_async", AsyncMock(side_effect=RuntimeError("no results")))
    member = SimpleNamespace(voice=SimpleNamespace(channel=object()), display_name="Ian")
    result = asyncio.run(manager.play(SimpleNamespace(id=1), member, "missing"))
    assert result.status == "failed"
    assert result.error_code == "resolution_failed"
    assert result.queue_position is None


def test_first_song_waiting_for_speech_is_starting_not_queued(monkeypatch):
    import music.player as music_player

    manager = music_player.VoiceManager()
    track = _track("Get Lucky")
    player = SimpleNamespace(
        queue=deque(), current=None, has_active_track=False, is_connected=True,
        voice_client=SimpleNamespace(is_paused=lambda: False), connect=AsyncMock(),
        play_next=lambda: None,  # Standalone acknowledgement defers playback.
    )
    monkeypatch.setattr(manager, "get_player", lambda guild: player)
    monkeypatch.setattr(music_player, "_resolve_track_async", AsyncMock(return_value=track))
    member = SimpleNamespace(voice=SimpleNamespace(channel=object()), display_name="Ian")
    result = asyncio.run(manager.play(SimpleNamespace(id=1), member, "Get Lucky"))
    assert result.status == "starting"
    assert result.queue_position is None
    assert list(player.queue) == [track]


def test_single_song_requires_voice_membership():
    from music.player import VoiceManager

    result = asyncio.run(VoiceManager().play(SimpleNamespace(id=1), SimpleNamespace(voice=None), "song"))
    assert result.to_dict() == {
        "status": "failed", "error_code": "not_in_voice",
        "message": "User is not in a voice channel; cannot play music.",
    }


def test_play_tool_serializes_metadata_without_interpreting_title(monkeypatch):
    import json
    from bot import tool_executor
    from music.results import PlayResult

    outcome = PlayResult("queued", title="Now playing: a song", artist="Artist", queue_position=4)
    monkeypatch.setattr(tool_executor.voice_manager, "get_player", lambda guild: SimpleNamespace(text_channel=None))
    monkeypatch.setattr(tool_executor.voice_manager, "play", AsyncMock(return_value=outcome))
    message = SimpleNamespace(guild=object(), author=object(), channel=None)
    result = asyncio.run(tool_executor.execute_tool_call(
        {"name": "play_music", "arguments": {"query": "song"}}, message,
    ))
    assert json.loads(result) == outcome.to_dict()


def test_resolved_track_is_deferred_while_standalone_audio_is_playing(monkeypatch):
    player = GuildPlayer(SimpleNamespace(name="Test Guild"))
    player.voice_client = SimpleNamespace(
        is_connected=lambda: True,
        is_playing=lambda: True,
    )
    player.queue = deque()
    scheduled = []
    monkeypatch.setattr(player, "_schedule_start_when_free", lambda: scheduled.append(True))

    track = _track("Resolved song")
    player._play_resolved(track)

    assert player.current is None
    assert list(player.queue) == [track]
    assert scheduled == [True]


def test_failed_discord_play_restores_track_and_clears_current(monkeypatch):
    import music.player as music_player

    class FakeVoiceClient:
        client = SimpleNamespace(loop=None)

        def is_connected(self):
            return True

        def is_playing(self):
            return True

        def play(self, source, after):
            raise RuntimeError("Already playing audio.")

    class FakeMixerSource:
        def __init__(self, source, clip_buffer=None):
            pass

    player = GuildPlayer(SimpleNamespace(name="Test Guild"))
    player.voice_client = FakeVoiceClient()
    player.queue = deque()
    scheduled = []
    monkeypatch.setattr(player, "_schedule_start_when_free", lambda: scheduled.append(True))
    monkeypatch.setattr(music_player, "_ffmpeg_before_options", lambda track: "")
    monkeypatch.setattr(music_player.discord, "FFmpegPCMAudio", lambda *args, **kwargs: object())
    monkeypatch.setattr(music_player.discord, "PCMVolumeTransformer", lambda source, volume: source)

    import voice.tts as tts
    monkeypatch.setattr(tts, "MixerSource", FakeMixerSource)
    track = _track("Failed start")
    player._play_resolved(track)

    assert player.current is None
    assert list(player.queue) == [track]
    assert scheduled == [True]


def test_voice_recovery_preserves_current_track_before_rebinding():
    player = GuildPlayer(SimpleNamespace(name="Test Guild"))
    current = _track("Current track")
    upcoming = _track("Upcoming track")
    player.current = current
    player.queue = deque([upcoming])
    player.voice_client = SimpleNamespace()
    played = []
    player.play_next = lambda: played.append(True)

    async def run():
        await player.prepare_for_voice_recovery()
        replacement = SimpleNamespace()
        player.restore_after_voice_recovery(replacement)

    asyncio.run(run())

    assert player.voice_client is not None
    assert list(player.queue) == [current, upcoming]
    assert player.current is None
    assert played == [True]
