import asyncio
from collections import deque
from types import SimpleNamespace

from music.tracks import Track


def _track(title):
    return Track(
        title=title,
        stream_url=f"https://example.test/{title}",
        requested_by="Ian",
        webpage_url=f"https://example.test/{title}",
    )


def _message(content):
    return SimpleNamespace(content=content, guild=object(), channel=None)


def test_resolver_prefers_studio_or_remaster_over_unrequested_video():
    from music.resolver import _score_result

    query = "this must be the place talking heads"
    video = {
        "title": "Talking Heads - This Must Be the Place (Official Video)",
        "uploader": "Talking Heads",
        "duration": 250,
    }
    remaster = {
        "title": "This Must Be the Place (Naive Melody) [2003 Remaster]",
        "uploader": "Talking Heads",
        "duration": 250,
    }

    assert _score_result(remaster, query) > _score_result(video, query)


def test_explicit_video_request_does_not_penalize_official_video():
    from music.resolver import _score_result

    video = {
        "title": "Talking Heads - This Must Be the Place (Official Video)",
        "uploader": "Talking Heads",
        "duration": 250,
    }

    assert _score_result(video, "this must be the place talking heads video") > _score_result(
        video, "this must be the place talking heads"
    )


def test_resolver_penalizes_live_unless_requested():
    from music.resolver import _score_result

    live = {
        "title": "This Must Be the Place (Live)",
        "uploader": "Talking Heads",
        "duration": 250,
    }
    studio = {
        "title": "This Must Be the Place (Naive Melody) [2003 Remaster]",
        "uploader": "Talking Heads",
        "duration": 250,
    }

    assert _score_result(studio, "this must be the place talking heads") > _score_result(
        live, "this must be the place talking heads"
    )
    assert _score_result(live, "this must be the place talking heads live") > _score_result(
        live, "this must be the place talking heads"
    )


def test_remove_current_song_routes_to_skip(monkeypatch):
    import bot.tool_executor as executor

    player = SimpleNamespace(current=_track("Current"), queue=deque([_track("Next")] ))
    skipped = []

    async def skip(guild):
        skipped.append(guild)
        return "Skipped: Current"

    monkeypatch.setattr(executor.voice_manager, "get_player", lambda guild: player)
    monkeypatch.setattr(executor.voice_manager, "skip", skip)

    result = asyncio.run(
        executor._handle_delete_track(_message("¿Puedes quitar la canción?"), {})
    )

    assert result == "Skipped: Current"
    assert skipped
    assert [track.title for track in player.queue] == ["Next"]


def test_queue_position_removes_only_upcoming_track(monkeypatch):
    import bot.tool_executor as executor

    current = _track("Current")
    player = SimpleNamespace(
        current=current,
        queue=deque([_track("First"), _track("Second")]),
    )
    monkeypatch.setattr(executor.voice_manager, "get_player", lambda guild: player)

    result = asyncio.run(
        executor._handle_delete_track(
            _message("quita la canción número 2 de la cola"),
            {"positions": [2]},
        )
    )

    assert "Second" in result
    assert player.current is current
    assert [track.title for track in player.queue] == ["First"]


def test_stop_music_is_blocked_for_non_stop_request():
    import bot.tool_executor as executor

    result = asyncio.run(
        executor._handle_stop_music(_message("pon la canción Stop This Train"), {})
    )

    assert result.startswith("Stop cancelled")


def test_unknown_tool_returns_safe_error():
    import bot.tool_executor as executor

    result = asyncio.run(
        executor.execute_tool_call(
            {"name": "not_a_real_tool", "arguments": {}},
            _message("hello"),
        )
    )

    assert result == "Unknown tool: not_a_real_tool"


def test_tool_exception_is_returned_without_escaping(monkeypatch):
    import bot.tool_executor as executor

    async def broken_handler(message, args):
        raise RuntimeError("simulated failure")

    monkeypatch.setitem(executor.TOOL_HANDLERS, "test_failure", broken_handler)

    result = asyncio.run(
        executor.execute_tool_call(
            {"name": "test_failure", "arguments": {}},
            _message("hello"),
        )
    )

    assert result == "Tool error: simulated failure"
