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


def test_resolver_prefers_artist_official_channel_over_unrelated_reupload():
    from music.resolver import _score_result

    query = "what i mean modjo"
    official_lyric = {
        "title": "Modjo - What I Mean (Official Lyric Video)",
        "uploader": "ModjoOfficial",
        "duration": 235,
    }
    reupload = {
        "title": "What I Mean - Modjo",
        "uploader": "Tokzen Records",
        "duration": 235,
    }

    assert _score_result(official_lyric, query) > _score_result(reupload, query)


def test_resolver_prefers_official_lyric_over_official_video_for_audio():
    from music.resolver import _score_result

    query = "what i mean modjo"
    official_lyric = {
        "title": "Modjo - What I Mean (Official Lyric Video)",
        "uploader": "ModjoOfficial",
        "duration": 235,
    }
    official_video = {
        "title": "Modjo - What I Mean (Official Music Video)",
        "uploader": "ModjoOfficial",
        "duration": 235,
    }

    assert _score_result(official_lyric, query) > _score_result(official_video, query)


def test_resolver_rejects_playable_but_irrelevant_results():
    from music.resolver import _filter_irrelevant_entries

    candidates = [
        (10, {"title": "Stay up late- Talking heads Lyrics", "uploader": "dlamitp"}),
        (20, {"title": "Talking Heads - Walk It Down (Official Audio)", "uploader": "Talking Heads"}),
    ]

    filtered = _filter_irrelevant_entries(
        candidates,
        "plate walk it down talking heads official audio",
    )

    assert [entry["title"] for _, entry in filtered] == [
        "Talking Heads - Walk It Down (Official Audio)"
    ]


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


def test_malformed_tool_arguments_are_rejected_before_dispatch(monkeypatch):
    import bot.tool_executor as executor

    called = []

    async def handler(message, args):
        called.append(args)
        return "should not run"

    monkeypatch.setitem(executor.TOOL_HANDLERS, "play_music", handler)
    result = asyncio.run(
        executor.execute_tool_call(
            {"name": "play_music", "arguments": {"query": 123}},
            _message("play something"),
        )
    )

    assert result.startswith("Invalid arguments for play_music:")
    assert not called


def test_unknown_tool_arguments_are_rejected():
    import bot.tool_executor as executor

    result = asyncio.run(
        executor.execute_tool_call(
            {"name": "pause_music", "arguments": {"unexpected": True}},
            _message("pause"),
        )
    )

    assert result == "Invalid arguments for pause_music: Unexpected argument(s): unexpected."


def test_leave_voice_stops_listener_and_disconnects_player(monkeypatch):
    import bot.tool_executor as executor
    import voice.listener as listener

    events = []

    class Player:
        async def disconnect(self):
            events.append("player-disconnected")

    async def stop_listening(guild):
        events.append("listener-stopped")

    monkeypatch.setattr(executor.voice_manager, "get_player", lambda guild: Player())
    monkeypatch.setattr(listener.voice_listener_manager, "stop_listening", stop_listening)

    result = asyncio.run(executor._handle_leave_voice(_message("leave"), {}))

    assert result == "Leaving the voice channel."
    assert events == ["listener-stopped", "player-disconnected"]


def test_music_control_confirmation_uses_actual_tool_result(monkeypatch):
    import voice.handler as handler

    async def send_to_openai(payload):
        assert "Removed one song" in payload["messages"][1]["content"]
        return {"choices": [{"message": {"content": "Removed the queued song."}}]}

    monkeypatch.setattr(handler, "send_to_openai", send_to_openai)

    result = asyncio.run(
        handler._music_action_confirmation(
            "delete position one on queue",
            "delete_track",
            "Removed one song: 'Example'",
        )
    )

    assert result == "Removed the queued song."


def test_undo_confirmation_is_casual_and_does_not_include_song_details(monkeypatch):
    import voice.handler as handler

    async def send_to_openai(payload):
        prompt = payload["messages"][0]["content"]
        assert "casually and briefly" in prompt
        assert "Do not mention the song, artist" in prompt
        return {"choices": [{"message": {"content": "Got it, I deleted it."}}]}

    monkeypatch.setattr(handler, "send_to_openai", send_to_openai)

    result = asyncio.run(
        handler._music_action_confirmation(
            "wrong song",
            "undo_last_song_request",
            "Deleted the most recently requested song.",
        )
    )

    assert result == "Got it, I deleted it."


def test_skip_confirmation_is_brief_and_does_not_include_song_details(monkeypatch):
    import voice.handler as handler

    async def send_to_openai(payload):
        prompt = payload["messages"][0]["content"]
        assert "Okay, skipped it" in prompt
        assert "Do not mention the song, artist" in prompt
        return {"choices": [{"message": {"content": "Okay, skipped it."}}]}

    monkeypatch.setattr(handler, "send_to_openai", send_to_openai)

    result = asyncio.run(
        handler._music_action_confirmation(
            "skip it",
            "skip_track",
            "Skipped: A song with a long title.",
        )
    )

    assert result == "Okay, skipped it."


def test_undo_last_song_request_removes_newest_queued_song(monkeypatch):
    import bot.tool_executor as executor

    newest = _track("Newest")
    player = SimpleNamespace(current=_track("Playing"), queue=deque([_track("Older"), newest]))
    monkeypatch.setattr(executor.voice_manager, "get_player", lambda guild: player)

    result = asyncio.run(
        executor._handle_undo_last_song_request(_message("wrong song"), {})
    )

    assert result == "Deleted the most recently requested song."
    assert [track.title for track in player.queue] == ["Older"]
    assert player.current.title == "Playing"


def test_undo_last_song_request_removes_only_current_song_when_queue_is_empty(monkeypatch):
    import bot.tool_executor as executor

    player = SimpleNamespace(
        current=_track("Playing"), queue=deque(), voice_client=None, is_playing=False,
    )
    monkeypatch.setattr(executor.voice_manager, "get_player", lambda guild: player)

    result = asyncio.run(
        executor._handle_undo_last_song_request(_message("canción equivocada"), {})
    )

    assert result == "Deleted the most recently requested song."
    assert player.current is None
