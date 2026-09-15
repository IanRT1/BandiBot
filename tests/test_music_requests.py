import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from music.requests import song_query
from music.results import PlayResult
import json


def test_recording_labels_use_music_metadata_not_channel():
    from music.resolver import _recording_labels
    assert _recording_labels({"track": "Song", "artist": "Artist", "title": "Artist - Song (Official Video)", "uploader": "Reupload channel"}) == ("Song", "Artist")
    assert _recording_labels({"title": "Artist - Song", "artists": ["Artist"], "uploader": "Reupload channel"}) == ("Song", "Artist")
    assert _recording_labels({"title": "Artist - Song", "uploader": "Reupload channel"}) == ("Artist - Song", None)


@pytest.mark.parametrize("original", [
    "Reproduce, por fa, de Justin Quiles.",
    "Play, por fa, Justin Huiles.",
    "Play Please by Chris Isaak",
    "Play anything by Justin Quiles",
])
def test_missing_title_preserves_request_instead_of_artist_only_search(original):
    song = {"title": "", "artist": "Justin Quiles", "version": "", "source": ""}
    assert song_query(song, original_text=original) == original


def test_complete_identity_and_exact_sources_keep_existing_fast_path():
    song = {"title": "Stop This Train", "artist": "John Mayer", "version": "", "source": ""}
    assert song_query(song, original_text="Could you play Stop This Train by John Mayer please?") == "Stop This Train John Mayer"
    direct = {"title": "", "artist": "", "version": "", "source": "kUGVuqakYtE"}
    assert song_query(direct, original_text="Play this link") == "kUGVuqakYtE"


def test_recovery_preserves_original_title_language_and_rejects_invention():
    from music.requests import recover_song_query
    async def run():
        for title, expected in (("por fa", "por fa Justin Quiles"), ("Please", None), ("Other song", None)):
            generate = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps({
                "title_quote": title, "artist": "Justin Quiles", "version": "",
            })}}]})
            assert await recover_song_query("Reproduce, por fa, de Justin Kwiles.", generate) == expected
            generate.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("song,expected", [
    ({"title": "Porfa", "artist": "Justin Quiles", "version": "", "source": ""}, "Porfa Justin Quiles"),
    ({"title": "Stop This Train", "artist": "John Mayer", "version": "live", "source": ""}, "Stop This Train John Mayer live"),
    ({"title": "Add It Up", "artist": "Violent Femmes", "version": "", "source": ""}, "Add It Up Violent Femmes"),
    ({"title": "", "artist": "", "version": "", "source": "kUGVuqakYtE"}, "kUGVuqakYtE"),
])
def test_song_identity_preserves_requested_terms(song, expected):
    assert song_query(song) == expected


@pytest.mark.parametrize("song", [
    {}, {"title": "song"},
    {"title": "", "artist": "", "version": "live", "source": ""},
    {"title": "song", "artist": "", "version": "", "source": "video-id"},
    {"title": None, "artist": "", "version": "", "source": ""},
    {"title": "x" * 501, "artist": "", "version": "", "source": ""},
])
def test_invalid_identity_is_rejected_before_resolution(song):
    with pytest.raises(ValueError):
        song_query(song)


def test_structured_tool_reaches_existing_controller(monkeypatch):
    from bot import tool_executor
    player = SimpleNamespace(text_channel=object())
    monkeypatch.setattr(tool_executor.voice_manager, "get_player", lambda _: player)
    play = AsyncMock(return_value=PlayResult("queued", title="Porfa", queue_position=1))
    monkeypatch.setattr(tool_executor.voice_manager, "play", play)
    message = SimpleNamespace(guild=object(), author=object(), channel=None)
    song = {"title": "Porfa", "artist": "Justin Quiles", "version": "", "source": ""}
    result = asyncio.run(tool_executor.execute_tool_call({"name": "play_music", "arguments": {"song": song}}, message))
    play.assert_awaited_once_with(message.guild, message.author, "Porfa Justin Quiles")
    assert '"queued"' in result
