"""
tests/test_search_acknowledgement.py

Offline checks for tailored voice search acknowledgements.

Coverage:
  Compact generation requests, parallel search and speech, graceful speech
  failure, and cancellation of both tasks when the interaction is interrupted.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from voice import handler


def test_pending_song_never_announces_playing_or_calls_model(monkeypatch):
    model = AsyncMock()
    monkeypatch.setattr(handler, "send_to_openai", model)
    assert asyncio.run(handler._song_confirmation("Play a song", "song", searching=True)) == ""
    model.assert_not_awaited()


def test_voice_model_receives_stop_context_and_live_search(monkeypatch):
    from music.player import voice_manager

    monkeypatch.setattr(handler, "_build_voice_context", lambda *args: ("context", True))
    monkeypatch.setattr(handler, "build_instruction", lambda **kwargs: "instructions")
    monkeypatch.setattr(voice_manager, "get_player", lambda guild: SimpleNamespace(
        current=SimpleNamespace(paused_at=None, title="Existing song"), queue=[],
    ))

    async def generate(payload, tools=None):
        prompt = "\n".join(message["content"] for message in payload["messages"])
        assert "speech was just interrupted by the wake word: True" in prompt
        assert "music track loaded: True" in prompt
        assert "speech is already stopped" in prompt
        assert "web_search" in {tool["function"]["name"] for tool in tools}
        assert "play_music" in {tool["function"]["name"] for tool in tools}
        assert "never substitute a text promise" in prompt
        return {"choices": [{"message": {"content": "Stopped."}}]}

    monkeypatch.setattr(handler, "send_to_openai", generate)
    assert asyncio.run(handler.handle_voice_command(
        "Stop speaking!", SimpleNamespace(name="Ian", nick=None), SimpleNamespace(name="server"),
        SimpleNamespace(user=SimpleNamespace(display_name="BandiBot")), [],
        speech_was_interrupted=True,
    )) == ("Stopped.", False)


@pytest.mark.parametrize("question", ["¿Cómo van los Chargers?", "Who is playing tonight?"])
def test_live_search_remains_available_despite_lore_or_local_routing(question):
    from bot.tool_schemas import select_tools_for_request

    tools = select_tools_for_request(
        question, lore_is_confident=True, has_lore_context=True, allow_live_search=True,
    )
    assert "web_search" in {tool["function"]["name"] for tool in tools}


@pytest.mark.parametrize("question", [
    "Search, it's my life bon jovi.", "Encuentra una canción para escuchar",
    "It's My Life", "Who wrote It's My Life?",
])
def test_voice_routing_keeps_song_and_web_tools_available(question):
    from bot.tool_schemas import select_tools_for_request

    tools = select_tools_for_request(
        question, lore_is_confident=True, has_lore_context=True,
        allow_live_search=True, allow_song_requests=True,
    )
    names = [tool["function"]["name"] for tool in tools]
    assert "play_music" in names
    assert "web_search" in names
    assert len(names) == len(set(names))


def test_acknowledgement_uses_compact_generation_request(monkeypatch):
    async def generate(payload):
        assert len(payload["messages"]) == 2
        assert payload["messages"][-1]["content"] == "¿Cómo está el clima en San Diego?"
        assert payload["max_completion_tokens"] == 80
        assert payload["reasoning_effort"] == "none"
        return {"choices": [{"message": {"content": "Un momento, reviso el clima en San Diego."}}]}

    monkeypatch.setattr(handler, "send_to_openai", generate)
    result = asyncio.run(handler._search_acknowledgement("¿Cómo está el clima en San Diego?"))
    assert result == "Un momento, reviso el clima en San Diego."


def test_search_and_acknowledgement_overlap_and_speech_failure_is_optional(monkeypatch):
    async def run():
        search_started = asyncio.Event()
        speech_started = asyncio.Event()

        async def search(*args):
            search_started.set()
            await speech_started.wait()
            return "search answer"

        async def acknowledge(*args):
            speech_started.set()
            await search_started.wait()
            raise RuntimeError("speech unavailable")

        monkeypatch.setattr(handler, "execute_tool_call", search)
        monkeypatch.setattr(handler, "_speak_search_acknowledgement", acknowledge)
        result = await asyncio.wait_for(
            handler._execute_web_search_tool({}, SimpleNamespace(guild=object()), "news"),
            timeout=1,
        )
        assert result == "search answer"

    asyncio.run(run())


def test_interruption_cancels_search_and_acknowledgement(monkeypatch):
    async def run():
        started = [asyncio.Event(), asyncio.Event()]
        stopped = []

        async def work(index):
            started[index].set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(index)

        monkeypatch.setattr(handler, "execute_tool_call", lambda *args: work(0))
        monkeypatch.setattr(handler, "_speak_search_acknowledgement", lambda *args: work(1))
        task = asyncio.create_task(handler._execute_web_search_tool(
            {}, SimpleNamespace(guild=object()), "news",
        ))
        await asyncio.gather(*(event.wait() for event in started))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(stopped) == [0, 1]

    asyncio.run(run())


@pytest.mark.parametrize("outcome, expected", [
    ({"status": "starting", "title": "Track"}, "Starting: Track."),
    ({"status": "playing", "title": "Track"}, "Playing: Track."),
    ({"status": "queued", "title": "Track", "queue_position": 4}, "Queued: Track, position 4."),
    ({"status": "failed", "message": "No match."}, "No match."),
])
def test_song_confirmation_depends_only_on_executed_outcome(monkeypatch, outcome, expected):
    execute = AsyncMock(return_value=json.dumps(outcome))
    model = AsyncMock()
    monkeypatch.setattr(handler, "_execute_playback_tool", execute)
    monkeypatch.setattr(handler, "send_to_openai", model)
    result = asyncio.run(handler._execute_song_request(
        {"name": "play_music", "arguments": {"query": "anything"}}, object(), "play anything"))
    assert result == (json.dumps(outcome), expected)
    model.assert_not_awaited()


def test_song_gets_final_confirmation_when_current_track_ends_during_search(monkeypatch):
    from music.player import voice_manager

    player = SimpleNamespace(
        has_active_track=True,
        is_connected=True,
        voice_client=SimpleNamespace(is_paused=lambda: False),
    )
    monkeypatch.setattr(voice_manager, "get_player", lambda guild: player)
    search_started = asyncio.Event()
    search_finished = asyncio.Event()
    outcome = json.dumps({
        "status": "playing",
        "title": "Homemade Dynamite",
        "artist": "Lorde",
    })

    async def play(*args):
        search_started.set()
        await search_finished.wait()
        return outcome

    confirm = AsyncMock(return_value="Playing Homemade Dynamite by Lorde.")
    announce = AsyncMock()
    monkeypatch.setattr(handler, "_execute_playback_tool", play)
    monkeypatch.setattr(handler, "_song_confirmation", confirm)

    async def run():
        request = asyncio.create_task(handler._execute_song_request(
            {"name": "play_music", "arguments": {"query": "Homemade Dynamite Lorde"}},
            SimpleNamespace(guild=object()),
            "Play Homemade Dynamite by Lorde",
        ))
        await asyncio.wait_for(search_started.wait(), 1)
        player.has_active_track = False
        search_finished.set()
        return await asyncio.wait_for(request, 1)

    assert asyncio.run(run()) == (
        outcome, "Playing Homemade Dynamite by Lorde."
    )
    confirm.assert_awaited_once_with(
        "Play Homemade Dynamite by Lorde", outcome, searching=False
    )
