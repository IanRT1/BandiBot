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

import pytest

from voice import handler


@pytest.mark.parametrize("text", ["Stop!", "stop", "¡Para!", "Detente."])
@pytest.mark.parametrize("queued", [False, True])
def test_bare_stop_controls_music_after_interrupted_acknowledgement(monkeypatch, text, queued):
    from unittest.mock import AsyncMock
    from music.player import voice_manager

    player = SimpleNamespace(current=None if queued else object(),
                             queue=[object()] if queued else [], text_channel=None)
    monkeypatch.setattr(voice_manager, "get_player", lambda guild: player)
    execute = AsyncMock(return_value="Stopped and cleared the queue.")
    model = AsyncMock()
    monkeypatch.setattr(handler, "execute_tool_call", execute)
    monkeypatch.setattr(handler, "send_to_openai", model)
    result = asyncio.run(handler.handle_voice_command(
        text, SimpleNamespace(name="Ian", nick=None), object(), object(), [],
        speech_was_interrupted=True,
    ))
    assert result == ("Stopped and cleared the queue.", False)
    execute.assert_awaited_once()
    assert execute.call_args.args[0] == {"name": "stop_music", "arguments": {}}
    assert execute.call_args.args[1].content == text
    model.assert_not_awaited()


def test_bare_stop_controls_an_in_progress_song_request(monkeypatch):
    from unittest.mock import AsyncMock
    from music.player import voice_manager

    player = SimpleNamespace(
        current=None, queue=[], text_channel=None, has_pending_play_requests=True
    )
    monkeypatch.setattr(voice_manager, "get_player", lambda guild: player)
    execute = AsyncMock(return_value="Stopped and cleared the queue.")
    model = AsyncMock()
    monkeypatch.setattr(handler, "execute_tool_call", execute)
    monkeypatch.setattr(handler, "send_to_openai", model)

    result = asyncio.run(handler.handle_voice_command(
        "Stop!", SimpleNamespace(name="Ian", nick=None), object(), object(), []
    ))

    assert result == ("Stopped and cleared the queue.", False)
    execute.assert_awaited_once()
    model.assert_not_awaited()


def test_song_acknowledgement_is_spoken_when_resolution_finishes_first(monkeypatch):
    from unittest.mock import AsyncMock
    from voice.listener import voice_listener_manager
    import voice.tts as tts

    client = SimpleNamespace(is_connected=lambda: True)
    session = SimpleNamespace(_voice_client=client, clip_buffer=None)
    monkeypatch.setattr(voice_listener_manager, "get_session", lambda guild: session)
    monkeypatch.setattr(handler, "_song_confirmation", AsyncMock(return_value="Reproduciendo Trains de Porcupine Tree."))
    speak = AsyncMock()
    monkeypatch.setattr(tts, "speak", speak)

    async def run():
        task = asyncio.create_task(asyncio.sleep(0))
        await task
        await handler._announce_song_search(object(), "Play Trains", "Trains Porcupine Tree", task)

    asyncio.run(run())
    speak.assert_awaited_once()
    assert speak.call_args.args[1] == "Reproduciendo Trains de Porcupine Tree."


def test_song_search_acknowledgement_requests_minimal_playing_status(monkeypatch):
    async def generate(payload):
        instruction = payload["messages"][0]["content"]
        assert "Respond in English" in instruction
        assert "Do not say you are searching" in instruction
        assert "Do not use Portuguese" in instruction
        assert "Keep it minimal" in instruction
        return {"choices": [{"message": {"content": "Reproduciendo Trains de Porcupine Tree."}}]}

    monkeypatch.setattr(handler, "send_to_openai", generate)
    result = asyncio.run(handler._song_confirmation(
        "Play Trains by Porcupine Tree.",
        "Trains Porcupine Tree",
        searching=True,
    ))
    assert result == "Reproduciendo Trains de Porcupine Tree."


def test_song_search_acknowledgement_selects_spanish_for_spanish_request(monkeypatch):
    async def generate(payload):
        assert "Respond in Spanish" in payload["messages"][0]["content"]
        return {"choices": [{"message": {"content": "Reproduciendo Trains."}}]}

    monkeypatch.setattr(handler, "send_to_openai", generate)
    result = asyncio.run(handler._song_confirmation(
        "Reproduce Trains de Porcupine Tree.",
        "Trains Porcupine Tree",
        searching=True,
    ))
    assert result == "Reproduciendo Trains."


def test_voice_model_receives_stop_context_and_live_search(monkeypatch):
    from music.player import voice_manager

    monkeypatch.setattr(handler, "_build_voice_context", lambda *args: ("context", True))
    monkeypatch.setattr(handler, "build_instruction", lambda **kwargs: "instructions")
    monkeypatch.setattr(voice_manager, "get_player", lambda guild: SimpleNamespace(
        current=SimpleNamespace(paused_at=None), queue=[],
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


@pytest.mark.parametrize("busy, outcome, expected_confirmation", [
    (False, {"status": "starting", "title": "Get Lucky", "artist": "Daft Punk"}, ""),
    (False, {"status": "playing", "title": "The Lady Don't Mind", "artist": "Talking Heads"}, ""),
    (True, {"status": "queued", "title": "The Lady Don't Mind", "artist": "Talking Heads", "queue_position": 4}, "confirmed"),
    (True, {"status": "playing", "title": "The Lady Don't Mind", "artist": "Talking Heads"}, ""),
    (False, {"status": "failed", "error_code": "resolution_failed", "message": "No matching track found."}, "confirmed"),
])
def test_song_acknowledgement_depends_on_actual_outcome(
    monkeypatch, busy, outcome, expected_confirmation,
):
    from music.player import voice_manager

    monkeypatch.setattr(voice_manager, "get_player", lambda guild: SimpleNamespace(
        has_active_track=busy, is_connected=False,
    ))
    calls = []
    outcome = json.dumps(outcome)

    async def run():
        searching = asyncio.Event()

        async def play(*args):
            if not busy:
                await searching.wait()
            return outcome

        async def announce(guild, text, query, task):
            assert not task.done()
            assert query == "The Lady Don't Mind Talking Heads"
            calls.append("searching")
            searching.set()

        async def confirm(text, details, *, searching):
            assert not searching
            assert details == outcome
            calls.append("confirmed")
            return "confirmed"

        monkeypatch.setattr(handler, "_execute_playback_tool", play)
        monkeypatch.setattr(handler, "_announce_song_search", announce)
        monkeypatch.setattr(handler, "_song_confirmation", confirm)
        result = await asyncio.wait_for(handler._execute_song_request(
            {"name": "play_music", "arguments": {"query": "The Lady Don't Mind Talking Heads"}},
            SimpleNamespace(guild=object()), "Pon Lady Don't Mind de Talking Heads",
        ), 1)
        assert result == (outcome, expected_confirmation)

    asyncio.run(run())
    assert ("searching" in calls) is (not busy)
    assert ("confirmed" in calls) is bool(expected_confirmation)
