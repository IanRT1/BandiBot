import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.interactions import execute_turn, render_music_result, ExecutionLedger, RequestContext
from bot.tool_executor import is_music_tool


def test_music_receipt_cannot_rewrite_queued_into_playing():
    result = render_music_result({"status": "queued", "title": "Playing Now", "queue_position": 2})
    assert result == "Added at position 2: Playing Now."


def test_control_receipt_preserves_requested_language():
    assert render_music_result({"status": "completed", "action": "move_track"}, language="es") == "Moví la canción."


def test_mixed_tool_turn_preserves_information_and_execution_ids():
    message = SimpleNamespace(id=12, guild=SimpleNamespace(id=1), author=SimpleNamespace(id=2))
    calls = [
        {"id": "a", "name": "play_music", "arguments": {"query": "song"}},
        {"id": "b", "name": "web_search", "arguments": {"question": "weather"}},
    ]
    execute = AsyncMock(side_effect=[json.dumps({"status": "queued", "title": "song", "queue_position": 3}), "weather result"])
    turn = asyncio.run(execute_turn(calls, message, "play song and weather", execute, is_music_tool))
    assert turn.receipts == ["Added at position 3: song."]
    assert turn.has_information
    assert [entry["tool_call_id"] for entry in turn.messages[1:]] == ["a", "b"]
    assert turn.messages[0]["content"] is None


@pytest.mark.parametrize("text", ["Stop!", "¡Para!", "Detente."])
def test_stop_uses_shared_model_intent_and_actual_result(monkeypatch, text):
    from voice import handler
    from music.player import voice_manager
    player = SimpleNamespace(current=SimpleNamespace(paused_at=None, title="Existing song"), queue=[], text_channel=None)
    monkeypatch.setattr(voice_manager, "get_player", lambda guild: player)
    monkeypatch.setattr(handler, "_build_voice_context", lambda *args: ("context", False))
    monkeypatch.setattr(handler, "build_instruction", lambda **kwargs: "instructions")
    model = AsyncMock(return_value={"choices": [{"message": {"tool_calls": [
        {"id": "stop", "name": "stop_music", "arguments": {"intent_evidence": text, "response_language": "en"}},
    ]}}]})
    monkeypatch.setattr(handler, "send_to_openai", model)
    execute = AsyncMock(return_value=json.dumps({"action": "stop", "status": "completed", "message": "Stopped and cleared the queue."}))
    monkeypatch.setattr(handler, "execute_tool_call", execute)
    result = asyncio.run(handler.handle_voice_command(text, SimpleNamespace(name="Ian", nick=None),
        SimpleNamespace(id=1, name="Server"), SimpleNamespace(user=SimpleNamespace(display_name="Bot")), []))
    assert result == ("Stopped and cleared the queue.", False)
    assert model.await_count == 1
    assert execute.await_count == 1


def test_unbound_control_evidence_does_not_mutate_music(monkeypatch):
    from bot import tool_executor
    stop = AsyncMock()
    monkeypatch.setattr(tool_executor.voice_manager, "stop", stop)
    result = asyncio.run(tool_executor.execute_tool_call({"name": "stop_music", "arguments": {
        "intent_evidence": "stop music",
    }}, SimpleNamespace(content="play Stop This Train")))
    assert "cancelled" in result.lower()
    stop.assert_not_awaited()


def test_delivery_retry_does_not_execute_control_twice():
    async def run():
        ledger = ExecutionLedger()
        context = RequestContext("request", 1, 2, "text", "skip")
        command = {"name": "skip_track", "arguments": {}}
        dispatch = AsyncMock(return_value="Skipped.")
        for _ in range(2):
            assert await ledger.execute(context, 0, command, object(), dispatch, durable=True) == "Skipped."
        dispatch.assert_awaited_once()
        changed = {"name": "stop_music", "arguments": {}}
        assert "different action" in await ledger.execute(context, 0, changed, object(), dispatch, durable=True)
        dispatch.assert_awaited_once()
    asyncio.run(run())


def test_voice_mixed_request_keeps_both_receipt_and_information(monkeypatch):
    from voice import handler
    from music.player import voice_manager
    player = SimpleNamespace(current=None, queue=[], text_channel=None)
    monkeypatch.setattr(voice_manager, "get_player", lambda guild: player)
    monkeypatch.setattr(handler, "_build_voice_context", lambda *args: ("context", False))
    monkeypatch.setattr(handler, "build_instruction", lambda **kwargs: "instructions")
    model = AsyncMock(side_effect=[{"choices": [{"message": {"tool_calls": [
        {"id": "music", "name": "play_music", "arguments": {"query": "song"}},
        {"id": "info", "name": "get_member_activity", "arguments": {}},
    ]}}]}, {"choices": [{"message": {"content": "Two members are online."}}]}])
    monkeypatch.setattr(handler, "send_to_openai", model)
    execute = AsyncMock(side_effect=[json.dumps({"status": "queued", "title": "song", "queue_position": 1}), "two online"])
    monkeypatch.setattr(handler, "execute_tool_call", execute)
    result = asyncio.run(handler.handle_voice_command("Play a song and who is online?",
        SimpleNamespace(name="Ian", nick=None), SimpleNamespace(id=1, name="Server"),
        SimpleNamespace(user=SimpleNamespace(display_name="Bot")), []))
    assert result == ("Queued: song, position 1. Two members are online.", False)
    assert execute.await_count == 2


def test_voice_receipt_is_short_but_text_preserves_metadata():
    outcome = {"status": "queued", "title": "Song",
               "artist": "Artist", "queue_position": 2, "uploader": "Uploader"}
    assert render_music_result(outcome, origin="voice") == "Queued: Song by Artist, position 2."
    assert render_music_result(outcome, origin="voice", language="es") == "Agregado: Song, de Artist, posición 2."
    assert "Uploader" not in render_music_result(outcome, origin="voice")
    text = render_music_result(outcome)
    assert outcome["title"] in text and outcome["artist"] in text


def test_voice_receipt_preserves_clarification():
    outcome = {"status": "needs_clarification", "message": "Studio or live version?"}
    assert render_music_result(outcome, origin="voice") == outcome["message"]
