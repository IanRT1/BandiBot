import asyncio
from unittest.mock import Mock, AsyncMock

import pytest

from core import capabilities
from core.preflight import run_preflight
from bot import tool_schemas


@pytest.mark.parametrize("key,expected", [(None, False), ("", False), ("  ", False), ("configured-key", True)])
def test_preflight_controls_every_tool_route(monkeypatch, key, expected):
    monkeypatch.setattr("core.preflight.Path.is_file", lambda _: True)
    environment = {
        "DISCORD_TOKEN": "test", "OPENAI_API_KEY": "test", "DEEPGRAM_API_KEY": "test",
    }
    if key is not None:
        environment["GEMINI_API_KEY"] = key
    result = run_preflight(environ=environment)
    assert result.ok
    capabilities.configure_optional_tools(result.enabled_optional_tools)
    assert capabilities.tool_available("web_search") is expected
    for options in ({}, {"allow_live_search": True}, {"allow_song_requests": True},
                    {"allow_live_search": True, "allow_song_requests": True}):
        tools = tool_schemas.select_tools_for_request("What is the weather today?", **options)
        assert ("web_search" in {t["function"]["name"] for t in tools}) is expected
    for helper in (tool_schemas.tools_without_context_lookups, tool_schemas.tools_without_server_info):
        assert ("web_search" in {t["function"]["name"] for t in helper()}) is expected
    from bot.handlers import build_instruction
    instruction = build_instruction("BandiBot", "Test server")
    assert ("use web_search before answering" in instruction) is expected


def test_disabled_tool_cannot_execute(monkeypatch):
    from bot import tool_executor
    capabilities.configure_optional_tools(frozenset())
    handler = Mock(side_effect=AssertionError("Provider must not be contacted"))
    monkeypatch.setitem(tool_executor.TOOL_HANDLERS, "web_search", handler)
    result = asyncio.run(tool_executor.execute_tool_call({"name": "web_search", "arguments": {"question": "news"}}, None))
    assert result == "Unknown tool: web_search"
    handler.assert_not_called()


def test_model_boundary_filters_static_catalog(monkeypatch):
    from bot import openai_client
    capabilities.configure_optional_tools(frozenset())
    response = Mock()
    response.usage = None
    response.choices = [Mock(message=Mock(content="hello", tool_calls=None))]
    create = AsyncMock(return_value=response)
    monkeypatch.setattr(openai_client._client.chat.completions, "create", create)
    asyncio.run(openai_client.send_to_openai({"messages": []}, tools=tool_schemas.ALL_TOOLS))
    assert "web_search" not in {t["function"]["name"] for t in create.call_args.kwargs["tools"]}
    assert "play_music" in {t["function"]["name"] for t in create.call_args.kwargs["tools"]}
