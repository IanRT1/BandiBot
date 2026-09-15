from bot.handlers import _direct_youtube_tool_response


def test_bare_youtube_video_id_routes_directly_to_play_music():
    response = _direct_youtube_tool_response("kUGVuqakYtE")

    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["name"] == "play_music"
    assert tool_call["arguments"] == {"query": "kUGVuqakYtE"}


def test_play_command_with_markdown_youtube_url_routes_directly():
    url = "https://www.youtube.com/watch?v=jRHWHkADjMo"
    response = _direct_youtube_tool_response(f"play this [{url}]({url})")

    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["name"] == "play_music"
    assert tool_call["arguments"] == {"query": url}


def test_non_video_id_still_uses_normal_chat_routing():
    assert _direct_youtube_tool_response("hello there") is None
    assert _direct_youtube_tool_response("kUGVuqakYtE extra") is None
    assert _direct_youtube_tool_response("https://youtu.be/kUGVuqakYtE") is None
    assert _direct_youtube_tool_response("what is this https://youtu.be/kUGVuqakYtE") is None
