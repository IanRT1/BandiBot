"""Thin async wrapper around the OpenAI SDK for chat completions."""

import json
import logging
import os

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

# Single shared async client. Reads OPENAI_API_KEY from env automatically
# (loaded by main.py via dotenv). Manages its own connection pool.
_client = AsyncOpenAI()

# Models — overridable via .env
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


# Music tool schemas
MUSIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": (
                "Play a song or audio in the user's current voice channel. "
                "Accepts either a YouTube URL or a free-text search query "
                "(e.g. 'Pink Floyd Time'). If something is already playing, "
                "the new track is added to the queue. Use this whenever the "
                "user asks to play, queue, or put on music."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "URL or search query for the track to play.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_track",
            "description": "Skip the currently playing track and move to the next in queue.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_music",
            "description": "Pause the currently playing track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_music",
            "description": "Resume a paused track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_music",
            "description": "Stop playback and clear the entire queue. Bot stays in the voice channel.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "leave_voice",
            "description": "Disconnect from the voice channel entirely.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "now_playing",
            "description": "Get the title of the currently playing track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_queue",
            "description": "List the upcoming tracks in the queue.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Member activity tool — fetches real-time server presence on demand
GET_MEMBER_ACTIVITY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_member_activity",
            "description": (
                "Get real-time information about who is currently online, "
                "what they are playing, who is in voice channels, and their "
                "roles and permissions. Use this when the user asks about "
                "server members, who is online, who is in voice, or what "
                "people are doing."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

# Combined tool list sent on every chat call
ALL_TOOLS = MUSIC_TOOLS + GET_MEMBER_ACTIVITY_TOOL


async def send_to_openai(payload, tools=None):
    """Send a chat completion request and return a dict-shaped response.

    Accepts the payload shape handlers.py builds:
        {"messages": [...], "temperature": 0.5}

    If `tools` is provided, the LLM may return a tool-call instead of text.
    The returned dict uses one of two shapes:

      Text response:
        {"choices": [{"message": {"content": "..."}}]}

      Tool-call response:
        {"choices": [{"message": {
            "content": None,
            "tool_calls": [
                {"id": "...", "name": "...", "arguments": {...}},
                ...
            ]
        }}]}

    Returns None on failure.
    """
    try:
        model = payload.get("model", DEFAULT_MODEL)
        messages = payload["messages"]
        temperature = payload.get("temperature", 0.5)

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools

        response = await _client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
            return {
                "choices": [{
                    "message": {
                        "content": msg.content,
                        "tool_calls": tool_calls,
                    }
                }]
            }

        return {
            "choices": [{
                "message": {
                    "content": msg.content,
                }
            }]
        }

    except RateLimitError as e:
        logger.error(f"OpenAI rate limit hit: {e}")
        return None
    except APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        return None
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {e}")
        return None