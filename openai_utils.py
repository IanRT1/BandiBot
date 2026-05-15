"""
openai_utils.py

Thin async wrapper around the OpenAI SDK for BandiBot's chat completions.

Manages a single shared AsyncOpenAI client and exposes a unified
send_to_openai() function that handles both plain text responses and
tool call responses in a consistent dict-shaped format.

Tool definitions:
  MUSIC_TOOLS          → playback control (play, queue_bulk, skip, pause, etc.)
  GET_MEMBER_ACTIVITY_TOOL → real-time server presence and voice channel info
  GET_SERVER_INFO_TOOL → static server history and lore from server_info.txt
  ALL_TOOLS            → combined list sent on every chat completion request

Tool call response shape:
  {"choices": [{"message": {"content": None, "tool_calls": [...]}}]}

Text response shape:
  {"choices": [{"message": {"content": "..."}}]}

Model is configurable via OPENAI_MODEL env var, defaults to gpt-5.4-nano.
All API errors are caught and logged; None is returned on failure.
"""

import json
import logging
import os

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

_client = AsyncOpenAI()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")


MUSIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": (
                "Play a single song or audio in the user's current voice channel. "
                "Pass the user's exact words as the query — do NOT interpret, translate, or try to guess the correct song title. "
                "Accepts either a YouTube URL or a free-text search query. "
                "If something is already playing, the new track is added to the queue. "
                "Use this whenever the user asks to play or queue a SINGLE song."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's exact words as the search query. Do not interpret or modify.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queue_bulk",
            "description": (
                "Queue multiple songs at once. Use this when the user asks to add several songs, "
                "provides a list of songs, or pastes a YouTube playlist URL. "
                "For text lists, each item becomes a separate search query. "
                "For a YouTube playlist URL, pass it as the single item in the list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of search queries or URLs. Each item is a separate song. "
                            "For a YouTube playlist, pass the playlist URL as the only item."
                        ),
                    },
                    "is_playlist": {
                        "type": "boolean",
                        "description": "True if the input is a YouTube playlist URL, False for text search queries.",
                    },
                },
                "required": ["queries", "is_playlist"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_track",
            "description": "Move a song in the queue to a different position. Can reference the track by its current position number or by name/partial name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_position": {
                        "type": "integer",
                        "description": "Current position of the track in the queue (1-based). Optional if track_name is provided.",
                    },
                    "to_position": {
                        "type": "integer",
                        "description": "Target position in the queue (1-based).",
                    },
                    "track_name": {
                        "type": "string",
                        "description": "Name or partial name of the track to move. Used if from_position is not provided.",
                    },
                },
                "required": ["to_position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_track",
            "description": (
                "Remove a specific song from the queue without affecting playback. "
                "Can reference the track by position number, by name/partial name, "
                "or by recency (e.g. 'the last song you added', 'the one I just queued'). "
                "Does NOT affect the currently playing track."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "integer",
                        "description": "Queue position of the track to remove (1-based). Optional if track_name is provided.",
                    },
                    "track_name": {
                        "type": "string",
                        "description": "Name or partial name of the track to remove. Also handles relative references like 'last song'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "join_voice",
            "description": (
                "Join the voice channel that the user is currently in. "
                "Use this when the user asks the bot to join, come, or enter their voice channel. "
                "Works even if no music is requested."
            ),
            "parameters": {"type": "object", "properties": {}},
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

GET_SERVER_INFO_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_server_info",
            "description": (
                "Get information about this Discord server from its history and lore document. "
                "Use this when the user asks something about the server — its history, rules, "
                "events, or any other server-specific knowledge. "
                "Pass the user's question so the response can be tailored to what was asked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question about the server, used to contextualize the response.",
                    }
                },
                "required": ["question"],
            },
        },
    }
]

ALL_TOOLS = MUSIC_TOOLS + GET_MEMBER_ACTIVITY_TOOL + GET_SERVER_INFO_TOOL


async def send_to_openai(payload, tools=None):
    """Send a chat completion request and return a dict-shaped response."""
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

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

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
                }],
                "usage": usage,
            }

        return {
            "choices": [{
                "message": {
                    "content": msg.content,
                }
            }],
            "usage": usage,
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