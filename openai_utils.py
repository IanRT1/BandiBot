"""Thin async wrapper around the OpenAI SDK for chat completions and categorization."""

import json
import logging
import os

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

# Load category definitions once at module import (was already doing this)
with open("config.json", "r", encoding="utf-8") as _f:
    _CONFIG = json.load(_f)

CATEGORY_LIST = _CONFIG["categories"]

# Single shared async client. Reads OPENAI_API_KEY from env automatically
# (loaded by main.py via dotenv). Manages its own connection pool — don't
# create one per request like the old aiohttp code did.
_client = AsyncOpenAI()

# Models — overridable via .env. Categorization gets a cheaper model since
# it's a simple classification task.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
CATEGORIZER_MODEL = os.getenv("OPENAI_CATEGORIZER_MODEL", "gpt-5.4-mini")


# Tool schemas the LLM sees on every chat call. These tell GPT what
# music-related functions exist and when to call them. The LLM picks one
# (or none) based on the user's message.
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


async def categorize_message(message):
    """Classify a message into one or more configured categories.

    Returns a list of category names from CATEGORY_LIST that the model
    judged applicable. Returns an empty list on any failure.
    """
    category_instructions = "\n".join(f"{k}: {v}" for k, v in CATEGORY_LIST.items())

    system_prompt = (
        "Your whole purpose is to categorize prompt intents. Use only the "
        f"following categories for labeling: \n{category_instructions}\n "
        "Identify the best category for the given message and output only "
        "using the category names from the list that apply."
    )

    try:
        response = await _client.chat.completions.create(
            model=CATEGORIZER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Discord Message: "{message}"'},
            ],
        )
        output_text = response.choices[0].message.content.strip()
        return [cat for cat in CATEGORY_LIST if cat in output_text]

    except (APIError, APIConnectionError, RateLimitError) as e:
        logger.error(f"OpenAI categorization failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during categorization: {e}")
        return []


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

        # Normalize the response. If the model called tools, surface them in
        # a structured way so handlers.py can dispatch.
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
                        "content": msg.content,  # may be None when tool-calling
                        "tool_calls": tool_calls,
                    }
                }]
            }

        # Plain text response — same shape as before
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