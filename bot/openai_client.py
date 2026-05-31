"""
bot/openai_client.py

Thin async wrapper around the OpenAI SDK for BandiBot's chat completions.

Manages a single shared AsyncOpenAI client and exposes a unified
send_to_openai() function that handles both plain text responses and
tool call responses in a consistent dict-shaped format.

Tool definitions:
  Tool schemas live in bot/tool_schemas.py. This module only sends requests
  and normalizes OpenAI SDK responses into the dict shape expected by handlers.

Tool call response shape:
  {"choices": [{"message": {"content": None, "tool_calls": [...]}}]}

Text response shape:
  {"choices": [{"message": {"content": "..."}}]}

Model is configurable via OPENAI_MODEL env var, defaults to gpt-5.4-nano.
All API errors are caught and logged; None is returned on failure.
"""

import json
import logging
from core.config import OPENAI_API_KEY, OPENAI_MODEL
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)  

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

DEFAULT_MODEL = OPENAI_MODEL


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
