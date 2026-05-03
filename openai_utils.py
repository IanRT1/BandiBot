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
        # Same filtering approach as before: keep only category names that
        # actually appear in the model's output.
        return [cat for cat in CATEGORY_LIST if cat in output_text]

    except (APIError, APIConnectionError, RateLimitError) as e:
        logger.error(f"OpenAI categorization failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during categorization: {e}")
        return []


async def send_to_openai(payload):
    """Send a chat completion request and return a dict-shaped response.

    Accepts the same payload shape handlers.py already builds:
        {"model": "...", "messages": [...], "temperature": 0.5}

    Returns a dict matching the legacy HTTP response shape so handlers.py's
    `data["choices"][0]["message"]["content"]` parsing keeps working:
        {"choices": [{"message": {"content": "..."}}]}

    Returns None on failure.
    """
    try:
        # Default model if handlers.py didn't specify one (it does, but be safe)
        model = payload.get("model", DEFAULT_MODEL)
        messages = payload["messages"]
        temperature = payload.get("temperature", 0.5)

        response = await _client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        # Normalize Pydantic response back to the dict shape handlers.py expects.
        # If you later rewrite process_openai_response to take the SDK object
        # directly, you can drop this normalization.
        return {
            "choices": [
                {
                    "message": {
                        "content": response.choices[0].message.content,
                    }
                }
            ]
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