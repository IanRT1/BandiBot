"""
voice_handler.py

Bridges the voice pipeline to the existing LLM + tool call logic.
Uses the same instructions.txt as the text handler for consistency.
"""

import logging
import json
import time

import discord

from utils import clean_username, get_current_pst_time, get_current_pst_date
from openai_utils import send_to_openai, ALL_TOOLS
from handlers import (
    build_instruction,
    _execute_tool_call,
    _is_music_tool,
)

logger = logging.getLogger(__name__)


def _build_voice_context(member: discord.Member, guild: discord.Guild) -> str:
    user_nick_or_name = clean_username(
        getattr(member, 'nick', None),
        member.name
    )
    return (
        f"**Server Information**:\n"
        f"- Server Name: {guild.name}\n"
        f"- Bot Name: BandiBot\n"
        f"- Current Date: {get_current_pst_date()}\n"
        f"- Server Time: {get_current_pst_time()}\n"
        f"- Current User: {user_nick_or_name}\n"
        f"- Interaction Mode: Voice"
    )


class _FakeMsgProxy:
    def __init__(self, guild: discord.Guild, member: discord.Member):
        self.guild = guild
        self.author = member
        self.channel = None


async def handle_voice_command(
    text: str,
    member: discord.Member,
    guild: discord.Guild,
    client: discord.Client,
    history: list,
) -> str:
    user_name = clean_username(getattr(member, 'nick', None), member.name)
    t_start = time.perf_counter()

    instruction = build_instruction(
        bot_display_name=client.user.display_name,
        server_name=guild.name,
    )

    context_info = _build_voice_context(member, guild)

    messages = [
        {"role": "system", "content": instruction},
        {"role": "system", "content": context_info},
        {"role": "system", "content": (
            "You are in a voice channel. Keep responses to 1-2 sentences max. "
            "No markdown. Spanish or English only. "
            "The conversation history below is PAST CONTEXT ONLY. "
            "IGNORE previous music requests. Only act on the CURRENT COMMAND at the end."
        )},
    ]

    # Add history with explicit past labels
    for msg in history[:-1]:
        content = msg["content"]
        if msg["role"] == "user":
            messages.append({"role": "user", "content": f"[PAST] {content}"})
        else:
            messages.append({"role": "assistant", "content": f"[PAST] {content}"})

    # Current command — explicitly separated
    messages.append({"role": "system", "content": "━━━ CURRENT COMMAND BELOW — ACT ON THIS ONLY ━━━"})
    messages.append({"role": "user", "content": f"[{user_name}] {text}"})

    response_data = await send_to_openai(
        {"messages": messages, "temperature": 0.5},
        tools=ALL_TOOLS,
    )

    if not response_data:
        return "Lo siento, algo salió mal."

    msg = response_data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls")

    if tool_calls:
        proxy = _FakeMsgProxy(guild, member)
        called_music = any(_is_music_tool(tc["name"]) for tc in tool_calls)

        messages.append({
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            result = await _execute_tool_call(tc, proxy)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        if called_music:
            elapsed = (time.perf_counter() - t_start) * 1000
            tool_names = [tc["name"] for tc in tool_calls]
            logger.info(f"[llm]  ← {', '.join(tool_names)} ({elapsed:.0f}ms) | no TTS")
            return ""

        response_data = await send_to_openai(
            {"messages": messages, "temperature": 0.5},
        )

        if not response_data:
            return "Listo."

    raw = response_data["choices"][0]["message"].get("content", "")
    response_text = raw.strip() if raw else "Listo."

    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(f"[llm]  ← {elapsed:.0f}ms | {response_text[:60]!r}")

    return response_text