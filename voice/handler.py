"""
voice/handler.py

Bridges the voice pipeline to the LLM and tool execution layer.

Called by voice_listener.py after STT produces a transcript. Builds the
LLM prompt with voice-specific constraints, executes any tool calls, and
returns a plain text response string for TTS — or an empty string if the
command was a music action that requires no spoken confirmation.

Prompt design:
  History is explicitly labeled [PAST] to prevent the LLM from re-executing
  old music commands from earlier in the session. The current command is
  separated by a visible divider so the model always acts on the right input.

Response constraints:
  1-2 sentences max, no markdown, no emojis, Spanish or English only.
  These constraints are enforced via the system prompt since voice output
  is fed directly to TTS with no post-processing.

Music commands:
  When a music tool is called, no TTS response is generated — the music
  action itself is the feedback. An empty string is returned so the voice
  pipeline knows to skip TTS entirely.

_FakeMsgProxy:
  A minimal stand-in for discord.Message that lets _execute_tool_call()
  in handlers.py operate identically whether called from text or voice.
"""

import logging
import json
import time

import discord

from bot.utils import clean_username, get_current_pst_time, get_current_pst_date
from bot.openai_client import send_to_openai, ALL_TOOLS
from bot.handlers import (
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
        from music.player import voice_manager
        player = voice_manager.get_player(guild)
        self.channel = player.text_channel  


async def handle_voice_command(
    text: str,
    member: discord.Member,
    guild: discord.Guild,
    client: discord.Client,
    history: list,
) -> tuple[str, bool]:
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
            "You are in a voice channel. Keep responses brief and to the point "
            "No markdown and no emojies. Spanish or English only. "
            "The conversation history below is PAST CONTEXT ONLY. "
            "IGNORE previous music requests. Only act on the CURRENT COMMAND at the end."
            "If the user asks you to leave, craft a goodbye message before leaving according to the context and history of conversation."
        )},
    ]

    for msg in history[:-1]:
        content = msg["content"]
        if msg["role"] == "user":
            messages.append({"role": "user", "content": f"[PAST] {content}"})
        else:
            messages.append({"role": "assistant", "content": f"[PAST] {content}"})

    messages.append({"role": "system", "content": "━━━ CURRENT COMMAND BELOW — ACT ON THIS ONLY ━━━"})
    messages.append({"role": "user", "content": f"[{user_name}] {text}"})

    response_data = await send_to_openai(
        {"messages": messages, "temperature": 0.5},
        tools=ALL_TOOLS,
    )

    if not response_data:
        return "Lo siento, algo salió mal.", False

    msg = response_data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls")

    should_leave = False

    if tool_calls:
        proxy = _FakeMsgProxy(guild, member)
        called_music = any(_is_music_tool(tc["name"]) for tc in tool_calls)
        should_leave = any(tc["name"] == "leave_voice" for tc in tool_calls)

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
            return "", False

        response_data = await send_to_openai(
            {"messages": messages, "temperature": 0.5},
        )

        if not response_data:
            return "Listo.", should_leave

    raw = response_data["choices"][0]["message"].get("content", "")
    response_text = raw.strip() if raw else "Listo."

    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(f"[llm]  ← {elapsed:.0f}ms | speaking ({len(response_text)} chars): {response_text!r}")

    return response_text, should_leave