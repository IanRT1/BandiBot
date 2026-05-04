import asyncio
import json
import logging
import os
import re
import time
from textwrap import dedent
from zoneinfo import ZoneInfo

import discord

from utils import (
    clean_username,
    get_current_pst_time,
    get_current_pst_date,
    get_server_info,
)
from openai_utils import categorize_message, send_to_openai, MUSIC_TOOLS
from music import voice_manager

logger = logging.getLogger(__name__)

# Load config once at import time, not on every message
with open("config.json", "r", encoding="utf-8") as _f:
    _CONFIG = json.load(_f)

# Load static server lore (optional). Sent to the LLM on every message,
# so keep the file lean — extra tokens cost money. Reloaded only on restart.
_SERVER_LORE = ""
if os.path.exists("server_info.md"):
    with open("server_info.md", "r", encoding="utf-8") as _f:
        _SERVER_LORE = _f.read().strip()
    logger.info(f"Loaded server_info.md ({len(_SERVER_LORE)} chars)")

_PACIFIC = ZoneInfo("America/Los_Angeles")


def _strip_bot_mentions(message, client):
    """
    Remove bot mentions from message content cleanly.

    Replaces the original `content.split()[0]` + `replace(...)` approach,
    which broke when the mention wasn't the first token or appeared
    multiple times in the message.
    """
    content = message.content
    for mention_str in (client.user.mention, f"<@!{client.user.id}>"):
        content = content.replace(mention_str, "")
    return content.strip()


def build_context_info(message, categories, client, server_info):
    """Build context block for the system prompt. server_info is passed in
    so we don't recompute it for the member-activity branch."""
    user_nick_or_name = clean_username(message.author.nick, message.author.name)
    server_name = message.guild.name
    channel_name = message.channel.name
    bot_display_name = client.user.display_name
    user_message = _strip_bot_mentions(message, client)
    creation_date = message.guild.created_at.strftime("%Y-%m-%d")
    server_owner = clean_username(message.guild.owner.nick, message.guild.owner.name)
    total_members = message.guild.member_count
    current_pst_time = get_current_pst_time()
    current_pst_date = get_current_pst_date()
    online_members_count = server_info["online_count"]
    online_users_list = ", ".join(info[0] for info in server_info["online_members"])

    context = dedent(
        f"""
        **Server Information**:
        - Server Name: {server_name}
        - Active Channel: {channel_name}
        - Bot Name: {bot_display_name}
        - Server Creation Date: {creation_date}
        - Server Owner: {server_owner}
        - Total Members: {total_members}
        - Online Members: {online_members_count}
        - Online Users: {online_users_list}
        - Current Date: {current_pst_date}
        - Server Time: {current_pst_time}
        - Current User: {user_nick_or_name}
        """
    ).strip()

    if "Member Activity" in categories:
        context += "\n" + build_member_activity_context(server_info)

    return context, user_message


def build_member_activity_context(server_info):
    """Build the member-activity context block. Takes pre-computed server_info
    instead of recomputing it (was being called twice per message)."""
    online_table = "\n".join(
        f"{m[0]} - Roles: {', '.join(m[1])} - Joined {m[2]} days ago - Permissions: {' '.join(m[3])}"
        for m in server_info["online_members"]
    ) or "No one"

    playing_info = ", ".join(
        f"{m[0]} is playing {m[1]}" for m in server_info["members_playing"]
    ) or "No one is playing any games"

    voice_chat_info_list = [
        f"In '{vc_name}': {', '.join(members)}"
        for vc_name, members in server_info["voice_channels_info"].items()
    ]
    voice_chat_info = "; ".join(voice_chat_info_list) or "No one is in voice chat"

    return dedent(
        f"""
        **Member Activities**:
        - Online Members:
        {online_table}
        - Current Activities: {playing_info}
        - Members in Voice Chat: {voice_chat_info}
        """
    ).strip()


def build_instruction(categories, bot_display_name, server_name):
    """Build the system instruction from config.json (loaded once at module level).

    The `initial` instruction in config.json supports {bot_display_name} and
    {server_name} placeholders, which are filled in here. Static server lore
    from server_info.md is appended if present.
    """
    instruction = _CONFIG["instructions"]["initial"].format(
        bot_display_name=bot_display_name,
        server_name=server_name,
    )

    if _SERVER_LORE:
        instruction += f"\n\n# Server-specific knowledge:\n{_SERVER_LORE}"

    special_instructions = _CONFIG["special_instructions"]
    for category in categories:
        if category in special_instructions:
            instruction += (
                f"\nThere are your special instructions for this message: "
                f"{special_instructions[category]}"
            )
    return instruction


async def _execute_tool_call(tool_call, message):
    """Run a single LLM-requested tool call and return the result string.

    The result string is sent back to the LLM as the tool result, which the
    LLM then uses to compose its final chat reply.
    """
    name = tool_call["name"]
    args = tool_call["arguments"]
    guild = message.guild
    requester = message.author

    logger.info(f"  tool call: {name}({args})")

    try:
        if name == "play_music":
            return await voice_manager.play(guild, requester, args.get("query", ""))
        elif name == "skip_track":
            return await voice_manager.skip(guild)
        elif name == "pause_music":
            return await voice_manager.pause(guild)
        elif name == "resume_music":
            return await voice_manager.resume(guild)
        elif name == "stop_music":
            return await voice_manager.stop(guild)
        elif name == "leave_voice":
            return await voice_manager.leave(guild)
        elif name == "now_playing":
            return await voice_manager.now_playing(guild)
        elif name == "get_queue":
            return await voice_manager.get_queue(guild)
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        logger.error(f"  tool {name} raised: {e}")
        return f"Tool error: {e}"


async def handle_bot_mention(message, client):
    """Handle a message that mentions the bot, with structured timing logs.

    Runs categorization, history fetch, and server-info gathering all in
    parallel since none of them depend on each other's results. Then makes
    an LLM call with music tools available; if the LLM calls a tool, we
    execute it and make a second LLM call with the tool result so it can
    compose a natural reply.
    """
    user_name = clean_username(message.author.nick, message.author.name)
    msg_len = len(message.content)
    t_start = time.perf_counter()

    logger.info(f"→ {user_name} ({msg_len} chars): {message.content[:80]!r}")

    t_prep = time.perf_counter()
    categories, history_messages, server_info = await asyncio.gather(
        categorize_message(message.content),
        fetch_recent_messages(message.channel, client, limit=20),
        asyncio.to_thread(get_server_info, message.guild),
    )
    prep_ms = (time.perf_counter() - t_prep) * 1000

    logger.info(f"  categorized as {categories or ['(none)']} | prep took {prep_ms:.0f}ms")

    context_info, user_message = build_context_info(message, categories, client, server_info)
    instruction = build_instruction(
        categories,
        client.user.display_name,
        message.guild.name,
    )

    # Build the full message list. History is spliced in as actual role-tagged
    # turns (user/assistant) instead of being stuffed into a system block as a
    # transcript — this stops the model from mimicking transcript formatting
    # in its replies.
    messages = [
        {"role": "system", "content": instruction},
        {"role": "system", "content": context_info},
        *history_messages,
        {"role": "user", "content": f"[{user_name}] {user_message}"},
    ]

    # First LLM call — may return text or tool calls
    t_llm = time.perf_counter()
    response_data = await send_to_openai(
        {"messages": messages, "temperature": 0.5},
        tools=MUSIC_TOOLS,
    )
    llm_ms = (time.perf_counter() - t_llm) * 1000

    if not response_data:
        logger.error(f"  LLM call failed after {llm_ms:.0f}ms")
        try:
            await message.reply(
                "Sorry, I encountered an issue processing your request. Please try again later.",
                mention_author=False,
            )
        except Exception as e:
            logger.error(f"Failed to send error reply: {e}")
        return

    msg = response_data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls")

    # If the LLM called tools, run them and ask the LLM again for a final reply
    if tool_calls:
        # Append the assistant's tool-call message to the conversation
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

        # Execute each tool, append result message
        for tc in tool_calls:
            result = await _execute_tool_call(tc, message)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        # Second LLM call — composes natural reply using tool results
        t_llm2 = time.perf_counter()
        response_data = await send_to_openai(
            {"messages": messages, "temperature": 0.5},
        )
        llm2_ms = (time.perf_counter() - t_llm2) * 1000
        llm_ms += llm2_ms

        if not response_data:
            logger.error(f"  follow-up LLM call failed after {llm2_ms:.0f}ms")
            try:
                await message.reply(
                    "Hice la acción pero algo falló al armar la respuesta.",
                    mention_author=False,
                )
            except Exception as e:
                logger.error(f"Failed to send error reply: {e}")
            return

    response_text = process_openai_response(data=response_data, message=message, client=client)
    await send_response_to_channel(message, response_text)

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"← replied to {user_name} ({len(response_text)} chars) | "
        f"llm {llm_ms:.0f}ms | total {total_ms:.0f}ms"
    )


async def fetch_recent_messages(channel, client, limit=20):
    """Fetch recent messages and return them as a list of role-tagged
    OpenAI message dicts (chronological order).

    Past bot messages → role 'assistant' (clean content, no name prefix).
    Past user messages → role 'user' with '[Name] content' format so the
    model knows who said what without seeing transcript-style 'Name:' lines.
    """
    messages = [msg async for msg in channel.history(limit=limit)]
    messages.reverse()

    bot_id = client.user.id

    # Build a member lookup once for cleaning up @mentions inside message content
    member_mention_map = {
        member.mention: clean_username(member.nick, member.name)
        for member in channel.guild.members
    }

    history = []
    for msg in messages:
        # Skip empty messages (e.g. embeds-only)
        if not msg.content:
            continue

        content = msg.content
        for mention_str, replacement in member_mention_map.items():
            content = content.replace(mention_str, replacement)

        if msg.author.id == bot_id:
            # Bot's past messages — clean assistant turns, no name prefix
            history.append({"role": "assistant", "content": content})
        else:
            # User messages — tag with the speaker's name in brackets
            if isinstance(msg.author, discord.Member):
                speaker = clean_username(msg.author.nick, msg.author.name)
            else:
                speaker = msg.author.name
            history.append({"role": "user", "content": f"[{speaker}] {content}"})

    return history


def process_openai_response(data, message, client):
    """Extract and clean the assistant's response text from the API payload."""
    if "choices" not in data:
        logger.error(f"Unexpected OpenAI API response: {data}")
        return "Sorry, I encountered an issue processing your request."

    raw_content = data["choices"][0]["message"]["content"]
    # If LLM returned tool calls only with no text, give it a generic ack
    if not raw_content:
        return "Listo."

    response_text = raw_content.strip()
    bot_display_name = client.user.display_name
    user_name = clean_username(message.author.nick, message.author.name)

    # Patterns the model sometimes leaks from the system prompt format
    patterns_to_strip = [
        re.compile(r"^" + re.escape(bot_display_name) + r":"),
        re.compile(r"^\[\d{1,2}:\d{2} (AM|PM)\] " + re.escape(bot_display_name) + r":"),
        re.compile(r"^" + re.escape(user_name) + r"[:,\s]"),
        re.compile(r"^\[" + re.escape(user_name) + r"\]\s*"),
    ]
    for pattern in patterns_to_strip:
        if pattern.match(response_text):
            response_text = pattern.sub("", response_text).strip()

    return response_text


async def send_response_to_channel(message, response_text):
    """Send the bot's response as a native Discord reply to the user's message."""
    try:
        await message.reply(response_text, mention_author=False)
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        try:
            await message.reply(
                "An error occurred. Please try again...",
                mention_author=False,
            )
        except Exception as e2:
            # If even the fallback fails, log and give up — don't crash the handler
            logger.error(f"Fallback send also failed: {e2}")