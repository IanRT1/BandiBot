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
from openai_utils import send_to_openai, ALL_TOOLS
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
    """Remove bot mentions from message content cleanly."""
    content = message.content
    for mention_str in (client.user.mention, f"<@!{client.user.id}>"):
        content = content.replace(mention_str, "")
    return content.strip()


def build_context_info(message, client):
    """Build base context block for the system prompt.

    No longer takes categories or server_info — member activity is fetched
    on-demand via the get_member_activity tool instead of being pre-computed
    and conditionally included.
    """
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

    context = dedent(
        f"""
        **Server Information**:
        - Server Name: {server_name}
        - Active Channel: {channel_name}
        - Bot Name: {bot_display_name}
        - Server Creation Date: {creation_date}
        - Server Owner: {server_owner}
        - Total Members: {total_members}
        - Current Date: {current_pst_date}
        - Server Time: {current_pst_time}
        - Current User: {user_nick_or_name}
        """
    ).strip()

    return context, user_message


def build_member_activity_context(guild):
    """Fetch and format member activity on-demand.

    Called only when the LLM requests it via the get_member_activity tool,
    not on every message.
    """
    server_info = get_server_info(guild)

    online_users_list = ", ".join(info[0] for info in server_info["online_members"])
    online_count = server_info["online_count"]

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
        **Member Activity**:
        - Online: {online_count} ({online_users_list})
        - Online Members Detail:
        {online_table}
        - Current Activities: {playing_info}
        - Members in Voice Chat: {voice_chat_info}
        """
    ).strip()


def build_instruction(bot_display_name, server_name):
    """Build the system instruction from config.json.

    No longer takes categories — special instructions were tied to
    categorization which has been removed.
    """
    instruction = _CONFIG["instructions"]["initial"].format(
        bot_display_name=bot_display_name,
        server_name=server_name,
    )

    if _SERVER_LORE:
        instruction += f"\n\n# Server-specific knowledge:\n{_SERVER_LORE}"

    return instruction


def _is_music_tool(name: str) -> bool:
    """Return True if the tool name is a music-related tool."""
    return name in {
        "play_music", "skip_track", "pause_music", "resume_music",
        "stop_music", "leave_voice", "now_playing", "get_queue",
    }


async def _execute_tool_call(tool_call, message):
    """Run a single LLM-requested tool call and return the result string."""
    name = tool_call["name"]
    args = tool_call["arguments"]
    guild = message.guild
    requester = message.author

    logger.info(f"  tool call: {name}({args})")

    try:
        if name == "play_music":
            result = await voice_manager.play(guild, requester, args.get("query", ""))
            player = voice_manager.get_player(guild)

            if result.startswith("Now playing:"):
                from now_playing_view import post_now_playing
                track = player.current
                if track:
                    await post_now_playing(
                        message.channel,
                        player,
                        title=track.title,
                        artist=track.artist,
                        duration_seconds=track.duration,
                        queue_size=len(player.queue),
                        requested_by=track.requested_by,
                        thumbnail_url=track.thumbnail,
                    )

            elif result.startswith("Queued"):
                from now_playing_view import update_now_playing_queue
                await update_now_playing_queue(player, len(player.queue))

            return result

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
        elif name == "get_member_activity":
            return await asyncio.to_thread(build_member_activity_context, guild)
        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        logger.error(f"  tool {name} raised: {e}")
        return f"Tool error: {e}"

async def handle_bot_mention(message, client):
    """Handle a message that mentions the bot.

    Flow:
    - Fetch history only (no pre-computed server_info, no categorization)
    - Single LLM call with all tools available
    - If music tool called: execute → lightweight reply call (no full context)
    - If get_member_activity called: execute → full context reply call
    - If no tool: use first response directly
    """
    user_name = clean_username(message.author.nick, message.author.name)
    msg_len = len(message.content)
    t_start = time.perf_counter()

    logger.info(f"→ {user_name} ({msg_len} chars): {message.content[:80]!r}")

    # Fetch history only — server info deferred to tool calls
    t_prep = time.perf_counter()
    history_messages = await fetch_recent_messages(message.channel, client, limit=20)
    prep_ms = (time.perf_counter() - t_prep) * 1000

    logger.info(f"  prep took {prep_ms:.0f}ms")

    context_info, user_message = build_context_info(message, client)
    instruction = build_instruction(
        client.user.display_name,
        message.guild.name,
    )

    messages = [
        {"role": "system", "content": instruction},
        {"role": "system", "content": context_info},
        *history_messages,
        {"role": "user", "content": f"[{user_name}] {user_message}"},
    ]

    # Single LLM call — decides to chat or call a tool
    t_llm = time.perf_counter()
    response_data = await send_to_openai(
        {"messages": messages, "temperature": 0.5},
        tools=ALL_TOOLS,
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

    if tool_calls:
        called_music = any(_is_music_tool(tc["name"]) for tc in tool_calls)

        # Append assistant's tool-call turn
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

        # Execute all tools, append results
        for tc in tool_calls:
            result = await _execute_tool_call(tc, message)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        # Second LLM call to compose the reply.
        # Music commands get a lightweight prompt — no need for full server
        # context or history just to say "Listo, poniendo X".
        if called_music:
            tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
            reply_messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are {client.user.display_name}, a chill Discord bot. "
                        f"Respond naturally in the same language as the user. "
                        f"Keep it short — one or two sentences max."
                    ),
                },
                {"role": "user", "content": f"[{user_name}] {user_message}"},
                {
                    "role": "assistant",
                    "content": None,
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
                },
                *[{"role": "tool", "tool_call_id": tc["id"], "content": r}
                for tc, r in zip(tool_calls, tool_results)],
            ]
        else:
            # Non-music tools (member activity etc.) — use full context
            reply_messages = messages

        t_llm2 = time.perf_counter()
        response_data = await send_to_openai(
            {"messages": reply_messages, "temperature": 0.5},
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
    """Fetch recent messages and return them as role-tagged OpenAI message dicts."""
    messages = [msg async for msg in channel.history(limit=limit)]
    messages.reverse()

    bot_id = client.user.id

    member_mention_map = {
        member.mention: clean_username(member.nick, member.name)
        for member in channel.guild.members
    }

    history = []
    for msg in messages:
        if not msg.content:
            continue

        content = msg.content
        for mention_str, replacement in member_mention_map.items():
            content = content.replace(mention_str, replacement)

        if msg.author.id == bot_id:
            history.append({"role": "assistant", "content": content})
        else:
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
    if not raw_content:
        return "Listo."

    response_text = raw_content.strip()
    bot_display_name = client.user.display_name
    user_name = clean_username(message.author.nick, message.author.name)

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
    """Send the bot's response as a native Discord reply."""
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
            logger.error(f"Fallback send also failed: {e2}")