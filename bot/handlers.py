"""
bot/handlers.py

Message handling for BandiBot's text channel interactions.

Processes @mention messages, builds LLM context, routes tool calls through
bot/tool_executor.py, and sends responses back to the Discord channel.

Request flow:
  @mention received → check for audio attachments (mp3/wav/flac/opus)
  → if found: extract metadata + cover art, queue directly, skip LLM entirely
  → if not: fetch channel history → build context → LLM call
  → tool calls executed → follow-up LLM call → reply sent

Tool execution:
  Tool calls route through execute_tool_call(), which is shared between text
  commands (real discord.Message) and voice commands (_FakeMsgProxy).
  Music tool responses use a stripped-down follow-up prompt for concise
  confirmations. Non-music tools use the full conversation context.

Context building:
  - instructions.txt    → bot identity and behavior rules (loaded once)
  - server_info.txt     → private server lore, retrieved locally before the LLM call
  - channel history     → last 20 messages for conversational continuity
  - member activity     → real-time presence fetched on demand via tool call
  - server context      → name, owner, channel, time, current user

Music tools bypass the full follow-up flow and use a minimal prompt
to keep confirmations short and language-matched to the user.
"""
import re
import time
import json
import logging
from textwrap import dedent

import discord

from bot.utils import (
    clean_username,
    get_current_pst_time,
    get_current_pst_date,
    get_server_info,
)
from bot.tool_schemas import (
    select_tools_for_request,
)
from bot.openai_client import send_to_openai
from bot.tool_executor import execute_tool_call, is_music_tool
from core.config import OPENAI_MODEL
from core.interaction_logging import log_message, log_done, track_token_usage
from bot.retrieval import (
    format_retrieved_context,
    load_context_file,
    load_server_lore,
    build_retrieval_query,
    retrieve_with_confidence,
    should_retrieve_lore,
)

from music.player import voice_manager
from music.attachments import get_audio_attachments, handle_audio_attachments

logger = logging.getLogger(__name__)

_INSTRUCTIONS_TEMPLATE = load_context_file("instructions.txt")
_SERVER_LORE = load_server_lore(OPENAI_MODEL)


def _strip_bot_mentions(message, client):
    content = message.content
    for mention_str in (client.user.mention, f"<@!{client.user.id}>"):
        content = content.replace(mention_str, "")
    return content.strip()


def build_context_info(message, client, history_messages=None):
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
    retrieval_query = build_retrieval_query(user_message, history_messages)
    if should_retrieve_lore(retrieval_query, _SERVER_LORE):
        lore_chunks, lore_is_confident = retrieve_with_confidence(
            _SERVER_LORE, retrieval_query
        )
    else:
        lore_chunks, lore_is_confident = [], False

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

    if lore_chunks:
        logger.debug("[rag] retrieved %d server-lore chunk(s) for text context", len(lore_chunks))
        context += "\n\n" + format_retrieved_context(lore_chunks)

    return context, user_message, lore_is_confident, bool(lore_chunks)


def build_server_info_context(question: str = "") -> str:
    if not _SERVER_LORE:
        return "No server info available."
    chunks, _ = retrieve_with_confidence(_SERVER_LORE, question)
    if not chunks:
        logger.debug("[rag] no matching lore for server-info lookup")
        return (
            "No directly matching server lore was found for this question. "
            "Do not infer or invent an answer; state that the server information "
            "is not documented if necessary."
        )
    lore = "\n\n".join(chunks)
    return "The following is the official server history and lore. Treat it as confirmed canon and answer based on it directly:\n\n" + lore


def build_member_activity_context(guild):
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
    return _INSTRUCTIONS_TEMPLATE.format(
        bot_display_name=bot_display_name,
        server_name=server_name,
    )


@track_token_usage
async def handle_bot_mention(message, client):
    user_name = clean_username(message.author.nick, message.author.name)
    t_start = time.perf_counter()

    log_message(logger, "chat", "user", user_name, _strip_bot_mentions(message, client))

    # ── Audio attachment shortcut — bypass LLM entirely ───────────────────────
    audio_attachments = get_audio_attachments(message)
    if audio_attachments:
        logger.info(f"  {len(audio_attachments)} audio attachment(s) detected — queuing directly")
        await handle_audio_attachments(message, audio_attachments)
        log_done(logger, "chat", (time.perf_counter() - t_start) * 1000)
        return

    async with message.channel.typing():
        t_prep = time.perf_counter()
        history_messages = await fetch_recent_messages(
            message.channel,
            client,
            limit=20,
            exclude_message_id=message.id,
        )
        prep_ms = (time.perf_counter() - t_prep) * 1000

        logger.debug("[chat] context prepared in %.0fms", prep_ms)

        (
            context_info,
            user_message,
            has_retrieved_lore,
            has_lore_context,
        ) = build_context_info(message, client, history_messages)
        instruction = build_instruction(
            client.user.display_name,
            message.guild.name,
        )

        past_history = [
            {
                "role": msg["role"],
                "content": f"[PAST] {msg['content']}",
            }
            for msg in history_messages
        ]

        messages = [
            {"role": "system", "content": instruction},
            {"role": "system", "content": context_info},
            {"role": "system", "content": (
                "The channel history below is PAST CONTEXT ONLY. "
                "For tool calls, especially music tools, act only on the CURRENT MESSAGE at the end. "
                "Never play, queue, skip, delete, or move music based on a past message. "
                "For music requests, clean only the CURRENT MESSAGE into the YouTube search query. "
                "If the user gives a plausible title plus artist, preserve it literally and do not replace it with a more famous song by that artist. "
                "Command-like words inside a requested song title are not control commands. "
                "Only stop or clear music when the current message explicitly asks to stop playback or clear the queue. "
                "When the user asks to remove/quit the song currently playing without naming a queue position, use skip_track; "
                "use delete_track only for a song in the upcoming queue. "
                "If the user provides a YouTube video ID, pass that exact ID unchanged."
            )},
            *past_history,
        ]

        player = voice_manager.get_player(message.guild)
        if player.current or player.queue:
            queue_str = await voice_manager.get_queue(message.guild)
            messages.append({"role": "system", "content": f"**Current Queue:**\n{queue_str}\n\nDo NOT re-queue any song already in this list."})

        messages.extend([
            {"role": "system", "content": "━━━ CURRENT MESSAGE BELOW — ACT ON THIS ONLY ━━━"},
            {"role": "user", "content": f"[{user_name}] {user_message}"},
        ])

        t_llm = time.perf_counter()
        available_tools = select_tools_for_request(
            user_message,
            lore_is_confident=has_retrieved_lore,
            has_lore_context=has_lore_context,
        )

        response_data = await send_to_openai(
            {"messages": messages, "temperature": 0.5},
            tools=available_tools,
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
            called_music = any(is_music_tool(tc["name"]) for tc in tool_calls)

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
                result = await execute_tool_call(tc, message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            if called_music:
                tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
                reply_messages = [
                    {
                        "role": "system",
                        "content": (
                            f"You are {client.user.display_name}, a chill Discord bot. "
                            f"Respond in the same language the user wrote in. The user wrote: '{user_message}'. Match that language exactly. "
                            f"Keep it short — one or two sentences max. "
                            f"The tool result tells you exactly what happened — confirm it confidently, don't ask for clarification."
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
    log_message(logger, "chat", "bot", client.user.display_name, response_text)
    logger.debug(
        "[chat] response processed | chars=%d | llm=%.0fms | total=%.0fms",
        len(response_text), llm_ms, total_ms,
    )
    log_done(logger, "chat", total_ms)


async def fetch_recent_messages(channel, client, limit=20, exclude_message_id=None):
    messages = [msg async for msg in channel.history(limit=limit)]
    messages.reverse()

    bot_id = client.user.id

    member_mention_map = {
        member.mention: clean_username(member.nick, member.name)
        for member in channel.guild.members
    }

    MAX_MESSAGE_CHARS = 500

    history = []
    for msg in messages:
        if exclude_message_id is not None and msg.id == exclude_message_id:
            continue
        if not msg.content:
            continue

        content = msg.content
        for mention_str, replacement in member_mention_map.items():
            content = content.replace(mention_str, replacement)

        if len(content) > MAX_MESSAGE_CHARS:
            content = content[:MAX_MESSAGE_CHARS] + "… [truncated]"

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
