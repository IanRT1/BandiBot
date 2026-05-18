"""
bot/handlers.py

Message handling and tool execution for BandiBot's text channel interactions.

Processes @mention messages, builds LLM context, executes tool calls, and
sends responses back to the Discord channel. Shared with the voice pipeline
via _FakeMsgProxy so tool execution logic is never duplicated.

Request flow:
  @mention received → check for audio attachments (mp3/wav/flac/opus)
  → if found: extract metadata + cover art, queue directly, skip LLM entirely
  → if not: fetch channel history → build context → LLM call
  → tool calls executed → follow-up LLM call → reply sent

Tool execution:
  All tool calls route through _execute_tool_call(), which is shared between
  text commands (real discord.Message) and voice commands (_FakeMsgProxy).
  Music tool responses use a stripped-down follow-up prompt for concise
  confirmations. Non-music tools use the full conversation context.

Context building:
  - instructions.txt    → bot identity and behavior rules (loaded once)
  - server_info.txt     → static server lore (loaded once, fetched on demand)
  - channel history     → last 20 messages for conversational continuity
  - member activity     → real-time presence fetched on demand via tool call
  - server context      → name, owner, channel, time, current user

Music tools bypass the full follow-up flow and use a minimal prompt
to keep confirmations short and language-matched to the user.
"""
import asyncio
import base64
import io
import json
import logging
import os
import re
import struct
import time
from textwrap import dedent
from zoneinfo import ZoneInfo

import aiohttp
import discord
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis

from bot.utils import (
    clean_username,
    get_current_pst_time,
    get_current_pst_date,
    get_server_info,
)
from bot.openai_client import send_to_openai, ALL_TOOLS
from music.player import voice_manager, Track

logger = logging.getLogger(__name__)

# Load instructions template once at import time
with open("data/instructions.txt", "r", encoding="utf-8") as _f:
    _INSTRUCTIONS_TEMPLATE = _f.read().strip()
logger.info(f"Loaded instructions.txt ({len(_INSTRUCTIONS_TEMPLATE)} chars)")

# Load static server lore once at import time
_SERVER_LORE = ""
if os.path.exists("data/server_info.txt"):
    with open("data/server_info.txt", "r", encoding="utf-8") as _f:
        _SERVER_LORE = _f.read().strip()
    logger.info(f"Loaded server_info.txt ({len(_SERVER_LORE)} chars)")

_PACIFIC = ZoneInfo("America/Los_Angeles")

# Audio file extensions treated as direct playback attachments
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".opus", ".ogg", ".m4a", ".aac"}


def _strip_bot_mentions(message, client):
    content = message.content
    for mention_str in (client.user.mention, f"<@!{client.user.id}>"):
        content = content.replace(mention_str, "")
    return content.strip()


def build_context_info(message, client):
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


def build_server_info_context(question: str = "") -> str:
    if not _SERVER_LORE:
        return "No server info available."
    return (
        "The following is the official server history and lore. "
        "Treat it as confirmed canon and answer based on it directly:\n\n"
        + _SERVER_LORE
    )


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


def _is_music_tool(name: str) -> bool:
    return name in {
        "play_music", "skip_track", "pause_music", "resume_music",
        "stop_music", "move_track", "delete_track", "queue_bulk",
    }


def _get_audio_attachments(message: discord.Message) -> list[discord.Attachment]:
    """Return attachments whose extension is in _AUDIO_EXTENSIONS."""
    result = []
    for attachment in message.attachments:
        ext = os.path.splitext(attachment.filename)[1].lower()
        if ext in _AUDIO_EXTENSIONS:
            result.append(attachment)
    return result


async def _extract_audio_metadata(url: str, ext: str) -> dict:
    """Download attachment bytes and extract title, artist, duration, cover art via mutagen."""
    result = {
        "title": None,
        "artist": None,
        "duration": None,
        "thumbnail_bytes": None,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logger.warning(f"[music] metadata fetch returned {resp.status}")
                    return result
                data = await resp.read()

        buf = io.BytesIO(data)

        if ext == ".flac":
            f = FLAC(buf)
            result["title"] = f.get("title", [None])[0]
            result["artist"] = f.get("artist", [None])[0] or f.get("performer", [None])[0]
            result["duration"] = int(f.info.length) if f.info else None
            if f.pictures:
                result["thumbnail_bytes"] = f.pictures[0].data

        elif ext == ".mp3":
            buf.seek(0)
            f = MP3(buf)
            result["duration"] = int(f.info.length) if f.info else None
            try:
                buf.seek(0)
                tags = ID3(buf)
                tit2 = tags.get("TIT2")
                tpe1 = tags.get("TPE1")
                result["title"] = str(tit2) if tit2 else None
                result["artist"] = str(tpe1) if tpe1 else None
                apic = tags.getall("APIC")
                if apic:
                    result["thumbnail_bytes"] = apic[0].data
            except Exception:
                pass

        elif ext in (".m4a", ".aac"):
            buf.seek(0)
            f = MP4(buf)
            result["duration"] = int(f.info.length) if f.info else None
            tags = f.tags or {}
            title = tags.get("\xa9nam")
            artist = tags.get("\xa9ART")
            result["title"] = title[0] if title else None
            result["artist"] = artist[0] if artist else None
            covr = tags.get("covr")
            if covr:
                result["thumbnail_bytes"] = bytes(covr[0])

        elif ext in (".ogg", ".opus"):
            buf.seek(0)
            f = OggVorbis(buf)
            result["title"] = f.get("title", [None])[0]
            result["artist"] = f.get("artist", [None])[0]
            result["duration"] = int(f.info.length) if f.info else None
            # Cover art in Ogg is stored as a base64-encoded FLAC PICTURE block
            pic_b64 = f.get("metadata_block_picture", [None])[0]
            if pic_b64:
                try:
                    pic_data = base64.b64decode(pic_b64)
                    offset = 4  # skip picture type int
                    mime_len = struct.unpack(">I", pic_data[offset:offset + 4])[0]
                    offset += 4 + mime_len
                    desc_len = struct.unpack(">I", pic_data[offset:offset + 4])[0]
                    offset += 4 + desc_len + 16  # skip description + width/height/depth/colors ints
                    img_len = struct.unpack(">I", pic_data[offset:offset + 4])[0]
                    offset += 4
                    result["thumbnail_bytes"] = pic_data[offset:offset + img_len]
                except Exception as e:
                    logger.warning(f"[music] ogg cover art parse failed: {e}")

    except Exception as e:
        logger.error(f"[music] metadata extraction failed: {e}")

    return result


async def _handle_audio_attachments(
    message: discord.Message,
    attachments: list[discord.Attachment],
    client,
) -> None:
    """Queue audio attachments directly into the music system, bypassing LLM.

    Downloads each attachment to extract embedded metadata (title, artist,
    duration, cover art). Falls back to filename if tags are absent.
    """
    requester = message.author

    if not requester.voice or not requester.voice.channel:
        await message.reply(
            "Únete a un canal de voz primero para que pueda reproducir el archivo.",
            mention_author=False,
        )
        return

    guild = message.guild
    voice_channel = requester.voice.channel
    player = voice_manager.get_player(guild)
    requested_by = clean_username(requester.nick, requester.name)

    async with voice_manager._get_play_lock(guild.id):
        await player.connect(voice_channel)

        if not player.text_channel:
            player.text_channel = message.channel

        queued_titles = []
        is_busy = player.is_playing or (
            player.is_connected and player.voice_client.is_paused()
        )

        for attachment in attachments:
            ext = os.path.splitext(attachment.filename)[1].lower()
            filename_title = os.path.splitext(attachment.filename)[0]

            logger.info(f"[music] extracting metadata from {attachment.filename!r}")
            meta = await _extract_audio_metadata(attachment.url, ext)

            title           = meta["title"] or filename_title
            artist          = meta["artist"]
            duration        = meta["duration"]
            thumbnail_bytes = meta["thumbnail_bytes"]

            track = Track(
                title=title,
                stream_url=attachment.url,
                requested_by=requested_by,
                webpage_url=attachment.url,
                duration=duration,
                thumbnail=None,
                thumbnail_bytes=thumbnail_bytes,
                artist=artist,
                resolved=True,
            )
            player.queue.append(track)
            queued_titles.append(title)
            logger.info(
                f"[music] queued attachment: {title!r} | "
                f"artist={artist!r} | duration={duration}s | "
                f"cover={'yes' if thumbnail_bytes else 'no'}"
            )

        if not is_busy:
            player.play_next()

    if len(queued_titles) == 1:
        if is_busy:
            await message.reply(f"Agregado a la cola: **{queued_titles[0]}**", mention_author=False)
        else:
            await message.reply(f"Reproduciendo: **{queued_titles[0]}**", mention_author=False)
    else:
        titles_str = "\n".join(f"• {t}" for t in queued_titles)
        await message.reply(
            f"Agregados {len(queued_titles)} archivos a la cola:\n{titles_str}",
            mention_author=False,
        )


async def _execute_tool_call(tool_call, message):
    name = tool_call["name"]
    args = tool_call["arguments"]
    guild = message.guild
    requester = message.author

    logger.info(f"  tool call: {name}({args})")

    try:
        if name == "play_music":
            try:
                result = await asyncio.wait_for(
                    voice_manager.play(guild, requester, args.get("query", "")),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                return "Took too long to resolve that track. Try again."

            player = voice_manager.get_player(guild)

            if not player.text_channel and message.channel:
                player.text_channel = message.channel

            if result.startswith("Now playing:"):
                if message.channel:
                    from music.now_playing import post_now_playing
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
                            thumbnail_bytes=track.thumbnail_bytes,
                        )

            elif result.startswith("Queued"):
                if message.channel:
                    from music.now_playing import update_now_playing_queue
                    await update_now_playing_queue(player, len(player.queue))
                    from voice.handler import _FakeMsgProxy
                    if isinstance(message, _FakeMsgProxy):
                        await message.channel.send(result)

            return result

        elif name == "queue_bulk":
            queries = args.get("queries", [])
            is_playlist = args.get("is_playlist", False)

            if not queries:
                return "No songs provided."

            if is_playlist:
                result = await voice_manager.queue_playlist(guild, requester, queries[0])
            else:
                result = await voice_manager.queue_bulk(guild, requester, queries)

            player = voice_manager.get_player(guild)
            if not player.text_channel and message.channel:
                player.text_channel = message.channel

            if message.channel:
                from music.now_playing import update_now_playing_queue
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
            from voice.listener import voice_listener_manager
            player = voice_manager.get_player(guild)
            player.queue.clear()
            if player.is_playing:
                player.voice_client.stop_playing()
            return "Leaving the voice channel."
        elif name == "now_playing":
            return await voice_manager.now_playing(guild)
        elif name == "get_queue":
            return await voice_manager.get_queue(guild)
        elif name == "move_track":
            from_pos = args.get("from_position")
            to_pos = args.get("to_position")
            if from_pos is None:
                query = args.get("track_name", "").lower()
                player = voice_manager.get_player(guild)
                queue_list = list(player.queue)
                match = next((i+1 for i, t in enumerate(queue_list) if query in t.title.lower()), None)
                if not match:
                    return f"Could not find '{query}' in queue."
                from_pos = match
            return await voice_manager.move_track(guild, int(from_pos), int(to_pos))
        elif name == "delete_track":
            position = args.get("position")
            if position is None:
                query = args.get("track_name", "").lower()
                player = voice_manager.get_player(guild)
                queue_list = list(player.queue)
                if query in ("last", "last song", "última", "última canción"):
                    position = len(queue_list)
                else:
                    match = next((i+1 for i, t in enumerate(queue_list) if query in t.title.lower()), None)
                    if not match:
                        return f"Could not find '{query}' in queue."
                    position = match
            return await voice_manager.delete_track(guild, int(position))
        elif name == "join_voice":
            if not requester.voice or not requester.voice.channel:
                return "User is not in a voice channel."
            from voice.listener import voice_listener_manager
            channel = requester.voice.channel
            loop = asyncio.get_event_loop()
            bot_client = guild.me._state._get_client()
            await voice_listener_manager.start_listening(guild, channel, bot_client, loop)
            player = voice_manager.get_player(guild)
            player.text_channel = message.channel
            return f"Joined {channel.name}."
        elif name == "get_server_info":
            question = args.get("question", "")
            result = build_server_info_context(question)
            return result
        elif name == "get_member_activity":
            return await asyncio.to_thread(build_member_activity_context, guild)
        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        logger.error(f"  tool {name} raised: {e}")
        return f"Tool error: {e}"


async def handle_bot_mention(message, client):
    user_name = clean_username(message.author.nick, message.author.name)
    msg_len = len(message.content)
    t_start = time.perf_counter()
    total_tokens = 0

    logger.info(f"→ {user_name} ({msg_len} chars): {message.content[:80]!r}")

    # ── Audio attachment shortcut — bypass LLM entirely ───────────────────────
    audio_attachments = _get_audio_attachments(message)
    if audio_attachments:
        logger.info(f"  {len(audio_attachments)} audio attachment(s) detected — queuing directly")
        await _handle_audio_attachments(message, audio_attachments, client)
        return

    async with message.channel.typing():
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

        total_tokens += response_data.get("usage", {}).get("total_tokens", 0)

        msg = response_data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls")

        if tool_calls:
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
                result = await _execute_tool_call(tc, message)
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

            total_tokens += response_data.get("usage", {}).get("total_tokens", 0)

        response_text = process_openai_response(data=response_data, message=message, client=client)
        await send_response_to_channel(message, response_text)

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"← replied to {user_name} ({len(response_text)} chars) | "
        f"llm {llm_ms:.0f}ms | tokens {total_tokens} | total {total_ms:.0f}ms"
    )


async def fetch_recent_messages(channel, client, limit=20):
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