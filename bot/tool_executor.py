"""
bot/tool_executor.py

Shared tool-call execution for BandiBot's text and voice interactions.

The LLM-facing function schemas live in bot/openai_client.py; this module is
the runtime dispatch layer that performs the side effects for those schemas.
Both text messages (real discord.Message instances) and voice commands
(_FakeMsgProxy from voice/handler.py) route through execute_tool_call() so
music, voice, clip, and server-context behavior stays consistent.

Responsibilities:
  - Music playback tools     → play, queue, skip, pause, resume, stop
  - Queue mutation tools     → move/delete tracks and update Now Playing
  - Voice session tools      → join voice, leave voice, export clips
  - Context lookup tools     → server lore and member activity

Boundary:
  This module executes already-decided tool calls. It does not define the tool
  schemas, build LLM prompts, transcribe audio, synthesize speech, or manage
  Discord event routing.
"""

import asyncio
import logging
import re
from collections import deque

from music.player import voice_manager

logger = logging.getLogger(__name__)


def is_music_tool(name: str) -> bool:
    return name in {
        "play_music", "skip_track", "pause_music", "resume_music",
        "stop_music", "move_track", "delete_track", "queue_bulk",
    }


async def _handle_play_music(message, args):
    guild = message.guild
    requester = message.author
    is_voice = _is_voice_proxy(message)
    player = voice_manager.get_player(guild)
    if not player.text_channel and message.channel:
        player.text_channel = message.channel

    query = args.get("query", "")
    search_msg = None
    if is_voice and message.channel:
        try:
            search_msg = await message.channel.send(f"Heard: *{query}*")
        except Exception:
            pass

    try:
        result = await asyncio.wait_for(
            voice_manager.play(guild, requester, query),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        if search_msg:
            try:
                await search_msg.edit(content="Took too long to resolve that track.")
            except Exception:
                pass
        return "Took too long to resolve that track. Try again."
    except asyncio.CancelledError:
        if search_msg:
            try:
                await search_msg.delete()
            except Exception:
                pass
        raise

    if search_msg:
        try:
            if result.startswith("Queued"):
                track_title = result.split(": ", 1)[1] if ": " in result else result
                await search_msg.edit(content=f"Queued: **{track_title}**")
            elif result.startswith("Now playing:"):
                await search_msg.delete()
            else:
                await search_msg.edit(content=result)
        except Exception:
            pass

    if result.startswith("Queued") and message.channel:
        from music.now_playing import update_now_playing_queue
        await update_now_playing_queue(player, len(player.queue))

    return result


async def _handle_queue_bulk(message, args):
    queries = args.get("queries", [])
    is_playlist = args.get("is_playlist", False)
    if not queries:
        return "No songs provided."

    guild = message.guild
    requester = message.author
    player = voice_manager.get_player(guild)
    if is_playlist:
        result = await voice_manager.queue_playlist(
            guild, requester, queries[0], text_channel=message.channel
        )
    else:
        result = await voice_manager.queue_bulk(
            guild, requester, queries, text_channel=message.channel
        )

    if message.channel:
        from music.now_playing import update_now_playing_queue
        await update_now_playing_queue(player, len(player.queue))
    return result


async def _handle_skip_track(message, args):
    return await voice_manager.skip(message.guild)


async def _handle_pause_music(message, args):
    return await voice_manager.pause(message.guild)


async def _handle_resume_music(message, args):
    return await voice_manager.resume(message.guild)


async def _handle_stop_music(message, args):
    if not _has_explicit_stop_intent(message):
        logger.warning(
            "  stop_music blocked: message did not explicitly ask to stop or clear music"
        )
        return (
            "Stop cancelled: the message did not explicitly ask to stop music "
            "or clear the queue."
        )
    return await voice_manager.stop(message.guild)


async def _handle_leave_voice(message, args):
    player = voice_manager.get_player(message.guild)
    player.queue.clear()
    if player.is_playing:
        player.voice_client.stop_playing()
    return "Leaving the voice channel."


async def _handle_now_playing(message, args):
    return await voice_manager.now_playing(message.guild)


async def _handle_get_queue(message, args):
    return await voice_manager.get_queue(message.guild)


async def _handle_move_track(message, args):
    from_pos = args.get("from_position")
    to_pos = args.get("to_position")
    if from_pos is None:
        query = args.get("track_name", "").lower()
        player = voice_manager.get_player(message.guild)
        queue_list = list(player.queue)
        match = next((i + 1 for i, t in enumerate(queue_list) if query in t.title.lower()), None)
        if not match:
            return f"Could not find '{query}' in queue."
        from_pos = match
    result = await voice_manager.move_track(message.guild, int(from_pos), int(to_pos))
    if _is_voice_proxy(message) and message.channel:
        await message.channel.send(result)
    return result


async def _handle_delete_track(message, args):
    if not _has_explicit_delete_intent(message):
        return "Delete cancelled: the message did not explicitly ask to remove a track."

    player = voice_manager.get_player(message.guild)
    queue_list = list(player.queue)
    positions = args.get("positions")
    track_name = args.get("track_name")

    # Natural language such as "quitar la canción" refers to the active song
    # unless the user explicitly identifies an upcoming queue item.  The
    # model normally selects skip_track after the schema guidance, but keep
    # this guard here so a delete_track call cannot silently leave playback
    # unchanged when the intent is clearly to skip the current track.
    if (
        player.current
        and not positions
        and not track_name
        and _has_current_track_skip_intent(message)
    ):
        return await voice_manager.skip(message.guild)

    if not positions:
        position = args.get("position")
        if position is None:
            query = (track_name or "").lower()
            if query in ("last", "last song", "última", "última canción"):
                position = len(queue_list)
            else:
                match = next((i + 1 for i, t in enumerate(queue_list) if query in t.title.lower()), None)
                if not match:
                    return f"Could not find '{query}' in queue."
                position = match
        positions = [position]

    max_pos = len(queue_list)
    invalid = [p for p in positions if p < 1 or p > max_pos]
    if invalid:
        return f"Invalid position(s): {invalid}. Queue has {max_pos} songs."

    results = []
    for pos in sorted(positions, reverse=True):
        track = queue_list.pop(pos - 1)
        results.append(track.title)

    player.queue = deque(queue_list)
    result = f"Removed {len(results)} song(s): " + ", ".join(f"'{t}'" for t in results)
    if _is_voice_proxy(message) and message.channel:
        await message.channel.send(result)
    return result


async def _handle_join_voice(message, args):
    requester = message.author
    if not requester.voice or not requester.voice.channel:
        return "User is not in a voice channel."
    from voice.listener import voice_listener_manager
    channel = requester.voice.channel
    loop = asyncio.get_event_loop()
    bot_client = message.guild.me._state._get_client()
    await voice_listener_manager.start_listening(message.guild, channel, bot_client, loop)
    player = voice_manager.get_player(message.guild)
    player.text_channel = message.channel
    return f"Joined {channel.name}."


async def _handle_get_server_info(message, args):
    from bot.handlers import build_server_info_context
    return build_server_info_context(args.get("question", ""))


async def _handle_get_member_activity(message, args):
    from bot.handlers import build_member_activity_context
    return await asyncio.to_thread(build_member_activity_context, message.guild)


async def _handle_clip_audio(message, args):
    from voice.clips import send_recent_clip
    return await send_recent_clip(message.guild, message.author, message.channel)


async def _handle_web_search(message, args):
    from bot.google_search import search_web
    return await search_web(args.get("question", ""))


TOOL_HANDLERS = {
    "play_music": _handle_play_music,
    "queue_bulk": _handle_queue_bulk,
    "skip_track": _handle_skip_track,
    "pause_music": _handle_pause_music,
    "resume_music": _handle_resume_music,
    "stop_music": _handle_stop_music,
    "leave_voice": _handle_leave_voice,
    "now_playing": _handle_now_playing,
    "get_queue": _handle_get_queue,
    "move_track": _handle_move_track,
    "delete_track": _handle_delete_track,
    "join_voice": _handle_join_voice,
    "get_server_info": _handle_get_server_info,
    "get_member_activity": _handle_get_member_activity,
    "clip_audio": _handle_clip_audio,
    "web_search": _handle_web_search,
}


async def execute_tool_call(tool_call, message):
    name = tool_call["name"]
    args = tool_call["arguments"]
    logger.info("[tool] %s", name)

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"

    try:
        return await handler(message, args)
    except Exception as e:
        logger.error(f"  tool {name} raised: {e}")
        return f"Tool error: {e}"


def _is_voice_proxy(message) -> bool:
    return message.__class__.__name__ == "_FakeMsgProxy"


def _has_explicit_delete_intent(message) -> bool:
    content = getattr(message, "content", "")
    if not content:
        return True
    lowered = content.lower()
    delete_words = (
        "remove", "delete", "clear", "drop", "take out", "take off",
        "quita", "quitar", "borra", "borrar", "elimina", "eliminar",
        "saca", "sacar",
    )
    return any(word in lowered for word in delete_words)


def _has_current_track_skip_intent(message) -> bool:
    """Return true for remove/skip wording aimed at the active track."""
    content = getattr(message, "content", "")
    if not content:
        return False
    lowered = content.lower()

    # Queue-specific wording means the user wants delete_track instead.
    if re.search(r"\b(?:queue|cola)\b", lowered):
        return False
    if re.search(r"\b(?:position|posici[oó]n|number|n[uú]mero)\b", lowered):
        return False

    return any(
        re.search(pattern, lowered)
        for pattern in (
            r"\bquitar(?:me)?\s+(?:la\s+)?canci[oó]n\b",
            r"\bquita(?:me)?\s+(?:la\s+)?canci[oó]n\b",
            r"\bsaca(?:r)?\s+(?:la\s+)?canci[oó]n\b",
            r"\b(?:remove|take out|take off)\s+(?:the\s+)?(?:current|this|playing)\s*(?:song|track)?\b",
        )
    )


def _has_explicit_stop_intent(message) -> bool:
    content = getattr(message, "content", "")
    if not content:
        return False

    lowered = content.lower()
    starts_as_play_request = re.match(
        r"^\s*(?:<@!?\d+>\s*)?"
        r"(?:play|queue|add|put on|pon|poner|reproduce|toca)\b",
        lowered,
    )
    if starts_as_play_request:
        return False

    stop_patterns = (
        r"\bstop\b",
        r"\bstop\s+(?:music|playback|playing|the song|the queue)\b",
        r"\bclear\s+(?:the\s+)?queue\b",
        r"\bclear\s+(?:all\s+)?music\b",
        r"\bempty\s+(?:the\s+)?queue\b",
        r"\bturn\s+off\s+(?:the\s+)?music\b",
        r"\bshut\s+up\b",
        r"\bstfu\b",
        r"\bpara\b",
        r"\bparen\b",
        r"\bdeten(?:er|te|lo|la)?\b",
        r"\bcancela(?:r)?\b",
        r"\blimpia\s+(?:la\s+)?cola\b",
        r"\bvac[ií]a\s+(?:la\s+)?cola\b",
        r"\bborra\s+(?:la\s+)?cola\b",
        r"\bquita\s+todo\b",
        r"\bapaga\s+(?:la\s+)?m[uú]sica\b",
    )
    return any(re.search(pattern, lowered) for pattern in stop_patterns)
