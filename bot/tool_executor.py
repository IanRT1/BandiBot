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
import json
import logging
import re
from collections import deque

from music.player import voice_manager
from music.results import PlayResult

logger = logging.getLogger(__name__)


def is_music_tool(name: str) -> bool:
    return name in {
        "play_music", "select_song_candidate", "skip_track", "pause_music", "resume_music",
        "stop_music", "move_track", "delete_track", "queue_bulk",
        "undo_last_song_request",
    }


async def _handle_play_music(message, args):
    guild = message.guild
    requester = message.author
    is_voice = _is_voice_proxy(message)
    player = voice_manager.get_player(guild)
    if not player.text_channel and message.channel:
        player.text_channel = message.channel

    from music.requests import song_query
    from music.identity import clarifications
    from bot.interactions import current_request

    context = current_request.get()
    original_text = context.original_text if context else getattr(message, "content", None)
    query = song_query(args["song"], original_text=original_text) if "song" in args else args.get("query", "")
    clarifications.clear(getattr(guild, "id", None), getattr(requester, "id", None))
    search_msg = None
    if is_voice and message.channel:
        try:
            search_msg = await message.channel.send(f"Heard: *{query}*")
        except Exception:
            pass

    try:
        # The controller owns the deadline and commit cancellation together.
        # An outer wait_for could report failure while shielded work still commits.
        result = await voice_manager.play(guild, requester, query)
    except asyncio.TimeoutError:
        if search_msg:
            try:
                await search_msg.edit(content="Took too long to resolve that track.")
            except Exception:
                pass
        return json.dumps(PlayResult("failed", error_code="timeout", message="Took too long to resolve that track. Try again.").to_dict())
    except asyncio.CancelledError:
        if search_msg:
            try:
                await search_msg.delete()
            except Exception:
                pass
        raise

    if search_msg:
        try:
            if result.status == "queued":
                track_title = result.title
                await search_msg.edit(content=f"Queued: **{track_title}** — position {result.queue_position}")
            elif result.status in {"playing", "starting"}:
                await search_msg.delete()
            else:
                await search_msg.edit(content=result.message)
        except Exception:
            pass

    if result.status == "queued" and message.channel:
        from music.now_playing import update_now_playing_queue
        try:
            await update_now_playing_queue(player, len(player.queue))
        except Exception:
            logger.warning("[music] queued track but could not refresh queue banner", exc_info=True)

    return json.dumps(result.to_dict(), ensure_ascii=False)


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
    return json.dumps(result.to_dict(), ensure_ascii=False)


async def _handle_skip_track(message, args):
    return await voice_manager.skip(message.guild)


async def _handle_pause_music(message, args):
    return await voice_manager.pause(message.guild)


async def _handle_resume_music(message, args):
    return await voice_manager.resume(message.guild)


async def _handle_stop_music(message, args):
    if not _action_is_grounded(message, args):
        logger.warning(
            "  stop_music blocked: message did not explicitly ask to stop or clear music"
        )
        return (
            "Stop cancelled: the message did not explicitly ask to stop music "
            "or clear the queue."
        )
    return await voice_manager.stop(message.guild)


async def _handle_leave_voice(message, args):
    from voice.listener import voice_listener_manager

    player = voice_manager.get_player(message.guild)
    # Voice commands speak the goodbye before leaving; the voice pipeline owns
    # the final disconnect after that response completes.
    if _is_voice_proxy(message):
        player.schedule_leave()
        return "Leaving the voice channel."

    await voice_listener_manager.stop_listening(message.guild)
    await player.disconnect()
    return "Leaving the voice channel."


async def _handle_now_playing(message, args):
    return await voice_manager.now_playing(message.guild)


async def _handle_get_queue(message, args):
    return await voice_manager.get_queue(message.guild)


async def _handle_move_track(message, args):
    from bot.interactions import current_request
    context = current_request.get()
    player = voice_manager.get_player(message.guild)
    if context and context.queue_revision is not None and context.queue_revision != player.operations.revision:
        return "The queue changed; please select the track again."
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
    if not _action_is_grounded(message, args):
        return "Delete cancelled: the message did not explicitly ask to remove a track."

    if args.get("pending_request_id"):
        return await voice_manager.cancel_pending(message.guild, args["pending_request_id"])

    player = voice_manager.get_player(message.guild)
    queue_list = list(player.queue)
    positions = args.get("positions")
    track_name = args.get("track_name")

    if not positions and not args.get("position") and not track_name:
        return "Please identify an upcoming queue entry; use skip for the current track."

    if not positions:
        position = args.get("position")
        if position is None:
            query = (track_name or "").lower()
            matches = [i + 1 for i, track in enumerate(queue_list) if query and query in track.title.lower()]
            if len(matches) != 1:
                return "Please identify a unique queue entry or its position."
            position = matches[0]
        positions = [position]

    max_pos = len(queue_list)
    invalid = [p for p in positions if p < 1 or p > max_pos]
    if invalid:
        return f"Invalid position(s): {invalid}. Queue has {max_pos} songs."

    from bot.interactions import current_request
    context = current_request.get()
    revision = context.queue_revision if context else None
    if context and context.queue_entries and (args.get("positions") or args.get("position")):
        if any(pos > len(context.queue_entries) for pos in positions):
            return "The queue changed; please select the track again."
        entry_ids = [context.queue_entries[pos - 1] for pos in sorted(set(positions), reverse=True)]
    else:
        entry_ids = [queue_list[pos - 1].entry_id for pos in sorted(set(positions), reverse=True)]
    result = await voice_manager.delete_entries(message.guild, entry_ids, revision=revision)

    if _is_voice_proxy(message) and message.channel:
        await message.channel.send(result)
    return result


async def _handle_undo_last_song_request(message, args):
    if not _action_is_grounded(message, args):
        return "Undo cancelled: the current request did not authorize that action."
    return await voice_manager.undo(message.guild)


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


async def _handle_select_song_candidate(message, args):
    from music.identity import clarifications
    from music.resolver import _entry_webpage_url
    entry = clarifications.take(message.guild.id, message.author.id, args["candidate_id"])
    if entry is None:
        return json.dumps(PlayResult("failed", message="That song choice expired or belongs to another request.").to_dict())
    url = _entry_webpage_url(entry)
    if not url:
        return json.dumps(PlayResult("failed", message="That candidate has no playable source.").to_dict())
    return await _handle_play_music(message, {"query": url})


TOOL_HANDLERS = {
    "select_song_candidate": _handle_select_song_candidate,
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
    "undo_last_song_request": _handle_undo_last_song_request,
    "join_voice": _handle_join_voice,
    "get_server_info": _handle_get_server_info,
    "get_member_activity": _handle_get_member_activity,
    "clip_audio": _handle_clip_audio,
    "web_search": _handle_web_search,
}

_TOOL_ARGUMENT_KEYS = {
    "select_song_candidate": {"candidate_id"},
    "play_music": {"query", "song"},
    "queue_bulk": {"queries", "is_playlist"},
    "move_track": {"from_position", "to_position", "track_name"},
    "delete_track": {"positions", "position", "track_name", "pending_request_id"},
    "undo_last_song_request": set(),
    "get_server_info": {"question"},
    "web_search": {"question"},
}
_TOOL_REQUIRED_ARGUMENTS = {
    "select_song_candidate": {"candidate_id"},
    "play_music": set(),
    "queue_bulk": {"queries", "is_playlist"},
    "move_track": {"to_position"},
    "get_server_info": {"question"},
    "web_search": {"question"},
}


def _validate_tool_call(name, args) -> str | None:
    """Validate model output before handing it to a side-effecting handler."""
    if not isinstance(args, dict):
        return "Tool arguments must be an object."

    allowed = _TOOL_ARGUMENT_KEYS.get(name, set()) | {"response_language"}
    if name in {"stop_music", "delete_track", "undo_last_song_request"}:
        allowed |= {"intent_evidence"}
    unexpected = set(args) - allowed
    if unexpected:
        return f"Unexpected argument(s): {', '.join(sorted(unexpected))}."

    missing = _TOOL_REQUIRED_ARGUMENTS.get(name, set()) - set(args)
    if missing:
        return f"Missing required argument(s): {', '.join(sorted(missing))}."

    if name == "select_song_candidate":
        if not isinstance(args["candidate_id"], str) or len(args["candidate_id"]) > 64:
            return "Invalid candidate ID."
    if name == "play_music":
        if "song" in args:
            if "query" in args:
                return "Use song identity or a legacy query, not both."
            from music.requests import song_query
            try:
                song_query(args["song"])
            except ValueError as exc:
                return str(exc)
            return None
        # Internal direct-link routing and older callers still supply query.
        if not isinstance(args.get("query"), str) or not args["query"].strip():
            return "The music query must be a non-empty string."
        if len(args["query"]) > 500:
            return "The music query is too long."
    elif name == "queue_bulk":
        queries = args["queries"]
        if not isinstance(queries, list) or not all(isinstance(query, str) for query in queries):
            return "Queue queries must be a list of strings."
        if not queries or len(queries) > 50:
            return "Queue queries must contain between 1 and 50 items."
        if any(not query.strip() or len(query) > 500 for query in queries):
            return "Each queue query must be a non-empty string of 500 characters or fewer."
        if not isinstance(args["is_playlist"], bool):
            return "is_playlist must be a boolean."
    elif name == "move_track":
        if not isinstance(args["to_position"], int) or isinstance(args["to_position"], bool):
            return "to_position must be an integer."
        if args["to_position"] < 1:
            return "to_position must be at least 1."
        if "from_position" in args and (
            not isinstance(args["from_position"], int)
            or isinstance(args["from_position"], bool)
            or args["from_position"] < 1
        ):
            return "from_position must be a positive integer."
        if "track_name" in args and (
            not isinstance(args["track_name"], str) or len(args["track_name"]) > 500
        ):
            return "track_name must be a string of 500 characters or fewer."
    elif name == "delete_track":
        if "pending_request_id" in args and (not isinstance(args["pending_request_id"], str) or len(args["pending_request_id"]) > 128):
            return "Invalid pending request ID."
        if "positions" in args and (
            not isinstance(args["positions"], list)
            or not all(isinstance(position, int) and not isinstance(position, bool) for position in args["positions"])
        ):
            return "positions must be a list of integers."
        if "position" in args and (
            not isinstance(args["position"], int)
            or isinstance(args["position"], bool)
        ):
            return "position must be an integer."
        if "track_name" in args and not isinstance(args["track_name"], str):
            return "track_name must be a string."
    elif name in {"get_server_info", "web_search"}:
        if not isinstance(args["question"], str) or not args["question"].strip():
            return "question must be a non-empty string."
        if len(args["question"]) > 2000:
            return "question is too long."

    return None


async def execute_tool_call(tool_call, message):
    if not isinstance(tool_call, dict):
        return "Invalid tool call."
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})
    if not isinstance(name, str):
        return "Invalid tool name."
    logger.info("[tool] %s", name)

    if isinstance(args, dict) and args.get("response_language", "en") not in {"en", "es"}:
        return "Invalid response language."

    from core.capabilities import tool_available

    handler = TOOL_HANDLERS.get(name) if tool_available(name) else None
    if handler is None:
        return f"Unknown tool: {name}"

    validation_error = _validate_tool_call(name, args)
    if validation_error:
        logger.warning("[tool] rejected %s: %s", name, validation_error)
        return f"Invalid arguments for {name}: {validation_error}"

    try:
        result = await handler(message, args)
        return json.dumps(result.to_dict(), ensure_ascii=False) if hasattr(result, "to_dict") else result
    except Exception as e:
        logger.error(f"  tool {name} raised: {e}")
        if name == "play_music":
            return json.dumps(PlayResult("failed", error_code="tool_error", message=f"Tool error: {e}").to_dict())
        return f"Tool error: {e}"


def _is_voice_proxy(message) -> bool:
    return getattr(message, "origin", None) == "voice"


def _action_is_grounded(message, args) -> bool:
    """Validate evidence binding; language interpretation belongs to the model."""
    evidence = args.get("intent_evidence")
    return bool(isinstance(evidence, str) and evidence.strip()
                and evidence.casefold() in (getattr(message, "content", "") or "").casefold())
