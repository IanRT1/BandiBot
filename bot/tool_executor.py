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
from collections import deque

from music.player import voice_manager

logger = logging.getLogger(__name__)


def is_music_tool(name: str) -> bool:
    return name in {
        "play_music", "skip_track", "pause_music", "resume_music",
        "stop_music", "move_track", "delete_track", "queue_bulk",
    }


async def execute_tool_call(tool_call, message):
    name = tool_call["name"]
    args = tool_call["arguments"]
    guild = message.guild
    requester = message.author

    logger.info(f"  tool call: {name}({args})")

    try:
        if name == "play_music":
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
                        await search_msg.edit(
                            content=f"Heard: *{query}*\nTook too long to resolve that track."
                        )
                    except Exception:
                        pass
                return "Took too long to resolve that track. Try again."

            if search_msg:
                try:
                    if result.startswith("Queued"):
                        track_title = result.split(": ", 1)[1] if ": " in result else result
                        await search_msg.edit(content=f"Heard: *{query}*\nQueued: **{track_title}**")
                    elif result.startswith("Now playing:"):
                        track_title = result.split(": ", 1)[1] if ": " in result else result
                        await search_msg.edit(content=f"Heard: *{query}*\nNow playing: **{track_title}**")
                    else:
                        await search_msg.edit(content=f"Heard: *{query}*\n{result}")
                except Exception:
                    pass

            if result.startswith("Queued"):
                if message.channel:
                    from music.now_playing import update_now_playing_queue
                    await update_now_playing_queue(player, len(player.queue))

            return result

        if name == "queue_bulk":
            queries = args.get("queries", [])
            is_playlist = args.get("is_playlist", False)

            if not queries:
                return "No songs provided."

            player = voice_manager.get_player(guild)

            if is_playlist:
                result = await voice_manager.queue_playlist(guild, requester, queries[0], text_channel=message.channel)
            else:
                result = await voice_manager.queue_bulk(guild, requester, queries, text_channel=message.channel)

            if message.channel:
                from music.now_playing import update_now_playing_queue
                await update_now_playing_queue(player, len(player.queue))

            return result

        if name == "skip_track":
            return await voice_manager.skip(guild)
        if name == "pause_music":
            return await voice_manager.pause(guild)
        if name == "resume_music":
            return await voice_manager.resume(guild)
        if name == "stop_music":
            return await voice_manager.stop(guild)
        if name == "leave_voice":
            player = voice_manager.get_player(guild)
            player.queue.clear()
            if player.is_playing:
                player.voice_client.stop_playing()
            return "Leaving the voice channel."
        if name == "now_playing":
            return await voice_manager.now_playing(guild)
        if name == "get_queue":
            return await voice_manager.get_queue(guild)
        if name == "move_track":
            from_pos = args.get("from_position")
            to_pos = args.get("to_position")
            if from_pos is None:
                query = args.get("track_name", "").lower()
                player = voice_manager.get_player(guild)
                queue_list = list(player.queue)
                match = next((i + 1 for i, t in enumerate(queue_list) if query in t.title.lower()), None)
                if not match:
                    return f"Could not find '{query}' in queue."
                from_pos = match
            result = await voice_manager.move_track(guild, int(from_pos), int(to_pos))
            if _is_voice_proxy(message) and message.channel:
                await message.channel.send(result)
            return result
        if name == "delete_track":
            player = voice_manager.get_player(guild)
            queue_list = list(player.queue)

            positions = args.get("positions")
            if not positions:
                position = args.get("position")
                if position is None:
                    query = args.get("track_name", "").lower()
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
        if name == "join_voice":
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
        if name == "get_server_info":
            from bot.handlers import build_server_info_context
            question = args.get("question", "")
            return build_server_info_context(question)
        if name == "get_member_activity":
            from bot.handlers import build_member_activity_context
            return await asyncio.to_thread(build_member_activity_context, guild)
        if name == "clip_audio":
            from voice.clips import send_recent_clip
            return await send_recent_clip(guild, requester, message.channel)

        return f"Unknown tool: {name}"

    except Exception as e:
        logger.error(f"  tool {name} raised: {e}")
        return f"Tool error: {e}"


def _is_voice_proxy(message) -> bool:
    return message.__class__.__name__ == "_FakeMsgProxy"
