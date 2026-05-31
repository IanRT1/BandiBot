"""
voice/clips.py

Voice-channel clip export for BandiBot.

GuildVoiceSession owns a rolling PCM buffer containing recent mixed voice,
music, and TTS frames. This module turns that buffer into a Discord-ready MP3
attachment when the LLM invokes the clip_audio tool.

Pipeline:
  GuildVoiceSession clip buffer → raw 48kHz stereo PCM → ffmpeg MP3 encode
  → timestamped Discord file upload → tool result string

Operational behavior:
  - Returns clear tool-result text for every failure path
  - Uses ffmpeg through stdin/stdout so no temporary files are created
  - Names clips with requester and Pacific timestamp for easy identification
  - Leaves capture ownership in voice/listener.py and tool dispatch ownership
    in bot/tool_executor.py
"""

import io
import logging
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import discord

from bot.utils import clean_username

logger = logging.getLogger(__name__)
_PACIFIC = ZoneInfo("America/Los_Angeles")


async def send_recent_clip(guild, requester, channel) -> str:
    from voice.listener import voice_listener_manager

    session = voice_listener_manager.get_session(guild)
    if not session or not session.sink:
        return "Not in a voice channel."

    pcm = session.get_clip_pcm()
    if not pcm:
        logger.warning("[clip] no audio in buffer")
        return "No audio captured yet."

    logger.info(f"[clip] encoding {len(pcm)} bytes of PCM")
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "s16le", "-ar", "48000", "-ac", "2",
                "-i", "pipe:0",
                "-codec:a", "libmp3lame", "-q:a", "4",
                "-f", "mp3", "pipe:1",
            ],
            input=pcm,
            capture_output=True,
            timeout=15,
        )
        mp3_bytes = proc.stdout
        if not mp3_bytes:
            logger.error(f"[clip] ffmpeg produced no output. stderr: {proc.stderr.decode()}")
            return "Failed to encode clip."
        logger.info(f"[clip] encoded {len(mp3_bytes)} bytes, sending to channel={channel}")
    except Exception as e:
        logger.error(f"[clip] ffmpeg error: {e}")
        return "Failed to encode clip."

    if not channel:
        logger.warning("[clip] no text channel to send to")
        return "Clip encoded but no channel to send it to."

    timestamp = datetime.now(_PACIFIC).strftime("%Y-%m-%d_%H-%M")
    requester_name = clean_username(getattr(requester, "nick", None), requester.name).replace(" ", "_")
    filename = f"clip_{requester_name}_{timestamp}.mp3"

    await channel.send(
        file=discord.File(fp=io.BytesIO(mp3_bytes), filename=filename)
    )
    logger.info(f"[clip] sent as {filename}")
    return f"Clip sent to #{channel.name} as {filename}."
