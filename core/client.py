"""
core/client.py

Entry point for BandiBot — Discord voice and chat assistant.

Manages the Discord client lifecycle, routes incoming messages and voice
state changes to the appropriate handlers, and implements exponential
backoff reconnection logic for network-level failures.

Architecture overview:
  Text commands  → handle_bot_mention (bot/handlers.py)
  Voice commands → voice_listener_manager (voice/listener.py)
  Music playback → voice_manager (music/player.py)
  TTS / mixing   → speak / MixerSource (voice/tts.py)
  STT            → transcribe (voice/stt.py)

The bot responds to @mentions in text channels and to a custom wake word
in voice channels. Voice interactions run a full wake word → VAD → STT →
LLM → TTS pipeline with interruption support and mid-speech cancellation.
"""
import os
import re
import sys
import time
from collections import deque

# ── Must be set before any HuggingFace/Kokoro imports ────────────────────────
# Prevents Kokoro from hitting huggingface.co on every TTS call to check
# for voice file updates. Uses cached files only.
os.environ["HF_HUB_OFFLINE"] = "1"

# ── Stderr filter — must be installed before any discord/voice/oww imports ───
# discord-ext-voice-recv and openwakeword write warnings directly to stderr
# via C extensions, bypassing Python's logging system entirely. We intercept
# here before those modules are imported so we catch everything.

class _PacketLossFilter:
    """
    Suppress noisy stderr warnings from discord-ext-voice-recv and openwakeword.

    - tflite runtime warnings: suppressed entirely (fires once per OWW model load)
    - packet loss warnings: aggregated and emitted as a single warning only when
      THRESH events accumulate within the WINDOW rolling period, then resets.
    """
    PATTERN        = re.compile(r"packets were lost being flushed")
    PATTERN_TFLITE = re.compile(r"tflite runtime")
    WINDOW  = 600   # seconds
    THRESH  = 100   # events within window before emitting aggregated warning

    def __init__(self, real_stderr):
        self._stderr = real_stderr
        self._times: deque = deque()

    def write(self, msg: str):
        if self.PATTERN_TFLITE.search(msg):
            return
        if self.PATTERN.search(msg):
            now = time.monotonic()
            self._times.append(now)
            while self._times and self._times[0] < now - self.WINDOW:
                self._times.popleft()
            if len(self._times) >= self.THRESH:
                self._stderr.write(
                    f"[WARNING] [voice] {len(self._times)} packet loss events "
                    f"in the last {self.WINDOW // 60} minutes — possible network instability\n"
                )
                self._stderr.flush()
                self._times.clear()
            return
        self._stderr.write(msg)

    def flush(self):
        self._stderr.flush()

    def fileno(self):
        return self._stderr.fileno()


sys.stderr = _PacketLossFilter(sys.stderr)

# ─────────────────────────────────────────────────────────────────────────────

import logging
import random
import asyncio

import discord
import aiohttp

import voice.tts
from core.config import DISCORD_TOKEN, LOG_LEVEL
from bot.handlers import handle_bot_mention
from music.player import voice_manager
from voice.listener import voice_listener_manager

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openwakeword").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

intents = discord.Intents.all()
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    logger.info(f"Logged in as {client.user} (ID: {client.user.id})")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.mention_everyone or client.user in message.mentions:
        await handle_bot_mention(message, client)


@client.event
async def on_voice_state_update(member, before, after):
    # When the BOT joins a voice channel — start listening automatically
    if member == client.user:
        if after.channel is not None and before.channel != after.channel:
            await asyncio.sleep(1.0)
            loop = asyncio.get_event_loop()
            await voice_listener_manager.start_listening(
                member.guild, after.channel, client, loop
            )
        elif after.channel is None:
            await voice_listener_manager.stop_listening(member.guild)
        return

    guild = member.guild
    bot_member = guild.me

    if not bot_member.voice or not bot_member.voice.channel:
        return

    bot_channel = bot_member.voice.channel

    if before.channel != bot_channel:
        return

    if len(bot_channel.members) == 1:
        logger.info(f"[voice] everyone left {bot_channel.name} in {guild.name}, disconnecting")
        player = voice_manager.get_player(guild)
        if player._now_playing_view:
            await player._now_playing_view.stop_updates()
            if player._now_playing_view.message:
                try:
                    await player._now_playing_view.message.delete()
                except Exception:
                    pass
                player._now_playing_view.message = None
        await player.disconnect()
        await voice_listener_manager.stop_listening(guild)


MAX_RETRIES = 10


def main():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            client.run(DISCORD_TOKEN, log_handler=None)
            logger.info("Bot exited cleanly.")
            break
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user.")
            break
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as e:
            retries += 1
            wait_time = (2 ** retries) + random.randint(0, 10)
            logger.critical(f"Network error: {e}. Retry {retries}/{MAX_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logger.critical(f"Unexpected error: {e}. Exiting.")
            break
    else:
        logger.critical(f"Exceeded {MAX_RETRIES} retries. Giving up.")


if __name__ == "__main__":
    main()