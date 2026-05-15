"""
main.py

Entry point for BandiBot — Discord voice and chat assistant.

Manages the Discord client lifecycle, routes incoming messages and voice
state changes to the appropriate handlers, and implements exponential
backoff reconnection logic for network-level failures.

Architecture overview:
  Text commands  → handle_bot_mention (handlers.py)
  Voice commands → voice_listener_manager (voice_listener.py)
  Music playback → voice_manager (music.py)
  TTS / mixing   → speak / MixerSource (tts.py)
  STT            → transcribe (stt.py)

The bot responds to @mentions in text channels and to a custom wake word
in voice channels. Voice interactions run a full wake word → VAD → STT →
LLM → TTS pipeline with interruption support and mid-speech cancellation.
"""
import logging
import os
import random
import time
import asyncio

import discord
import aiohttp
from dotenv import load_dotenv

load_dotenv()

from handlers import handle_bot_mention
from music import voice_manager
from voice_listener import voice_listener_manager

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set in .env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openwakeword").setLevel(logging.ERROR)

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

if __name__ == "__main__":
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