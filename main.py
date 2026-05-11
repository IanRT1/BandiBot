import logging
import os
import random
import time

import discord
import aiohttp
from dotenv import load_dotenv

load_dotenv()

from handlers import handle_bot_mention
from music import voice_manager

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set in .env")

# Logging setup — configures the root logger for the whole app.
# Other modules just do `logger = logging.getLogger(__name__)` and inherit this.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Silence library noise — we only want our own structured logs
logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Requires "Server Members Intent" and "Message Content Intent" (and "Presence
# Intent" if used) toggled ON in the Discord Developer Portal → Bot settings.
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
    # Ignore bot's own voice state changes
    if member == client.user:
        return

    guild = member.guild
    bot_member = guild.me

    # Bot isn't in a voice channel — nothing to do
    if not bot_member.voice or not bot_member.voice.channel:
        return

    bot_channel = bot_member.voice.channel

    # Only care if the member left the bot's channel
    if before.channel != bot_channel:
        return

    # If bot is now alone, clean up and disconnect
    if len(bot_channel.members) == 1:  # only the bot remains
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