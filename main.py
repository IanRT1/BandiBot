import discord
import openai
import random
import aiohttp
import time
from config import OPENAI_API_KEY, DISCORD_TOKEN
from logger import logger
from handlers import handle_bot_mention
from openai_utils import categorize_message

# Initialize Discord client with specified intents
intents = discord.Intents.all()
client = discord.Client(intents=intents)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Event handler for when the bot is ready
@client.event
async def on_ready():
    pass

# Event handler for incoming messages
@client.event
async def on_message(message):
    # Avoid processing messages sent by the bot itself
    if message.author == client.user:
        return

    # Handle mentions of the bot
    if message.mention_everyone or client.user in message.mentions:
        categories = await categorize_message(message.content)
        await handle_bot_mention(
            message, categories, client
        )  # passing client as a parameter

# Start the bot
MAX_RETRIES = 10

if __name__ == "__main__":
    retries = 0
    while retries < MAX_RETRIES:
        try:
            client.run(DISCORD_TOKEN)
            # If successful, reset retries
            retries = 0
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as e:
            logger.critical(f"Network error: {e}.")
            retries += 1
            wait_time = (2**retries) + random.randint(
                0, 10
            )  # Exponential backoff with jitter
            logger.critical(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            logger.critical(f"Unexpected error: {e}. Exiting.")
            break
