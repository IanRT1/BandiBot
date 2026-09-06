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
import warnings
from collections import deque
from pathlib import Path

# ── Must be set before any HuggingFace/Kokoro imports ────────────────────────
# Prevents Kokoro from hitting huggingface.co on every TTS call to check
# for voice file updates. Uses cached files only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# These are known startup warnings from optional ML dependencies. Keep other
# warnings visible so new compatibility problems are not hidden.
warnings.filterwarnings(
    "ignore",
    message=r"Defaulting repo_id to hexgrad/Kokoro-82M.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"dropout option adds dropout after all but last recurrent layer.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.nn\.utils\.weight_norm is deprecated.*",
    category=FutureWarning,
)

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
        if (
            "All log messages before absl::InitializeLog()" in msg
            or "oneDNN custom operations are on" in msg
            or "Defaulting repo_id to hexgrad/Kokoro-82M" in msg
            or "torch.nn.utils.weight_norm is deprecated" in msg
            or msg.strip() == "WeightNorm.apply(module, name, dim)"
        ):
            return
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

from dotenv import load_dotenv

load_dotenv()
console_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.setLevel(console_level)

session_log_path = Path(__file__).resolve().parents[1] / "logs" / "session.log"
session_log_path.parent.mkdir(parents=True, exist_ok=True)
session_handler = logging.FileHandler(session_log_path, mode="w", encoding="utf-8")
session_handler.setLevel(logging.DEBUG)
session_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
root_logger.addHandler(session_handler)

logger = logging.getLogger(__name__)
logger.info("[startup] initializing BandiBot")
from core.preflight import run_preflight
from bot.retrieval import warm_retrieval

if not run_preflight(warmup=warm_retrieval).ok:
    logger.critical("[startup] preflight failed; bot not started")
    raise SystemExit(1)

logging.getLogger("torio").setLevel(logging.WARNING)
logging.getLogger("torchaudio").setLevel(logging.WARNING)

import discord
import aiohttp

import voice.tts
from core.config import DISCORD_TOKEN
from bot.handlers import handle_bot_mention
from music.player import voice_manager
from voice.listener import voice_listener_manager

logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openwakeword").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

intents = discord.Intents.all()


class BandiBotClient(discord.Client):
    """Discord client with one idempotent application-shutdown cleanup path."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shutdown_started = False

    async def close(self):
        if not self._shutdown_started:
            self._shutdown_started = True
            logger.info("[shutdown] stopping voice listeners and music players")
            await voice_listener_manager.shutdown()
            await voice_manager.shutdown()
        await super().close()


client = BandiBotClient(intents=intents)


@client.event
async def on_ready():
    logger.info("[startup] logged in as %s", client.user)


_processed_messages: set[int] = set()

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.mention_everyone or client.user in message.mentions:
        if message.id in _processed_messages:
            return
        _processed_messages.add(message.id)
        await handle_bot_mention(message, client)
        _processed_messages.discard(message.id)


@client.event
async def on_voice_state_update(member, before, after):
    # When the BOT joins a voice channel — start listening automatically
    if member.id == client.user.id:
        if after.channel is not None and before.channel != after.channel:
            # join_voice() and music playback start the listener explicitly.
            # Discord emits this event for those joins as well, so do not
            # create a competing session while that startup is in progress.
            if voice_listener_manager.get_session(member.guild) is not None:
                return
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
    logger.info("[startup] bot initialized; connecting to Discord")
    while retries < MAX_RETRIES:
        try:
            if retries:
                logger.info("[startup] retrying connection (attempt %d)", retries + 1)
            client.run(DISCORD_TOKEN, log_handler=None)
            logger.info("[startup] bot exited cleanly")
            break
        except KeyboardInterrupt:
            logger.info("[startup] shutdown requested by user")
            break
        except discord.LoginFailure as e:
            logger.critical("[startup] Discord authentication failed: %s", e)
            break
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            discord.GatewayNotFound,
            discord.ConnectionClosed,
        ) as e:
            retries += 1
            wait_time = min(60, (2 ** retries) + random.randint(0, 10))
            logger.critical(
                "[startup] network error: %s; retry %d/%d in %ds",
                e, retries, MAX_RETRIES, wait_time,
            )
            time.sleep(wait_time)
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logger.exception("[startup] unexpected error after %d attempts; exiting", retries)
                break
            wait_time = min(60, 2 ** retries)
            logger.exception(
                "[startup] unexpected recoverable error; retry %d/%d in %ds",
                retries, MAX_RETRIES, wait_time,
            )
            time.sleep(wait_time)
    else:
        logger.critical("[startup] exceeded %d retries; giving up", MAX_RETRIES)


if __name__ == "__main__":
    main()
