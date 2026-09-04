"""
core/config.py

Centralized environment configuration for BandiBot.

All environment variables are loaded here once at startup.
Import from this module instead of calling os.getenv() directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Discord ───────────────────────────────────────────────────
DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set in .env")

# ── OpenAI ────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

# ── Deepgram ──────────────────────────────────────────────────
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")

# ── TTS Provider ──────────────────────────────────────────────
# Code-level operational choice, not a secret.
# Options: "deepgram" | "kokoro"
TTS_PROVIDER: str = "kokoro"
if TTS_PROVIDER not in {"deepgram", "kokoro"}:
    raise RuntimeError("TTS_PROVIDER must be either 'deepgram' or 'kokoro'")

# ── YouTube / yt-dlp ─────────────────────────────────────────
# Optional JavaScript challenge solver settings for modern YouTube extraction.
YOUTUBE_JS_RUNTIME: str = os.getenv("YOUTUBE_JS_RUNTIME", "").strip()
YOUTUBE_REMOTE_COMPONENTS: str = os.getenv("YOUTUBE_REMOTE_COMPONENTS", "").strip()

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG").upper()
