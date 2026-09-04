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

# Optional web-search answer provider. The bot remains usable without it and
# reports a clear tool error when web search is requested without a key.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_SEARCH_MODEL: str = os.getenv("GEMINI_SEARCH_MODEL", "gemini-3.8-flash")

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
# Fixed runtime settings for this deployment's YouTube challenge solver.
YOUTUBE_JS_RUNTIME: str = "node"
YOUTUBE_REMOTE_COMPONENTS: str = "ejs:github"

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG").upper()
