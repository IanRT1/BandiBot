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
# Fixed grounded-search model for this deployment.
GEMINI_SEARCH_MODEL: str = "gemini-3.8-flash"

# ── Deepgram ──────────────────────────────────────────────────
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")

# ── TTS Providers ────────────────────────────────────────────
# Change TTS_PROVIDER and restart the bot to switch providers.
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "kokoro").lower()
if TTS_PROVIDER not in {"deepgram", "kokoro", "elevenlabs"}:
    raise RuntimeError("TTS_PROVIDER must be 'deepgram', 'kokoro', or 'elevenlabs'")

ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_MODEL: str = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# ── YouTube / yt-dlp ─────────────────────────────────────────
# Fixed runtime settings for this deployment's YouTube challenge solver.
YOUTUBE_JS_RUNTIME: str = "node"
YOUTUBE_REMOTE_COMPONENTS: str = "ejs:github"

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
# Message and response previews remain visible for operational transparency.
# Set to 0 if deployments should keep conversation contents out of logs.
LOG_SENSITIVE_CONTENT: bool = os.getenv("LOG_SENSITIVE_CONTENT", "1") == "1"
