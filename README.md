# BandiBot

> A self-hosted Discord bot with continuous voice listening, wake word detection, real-time speech-to-text, LLM reasoning with tool calls, TTS mixed over music, and a full music queue system — all running locally on your own machine.

<br>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Wake Word Setup](#wake-word-setup)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [License](#license)

---

## Overview

BandiBot is a fully self-hosted Discord bot designed for friend group servers. It listens continuously in voice channels for a custom wake word, transcribes commands via Deepgram, reasons through them with an OpenAI LLM, and responds via the configured text-to-speech provider — all while music plays uninterrupted in the background.

Every instance is independently hosted by the user. There is no central server, no shared infrastructure, and no data leaving your machine except to the APIs you configure (Discord, OpenAI, Deepgram).

---

## Features

### Voice Pipeline
- **Custom wake word detection** via a trained openWakeWord ONNX model
- **Per-user voice activity detection** using Silero VAD with configurable grace periods and silence thresholds
- **Speech-to-text** via Deepgram Nova-3
- **LLM reasoning** with full tool call support via OpenAI
- **Text-to-speech** via Kokoro by default, with a Deepgram Aura-2 path available in code; speech is mixed directly into the music stream at PCM level so music never pauses
- **Mid-speech interruption** — trigger the wake word while the bot is speaking to immediately cancel and start a new command
- **Per-user isolation** — only the user who triggered the wake word has their audio captured; other speakers are ignored
- **Voice music feedback** — voice-detected song requests post the heard query first, then replace that same message with the final queued or now-playing result

### Music
- **YouTube playback** via yt-dlp and FFmpeg with loudness normalization
- **Queue management** — add, remove, move, skip, pause, resume, and stop through natural language, with loop and shuffle available from the Now Playing controls
- **Audio attachments** — attached audio files can be queued directly, with metadata and embedded cover art extracted when available
- **Voice clips** — save the last 30 seconds of voice-channel audio as an MP3 clip
- **Single song queuing** — resolved on the spot before playing
- **Bulk song queuing** — multiple songs or YouTube playlist URLs queued instantly as placeholders, resolved one track ahead in the background as songs play
- **Graceful error handling** — unresolvable tracks are skipped with a notification at playback time
- **Now Playing embed** with live timer, album art banner, next track preview, and interactive button controls (previous, play/pause, skip, stop, queue, loop, shuffle, copy link)

### Text Commands
- Full conversational LLM responses via @mention
- Music control via natural language
- Server member activity and presence lookup
- Server history and lore via configurable `server_info.txt`
- Recent channel history included in every request for conversational continuity

---

## Architecture

```
Discord Gateway
      │
      ├── on_message (@mention)
      │         └── bot/handlers.py
      │                   ├── LLM (bot/openai_client.py)
      │                   ├── Tool schemas (bot/tool_schemas.py)
      │                   └── Tool execution (bot/tool_executor.py)
      │                             ├── music/player.py (VoiceManager)
      │                             ├── voice/listener.py / voice/clips.py
      │                             └── bot/handlers.py + bot/utils.py (server context)
      │
      └── on_voice_state_update
                └── voice/listener.py (GuildVoiceSession)
                          ├── BandiBotSink (audio receive)
                          │         ├── Wake word detection (openWakeWord)
                          │         └── VAD (Silero)
                          ├── voice/stt.py (Deepgram Nova-3)
                          ├── voice/handler.py (LLM + tool calls)
                          └── voice/tts.py (TTS orchestration)
                                    ├── voice/tts_providers.py (Kokoro / Deepgram Aura-2)
                                    ├── voice/tts_sources.py (MixerSource / StandaloneSource)
                                    └── music/player.py (FFmpeg → MixerSource → Discord)
```

### Key Design Decisions

**TTS mixed into music at PCM level** — `MixerSource` wraps the FFmpeg audio source and injects TTS frames directly, ducking music volume during speech. No pausing, no restarting.

**Placeholder queue with one-ahead resolution** — bulk queued songs appear instantly in the queue. The background resolver pre-loads only the next track while the current one plays, then triggers the next resolution when the song changes. Respectful of YouTube's API and accurate to actual queue state.

**Interruption system across async and audio threads** — wake word detection runs on the audio thread. When it fires mid-TTS, `cancel_tts()` immediately clears the TTS buffer, and `_interrupt_current()` cancels the asyncio pipeline task. Music commands are exempt from cancellation and always run to completion.

**Per-user state machines** — each user in the voice channel has independent wake word detection, VAD state, and capture buffers. Packet loss or bad audio from one user does not affect others.

**Music starts voice listening** — when a text command makes the bot join voice for music playback, `music/player.py` explicitly starts the wake-word listener on the same voice client. The Discord voice-state event also starts listening, and the listener manager de-duplicates repeated starts.

---

## Requirements

### System
- Python 3.11+
- FFmpeg installed and available in `PATH`
- espeak-ng installed and available in `PATH` when using the default Kokoro TTS provider

### Python Dependencies

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

### API Keys
- [Discord Developer Portal](https://discord.com/developers/applications) — Bot token
- [OpenAI Platform](https://platform.openai.com/) — API key
- [Deepgram Console](https://console.deepgram.com/) — API key used for STT, and also for TTS if the Deepgram TTS path is selected in code

### FFmpeg

FFmpeg is required for audio decoding and loudness normalization. It is **not** installable via pip.

**Windows:**
```bash
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

### espeak-ng

The current code defaults to Kokoro TTS in `core/config.py`. Kokoro requires `espeak-ng` as a system dependency.

Verify installation:
```bash
espeak-ng --version
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/IanRT1/BandiBot.git
cd BandiBot
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Install the package**

```bash
pip install -e .
```

**4. Create your `.env` file**

```bash
cp .env.example .env
```

Edit `.env` with your API keys (see [Configuration](#configuration)).

**5. Create a Discord bot**

- Go to the [Discord Developer Portal](https://discord.com/developers/applications)
- Create a new application and add a Bot
- Enable the following Privileged Gateway Intents:
  - `MESSAGE CONTENT INTENT`
  - `SERVER MEMBERS INTENT`
  - `PRESENCE INTENT`
- Copy the bot token into your `.env`
- Invite the bot to your server with the `bot` scope and the permissions needed for message reading, voice connect/speak, embeds, attachments, and message management. `Administrator` is the simplest private-server setup, but not strictly required.

**6. Place your wake word model**

Put your `BandiBot.onnx` wake word model file in the `assets/` directory. See [Wake Word Setup](#wake-word-setup) for how to train one.

**7. Run the bot**

```bash
bandibot
```

---

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

YouTube challenge solving is configured centrally in `core/config.py` with Node
and the `ejs:github` remote component source. Node 22+ must be installed and
available on PATH.

If YouTube starts returning playable-search results but every stream fails with
`HTTP Error 403: Forbidden`, update yt-dlp to the current pre-release/nightly:

```powershell
python -m pip install -U --pre "yt-dlp[default]"
```

---

## Wake Word Setup

BandiBot uses a custom [openWakeWord](https://github.com/dscripka/openWakeWord) ONNX model for wake word detection. The model file must be named `BandiBot.onnx` and placed in the `assets/` directory.

### Training a Custom Model

The easiest way to train a model is via the official Google Colab notebook:

👉 [openWakeWord Training Notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb)

Provide your wake word text (e.g. "BandiBot"), let it generate synthetic samples, train, and export the resulting `.onnx` file.

### Tuning Detection

Adjust the following constants in `voice/listener.py` to tune detection:

| Constant | Default | Description |
|---|---|---|
| `WAKEWORD_THRESHOLD` | `0.05` | Minimum score per chunk to count as a hit |
| `SMOOTHING_WINDOW` | `3` | Number of recent chunks to evaluate |
| `HITS_REQUIRED` | `2` | Chunks above threshold needed to trigger |
| `WAKEWORD_COOLDOWN` | `2` | Seconds between triggers |

### Notes on Discord Audio

Discord's Opus codec introduces audio degradation compared to a direct microphone signal. Models trained on clean microphone audio will score lower on Discord-processed audio. For best results, record training samples through Discord itself using the `debug_capture.wav` output (enable in `voice/listener.py`) and include those in your training set.

---

## Project Structure

```
BandiBot/
├── core/
│   ├── client.py           # Discord client, event routing, reconnection logic
│   └── config.py           # Centralized environment variable loading
│
├── voice/
│   ├── listener.py         # Wake word, VAD, STT, TTS pipeline per guild
│   ├── handler.py          # Voice command LLM bridge, _FakeMsgProxy
│   ├── audio.py            # Stateless PCM conversion helpers
│   ├── clips.py            # Last-30-seconds voice clip export
│   ├── stt.py              # Deepgram STT wrapper
│   ├── tts.py              # TTS orchestration, cancellation, activation sound
│   ├── tts_providers.py    # Kokoro and Deepgram TTS provider adapters
│   └── tts_sources.py      # MixerSource and StandaloneSource audio sources
│
├── music/
│   ├── player.py           # Music queue, guild playback state, FFmpeg playback
│   ├── resolver.py         # yt-dlp search, URL resolution, playlist extraction
│   ├── tracks.py           # Shared Track model
│   ├── attachments.py      # Uploaded audio ingestion and metadata extraction
│   ├── now_playing.py      # Now Playing embed, button controls, live timer
│   └── banner.py           # Banner image generation (thumbnail + text overlay)
│
├── bot/
│   ├── handlers.py         # Text command handling, LLM context, replies
│   ├── openai_client.py    # OpenAI SDK wrapper
│   ├── tool_schemas.py     # OpenAI tool definitions
│   ├── tool_executor.py    # Shared text/voice tool execution
│   └── utils.py            # Shared utilities, member presence, time formatting
│
├── assets/
│   ├── BandiBot.onnx       # Wake word model expected by voice/listener.py
│   └── wake_activation.wav # Wake word confirmation sound
│
├── data/
│   ├── instructions.txt    # Bot identity and behavior prompt (editable)
│   └── server_info.txt     # Server history and lore (editable)
│
├── __main__.py             # Package entry point
├── pyproject.toml          # Package config, bandibot CLI entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .env                    # Secrets/API keys (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Customization

### Bot Personality

Edit `data/instructions.txt` to change the bot's identity, language rules, tone, and behavior. The file is a plain text prompt loaded at startup. Variables `{bot_display_name}` and `{server_name}` are injected automatically.

### Server Lore

Edit `data/server_info.txt` to add your server's history, rules, events, member nicknames, and any other context you want the bot to know. This file is fetched on demand via the `get_server_info` tool when users ask server-specific questions.

### Wake Word

Replace `assets/BandiBot.onnx` with any openWakeWord-compatible ONNX model. Update `WAKEWORD_MODEL_PATH` in `voice/listener.py` if you rename the file.

### TTS Provider And Voice

The provider is selected by `TTS_PROVIDER` in `.env` and takes effect after a
restart. Supported providers are `kokoro` (default), `deepgram`, and
`elevenlabs`. All providers expose the same PCM streaming interface, so the
Discord playback and music-mixing code is provider-independent.

For Kokoro, change `KOKORO_VOICE`, `KOKORO_LANG`, and `KOKORO_SPEED` in `voice/tts_providers.py`.

For Deepgram, change `TTS_MODEL` in `voice/tts_providers.py` to any [Deepgram Aura-2 voice](https://developers.deepgram.com/docs/tts-models). The current Deepgram model constant is `aura-2-javier-es` (Spanish male).

For ElevenLabs, set `TTS_PROVIDER=elevenlabs`, provide `ELEVENLABS_API_KEY`,
and optionally change `ELEVENLABS_VOICE_ID` or `ELEVENLABS_MODEL` in `.env`.
ElevenLabs audio is streamed as 24 kHz PCM and converted to Discord's 48 kHz
PCM format automatically.

Low-level Discord audio buffering lives in `voice/tts_sources.py`. Provider output conversion, including Kokoro's 24kHz float32 to 48kHz int16 PCM conversion, routes through helpers in `voice/audio.py`.

### STT Language

The default STT language is `multi` in `voice/stt.py`, which supports English/Spanish code-switching better than pinning recognition to only `es`. Change `STT_LANGUAGE` to any [Deepgram-supported language code](https://developers.deepgram.com/docs/languages) if you want to force one language.

### Music Volume

Adjust `DEFAULT_VOLUME` in `music/player.py` (0.0–1.0) and `MUSIC_DUCK_VOLUME` in `voice/tts_sources.py` (volume multiplier applied to music while TTS is speaking).

### Hidden Voice Channels

Add voice channel IDs to `HIDDEN_VOICE_CHANNEL_IDS` in `bot/utils.py` to prevent those channels and their members from ever being reported to the LLM.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
