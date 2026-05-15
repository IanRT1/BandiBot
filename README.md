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

BandiBot is a fully self-hosted Discord bot designed for friend group servers. It listens continuously in voice channels for a custom wake word, transcribes commands via Deepgram, reasons through them with an OpenAI LLM, and responds via text-to-speech — all while music plays uninterrupted in the background.

Every instance is independently hosted by the user. There is no central server, no shared infrastructure, and no data leaving your machine except to the APIs you configure (Discord, OpenAI, Deepgram).

---

## Features

### Voice Pipeline
- **Custom wake word detection** via a trained openWakeWord ONNX model
- **Per-user voice activity detection** using Silero VAD with configurable grace periods and silence thresholds
- **Speech-to-text** via Deepgram Nova-3
- **LLM reasoning** with full tool call support via OpenAI
- **Text-to-speech** via Deepgram Aura-2, mixed directly into the music stream at PCM level — music never pauses
- **Mid-speech interruption** — trigger the wake word while the bot is speaking to immediately cancel and start a new command
- **Per-user isolation** — only the user who triggered the wake word has their audio captured; other speakers are ignored

### Music
- **YouTube playback** via yt-dlp and FFmpeg with loudness normalization
- **Queue management** — add, remove, move, shuffle, loop
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
      │         └── handlers.py
      │                   ├── LLM (openai_utils.py)
      │                   └── Tool execution (_execute_tool_call)
      │                             ├── music.py (VoiceManager)
      │                             ├── voice_listener.py
      │                             └── handlers.py (server info, member activity)
      │
      └── on_voice_state_update
                └── voice_listener.py (GuildVoiceSession)
                          ├── BandiBotSink (audio receive)
                          │         ├── Wake word detection (openWakeWord)
                          │         └── VAD (Silero)
                          ├── stt.py (Deepgram Nova-3)
                          ├── voice_handler.py (LLM + tool calls)
                          └── tts.py (Deepgram Aura-2 → MixerSource)
                                    └── music.py (FFmpeg → MixerSource → Discord)
```

### Key Design Decisions

**TTS mixed into music at PCM level** — `MixerSource` wraps the FFmpeg audio source and injects TTS frames directly, ducking music volume during speech. No pausing, no restarting.

**Placeholder queue with one-ahead resolution** — bulk queued songs appear instantly in the queue. The background resolver pre-loads only the next track while the current one plays, then triggers the next resolution when the song changes. Respectful of YouTube's API and accurate to actual queue state.

**Interruption system across async and audio threads** — wake word detection runs on the audio thread. When it fires mid-TTS, `cancel_tts()` immediately clears the TTS buffer, and `_interrupt_current()` cancels the asyncio pipeline task. Music commands are exempt from cancellation and always run to completion.

**Per-user state machines** — each user in the voice channel has independent wake word detection, VAD state, and capture buffers. Packet loss or bad audio from one user does not affect others.

---

## Requirements

### System
- Python 3.11+
- FFmpeg installed and available in `PATH`

### Python Dependencies

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

### API Keys
- [Discord Developer Portal](https://discord.com/developers/applications) — Bot token
- [OpenAI Platform](https://platform.openai.com/) — API key
- [Deepgram Console](https://console.deepgram.com/) — API key (used for both STT and TTS)

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

**3. Create your `.env` file**

```bash
cp .env.example .env
```

Edit `.env` with your API keys (see [Configuration](#configuration)).

**4. Create a Discord bot**

- Go to the [Discord Developer Portal](https://discord.com/developers/applications)
- Create a new application and add a Bot
- Enable the following Privileged Gateway Intents:
  - `MESSAGE CONTENT INTENT`
  - `SERVER MEMBERS INTENT`
  - `PRESENCE INTENT`
- Copy the bot token into your `.env`
- Invite the bot to your server with the `bot` and `applications.commands` scopes and `Administrator` permissions

**5. Place your wake word model**

Put your `BandiBot.onnx` wake word model file in the root project directory. See [Wake Word Setup](#wake-word-setup) for how to train one.

**6. Run the bot**

```bash
python main.py
```

---

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

---

## Wake Word Setup

BandiBot uses a custom [openWakeWord](https://github.com/dscripka/openWakeWord) ONNX model for wake word detection. The model file must be named `BandiBot.onnx` and placed in the root project directory.

### Training a Custom Model

The easiest way to train a model is via the official Google Colab notebook:

👉 [openWakeWord Training Notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb)

Provide your wake word text (e.g. "BandiBot"), let it generate synthetic samples, train, and export the resulting `.onnx` file.

### Testing Your Model

Use the included `openwakeword_test.py` to test detection sensitivity before deploying:

```bash
python openwakeword_test.py
```

Adjust the following constants in `voice_listener.py` to tune detection:

| Constant | Default | Description |
|---|---|---|
| `WAKEWORD_THRESHOLD` | `0.01` | Minimum score per chunk to count as a hit |
| `SMOOTHING_WINDOW` | `3` | Number of recent chunks to evaluate |
| `HITS_REQUIRED` | `2` | Chunks above threshold needed to trigger |
| `WAKEWORD_COOLDOWN` | `2` | Seconds between triggers |

### Notes on Discord Audio

Discord's Opus codec introduces audio degradation compared to a direct microphone signal. Models trained on clean microphone audio will score lower on Discord-processed audio. For best results, record training samples through Discord itself using the `debug_capture.wav` output (enable in `voice_listener.py`) and include those in your training set.

---

## Project Structure

```
BandiBot/
├── main.py               # Entry point, Discord client, event routing
├── handlers.py           # Text command handling, LLM context, tool execution
├── voice_handler.py      # Voice command LLM bridge, _FakeMsgProxy
├── voice_listener.py     # Wake word, VAD, STT, TTS pipeline per guild
├── music.py              # Music queue, yt-dlp resolution, FFmpeg playback
├── now_playing_view.py   # Now Playing embed, button controls, live timer
├── banner.py             # Banner image generation (thumbnail + text overlay)
├── tts.py                # Deepgram TTS, MixerSource, audio mixing
├── stt.py                # Deepgram STT wrapper
├── openai_utils.py       # OpenAI SDK wrapper, tool definitions
├── utils.py              # Shared utilities, member presence, time formatting
├── instructions.txt      # Bot identity and behavior prompt (editable)
├── server_info.txt       # Server history and lore (editable)
├── BandiBot.onnx         # Wake word model (not included, train your own)
├── wake_activation.wav   # Wake word confirmation sound
├── requirements.txt      # Python dependencies
└── .env                  # API keys (not committed)
```

---

## Customization

### Bot Personality

Edit `instructions.txt` to change the bot's identity, language rules, tone, and behavior. The file is a plain text prompt loaded at startup. Variables `{bot_display_name}` and `{server_name}` are injected automatically.

### Server Lore

Edit `server_info.txt` to add your server's history, rules, events, member nicknames, and any other context you want the bot to know. This file is fetched on demand via the `get_server_info` tool when users ask server-specific questions.

### Wake Word

Replace `BandiBot.onnx` with any openWakeWord-compatible ONNX model. Update `WAKEWORD_MODEL_PATH` in `voice_listener.py` if you rename the file.

### TTS Voice

Change `TTS_MODEL` in `tts.py` to any [Deepgram Aura-2 voice](https://developers.deepgram.com/docs/tts-models). Default is `aura-2-javier-es` (Spanish male).

### STT Language

Change the `language` field in `stt.py` `_OPTIONS` to any [Deepgram-supported language code](https://developers.deepgram.com/docs/languages).

### Music Volume

Adjust `DEFAULT_VOLUME` in `music.py` (0.0–1.0) and `MUSIC_DUCK_VOLUME` in `tts.py` (volume multiplier applied to music while TTS is speaking).

### Hidden Voice Channels

Add voice channel IDs to `HIDDEN_VOICE_CHANNEL_IDS` in `utils.py` to prevent those channels and their members from ever being reported to the LLM.

---

