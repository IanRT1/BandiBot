# BandiBot

> A self-hosted Discord AI assistant with continuous voice listening, wake-word detection, real-time speech recognition, LLM tool calling, Google Search integration, multi-provider text-to-speech with automatic fallback, TTS mixed over music, and a full YouTube music queue system — running on your own machine.

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
- [Testing](#testing)
- [License](#license)

---

## Overview

BandiBot is a fully self-hosted Discord bot designed for friend group servers. It listens continuously in voice channels for a custom wake word, transcribes commands via Deepgram, reasons through them with an OpenAI LLM, and responds via the configured text-to-speech provider — all while music plays uninterrupted in the background.

Every instance is independently hosted by the user. There is no central server, no shared infrastructure, and no data leaving your machine except to the APIs you configure (Discord, OpenAI, Deepgram, Gemini, and optionally ElevenLabs).

---

## Features

### Voice Pipeline
- **Custom wake word detection** via a trained openWakeWord ONNX model
- **Per-user voice activity detection** using Silero VAD with configurable grace periods and silence thresholds
- **Speech-to-text** via Deepgram Nova-3
- **LLM reasoning** with full tool call support via OpenAI
- **Text-to-speech** via Kokoro, Deepgram Aura-2, or ElevenLabs; speech is mixed directly into the music stream at PCM level so music never pauses
- **Automatic TTS fallback** — if a remote provider fails before producing audio, the response automatically falls back to local Kokoro
- **Mid-speech interruption** — trigger the wake word while the bot is speaking to immediately cancel and start a new command
- **Per-user isolation** — only the user who triggered the wake word has their audio captured; other speakers are ignored
- **Voice music feedback** — voice-detected song requests post the heard query first, then edit it when queued or remove it when playback starts and the Now Playing UI takes over
- **Minimal voice acknowledgements** — song requests use a short English or Spanish playing/queued confirmation instead of narrating the search process
- **Timeout recovery** — STT, voice-command, and TTS stages have independent timeouts and reset processing state when a provider hangs
- **Clean voice lifecycle** — failed or timed-out commands release their pipeline state so later wake-word requests are not permanently blocked; listener start/stop operations are serialized to prevent duplicate voice sessions
- **Voice disconnect recovery** — a sustained voice disconnect gets a 15-second grace period, then the stale client is replaced with up to three bounded reconnect attempts while preserving the active track and queue

### Music
- **YouTube playback** via yt-dlp and FFmpeg with loudness normalization
- **Queue management** — add, remove, move, skip, pause, resume, and stop through natural language, with loop and shuffle available from the Now Playing controls
- **Audio attachments** — attached audio files can be queued directly, with metadata and embedded cover art extracted when available
- **Voice clips** — save the last 30 seconds of voice-channel audio as an MP3 clip
- **Single song queuing** — resolved on the spot before playing, with unrequested music videos penalized in favor of audio, studio, and remaster results
- **Bulk song queuing** — multiple songs or YouTube playlist URLs queued instantly as placeholders, resolved one track ahead in the background as songs play
- **Graceful error handling** — unresolvable tracks are skipped with a notification at playback time
- **Playback race recovery** — tracks that finish resolving while activation/TTS audio is playing wait safely instead of causing Discord's `Already playing audio` error; failed starts restore the track to the queue
- **Stream URL refresh** — stale signed YouTube stream URLs are refreshed from the existing video URL before playback, avoiding a second search/ranking pass; playback failures can still search for an alternative result
- **Now Playing embed** with live timer, album art banner, next track preview, and interactive button controls (previous, play/pause, skip, stop, queue, loop, shuffle, copy link)

### Text Commands
- Full conversational LLM responses via @mention
- Music control via natural language
- Google Search-grounded answers for current web questions
- Local RAG retrieval for server lore, avoiding an extra LLM tool-call round trip when relevant context is found
- Server member activity and presence lookup
- Private server history and lore via local `server_info.txt`
- Recent channel history included in every request for conversational continuity

### Reliability and Privacy
- **Structured runtime logs** with separate startup, chat, voice, STT, TTS, tool, and RAG events
- **Two-session diagnostics** — `logs/session.log` captures the current run and `logs/session.previous.log` preserves the immediately preceding run; both always capture `DEBUG` detail locally
- **Selective tool routing** — obvious music, voice, web, member-activity, lore, and casual requests receive only the smallest safe tool set; ambiguous requests retain the full fallback set
- **Transparent chat logs** — message and response previews are shown by default; set `LOG_SENSITIVE_CONTENT=0` to hide conversation contents
- **Quiet local embeddings** — casual or very short messages without a lore/name signal skip semantic retrieval, and Sentence Transformers progress output is disabled
- **Startup preflight** — required configuration, runtime dependencies, voice assets, and context files are validated before connecting to Discord
- **Graceful shutdown and recovery** — active voice/music sessions are cleaned up on exit; transient connection failures retry with bounded backoff, while invalid credentials stop immediately
- **Connection-aware logging** — bursts of Discord voice crypto/decryption packet errors produce one aggregated warning instead of flooding the log
- **Quiet startup logging** — normal connection attempts are implicit; retry attempts are logged only after a connection failure
- **Single-instance guard** — a process lock prevents multiple BandiBot instances from sharing one Discord token and invalidating each other's voice sessions
- **Private deployment context** — personal instructions and server lore stay local and are excluded from Git
- **Generic tracked templates** — new deployments can use safe `*.example.txt` files without exposing server-specific information
- **Offline automated tests** for RAG, music intent routing, voice timeout recovery, TTS fallback, tool errors, voice lifecycle cleanup, acknowledgement language, and repository hygiene

---

## Architecture

```
Discord Gateway
      │
      ├── on_message (@mention)
      │         └── bot/handlers.py
      │                   ├── LLM (bot/openai_client.py)
      │                   ├── Local lore RAG (bot/retrieval.py)
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
                                     ├── voice/tts_providers.py (Kokoro / Deepgram / ElevenLabs registry)
                                     ├── voice/tts_sources.py (MixerSource / StandaloneSource)
                                     └── music/player.py (FFmpeg → MixerSource → Discord)
```

### Key Design Decisions

**TTS mixed into music at PCM level** — `MixerSource` wraps the FFmpeg audio source and injects TTS frames directly, ducking music volume during speech. No pausing, no restarting.

**Placeholder queue with one-ahead resolution** — bulk queued songs appear instantly in the queue. The background resolver pre-loads only the next track while the current one plays, then triggers the next resolution when the song changes. Respectful of YouTube's API and accurate to actual queue state.

**Interruption system across async and audio threads** — wake word detection runs on the audio thread. When it fires mid-TTS, `cancel_tts()` immediately clears the TTS buffer, and `_interrupt_current()` cancels the asyncio pipeline task. Music commands are exempt from cancellation and always run to completion.

**Per-user state machines** — each user in the voice channel has independent wake word detection, VAD state, and capture buffers. Packet loss or bad audio from one user does not affect others.

**Music starts voice listening** — when a text command makes the bot join voice for music playback, `music/player.py` explicitly starts the wake-word listener on the same voice client. The Discord voice-state event remains a fallback for joins that did not originate in the command path, while the listener manager serializes lifecycle operations and avoids competing sessions.

**Playback state recovery** — resolver callbacks re-check whether Discord is
already playing standalone activation/TTS audio. If a playback start is
rejected, the track is restored to the queue and phantom `current` state is
cleared so later music commands continue normally.

**Private deployment context** — personal instructions and server lore are ignored by Git. Generic `.example.txt` templates are tracked and used automatically when the private files are absent.

**Hybrid local RAG for server lore** — server-lore files are split into sections
and searched locally before the LLM request. BM25/fuzzy matching protects
server-specific names and speech-to-text corrections, while a cached
multilingual embedding model recovers paraphrased questions. Document-chunk
embeddings are precomputed and reused, so each query only embeds the question.
The top three matches are ranked together and capped before being added to the
prompt. Strong matches omit the redundant `get_server_info` tool so the answer
uses one LLM call; weak matches keep the tool available as a fallback. The
embedding model is loaded lazily and the system degrades to BM25 retrieval if
it is unavailable. Parent section labels, conservative singular/plural
matching, corpus-derived BM25 weighting, and evidence requirements for names,
headings, or multiple terms improve broad questions without hardcoded language
lists or private server names. Live Discord facts such as server creation date
remain structured context.

The retriever normalizes accents and tolerates conservative speech-to-text
spelling variations for names. If no lore matches, the full private file is
not sent as a fallback; the bot reports that the fact is not documented instead
of guessing.

Short follow-up questions inherit the latest user question for retrieval, so a
sequence such as "Who created BandiBot?" followed by "When?" can retrieve the
same focused lore section without storing permanent conversation memory.

**Shared tool execution** — text and voice commands use the same tool executor.
Music intent rules distinguish skipping the currently playing song from deleting
an upcoming queue item, including Spanish phrasing such as `quitar la canción`.

**Voice receive recovery** — bursts of Discord crypto/decryption failures are
grouped into one warning and trigger a receive-sink restart, while unrelated
voice processing errors remain visible individually.

**Voice connection recovery** — the listener watchdog waits through short-lived
Discord reconnects, then force-closes a stale voice client and retries a fresh
connection with bounded exponential backoff. The music player detaches from the
stale client, restores the interrupted track to the front of the queue, and
resumes playback on the replacement client. This is separate from the main
Discord gateway retry loop.

**Structured music outcomes** — `music/results.py` represents play success,
queueing, startup, and failure as structured data. Text and voice confirmations
therefore use the actual playback result instead of parsing display strings.

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
- [Deepgram Console](https://console.deepgram.com/) — API key used for STT and optional Deepgram TTS
- [Google AI Studio](https://aistudio.google.com/app/apikey) — API key for Google Search grounding
- [ElevenLabs](https://elevenlabs.io/app/settings/api-keys) — optional API key for ElevenLabs TTS

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

The default provider is Kokoro. Kokoro requires `espeak-ng` as a system dependency. If you select ElevenLabs or Deepgram, the corresponding API key is required for remote synthesis; Kokoro remains the automatic fallback.

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

The `bandibot` console command is provided by the editable package install.
Stop any older BandiBot process before restarting so two Discord voice sessions
cannot compete for the same bot account.

---

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5.6-luna
DEEPGRAM_API_KEY=your_deepgram_api_key
GEMINI_API_KEY=your_gemini_api_key

# Optional remote TTS provider
TTS_PROVIDER=kokoro
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL=eleven_multilingual_v2

# Logging: INFO is the clean default; use DEBUG for timing/RAG diagnostics.
LOG_LEVEL=INFO
# Set to 0 to hide message/response contents.
LOG_SENSITIVE_CONTENT=1
```

`TTS_PROVIDER` supports `kokoro`, `deepgram`, and `elevenlabs`. Provider changes take effect after restarting the bot. `GEMINI_SEARCH_MODEL` is fixed in `core/config.py` and is not a secret.

The voice assistant currently constrains generated spoken acknowledgements to
English or Spanish based on the recognized command. STT itself defaults to
Deepgram's multilingual `multi` mode and can handle code-switching.

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
│   ├── config.py           # Centralized environment variable loading
│   ├── instance_lock.py     # Single-process runtime guard
│   ├── interaction_logging.py # Interaction timing, privacy, and usage logs
│   ├── session_logs.py       # Two-file session log rotation
│   └── preflight.py         # Startup dependency and asset validation
│
├── voice/
│   ├── listener.py         # Wake word, VAD, STT, TTS pipeline per guild
│   ├── handler.py          # Voice command LLM bridge, _FakeMsgProxy
│   ├── audio.py            # Stateless PCM conversion helpers
│   ├── clips.py            # Last-30-seconds voice clip export
│   ├── stt.py              # Deepgram STT wrapper
│   ├── tts.py              # TTS orchestration, cancellation, activation sound
│   ├── tts_providers.py    # TTS provider interface, adapters, registry, fallback support
│   └── tts_sources.py      # MixerSource and StandaloneSource audio sources
│
├── music/
│   ├── player.py           # Music queue, guild playback state, FFmpeg playback
│   ├── resolver.py         # yt-dlp search, URL resolution, playlist extraction
│   ├── tracks.py           # Shared Track model
│   ├── results.py           # Structured play/queue/failure outcomes
│   ├── attachments.py      # Uploaded audio ingestion and metadata extraction
│   ├── now_playing.py      # Now Playing embed, button controls, live timer
│   └── banner.py           # Banner image generation (thumbnail + text overlay)
│
├── bot/
│   ├── handlers.py         # Text command handling, LLM context, replies
│   ├── google_search.py    # Gemini Google Search grounding adapter
│   ├── retrieval.py        # Local hybrid RAG chunking and retrieval
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
│   ├── instructions.example.txt  # Generic tracked prompt template
│   └── server_info.example.txt   # Generic tracked server-context template
│
├── tests/
│   ├── test_retrieval.py           # Local RAG and lore fallback tests
│   ├── test_music_tools.py         # Music intent and tool safety tests
│   ├── test_music_player.py        # Playback race and queue restoration tests
│   ├── test_interaction_logging.py # Privacy and usage logging tests
│   ├── test_instance_lock.py         # Single-process guard tests
│   ├── test_preflight.py            # Startup validation tests
│   ├── test_runtime_shutdown.py     # Voice/music cleanup tests
│   ├── test_search_acknowledgement.py # Voice search/playing acknowledgements
│   ├── test_session_logs.py          # Current/previous log rotation tests
│   ├── test_tts.py                 # TTS providers, conversion, and fallback
│   ├── test_voice_timeouts.py      # STT/LLM timeout state recovery
│   └── test_repository_hygiene.py  # Private files and cache ignore rules
│
├── __main__.py             # Direct module entry point
├── pyproject.toml          # Package config, bandibot CLI entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .env                    # Secrets/API keys (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Testing

The test suite is offline and mocks provider behavior; it does not contact
Discord, Gemini, Deepgram, or ElevenLabs and does not spend API credits.

```powershell
python -m pytest tests -q
```

The tests cover:

- Local RAG relevance, accent normalization, STT name variations, and safe
  unmatched-query fallback
- Music routing between skipping the active song and deleting queued songs
- Queue/tool safety, including blocked stop requests, unknown tools, and
  contained tool exceptions
- STT and LLM timeout recovery so voice users are not left permanently locked
- TTS provider registration, PCM conversion, error parsing, automatic
  ElevenLabs-to-Kokoro fallback, and prevention of repeated speech after a
  partial provider stream
- Runtime voice/music shutdown cleanup, listener lifecycle serialization, and
  failure isolation
- Voice search acknowledgement overlap, English/Spanish routing, and
  cancellation behavior
- Private context separation and ignored test-cache directories

The suite currently contains 110 tests. The only expected warning is Python's
`audioop` deprecation warning from the Discord dependency.

For a local syntax check, compile the edited modules with:

```powershell
python -m py_compile core/client.py music/player.py voice/handler.py voice/listener.py
```

The test suite sets placeholder API keys and disables semantic embeddings, so
it does not require production secrets, private context files, network access,
or model downloads.

---

## Customization

### Bot Personality

Copy `data/instructions.example.txt` to `data/instructions.txt` and edit the private file to change the bot's identity, language rules, tone, and behavior. The private file is ignored by Git; the generic example is used if it is absent. Variables `{bot_display_name}` and `{server_name}` are injected automatically.

### Server Lore

Copy `data/server_info.example.txt` to `data/server_info.txt` and edit the private file to add your server's history, rules, events, member nicknames, and other context. This file is ignored by Git and retrieved locally as relevant context before the LLM request. Keep lore organized under Markdown headings so retrieval can return focused sections. The `get_server_info` tool remains available as a controlled fallback when local retrieval does not answer a request.

### Wake Word

Replace `assets/BandiBot.onnx` with any openWakeWord-compatible ONNX model. Update `WAKEWORD_MODEL_PATH` in `voice/listener.py` if you rename the file.

### TTS Provider And Voice

The provider is selected by `TTS_PROVIDER` in `.env` and takes effect after a
restart. Supported providers are `kokoro` (default), `deepgram`, and
`elevenlabs`. All providers expose the same PCM streaming interface, so the
Discord playback and music-mixing code is provider-independent.

For Kokoro, change `KOKORO_VOICE`, `KOKORO_LANG`, and `KOKORO_SPEED` in `voice/tts_providers.py`.

For Deepgram, change `DEEPGRAM_MODEL` in `voice/tts_providers.py` to any [Deepgram Aura-2 voice](https://developers.deepgram.com/docs/tts-models). The current model is `aura-2-javier-es` (Spanish male).

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
