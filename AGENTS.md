# Repository Guidelines

## Project Structure & Module Organization

BandiBot is a Python 3.11+ Discord bot organized by runtime concern. `core/` contains startup, client wiring, and configuration. `bot/` handles text commands, OpenAI tool schemas, tool execution, and Discord utility helpers. `voice/` owns wake-word detection, VAD, STT, TTS, audio mixing, and voice clips. `music/` owns queue state, yt-dlp resolution, playback, attachments, and now-playing UI. Static runtime files live in `assets/`, including `BandiBot.onnx` and `wake_activation.wav`. Editable prompt/context files live in `data/`. The package entry points are `__main__.py` and the `bandibot` console script defined in `pyproject.toml`.

## Build, Test, and Development Commands

- `python -m venv .venv` creates a local virtual environment.
- `pip install -r requirements.txt` installs runtime dependencies.
- `pip install -e .` installs the package and exposes the `bandibot` command.
- `bandibot` runs the bot through `core.client:main`.
- `python -m bandibot` runs the package entry point when installed/importable.

System dependencies are not installed by pip. Ensure `ffmpeg` is on `PATH`; install `espeak-ng` when using the default Kokoro TTS provider.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, type hints where they clarify async boundaries or shared models, and descriptive snake_case names for modules, functions, and variables. Keep Discord event orchestration in `core/` or `bot/`; keep audio pipeline logic in `voice/`; keep playback and queue behavior in `music/`. Prefer small async helpers over large event handlers, and keep environment reads centralized through `core/config.py`.

## Testing Guidelines

There is no committed test suite yet. For changes, at minimum run `python -m py_compile` on edited Python files or the touched package directories. When adding tests, place them under `tests/`, use `pytest`, name files `test_*.py`, and mock Discord, OpenAI, Deepgram, and yt-dlp calls rather than hitting live services.

## Commit & Pull Request Guidelines

Recent history mostly uses short Conventional Commit prefixes such as `fix:`, `docs:`, `refactor:`, and `feat:`. Keep commit subjects imperative and scoped, for example `fix: prevent duplicate voice listener starts`. Pull requests should describe behavior changes, list manual verification steps, link related issues, and include screenshots or Discord embed captures for visible UI changes.

## Security & Configuration Tips

Never commit `.env`, API keys, Discord tokens, generated voice clips, or local cache/build artifacts. Use `.env.example` for documented configuration only. Treat `data/server_info.txt` as potentially sensitive server context and review it before sharing logs or examples.
