"""Runtime and packaged resource paths."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def runtime_root() -> Path:
    """Return the writable root used for user data and operational files."""
    return Path(os.getenv("BANDIBOT_RUNTIME_DIR", Path.cwd())).expanduser().resolve()


def data_root() -> Path:
    """Return the writable user data directory."""
    return Path(os.getenv("BANDIBOT_DATA_DIR", runtime_root() / "data")).expanduser().resolve()


def packaged_data_root() -> Path:
    """Return the read-only templates shipped with the application."""
    return PACKAGE_ROOT / "data"


def assets_root() -> Path:
    """Return the bundled runtime assets."""
    return PACKAGE_ROOT / "assets"


def context_path(filename: str) -> Path:
    """Prefer a user override and fall back to a packaged example template."""
    private_path = data_root() / filename
    if private_path.is_file():
        return private_path
    return packaged_data_root() / (
        filename.removesuffix(".txt") + ".example.txt"
    )
