"""Small rolling window for BandiBot session diagnostics."""

from __future__ import annotations

import os
from pathlib import Path


def rotate_session_log(current_path: Path) -> Path:
    """Move the current session log into the single previous-session slot."""
    current_path = Path(current_path)
    previous_path = current_path.with_name(
        f"{current_path.stem}.previous{current_path.suffix}"
    )
    if current_path.is_file():
        os.replace(current_path, previous_path)
    return previous_path
