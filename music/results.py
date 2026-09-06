"""
music/results.py

Structured outcomes for individual music requests.

Fields:
  status identifies playback, queueing, or failure. Track metadata and the
  one-based upcoming queue position come from playback state. Failures carry
  a stable error code and a human-readable message.

Boundary:
  Tool execution serializes these results as JSON for text and voice models.
  Display wording is independent of the fields used to make decisions.
"""

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class PlayResult:
    status: Literal["playing", "starting", "queued", "failed"]
    title: str | None = None
    artist: str | None = None
    queue_position: int | None = None
    error_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}
