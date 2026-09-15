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


class ActionResult(str):
    """Typed control outcome with string compatibility for existing UI callers."""
    def __new__(cls, action: str, status: str, message: str):
        value = super().__new__(cls, message)
        value.action = action
        value.status = status
        return value

    def to_dict(self):
        return {"action": self.action, "status": self.status, "message": str(self)}


@dataclass(frozen=True)
class PlayResult:
    status: Literal["accepted", "resolving", "needs_clarification", "playing", "starting", "queued", "failed", "cancelled"]
    title: str | None = None
    artist: str | None = None
    queue_position: int | None = None
    error_code: str | None = None
    message: str | None = None
    entry_id: str | None = None
    operation_id: str | None = None
    revision: int | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}
