"""Speech delivery is independent of command success."""
from enum import Enum


class SpeechResult(str, Enum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    STALE = "stale"
