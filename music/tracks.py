"""
music/tracks.py

Shared track model for BandiBot's music pipeline.

Track instances are the boundary object between query resolution, uploaded
audio attachments, the playback queue, and the Now Playing display. The model
keeps playback timing, resolved metadata, and deferred-resolution state in one
small structure so the rest of the music system can pass tracks around without
duplicating state fields.

Lifecycle:
  Resolved track      -> has stream_url and can be played immediately
  Placeholder track   -> has resolved=False and query set for later resolution
  Errored placeholder -> has resolved=True and error set so playback can skip it

Metadata fields:
  title/requested_by identify the user-facing queue item. webpage_url points
  back to the source, duration/artist/thumbnail feed the Now Playing view, and
  thumbnail_bytes carries embedded cover art from uploaded audio files.

Playback timing:
  started_at, paused_at, and total_paused are maintained by music.player so the
  progress display can report elapsed playback time without coupling the view
  layer to Discord voice internals.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    title: str
    stream_url: str
    requested_by: str
    webpage_url: str
    duration: Optional[int] = None
    thumbnail: Optional[str] = None
    thumbnail_bytes: Optional[bytes] = None
    artist: Optional[str] = None
    http_headers: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    paused_at: Optional[float] = None
    total_paused: float = 0.0
    resolved: bool = True
    resolved_at: float = field(default_factory=time.time)
    query: Optional[str] = None
    error: Optional[str] = None
    playback_failures: int = 0
