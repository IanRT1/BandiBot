"""
music/resolver.py

YouTube and playlist resolution for BandiBot's music system.

This module owns the yt-dlp boundary: turning a user query, direct URL, or
playlist URL into Track objects that the player can queue and stream. Playback
state, Discord voice connections, queue mutation, and Now Playing updates stay
in music.player; this file only resolves media metadata and stream URLs.

Resolution modes:
  Direct URL   -> resolved immediately into a playable stream URL
  Text query   -> searches YouTube, scores candidates, returns best match
  Playlist URL -> extracts flat entries as unresolved placeholder tracks

Search ranking:
  Text queries request the top five YouTube results, then prefer title word
  overlap while softly penalizing very long videos and likely non-song content
  such as reactions, reviews, commentary, interviews, and full albums.

Playlist behavior:
  Playlist entries are intentionally not fully resolved here. They enter the
  queue as lightweight placeholders so large playlists can enqueue quickly;
  music.player resolves each item one ahead as playback progresses.

Output:
  Returns Track objects with normalized title, stream URL, source URL,
  duration, thumbnail URL, artist/uploader metadata, and deferred-resolution
  fields populated for playlist placeholders.
"""

import logging
import re
import time

import yt_dlp

from music.tracks import Track

logger = logging.getLogger(__name__)

_YDL_OPTS = {
    "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best[acodec!=none]/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
    "ignoreerrors": True,
}

_YDL_OPTS_FLAT = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": True,
    "ignoreerrors": True,
}


def resolve_track(query: str, requested_by: str) -> Track:
    if query.startswith("http://") or query.startswith("https://"):
        logger.info(f"  [yt-dlp] starting resolution for {query!r}")
        t = time.time()
        with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
            info = ydl.extract_info(query, download=False)
        logger.info(f"  [yt-dlp] resolved in {time.time() - t:.2f}s")

        if "entries" in info:
            entry = next((e for e in info["entries"] if e), None)
            if not entry:
                raise Exception("No playable results found.")
            info = entry

        logger.info(f"  [yt-dlp] using: {info.get('title', '?')!r}")
        return Track(
            title=info.get("title", "Unknown title"),
            stream_url=info["url"],
            requested_by=requested_by,
            webpage_url=info.get("webpage_url", ""),
            duration=info.get("duration"),
            thumbnail=info.get("thumbnail"),
            artist=info.get("uploader") or info.get("channel"),
        )

    query_lower = query.lower()
    search_query = f"ytsearch5:{query}"
    logger.info(f"  [yt-dlp] searching {search_query!r}")
    t = time.time()

    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        info = ydl.extract_info(search_query, download=False)
    logger.info(f"  [yt-dlp] resolved in {time.time() - t:.2f}s")

    entries = [e for e in info.get("entries", []) if e]
    if not entries:
        raise Exception("No playable results found.")

    best = max(entries, key=lambda entry: _score_result(entry, query_lower))
    logger.info(f"  [yt-dlp] using: {best.get('title', '?')!r}")
    return Track(
        title=best.get("title", "Unknown title"),
        stream_url=best["url"],
        requested_by=requested_by,
        webpage_url=best.get("webpage_url", ""),
        duration=best.get("duration"),
        thumbnail=best.get("thumbnail"),
        artist=best.get("uploader") or best.get("channel"),
    )


def extract_playlist(url: str, requested_by: str) -> list[Track]:
    """Extract playlist entries as unresolved placeholder tracks."""
    if "list=" in url:
        match = re.search(r"list=([A-Za-z0-9_-]+)", url)
        if match:
            url = f"https://www.youtube.com/playlist?list={match.group(1)}"

    logger.info(f"  [yt-dlp] extracting playlist {url!r}")
    t = time.time()

    with yt_dlp.YoutubeDL(_YDL_OPTS_FLAT) as ydl:
        info = ydl.extract_info(url, download=False)

    logger.info(f"  [yt-dlp] playlist extracted in {time.time() - t:.2f}s")

    tracks = []
    for entry in info.get("entries", []):
        if not entry:
            continue
        title = entry.get("title") or entry.get("id") or "Unknown"
        entry_url = entry.get("url") or entry.get("webpage_url") or url
        tracks.append(
            Track(
                title=title,
                stream_url="",
                requested_by=requested_by,
                webpage_url=entry_url,
                resolved=False,
                query=entry_url,
            )
        )

    logger.info(f"  [yt-dlp] found {len(tracks)} playlist entries")
    return tracks


def _score_result(entry: dict, query_lower: str) -> int:
    title = (entry.get("title") or "").lower()
    duration = entry.get("duration") or 0
    score = 0

    if duration > 600:
        score -= 2
    if duration > 1200:
        score -= 3

    query_words = set(query_lower.split())
    title_words = set(title.split())
    score += len(query_words & title_words)

    unwanted = ("reaction", "review", "full album", "commentary", "explained", "interview")
    wanted_override = any(w in query_lower for w in unwanted)
    if not wanted_override and any(w in title for w in unwanted):
        score -= 3

    return score
