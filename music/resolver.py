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
  Text queries request the top seven YouTube results, then prefer title word
  overlap and canonical studio signals such as Official Audio, Artist - Topic,
  and remaster uploads. Unrequested music videos are penalized because they can
  include intros, outros, or visual-only content; video results are favored when
  the user explicitly asks for a video. Live/remix/cover/slowed/etc. results are
  penalized unless the user explicitly asked for that variant.

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
import unicodedata
from urllib.parse import parse_qs, urlparse

import yt_dlp
import yt_dlp.version

from core.config import (
    YOUTUBE_JS_RUNTIME,
    YOUTUBE_REMOTE_COMPONENTS,
)
from music.tracks import Track

logger = logging.getLogger(__name__)

_SEARCH_RESULT_COUNT = 7
_YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_MIN_RETRY_QUERY_TOKEN_MATCH = 0.6
_MAX_RESOLVE_CANDIDATES = 4
_MISSING_QUERY_TOKEN_PENALTY = 10
_MIN_RELEVANT_QUERY_RATIO = 0.5
_LOGGED_YTDLP_CONFIG = False
_QUERY_FILLER_TOKENS = {
    "by", "ft", "feat", "featuring", "with", "and", "the", "a", "an",
    "official", "audio", "video",
}

_REQUESTABLE_VARIANTS = (
    "live", "remix", "acoustic", "cover", "instrumental", "karaoke",
    "slowed", "sped up", "nightcore", "mashup", "extended", "demo",
    "radio edit", "clean", "explicit", "remaster", "remastered",
)

_VARIANT_PENALTIES = {
    "live": -14,
    "concert": -12,
    "remix": -8,
    "mix": -4,
    "acoustic": -6,
    "cover": -9,
    "instrumental": -8,
    "karaoke": -10,
    "slowed": -9,
    "sped up": -9,
    "speed up": -9,
    "nightcore": -9,
    "mashup": -9,
    "bass boosted": -8,
    "8d": -8,
    "reverb": -7,
    "extended": -4,
    "demo": -6,
    "alternate": -5,
    "edit": -3,
    "radio edit": -3,
    "clean": -4,
    "rare version": -6,
    "version": -2,
    "vinyl": -4,
    "special": -5,
}

_NON_SONG_PENALTIES = {
    "reaction": -10,
    "review": -8,
    "full album": -10,
    "commentary": -8,
    "explained": -8,
    "interview": -8,
    "tutorial": -8,
    "analysis": -8,
    "lesson": -10,
    "guitar lesson": -12,
    "subtitulada": -3,
    "subtitulado": -3,
    "sub espanol": -3,
    "sub español": -3,
    "lyrics": -4,
    "lyric video": -4,
    "letra": -4,
    "com letra": -4,
    "karaoke": -10,
    "parodia": -10,
    "parody": -10,
    "la voz": -18,
    "the voice": -18,
    "la voz kids": -22,
    "semifinal": -12,
    "finalistas": -12,
    "finalist": -12,
    "audition": -14,
    "contestant": -14,
    "cantan": -10,
    "siempre en domingo": -14,
    "la casa de los famosos": -14,
    "talent show": -16,
}

_BOOTLEG_DATE_RE = re.compile(
    r"\b("
    r"jan|january|feb|february|mar|march|apr|april|may|jun|june|"
    r"jul|july|aug|august|sep|sept|september|oct|october|"
    r"nov|november|dec|december"
    r")\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

_BOOTLEG_LOCATION_WORDS = (
    "san francisco", "los angeles", "new york", "london", "paris",
    "tokyo", "chicago", "boston", "toronto", "berlin", "amsterdam",
)


class _YtdlpLogger:
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


_YTDLP_LOGGER = _YtdlpLogger()

_YDL_OPTS_BASE = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "logger": _YTDLP_LOGGER,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
    "ignoreerrors": False,
    "socket_timeout": 15,
    "retries": 2,
    "fragment_retries": 2,
}

_YDL_OPTS_FLAT_BASE = {
    "quiet": True,
    "no_warnings": True,
    "logger": _YTDLP_LOGGER,
    "extract_flat": True,
    "ignoreerrors": True,
    "socket_timeout": 15,
    "retries": 2,
}

_YDL_OPTS_SEARCH_BASE = {
    "quiet": True,
    "no_warnings": True,
    "logger": _YTDLP_LOGGER,
    "extract_flat": True,
    "default_search": "ytsearch",
    "ignoreerrors": True,
    "socket_timeout": 10,
    "retries": 1,
}


def _apply_youtube_session_opts(opts: dict) -> dict:
    opts = dict(opts)
    if YOUTUBE_JS_RUNTIME:
        runtime, _, runtime_path = YOUTUBE_JS_RUNTIME.partition(":")
        runtime_config = {}
        if runtime_path:
            runtime_config["path"] = runtime_path
        opts["js_runtimes"] = {runtime.strip().lower(): runtime_config}

    if YOUTUBE_REMOTE_COMPONENTS:
        opts["remote_components"] = [
            component.strip()
            for component in YOUTUBE_REMOTE_COMPONENTS.split(",")
            if component.strip()
        ]

    return opts


_YDL_OPTS = _apply_youtube_session_opts(_YDL_OPTS_BASE)
_YDL_OPTS_FLAT = _apply_youtube_session_opts(_YDL_OPTS_FLAT_BASE)
_YDL_OPTS_SEARCH = _apply_youtube_session_opts(_YDL_OPTS_SEARCH_BASE)


def resolve_track(
    query: str,
    requested_by: str,
    exclude_webpage_urls: set[str] | None = None,
) -> Track:
    _log_ytdlp_config_once()
    query = query.strip()
    direct_url = _direct_video_url(query)
    if direct_url:
        return _resolve_direct_url(direct_url, requested_by)

    query_lower = _normalize_text(query)
    exclude_webpage_urls = exclude_webpage_urls or set()
    scored_entries = _search_and_score(query, query_lower)
    scored_entries = _filter_excluded_entries(scored_entries, exclude_webpage_urls)
    if exclude_webpage_urls:
        scored_entries = _filter_weak_retry_entries(scored_entries, query_lower)
    if not scored_entries:
        raise Exception("No playable results found.")

    candidate_entries = list(scored_entries)
    best_score, best = max(candidate_entries, key=lambda item: item[0])
    if _should_retry_official_audio(query_lower, best_score, best):
        retry_query = f"{query} official audio"
        logger.debug(f"  [yt-dlp] retrying with studio-biased query {retry_query!r}")
        retry_scored_entries = _search_and_score(retry_query, query_lower)
        retry_scored_entries = _filter_excluded_entries(
            retry_scored_entries,
            exclude_webpage_urls,
        )
        if exclude_webpage_urls:
            retry_scored_entries = _filter_weak_retry_entries(
                retry_scored_entries,
                query_lower,
            )
        if retry_scored_entries:
            candidate_entries.extend(retry_scored_entries)

    candidate_entries = _filter_irrelevant_entries(candidate_entries, query_lower)
    if not candidate_entries:
        raise Exception("No sufficiently relevant playable results found.")

    _log_candidate_scores(candidate_entries)

    (
        winner_rank,
        winner_score,
        winner_candidate,
        best,
    ) = _select_playable_entry(candidate_entries)
    logger.debug(
        "  [yt-dlp] winner final_rank=%s score=%+d title=%r uploader=%r",
        winner_rank,
        winner_score,
        winner_candidate.get("title", "?"),
        winner_candidate.get("uploader") or winner_candidate.get("channel") or "?",
    )
    return Track(
        title=best.get("title", "Unknown title"),
        stream_url=best["url"],
        requested_by=requested_by,
        webpage_url=best.get("webpage_url", ""),
        duration=best.get("duration"),
        thumbnail=best.get("thumbnail"),
        artist=best.get("uploader") or best.get("channel"),
        http_headers=best.get("http_headers") or {},
        query=query,
    )


def _log_ytdlp_config_once():
    global _LOGGED_YTDLP_CONFIG
    if _LOGGED_YTDLP_CONFIG:
        return
    _LOGGED_YTDLP_CONFIG = True
    logger.debug(
        "[yt-dlp] version=%s js_runtime=%s remote_components=%s",
        yt_dlp.version.__version__,
        YOUTUBE_JS_RUNTIME or "default",
        YOUTUBE_REMOTE_COMPONENTS or "off",
    )


def _direct_video_url(query: str) -> str | None:
    if query.startswith("http://") or query.startswith("https://"):
        return query
    if _YOUTUBE_VIDEO_ID_RE.fullmatch(query):
        url = f"https://www.youtube.com/watch?v={query}"
        logger.debug(f"  [yt-dlp] detected YouTube video id {query!r}")
        return url
    return None


def _resolve_direct_url(url: str, requested_by: str) -> Track:
    logger.debug(f"  [yt-dlp] starting resolution for {url!r}")
    t = time.time()
    info = _resolve_playable_url(url)
    logger.debug(f"  [yt-dlp] resolved in {time.time() - t:.2f}s")

    logger.debug(f"  [yt-dlp] winner title={info.get('title', '?')!r}")
    return Track(
        title=info.get("title", "Unknown title"),
        stream_url=info["url"],
        requested_by=requested_by,
        webpage_url=info.get("webpage_url", url),
        duration=info.get("duration"),
        thumbnail=info.get("thumbnail"),
        artist=info.get("uploader") or info.get("channel"),
        http_headers=info.get("http_headers") or {},
        query=url,
    )


def _search_and_score(search_text: str, query_lower: str) -> list[tuple[int, dict]]:
    search_query = f"ytsearch{_SEARCH_RESULT_COUNT}:{search_text}"
    logger.debug(f"  [yt-dlp] searching {search_query!r}")
    t = time.time()

    with yt_dlp.YoutubeDL(_YDL_OPTS_SEARCH) as ydl:
        info = ydl.extract_info(search_query, download=False)
    logger.debug(f"  [yt-dlp] resolved in {time.time() - t:.2f}s")

    entries = [e for e in info.get("entries", []) if e]
    scored_entries = [
        (_score_result(entry, query_lower), entry)
        for entry in entries
    ]
    return scored_entries


def _select_playable_entry(
    scored_entries: list[tuple[int, dict]],
) -> tuple[int, int, dict, dict]:
    checked = 0
    for rank, (score, entry) in enumerate(
        _dedupe_scored_entries(scored_entries),
        start=1,
    ):
        if checked >= _MAX_RESOLVE_CANDIDATES:
            break
        checked += 1
        title = entry.get("title", "?")
        resolved_entry = _resolve_playable_candidate(entry)
        if resolved_entry:
            return rank, score, entry, resolved_entry
        logger.debug(
            "  [yt-dlp] skipped unplayable final_rank=%s score=%+d title=%r",
            rank,
            score,
            title,
        )
    raise Exception("No playable results found.")


def _resolve_playable_candidate(entry: dict) -> dict | None:
    candidate_url = _entry_webpage_url(entry)
    if not candidate_url:
        return None

    try:
        return _resolve_playable_url(candidate_url)
    except Exception:
        return None


def _resolve_playable_url(url: str) -> dict:
    info = _extract_one(url)
    if not info.get("url"):
        raise Exception("Resolved candidate has no stream URL.")
    return info


def _extract_one(input_url: str) -> dict:
    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        info = ydl.extract_info(input_url, download=False)

    if not info:
        raise Exception("No playable results found.")

    if "entries" in info:
        entry = next((e for e in info["entries"] if e), None)
        if not entry:
            raise Exception("No playable results found.")
        return entry

    return info


def _entry_webpage_url(entry: dict) -> str | None:
    url = entry.get("webpage_url")
    if url:
        return url
    url = entry.get("url")
    if url and _YOUTUBE_VIDEO_ID_RE.fullmatch(url):
        return f"https://www.youtube.com/watch?v={url}"
    if url and str(url).startswith(("http://", "https://")):
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
            return url
    video_id = entry.get("id")
    if video_id and _YOUTUBE_VIDEO_ID_RE.fullmatch(video_id):
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def _dedupe_scored_entries(scored_entries: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    deduped = []
    seen = set()
    for score, entry in sorted(scored_entries, key=lambda item: item[0], reverse=True):
        key = entry.get("webpage_url") or entry.get("id") or entry.get("url")
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, entry))
    return deduped


def _log_candidate_scores(scored_entries: list[tuple[int, dict]]):
    ranked_entries = _dedupe_scored_entries(scored_entries)[:3]
    for rank, (score, entry) in enumerate(ranked_entries, start=1):
        logger.debug(
            "  [yt-dlp] candidate #%d score=%+d title=%r uploader=%r",
            rank,
            score,
            entry.get("title", "?"),
            entry.get("uploader") or entry.get("channel") or "?",
        )


def _filter_excluded_entries(
    scored_entries: list[tuple[int, dict]],
    exclude_webpage_urls: set[str],
) -> list[tuple[int, dict]]:
    if not exclude_webpage_urls:
        return scored_entries

    filtered = [
        (score, entry)
        for score, entry in scored_entries
        if entry.get("webpage_url") not in exclude_webpage_urls
    ]
    skipped = len(scored_entries) - len(filtered)
    if skipped:
        logger.debug(f"  [yt-dlp] skipped {skipped} previously failed candidate(s)")
    return filtered


def _filter_weak_retry_entries(
    scored_entries: list[tuple[int, dict]],
    query_lower: str,
) -> list[tuple[int, dict]]:
    query_tokens = [
        token for token in _tokens(query_lower)
        if token not in _QUERY_FILLER_TOKENS
    ]
    if len(query_tokens) <= 1:
        return scored_entries

    filtered = []
    for score, entry in scored_entries:
        candidate_text = _normalize_text(
            f"{entry.get('title') or ''} "
            f"{entry.get('uploader') or entry.get('channel') or ''}"
        )
        candidate_tokens = set(_tokens(candidate_text))
        matches = sum(1 for token in query_tokens if token in candidate_tokens)
        if matches / len(query_tokens) >= _MIN_RETRY_QUERY_TOKEN_MATCH:
            filtered.append((score, entry))

    skipped = len(scored_entries) - len(filtered)
    if skipped:
        logger.debug(f"  [yt-dlp] skipped {skipped} weak retry candidate(s)")
    return filtered


def _filter_irrelevant_entries(
    scored_entries: list[tuple[int, dict]],
    query_lower: str,
) -> list[tuple[int, dict]]:
    """Reject playable results that match too little of a multi-word request."""
    query_tokens = {
        token for token in _tokens(query_lower)
        if token not in _QUERY_FILLER_TOKENS and len(token) > 2
    }
    if len(query_tokens) < 2:
        return scored_entries

    filtered = []
    for score, entry in scored_entries:
        candidate_tokens = set(_tokens(
            f"{entry.get('title') or ''} "
            f"{entry.get('uploader') or entry.get('channel') or ''}"
        ))
        match_ratio = len(query_tokens & candidate_tokens) / len(query_tokens)
        if match_ratio >= _MIN_RELEVANT_QUERY_RATIO:
            filtered.append((score, entry))
        else:
            logger.debug(
                "  [yt-dlp] rejected low-relevance candidate ratio=%.2f title=%r uploader=%r",
                match_ratio,
                entry.get("title", "?"),
                entry.get("uploader") or entry.get("channel") or "?",
            )
    return filtered

def extract_playlist(url: str, requested_by: str) -> list[Track]:
    """Extract playlist entries as unresolved placeholder tracks."""
    original_url = url
    if "list=" in url:
        match = re.search(r"list=([A-Za-z0-9_-]+)", url)
        if match:
            playlist_id = match.group(1)
            if playlist_id.startswith("RD"):
                video_url = _extract_video_url(original_url)
                if video_url:
                    logger.info(
                        f"  [yt-dlp] radio playlist {playlist_id!r} is not extractable; "
                        f"falling back to video {video_url!r}"
                    )
                    return [_placeholder_track(video_url, requested_by)]
            url = f"https://www.youtube.com/playlist?list={playlist_id}"

    logger.info(f"  [yt-dlp] extracting playlist {url!r}")
    t = time.time()

    with yt_dlp.YoutubeDL(_YDL_OPTS_FLAT) as ydl:
        info = ydl.extract_info(url, download=False)

    logger.info(f"  [yt-dlp] playlist extracted in {time.time() - t:.2f}s")

    if not info:
        video_url = _extract_video_url(original_url)
        if video_url:
            logger.info(f"  [yt-dlp] playlist returned no data; falling back to video {video_url!r}")
            return [_placeholder_track(video_url, requested_by)]
        raise Exception("No playable playlist data returned.")

    tracks = []
    for entry in info.get("entries", []):
        if not entry:
            continue
        title = entry.get("title") or entry.get("id") or "Unknown"
        entry_url = entry.get("url") or entry.get("webpage_url") or url
        tracks.append(_placeholder_track(entry_url, requested_by, title=title))

    logger.info(f"  [yt-dlp] found {len(tracks)} playlist entries")
    return tracks


def _placeholder_track(url: str, requested_by: str, title: str | None = None) -> Track:
    return Track(
        title=title or url,
        stream_url="",
        requested_by=requested_by,
        webpage_url=url,
        resolved=False,
        query=url,
    )


def _extract_video_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        video_id = parsed.path.strip("/").split("/", 1)[0]
    else:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    if not video_id:
        return None
    return f"https://www.youtube.com/watch?v={video_id}"


def _score_result(entry: dict, query_lower: str) -> int:
    title = _normalize_text(entry.get("title") or "")
    uploader = _normalize_text(entry.get("uploader") or entry.get("channel") or "")
    duration = entry.get("duration") or 0
    score = 0

    if 90 <= duration <= 540:
        score += 2
    if duration and duration < 60:
        score -= 6
    if duration > 600:
        score -= 4
    if duration > 1200:
        score -= 8

    query_words = set(_tokens(query_lower))
    title_words = set(_tokens(title))
    uploader_words = set(_tokens(uploader))
    candidate_words = title_words | uploader_words
    important_query_words = [
        word for word in _tokens(query_lower)
        if word not in _QUERY_FILLER_TOKENS
    ]
    missing_query_words = [
        word for word in important_query_words
        if word not in candidate_words
    ]
    overlap = query_words & title_words
    score += len(overlap) * 2

    compact_title_matches = _compact_title_matches(query_lower, title)
    if compact_title_matches:
        score += compact_title_matches * 5

    uploader_overlap = query_words & uploader_words
    score += len(uploader_overlap) * 3

    combined_overlap = overlap | uploader_overlap
    if query_words and len(overlap) / len(query_words) >= 0.75:
        score += 3
    if query_words and len(combined_overlap) / len(query_words) >= 0.75:
        score += 4

    if missing_query_words:
        score -= len(missing_query_words) * _MISSING_QUERY_TOKEN_PENALTY
        if len(important_query_words) >= 2:
            score -= 4

    # A title plus artist/uploader match is often the clean studio upload.
    # Do not grant this to random reuploads that stuff artist + title together
    # in the title while the uploader contributes no query match.
    if (
        uploader_overlap
        and query_words
        and len(combined_overlap) / len(query_words) >= 0.75
    ):
        score += 5

    if (
        uploader_overlap
        and len(title_words) <= 4
        and query_words
        and (
            len(combined_overlap) / len(query_words) >= 0.75
            or compact_title_matches
        )
    ):
        score += 8
    if uploader_overlap and compact_title_matches and len(title_words) <= 2:
        score += 12

    if "official audio" in title or "audio oficial" in title:
        score += 8
    elif "audio" in title:
        score += 3 if uploader_overlap else 1

    requested_video = _contains_phrase(query_lower, "video")
    if (
        "official video" in title
        or "official music video" in title
        or "video oficial" in title
    ):
        score += 3 if requested_video else -8
    elif "video" in title and not requested_video:
        score -= 6
    if "remaster" in title or "remastered" in title:
        score += 4

    if uploader.endswith(" - topic"):
        score += 8
    if "vevo" in uploader:
        score += 4
    if "official" in uploader or "oficial" in uploader:
        score += 3

    # Artist channels often compact the artist name into values such as
    # "ModjoOfficial". Reward that authority signal when the same name also
    # appears in the candidate title, without boosting unrelated reuploads.
    authoritative_uploader = (
        "official" in uploader
        or "oficial" in uploader
        or " - topic" in uploader
        or "vevo" in uploader
    )
    if (
        authoritative_uploader
        and any(
            len(word) >= 4 and word in uploader
            for word in title_words
            if word not in _QUERY_FILLER_TOKENS
        )
    ):
        score += 8

    # For ordinary playback, an official lyric upload is preferable to an
    # official music video because it usually provides a cleaner audio-first
    # stream. The bonus is limited to authoritative uploaders.
    if authoritative_uploader and _contains_phrase(title, "lyric"):
        score += 6

    requested_variants = {
        variant for variant in _REQUESTABLE_VARIANTS
        if _contains_phrase(query_lower, variant)
    }

    for phrase, penalty in _VARIANT_PENALTIES.items():
        if _contains_phrase(title, phrase) and not _variant_requested(phrase, requested_variants):
            score += penalty

    candidate_text = f"{title} {uploader}"
    for phrase, penalty in _NON_SONG_PENALTIES.items():
        if _contains_phrase(candidate_text, phrase) and not _contains_phrase(query_lower, phrase):
            score += penalty

    if not _variant_requested("live", requested_variants):
        if _BOOTLEG_DATE_RE.search(title):
            score -= 10
        if any(location in title for location in _BOOTLEG_LOCATION_WORDS):
            score -= 4

    return score


def _should_retry_official_audio(query_lower: str, best_score: int, best: dict) -> bool:
    if "official audio" in query_lower:
        return False

    requested_variants = {
        variant for variant in _REQUESTABLE_VARIANTS
        if _contains_phrase(query_lower, variant)
    }
    if requested_variants & {
        "live", "remix", "acoustic", "cover", "instrumental", "karaoke",
        "slowed", "sped up", "nightcore", "mashup", "extended", "demo",
    }:
        return False

    return best_score < 8 or _has_unrequested_bad_markers(best, query_lower)


def _has_unrequested_bad_markers(entry: dict, query_lower: str) -> bool:
    title = _normalize_text(entry.get("title") or "")
    uploader = _normalize_text(entry.get("uploader") or entry.get("channel") or "")
    candidate_text = f"{title} {uploader}"
    requested_variants = {
        variant for variant in _REQUESTABLE_VARIANTS
        if _contains_phrase(query_lower, variant)
    }

    for phrase in _VARIANT_PENALTIES:
        if _contains_phrase(title, phrase) and not _variant_requested(phrase, requested_variants):
            return True

    for phrase in _NON_SONG_PENALTIES:
        if _contains_phrase(candidate_text, phrase) and not _contains_phrase(query_lower, phrase):
            return True

    if not _variant_requested("live", requested_variants):
        if _BOOTLEG_DATE_RE.search(title):
            return True
        if any(location in title for location in _BOOTLEG_LOCATION_WORDS):
            return True

    return False


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(text))


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _compact_title_matches(query_lower: str, title: str) -> int:
    """Count title words that match adjacent query words joined together.

    YouTube often returns studio uploads with compact spellings like
    "Evenflow" while the user asks for "Even Flow". Without this, official
    videos with spaced words can beat the cleaner studio result.
    """
    query_tokens = _tokens(query_lower)
    title_tokens = set(_tokens(title))
    matches = 0
    for size in (2, 3):
        for i in range(0, len(query_tokens) - size + 1):
            compact = "".join(query_tokens[i:i + size])
            if len(compact) >= 5 and compact in title_tokens:
                matches += 1
    return matches


def _contains_phrase(text: str, phrase: str) -> bool:
    return _normalize_text(phrase) in _normalize_text(text)


def _variant_requested(phrase: str, requested_variants: set[str]) -> bool:
    if phrase in requested_variants:
        return True
    if phrase == "concert" and "live" in requested_variants:
        return True
    if phrase in {"sped up", "speed up"} and {"sped up", "speed up"} & requested_variants:
        return True
    if phrase in {"edit", "radio edit"} and "radio edit" in requested_variants:
        return True
    return False
