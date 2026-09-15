"""Translate structured song identity into resolver input without word removal."""


class SongSearchQuery(str):
    """Resolver input retaining the original single-song request for recovery."""

    def __new__(cls, value: str, original_text: str | None = None):
        query = super().__new__(cls, value)
        query.original_text = original_text if original_text is not None else value
        return query


class LosslessSongQuery(SongSearchQuery):
    """Original request retained because extraction omitted its possible title."""


def song_query(song: dict, *, original_text: str | None = None) -> str:
    if not isinstance(song, dict) or set(song) != {"title", "artist", "version", "source"}:
        raise ValueError("song must contain title, artist, version, and source.")
    if any(not isinstance(value, str) for value in song.values()):
        raise ValueError("Song identity fields must be strings.")
    if song["source"]:
        if any(song[field] for field in ("title", "artist", "version")):
            raise ValueError("Use either a direct source or song identity fields, not both.")
        query = song["source"]
    else:
        if not (song["title"].strip() or song["artist"].strip()):
            raise ValueError("A song title, artist, or direct source is required.")
        # Missing title is not proof that the user requested any song by this
        # artist. Search the lossless request instead of silently discarding
        # words the model may have mistaken for politeness. Existing identity
        # review interprets the results against this same original request.
        if not song["title"].strip() and original_text and original_text.strip():
            query = LosslessSongQuery(original_text.strip())
        else:
            query = " ".join(song[field].strip() for field in ("title", "artist", "version") if song[field].strip())
    if not query.strip() or len(query) > 500:
        raise ValueError("Song resolver input must contain 1 to 500 characters.")
    if original_text and not song["source"] and not isinstance(query, SongSearchQuery):
        query = SongSearchQuery(query, original_text)
    return query


async def recover_song_query(original_text: str, generate) -> str | None:
    """One failed-search repair; the resolver must verify results against the original."""
    import json

    response = await generate({"messages": [
        {"role": "system", "content": (
            "Recover the intended music search from a speech transcript after a failed search. "
            "The prior extraction or phonetic spelling may be wrong. Recover title and artist together. "
            "Apparent politeness between the action and artist may itself be the title. "
            "Ignore transcription commas and correct plausible phonetic artist spellings. "
            "Preserve the title in its original language, even when it resembles a courtesy phrase. "
            "Do not translate or substitute the title. Return JSON only with title_quote, artist, version "
            "as strings. title_quote must COPY the exact title words from the transcript, preserving spaces "
            "and spelling. The title may be between the action and the artist, separated by commas or 'by/de'. "
            "If no title can be recovered, return an empty title_quote."
        )}, {"role": "user", "content": str(original_text)},
    ], "max_completion_tokens": 180, "reasoning_effort": "none"})
    try:
        song = json.loads(response["choices"][0]["message"]["content"])
        if not isinstance(song, dict) or not isinstance(song.get("title_quote"), str) or not song["title_quote"].strip():
            return None
        title = song["title_quote"].strip()
        if title.casefold() not in original_text.casefold():
            return None
        return song_query({"title": title, "artist": song.get("artist", ""),
                           "version": song.get("version", ""), "source": ""})
    except (ValueError, TypeError, KeyError, IndexError):
        return None
