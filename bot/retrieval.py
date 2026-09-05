"""Small local retrieval layer for server knowledge.

This is the retrieval part of BandiBot's RAG pipeline. It intentionally uses
local lexical scoring instead of another API or model: the retrieved excerpts
are added to the single LLM request before generation.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
_STOPWORDS = {
    "a", "al", "and", "como", "con", "cuál", "cual", "de", "del", "el",
    "en", "es", "for", "how", "is", "la", "las", "los", "me", "of", "on",
    "qué", "que", "se", "sobre", "the", "un", "una", "what", "when", "who",
    "why", "with", "y", "yo", "server", "servidor",
}

# Stopwords are compared after accent normalization, so "qué", "quién",
# and "quien" behave identically.
_STOPWORDS = {
    "".join(
        char for char in unicodedata.normalize("NFD", word.lower())
        if unicodedata.category(char) != "Mn"
    )
    for word in _STOPWORDS
} | {"quien", "quienes", "cual", "cuales"}


def load_context_file(filename: str) -> str:
    """Load a private context file, falling back to its tracked template."""
    private_path = os.path.join("data", filename)
    example_path = os.path.join(
        "data", filename.removesuffix(".txt") + ".example.txt"
    )
    path = private_path if os.path.exists(private_path) else example_path
    try:
        with open(path, "r", encoding="utf-8") as context_file:
            return context_file.read().strip()
    except OSError as exc:
        logger.error("Unable to load context file %s: %s", path, exc)
        return ""


def load_server_lore(model_name: str = "") -> str:
    return load_context_file("server_info.txt").replace("{model_name}", model_name)


def _normalize(text: str) -> str:
    without_marks = "".join(
        char for char in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(char) != "Mn"
    )
    return without_marks


def _terms(text: str) -> set[str]:
    return {
        token for token in _TOKEN_RE.findall(_normalize(text))
        if len(token) > 2 and token not in _STOPWORDS
    }


def _fold_repeated_letters(term: str) -> str:
    """Fold repeated letters for small speech-to-text spelling slips."""
    return re.sub(r"(.)\1+", r"\1", term)


def _terms_match(left: str, right: str) -> bool:
    """Match exact terms plus conservative STT typos in longer words."""
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4:
        return False

    left = _fold_repeated_letters(left)
    right = _fold_repeated_letters(right)
    if abs(len(left) - len(right)) > 1:
        return False

    # Accept at most one insertion, deletion, or substitution after folding
    # repeated letters (e.g. pollo -> polo vs Poyo). This avoids broad fuzzy
    # matching that could attach unrelated lore to a question.
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1:] == shorter:
            return True
    return False


def _proper_name_terms(chunk: str) -> set[str]:
    """Return capitalized non-heading terms eligible for STT fuzzy matching."""
    names = set()
    for line in chunk.splitlines():
        words = _TOKEN_RE.findall(line)
        for index, word in enumerate(words):
            if not word[:1].isupper():
                continue
            # Avoid treating a heading's first word as a person/name. Body
            # lines and bullet entries may legitimately begin with a name.
            stripped = line.lstrip()
            if index == 0 and stripped.startswith("#"):
                continue
            names.add(_normalize(word))
    return names


def split_context_chunks(document: str) -> list[str]:
    """Split markdown lore into heading-based chunks for retrieval."""
    if not document.strip():
        return []
    chunks = [chunk.strip() for chunk in re.split(r"(?=^#{1,6}\s)", document, flags=re.MULTILINE)]
    return [chunk for chunk in chunks if chunk]


def retrieve_relevant_chunks(
    document: str,
    query: str,
    *,
    max_chunks: int = 3,
    min_score: float = 0.2,
) -> list[str]:
    """Return the highest-scoring lore chunks relevant to the query."""
    query_terms = _terms(query)
    if not query_terms:
        return []

    scored: list[tuple[float, str]] = []
    for chunk in split_context_chunks(document):
        chunk_terms = _terms(chunk)
        proper_name_terms = _proper_name_terms(chunk)
        overlap = {
            query_term
            for query_term in query_terms
            if query_term in chunk_terms
            or any(
                _terms_match(query_term, chunk_term)
                for chunk_term in proper_name_terms
            )
        }
        if not overlap:
            continue
        score = len(overlap) / len(query_terms)
        heading = chunk.splitlines()[0] if chunk.splitlines() else ""
        if query_terms & _terms(heading):
            score += 0.15
        if score >= min_score:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:max_chunks]]


def format_retrieved_context(chunks: list[str]) -> str:
    if not chunks:
        return ""
    return (
        "Relevant server knowledge retrieved locally. Treat these excerpts as "
        "confirmed context and answer directly without calling get_server_info. "
        "Voice transcription may slightly misspell names; match a user-mentioned "
        "name to a close name in these excerpts when the surrounding context fits, "
        "and do not reject the fact only because of that spelling difference:\n\n"
        + "\n\n".join(chunks)
    )
