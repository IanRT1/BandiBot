"""
bot/retrieval.py

Local hybrid retrieval for BandiBot's private server lore.

Retrieval flow:
  server_info.txt → split into heading-based lore chunks
  → BM25 + conservative fuzzy name matching
  → cached multilingual embedding similarity
  → combined ranking → capped context excerpts for the LLM

Indexing:
  Chunk embeddings are precomputed and cached per document. Query embeddings
  are cached separately so repeated requests do not re-encode the lore.

Fallback behavior:
  Retrieval is local and privacy-preserving. Strong matches can omit redundant
  context-lookup tools, while weak matches keep those tools available. If the
  embedding dependency or model is unavailable, retrieval safely falls back to
  BM25 and fuzzy lexical matching.
"""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_RETRIEVED_CHARS = 8000
SEMANTIC_MIN_SCORE = 0.62
SEMANTIC_MIN_MARGIN = 0.03
_embedding_model = None
_embedding_model_failed = False
_index_cache: dict[str, "_LoreIndex"] = {}
_query_embedding_cache: dict[str, object] = {}


@dataclass(frozen=True)
class _ScoredChunk:
    chunk: str
    lexical_score: float
    semantic_score: float
    combined_score: float


@dataclass
class _LoreIndex:
    chunks: list[str]
    terms: list[list[str]]
    term_frequencies: list[Counter[str]]
    document_frequencies: Counter[str]
    average_length: float
    proper_names: list[set[str]]
    embeddings: object = None


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
        token for token in _TOKEN_RE.findall(_normalize(text)) if len(token) > 2
    }


def _term_list(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(_normalize(text)) if len(token) > 2]


def should_retrieve_lore(query: str, document: str = "") -> bool:
    """Return whether a message merits semantic lore retrieval.

    This gate is language-neutral: it does not classify greetings or maintain
    phrase lists. It keeps direct name/lore matches, including STT name slips,
    and avoids loading the embedding model for very short messages with no
    lexical connection to the lore. Longer paraphrases still reach semantic
    retrieval; the normal tool fallback handles short unmatched questions.
    """
    query_terms = _terms(query)
    if not query_terms:
        return False
    if not document:
        return len(query_terms) >= 3

    document_terms = _terms(document)
    if query_terms & document_terms:
        return True

    proper_names = {
        name
        for chunk in split_context_chunks(document)
        for name in _proper_name_terms(chunk)
    }
    if any(
        _terms_match(query_term, name)
        for query_term in query_terms
        for name in proper_names
    ):
        return True
    return len(query_terms) >= 3


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


def _lexical_terms_match(left: str, right: str) -> bool:
    """Match exact terms plus safe singular/plural variants."""
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4:
        return False
    return (
        left + "s" == right
        or right + "s" == left
        or left + "es" == right
        or right + "es" == left
    )


def _proper_name_terms(chunk: str) -> set[str]:
    """Return capitalized non-heading terms eligible for STT fuzzy matching."""
    names = set()
    for line in chunk.splitlines():
        words = _TOKEN_RE.findall(line)
        for index, word in enumerate(words):
            if not word[:1].isupper():
                continue
            # Avoid treating the first word of a normal sentence as a name.
            # Bullet entries may legitimately begin with a member name.
            stripped = line.lstrip()
            if (
                index == 0
                and not stripped.startswith(("#", "-", "*"))
                and len(_normalize(word)) < 4
            ):
                continue
            names.add(_normalize(word))
    return names


def _get_embedding_model():
    """Load the multilingual encoder lazily so startup stays lightweight."""
    global _embedding_model, _embedding_model_failed
    if os.getenv("BANDIBOT_DISABLE_SEMANTIC_RAG") == "1":
        return None
    if _embedding_model is not None or _embedding_model_failed:
        return _embedding_model
    try:
        # BandiBot uses the PyTorch backend. Prevent an installed TensorFlow /
        # Keras 3 stack from making Transformers import fail before the model
        # is even loaded.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        for library_logger in (
            "sentence_transformers",
            "transformers",
            "huggingface_hub",
        ):
            logging.getLogger(library_logger).setLevel(logging.WARNING)
        from sentence_transformers import SentenceTransformer

        logger.debug("[rag] loading local embedding model %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.debug("[rag] local embedding model ready")
    except Exception as exc:
        _embedding_model_failed = True
        logger.warning("[rag] semantic retrieval unavailable; using lexical retrieval: %s", exc)
    return _embedding_model


def _build_index(document: str) -> _LoreIndex:
    cached = _index_cache.get(document)
    if cached is not None:
        return cached

    chunks = split_context_chunks(document)
    terms = [_term_list(chunk) for chunk in chunks]
    term_frequencies = [Counter(chunk_terms) for chunk_terms in terms]
    document_frequencies = Counter(
        term for chunk_terms in terms for term in set(chunk_terms)
    )
    average_length = sum(map(len, terms)) / max(1, len(terms))
    index = _LoreIndex(
        chunks=chunks,
        terms=terms,
        term_frequencies=term_frequencies,
        document_frequencies=document_frequencies,
        average_length=average_length,
        proper_names=[_proper_name_terms(chunk) for chunk in chunks],
    )
    _index_cache[document] = index
    if len(_index_cache) > 8:
        _index_cache.pop(next(iter(_index_cache)))
    return index


def _semantic_scores(query: str, index: _LoreIndex) -> list[float]:
    """Return cosine similarities using embeddings precomputed per document."""
    model = _get_embedding_model()
    if model is None or not index.chunks:
        return [0.0] * len(index.chunks)

    if index.embeddings is None:
        index.embeddings = model.encode(
            index.chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    query_embedding = _query_embedding_cache.get(query)
    if query_embedding is None:
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        _query_embedding_cache[query] = query_embedding
        if len(_query_embedding_cache) > 64:
            _query_embedding_cache.pop(next(iter(_query_embedding_cache)))
    return [max(0.0, float(query_embedding @ embedding)) for embedding in index.embeddings]


def split_context_chunks(document: str) -> list[str]:
    """Split markdown lore into heading-based chunks for retrieval.

    Child chunks retain their parent heading as lightweight metadata. This lets
    a broad query such as "server history" find the relevant event entries
    without copying the whole parent section into every retrieved excerpt.
    """
    if not document.strip():
        return []
    raw_chunks = [
        chunk.strip()
        for chunk in re.split(r"(?=^#{1,6}\s)", document, flags=re.MULTILINE)
    ]
    raw_chunks = [chunk for chunk in raw_chunks if chunk]

    chunks: list[str] = []
    heading_path: list[tuple[int, str]] = []
    for chunk in raw_chunks:
        first_line, _, _ = chunk.partition("\n")
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", first_line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_path = [
                (parent_level, title)
                for parent_level, title in heading_path
                if parent_level < level
            ]
            heading_path.append((level, heading_match.group(2)))

        metadata = "\n".join(
            f"[Section: {title}]" for _, title in heading_path[:-1]
        )
        chunks.append(f"{chunk}\n{metadata}" if metadata else chunk)
    return chunks


def retrieve_relevant_chunks(
    document: str,
    query: str,
    *,
    max_chunks: int = 3,
    min_score: float = 0.2,
) -> list[str]:
    """Return the highest-scoring lore chunks relevant to the query.

    Lexical matching remains the precision anchor for names and STT errors;
    local multilingual embeddings recover paraphrased questions. If the
    optional encoder cannot load, this safely degrades to lexical retrieval.
    """
    query_terms = _terms(query)
    if not query_terms:
        return []

    index = _build_index(document)
    semantic_scores = _semantic_scores(query, index)
    query_term_list = list(query_terms)
    total_documents = len(index.chunks)
    bm25_scores: list[float] = []
    for term_frequencies, terms in zip(index.term_frequencies, index.terms):
        score = 0.0
        document_length = len(terms)
        for term in query_term_list:
            term_frequency = term_frequencies.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = index.document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_factor = 0.25 + 0.75 * document_length / max(1, index.average_length)
            score += inverse_document_frequency * (
                term_frequency * 2.5 / (term_frequency + 1.5 * length_factor)
            )
        bm25_scores.append(score)
    max_bm25 = max(bm25_scores, default=0.0)
    ranked_semantic_scores = sorted(semantic_scores, reverse=True)
    semantic_margin = (
        ranked_semantic_scores[0] - ranked_semantic_scores[1]
        if len(ranked_semantic_scores) > 1
        else ranked_semantic_scores[0] if ranked_semantic_scores else 0.0
    )
    allow_semantic_only = (
        bool(ranked_semantic_scores)
        and ranked_semantic_scores[0] >= SEMANTIC_MIN_SCORE
        and semantic_margin >= SEMANTIC_MIN_MARGIN
    )

    scored: list[_ScoredChunk] = []
    for index_number, (chunk, semantic_score) in enumerate(zip(index.chunks, semantic_scores)):
        chunk_terms = set(index.terms[index_number])
        proper_name_terms = index.proper_names[index_number]
        overlap = {
            query_term
            for query_term in query_term_list
            if query_term in chunk_terms
            or any(
                _terms_match(query_term, chunk_term)
                for chunk_term in proper_name_terms
            )
        }
        name_overlap = {
            query_term
            for query_term in query_term_list
            if any(
                _terms_match(query_term, chunk_name)
                for chunk_name in proper_name_terms
            )
        }
        fuzzy_name_score = len(overlap) / len(query_terms)
        lexical_score = max(
            fuzzy_name_score,
            bm25_scores[index_number] / max_bm25 if max_bm25 else 0.0,
        )
        heading = chunk.splitlines()[0] if chunk.splitlines() else ""
        heading_match = any(
            _lexical_terms_match(query_term, heading_term)
            for query_term in query_terms
            for heading_term in _terms(heading)
        )
        if heading_match:
            lexical_score = max(0.35, min(1.0, lexical_score + 0.15))
        if name_overlap:
            lexical_score = max(0.5, lexical_score)

        lexical_evidence = (
            len(overlap) >= 2
            or any(len(term) >= 5 for term in overlap)
            or heading_match
            or bool(name_overlap)
        )

        # Keep lexical matching dominant for proper names and short server
        # lore. Semantic-only matches still need a meaningful similarity.
        combined_score = 0.6 * lexical_score + 0.4 * semantic_score
        if (lexical_score >= min_score and lexical_evidence) or (
            allow_semantic_only and semantic_score >= SEMANTIC_MIN_SCORE
        ):
            scored.append(_ScoredChunk(chunk, lexical_score, semantic_score, combined_score))

    scored.sort(key=lambda item: item.combined_score, reverse=True)
    selected: list[str] = []
    total_chars = 0
    for item in scored:
        if len(selected) >= max_chunks:
            break
        if selected and total_chars + len(item.chunk) > MAX_RETRIEVED_CHARS:
            continue
        if item.combined_score < min_score and not (
            allow_semantic_only and item.semantic_score >= SEMANTIC_MIN_SCORE
        ):
            continue
        selected.append(item.chunk)
        total_chars += len(item.chunk)
    return selected


def retrieve_with_confidence(
    document: str,
    query: str,
    *,
    max_chunks: int = 3,
    min_score: float = 0.2,
) -> tuple[list[str], bool]:
    """Return lore plus whether retrieval is strong enough to suppress tools."""
    chunks = retrieve_relevant_chunks(
        document, query, max_chunks=max_chunks, min_score=min_score
    )
    if not chunks:
        return [], False

    query_terms = _terms(query)
    top_chunk = chunks[0]
    proper_names = _proper_name_terms(top_chunk)
    name_match = any(
        any(_terms_match(query_term, name) for name in proper_names)
        for query_term in query_terms
    )
    lexical_overlap = sum(
        1 for query_term in query_terms
        if any(
            _lexical_terms_match(query_term, chunk_term)
            for chunk_term in _terms(top_chunk)
        )
        or any(_terms_match(query_term, name) for name in proper_names)
    ) / max(1, len(query_terms))
    heading_terms = _terms(top_chunk.splitlines()[0])
    heading_match = any(
        _lexical_terms_match(query_term, heading_term)
        for query_term in query_terms
        for heading_term in heading_terms
    )
    return chunks, name_match or heading_match or lexical_overlap >= 0.6


def format_retrieved_context(chunks: list[str]) -> str:
    if not chunks:
        return ""
    return (
        "Relevant server knowledge retrieved locally. Treat these excerpts as "
        "confirmed context and answer directly without calling get_server_info. "
        "Voice transcription may slightly misspell names; match a user-mentioned "
        "name to a close name in these excerpts when the surrounding context fits, "
        "and do not reject the fact only because of that spelling difference:\n\n"
        + "\n\n".join(chunks)[:MAX_RETRIEVED_CHARS]
    )
