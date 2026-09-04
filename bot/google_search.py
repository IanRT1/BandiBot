"""Google-grounded web answers for BandiBot.

Gemini performs the search, synthesis, and citation selection. This module
only adapts the Gemini Interactions API response into concise Discord text.
"""

import logging

import aiohttp

from core.config import GEMINI_API_KEY, GEMINI_SEARCH_MODEL

logger = logging.getLogger(__name__)

_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def _extract_text_and_sources(data: dict) -> tuple[str, list[tuple[str, str]]]:
    """Extract model text and citation links from current or legacy responses."""
    text_parts: list[str] = []
    sources: list[tuple[str, str]] = []
    steps = data.get("steps") or data.get("outputs") or []

    for step in steps:
        step_type = step.get("type")
        if step_type in {"text", "model_output"}:
            if step_type == "text":
                text = step.get("text", "")
                annotations = step.get("annotations", [])
            else:
                blocks = step.get("content", [])
                text = " ".join(
                    block.get("text", "")
                    for block in blocks
                    if block.get("type") == "text"
                )
                annotations = [
                    annotation
                    for block in blocks
                    for annotation in block.get("annotations", [])
                ]
            if text:
                text_parts.append(text)
            for annotation in annotations:
                url = annotation.get("url") or annotation.get("source")
                title = annotation.get("title") or url
                if url and (title, url) not in sources:
                    sources.append((title, url))

    # Some API revisions expose the final answer directly.
    if not text_parts and data.get("output_text"):
        text_parts.append(data["output_text"])

    return "\n".join(text_parts).strip(), sources


async def search_web(question: str) -> str:
    """Return a concise, Google-grounded answer with source links."""
    if not GEMINI_API_KEY:
        return "Web search is not configured. Add GEMINI_API_KEY to .env."

    question = question.strip()
    if not question:
        return "I need a question to search for."

    payload = {
        "model": GEMINI_SEARCH_MODEL,
        "input": (
            "Answer the user's question concisely using Google Search Grounding. "
            "Give the single most useful answer first. Prefer current, authoritative sources. "
            "If the answer is uncertain or sources disagree, say so briefly.\n\n"
            f"User question: {question}"
        ),
        "tools": [{"type": "google_search", "search_types": ["web_search"]}],
    }
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(_INTERACTIONS_URL, json=payload, headers=headers) as response:
                data = await response.json(content_type=None)
                if response.status >= 400:
                    logger.error(
                        "[web] search failed with HTTP %s: %s",
                        response.status,
                        _format_api_error(data),
                    )
                    return "Google web search failed. Try again later."
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.error("Gemini web search connection failed: %s", exc)
        return "Google web search is temporarily unavailable."

    answer, sources = _extract_text_and_sources(data)
    if not answer:
        return "I couldn't find a useful answer from Google."

    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(
            f"- [{title}]({url})" for title, url in sources[:5]
        )
    return answer


def _format_api_error(data) -> str:
    """Expose useful provider diagnostics without logging request secrets."""
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        parts = [str(error[key]) for key in ("type", "code", "message", "status") if error.get(key)]
        if parts:
            return " | ".join(parts)[:500]
    return str(error)[:500]
