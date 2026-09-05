from datetime import datetime, timezone
from types import SimpleNamespace

from bot.retrieval import format_retrieved_context, retrieve_relevant_chunks
from bot.tool_schemas import tools_without_context_lookups


def test_retrieval_returns_only_relevant_server_lore():
    document = """# Server history
The server began as a small gaming community in 2020.

# Tournament
The annual tournament started in 2024.
"""

    chunks = retrieve_relevant_chunks(document, "What happened during the annual tournament?")

    assert len(chunks) == 1
    assert "annual tournament" in chunks[0]
    assert "gaming community" not in chunks[0]


def test_retrieval_returns_no_context_for_unrelated_question():
    document = "# Server history\nThe server began as a gaming community."

    assert retrieve_relevant_chunks(document, "What is the weather today?") == []


def test_retrieval_handles_stt_spelling_variation_for_member_name():
    document = "# Members\nPoyo pretende ser un DJ."

    chunks = retrieve_relevant_chunks(document, "¿Quién es el pollo?")

    assert len(chunks) == 1
    assert "Poyo" in chunks[0]


def test_retrieval_handles_name_spelling_variation():
    document = "# Members\nLa novia de Trabis se llama Beyra."

    chunks = retrieve_relevant_chunks(document, "¿Sabes quién es Beira?")

    assert len(chunks) == 1
    assert "Beyra" in chunks[0]


def test_retrieved_context_explicitly_prevents_redundant_tool_call():
    context = format_retrieved_context(["# History\nThe server began in 2020."])
    names = {tool["function"]["name"] for tool in tools_without_context_lookups()}

    assert "answer directly without calling get_server_info" in context
    assert "get_server_info" not in names
    assert "get_member_activity" not in names


def test_text_context_includes_retrieved_lore_before_llm(monkeypatch):
    import bot.handlers as handlers

    monkeypatch.setattr(
        handlers,
        "_SERVER_LORE",
        "# Tournament\nThe annual tournament started in 2024.",
    )
    owner = SimpleNamespace(nick=None, name="Owner")
    author = SimpleNamespace(nick=None, name="Ian")
    guild = SimpleNamespace(
        name="Test Server",
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        owner=owner,
        member_count=3,
    )
    message = SimpleNamespace(
        content="<@123> What happened during the annual tournament?",
        author=author,
        guild=guild,
        channel=SimpleNamespace(name="general"),
    )
    client = SimpleNamespace(
        user=SimpleNamespace(mention="<@123>", id=123, display_name="BandiBot")
    )

    context, user_message, has_lore = handlers.build_context_info(message, client)

    assert has_lore is True
    assert "annual tournament started in 2024" in context
    assert user_message == "What happened during the annual tournament?"


def test_unmatched_server_info_lookup_does_not_fallback_to_full_document(monkeypatch):
    import bot.handlers as handlers

    monkeypatch.setattr(
        handlers,
        "_SERVER_LORE",
        "# History\nThe server began in 2020.\n\n# Members\nIan founded the bot.",
    )

    result = handlers.build_server_info_context("What is the weather today?")

    assert "No directly matching server lore" in result
    assert "The server began in 2020" not in result
    assert "Ian founded the bot" not in result
