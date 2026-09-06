from datetime import datetime, timezone
from types import SimpleNamespace

from bot.retrieval import (
    build_retrieval_query,
    format_retrieved_context,
    retrieve_relevant_chunks,
    retrieve_with_confidence,
    should_retrieve_lore,
)
from bot.tool_schemas import (
    MUSIC_TOOLS,
    VOICE_TOOLS,
    WEB_SEARCH_TOOL,
    select_tools_for_request,
    tools_without_context_lookups,
    tools_without_context_lookups_or_web_search,
    tools_without_web_search,
)


def _tool_names(tools):
    return {tool["function"]["name"] for tool in tools}


def test_tool_routing_uses_small_groups_for_obvious_requests():
    assert select_tools_for_request("Play a song") == MUSIC_TOOLS
    assert select_tools_for_request("unete al canal") == VOICE_TOOLS
    assert select_tools_for_request("what is the weather today?") == WEB_SEARCH_TOOL
    assert select_tools_for_request("hello") == []


def test_tool_routing_preserves_lore_answers_without_tools():
    assert select_tools_for_request(
        "¿Quién es Poyo?",
        lore_is_confident=True,
    ) == []


def test_tool_routing_combines_groups_for_mixed_intents():
    names = _tool_names(
        select_tools_for_request("Play music and tell me the weather today")
    )

    assert "play_music" in names
    assert "web_search" in names


def test_explicit_web_intent_survives_confident_lore_context():
    names = _tool_names(
        select_tools_for_request(
            "¿Cuál es el clima hoy?",
            lore_is_confident=True,
        )
    )

    assert names == {"web_search"}


def test_tool_routing_keeps_ambiguous_questions_safe():
    names = _tool_names(select_tools_for_request("Can you help me?"))
    assert "play_music" in names
    assert "web_search" in names


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


def test_casual_messages_skip_lore_retrieval():
    document = "# Members\nMemberAlpha is a member."

    assert should_retrieve_lore("Hello, bot!", document) is False
    assert should_retrieve_lore("¿Quién es MemberAlpha?", document) is True


def test_short_follow_up_inherits_the_previous_user_question():
    history = [{"role": "user", "content": "¿Quién es el creador de BandiBot?"}]

    query = build_retrieval_query("¿Cuándo?", history)

    assert "creador de BandiBot" in query
    assert "¿Cuándo?" in query


def test_creator_question_can_match_a_single_strong_lore_term():
    document = "### Ian — creator\nIan creó y desarrolla BandiBot."

    chunks = retrieve_relevant_chunks(document, "¿Quién te creó?")

    assert chunks
    assert chunks[0].startswith("### Ian")


def test_long_message_is_not_augmented_with_history():
    history = [{"role": "user", "content": "A previous unrelated question"}]

    assert build_retrieval_query("Tell me the complete server history", history) == (
        "Tell me the complete server history"
    )


def test_retrieval_returns_no_context_for_unrelated_question():
    document = "# Server history\nThe server began as a gaming community."

    assert retrieve_relevant_chunks(document, "What is the weather today?") == []


def test_retrieval_ignores_generic_current_information_words():
    document = "# History\nThe server began in 2020 and has a recent event."

    assert retrieve_relevant_chunks(document, "What is the weather today?") == []
    assert retrieve_relevant_chunks(document, "What is the latest exchange rate?") == []


def test_retrieval_matches_safe_singular_plural_variants():
    document = "## Conceptos\nEsta sección explica un concepto importante del servidor."

    chunks = retrieve_relevant_chunks(document, "¿Qué conceptos explica el servidor?")

    assert chunks
    assert chunks[0].startswith("## Conceptos")


def test_retrieval_does_not_fuzzy_match_ordinary_sentence_words():
    document = "# Notes\nGroupAlpha is the group in which UserAlpha currently se encuentra."

    assert retrieve_relevant_chunks(document, "¿Cómo te encuentras?") == []


def test_retrieval_handles_stt_spelling_variation_for_member_name():
    document = "# Members\nMira pretende ser un DJ."

    chunks = retrieve_relevant_chunks(document, "¿Quién es Mina?")

    assert len(chunks) == 1
    assert "Mira" in chunks[0]


def test_retrieval_handles_name_spelling_variation():
    document = "# Members\nLa pareja de MemberAlpha se llama Lara."

    chunks = retrieve_relevant_chunks(document, "¿Sabes quién es Lira?")

    assert len(chunks) == 1
    assert "Lara" in chunks[0]


def test_retrieval_confidence_keeps_tools_for_weak_match():
    document = "# History\nThe server began as a gaming community in 2020."

    chunks, confident = retrieve_with_confidence(document, "community origin")

    assert chunks
    assert confident is False


def test_semantic_retrieval_can_recover_a_paraphrase(monkeypatch):
    import bot.retrieval as retrieval

    class FakeEmbeddingModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            assert normalize_embeddings is True
            vectors = []
            for text in texts:
                vectors.append(
                    [1.0, 0.0]
                    if "founding" in text or "origin" in text or "server began" in text
                    else [0.0, 1.0]
                )
            return __import__("numpy").array(vectors, dtype=float)

    monkeypatch.delenv("BANDIBOT_DISABLE_SEMANTIC_RAG", raising=False)
    monkeypatch.setattr(retrieval, "_embedding_model", FakeEmbeddingModel())
    monkeypatch.setattr(retrieval, "_embedding_model_failed", False)
    retrieval._index_cache.clear()
    retrieval._query_embedding_cache.clear()

    document = "# History\nThe server began as a gaming community in 2020."
    chunks = retrieve_relevant_chunks(document, "Tell me about the founding origin")

    assert len(chunks) == 1
    assert "server began" in chunks[0]


def test_bm25_prefers_repeated_query_terms():
    document = (
        "# General\nThe server has many activities.\n\n"
        "# Tournament\nThe annual tournament tournament schedule and tournament results."
    )

    chunks = retrieve_relevant_chunks(document, "tournament results")

    assert chunks[0].startswith("# Tournament")


def test_child_chunks_retain_parent_section_for_broad_questions():
    document = (
        "## History and events\n\n"
        "### Server founding\nThe server began as a gaming community.\n\n"
        "### First tournament\nThe first tournament happened in 2024."
    )

    chunks = retrieve_relevant_chunks(document, "What happened in the server history?")

    assert chunks
    assert any("[Section: History and events]" in chunk for chunk in chunks)


def test_semantic_only_match_requires_a_clear_lead(monkeypatch):
    import bot.retrieval as retrieval

    class AmbiguousEmbeddingModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            assert normalize_embeddings is True
            return __import__("numpy").array(
                [[0.8, 0.6] for _ in texts], dtype=float
            )

    monkeypatch.delenv("BANDIBOT_DISABLE_SEMANTIC_RAG", raising=False)
    monkeypatch.setattr(retrieval, "_embedding_model", AmbiguousEmbeddingModel())
    monkeypatch.setattr(retrieval, "_embedding_model_failed", False)
    retrieval._index_cache.clear()
    retrieval._query_embedding_cache.clear()

    document = "# History\nThe server began as a gaming community.\n\n# Members\nIan founded it."

    assert retrieve_relevant_chunks(document, "Tell me about the weather") == []


def test_embeddings_are_precomputed_once_per_document(monkeypatch):
    import bot.retrieval as retrieval

    class FakeEmbeddingModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            self.calls.append(list(texts))
            return __import__("numpy").array([[1.0, 0.0] for _ in texts], dtype=float)

    model = FakeEmbeddingModel()
    monkeypatch.delenv("BANDIBOT_DISABLE_SEMANTIC_RAG", raising=False)
    monkeypatch.setattr(retrieval, "_embedding_model", model)
    monkeypatch.setattr(retrieval, "_embedding_model_failed", False)
    retrieval._index_cache.clear()
    retrieval._query_embedding_cache.clear()

    document = "# History\nThe server began in 2020.\n\n# Members\nIan founded it."
    retrieve_relevant_chunks(document, "server origin")
    retrieve_relevant_chunks(document, "who founded it")

    assert len(model.calls) == 3
    assert len(model.calls[0]) == 2  # chunks encoded once as the document index
    assert len(model.calls[1]) == 1
    assert len(model.calls[2]) == 1


def test_retrieval_confidence_is_strong_for_member_name_match():
    document = "# Members\nLa pareja de MemberAlpha se llama Lara."

    chunks, confident = retrieve_with_confidence(document, "¿Quién es Lira?")

    assert chunks
    assert confident is True


def test_retrieved_context_explicitly_prevents_redundant_tool_call():
    context = format_retrieved_context(["# History\nThe server began in 2020."])
    names = {tool["function"]["name"] for tool in tools_without_context_lookups()}

    assert "answer directly without calling get_server_info" in context
    assert "get_server_info" not in names
    assert "get_member_activity" not in names


def test_server_lore_tool_sets_do_not_offer_web_search():
    assert "web_search" not in {
        tool["function"]["name"] for tool in tools_without_web_search()
    }
    assert "web_search" not in {
        tool["function"]["name"]
        for tool in tools_without_context_lookups_or_web_search()
    }
    assert "web_search" in {
        tool["function"]["name"] for tool in tools_without_context_lookups()
    }


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

    context, user_message, has_lore, has_lore_context = handlers.build_context_info(message, client)

    assert has_lore is True
    assert has_lore_context is True
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
