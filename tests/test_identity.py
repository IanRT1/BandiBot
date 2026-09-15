import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from music.identity import Clarifications, IdentityReviewRequired, review_identity
from music.resolver import _supported_identity_entries


def test_failed_match_does_not_create_menu_and_corrected_request_can_resolve(monkeypatch):
    from music import player
    from music.identity import NeedsClarification, clarifications
    from bot.interactions import current_request, RequestContext

    unrelated = [candidate("U Smile", "Justin Bieber"), candidate("Flatline", "Justin Bieber")]
    found = object()
    resolver = Mock(side_effect=[IdentityReviewRequired("unmatched transcript", unrelated), found])
    monkeypatch.setattr(player, "resolve_track", resolver)
    monkeypatch.setattr("music.identity.review_identity", AsyncMock(return_value=[]))
    monkeypatch.setattr(clarifications, "pending", {})

    async def run():
        token = current_request.set(RequestContext("regression", 100, 200, "voice", "Play, por fa, by Justin Willis."))
        try:
            clarifications.put(100, 200, [entry for _, entry in unrelated])
            with pytest.raises(NeedsClarification) as failure:
                await player._resolve_track_async("Play, por fa Justin Willis", "Test user")
            assert "U Smile" not in failure.value.result.message
            assert "Flatline" not in failure.value.result.message
            assert clarifications.context(100, 200) == ""
            assert await player._resolve_track_async("Porfa Justin Quiles", "Test user") is found
            assert resolver.call_count == 2
        finally:
            current_request.reset(token)
    asyncio.run(run())


def test_optional_suggestions_explicitly_allow_replacement_and_clear_per_user():
    store = Clarifications()
    store.put(1, 2, [{"title": "A"}])
    store.put(1, 3, [{"title": "B"}])
    assert "Never insist" in store.context(1, 2)
    store.clear(1, 2)
    assert store.context(1, 2) == ""
    assert store.context(1, 3)


@pytest.mark.parametrize("title_missing", [True, False])
def test_lost_title_recovery_is_bounded_and_cannot_bypass_identity_review(monkeypatch, title_missing):
    from music import player
    from music.requests import LosslessSongQuery, SongSearchQuery
    from music.identity import NeedsClarification

    unrelated = [candidate("Other song", "Artist")]
    resolver = Mock(side_effect=[IdentityReviewRequired("original", []), IdentityReviewRequired("repaired", unrelated)])
    repair = AsyncMock(return_value="repaired")
    review = AsyncMock(return_value=[])
    monkeypatch.setattr(player, "resolve_track", resolver)
    monkeypatch.setattr("music.requests.recover_song_query", repair)
    monkeypatch.setattr("music.identity.review_identity", review)
    query = LosslessSongQuery("original") if title_missing else SongSearchQuery("extracted", "original")
    with pytest.raises(NeedsClarification):
        asyncio.run(player._resolve_track_async(query, "Test"))
    repair.assert_awaited_once()
    assert repair.call_args.args[0] == "original"
    review.assert_awaited_once()
    assert resolver.call_count == 2
    assert resolver.call_args.kwargs == {"require_identity_review": True}


def test_successful_search_does_not_call_recovery_model(monkeypatch):
    from music import player
    from music.requests import SongSearchQuery
    track = object()
    repair = AsyncMock()
    monkeypatch.setattr(player, "resolve_track", Mock(return_value=track))
    monkeypatch.setattr("music.requests.recover_song_query", repair)
    assert asyncio.run(player._resolve_track_async(SongSearchQuery("song artist", "play song by artist"), "Test")) is track
    repair.assert_not_awaited()


def candidate(title, uploader="", score=0):
    return score, {"title": title, "uploader": uploader}


def test_incidental_overlap_cannot_authorize_wrong_song():
    candidates = [candidate("Feid, Justin Quiles - Porfa", "Feid", 20),
                  candidate("Justin Quiles - Loco Por Verte", "GLAD Empire", -26)]
    assert _supported_identity_entries(candidates, "por fa justin kills") == []
    assert _supported_identity_entries(candidates, "por fa justin quiles") == candidates[:1]


def test_identity_gate_does_not_use_upload_quality_as_match_evidence():
    candidates = [candidate("Wrong Song", "Correct Artist", 100),
                  candidate("Right Song", "Correct Artist", -20)]
    assert _supported_identity_entries(candidates, "right song correct artist") == candidates[1:]


def test_review_rejects_invented_candidate_ids_and_ungrounded_evidence():
    review = IdentityReviewRequired("Por Fa Justin Kills", [candidate("Porfa", "Justin Quiles")])

    async def run():
        for decision in [
            {"status": "selected", "candidate_ids": ["https://invented"], "title_evidence": "Porfa", "artist_evidence": "Justin Quiles"},
            {"status": "selected", "candidate_ids": ["0"], "title_evidence": "Another song", "artist_evidence": "Justin Quiles"},
            {"status": "ambiguous", "candidate_ids": ["0"]},
        ]:
            async def generate(payload):
                return {"choices": [{"message": {"content": json.dumps(decision)}}]}
            assert await review_identity(review, generate) == []
    asyncio.run(run())


def test_review_can_select_existing_identity_with_literal_evidence():
    entries = [candidate("Porfa", "Justin Quiles"), candidate("Loco Por Verte", "Justin Quiles")]
    async def generate(payload):
        return {"choices": [{"message": {"content": json.dumps({
            "status": "selected", "candidate_ids": ["0"],
            "title_evidence": "Porfa", "artist_evidence": "Justin Quiles",
        })}}]}
    assert asyncio.run(review_identity(IdentityReviewRequired("Por Fa Justin Kills", entries), generate)) == entries[:1]


def test_clarification_is_scoped_expires_and_cannot_be_reused(monkeypatch):
    store = Clarifications()
    choices = store.put(1, 2, [{"title": "A"}])
    key = next(iter(choices))
    assert store.take(1, 3, key) is None
    assert store.take(2, 2, key) is None
    assert store.take(1, 2, key) == {"title": "A"}
    assert store.take(1, 2, key) is None
    store.put(1, 2, [{"title": "B"}])
    monkeypatch.setattr("music.identity.time.monotonic", lambda: float("inf"))
    assert store.context(1, 2) == ""
