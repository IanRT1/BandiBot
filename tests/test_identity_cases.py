"""Held-out candidate fixtures: no aliases or private server context."""
import pytest

from music.resolver import _supported_identity_entries


CASES = [
    ("even flow pearl jam", [("Pearl Jam - Evenflow", "Pearl Jam"), ("Pearl Jam - Alive", "Pearl Jam")], [0]),
    ("hotel california eagles live", [("Eagles - Hotel California (Live)", "Eagles"), ("Eagles - Hotel California", "Eagles")], [0]),
    ("walk it down talking heads", [("Talking Heads - Walk It Down", "Talking Heads"), ("Talking Heads - Once in a Lifetime", "Talking Heads")], [0]),
    ("stop this train john mayer", [("Stop This Train", "John Mayer"), ("Slow Dancing in a Burning Room", "John Mayer")], [0]),
    ("beyonce halo", [("Beyoncé - Halo", "Beyoncé"), ("Beyoncé - Hello", "Beyoncé")], [0]),
    ("numb linkin park", [("Numb", "Linkin Park"), ("Numb", "U2")], [0]),
    ("por fa justin kills", [("Feid, Justin Quiles - Porfa", "Feid"), ("Justin Quiles - Loco Por Verte", "GLAD Empire")], []),
    ("this one guarded playboy cardi", [("Playboi Carti - DIS 1 GOT IT", "Playboi Carti"), ("Playboi Carti - TOXIC", "Playboi Carti")], []),
    ("invented title existing artist", [("Existing Artist - Different Song", "Existing Artist")], []),
]


@pytest.mark.parametrize("query,entries,expected", CASES)
def test_identity_fast_path_requires_recording_evidence(query, entries, expected):
    candidates = [(100 - i, {"title": title, "uploader": artist}) for i, (title, artist) in enumerate(entries)]
    result = _supported_identity_entries(candidates, query)
    assert result == [candidates[i] for i in expected]
