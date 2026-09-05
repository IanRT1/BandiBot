from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_private_context_files_are_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "data/instructions.txt" in gitignore
    assert "data/server_info.txt" in gitignore


def test_example_context_files_do_not_contain_private_member_lore():
    instructions = (ROOT / "data/instructions.example.txt").read_text(encoding="utf-8")
    server_info = (ROOT / "data/server_info.example.txt").read_text(encoding="utf-8")

    private_names = ("DJ Bonk", "Trabis", "Poyo", "Beyra", "Bandía")
    for name in private_names:
        assert name not in instructions
        assert name not in server_info


def test_test_cache_directories_are_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "__pycache__/" in gitignore
    assert ".pytest_cache/" in gitignore
    assert "pytest-cache-files*/" in gitignore
