import pytest

from core.settings import load_settings


def test_missing_file_and_partial_settings(tmp_path):
    path = tmp_path / "config.toml"
    assert load_settings(path)["music"]["pending_limit"] == 32
    path.write_text('[logging]\nlevel = "debug"\nsensitive_content = false\n')
    settings = load_settings(path)
    assert settings["logging"] == {"level": "DEBUG", "sensitive_content": False}
    assert settings["music"]["prepare_concurrency"] == 3


@pytest.mark.parametrize("content", [
    '[music]\npending_limit = 0',
    '[music]\nprepare_concurrency = 1.5',
    '[music]\noperation_timeout_seconds = nan',
    '[music]\noperation_timeout_seconds = true',
    '[music]\npending_limt = 3',
    '[logging]\nlevel = "verbose"',
    '[logging]\nsensitive_content = "false"',
    '[unknown]\nvalue = 1',
    'invalid toml',
])
def test_invalid_settings_fail_with_file_path(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content)
    with pytest.raises(ValueError, match="config.toml"):
        load_settings(path)


def test_environment_does_not_override_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MUSIC_PENDING_LIMIT", "1")
    path = tmp_path / "config.toml"
    path.write_text('[music]\npending_limit = 12\n[logging]\nlevel = "WARNING"')
    settings = load_settings(path)
    assert settings["music"]["pending_limit"] == 12
    assert settings["logging"]["level"] == "WARNING"
