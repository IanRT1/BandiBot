from core.preflight import run_preflight
from unittest.mock import Mock


def _valid_environment():
    return {
        "DISCORD_TOKEN": "discord",
        "OPENAI_API_KEY": "openai",
        "DEEPGRAM_API_KEY": "deepgram",
        "TTS_PROVIDER": "kokoro",
    }


def test_preflight_passes_with_required_configuration(monkeypatch):
    monkeypatch.setattr("core.preflight.shutil.which", lambda _: "available")
    monkeypatch.setattr("core.preflight.Path.is_file", lambda _: True)

    result = run_preflight(environ=_valid_environment())

    assert result.ok is True
    assert result.errors == ()
    assert "ffmpeg=ok" in result.summary
    assert "context=ok" in result.summary


def test_preflight_reports_missing_required_configuration(monkeypatch):
    environment = _valid_environment()
    environment.pop("OPENAI_API_KEY")
    monkeypatch.setattr("core.preflight.Path.is_file", lambda _: True)

    result = run_preflight(environ=environment)

    assert result.ok is False
    assert any("OPENAI_API_KEY" in error for error in result.errors)


def test_preflight_keeps_optional_dependency_failures_as_warnings(monkeypatch):
    monkeypatch.setattr("core.preflight.shutil.which", lambda _: None)
    monkeypatch.setattr("core.preflight.Path.is_file", lambda _: True)

    result = run_preflight(environ=_valid_environment())

    assert result.ok is True
    assert any("ffmpeg" in warning for warning in result.warnings)
    assert any("node" in warning for warning in result.warnings)


def test_preflight_includes_warmup_in_single_summary(monkeypatch, caplog):
    monkeypatch.setattr("core.preflight.shutil.which", lambda _: "available")
    monkeypatch.setattr("core.preflight.Path.is_file", lambda _: True)
    warmup = Mock(return_value="ready")
    with caplog.at_level("INFO"):
        result = run_preflight(environ=_valid_environment(), warmup=warmup)
    warmup.assert_called_once_with()
    assert any(item.startswith("retrieval=ready (") for item in result.summary)
    assert len(caplog.messages) == 1
    assert "retrieval=ready" in caplog.messages[0]


def test_preflight_skips_warmup_when_required_checks_fail(monkeypatch):
    monkeypatch.setattr("core.preflight.Path.is_file", lambda _: True)
    warmup = Mock()
    assert not run_preflight(environ={}, warmup=warmup).ok
    warmup.assert_not_called()


def test_preflight_warmup_failure_is_optional(monkeypatch):
    monkeypatch.setattr("core.preflight.Path.is_file", lambda _: True)
    result = run_preflight(
        environ=_valid_environment(), warmup=Mock(side_effect=RuntimeError("unavailable")),
    )
    assert result.ok
    assert any(item.startswith("retrieval=fallback (") for item in result.summary)
