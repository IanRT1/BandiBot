from core.preflight import run_preflight


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
