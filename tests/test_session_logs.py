"""Offline tests for the two-file session log rotation."""

from core.session_logs import rotate_session_log


def test_rotation_keeps_current_and_previous_session_only(tmp_path):
    current = tmp_path / "session.log"
    previous = tmp_path / "session.previous.log"

    current.write_text("first run", encoding="utf-8")
    assert rotate_session_log(current) == previous
    assert previous.read_text(encoding="utf-8") == "first run"
    assert not current.exists()

    current.write_text("second run", encoding="utf-8")
    rotate_session_log(current)
    assert previous.read_text(encoding="utf-8") == "second run"


def test_rotation_is_safe_when_no_current_log_exists(tmp_path):
    current = tmp_path / "session.log"

    previous = rotate_session_log(current)

    assert previous == tmp_path / "session.previous.log"
    assert not previous.exists()
