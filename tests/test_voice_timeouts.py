import asyncio
from types import SimpleNamespace


class _FakeUserState:
    def __init__(self):
        self.interrupted = False
        self.state = "processing"
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.state = "idle"
        self.interrupted = False


class _FakeSink:
    def __init__(self, user_state):
        self.user_state = user_state

    def _get_user(self, uid):
        return self.user_state

    def _reset_user(self, uid):
        self.user_state.reset()


def _make_session(monkeypatch):
    import voice.listener as listener

    member = SimpleNamespace(id=7, name="Ian", display_name="Ian")
    user_state = _FakeUserState()
    guild = SimpleNamespace(get_member=lambda uid: member)
    session = object.__new__(listener.GuildVoiceSession)
    session.guild = guild
    session.client = None
    session.sink = _FakeSink(user_state)
    session._voice_client = None
    session._pipeline_task = None
    session._pipeline_is_music = False
    session._interrupted_pipeline_uids = set()
    session._protected_pipeline_tasks = set()
    session._sessions = {}
    session.get_session = lambda current_member: SimpleNamespace(
        add=lambda role, content: None,
        get_history=lambda: [],
    )
    return listener, session, user_state


def test_stt_timeout_resets_voice_state(monkeypatch):
    listener, session, user_state = _make_session(monkeypatch)
    import voice.stt as stt

    async def slow_transcribe(wav_bytes):
        await asyncio.sleep(0.05)
        return "never returned"

    monkeypatch.setattr(listener, "STT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(stt, "transcribe", slow_transcribe)

    async def run():
        await session.on_speech_captured(7, b"wav")
        await session._pipeline_task

    asyncio.run(run())

    assert user_state.reset_calls >= 1
    assert session._pipeline_task is None


def test_command_timeout_resets_voice_state(monkeypatch):
    listener, session, user_state = _make_session(monkeypatch)
    import voice.handler as handler
    import voice.stt as stt

    async def fast_transcribe(wav_bytes):
        return "haz algo"

    async def slow_command(**kwargs):
        await asyncio.sleep(0.05)
        return "too late", False

    monkeypatch.setattr(listener, "VOICE_COMMAND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(stt, "transcribe", fast_transcribe)
    monkeypatch.setattr(handler, "handle_voice_command", slow_command)

    async def run():
        await session.on_speech_captured(7, b"wav")
        await session._pipeline_task

    asyncio.run(run())

    assert user_state.reset_calls >= 1
    assert session._pipeline_task is None


def test_crypto_error_detection_matches_packet_decryption_errors():
    from voice.listener import _is_crypto_error

    assert _is_crypto_error(RuntimeError("CryptoError decoding packet data"))
    assert _is_crypto_error(RuntimeError("failed to decrypt voice packet"))
    assert not _is_crypto_error(RuntimeError("audio buffer is empty"))
