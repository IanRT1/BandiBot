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


def test_interrupted_speech_logs_usage_and_passes_prior_interruption(monkeypatch, caplog):
    import logging
    import voice.handler as handler
    import voice.stt as stt
    import voice.tts as tts
    from core.interaction_logging import record_token_usage, record_usage

    _, session, _ = _make_session(monkeypatch)
    session.client = SimpleNamespace(user=SimpleNamespace(display_name="BandiBot"))
    session.clip_buffer = None
    session._speech_interrupted_for = {7}

    async def run():
        speaking = asyncio.Event()

        async def transcribe(_):
            record_usage("deepgram", "audio_seconds", 2)
            return "Stop"

        async def command(**kwargs):
            assert kwargs["speech_was_interrupted"] is True
            record_token_usage("openai", 100)
            return "Stopped.", False

        async def speak(*args, **kwargs):
            record_usage("elevenlabs", "credits", 10)
            speaking.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(stt, "transcribe", transcribe)
        monkeypatch.setattr(handler, "handle_voice_command", command)
        monkeypatch.setattr(tts, "speak", speak)
        await session.on_speech_captured(7, b"wav")
        task = session._pipeline_task
        await asyncio.wait_for(speaking.wait(), 1)
        task.cancel()
        await task

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    summaries = [message for message in caplog.messages if "total)" in message]
    assert len(summaries) == 1
    assert "<- interrupted" in summaries[0]
    assert "openai=100 tokens | elevenlabs=10 credits | deepgram=2.0s audio" in summaries[0]
    assert not session._speech_interrupted_for


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


def test_crypto_packet_log_filter_suppresses_individual_events_and_reports_burst(caplog):
    import logging

    from voice.listener import _CryptoPacketLogFilter

    current_time = [0.0]
    packet_filter = _CryptoPacketLogFilter(clock=lambda: current_time[0])
    records = [
        logging.LogRecord(
            "discord.ext.voice_recv.reader",
            logging.ERROR,
            __file__,
            1,
            "CryptoError decoding packet data",
            (),
            None,
        )
        for _ in range(5)
    ]

    with caplog.at_level(logging.WARNING, logger="voice.listener"):
        results = [packet_filter.filter(record) for record in records]

    assert results == [False] * 5
    warnings = [record for record in caplog.records if "packet decryption errors" in record.message]
    assert len(warnings) == 1
    assert "5 Discord voice packet decryption errors" in warnings[0].message


def test_crypto_packet_log_filter_reports_again_after_quiet_window(caplog):
    import logging

    from voice.listener import _CryptoPacketLogFilter

    current_time = [0.0]
    packet_filter = _CryptoPacketLogFilter(clock=lambda: current_time[0])

    def record():
        return logging.LogRecord(
            "discord.ext.voice_recv.reader",
            logging.ERROR,
            __file__,
            1,
            "CryptoError decoding packet data",
            (),
            None,
        )

    with caplog.at_level(logging.WARNING, logger="voice.listener"):
        for _ in range(5):
            packet_filter.filter(record())
        current_time[0] = 11.0
        for _ in range(5):
            packet_filter.filter(record())

    warnings = [record for record in caplog.records if "packet decryption errors" in record.message]
    assert len(warnings) == 2
