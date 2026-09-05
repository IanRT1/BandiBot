from collections import deque
from types import SimpleNamespace

from music.player import GuildPlayer
from music.tracks import Track


def _track(title):
    return Track(
        title=title,
        stream_url="https://example.test/stream",
        requested_by="Ian",
        webpage_url="https://example.test/video",
    )


def test_resolved_track_is_deferred_while_standalone_audio_is_playing(monkeypatch):
    player = GuildPlayer(SimpleNamespace(name="Test Guild"))
    player.voice_client = SimpleNamespace(
        is_connected=lambda: True,
        is_playing=lambda: True,
    )
    player.queue = deque()
    scheduled = []
    monkeypatch.setattr(player, "_schedule_start_when_free", lambda: scheduled.append(True))

    track = _track("Resolved song")
    player._play_resolved(track)

    assert player.current is None
    assert list(player.queue) == [track]
    assert scheduled == [True]


def test_failed_discord_play_restores_track_and_clears_current(monkeypatch):
    import music.player as music_player

    class FakeVoiceClient:
        client = SimpleNamespace(loop=None)

        def is_connected(self):
            return True

        def is_playing(self):
            return True

        def play(self, source, after):
            raise RuntimeError("Already playing audio.")

    class FakeMixerSource:
        def __init__(self, source, clip_buffer=None):
            pass

    player = GuildPlayer(SimpleNamespace(name="Test Guild"))
    player.voice_client = FakeVoiceClient()
    player.queue = deque()
    scheduled = []
    monkeypatch.setattr(player, "_schedule_start_when_free", lambda: scheduled.append(True))
    monkeypatch.setattr(music_player, "_ffmpeg_before_options", lambda track: "")
    monkeypatch.setattr(music_player.discord, "FFmpegPCMAudio", lambda *args, **kwargs: object())
    monkeypatch.setattr(music_player.discord, "PCMVolumeTransformer", lambda source, volume: source)

    import voice.tts as tts
    monkeypatch.setattr(tts, "MixerSource", FakeMixerSource)
    track = _track("Failed start")
    player._play_resolved(track)

    assert player.current is None
    assert list(player.queue) == [track]
    assert scheduled == [True]
