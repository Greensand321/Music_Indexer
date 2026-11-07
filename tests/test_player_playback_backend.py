"""Playback backend seam (gui/playback_backend.py) — Qt-free.

PLY-01: python-vlc installed but libVLC missing makes ``import vlc`` raise a
non-ImportError; that used to crash window construction. ``create_backend``
must never raise and must explain why playback is unavailable.
"""
from __future__ import annotations

import enum
import sys
import types

import pytest

from gui import playback_backend as pb
from gui.playback_backend import PlaybackState


def _fake_vlc_module(instance_returns_none: bool = False) -> types.ModuleType:
    mod = types.ModuleType("vlc")
    calls: list = []

    class State(enum.Enum):
        NothingSpecial = 0
        Opening = 1
        Buffering = 2
        Playing = 3
        Paused = 4
        Stopped = 5
        Ended = 6
        Error = 7

    class Media:
        def __init__(self, path):
            self.path = path

    class MediaPlayer:
        def __init__(self):
            self.media = None
            self.state = State.NothingSpecial

        def set_media(self, m):
            calls.append(("set_media", m.path))
            self.media = m

        def play(self):
            calls.append(("play",))
            if self.media is None:
                return -1
            self.state = State.Playing
            return 0

        def set_pause(self, flag):
            calls.append(("set_pause", flag))
            self.state = State.Paused

        def pause(self):  # toggle — must not be used when set_pause exists
            calls.append(("pause-toggle",))

        def stop(self):
            self.state = State.Stopped

        def get_state(self):
            return self.state

        def is_playing(self):
            return self.state == State.Playing

        def get_length(self):
            return 5000

        def get_time(self):
            return 1200

        def set_time(self, t):
            calls.append(("set_time", t))

        def audio_set_volume(self, v):
            calls.append(("volume", v))

    class Instance:
        def __new__(cls, *args):
            if instance_returns_none:
                return None
            return super().__new__(cls)

        def media_new(self, path):
            return Media(path)

        def media_player_new(self):
            return MediaPlayer()

    mod.State = State
    mod.Instance = Instance
    mod.CALLS = calls
    return mod


def test_missing_python_vlc_gives_null_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "vlc", None)   # import vlc -> ImportError
    backend = pb.create_backend()
    assert isinstance(backend, pb.NullBackend)
    assert backend.available is False
    assert "python-vlc" in backend.unavailable_reason


def test_libvlc_missing_non_importerror_does_not_raise(monkeypatch, tmp_path):
    """python-vlc raises NotImplementedError('Cannot find libvlc lib') at import."""
    (tmp_path / "vlc.py").write_text("raise NotImplementedError('Cannot find libvlc lib')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "vlc", raising=False)
    backend = pb.create_backend()
    assert isinstance(backend, pb.NullBackend)
    assert "Cannot find libvlc lib" in backend.unavailable_reason
    monkeypatch.delitem(sys.modules, "vlc", raising=False)


def test_vlc_instance_none_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "vlc", _fake_vlc_module(instance_returns_none=True))
    backend = pb.create_backend()
    assert not backend.available
    assert "libVLC failed to start" in backend.unavailable_reason


def test_null_backend_never_pretends_to_play():
    b = pb.NullBackend("because")
    with pytest.raises(pb.BackendUnavailable):
        b.load("x.mp3")
    assert b.play() is False
    assert b.state() is PlaybackState.IDLE
    assert b.is_playing() is False
    assert b.time_ms() == -1 and b.length_ms() == -1


def test_vlc_backend_maps_states_and_uses_explicit_pause():
    mod = _fake_vlc_module()
    b = pb.VlcBackend(mod)
    assert b.available and b.name == "VLC"
    assert b.state() is PlaybackState.IDLE
    b.load("/music/a.flac")
    b.set_volume(55)
    assert b.play() is True
    assert b.state() is PlaybackState.PLAYING
    b.pause()
    assert ("set_pause", 1) in mod.CALLS and ("pause-toggle",) not in mod.CALLS
    assert b.state() is PlaybackState.PAUSED
    assert b.length_ms() == 5000 and b.time_ms() == 1200
    b.set_time_ms(-5)
    assert ("set_time", 0) in mod.CALLS
    # The volume is re-applied after play() (VLC drops it before the stream opens).
    assert mod.CALLS.count(("volume", 55)) == 2


def test_vlc_backend_play_without_media_reports_failure():
    b = pb.VlcBackend(_fake_vlc_module())
    assert b.play() is False
