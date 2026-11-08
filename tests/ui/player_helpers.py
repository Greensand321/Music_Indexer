"""Test helpers for the Player area: a recording fake playback backend.

Usage::

    from player_helpers import FakeBackend, make_player

    ws, fake = make_player(qtbot)            # PlayerWorkspace with a FakeBackend
    ws.load_tracks_and_play(paths)
    assert fake.loaded == paths[0]
    fake.finish()                            # the track reaches its end
    ws._tick_position()                      # what the 400 ms timer would do
"""
from __future__ import annotations

from gui.playback_backend import PlaybackBackend, PlaybackState


class FakeBackend(PlaybackBackend):
    """Plays nothing; records every command; state is driven by the test."""

    name = "Fake"
    available = True

    def __init__(self, length_ms: int = 180_000) -> None:
        self.calls: list[tuple] = []
        self.loaded: str | None = None
        self._state = PlaybackState.IDLE
        self._time = 0
        self.length = length_ms
        self.volume: int | None = None
        self.fail_paths: set[str] = set()   # load() raises for these
        self.released = False

    # ── PlaybackBackend ───────────────────────────────────────────────────
    def load(self, path: str) -> None:
        self.calls.append(("load", path))
        if path in self.fail_paths:
            raise RuntimeError("cannot open")
        self.loaded = path
        self._state = PlaybackState.IDLE
        self._time = 0

    def play(self) -> bool:
        self.calls.append(("play", self.loaded))
        if self.loaded is None:
            return False
        self._state = PlaybackState.PLAYING
        return True

    def pause(self) -> None:
        self.calls.append(("pause",))
        if self._state is PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        self.calls.append(("stop",))
        if self.loaded is not None:
            self._state = PlaybackState.STOPPED

    def state(self) -> PlaybackState:
        return self._state

    def length_ms(self) -> int:
        return self.length if self.loaded else -1

    def time_ms(self) -> int:
        return self._time if self.loaded else -1

    def set_time_ms(self, ms: int) -> None:
        self.calls.append(("set_time", ms))
        self._time = ms

    def set_volume(self, value: int) -> None:
        self.calls.append(("volume", value))
        self.volume = value

    def release(self) -> None:
        self.released = True

    # ── test controls ─────────────────────────────────────────────────────
    def finish(self) -> None:
        self._state = PlaybackState.ENDED

    def fail(self) -> None:
        self._state = PlaybackState.ERROR

    def plays(self) -> list[str]:
        """Paths passed to load() whose play() followed, in order."""
        out, last = [], None
        for call in self.calls:
            if call[0] == "load":
                last = call[1]
            elif call[0] == "play" and call[1] == last and last is not None:
                out.append(last)
                last = None
        return out

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c[0] == name)


def make_player(qtbot, library_path: str = "", backend: PlaybackBackend | None = None, show=True):
    from gui.workspaces.player import PlayerWorkspace

    fake = backend if backend is not None else FakeBackend()
    ws = PlayerWorkspace(library_path=library_path, backend=fake)
    qtbot.addWidget(ws)
    if show:
        ws.resize(1100, 760)
        ws.show()
    return ws, fake


def wire_bar(ws, bar) -> None:
    """Connect ``bar`` (a NowPlayingBar) to ``ws`` the way AlphaDEXWindow
    does (gui/main_window.py, "Wire PlayerWorkspace ↔ NowPlayingBar"); the
    capsule is the Player's only transport, so transport tests click it."""
    ws.now_playing_changed.connect(bar.update_now_playing)
    ws.playback_state_changed.connect(bar.set_playing)
    ws.position_changed.connect(bar.update_position)
    bar.play_pause_requested.connect(ws._on_play_pause)
    bar.next_requested.connect(ws.play_next)
    bar.prev_requested.connect(ws.play_prev)
    bar.seek_requested.connect(ws.seek_to_ms)
    bar.volume_changed.connect(ws.set_volume)
    ws.volume_changed.connect(bar.set_volume)
    bar.shuffle_requested.connect(ws.set_shuffle)
    bar.repeat_requested.connect(ws.set_repeat)
    ws.shuffle_changed.connect(bar.set_shuffle)
    ws.repeat_changed.connect(bar.set_repeat)
    ws.transport_changed.connect(bar.set_transport_state)
    ws.transport_pinned.connect(bar.set_pinned)
    bar.set_transport_state(*ws.transport_state())


def make_player_with_bar(qtbot, library_path: str = "", backend: PlaybackBackend | None = None,
                         show=True):
    """``(ws, fake, bar)``: a PlayerWorkspace and a NowPlayingBar wired to it."""
    from gui.widgets.now_playing_bar import NowPlayingBar

    bar = NowPlayingBar()
    qtbot.addWidget(bar)
    bar.resize(880, NowPlayingBar.HEIGHT)
    fake = backend if backend is not None else FakeBackend()
    from gui.workspaces.player import PlayerWorkspace
    ws = PlayerWorkspace(library_path=library_path, backend=fake)
    qtbot.addWidget(ws)
    wire_bar(ws, bar)
    if show:
        ws.resize(1100, 760)
        ws.show()
    return ws, fake, bar


def wait_scan(ws, threads, timeout_ms: int = 15_000) -> None:
    """Wait for the library scan (and any tag/art workers) to finish."""
    assert threads.wait_all(timeout_ms) == []
    assert ws._scan_finished


def make_tracks(tmp_path, n: int = 4, ext: str = ".mp3") -> list[str]:
    """``n`` dummy (untagged, undecodable) audio files — enough for queue logic."""
    paths = []
    for i in range(n):
        p = tmp_path / f"{i + 1:02d} - Track {i + 1}{ext}"
        p.write_bytes(b"\0" * 64)
        paths.append(str(p))
    return paths
