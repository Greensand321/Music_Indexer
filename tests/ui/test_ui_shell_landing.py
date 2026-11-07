"""MosaicLanding: exit sequence, album-art scanner lifetime, diagnostics."""
from __future__ import annotations

import logging
import os
import time

import pytest
from PySide6 import QtCore

from gui.widgets import landing as landing_mod
from gui.widgets.landing import MosaicLanding
from gui.workers import job_registry
from shell_helpers import cta_button, make_album_tree, wait_until
from ui_harness import process_events

GEO = QtCore.QRect(0, 0, 1300, 860)


@pytest.fixture
def make_landing(qtbot):
    made = []

    def factory(saved: str = "", missing: str = "") -> MosaicLanding:
        landing = MosaicLanding(GEO, saved, missing_path=missing)
        qtbot.addWidget(landing)
        made.append(landing)
        return landing

    yield factory
    for landing in made:
        landing._stop_scanner()


@pytest.mark.motion(time_scale=1.0)
def test_initialize_during_logo_pause_still_exits(make_landing, qtbot):
    """DS-01: the logo-pause fly-in used to fire mid-scatter, stop the scatter's
    animations, and leave the landing on screen forever."""
    landing = make_landing()
    selected = []
    landing.library_selected.connect(selected.append)
    with qtbot.waitSignal(landing.faded_in, timeout=3000):
        landing.show_animated()
    qtbot.wait(330)                       # inside the 600 ms logo pause
    landing._accept("/some/library")
    assert not landing._fly_in_timer.isActive()
    with qtbot.waitSignal(landing.finished, timeout=4000):
        pass
    assert selected == ["/some/library"]
    assert landing._fly_in_grp is None    # the fly-in never started
    assert not landing.isVisible()


def test_accept_runs_once_and_never_snaps_tiles(make_landing, motion):
    """DS-23: a second Initialize / tile click mid-scatter used to snap every
    tile back to its grid slot and restart the scatter."""
    landing = make_landing()
    landing.show()
    selected = []
    landing.library_selected.connect(selected.append)
    before = [t.pos() for t in landing._tiles]
    landing._accept("/lib/a")
    group = landing._fly_out_grp
    # Tiles start the scatter from where they are (off-screen here: no fly-in yet).
    assert [t.pos() for t in landing._tiles] == before
    landing._accept("/lib/b")
    landing._on_tile_clicked("/lib/a/Album")
    assert landing._fly_out_grp is group
    motion.settle(3000)
    assert wait_until(lambda: bool(selected), 3000)
    process_events(50)
    assert selected == ["/lib/a"]


def test_exit_safety_timer_fades_out_if_scatter_is_interrupted(make_landing):
    landing = make_landing()
    landing.show()
    selected = []
    landing.library_selected.connect(selected.append)
    landing._accept("/lib")
    landing._fly_out_grp.stop()           # finished() will never be emitted
    assert wait_until(lambda: bool(selected), 3000)
    assert selected == ["/lib"]


class _SlowScandir:
    """os.scandir that is slow, so the scanner is still walking when we stop it."""

    def __init__(self, real, delay: float = 0.05):
        self.real = real
        self.delay = delay
        self.calls = 0

    def __call__(self, path):
        self.calls += 1
        time.sleep(self.delay)
        return self.real(path)


def _deep_library(root) -> str:
    root = str(root)
    for a in range(30):
        for b in range(4):
            os.makedirs(os.path.join(root, f"A{a}", f"B{b}", "CD1"), exist_ok=True)
    return root


def test_art_scanner_is_registered_and_stops_on_close(make_landing, tmp_path, monkeypatch, threads):
    """SHELL-02: closing the landing (Alt+F4) mid-scan aborted the process."""
    lib = _deep_library(tmp_path / "Lib")
    slow = _SlowScandir(os.scandir)
    monkeypatch.setattr(landing_mod.os, "scandir", slow)
    terminated = []
    monkeypatch.setattr(landing_mod._ArtScanner, "terminate", lambda self: terminated.append(self))

    landing = make_landing(saved=lib)
    scanner = landing._scanner
    assert wait_until(lambda: slow.calls > 0, 3000)
    assert scanner.isRunning()
    assert "Album-art scan" in [j.name for j in job_registry().jobs()]

    landing.show()
    landing.close()
    assert landing._scanner is None
    assert threads.wait_all(3000) == []          # the walk itself is interruptible
    assert slow.calls < 120                       # stopped well before walking everything
    assert terminated == []


def test_stopping_the_scanner_never_blocks_the_gui(make_landing, tmp_path, monkeypatch, threads):
    """SHELL-03: _done() used to wait(2000) on the GUI thread, then terminate()."""
    lib = _deep_library(tmp_path / "Lib")
    slow = _SlowScandir(os.scandir, delay=0.2)
    monkeypatch.setattr(landing_mod.os, "scandir", slow)
    terminated = []
    monkeypatch.setattr(landing_mod._ArtScanner, "terminate", lambda self: terminated.append(self))

    landing = make_landing(saved=lib)
    assert wait_until(lambda: slow.calls > 0, 3000)
    t0 = time.monotonic()
    landing._done()
    assert (time.monotonic() - t0) < 0.1
    assert threads.wait_all(3000) == []
    assert terminated == []


def test_scanner_drops_art_after_exit(make_landing, tmp_path):
    lib = make_album_tree(tmp_path / "Lib")
    landing = make_landing(saved=lib)
    scanner = landing._scanner
    landing._accept(lib)
    assert landing._scanner is None
    assert scanner._cancel.is_set()


def test_art_scan_summary_is_logged_not_printed(tmp_path, capfd, caplog, monkeypatch, qapp):
    """DS-26 / SHELL-26: no ``[ArtScan]`` line on stderr in a normal run."""
    lib = make_album_tree(tmp_path / "Lib", artists=1, albums=1, tracks=1)
    monkeypatch.setattr(landing_mod._ArtHistory, "_PATH", str(tmp_path / "hist.json"))
    scanner = landing_mod._ArtScanner(lib, 4)
    scanner.run()                                  # synchronously, on this thread
    assert "[ArtScan]" not in capfd.readouterr().err
    caplog.set_level(logging.DEBUG, logger=landing_mod.__name__)
    landing_mod._ArtScanner(lib, 4).run()
    assert any("[ArtScan]" in r.getMessage() for r in caplog.records)


def test_landing_reads_gradient_setting_once(make_landing, monkeypatch):
    import gui.widgets.gradient_bg as gb

    calls = []
    monkeypatch.setattr(gb, "_gradient_enabled", lambda: calls.append(1) or True)
    landing = make_landing()
    landing.show()
    for _ in range(5):
        landing.grab()
    assert len(calls) == 1


def test_missing_library_card(make_landing, tmp_path):
    landing = make_landing(missing=str(tmp_path / "gone"))
    assert cta_button(landing).text() == "Choose Library Folder"
    assert "not found" in landing._cta.tagline.text().lower()
    assert landing._scanner is None
    assert landing_mod._nearest_existing_dir(str(tmp_path / "gone" / "deeper")) == str(tmp_path)
