"""Startup sequence: landing → deferred main window → cross-fade → library.

Drives ``alpha_dex_gui.build_startup`` (everything ``main()`` does except
``app.exec()``) through the paths users hit: Initialize, a very fast click,
an album-tile click, no saved library, a saved library that has gone missing,
and a main window that fails to build.
"""
from __future__ import annotations

import os

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

import alpha_dex_gui
from gui.main_window import AlphaDEXWindow
from gui.widgets import landing as landing_mod
from shell_helpers import cta_button, finish_startup, make_album_tree, wait_until, write_config
from ui_harness import process_events


@pytest.fixture
def count_library_fanouts(monkeypatch):
    """Count real library applications (each one starts one stats scan)."""
    calls: list[str] = []
    original = AlphaDEXWindow._update_library_stats

    def counting(self, path):
        calls.append(path)
        return original(self, path)

    monkeypatch.setattr(AlphaDEXWindow, "_update_library_stats", counting)
    return calls


@pytest.fixture
def app(qapp):
    return qapp


def _start(app, **kwargs):
    return alpha_dex_gui.build_startup(app, **kwargs)


def test_initialize_builds_window_after_first_frame_and_applies_library_once(
    app, tmp_path, threads, count_library_fanouts
):
    lib = make_album_tree(tmp_path / "Lib")
    write_config(library_root=lib)

    startup = _start(app)
    try:
        # SHELL-15: the main window is not built before the landing's first frames.
        assert startup.window is None
        assert startup.landing.isVisible()
        assert cta_button(startup.landing).text() == "Initialize"

        assert wait_until(lambda: startup.window is not None)
        # SHELL-04: building the window applies no library on its own.
        assert startup.window._library_path == ""
        assert count_library_fanouts == []

        cta_button(startup.landing).click()
        assert wait_until(lambda: startup._library_applied, 5000)

        assert startup.window.isVisible()
        assert not startup.landing.isVisible()
        assert startup.window.windowOpacity() == pytest.approx(1.0)
        assert startup.window._library_path == os.path.normpath(lib)
        assert count_library_fanouts == [os.path.normpath(lib)]   # once, not twice
        process_events(50)
        assert count_library_fanouts == [os.path.normpath(lib)]
    finally:
        finish_startup(startup, threads)


def test_fast_click_before_window_exists_builds_it_on_demand(
    app, tmp_path, threads, count_library_fanouts
):
    lib = make_album_tree(tmp_path / "Lib")
    write_config(library_root=lib)

    startup = _start(app)
    try:
        assert startup.window is None
        cta_button(startup.landing).click()          # before any event is processed
        assert wait_until(lambda: startup._library_applied, 8000)
        assert startup.window is not None and startup.window.isVisible()
        assert count_library_fanouts == [os.path.normpath(lib)]
    finally:
        finish_startup(startup, threads)


def test_tile_click_plays_that_folder_then_applies_library(
    app, tmp_path, threads, monkeypatch, count_library_fanouts
):
    lib = make_album_tree(tmp_path / "Lib")
    write_config(library_root=lib)
    order: list[tuple[str, str]] = []
    monkeypatch.setattr(AlphaDEXWindow, "play_directory", lambda self, d: order.append(("play", d)))
    original_set = AlphaDEXWindow.set_library

    def recording_set(self, path):
        order.append(("library", path))
        return original_set(self, path)

    monkeypatch.setattr(AlphaDEXWindow, "set_library", recording_set)

    startup = _start(app)
    try:
        assert wait_until(lambda: startup.window is not None)
        tiles = startup.landing._tiles
        first = os.path.join(lib, "Artist0", "Album0")
        second = os.path.join(lib, "Artist1", "Album1")
        tiles[0].set_dirpath(first)
        tiles[1].set_dirpath(second)
        QtTest.QTest.mouseClick(tiles[0], QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(tiles[1], QtCore.Qt.MouseButton.LeftButton)  # mid-scatter: ignored

        assert wait_until(lambda: startup._library_applied, 5000)
        assert order == [("play", first), ("library", lib)]
        assert len(count_library_fanouts) == 1
    finally:
        finish_startup(startup, threads)


def test_no_saved_library_offers_folder_choice(app, tmp_path, threads, dialogs):
    lib = make_album_tree(tmp_path / "Chosen")
    startup = _start(app)
    try:
        assert cta_button(startup.landing).text() == "Choose Library Folder"
        assert startup.landing._scanner is None      # nothing to scan yet

        dialogs.on_exec(QtWidgets.QDialog.DialogCode.Rejected)   # cancel: stay on landing
        cta_button(startup.landing).click()
        process_events(50)
        assert startup.landing.isVisible() and not startup.landing._exiting

        dialogs.accept_file_dialog(lib)
        cta_button(startup.landing).click()
        assert wait_until(lambda: startup._library_applied, 8000)
        assert startup.window._library_path == os.path.normpath(lib)
        import config
        assert config.load_config()["library_root"] == os.path.normpath(lib)
    finally:
        finish_startup(startup, threads)


def test_missing_saved_library_says_so_and_is_not_applied(app, tmp_path, threads, dialogs):
    gone = str(tmp_path / "Unplugged" / "Music")
    write_config(library_root=gone)

    startup = _start(app)
    try:
        landing = startup.landing
        assert cta_button(landing).text() == "Choose Library Folder"
        assert "not found" in landing._cta.tagline.text().lower()
        assert gone in landing._cta.tagline.toolTip()
        assert landing._scanner is None

        seen_start_dirs = []

        def cancel(dlg):
            seen_start_dirs.append(dlg.directory().absolutePath())
            return QtWidgets.QDialog.DialogCode.Rejected

        dialogs.on_exec(cancel)
        cta_button(landing).click()
        process_events(50)
        assert landing.isVisible()
        # The picker opens at the nearest folder that still exists.
        assert seen_start_dirs and os.path.normpath(seen_start_dirs[0]) == os.path.normpath(str(tmp_path))
        assert wait_until(lambda: startup.window is not None)
        assert startup.window._library_path == ""
        import config
        assert config.load_config()["library_root"] == gone   # not rewritten
    finally:
        finish_startup(startup, threads)


def test_window_build_failure_shows_error_and_exits(app, tmp_path, threads, dialogs):
    lib = make_album_tree(tmp_path / "Lib")
    exits: list[int] = []

    def broken_window():
        raise RuntimeError("Cannot find libvlc lib")

    startup = alpha_dex_gui.build_startup(
        app, saved_library=lib, window_factory=broken_window, on_fatal=exits.append
    )
    try:
        assert wait_until(lambda: bool(exits), 5000)
        assert exits == [1]
        assert startup.exit_code == 1
        assert startup.window is None
        assert "libvlc" in startup.error
        crit = dialogs.calls_of("critical")
        assert crit and "could not start" in crit[0].title
        assert not startup.landing.isVisible()

        # A later hand-off (e.g. a click that was already in flight) does not
        # retry, re-poll or show a second dialog.
        startup._landing_to_main(lib)
        process_events(50)
        assert exits == [1]
        assert len(dialogs.calls_of("critical")) == 1
    finally:
        finish_startup(startup, threads)


def test_window_build_failure_on_fast_click(app, tmp_path, threads, dialogs):
    lib = make_album_tree(tmp_path / "Lib")
    exits: list[int] = []

    def broken_window():
        raise ValueError("boom")

    startup = alpha_dex_gui.build_startup(
        app, saved_library=lib, window_factory=broken_window, on_fatal=exits.append
    )
    try:
        cta_button(startup.landing).click()
        assert wait_until(lambda: bool(exits), 5000)
        process_events(100)
        assert exits == [1]
        assert len(dialogs.calls_of("critical")) == 1
    finally:
        finish_startup(startup, threads)


def test_main_installs_freeze_support_first():
    src = open(alpha_dex_gui.__file__, encoding="utf-8").read()
    head = src.split("import os", 1)[0]
    assert "multiprocessing.freeze_support()" in head
