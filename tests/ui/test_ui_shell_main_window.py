"""Main window shell: library selection, stats, keyboard mode, settings, nav."""
from __future__ import annotations

import os
import time

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

import gui.main_window as mw
from gui.main_window import AlphaDEXWindow, _NAV_ORDER
from gui.widgets.sidebar import NAV_STRUCTURE
from shell_helpers import make_album_tree, wait_until, write_config
from ui_harness import process_events


@pytest.fixture
def fanouts(monkeypatch):
    calls: list[str] = []
    original = AlphaDEXWindow._update_library_stats

    def counting(self, path):
        calls.append(path)
        return original(self, path)

    monkeypatch.setattr(AlphaDEXWindow, "_update_library_stats", counting)
    return calls


# ── one validated library path (SHELL-04/05/08/09) ─────────────────────────

def test_window_does_not_apply_saved_library_on_construction(qtbot, tmp_path, threads, fanouts):
    lib = make_album_tree(tmp_path / "Lib")
    write_config(library_root=lib)
    window = AlphaDEXWindow()
    qtbot.addWidget(window)
    assert window._library_path == ""
    assert fanouts == []
    assert threads.started == []          # no Player scan, no stats scan yet


def test_same_library_twice_is_a_noop(main_window, tmp_path, threads, fanouts):
    lib = make_album_tree(tmp_path / "Lib")
    assert main_window.set_library(lib)
    assert main_window.set_library(lib + os.sep)       # same folder, other spelling
    main_window._top_bar.library_changed.emit(lib)    # Change Library → same folder
    assert fanouts == [os.path.normpath(lib)]
    assert threads.wait_all(10_000) == []


def test_missing_library_is_rejected_with_a_message(main_window, tmp_path, threads, dialogs, fanouts):
    lib = make_album_tree(tmp_path / "Lib")
    main_window.set_library(lib)
    assert threads.wait_all(10_000) == []
    gone = str(tmp_path / "gone")

    main_window._top_bar.library_changed.emit(gone)
    assert main_window.set_library(gone) is False

    assert main_window._library_path == os.path.normpath(lib)
    assert main_window._top_bar.library_path == os.path.normpath(lib)
    assert all(ws._library_path == os.path.normpath(lib) for ws in main_window._workspaces.values())
    warnings = dialogs.calls_of("warning")
    assert len(warnings) == 2 and "not found" in warnings[0].title.lower()
    import config
    assert config.load_config()["library_root"] == os.path.normpath(lib)
    assert fanouts == [os.path.normpath(lib)]


def test_stale_stats_are_discarded(main_window, tmp_path, threads, monkeypatch):
    """SHELL-05: A's slow scan finishing after B's used to show A's counts."""
    a = make_album_tree(tmp_path / "A", artists=1, albums=1, tracks=5)
    b = make_album_tree(tmp_path / "B", artists=2, albums=1, tracks=1)
    real = mw.library_stats

    def slow_for_a(path, should_stop=lambda: False):
        if os.path.normpath(path) == os.path.normpath(a):
            for _ in range(40):          # ~0.4 s, ignoring cancellation on purpose
                time.sleep(0.01)
        return real(path, lambda: False)

    monkeypatch.setattr(mw, "library_stats", slow_for_a)
    main_window.set_library(a)
    main_window.set_library(b)
    assert threads.wait_all(10_000) == []
    process_events(20)
    numbers = [n.text() for n, _u in main_window._top_bar._stat_widgets]
    assert numbers[0] == "2"            # B's two tracks, not A's five
    assert numbers[2] == "2"


def test_stats_worker_does_not_shadow_qthread_finished():
    assert "finished" not in mw._StatsWorker.__dict__
    assert "stats_ready" in mw._StatsWorker.__dict__


# ── library statistics (SHELL-06) ───────────────────────────────────────────

def _touch(path, size=100):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * size)


def test_library_stats_skip_reserved_folders_and_survive_broken_links(tmp_path):
    root = tmp_path / "Music"
    _touch(str(root / "Artist A" / "Album 1" / "01.mp3"))
    _touch(str(root / "Artist A" / "Album 2" / "01.flac"))
    _touch(str(root / "Artist B" / "02.mp3"))                  # one level deep
    _touch(str(root / "loose.mp3"))                            # root: no artist
    _touch(str(root / "Quarantine" / "X" / "Y" / "dupe.mp3"))
    _touch(str(root / "Trash" / "junk.mp3"))
    _touch(str(root / "Docs" / "a.mp3"))
    _touch(str(root / "Artist A" / "Album 1" / "cover.jpg"))
    try:
        os.symlink(str(root / "nowhere.mp3"), str(root / "Artist B" / "broken.mp3"))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    tracks, size, artists = mw.library_stats(str(root))
    assert tracks == 4
    assert size == 400
    assert artists == 2                                         # Artist A, Artist B


def test_library_stats_stop_early():
    assert mw.library_stats(os.getcwd(), lambda: True) is None


# ── keyboard mode scoping (SHELL-11 / PLY-10) ────────────────────────────────

@pytest.fixture
def kb(main_window, monkeypatch):
    player = main_window._workspaces["player"]
    hits: list[str] = []
    for name in ("kb_lib_down", "kb_lib_right", "kb_lib_left", "kb_lib_up",
                 "_preview_next", "play_next", "play_prev", "_restart_current"):
        monkeypatch.setattr(player, name, lambda n=name: hits.append(n))
    return hits


def test_keyboard_mode_filter_only_installed_while_on(main_window):
    assert not main_window._keyboard_filter_installed
    main_window.set_keyboard_mode_global(True)
    assert main_window._keyboard_filter_installed
    main_window.set_keyboard_mode_global(False)
    assert not main_window._keyboard_filter_installed


def test_keyboard_mode_arrows_drive_player_on_player_page(main_window, kb, qtbot):
    main_window.set_keyboard_mode_global(True)
    main_window._on_nav_changed("player")
    process_events(10)
    table = main_window._workspaces["player"]._lib_table
    qtbot.keyClick(table, QtCore.Qt.Key.Key_Left)
    qtbot.keyClick(main_window._sidebar, QtCore.Qt.Key.Key_Down)
    assert kb == ["kb_lib_left", "kb_lib_down"]


def test_keyboard_mode_leaves_text_inputs_alone(main_window, kb, qtbot):
    main_window.set_keyboard_mode_global(True)
    main_window._on_nav_changed("player")
    search = main_window._workspaces["player"]._search
    search.setText("abc")
    search.setCursorPosition(3)
    qtbot.keyClick(search, QtCore.Qt.Key.Key_Left)
    assert search.cursorPosition() == 2
    assert kb == []


def test_keyboard_mode_does_nothing_on_other_pages(main_window, kb, qtbot):
    main_window.set_keyboard_mode_global(True)
    main_window._on_nav_changed("indexer")
    process_events(10)
    indexer = main_window._workspaces["indexer"]
    spin = indexer.findChild(QtWidgets.QSpinBox)
    edit = indexer.findChild(QtWidgets.QLineEdit)
    button = indexer.findChild(QtWidgets.QPushButton)
    for widget in (spin, edit, button):
        if widget is not None:
            qtbot.keyClick(widget, QtCore.Qt.Key.Key_Left)
    assert kb == []
    # ...except from the Now Playing bar itself.
    qtbot.keyClick(main_window._now_playing_bar._kb_btn, QtCore.Qt.Key.Key_Right)
    assert kb == ["kb_lib_right"]


def test_keyboard_mode_ignores_modal_dialogs(main_window, kb, qtbot):
    main_window.set_keyboard_mode_global(True)
    main_window._on_nav_changed("player")
    dlg = QtWidgets.QDialog(main_window)
    dlg.setModal(True)
    edit = QtWidgets.QLineEdit("xyz", dlg)
    QtWidgets.QVBoxLayout(dlg).addWidget(edit)
    dlg.show()
    qtbot.waitExposed(dlg)
    edit.setCursorPosition(3)
    try:
        qtbot.keyClick(edit, QtCore.Qt.Key.Key_Left)
        qtbot.keyClick(dlg, QtCore.Qt.Key.Key_Left)
        assert edit.cursorPosition() == 2
        assert kb == []
    finally:
        dlg.close()
        dlg.deleteLater()


# ── navigation (SHELL-27, SHELL-25 hook) ────────────────────────────────────

def test_shortcut_order_is_the_sidebar_order():
    from gui.navigation import PAGE_BY_KEY

    sidebar_order = [i.key for s in NAV_STRUCTURE for i in s.items]
    assert _NAV_ORDER == sidebar_order
    assert set(_NAV_ORDER) == set(PAGE_BY_KEY)


def test_status_bar_uses_sidebar_labels(main_window):
    main_window._on_nav_changed("graph")
    assert main_window.statusBar().currentMessage() == "Sound Map · Map"
    main_window._on_nav_changed("player")
    assert main_window.statusBar().currentMessage() == "Player"


def test_on_activated_hook_runs_when_a_page_is_shown(main_window, monkeypatch):
    graph = main_window._workspaces["graph"]
    seen = []
    monkeypatch.setattr(graph, "on_activated", lambda: seen.append(1), raising=False)
    main_window._on_nav_changed("graph")
    assert seen == [1]


# ── dialogs owned by the window (SHELL-17, SHELL-09) ────────────────────────

def test_settings_open_as_a_page_not_a_dialog(main_window, dialogs, qtbot):
    """Settings is a page now: Ctrl+, and the top bar's Settings button go
    there, and no modal dialog is opened (the harness would fail on one)."""
    from gui.dialogs.settings_drawer import SettingsDrawer

    main_window._top_bar.settings_requested.emit()
    assert main_window.current_page() is main_window._pages["settings"]
    main_window.select_page("home")
    qtbot.keyClick(main_window, QtCore.Qt.Key.Key_Comma, QtCore.Qt.KeyboardModifier.ControlModifier)
    process_events(5)
    assert main_window.current_page() is main_window._pages["settings"]
    assert dialogs.calls_of("exec") == []
    assert main_window.findChildren(SettingsDrawer) == []


def test_theme_button_opens_settings_appearance(main_window):
    main_window._on_theme_requested()
    page = main_window._pages["settings"]
    assert main_window.current_page() is page
    assert page.current_section() == "appearance"
    panel = main_window._workspaces["settings"]
    assert panel.theme_picker.isVisible()


def test_settings_default_library_switches_once(main_window, tmp_path, threads, dialogs, fanouts):
    lib = make_album_tree(tmp_path / "Lib")
    main_window._on_settings()
    settings = main_window._workspaces["settings"]
    settings._library_entry.setText(lib)
    assert settings._on_save()
    assert main_window._library_path == os.path.normpath(lib)
    assert fanouts == [os.path.normpath(lib)]
    assert threads.wait_all(10_000) == []


def test_config_backup_is_reported_in_the_log(qtbot, tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "_last_backup", None)
    with open(config.CONFIG_PATH, "w", encoding="utf-8") as fh:
        fh.write('{"metadata_api_key": "SECRET", ')        # truncated
    config.save_config(config.load_config())
    window = AlphaDEXWindow()
    qtbot.addWidget(window)
    text = window._log_drawer._text.toPlainText()
    assert "could not be read" in text and ".corrupt-" in text


# ── small shell fixes ───────────────────────────────────────────────────────

def test_top_bar_path_colour_follows_library_state(main_window, tmp_path, threads):
    from gui.themes.manager import get_manager

    t = get_manager().current
    bar = main_window._top_bar
    assert t.text_secondary in bar._path_label.styleSheet()
    main_window.set_library(make_album_tree(tmp_path / "Lib"))
    assert t.text_primary in bar._path_label.styleSheet()
    assert threads.wait_all(10_000) == []


def test_theme_switch_never_refreshes_a_page_shadow_twice(main_window, monkeypatch):
    """SHELL-22 / DS-21: shadows used to be rebuilt twice per switch (window +
    base). Shadows are now painted from the live theme, so a switch needs at
    most one cheap repaint per page — never a second rebuild."""
    from gui.themes.manager import get_manager
    from gui.workspaces.base import WorkspaceBase

    calls = []
    original = WorkspaceBase.refresh_shadows
    monkeypatch.setattr(WorkspaceBase, "refresh_shadows",
                        lambda self: (calls.append(type(self).__name__), original(self)))
    get_manager().apply("pearl")
    per_page = {name: calls.count(name) for name in set(calls)}
    assert all(n <= 1 for n in per_page.values()), per_page
