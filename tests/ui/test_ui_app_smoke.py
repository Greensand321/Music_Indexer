"""App-level smoke tests: the whole Qt window, every workspace, every theme.

These are deliberately broad. They catch the classes of bug that make an app
feel unstable — exceptions inside paint events and slots (pytest-qt fails the
test on any exception raised in a Qt virtual or slot), cross-thread Qt
warnings, threads left running, dialogs nobody planned for — across every
page, before any page-specific test exists.
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from ui_harness import process_events

from gui.main_window import _WORKSPACE_MAP
from gui.themes.tokens import THEMES

WORKSPACE_KEYS = list(_WORKSPACE_MAP)


def _paint(widget: QtWidgets.QWidget) -> QtGui.QPixmap:
    """Force a full synchronous paint of ``widget`` (exceptions surface here)."""
    pixmap = widget.grab()
    assert not pixmap.isNull()
    return pixmap


def test_window_builds_every_workspace(main_window):
    assert set(main_window._workspaces) == set(WORKSPACE_KEYS)
    for key, ws in main_window._workspaces.items():
        assert isinstance(ws, _WORKSPACE_MAP[key])
    # The first page shown is the Indexer.
    assert main_window._stack.currentWidget() is main_window._workspaces["indexer"]


def test_navigate_to_and_paint_every_workspace(main_window):
    for key in WORKSPACE_KEYS:
        main_window._on_nav_changed(key)
        process_events(10)
        ws = main_window._workspaces[key]
        assert main_window._stack.currentWidget() is ws, key
        assert ws.isVisible(), key
        _paint(main_window)


def test_unknown_navigation_key_is_ignored(main_window):
    before = main_window._stack.currentWidget()
    main_window._on_nav_changed("does_not_exist")
    assert main_window._stack.currentWidget() is before


def test_ctrl_number_shortcuts_follow_workspace_order(main_window, qtbot):
    for index, key in enumerate(WORKSPACE_KEYS[:9], start=1):
        qtbot.keyClick(main_window, QtCore.Qt.Key(QtCore.Qt.Key.Key_0.value + index),
                       QtCore.Qt.KeyboardModifier.ControlModifier)
        process_events(5)
        assert main_window._stack.currentWidget() is main_window._workspaces[key], key


def test_library_change_reaches_every_workspace(main_window, library, threads):
    root = str(library.root)
    main_window.set_library(root)
    assert threads.wait_all(10_000) == []
    for key, ws in main_window._workspaces.items():
        assert ws._library_path == root, key

    import config
    assert config.load_config().get("library_root") == root


def test_every_theme_applies_and_paints(main_window):
    from gui.themes.manager import get_manager

    mgr = get_manager()
    seen = []
    mgr.theme_changed.connect(lambda tokens: seen.append(tokens.key))
    for key in THEMES:
        mgr.apply(key)
        process_events(5)
        assert mgr.current.key == key
        _paint(main_window)
    assert seen == list(THEMES)


@pytest.mark.parametrize("key", WORKSPACE_KEYS)
def test_workspace_standalone_with_library(make_workspace, library, threads, key):
    """Each page must survive construction with a library, a library switch,
    a switch back, and painting — on its own, outside the main window."""
    ws = make_workspace(key, library_path=library.root)
    _paint(ws)
    other = library.root.parent / "Other Library"
    other.mkdir()
    ws.set_library_path(str(other))
    ws.set_library_path(str(library.root))
    process_events(20)
    _paint(ws)
    assert threads.wait_all(10_000) == []
