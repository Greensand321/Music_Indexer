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

from gui import navigation
from gui.main_window import _NAV_ORDER, _WORKSPACE_MAP
from gui.themes.tokens import THEMES

WORKSPACE_KEYS = list(_WORKSPACE_MAP)           # panels
PAGE_KEYS = [p.key for p in navigation.PAGES]   # sidebar pages
SECTION_KEYS = [f"{p.key}/{s.key}" for p in navigation.PAGES for s in p.sections]


def _paint(widget: QtWidgets.QWidget) -> QtGui.QPixmap:
    """Force a full synchronous paint of ``widget`` (exceptions surface here)."""
    pixmap = widget.grab()
    assert not pixmap.isNull()
    return pixmap


def test_window_builds_every_workspace(main_window):
    assert set(main_window._workspaces) == set(WORKSPACE_KEYS)
    for key, ws in main_window._workspaces.items():
        assert isinstance(ws, _WORKSPACE_MAP[key])
    # Every panel is used by a page, and every page is built.
    assert set(navigation.panel_keys()) == set(WORKSPACE_KEYS)
    assert list(main_window._pages) == PAGE_KEYS
    # The first page shown is Home.
    assert main_window._stack.currentWidget() is main_window._pages["home"]
    assert main_window.current_panel() is main_window._workspaces["home"]


def test_navigate_to_and_paint_every_workspace(main_window):
    """Every old panel key still works: it shows the page and section that
    hold that panel."""
    for key in WORKSPACE_KEYS:
        main_window._on_nav_changed(key)
        process_events(10)
        ws = main_window._workspaces[key]
        assert main_window.current_panel() is ws, key
        assert main_window._stack.currentWidget() is main_window.current_page(), key
        assert ws.isVisible(), key
        _paint(main_window)


# Pages and sections the information architecture added (the rest were
# workspaces before, and the theme tests already cover them).
NEW_SECTION_KEYS = [
    "home/home", "file_cleanup/file_cleanup", "exports/exports", "activity/activity",
    "duplicates/pair_review", "duplicates/fuzzy_finder", "duplicates/scan_engine",
    "duplicates/bucketing", "metadata/fill_genres", "metadata/normalize_genres",
    "sound_map/analyse", "sound_map/map", "mirror/verify",
    "settings/services", "settings/general", "settings/appearance",
]


def test_new_section_keys_exist():
    assert set(NEW_SECTION_KEYS) <= set(SECTION_KEYS)


@pytest.mark.parametrize("theme", ["midnight", "pearl"])
def test_every_page_and_section_builds_and_paints(main_window, theme):
    _paint_sections(main_window, theme, SECTION_KEYS)


@pytest.mark.parametrize("theme", [t for t in THEMES if t not in ("midnight", "pearl")])
def test_new_pages_paint_in_every_theme(main_window, theme):
    _paint_sections(main_window, theme, NEW_SECTION_KEYS)


def _paint_sections(main_window, theme, keys):
    from gui.themes.manager import get_manager

    get_manager().apply(theme)
    for key in keys:
        main_window._on_nav_changed(key)
        process_events(5)
        page_key, section_key = key.split("/")
        page = main_window._pages[page_key]
        assert main_window.current_page() is page, key
        assert page.current_section() == section_key, key
        assert page.current_panel().isVisible(), key
        _paint(main_window)


def test_unknown_navigation_key_is_ignored(main_window):
    before = main_window._stack.currentWidget()
    main_window._on_nav_changed("does_not_exist")
    assert main_window._stack.currentWidget() is before


def test_ctrl_number_shortcuts_follow_workspace_order(main_window, qtbot):
    assert _NAV_ORDER == PAGE_KEYS
    for index, key in enumerate(PAGE_KEYS[:9], start=1):
        qtbot.keyClick(main_window, QtCore.Qt.Key(QtCore.Qt.Key.Key_0.value + index),
                       QtCore.Qt.KeyboardModifier.ControlModifier)
        process_events(5)
        assert main_window._stack.currentWidget() is main_window._pages[key], key


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
