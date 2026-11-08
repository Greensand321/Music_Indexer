"""The Organize pages and Exports in the Liquid Glass look (redesign R2).

Library Sync used to be wider than the window at 1300 px and scroll sideways;
these tests pin that it fits the default and the minimum window, empty and
with results. They also cover what the restyle added: step pills that follow
the real state, stat tiles for the counts, empty states before a run, glyph
buttons (no emoji) with accessible names, glass item views, and the per-item
flags still reachable from the right-click menu.
"""
from __future__ import annotations

import unicodedata

import pytest
from PySide6 import QtCore, QtWidgets

from library_helpers import build_incoming, install_fake_fingerprints, wait_idle
from ui_harness import process_events

SYNC_WORKERS = ("_worker", "_build_worker", "_exec_worker")
SIZES = [(900, 600), (1300, 860)]


def _show(main_window, key, size):
    main_window.resize(*size)
    main_window._on_nav_changed(key)
    process_events(60)
    return main_window._workspaces[key]


def _hscroll(ws) -> int:
    process_events(30)
    return ws._scroll_area.horizontalScrollBar().maximum()


def _sync_with_results(main_window, library, tmp_path, monkeypatch, qtbot):
    install_fake_fingerprints(monkeypatch)
    incoming = tmp_path / "an incoming folder with a rather long name" / "and a deep path"
    build_incoming(incoming)
    main_window.set_library(str(library.root))
    ws = main_window._workspaces["library_sync"]
    ws._incoming_entry.setText(str(incoming))
    qtbot.mouseClick(ws._scan_btn, QtCore.Qt.MouseButton.LeftButton)
    wait_idle(qtbot, ws, SYNC_WORKERS)
    assert ws._incoming_table.topLevelItemCount() == 3
    return ws


# ── No sideways scrolling ─────────────────────────────────────────────────────

@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_library_sync_fits_without_horizontal_scroll(main_window, library, tmp_path,
                                                     monkeypatch, qtbot, size):
    """The known bug: the page's minimum width was ~1090 px (one button row),
    so it scrolled sideways even at the default 1300 × 860 window."""
    ws = _show(main_window, "library_sync", size)
    assert _hscroll(ws) == 0, "empty page scrolls sideways"

    ws = _sync_with_results(main_window, library, tmp_path, monkeypatch, qtbot)
    ws._incoming_table.setCurrentItem(ws._incoming_table.topLevelItem(0))
    qtbot.mouseClick(ws._build_plan_btn, QtCore.Qt.MouseButton.LeftButton)
    wait_idle(qtbot, ws, SYNC_WORKERS)
    assert ws._plan is not None
    _show(main_window, "library_sync", size)
    assert ws._review_box.isVisible() and ws._plan_stats.isVisible()
    assert _hscroll(ws) == 0, "page with results and a plan scrolls sideways"
    inner = ws._inner.minimumSizeHint().width()
    assert inner <= ws._scroll_area.viewport().width(), inner


@pytest.mark.parametrize("key", ["indexer", "file_cleanup", "exports"])
def test_other_organize_pages_fit_the_minimum_window(main_window, library, key):
    main_window.set_library(str(library.root))
    for size in SIZES:
        ws = _show(main_window, key, size)
        assert _hscroll(ws) == 0, (key, size)


def test_review_lists_sit_side_by_side_when_wide_and_stack_when_narrow(
        main_window, library, tmp_path, monkeypatch, qtbot):
    ws = _sync_with_results(main_window, library, tmp_path, monkeypatch, qtbot)
    _show(main_window, "library_sync", (900, 600))
    assert ws._review_split.orientation() == QtCore.Qt.Orientation.Vertical
    _show(main_window, "library_sync", (1300, 860))
    assert ws._review_split.orientation() == QtCore.Qt.Orientation.Horizontal
    _show(main_window, "library_sync", (900, 600))
    assert ws._review_split.orientation() == QtCore.Qt.Orientation.Vertical


# ── Library Sync: stats, empty state, flags, glass lists ──────────────────────

def test_sync_review_starts_empty_then_shows_stats_and_lists(
        main_window, library, tmp_path, monkeypatch, qtbot):
    ws = _show(main_window, "library_sync", (1300, 860))
    assert ws._review_empty.isVisible() and not ws._review_box.isVisible()
    assert ws._review_empty.title() == "No matches yet"

    ws = _sync_with_results(main_window, library, tmp_path, monkeypatch, qtbot)
    assert not ws._review_empty.isVisible() and ws._review_box.isVisible()
    assert ws.scan_stats() == {"incoming": "3", "new": "1", "matched": "2",
                               "upgrades": "1", "low_confidence": "0"}
    assert ws._steps.current() == ws.current_step() == 1
    # Raw engine status stays on the item; the list shows it in words.
    item = next(ws._incoming_table.topLevelItem(i) for i in range(3)
                if ws._incoming_table.topLevelItem(i).text(0) == "new_track.flac")
    assert item.text(1) == "NEW"
    delegate = ws._incoming_table.itemDelegateForColumn(1)
    assert delegate.displayText("EXACT_MATCH", QtCore.QLocale()) == "Exact match"


def test_right_click_flags_and_note_still_work(main_window, library, tmp_path, monkeypatch,
                                               qtbot, dialogs):
    ws = _sync_with_results(main_window, library, tmp_path, monkeypatch, qtbot)
    table = ws._incoming_table

    def right_click(name):
        item = next(table.topLevelItem(i) for i in range(table.topLevelItemCount())
                    if table.topLevelItem(i).text(0) == name)
        pos = table.visualItemRect(item).center()
        ws._on_incoming_context_menu(pos)
        return item, item.data(0, QtCore.Qt.ItemDataRole.UserRole)

    dialogs.on_menu("Copy")
    item, tid = right_click("new_track.flac")
    assert ws._review_state_store.is_copy_flagged(tid) and item.text(3) == "Copy"

    dialogs.on_menu("Replace")
    item, tid = right_click("northbound.flac")
    assert ws._review_state_store.replace_target(tid) and item.text(3) == "Replace"

    dialogs.on_menu("Clear flag")
    item, tid = right_click("northbound.flac")
    assert not ws._review_state_store.replace_target(tid) and item.text(3) == ""

    dialogs.on_menu("Add note…")
    dialogs.answer("getMultiLineText", ("check the mastering", True))
    item, tid = right_click("new_track.flac")
    assert ws._review_state_store.note_for(tid) == "check the mastering"
    assert item.text(4) == "check the mastering"
    menu_texts = dialogs.calls_of("menu")[0].text
    assert all(ord(ch) < 0x2190 or ch == "…" for ch in menu_texts), menu_texts


def test_plan_stats_follow_the_plan_and_hide_when_it_goes_stale(
        main_window, library, tmp_path, monkeypatch, qtbot):
    ws = _sync_with_results(main_window, library, tmp_path, monkeypatch, qtbot)
    assert ws.plan_stats() is None
    qtbot.mouseClick(ws._build_plan_btn, QtCore.Qt.MouseButton.LeftButton)
    wait_idle(qtbot, ws, SYNC_WORKERS)
    c = ws._plan.counts()
    assert ws.plan_stats() == {"add": str(c["copy"] + c["move"]), "replace": str(c["replace"]),
                               "skip": str(c["skip"]), "review": str(c["review"])}
    assert ws._steps.current() == 2
    ws._transfer_toggle.setChecked(False)          # the plan no longer matches its inputs
    assert ws.plan_stats() is None and not ws._execute_plan_btn.isEnabled()
    ws._transfer_toggle.setChecked(True)
    assert ws.plan_stats() is not None and ws._execute_plan_btn.isEnabled()


def test_sync_lists_are_glass_item_views(make_workspace):
    from gui.workspaces.organize_parts import ROW_HEIGHT, GlassRowDelegate

    ws = make_workspace("library_sync")
    for view in (ws._incoming_table, ws._existing_table):
        assert isinstance(view.itemDelegate(), GlassRowDelegate)
        assert not view.alternatingRowColors()
        assert view.styleSheet() == ""
        view.addTopLevelItem(QtWidgets.QTreeWidgetItem(["a track.flac", "NEW", "1.000", "", ""]))
        assert view.sizeHintForRow(0) >= ROW_HEIGHT
    assert ws._log_area.objectName() == "organizeLog" and ws._log_area.styleSheet() == ""


# ── Indexer: hero, step pills, stats, empty state ─────────────────────────────

def test_indexer_results_are_stat_tiles_with_an_empty_state(make_workspace, library, qtbot,
                                                             dialogs, threads):
    ws = make_workspace("indexer", library_path=library.root)
    assert ws._results_empty.isVisible() and not ws._stats_box.isVisible()
    assert ws._library_lbl.text() == str(library.root)
    assert ws._library_lbl.toolTip() == str(library.root)

    qtbot.mouseClick(ws._run_btn, QtCore.Qt.MouseButton.LeftButton)
    wait_idle(qtbot, ws)
    counts = ws._plan.counts()
    assert not ws._results_empty.isVisible() and ws._stats_box.isVisible()
    stats = ws.stats()
    for key in ("moves", "unchanged", "manual_review", "leftovers", "empty_dirs"):
        assert stats[key][0] == str(counts[key]), key
    assert ws._steps.current() == ws.current_step() == 1

    ws._dry_run_cb.setChecked(False)
    assert ws._steps.current() == 2
    dialogs.answer_yes()
    qtbot.mouseClick(ws._run_btn, QtCore.Qt.MouseButton.LeftButton)
    wait_idle(qtbot, ws)
    stats = ws.stats()
    assert stats["moves"] == (str(counts["moves"]), "moved")
    assert stats["manual_review"] == ("0", "problems")
    assert stats["leftovers"] == (str(counts["leftovers"]), "tidied")
    assert threads.wait_all(10_000) == []

    other = library.root.parent / "Somewhere else"
    other.mkdir()
    ws.set_library_path(str(other))
    assert ws._results_empty.isVisible() and not ws._stats_box.isVisible()
    assert ws._library_lbl.text() == str(other)


def test_indexer_keeps_its_advanced_controls(make_workspace):
    ws = make_workspace("indexer")
    adv = ws.findChild(QtWidgets.QWidget, "indexerAdvanced")
    for w in (ws._cross_album_cb, ws._flush_cache_cb, ws._max_workers_spin):
        assert adv.isAncestorOf(w)
    assert not ws._cross_album_cb.isEnabled() and not ws._flush_cache_cb.isEnabled()
    assert ws._max_workers_spin.isEnabled()
    assert ws.findChildren(QtWidgets.QLabel, "sectionTitle") == []      # no repeated title


# ── Buttons: glyphs, accessible names, no emoji ───────────────────────────────

def _is_emoji_or_symbol(ch: str) -> bool:
    return ord(ch) >= 0x2190 and ch not in "…–—‘’“”" and unicodedata.category(ch).startswith("S")


@pytest.mark.parametrize("key", ["indexer", "library_sync", "file_cleanup", "exports"])
def test_buttons_use_glyphs_and_have_names(make_workspace, key):
    ws = make_workspace(key)
    # Buttons the page shows (an EmptyState keeps a hidden, empty action button).
    buttons = [b for b in ws.findChildren(QtWidgets.QAbstractButton) if b.isVisibleTo(ws)]
    assert buttons
    for btn in buttons:
        text = btn.text()
        assert not any(_is_emoji_or_symbol(ch) for ch in text), (key, text)
        if not text:
            assert btn.accessibleName() and btn.toolTip(), (key, btn.objectName())
            assert btn.property("glyphName"), (key, btn.objectName())
    glyphs = {b.property("glyphName") for b in buttons if b.property("glyphName")}
    expected = {"indexer": {"play", "close", "external"},
                "library_sync": {"folder", "search", "list", "external", "play", "download"},
                "file_cleanup": {"search", "close", "undo"},
                "exports": {"download", "close", "external", "folder"}}[key]
    assert expected <= glyphs, (key, glyphs)


# ── File Cleanup / Exports: glass tiles, empty states, elided output ──────────

def test_file_cleanup_is_one_hero_tile_with_an_empty_state(make_workspace, tmp_path, qtbot,
                                                         threads):
    lib = tmp_path / "Empty"
    lib.mkdir()
    ws = make_workspace("file_cleanup", library_path=lib)
    tile = ws._cleanup_tile
    assert tile.objectName() == "headerCard"
    assert tile.empty_state is not None and tile.empty_state.isVisible()
    assert not tile.drawer_open()
    ws._cleanup_run_btn.click()
    qtbot.waitUntil(lambda: ws._cleanup_plan_worker is None and ws._cleanup_run_btn.isEnabled(),
                    timeout=5000)
    assert tile.drawer_open() and not tile.empty_state.isVisible()
    assert "Nothing to clean up" in ws._cleanup_status.text()
    ws.set_library_path(str(tmp_path))
    assert not tile.drawer_open() and tile.empty_state.isVisible()


def test_exports_show_the_written_file_elided(make_workspace, library, qtbot, threads):
    from gui.widgets.glass import ElidedLabel

    ws = make_workspace("exports", library_path=library.root)
    assert ws._codec_tile.empty_state.isVisible()
    assert all(isinstance(c, QtWidgets.QAbstractButton) and c.isChecked()
               for c in ws._codec_ext_cbs.values())
    ws._codec_ext_cbs[".flac"].click()
    assert not ws._codec_ext_cbs[".flac"].isChecked()
    ws._codec_run_btn.click()
    qtbot.waitUntil(lambda: ws._codec_worker is None, timeout=15000)
    out = ws._codec_out_path
    assert out and out.endswith("codec_file_list.txt")
    assert ws._codec_tile.output_path() == out
    label = ws._codec_tile.findChild(ElidedLabel, "toolOutputPath")
    assert label.toolTip() == out
    assert out not in ws._codec_status.text()      # the path is not crammed into the prose
    assert ws._codec_tile.status_chip.text == "Done"


@pytest.mark.parametrize("theme", ["midnight", "pearl"])
def test_pages_paint_in_classic_themes_and_reduced_transparency(make_workspace, theme):
    from gui.themes.manager import get_manager

    get_manager().apply(theme)
    for key in ("indexer", "library_sync", "file_cleanup", "exports"):
        ws = make_workspace(key)
        ws.grab()
    get_manager().set_reduce_transparency(True)
    try:
        for key in ("indexer", "library_sync", "file_cleanup", "exports"):
            make_workspace(key).grab()
    finally:
        get_manager().set_reduce_transparency(False)
