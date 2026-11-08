"""Duplicates page in the Liquid Glass look (R2): the parts the restyle added.

Find duplicates: hero card, an empty state before the first scan, the plan
and execution summaries as stat tiles, glass group rows that show each
group's disposition, glyphs (no emoji) on the buttons. Compare two files:
the verdict pill, the distance tiles and the explanation. The launchers: one
card each, no repeated page title. And every section fits the minimum window
(900 x 600) without horizontal scrolling.
"""
from __future__ import annotations

import re

import pytest
from PySide6 import QtCore, QtWidgets

from dupes_helpers import install_fake_fpcalc, record_browser, run_execute, run_scan
from ui_harness import process_events

SECTIONS = ("find", "compare", "pair_review", "fuzzy_finder", "scan_engine", "bucketing")
# Emoji and pictographs the old buttons carried (🔍 📄 ▶ 📊 ✕ ⚖ ✅ ◄ ►) and any
# other symbol outside the Basic Multilingual Plane.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿▶◄►⚖✅]")


@pytest.fixture
def dupes(make_workspace, library, monkeypatch):
    install_fake_fpcalc(monkeypatch)
    record_browser(monkeypatch)
    return make_workspace("duplicates", library_path=library.root)


def _unflag(ws):
    for group in ws._plan.groups:
        group.review_flags = []
    ws._plan.review_flags = []
    ws._plan.refresh_plan_signature()


# ── Widths ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [(900, 600), (1300, 860)])
def test_every_duplicates_section_fits_without_horizontal_scroll(main_window, library, monkeypatch,
                                                                  threads, qtbot, size):
    install_fake_fpcalc(monkeypatch)
    record_browser(monkeypatch)
    main_window.resize(*size)
    main_window.set_library(str(library.root))
    assert threads.wait_all(20_000) == []
    main_window._on_nav_changed("duplicates/find")
    run_scan(main_window._workspaces["duplicates"], qtbot)      # populated rows and tiles too
    for key in SECTIONS:
        main_window._on_nav_changed(f"duplicates/{key}")
        process_events(30)
        ws = main_window.current_panel()
        bar = ws._scroll_area.horizontalScrollBar()
        assert bar.maximum() == 0, (key, size, ws._inner.minimumSizeHint().width(),
                                    ws._scroll_area.viewport().width())


# ── Find duplicates ──────────────────────────────────────────────────────────

def test_empty_state_before_the_first_scan_then_results(dupes, qtbot):
    ws = dupes
    assert ws._results_stack.currentWidget() is ws._empty
    assert ws._empty.title() == "No scan yet"
    assert ws._library_lbl.text() == str(ws._library_path)
    assert ws._library_lbl.toolTip() == str(ws._library_path)
    process_events(5)
    # The card is as tall as the empty state, not as the results page it replaces.
    assert ws._results_stack.height() < ws._empty.sizeHint().height() + 24

    run_scan(ws, qtbot)
    process_events(5)
    assert ws._results_stack.currentWidget() is ws._results_page
    assert ws._results_stack.height() > ws._empty.sizeHint().height() + 100
    stats = ws._plan_stats
    assert stats.tile("groups").value() == "1"
    assert stats.tile("files").value() == "1"
    assert stats.tile("space").value() != "0 B"            # the MP3 copy's size
    assert not ws._outcome_stats.isVisibleTo(ws)
    assert ws._groups_table.topLevelItem(0).toolTip(0).startswith("Keeps: ")


def test_no_library_and_no_duplicates_have_their_own_empty_states(make_workspace, tmp_path, qtbot,
                                                                   monkeypatch):
    from library_factory import TrackSpec, build_library

    ws = make_workspace("duplicates")
    assert ws._empty.title() == "No library selected"

    install_fake_fpcalc(monkeypatch)
    record_browser(monkeypatch)
    root = tmp_path / "Lib"
    build_library(root, [TrackSpec("A", "One", "Alb", 1), TrackSpec("A", "Two", "Alb", 2, freq=500)])
    ws.set_library_path(str(root))
    assert ws._empty.title() == "No scan yet"
    run_scan(ws, qtbot)
    assert ws._results_stack.currentWidget() is ws._empty
    assert ws._empty.title() == "No duplicates found"


def test_dispositions_show_on_the_rows_and_drive_the_reclaimable_space(dupes, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    group = ws._groups[0]
    space = ws._plan_stats.tile("space").value()
    assert ws._group_disposition(group) == "quarantine"

    ws._delete_rb.setChecked(True)                 # global choice
    assert ws._group_disposition(group) == "delete"
    assert ws._plan_stats.tile("space").value() == space

    ws._groups_table.setCurrentItem(ws._groups_table.topLevelItem(0))
    assert ws._selected_lbl.text().startswith("Group 1 of 1")
    ws._disposition_combo.setCurrentIndex(ws._disposition_combo.findData("retain"))
    assert ws._group_disposition(group) == "retain"
    assert ws._plan_stats.tile("space").value() == "0 B"
    assert "Loser (retain)" in ws._group_details.toPlainText()

    ws._groups_table.viewport().grab()             # the delegate paints every tag without error


def test_execution_fills_the_outcome_tiles(dupes, dialogs, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    _unflag(ws)
    dialogs.answer_yes()
    run_execute(ws, qtbot)
    tiles = ws._outcome_stats
    assert tiles.isVisibleTo(ws)
    assert [tiles.tile(k).value() for k in tiles.keys()] == ["1", "0", "0", "0", "0"]
    assert ws._summary_reason.text().startswith("Execution complete: 1 quarantined")
    assert ws._prog_status.text().startswith("Execution complete: 1 quarantined")


def test_execution_counts_match_the_summary_sentence():
    from duplicate_consolidation_executor import ExecutionAction, ExecutionResult
    from gui.workspaces.duplicates import execution_counts, summarize_execution

    def result(*actions, success=False):
        return ExecutionResult(success=success, actions=list(actions), playlist_reports=[],
                               quarantine_index={}, report_paths={})

    mixed = result(
        ExecutionAction("loser_cleanup", "/a.mp3", "success", "Loser quarantined.",
                        metadata={"disposition": "quarantine"}),
        ExecutionAction("loser_cleanup", "/b.mp3", "failed", "Not removed: its winner is missing.",
                        metadata={"disposition": "quarantine", "skip_reason": "winner_missing"}),
        ExecutionAction("loser_cleanup", "/c.mp3", "failed", "Permission denied",
                        metadata={"disposition": "quarantine"}),
        ExecutionAction("loser_cleanup", "/d.mp3", "skipped", "Execution stopped.",
                        metadata={"skip_reason": "not_attempted"}),
    )
    counts = execution_counts(mixed)
    assert (counts["quarantined"], counts["deleted"], counts["refused"],
            counts["not_attempted"], counts["errors"]) == (1, 0, 1, 1, 1)
    assert counts["refusal_reasons"] == ["winner missing"]
    ok, message = summarize_execution(mixed)
    assert message.startswith("Execution finished with 1 error(s): Permission denied")
    assert "1 left in place for safety (winner missing)" in message and "1 not attempted" in message

    blocked = result(ExecutionAction("preflight", "execution", "blocked", "Library changed since preview"))
    counts = execution_counts(blocked)
    assert counts["blocked"] and counts["errors"] == 0
    assert summarize_execution(blocked)[1].startswith("Execution blocked — nothing was changed.")


def test_buttons_carry_glyphs_not_emoji(dupes, make_workspace):
    ws = dupes
    for btn in (ws._scan_btn, ws._preview_btn, ws._report_btn, ws._execute_btn, ws._cancel_btn):
        assert btn.property("glyphName"), btn.text()
        assert not btn.icon().isNull()
    assert ws._execute_btn.objectName() == "dangerBtn"
    assert ws._scan_btn.objectName() == "primaryBtn"
    sim = make_workspace("similarity")
    for key in ("pair_review", "fuzzy_finder", "scan_engine", "bucketing"):
        panel = make_workspace(key)
        assert panel.findChild(QtWidgets.QPushButton, "launcherOpenBtn").property("glyphName")
    for widget in (ws, sim):
        for btn in widget.findChildren(QtWidgets.QAbstractButton):
            assert not _EMOJI.search(btn.text()), btn.text()


def test_steps_are_a_pill_row_that_follows_the_flow(dupes, dialogs, qtbot):
    ws = dupes
    assert [lbl.text() for lbl in ws._step_labels] == ["Scan", "Review", "Execute"]
    assert ws._step_labels[0].parentWidget().objectName() == "dupStepRow"
    assert ws._current_step() == 0
    run_scan(ws, qtbot)
    assert ws._current_step() == 1
    _unflag(ws)
    dialogs.answer_yes()
    run_execute(ws, qtbot)
    assert ws._current_step() == 2


def test_cancel_and_progress_only_show_while_a_job_runs(dupes, qtbot):
    import threading

    import chromaprint_utils

    ws = dupes
    assert not ws._cancel_btn.isVisible() and not ws._prog_bar.isVisible()
    fake = chromaprint_utils.fingerprint_fpcalc       # the fixture's FakeFpcalc
    fake.gate = threading.Event()
    ws._scan_btn.click()
    assert ws._cancel_btn.isVisible() and ws._cancel_btn.isEnabled() and ws._prog_bar.isVisible()
    fake.gate.set()
    qtbot.waitUntil(lambda: not ws._scan_running, timeout=30_000)
    assert not ws._cancel_btn.isVisible() and not ws._prog_bar.isVisible()


def test_page_has_no_repeated_title(dupes, make_workspace):
    for ws in (dupes, make_workspace("similarity"), make_workspace("pair_review"),
               make_workspace("bucketing")):
        titles = [w for w in ws.findChildren(QtWidgets.QLabel) if w.objectName() == "sectionTitle"]
        assert titles == []
        assert ws.findChild(QtWidgets.QLabel, "dupHeroTitle") is not None


# ── Compare two files ────────────────────────────────────────────────────────

@pytest.fixture
def compare(make_workspace, tmp_path, monkeypatch):
    import chromaprint_utils
    from _helpers.dupes_chromaprint import real_fingerprints

    fps = real_fingerprints()
    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc",
                        lambda path, **kw: fps["songA.flac"] if str(path).endswith(".flac")
                        else fps["songA.mp3_64"])
    monkeypatch.setattr(chromaprint_utils, "missing_tools", lambda *a, **k: [])
    a, b = tmp_path / "Song.flac", tmp_path / "Song.mp3"
    for p in (a, b):
        p.write_bytes(b"x")
    ws = make_workspace("similarity")
    ws.files = (a, b)
    return ws


def test_compare_shows_a_verdict_pill_tiles_and_an_explanation(compare, qtbot):
    ws = compare
    assert ws._result_stack.currentWidget() is ws._empty
    a, b = ws.files
    ws._file_a.setText(str(a))
    ws._file_b.setText(str(b))
    assert ws._file_a.toolTip() == str(a)
    ws._run_btn.click()
    qtbot.waitUntil(lambda: ws._worker is None, timeout=20_000)

    assert ws._result_stack.currentWidget() is ws._result_page
    assert (ws._verdict.text(), ws._verdict.tone()) == ("Exact duplicate", "success")
    distance = float(ws._results_text.toPlainText().split("Fingerprint distance:")[1].split()[0])
    assert ws._stats.tile("distance").value() == f"{distance:.4f}"
    assert ws._stats.tile("overlap").value() != "–"
    assert ws._stats.tile("exact").value() == "0.0200"
    assert "exact threshold" in ws._explain_lbl.text()
    assert "mixed codecs" in ws._explain_lbl.text()

    assert not ws._results_text.isVisibleTo(ws)            # the full report folds away
    ws._report_toggle.click()
    assert ws._results_text.isVisibleTo(ws)

    ws._close_btn.click()                                   # Clear
    assert ws._result_stack.currentWidget() is ws._empty
    assert ws._file_a.text() == "" and ws._results_text.toPlainText() == ""


def test_compare_failure_is_a_danger_verdict(compare, qtbot, monkeypatch):
    import chromaprint_utils

    def broken(path, **_kw):
        raise chromaprint_utils.FingerprintError("decoder exploded")

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", broken)
    ws = compare
    a, b = ws.files
    ws._file_a.setText(str(a))
    ws._file_b.setText(str(b))
    ws._run_btn.click()
    qtbot.waitUntil(lambda: ws._worker is None, timeout=20_000)
    assert (ws._verdict.text(), ws._verdict.tone()) == ("Could not compare", "danger")
    assert "decoder exploded" in ws._explain_lbl.text()


def test_advanced_overrides_fold_away_and_name_the_settings_values(compare):
    ws = compare
    assert not ws._adv_box.isVisibleTo(ws) and not ws._near_t.isEnabled()
    assert "near 0.1" in ws._adv_summary.text()
    ws._adv_group.setChecked(True)
    assert ws._adv_box.isVisibleTo(ws) and ws._near_t.isEnabled()
    assert not ws._adv_summary.isVisibleTo(ws)


def test_a_dropped_audio_file_fills_its_row(compare, tmp_path):
    ws = compare
    a, _b = ws.files
    row = ws._file_rows[0]
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(a))])
    event = type("Drop", (), {"mimeData": lambda self: mime})()
    assert row._dropped_file(event) == str(a)
    other = QtCore.QMimeData()
    other.setUrls([QtCore.QUrl.fromLocalFile(str(tmp_path / "notes.txt"))])
    assert row._dropped_file(type("Drop", (), {"mimeData": lambda self: other})()) == ""


# ── Launchers ────────────────────────────────────────────────────────────────

def test_launcher_card_keeps_the_pill_beside_its_headline(make_workspace):
    ws = make_workspace("fuzzy_finder")
    title = ws.findChild(QtWidgets.QLabel, "dupHeroTitle")
    assert title.text() == ws.HEADLINE != ws.TITLE
    assert ws.pill.isVisible()
    assert abs(ws.pill.geometry().center().y() - title.geometry().center().y()) < 12
    assert ws.pill.x() > title.x()
    assert "not ready to test" in ws._note.text()
    plain = make_workspace("pair_review")
    assert plain.pill is None


# ── Shared parts and themes ──────────────────────────────────────────────────

def test_tile_grid_reflows_into_balanced_rows(qtbot):
    from gui.widgets.dupes_parts import TileGrid

    grid = TileGrid([(str(i), f"Tile {i}") for i in range(5)], min_tile_width=128)
    qtbot.addWidget(grid)
    assert grid.columns_for(1000) == 5
    assert grid.columns_for(560) == 3            # 4 fit, but 3 + 2 beats 4 + 1
    assert grid.columns_for(100) == 1
    grid.resize(560, 200)
    grid.show()
    process_events(5)
    assert grid.columns() == 3
    assert grid.minimumSizeHint().height() > grid.tile("0").sizeHint().height()


@pytest.mark.parametrize("theme", ["midnight", "pearl"])
def test_duplicates_sections_paint_in_classic_themes_and_reduced_transparency(dupes, make_workspace,
                                                                              qtbot, theme):
    from gui.themes.manager import get_manager

    run_scan(dupes, qtbot)
    sim = make_workspace("similarity")
    launcher = make_workspace("scan_engine")
    mgr = get_manager()
    mgr.apply(theme)
    for reduce in (False, True):
        mgr.set_reduce_transparency(reduce)
        process_events(5)
        for ws in (dupes, sim, launcher):
            assert not ws.grab().isNull()
    mgr.set_reduce_transparency(False)
