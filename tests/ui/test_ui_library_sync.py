"""Library Sync page, end to end against the synthetic library (area: library).

scan → review flags → build plan → confirm (queued via ``dialogs``) →
execute, then the filesystem is checked. Fingerprints are faked from the
title tag (no fpcalc in CI) by ``library_helpers.install_fake_fingerprints``.
"""
from __future__ import annotations

import json
import shutil
import threading

import pytest
from PySide6 import QtCore

from library_helpers import (
    build_incoming,
    content_multiset,
    install_fake_fingerprints,
    reserved_tree,
    tree,
    wait_idle,
)

ALL_WORKERS = ("_worker", "_build_worker", "_exec_worker")


@pytest.fixture
def opened(monkeypatch):
    urls = []
    monkeypatch.setattr("webbrowser.open", lambda url, *a, **k: urls.append(url) or True)
    return urls


@pytest.fixture
def env(library, tmp_path, monkeypatch, opened):
    install_fake_fingerprints(monkeypatch)
    incoming_root = tmp_path / "incoming"
    paths = build_incoming(incoming_root)
    lib = library.root
    (lib / "Quarantine").mkdir()
    shutil.copyfile(paths["new"], lib / "Quarantine" / "loser.flac")   # LIB-06 bait
    return lib, incoming_root, paths


def _page(make_workspace, lib, incoming):
    ws = make_workspace("library_sync", library_path=lib)
    ws._incoming_entry.setText(str(incoming))
    statuses = []
    ws.status_changed.connect(lambda text, colour: statuses.append((text, colour)))
    ws._statuses = statuses
    return ws


def _click(qtbot, button):
    assert button.isEnabled(), button.text()
    qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)


def _scan(qtbot, ws):
    _click(qtbot, ws._scan_btn)
    wait_idle(qtbot, ws, ALL_WORKERS)


def _build(qtbot, ws):
    _click(qtbot, ws._build_plan_btn)
    wait_idle(qtbot, ws, ALL_WORKERS)
    assert ws._plan is not None, ws._plan_status_lbl.text()


def _ids(ws):
    out = {}
    for i in range(ws._incoming_table.topLevelItemCount()):
        item = ws._incoming_table.topLevelItem(i)
        out[item.text(0)] = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
    return out


def test_scan_flag_build_execute_end_to_end(make_workspace, env, dialogs, qtbot, opened, threads):
    lib, incoming, paths = env
    lib_before = tree(lib)
    inc_before = tree(incoming)
    reserved_before = reserved_tree(lib)
    ws = _page(make_workspace, lib, incoming)
    assert ws.current_step() == 0

    _scan(qtbot, ws)
    assert ws.current_step() == 1                                           # DS-22
    rows = {ws._incoming_table.topLevelItem(i).text(0): ws._incoming_table.topLevelItem(i).text(1)
            for i in range(ws._incoming_table.topLevelItemCount())}
    assert rows["new_track.flac"] == "NEW"                                  # not matched to Quarantine/ (LIB-06)
    assert rows["loose_wire_copy.mp3"] == "EXACT_MATCH"
    summary = ws._match_summary_lbl.text()
    assert "1 new" in summary and "2 already in library" in summary         # LIB-20
    assert '"event"' not in ws._log_area.toPlainText()                      # no raw JSON
    assert ws._existing_table.topLevelItemCount() == 2                      # LIB-17
    assert ws._recompute_btn.isEnabled()

    ids = _ids(ws)
    ws._transfer_toggle.setChecked(False)                                   # Move mode
    ws._flag_for_copy(ids["new_track.flac"])                                # keep the original (LIB-03)
    ws._flag_for_replace(ids["northbound.flac"])                            # upgrade (LIB-04)
    _build(qtbot, ws)
    assert ws.current_step() == 2
    assert any(u.endswith("LibrarySyncPreview.html") for u in opened)

    dialogs.answer_yes()
    _click(qtbot, ws._execute_plan_btn)
    wait_idle(qtbot, ws, ALL_WORKERS)
    question = dialogs.calls_of("question")[-1].text
    assert "MOVE" in question and "1 track(s) added" in question and "1 existing track(s) replaced" in question

    lib_after = tree(lib)
    inc_after = tree(incoming)
    # Copy-flagged track: still in incoming, and now in the library.
    assert inc_after["drop/new_track.flac"] == inc_before["drop/new_track.flac"]
    added = [k for k, v in lib_after.items() if v == inc_before["drop/new_track.flac"]
             and not k.startswith("Quarantine/")]
    assert len(added) == 1 and added[0].startswith("By Year/2024/")
    # Duplicate: skipped — still in incoming, not copied into the library (LIB-05).
    assert "drop/loose_wire_copy.mp3" in inc_after
    assert list(lib_after.values()).count(inc_before["drop/loose_wire_copy.mp3"]) == \
        list(lib_before.values()).count(inc_before["drop/loose_wire_copy.mp3"])
    # Replace: moved into the MP3's slot; the MP3 is in the per-run backup.
    old = "Aurora Lines/Dawn Signals/02 - Northbound.mp3"
    assert old not in lib_after
    assert lib_after["Aurora Lines/Dawn Signals/02 - Northbound.flac"] == inc_before["drop/northbound.flac"]
    assert "drop/northbound.flac" not in inc_after
    assert any(k.startswith("Docs/Backups/LibrarySync/") and v == lib_before[old] for k, v in lib_after.items())
    # Nothing in the library was lost or overwritten; reserved folders untouched.
    assert not (content_multiset(lib_before) - content_multiset(lib_after))
    for rel, digest in reserved_before.items():
        assert lib_after.get(rel) == digest, rel
    audit = json.loads((lib / "Docs" / "LibrarySyncAudit.json").read_text())
    assert audit["plan"]["transfer_mode"] == "move"
    # The executed report opens (wrong summary key, LIB-11) and the plan is spent.
    assert any(u.endswith("LibrarySyncExecuted.html") for u in opened)
    assert "Execution complete" in ws._plan_status_lbl.text()
    assert ws._plan is None and not ws._execute_plan_btn.isEnabled()
    assert not ws._build_plan_btn.isEnabled()          # library changed: scan again
    assert threads.wait_all(10_000) == []


def test_any_input_change_invalidates_the_plan(make_workspace, env, qtbot):
    """LIB-10: Execute used to stay enabled for a plan the inputs no longer match."""
    lib, incoming, _paths = env
    ws = _page(make_workspace, lib, incoming)
    _scan(qtbot, ws)
    ids = _ids(ws)
    _build(qtbot, ws)
    assert ws._execute_plan_btn.isEnabled()

    ws._transfer_toggle.setChecked(False)
    assert not ws._execute_plan_btn.isEnabled()
    assert "out of date" in ws._plan_status_lbl.text()
    ws._transfer_toggle.setChecked(True)
    assert ws._execute_plan_btn.isEnabled()

    ws._flag_for_copy(ids["loose_wire_copy.mp3"])
    assert not ws._execute_plan_btn.isEnabled()
    ws._flag_clear(ids["loose_wire_copy.mp3"])
    assert ws._execute_plan_btn.isEnabled()

    ws._incoming_entry.setText(str(incoming) + "_other")
    assert not ws._execute_plan_btn.isEnabled() and not ws._build_plan_btn.isEnabled()
    ws._incoming_entry.setText(str(incoming))
    assert ws._execute_plan_btn.isEnabled()

    other = lib.parent / "Another Library"
    other.mkdir()
    ws.set_library_path(str(other))
    assert not ws._execute_plan_btn.isEnabled()


def test_rescan_disables_execute_while_running(make_workspace, env, qtbot):
    lib, incoming, _paths = env
    ws = _page(make_workspace, lib, incoming)
    _scan(qtbot, ws)
    _build(qtbot, ws)
    _click(qtbot, ws._scan_btn)
    assert not ws._execute_plan_btn.isEnabled() and not ws._build_plan_btn.isEnabled()
    wait_idle(qtbot, ws, ALL_WORKERS)
    assert ws._plan is None and not ws._execute_plan_btn.isEnabled()


@pytest.mark.parametrize("case", ["typo_existing", "missing_incoming", "nested"])
def test_bad_folders_are_refused_not_created(make_workspace, env, dialogs, qtbot, case):
    """LIB-15: a typo'd library used to be created and everything looked NEW."""
    lib, incoming, _paths = env
    ws = _page(make_workspace, lib, incoming)
    typo = lib.parent / "Musc"
    if case == "typo_existing":
        ws._existing_entry.setText(str(typo))
    elif case == "missing_incoming":
        ws._incoming_entry.setText(str(lib.parent / "nowhere"))
    else:
        nested = lib / "Incoming"
        nested.mkdir()
        ws._incoming_entry.setText(str(nested))
    _click(qtbot, ws._scan_btn)
    assert ws._worker is None
    assert dialogs.calls_of("warning")
    assert not typo.exists() and not (lib.parent / "nowhere").exists()


def test_threshold_input_is_validated_and_normalised(make_workspace, env, dialogs, qtbot, monkeypatch):
    """LIB-18: parse errors were swallowed and 'flac=0.1' never matched."""
    import library_sync  # the module the worker will import (not a stale copy)
    lib, incoming, _paths = env
    seen = {}
    real = library_sync.compare_libraries

    def spy(*a, **k):
        seen["thresholds"] = k.get("thresholds")
        return real(*a, **k)

    monkeypatch.setattr(library_sync, "compare_libraries", spy)
    ws = _page(make_workspace, lib, incoming)
    ws._threshold_entry.setText("abc")
    _click(qtbot, ws._scan_btn)
    assert ws._worker is None and "not a number" in dialogs.calls_of("warning")[-1].text
    ws._threshold_entry.setText("0.2")
    ws._overrides_text.setPlainText("flac=0.1\nMP3 = 0.25")
    _scan(qtbot, ws)
    assert seen["thresholds"] == {"default": 0.2, ".flac": 0.1, ".mp3": 0.25}


def test_html_export_includes_flags_and_notes(make_workspace, env, dialogs, qtbot, tmp_path):
    """LIB-19: the HTML report was written from an empty flag store."""
    lib, incoming, _paths = env
    ws = _page(make_workspace, lib, incoming)
    _scan(qtbot, ws)
    tid = _ids(ws)["new_track.flac"]
    ws._flag_for_copy(tid)
    ws._review_state_store.set_note(tid, "NOTE-XYZ")
    out = tmp_path / "review.html"
    dialogs.choose_save_file(out)
    _click(qtbot, ws._export_report_btn)
    text = out.read_text()
    assert "NOTE-XYZ" in text and "<td>Copy</td>" in text


def test_rescan_keeps_flags_that_still_apply(make_workspace, env, dialogs, qtbot):
    """LIB-23: every re-scan silently dropped all flags."""
    lib, incoming, paths = env
    ws = _page(make_workspace, lib, incoming)
    _scan(qtbot, ws)
    ids = _ids(ws)
    ws._flag_for_copy(ids["new_track.flac"])
    ws._flag_for_replace(ids["northbound.flac"])
    _scan(qtbot, ws)
    store = ws._review_state_store
    assert store.is_copy_flagged(ids["new_track.flac"]) and store.replace_target(ids["northbound.flac"])
    flags_col = {ws._incoming_table.topLevelItem(i).text(0): ws._incoming_table.topLevelItem(i).text(3)
                 for i in range(ws._incoming_table.topLevelItemCount())}
    assert "Copy" in flags_col["new_track.flac"] and "Replace" in flags_col["northbound.flac"]

    paths["new"].unlink()
    _scan(qtbot, ws)
    assert not store.flags.copy
    info = dialogs.calls_of("information")[-1]
    assert info.title == "Flags Updated" and "Copy flag" in info.text


def test_recompute_rematches_without_rescanning(make_workspace, env, qtbot, monkeypatch):
    """LIB-17: Recompute was never enabled (and would have rescanned)."""
    lib, incoming, _paths = env
    ws = _page(make_workspace, lib, incoming)
    _scan(qtbot, ws)
    calls = []
    monkeypatch.setattr("library_sync.compare_libraries", lambda *a, **k: calls.append(1))
    ws._threshold_entry.setText("0.1")
    _click(qtbot, ws._recompute_btn)
    wait_idle(qtbot, ws, ALL_WORKERS)
    assert calls == []
    assert ws._incoming_table.topLevelItemCount() == 3
    assert ws._build_plan_btn.isEnabled()


def test_save_session_is_restored(make_workspace, env, qtbot):
    lib, incoming, _paths = env
    ws = _page(make_workspace, lib, incoming)
    ws._threshold_entry.setText("0.25")
    ws._on_save_session()
    again = make_workspace("library_sync", library_path=lib)
    assert again._incoming_entry.text() == str(incoming)
    assert again._threshold_entry.text() == "0.25"
    assert not again._preset_entry.isEnabled()          # unread: disabled, says so


def test_cancel_scan_then_rescan_immediately(make_workspace, env, qtbot, monkeypatch, threads):
    """LIB-07 / LIB-21: cancel, re-run; a cancel is neutral, not a red Error."""
    import library_sync  # the module the worker will import (not a stale copy)
    lib, incoming, _paths = env
    gate = threading.Event()
    real = library_sync.fingerprint_generator.compute_fingerprints_parallel

    def slow(paths, db, log_callback=None, progress_callback=None, cancel_event=None, **kw):
        if not gate.is_set():
            while cancel_event is not None and not cancel_event.wait(0.01):
                pass
        return real(paths, db, log_callback, progress_callback, cancel_event=cancel_event, **kw)

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", slow)
    ws = _page(make_workspace, lib, incoming)
    for _ in range(3):
        _click(qtbot, ws._scan_btn)
        assert not ws._scan_btn.isEnabled()           # no double run
        _click(qtbot, ws._cancel_btn)
        wait_idle(qtbot, ws, ALL_WORKERS)
        assert ws._state_lbl.text() == "State: Cancelled"
        text, colour = ws._statuses[-1]
        assert text == "Cancelled"
        from gui.themes.manager import get_manager
        assert colour != get_manager().current.danger
    gate.set()
    _scan(qtbot, ws)
    assert ws._incoming_table.topLevelItemCount() == 3
    assert threads.wait_all(10_000) == []


def test_cancelled_execute_is_not_reported_as_success(make_workspace, env, dialogs, qtbot, monkeypatch):
    """LIB-11: cancelled / failed runs were reported as 'Execution complete'."""
    import library_sync  # the module the worker will import (not a stale copy)
    lib, incoming, _paths = env
    ws = _page(make_workspace, lib, incoming)
    _scan(qtbot, ws)
    _build(qtbot, ws)
    real = library_sync.execute_library_sync_plan

    def cancelled(plan, **kw):
        event = threading.Event()
        event.set()
        kw["cancel_event"] = event
        return real(plan, **kw)

    monkeypatch.setattr(library_sync, "execute_library_sync_plan", cancelled)
    before = tree(lib)
    dialogs.answer_yes()
    _click(qtbot, ws._execute_plan_btn)
    wait_idle(qtbot, ws, ALL_WORKERS)
    assert "cancelled" in ws._plan_status_lbl.text().lower()
    assert ws._statuses[-1][0] == "Cancelled"
    assert {k: v for k, v in tree(lib).items() if not k.startswith("Docs/")} == \
        {k: v for k, v in before.items() if not k.startswith("Docs/")}


def test_quit_during_scan_cancels_and_waits(main_window, env, dialogs, qtbot, monkeypatch):
    """LIB-08: closing the window mid-scan used to abort the process."""
    import library_sync  # the module the worker will import (not a stale copy)
    lib, incoming, _paths = env
    started = threading.Event()

    def stuck(paths, db, log_callback=None, progress_callback=None, cancel_event=None, **kw):
        started.set()
        while not cancel_event.wait(0.01):
            pass
        return []

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", stuck)
    main_window.set_library(str(lib))
    ws = main_window._workspaces["library_sync"]
    ws._incoming_entry.setText(str(incoming))
    _click(qtbot, ws._scan_btn)
    worker = ws._worker
    assert started.wait(10)
    dialogs.answer_yes()
    main_window.close()
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=10_000)
    assert not worker.isRunning()
