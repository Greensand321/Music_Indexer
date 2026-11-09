"""Duplicates workspace: scan → review → preview → execute, end to end.

Real synthetic library, real plan builder and executor; only fpcalc is faked
(``dupes_helpers.FakeFpcalc`` has the real function's signature and returns
genuine compressed Chromaprint strings).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PySide6 import QtWidgets

from dupes_helpers import install_fake_fpcalc, record_browser, run_execute, run_scan

YES = QtWidgets.QMessageBox.StandardButton.Yes
NO = QtWidgets.QMessageBox.StandardButton.No


@pytest.fixture
def dupes(make_workspace, library, monkeypatch):
    fake = install_fake_fpcalc(monkeypatch)
    opened = record_browser(monkeypatch)
    ws = make_workspace("duplicates", library_path=library.root)
    ws.fake_fpcalc = fake
    ws.opened = opened
    return ws


def _fault_line(library):
    flac = library.root / "Basalt" / "Granite" / "01 - Fault Line.flac"
    mp3 = library.root / "Basalt" / "Singles" / "Fault Line.mp3"
    assert flac.exists() and mp3.exists()
    return flac, mp3


def _step(ws) -> int:
    return ws._steps.current()


# ── DUP-01: fresh files really get fingerprinted ─────────────────────────────

def test_scan_fingerprints_a_fresh_library_and_groups_the_cross_codec_pair(dupes, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)

    log = ws._log_area.toPlainText()
    assert "Fingerprint failed" not in log, log
    assert "too many values to unpack" not in log
    assert len(ws.fake_fpcalc.calls) == len(library.audio_paths)
    flac, mp3 = _fault_line(library)
    assert len(ws._groups) == 1
    group = ws._groups[0]
    assert group.winner_path == str(flac) and group.losers == [str(mp3)]
    assert ws._groups_table.topLevelItemCount() == 1
    assert ws._execute_btn.isEnabled() and ws._preview_btn.isEnabled()
    # DS-22: the stepper moved on to "Review groups".
    assert _step(ws) == 1


def test_second_scan_uses_the_fingerprint_cache(dupes, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    first = len(ws.fake_fpcalc.calls)
    run_scan(ws, qtbot)
    assert len(ws.fake_fpcalc.calls) == first  # nothing re-fingerprinted
    assert len(ws._groups) == 1


# ── DUP-09 / DUP-14: the files the scan writes ───────────────────────────────

def test_scan_writes_preview_json_for_pair_review(dupes, library, qtbot):
    from gui.dialogs.pair_review_dialog import load_pairs

    run_scan(dupes, qtbot)
    preview_json = library.root / "Docs" / "duplicate_preview.json"
    assert preview_json.exists()
    payload = json.loads(preview_json.read_text(encoding="utf-8"))
    assert len(payload["plan"]["groups"]) == 1
    pairs = load_pairs(str(library.root))
    flac, mp3 = _fault_line(library)
    assert pairs and {pairs[0].left_path, pairs[0].right_path} == {str(flac), str(mp3)}


def test_metadata_cache_lands_in_library_docs_not_an_album_folder(make_workspace, tmp_path, qtbot, monkeypatch):
    from library_factory import TrackSpec, build_library

    install_fake_fpcalc(monkeypatch)
    record_browser(monkeypatch)
    root = tmp_path / "Lib"
    build_library(root, [
        TrackSpec("Art", "One", "Alb", 1, ext=".flac", freq=300),
        TrackSpec("Art", "One", "Alb", 1, ext=".mp3", freq=300, relpath="Art/Alb/One (copy).mp3"),
    ])
    ws = make_workspace("duplicates", library_path=root)
    run_scan(ws, qtbot)
    assert not (root / "Art" / "Alb" / "Docs").exists()
    assert not (root / "Art" / "Docs").exists()
    assert (root / "Docs" / ".duplicate_metadata_cache.json").exists()


# ── DUP-17: one clear fpcalc pre-flight message ──────────────────────────────

def test_missing_fpcalc_is_reported_once_before_scanning(dupes, dialogs, qtbot, monkeypatch, threads):
    import gui.workspaces.duplicates as mod

    monkeypatch.setattr(mod, "_missing_fingerprint_tools", lambda: ["fpcalc"])
    dialogs.answer_no()
    dupes._scan_btn.click()
    (question,) = dialogs.calls_of("question")
    assert "fpcalc" in question.text and "acoustid.org/chromaprint" in question.text
    assert not dupes._scan_running and threads.started == []
    assert "fpcalc not found" in dupes._prog_status.text()


def test_missing_fpcalc_can_scan_cached_fingerprints_only(dupes, dialogs, library, qtbot, monkeypatch):
    import gui.workspaces.duplicates as mod

    run_scan(dupes, qtbot)                    # fills the cache for every file
    extra = library.root / "Basalt" / "Granite" / "03 - Aftershock.flac"
    extra.write_bytes((library.root / "Basalt" / "Granite" / "02 - Magma.flac").read_bytes())
    calls = len(dupes.fake_fpcalc.calls)

    monkeypatch.setattr(mod, "_missing_fingerprint_tools", lambda: ["fpcalc", "ffmpeg"])
    dialogs.answer_yes()
    run_scan(dupes, qtbot)

    log = dupes._log_area.toPlainText()
    assert len(dupes.fake_fpcalc.calls) == calls          # nothing computed
    assert log.count("Fingerprint failed") == 0
    assert "1 file(s) skipped: not in the fingerprint cache and fpcalc is unavailable." in log
    assert len(dupes._groups) == 1


def test_fingerprint_failures_are_summarised_not_spammed(make_workspace, tmp_path, qtbot, monkeypatch):
    import chromaprint_utils
    import gui.workspaces.duplicates as mod
    from library_factory import TrackSpec, build_library

    root = tmp_path / "Lib"
    build_library(root, [TrackSpec("A", f"T{i}", "Alb", i, freq=200 + i) for i in range(25)])

    def broken(path, **_kw):
        raise chromaprint_utils.FingerprintError("FFmpeg error: bad data")

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", broken)
    monkeypatch.setattr(mod, "_missing_fingerprint_tools", lambda: [])
    ws = make_workspace("duplicates", library_path=root)
    run_scan(ws, qtbot)
    log = ws._log_area.toPlainText()
    assert log.count("! Fingerprint failed") == 20
    assert "5 more fingerprint failure(s) not shown" in log
    assert ws._prog_status.text() == "No fingerprinted tracks found."


# ── DUP-02: library change / quit while a job runs ───────────────────────────

def test_library_change_mid_scan_cancels_and_discards_results(dupes, library, tmp_path, qtbot, threads):
    import threading

    ws = dupes
    messages = []
    ws.log_message.connect(lambda msg, level: messages.append(msg))
    ws.fake_fpcalc.gate = threading.Event()
    ws._scan_btn.click()
    assert ws.fake_fpcalc.started.wait(10)
    old_worker = ws._worker
    other = tmp_path / "Other"
    other.mkdir()

    ws.set_library_path(str(other))         # used to drop the running QThread → abort
    assert old_worker._cancel_event.is_set()
    assert ws._scan_btn.isEnabled()          # the new library can be scanned at once
    ws.fake_fpcalc.gate.set()
    assert threads.wait_all(20_000) == []

    # The old library's late results never reach the new library's page.
    assert ws._groups == [] and ws._groups_table.topLevelItemCount() == 0
    assert ws._plan is None and ws._prog_status.text() == "Idle"
    assert not ws._execute_btn.isEnabled() and ws._log_area.toPlainText() == ""
    assert "Library changed: cancelled the running duplicate scan." in messages


def test_quitting_mid_scan_stops_the_worker(main_window, library, dialogs, qtbot, monkeypatch, threads):
    import threading

    fake = install_fake_fpcalc(monkeypatch)
    record_browser(monkeypatch)
    fake.gate = threading.Event()
    main_window.set_library(str(library.root))
    ws = main_window._workspaces["duplicates"]
    ws._scan_btn.click()
    assert fake.started.wait(10)

    dialogs.answer_yes()                    # "Background tasks running — stop and quit?"
    main_window.close()
    assert ws._worker._cancel_event.is_set()
    fake.gate.set()
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=20_000)
    assert threads.wait_all(20_000) == []


def test_scan_cannot_be_started_twice(dupes, qtbot, threads):
    import threading

    ws = dupes
    ws.fake_fpcalc.gate = threading.Event()
    ws._scan_btn.click()
    assert not ws._scan_btn.isEnabled()
    ws._on_scan()                            # a stray second trigger
    ws.fake_fpcalc.gate.set()
    qtbot.waitUntil(lambda: not ws._scan_running, timeout=30_000)
    assert threads.started.count("DupeBuildWorker") == 1


def test_cancel_scan(dupes, qtbot):
    import threading

    ws = dupes
    ws.fake_fpcalc.gate = threading.Event()
    ws._scan_btn.click()
    assert ws.fake_fpcalc.started.wait(10)
    ws._cancel_btn.click()
    ws.fake_fpcalc.gate.set()
    qtbot.waitUntil(lambda: not ws._scan_running, timeout=30_000)
    assert ws._prog_status.text() == "Cancelled."
    assert ws._plan is None and not ws._execute_btn.isEnabled()


# ── DUP-04: review flags get an honest path ─────────────────────────────────

def _flag_first_group(ws):
    ws._plan.groups[0].review_flags = ["Placeholder metadata detected"]
    ws._plan.refresh_plan_signature()


def test_flagged_plan_declined_moves_nothing(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    _flag_first_group(ws)
    flac, mp3 = _fault_line(library)

    dialogs.answer_no()
    run_execute(ws, qtbot)

    (question,) = dialogs.calls_of("question")
    assert "flagged for review" in question.text
    assert "Placeholder metadata detected" in question.text
    assert "Fault Line.flac" in question.text
    assert mp3.exists() and flac.exists()
    assert "Nothing was moved" in ws._prog_status.text()
    assert ws._execute_btn.isEnabled()        # still executable after reviewing


def test_flagged_plan_override_executes_and_quarantines(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    _flag_first_group(ws)
    flac, mp3 = _fault_line(library)

    dialogs.answer_yes().answer_yes()        # override review, then confirm
    run_execute(ws, qtbot)

    assert not mp3.exists() and flac.exists()
    assert (library.root / "Quarantine" / mp3.name).exists()
    assert ws._prog_status.text().startswith("Execution complete: 1 quarantined, 0 deleted")
    titles = [c.title for c in dialogs.calls_of("question")]
    assert titles == ["Groups flagged for review", "Confirm Execute"]


def test_execute_unflagged_plan_asks_only_for_confirmation(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    for g in ws._plan.groups:
        g.review_flags = []
    ws._plan.review_flags = []
    ws._plan.refresh_plan_signature()
    flac, mp3 = _fault_line(library)

    dialogs.answer_yes()
    run_execute(ws, qtbot)
    (confirm,) = dialogs.calls_of("question")
    assert confirm.title == "Confirm Execute"
    assert "1 duplicate file(s) will be moved to Quarantine/" in confirm.text
    assert "PERMANENTLY" not in confirm.text
    assert (library.root / "Quarantine" / mp3.name).exists()


def test_blocked_execution_reports_the_reason(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    flac, mp3 = _fault_line(library)
    time.sleep(0.01)
    mp3.write_bytes(mp3.read_bytes() + b"changed after the preview")

    dialogs.answer_yes().answer_yes()
    run_execute(ws, qtbot)
    status = ws._prog_status.text()
    assert status.startswith("Execution blocked — nothing was changed.")
    assert "Library changed since preview" in status
    assert mp3.exists()


# ── DUP-03: "Delete" means delete, behind a second, typed confirmation ───────

def test_delete_requires_typed_confirmation(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    flac, mp3 = _fault_line(library)
    ws._delete_rb.setChecked(True)

    dialogs.answer_yes().answer_yes()
    dialogs.answer("getText", ("delete", True))      # wrong word
    run_execute(ws, qtbot)
    assert mp3.exists()
    assert "Deletion not confirmed" in ws._prog_status.text()
    confirm = [c for c in dialogs.calls_of("question") if c.title == "Confirm Execute"][-1]
    assert "1 duplicate file(s) will be PERMANENTLY DELETED" in confirm.text
    assert "moved to Quarantine" not in confirm.text


def test_delete_really_deletes_after_confirmation(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    flac, mp3 = _fault_line(library)
    ws._delete_rb.setChecked(True)

    dialogs.answer_yes().answer_yes()
    dialogs.answer("getText", ("DELETE", True))
    run_execute(ws, qtbot)

    assert not mp3.exists() and flac.exists()
    assert not (library.root / "Quarantine" / mp3.name).exists()
    assert ws._prog_status.text().startswith("Execution complete: 0 quarantined, 1 deleted")
    report = json.loads(Path(ws._last_report_path).with_suffix(".json").read_text(encoding="utf-8"))
    assert any(a["step"] == "loser_cleanup" and a["detail"] == "Loser deleted." for a in report["actions"])
    # The preview was re-written to show the disposition that executed.
    preview = (library.root / "Docs" / "duplicate_preview.html").read_text(encoding="utf-8")
    assert f"{mp3} → delete" in preview and "→ quarantine" not in preview


def test_preview_export_failure_changes_nothing_and_retry_reexports(dupes, dialogs, library, qtbot, monkeypatch):
    import duplicate_consolidation

    ws = dupes
    run_scan(ws, qtbot)
    flac, mp3 = _fault_line(library)
    ws._delete_rb.setChecked(True)
    real = duplicate_consolidation.export_consolidation_preview_html

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(duplicate_consolidation, "export_consolidation_preview_html", broken)
    dialogs.answer_yes()                              # review override
    run_execute(ws, qtbot)
    assert "disk full" in dialogs.calls_of("warning")[-1].text
    assert ws._plan.groups[0].loser_disposition == {str(mp3): "quarantine"}
    assert mp3.exists()

    monkeypatch.setattr(duplicate_consolidation, "export_consolidation_preview_html", real)
    dialogs.answer_yes().answer_no()                  # override, then decline the confirm
    run_execute(ws, qtbot)
    assert ws._plan.groups[0].loser_disposition == {str(mp3): "delete"}
    preview = (library.root / "Docs" / "duplicate_preview.html").read_text(encoding="utf-8")
    assert f"{mp3} → delete" in preview
    assert mp3.exists()


def test_quarantine_is_still_the_default(dupes):
    assert dupes._quarantine_rb.isChecked() and not dupes._delete_rb.isChecked()


# ── DUP-16: the per-group disposition and playlist controls tell the truth ──

def test_per_group_retain_keeps_the_loser(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    flac, mp3 = _fault_line(library)
    ws._groups_table.setCurrentItem(ws._groups_table.topLevelItem(0))
    assert ws._disposition_combo.isEnabled()
    ws._disposition_combo.setCurrentIndex(ws._disposition_combo.findData("retain"))
    assert "Loser (retain)" in ws._group_details.toPlainText()

    dialogs.answer_yes().answer_yes()
    run_execute(ws, qtbot)
    assert mp3.exists()
    assert "1 duplicate file(s) will stay where they are" in dialogs.calls_of("question")[-1].text
    assert "0 quarantined, 0 deleted, 1 kept in place" in ws._prog_status.text()


def test_per_group_delete_override_also_needs_typed_confirmation(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    flac, mp3 = _fault_line(library)
    ws._groups_table.setCurrentItem(ws._groups_table.topLevelItem(0))
    ws._disposition_combo.setCurrentIndex(ws._disposition_combo.findData("delete"))

    dialogs.answer_yes().answer_yes()
    dialogs.answer("getText", ("", False))           # cancelled
    run_execute(ws, qtbot)
    assert mp3.exists()


def test_playlist_checkbox_is_shown_as_always_on(dupes):
    cb = dupes._playlists_cb
    assert cb.isChecked() and not cb.isEnabled()
    assert "Always on" in cb.toolTip()


# ── DUP-12 / DUP-13 / DS-22: after an execution ──────────────────────────────

def test_after_execute_report_opens_and_plan_is_spent(dupes, dialogs, library, qtbot):
    ws = dupes
    run_scan(ws, qtbot)
    dialogs.answer_yes().answer_yes()
    run_execute(ws, qtbot)

    assert ws._last_report_path.endswith("execution_report.html")
    assert ws.opened and ws.opened[-1].endswith("execution_report.html")
    assert not ws._execute_btn.isEnabled()                 # DUP-13
    assert "Scan again" in ws._prog_status.text()
    assert _step(ws) == 2                                   # DS-22

    ws.opened.clear()
    ws._report_btn.click()                                  # DUP-12
    assert ws.opened == [Path(ws._last_report_path).resolve().as_uri()]
    assert "duplicate_preview" not in ws.opened[0]


def test_open_report_finds_reports_from_earlier_sessions(dupes, dialogs, library, qtbot, make_workspace, monkeypatch):
    run_scan(dupes, qtbot)
    dialogs.answer_yes().answer_yes()
    run_execute(dupes, qtbot)

    fresh = make_workspace("duplicates", library_path=library.root)
    assert fresh._report_btn.isEnabled()
    opened = record_browser(monkeypatch)
    fresh._report_btn.click()
    assert opened and opened[0].endswith("execution_report.html")


# ── DUP-15: thresholds come from Settings ────────────────────────────────────

def test_thresholds_start_from_config_and_follow_settings(make_workspace, qtbot):
    from config import load_config, save_config

    cfg = load_config()
    cfg.update(exact_duplicate_threshold=0.011, near_duplicate_threshold=0.077, mixed_codec_threshold_boost=0.044)
    save_config(cfg)
    ws = make_workspace("duplicates")
    assert (ws._exact_spin.value(), ws._near_spin.value(), ws._boost_spin.value()) == (0.011, 0.077, 0.044)

    ws._near_spin.setValue(0.2)              # a local tweak survives re-showing...
    ws.hide()
    ws.show()
    ws.on_settings_saved()                   # ...and an unrelated Settings save
    assert ws._near_spin.value() == 0.2

    cfg = load_config()
    cfg["near_duplicate_threshold"] = 0.123  # ...until Settings saves a new value
    save_config(cfg)
    ws.on_settings_saved()                   # the main window calls this after a save
    assert ws._near_spin.value() == 0.123


def test_settings_save_reaches_pages_through_the_main_window(main_window):
    from config import load_config, save_config

    cfg = load_config()
    cfg["near_duplicate_threshold"] = 0.045
    save_config(cfg)
    main_window._on_settings_saved()
    assert main_window._workspaces["duplicates"]._near_spin.value() == 0.045
    assert main_window._workspaces["similarity"]._near_t.text() == "0.045"


def test_scan_uses_the_thresholds_shown(dupes, qtbot, monkeypatch):
    import duplicate_consolidation

    seen = {}
    real = duplicate_consolidation.build_consolidation_plan

    def spy(tracks, **kwargs):
        seen.update(kwargs)
        return real(tracks, **kwargs)

    monkeypatch.setattr(duplicate_consolidation, "build_consolidation_plan", spy)
    dupes._near_spin.setValue(0.05)
    run_scan(dupes, qtbot)
    assert seen["near_duplicate_threshold"] == 0.05
    assert seen["cancel_event"] is not None and seen["progress_callback"] is not None
