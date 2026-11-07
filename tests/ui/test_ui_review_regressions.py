"""Regressions for the independent concurrency review's crash reproductions.

The reviewer reproduced "QThread: Destroyed while thread is still running"
aborts on the Duplicates page (library change mid-scan, an immediate
rescan, closing mid-execute) before that page moved onto the JobRegistry.
These tests drive exactly those scenarios through the harness; a regression
would abort the test process, which is impossible to miss.
"""
from __future__ import annotations

import threading
import time

from dupes_helpers import install_fake_fpcalc, record_browser, run_scan
from ui_harness import process_events


def _unflag(ws) -> None:
    for g in ws._plan.groups:
        g.review_flags = []
    ws._plan.review_flags = []
    ws._plan.refresh_plan_signature()


def test_rescan_right_after_a_library_change_mid_scan(make_workspace, library, tmp_path,
                                                      monkeypatch, qtbot, threads):
    fake = install_fake_fpcalc(monkeypatch)
    record_browser(monkeypatch)
    ws = make_workspace("duplicates", library_path=library.root)
    fake.gate = threading.Event()
    ws._scan_btn.click()
    assert fake.started.wait(10)

    other = tmp_path / "Other"
    other.mkdir()
    ws.set_library_path(str(other))      # cancels the first scan...
    ws.set_library_path(str(library.root))
    assert ws._scan_btn.isEnabled()
    ws._scan_btn.click()                 # ...and a new one starts at once
    fake.gate.set()
    qtbot.waitUntil(lambda: not ws._scan_running, timeout=30_000)
    assert threads.wait_all(20_000) == []
    assert ws._plan is not None and ws._groups, "the fresh scan's results are shown"


def test_closing_mid_execute_asks_and_waits_for_the_file_moves(main_window, library, dialogs,
                                                               monkeypatch, qtbot, threads):
    import duplicate_consolidation_executor as executor

    install_fake_fpcalc(monkeypatch)
    record_browser(monkeypatch)
    main_window.set_library(str(library.root))
    ws = main_window._workspaces["duplicates"]
    run_scan(ws, qtbot)
    _unflag(ws)

    real = executor.execute_consolidation_plan
    moving = threading.Event()
    finished = threading.Event()

    def slow_execute(plan, config, *args, **kwargs):
        moving.set()
        time.sleep(0.5)                  # "moving files" — must not be cut off
        result = real(plan, config, *args, **kwargs)
        finished.set()
        return result

    monkeypatch.setattr(executor, "execute_consolidation_plan", slow_execute)
    dialogs.answer_yes()                 # Confirm Execute
    ws._execute_btn.click()
    assert moving.wait(10)

    dialogs.answer_yes()                 # "Background tasks running — stop and quit?"
    main_window.close()
    process_events(50)
    titles = [c.title for c in dialogs.calls_of("question")]
    assert titles == ["Confirm Execute", "Background tasks running"]
    assert main_window.isVisible(), "the window waits for the execute to stop"
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=30_000)
    assert finished.is_set(), "the file moves ran to their safe stopping point"
    assert threads.wait_all(10_000) == []
