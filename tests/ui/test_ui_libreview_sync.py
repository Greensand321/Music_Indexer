"""Library Sync page, second review round (area: libreview).

* A scan abandoned after a library change still flushes its last batch of
  log lines; they must never land in the new library's log.
* Executing a plan whose library changed since Build Plan reports the stale
  items instead of claiming a replace that did not happen.
"""
from __future__ import annotations

import shutil
import threading

import pytest
from PySide6 import QtCore

from library_helpers import build_incoming, install_fake_fingerprints, wait_idle

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
    return library.root, incoming_root, paths


def _click(qtbot, button):
    assert button.isEnabled(), button.text()
    qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)


def test_abandoned_scan_never_writes_into_the_new_librarys_log(make_workspace, env, qtbot, monkeypatch, threads):
    import library_sync  # the module the worker will import
    lib, incoming, _paths = env
    started, gate = threading.Event(), threading.Event()
    real = library_sync.fingerprint_generator.compute_fingerprints_parallel

    def slow(paths, db, log_callback=None, progress_callback=None, cancel_event=None, **kw):
        if not started.is_set():
            started.set()
            # Batched behind the "Fingerprinting…" line: flushed when the worker ends.
            log_callback("OLD-LIBRARY line 1")
            log_callback("OLD-LIBRARY line 2")
            gate.wait(10)
        return real(paths, db, log_callback, progress_callback, cancel_event=cancel_event, **kw)

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", slow)
    ws = make_workspace("library_sync", library_path=lib)
    ws._incoming_entry.setText(str(incoming))
    _click(qtbot, ws._scan_btn)
    old = ws._worker
    assert started.wait(10)
    qtbot.waitUntil(lambda: "Fingerprinting" in ws._log_area.toPlainText(), timeout=5_000)

    other = lib.parent / "Another Library"
    other.mkdir()
    ws.set_library_path(str(other))
    assert ws._worker is None
    assert "Fingerprinting" not in ws._log_area.toPlainText(), "the old library's log is cleared"

    old.log_line.emit("OLD-LIBRARY late line")      # what a late flush looks like
    gate.set()
    assert threads.wait_all(10_000) == []
    qtbot.wait(50)                                   # deliver any queued signals
    assert "OLD-LIBRARY" not in ws._log_area.toPlainText()


def test_execute_after_the_library_changed_reports_stale_items(make_workspace, env, dialogs, qtbot):
    lib, incoming, paths = env
    ws = make_workspace("library_sync", library_path=lib)
    ws._incoming_entry.setText(str(incoming))
    _click(qtbot, ws._scan_btn)
    wait_idle(qtbot, ws, ALL_WORKERS)
    ids = {ws._incoming_table.topLevelItem(i).text(0): ws._incoming_table.topLevelItem(i).data(0, QtCore.Qt.ItemDataRole.UserRole)
           for i in range(ws._incoming_table.topLevelItemCount())}
    ws._flag_for_replace(ids["northbound.flac"])
    _click(qtbot, ws._build_plan_btn)
    wait_idle(qtbot, ws, ALL_WORKERS)
    assert ws._plan is not None

    # The library is reorganised between Build Plan and Execute.
    old = lib / "Aurora Lines" / "Dawn Signals" / "02 - Northbound.mp3"
    moved = lib / "By Year" / "2019" / "Aurora Lines_02_Northbound.mp3"
    moved.parent.mkdir(parents=True)
    shutil.move(str(old), str(moved))

    dialogs.answer_yes()
    _click(qtbot, ws._execute_plan_btn)
    wait_idle(qtbot, ws, ALL_WORKERS)
    warning = dialogs.calls_of("warning")[-1]
    assert warning.title == "Execution Problems"
    assert "0 replaced" in warning.text and "library changed since the plan was built" in warning.text
    assert paths["upgrade"].exists(), "the stale Replace transferred nothing"
    assert not (lib / "Aurora Lines" / "Dawn Signals" / "02 - Northbound.flac").exists()
