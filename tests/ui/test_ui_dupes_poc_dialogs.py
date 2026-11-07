"""Tools → Diagnostics proofs of concept: they must not crash and not lie.

DUP-07 Fuzzy Finder always failed (wrong call) and would have found 0 pairs;
its filter controls did nothing. DUP-10 Scan Engine silently did nothing
without librosa, and opening it created Docs/. DUP-11 closing a dialog left
its worker running (Bucketing then popped "Open it now?" from a closed
dialog). DUP-19/20 workers were unparented and dialogs leaked per open.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest
from PySide6 import QtCore, QtWidgets

from _helpers.dupes_chromaprint import real_fingerprints

FPS = real_fingerprints()


def _show(qtbot, dlg):
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    return dlg


def _deleted(obj) -> bool:
    import shiboken6

    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete.value)
    QtWidgets.QApplication.processEvents()
    return not shiboken6.isValid(obj)


# ── Fuzzy Finder (DUP-07, DUP-16, DUP-21) ────────────────────────────────────

@pytest.fixture
def fingerprinted_library(tmp_path):
    from fingerprint_cache import flush_fingerprint_writes, store_fingerprint

    root = tmp_path / "Lib"
    tags = {"artist": "Artist", "title": "Song", "album": "Album"}
    entries = {
        root / "Artist" / "Album" / "01 - Song.flac": FPS["songA.flac"],
        root / "Artist" / "Album" / "01 - Song.mp3": FPS["songA.mp3_64"],
        root / "Artist" / "Album" / "02 - Other.mp3": FPS["songB.flac"],
    }
    db = str(root / "Docs" / ".duplicate_fingerprints.db")
    for path, fp in entries.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
        t = dict(tags, title="Other") if "Other" in path.name else tags
        store_fingerprint(str(path), db, None, fp, ext=path.suffix, tags=t)
    (root / "Artist" / "Album" / "03 - New.mp3").write_bytes(b"not fingerprinted yet")
    flush_fingerprint_writes(db, timeout=5)
    return root


def test_fuzzy_finds_the_cross_codec_pair(qtbot, fingerprinted_library):
    from gui.dialogs.fuzzy_dupe_dialog import FuzzyDupeDialog

    dlg = _show(qtbot, FuzzyDupeDialog(str(fingerprinted_library)))
    dlg._run_btn.click()
    qtbot.waitUntil(lambda: dlg._run_btn.isEnabled(), timeout=20_000)
    assert dlg._prog_status.text() == (
        "1 candidate pair(s) found; 1 track(s) not fingerprinted yet were skipped."
    )
    item = dlg._table.topLevelItem(0)
    assert (item.text(2), item.text(3)) == ("01 - Song.flac", "01 - Song.mp3")
    assert item.text(0) == "Exact duplicate"


def test_fuzzy_without_a_cache_says_what_to_do(qtbot, tmp_path):
    from gui.dialogs.fuzzy_dupe_dialog import FuzzyDupeDialog

    dlg = _show(qtbot, FuzzyDupeDialog(str(tmp_path)))
    dlg._run_btn.click()
    qtbot.waitUntil(lambda: dlg._run_btn.isEnabled(), timeout=10_000)
    assert "run a scan in Duplicates first" in dlg._prog_status.text()


def test_fuzzy_controls_without_a_backend_are_disabled(qtbot, tmp_path):
    from gui.dialogs.fuzzy_dupe_dialog import FuzzyDupeDialog

    dlg = _show(qtbot, FuzzyDupeDialog(str(tmp_path)))
    for widget in (dlg._min_word_len, dlg._min_shared, dlg._ignore_common, *dlg._field_cbs,
                   dlg._no_fp_cb, dlg._skip_fp_cb, dlg._include_non_cb, dlg._send_review_btn):
        assert not widget.isEnabled(), widget
    assert dlg._near_t.isEnabled() and dlg._boost_t.isEnabled()
    headers = [dlg._table.headerItem().text(i) for i in range(dlg._table.columnCount())]
    assert "Shared Words" not in headers


def test_fuzzy_close_mid_run_cancels_and_frees_the_dialog(qtbot, fingerprinted_library, monkeypatch, threads):
    import fingerprint_cache
    from gui.dialogs.fuzzy_dupe_dialog import FuzzyDupeDialog

    gate, started = threading.Event(), threading.Event()
    real = fingerprint_cache.get_cached_fingerprint_metadata

    def slow(*a, **k):
        started.set()
        gate.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(fingerprint_cache, "get_cached_fingerprint_metadata", slow)
    dlg = _show(qtbot, FuzzyDupeDialog(str(fingerprinted_library)))
    dlg._run_btn.click()
    assert started.wait(10)
    worker = dlg._worker
    dlg.close()
    assert worker._cancel.is_set()
    gate.set()
    assert threads.wait_all(10_000) == []
    assert _deleted(dlg)


# ── Scan Engine (DUP-10, DUP-11) ─────────────────────────────────────────────

def test_scan_engine_without_librosa_says_so_and_creates_nothing(qtbot, tmp_path, monkeypatch):
    import gui.dialogs.scan_engine_dialog as mod

    monkeypatch.setattr(mod, "_missing_engine_modules", lambda: ["librosa"])
    dlg = _show(qtbot, mod.ScanEngineDialog(str(tmp_path)))
    assert not dlg._run_btn.isEnabled()
    assert "needs librosa" in dlg._status_lbl.text()
    assert "pip install librosa" in dlg._run_btn.toolTip()
    assert not (tmp_path / "Docs").exists()          # opening the dialog writes nothing


def test_scan_engine_import_failure_is_shown(qtbot, tmp_path, monkeypatch, dialogs):
    import gui.dialogs.scan_engine_dialog as mod

    monkeypatch.setattr(mod, "_missing_engine_modules", lambda: [])
    broken = types.ModuleType("duplicate_scan_engine")

    def __getattr__(name):
        raise ImportError("No module named 'librosa'")

    broken.__getattr__ = __getattr__
    monkeypatch.setitem(sys.modules, "duplicate_scan_engine", broken)
    dlg = _show(qtbot, mod.ScanEngineDialog(str(tmp_path)))
    dlg._run_btn.click()
    (warning,) = dialogs.calls_of("warning")
    assert "librosa" in warning.text
    assert "scan engine unavailable" in dlg._log_box.toPlainText()


@pytest.fixture
def fake_scan_engine(monkeypatch):
    import gui.dialogs.scan_engine_dialog as mod

    monkeypatch.setattr(mod, "_missing_engine_modules", lambda: [])
    fake = types.ModuleType("duplicate_scan_engine")
    fake.gate = None
    fake.started = threading.Event()

    class DuplicateScanConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Summary:
        def __init__(self, cancelled=False):
            self.tracks_total, self.headers_updated, self.fingerprints_updated = 3, 3, 2
            self.edges_written, self.groups_written, self.cancelled = 1, 1, cancelled

    def run_duplicate_scan(library, db, config, log_callback=None, cancel_event=None):
        fake.started.set()
        log_callback("Found 3 audio files.")
        if fake.gate is not None:
            while not fake.gate.is_set() and not cancel_event.is_set():
                fake.gate.wait(0.01)
        return Summary(cancelled=cancel_event.is_set())

    fake.DuplicateScanConfig = DuplicateScanConfig
    fake.run_duplicate_scan = run_duplicate_scan
    monkeypatch.setitem(sys.modules, "duplicate_scan_engine", fake)
    return fake


def test_scan_engine_runs_and_reports(qtbot, tmp_path, fake_scan_engine):
    from gui.dialogs.scan_engine_dialog import ScanEngineDialog

    dlg = _show(qtbot, ScanEngineDialog(str(tmp_path)))
    dlg._run_btn.click()
    qtbot.waitUntil(lambda: dlg._worker is None, timeout=10_000)
    log = dlg._log_box.toPlainText()
    assert "Found 3 audio files." in log
    assert "Summary: 3 tracks, 3 headers updated, 2 fingerprints updated, 1 edges, 1 groups." in log
    assert (tmp_path / "Docs").is_dir()              # created by the run, not by opening


def test_scan_engine_close_mid_run_cancels(qtbot, tmp_path, fake_scan_engine, threads):
    from gui.dialogs.scan_engine_dialog import ScanEngineDialog

    fake_scan_engine.gate = threading.Event()
    dlg = _show(qtbot, ScanEngineDialog(str(tmp_path)))
    dlg._run_btn.click()
    assert fake_scan_engine.started.wait(10)
    worker = dlg._worker
    dlg.reject()                                      # Esc / Close / title bar
    assert worker._cancel_event.is_set()
    assert threads.wait_all(10_000) == []
    assert _deleted(dlg)


# ── Bucketing POC (DUP-11, DUP-17-style pre-flight) ──────────────────────────

@pytest.fixture
def fake_bucketing(monkeypatch):
    import chromaprint_utils
    import duplicate_bucketing_poc

    monkeypatch.setattr(chromaprint_utils, "missing_tools", lambda *a, **k: [])
    state = types.SimpleNamespace(gate=None, started=threading.Event(), report="")

    def run(folder, *, log_callback=None, cancel_event=None):
        state.started.set()
        log_callback("Duplicate Bucketing POC: scanning")
        if state.gate is not None:
            while not state.gate.is_set():
                if cancel_event.is_set():
                    raise duplicate_bucketing_poc.BucketingCancelled()
                state.gate.wait(0.01)
        state.report = f"{folder}/Docs/duplicate_bucketing_poc_report.html"
        return state.report

    monkeypatch.setattr(duplicate_bucketing_poc, "run_duplicate_bucketing_poc", run)
    return state


def test_bucketing_close_mid_run_cancels_without_popups(qtbot, tmp_path, fake_bucketing, dialogs, threads):
    from gui.dialogs.bucketing_poc_dialog import BucketingPocDialog

    fake_bucketing.gate = threading.Event()
    dlg = _show(qtbot, BucketingPocDialog(str(tmp_path)))
    dlg._run_btn.click()
    assert fake_bucketing.started.wait(10)
    worker = dlg._worker
    dlg.close()
    assert worker._cancel_event.is_set()
    assert threads.wait_all(10_000) == []
    assert dialogs.calls_of("question") == []         # no "Open it now?" from a closed dialog
    assert _deleted(dlg)


def test_bucketing_completion_offers_the_report(qtbot, tmp_path, fake_bucketing, dialogs, monkeypatch):
    import webbrowser
    from gui.dialogs.bucketing_poc_dialog import BucketingPocDialog

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url))
    dlg = _show(qtbot, BucketingPocDialog(str(tmp_path)))
    dialogs.answer_yes()
    dlg._run_btn.click()
    qtbot.waitUntil(lambda: dlg._worker is None, timeout=10_000)
    assert dlg._status_lbl.text() == "Completed"
    assert opened and opened[0].endswith("duplicate_bucketing_poc_report.html")


def test_bucketing_without_fpcalc_asks_first_and_says_so(qtbot, tmp_path, fake_bucketing, dialogs, monkeypatch, threads):
    import chromaprint_utils
    from gui.dialogs.bucketing_poc_dialog import BucketingPocDialog

    monkeypatch.setattr(chromaprint_utils, "missing_tools", lambda *a, **k: ["fpcalc"])
    dlg = _show(qtbot, BucketingPocDialog(str(tmp_path)))
    dialogs.answer_no()
    dlg._run_btn.click()
    assert threads.started == []
    assert "fpcalc not found" in dialogs.calls_of("question")[0].text

    dialogs.answer_yes().answer_no()                  # run anyway; don't open the report
    dlg._run_btn.click()
    qtbot.waitUntil(lambda: dlg._worker is None, timeout=10_000)
    assert dlg._status_lbl.text() == "Completed without fingerprints (fpcalc not found)"
