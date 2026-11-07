"""Cooperative cancellation for the two proof-of-concept duplicate engines
(so closing their dialogs mid-run can stop them, DUP-11)."""
from __future__ import annotations

import threading
import types

import pytest

from _helpers.module_isolation import isolated_modules


def test_bucketing_poc_stops_when_cancelled(tmp_path, monkeypatch):
    import duplicate_bucketing_poc as poc

    for name in ("a.mp3", "b.mp3"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(poc, "discover_files", lambda root: [str(tmp_path / "a.mp3"), str(tmp_path / "b.mp3")])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(poc.BucketingCancelled):
        poc.run_duplicate_bucketing_poc(str(tmp_path), log_callback=lambda m: None, cancel_event=cancel)
    assert not (tmp_path / "Docs" / "duplicate_bucketing_poc_report.html").exists()


def test_bucketing_poc_fingerprinting_stops_when_cancelled(tmp_path, monkeypatch):
    import chromaprint_utils
    import duplicate_bucketing_poc as poc

    calls = []
    cancel = threading.Event()

    def fpcalc(path, **_kw):
        calls.append(path)
        cancel.set()
        return None

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", fpcalc)
    paths = [str(tmp_path / f"t{i}.mp3") for i in range(20)]
    for p in paths:
        open(p, "wb").write(b"x")
    with pytest.raises(poc.BucketingCancelled):
        poc._compute_fingerprints(paths, str(tmp_path / "fp.db"), lambda m: None, max_workers=1,
                                  cancel_event=cancel)
    assert len(calls) == 1


def test_bucketing_poc_reports_missing_fingerprints(tmp_path, monkeypatch):
    """With fpcalc missing the report used to be written as if all was well."""
    import duplicate_bucketing_poc as poc
    from duplicate_bucketing_poc import TrackInfo

    info = {
        str(tmp_path / n): TrackInfo(str(tmp_path / n), "song", "artist", "album", None, None, 1, "MP3")
        for n in ("a.mp3", "b.flac")
    }
    for p in info:
        open(p, "wb").write(b"x")
    monkeypatch.setattr(poc, "discover_files", lambda root: list(info))
    monkeypatch.setattr(poc, "_extract_track_info", lambda paths, log, cancel=None: info)
    monkeypatch.setattr(poc, "_compute_fingerprints", lambda *a, **k: {p: None for p in info})
    logs = []
    report = poc.run_duplicate_bucketing_poc(str(tmp_path), log_callback=logs.append)
    assert report.endswith("duplicate_bucketing_poc_report.html")
    assert any("2 of 2 candidate track(s) could not be fingerprinted" in line for line in logs)


def test_scan_engine_stops_when_cancelled(tmp_path):
    librosa_stub = types.ModuleType("librosa")
    with isolated_modules({"librosa": librosa_stub}, fresh=["duplicate_scan_engine"]) as mods:
        engine = mods["duplicate_scan_engine"]
        (tmp_path / "a.mp3").write_bytes(b"x")
        cancel = threading.Event()
        cancel.set()
        logs = []
        summary = engine.run_duplicate_scan(str(tmp_path), str(tmp_path / "scan.db"),
                                            log_callback=logs.append, cancel_event=cancel)
    assert summary.cancelled is True
    assert summary.headers_updated == 0 and summary.edges_written == 0
    assert "Cancelled." in logs
