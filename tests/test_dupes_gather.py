"""Duplicate Finder Stage 1 (``_gather_tracks_for_plan``) without Qt threads.

DUP-01: the scan unpacked ``duration, fp = fingerprint_fpcalc(path)`` — but
the function returns one string, so every fresh file failed with "too many
values to unpack" and the scan found nothing.
"""
from __future__ import annotations

import os
import shutil
import threading

import pytest

pytest.importorskip("PySide6")  # gui.workspaces.duplicates imports the Qt shim

from _helpers.dupes_chromaprint import fake_fingerprint, write_fake_fpcalc


def _gather():
    from gui.workspaces.duplicates import _gather_tracks_for_plan
    return _gather_tracks_for_plan


def _write(path, data=b"audio"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _run(root, monkeypatch=None, *, compute_missing=True):
    logs, progress = [], []
    tracks = _gather()(
        str(root),
        str(root / "Docs" / ".duplicate_fingerprints.db"),
        logs.append,
        progress.append,
        threading.Event(),
        compute_missing=compute_missing,
    )
    return tracks, logs, progress


def test_fresh_files_are_fingerprinted_with_the_real_signature(tmp_path, monkeypatch):
    import chromaprint_utils

    calls = []

    def fpcalc(path, *, trim=True, start_sec=0.0, duration_sec=120.0,
               threshold_db=-50.0, min_silence_duration=0.5):
        calls.append(path)
        return fake_fingerprint(os.path.basename(path))   # a str, like the real one

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", fpcalc)
    a = _write(tmp_path / "Artist" / "Album" / "01 - a.flac")
    b = _write(tmp_path / "Artist" / "Album" / "02 - b.mp3")

    tracks, logs, progress = _run(tmp_path)

    assert not [line for line in logs if "Fingerprint failed" in line], logs
    assert sorted(t["path"] for t in tracks) == sorted([str(a), str(b)])
    assert all(t["fingerprint"] for t in tracks)
    assert progress[-1] == 60
    # Stored: a second pass computes nothing.
    calls.clear()
    tracks2, _logs, _ = _run(tmp_path)
    assert calls == [] and len(tracks2) == 2


def test_through_the_real_fingerprint_fpcalc_with_a_fake_binary(tmp_path, monkeypatch):
    if os.name == "nt" or shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg and a POSIX shell")
    from library_factory import TrackSpec, build_library
    from _helpers.module_isolation import isolated_modules

    write_fake_fpcalc(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ.get("PATH", ""))
    lib = build_library(tmp_path / "Lib", [
        TrackSpec("A", "Song", "Alb", 1, ext=".flac", seconds=2.0, freq=330),
        TrackSpec("A", "Song", "Alb", 1, ext=".mp3", seconds=2.0, freq=330, relpath="A/Alb/Song.mp3"),
    ])
    with isolated_modules(fresh=["chromaprint_utils"]):
        tracks, logs, _ = _run(lib.root)
    assert not [line for line in logs if line.startswith("!")], logs
    assert len(tracks) == 2 and all(t["fingerprint"] for t in tracks)


def test_reserved_top_level_folders_are_skipped(tmp_path, monkeypatch):
    import chromaprint_utils

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", lambda p, **k: fake_fingerprint(p))
    keep = _write(tmp_path / "Artist" / "Album" / "song.mp3")
    for reserved in ("Quarantine", "Docs", "Playlists", "Trash", "Not Sorted", "Manual Review"):
        _write(tmp_path / reserved / "song.mp3")
    tracks, _logs, _ = _run(tmp_path)
    assert [t["path"] for t in tracks] == [str(keep)]


def test_track_dicts_name_the_scan_root(tmp_path, monkeypatch):
    """DUP-14: without it the plan builder's metadata cache went to
    <common folder of the tracks>/Docs — inside an album folder."""
    import chromaprint_utils

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", lambda p, **k: fake_fingerprint(p))
    _write(tmp_path / "Art" / "Alb" / "song.mp3")
    tracks, _logs, _ = _run(tmp_path)
    assert tracks[0]["discovery"] == {"scan_roots": [str(tmp_path)]}


def test_without_fpcalc_only_cached_files_are_used_and_reported_once(tmp_path, monkeypatch):
    import chromaprint_utils

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", lambda p, **k: fake_fingerprint(p))
    _write(tmp_path / "A" / "one.mp3")
    _run(tmp_path)                                    # caches one.mp3
    for i in range(30):
        _write(tmp_path / "A" / f"new{i}.mp3")

    def must_not_run(*_a, **_k):
        raise AssertionError("fpcalc must not be called")

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", must_not_run)
    tracks, logs, _ = _run(tmp_path, compute_missing=False)
    assert [os.path.basename(t["path"]) for t in tracks] == ["one.mp3"]
    assert logs == ["30 file(s) skipped: not in the fingerprint cache and fpcalc is unavailable."]


def test_tool_missing_mid_scan_is_logged_once(tmp_path, monkeypatch):
    import chromaprint_utils

    def missing(_path, **_kw):
        raise chromaprint_utils.FingerprintError("Required tool 'fpcalc' not found")

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", missing)
    for i in range(10):
        _write(tmp_path / "A" / f"t{i}.mp3")
    tracks, logs, _ = _run(tmp_path)
    assert tracks == []
    assert sum("Required tool 'fpcalc' not found" in line for line in logs) == 1
    assert logs[-1] == "10 file(s) skipped: not in the fingerprint cache and fpcalc is unavailable."


def test_cancel_stops_between_files(tmp_path, monkeypatch):
    import chromaprint_utils

    cancel = threading.Event()
    calls = []

    def fpcalc(path, **_kw):
        calls.append(path)
        cancel.set()
        return fake_fingerprint(path)

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", fpcalc)
    for i in range(5):
        _write(tmp_path / "A" / f"t{i}.mp3")
    _gather()(str(tmp_path), str(tmp_path / "Docs" / "fp.db"), lambda m: None, lambda p: None, cancel)
    assert len(calls) == 1


def test_summarize_execution_surfaces_preflight_reasons(tmp_path):
    from duplicate_consolidation_executor import ExecutionAction, ExecutionResult
    from gui.workspaces.duplicates import summarize_execution

    blocked = ExecutionResult(
        success=False,
        actions=[ExecutionAction("preflight", "execution", "blocked",
                                 "Review-required groups present; enable override to continue.")],
        playlist_reports=[], quarantine_index={}, report_paths={},
    )
    ok, message = summarize_execution(blocked)
    assert not ok
    assert message == ("Execution blocked — nothing was changed. "
                       "Review-required groups present; enable override to continue.")

    done = ExecutionResult(
        success=True,
        actions=[
            ExecutionAction("loser_cleanup", "/a.mp3", "success", "Loser quarantined.",
                            metadata={"disposition": "quarantine"}),
            ExecutionAction("loser_cleanup", "/b.mp3", "success", "Loser deleted.",
                            metadata={"disposition": "delete"}),
            ExecutionAction("loser_cleanup", "/c.mp3", "skipped", "Loser retained by plan disposition.",
                            metadata={"disposition": "retain", "skip_reason": "retain"}),
        ],
        playlist_reports=[], quarantine_index={}, report_paths={},
    )
    assert summarize_execution(done) == (True, "Execution complete: 1 quarantined, 1 deleted, 1 kept in place.")
