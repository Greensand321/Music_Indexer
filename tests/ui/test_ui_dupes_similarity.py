"""Similarity Inspector (DUP-05, DUP-15, DUP-16, DUP-19).

DUP-05: pyacoustid hands back ``bytes``; the old distance could only parse
``str``, so every non-identical pair — a FLAC and its own MP3 included —
scored 1.0 "NOT A DUPLICATE".
"""
from __future__ import annotations

import threading

import pytest
from PySide6 import QtWidgets

from _helpers.dupes_chromaprint import real_fingerprints

FPS = real_fingerprints()


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "Lib"
    a = root / "Artist" / "Album" / "Song.flac"
    b = root / "Artist" / "Album" / "Song.mp3"
    for p in (a, b):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(p.suffix.encode() * 100)
    return root, a, b


@pytest.fixture
def fpcalc(monkeypatch):
    """Fake fpcalc with the real signature; records the parameters it got."""
    import chromaprint_utils

    calls = []

    def fake(path, **kwargs):
        calls.append((path, kwargs))
        return FPS["songA.flac"] if str(path).endswith(".flac") else FPS["songA.mp3_64"]

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", fake)
    monkeypatch.setattr(chromaprint_utils, "missing_tools", lambda *a, **k: [])
    return calls


def _run(ws, qtbot, a, b):
    ws._file_a.setText(str(a))
    ws._file_b.setText(str(b))
    ws._run_btn.click()
    qtbot.waitUntil(lambda: ws._worker is None, timeout=20_000)
    return ws._results_text.toPlainText()


def test_flac_and_its_mp3_are_duplicates_with_pyacoustid_bytes(make_workspace, files, qtbot, monkeypatch):
    import acoustid
    import chromaprint_utils

    root, a, b = files
    monkeypatch.setattr(chromaprint_utils, "missing_tools", lambda *a, **k: ["fpcalc"])
    monkeypatch.setattr(
        acoustid, "fingerprint_file",
        lambda path, *a, **k: (30.0, (FPS["songA.flac"] if path.endswith(".flac") else FPS["songA.mp3_64"]).encode()),
    )
    ws = make_workspace("similarity")
    report = _run(ws, qtbot, a, b)
    assert "VERDICT: EXACT DUPLICATE" in report, report
    distance = float(report.split("Fingerprint distance:")[1].split()[0])
    assert distance < 0.02
    assert "libchromaprint via pyacoustid (fpcalc not found)" in report
    assert "Mixed-codec: Yes" in report


def test_same_pipeline_as_the_duplicate_finder(make_workspace, files, qtbot, fpcalc):
    root, a, b = files
    ws = make_workspace("similarity")
    report = _run(ws, qtbot, a, b)
    assert "VERDICT: EXACT DUPLICATE" in report
    assert [kw for _p, kw in fpcalc] == [
        {"trim": True, "start_sec": 0.0, "duration_sec": 120.0, "threshold_db": -50.0, "min_silence_duration": 0.5}
    ] * 2


def test_reads_the_duplicate_finder_cache(make_workspace, files, qtbot, fpcalc):
    from fingerprint_cache import flush_fingerprint_writes, store_fingerprint

    root, a, b = files
    db = str(root / "Docs" / ".duplicate_fingerprints.db")
    store_fingerprint(str(a), db, None, FPS["songA.flac"])
    store_fingerprint(str(b), db, None, FPS["songB.flac"])
    flush_fingerprint_writes(db, timeout=5)

    ws = make_workspace("similarity", library_path=root)
    report = _run(ws, qtbot, a, b)
    assert fpcalc == []                                   # nothing recomputed
    assert report.count("Duplicate Finder cache") == 2
    assert "VERDICT: NOT A DUPLICATE" in report


def test_thresholds_come_from_settings(make_workspace, files, qtbot, fpcalc):
    from config import load_config, save_config

    cfg = load_config()
    cfg.update(exact_duplicate_threshold=0.0005, near_duplicate_threshold=0.0007,
               mixed_codec_threshold_boost=0.0)
    save_config(cfg)
    root, a, b = files
    ws = make_workspace("similarity")
    assert ws._near_t.text() == "0.0007" and not ws._near_t.isEnabled()
    report = _run(ws, qtbot, a, b)
    assert "Near threshold:           0.0007" in report
    assert "Effective threshold:      0.0007" in report
    assert "Thresholds from:          Settings" in report
    assert "VERDICT: NOT A DUPLICATE" in report    # 0.0045 > 0.0007


def test_advanced_overrides_are_applied(make_workspace, files, qtbot, fpcalc):
    root, a, b = files
    ws = make_workspace("similarity")
    ws._adv_group.setChecked(True)
    ws._trim_cb.setChecked(False)
    ws._fp_offset.setText("1500")
    ws._fp_dur.setText("30000")
    ws._silence_thresh.setText("-40")
    ws._silence_min.setText("250")
    ws._near_t.setText("0.0001")
    ws._exact_t.setText("0")
    ws._codec_boost.setText("0")
    report = _run(ws, qtbot, a, b)
    assert {tuple(sorted(kw.items())) for _p, kw in fpcalc} == {tuple(sorted({
        "trim": False, "start_sec": 1.5, "duration_sec": 30.0,
        "threshold_db": -40.0, "min_silence_duration": 0.25,
    }.items()))}
    assert "fpcalc with your overrides" in report
    assert "Thresholds from:          Advanced Overrides" in report
    assert "VERDICT: NOT A DUPLICATE" in report


def test_invalid_override_is_named_not_silently_replaced(make_workspace, files, qtbot, fpcalc, dialogs, threads):
    root, a, b = files
    ws = make_workspace("similarity")
    ws._adv_group.setChecked(True)
    ws._near_t.setText("0,1")
    ws._file_a.setText(str(a))
    ws._file_b.setText(str(b))
    ws._run_btn.click()
    (warning,) = dialogs.calls_of("warning")
    assert "Near threshold" in warning.text and "'0,1'" in warning.text
    assert threads.started == [] and fpcalc == []


def test_clear_discards_a_running_comparison(make_workspace, files, qtbot, monkeypatch, threads):
    import chromaprint_utils

    gate, started = threading.Event(), threading.Event()

    def slow(path, **_kw):
        started.set()
        gate.wait(10)
        return FPS["songA.flac"]

    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", slow)
    monkeypatch.setattr(chromaprint_utils, "missing_tools", lambda *a, **k: [])
    root, a, b = files
    ws = make_workspace("similarity")
    ws._file_a.setText(str(a))
    ws._file_b.setText(str(b))
    ws._run_btn.click()
    assert started.wait(10)
    ws._on_clear()
    assert ws._run_btn.isEnabled()
    gate.set()
    assert threads.wait_all(10_000) == []
    assert ws._results_text.toPlainText() == ""


def test_missing_file_is_reported(make_workspace, files, qtbot, dialogs, threads, tmp_path):
    root, a, _b = files
    ws = make_workspace("similarity")
    ws._file_a.setText(str(a))
    ws._file_b.setText(str(tmp_path / "nope.mp3"))
    ws._run_btn.click()
    (warning,) = dialogs.calls_of("warning")
    assert "nope.mp3" in warning.text
    assert threads.started == []


def test_backend_lossless_set_is_used(make_workspace, tmp_path, qtbot, fpcalc):
    a = tmp_path / "Song.aiff"
    b = tmp_path / "Song.flac"
    for p in (a, b):
        p.write_bytes(b"x")
    ws = make_workspace("similarity")
    report = _run(ws, qtbot, a, b)
    assert "Codec A: .aiff (lossless)" in report
    assert "Mixed-codec: No" in report


def test_a_clip_of_a_song_is_not_called_a_duplicate(make_workspace, files, qtbot, monkeypatch):
    """Compare two files gives the Duplicate Finder's verdict: a 30-second clip's
    print is a prefix of the song's, so the overlap alone scored it identical."""
    import chromaprint_utils
    from _helpers.dupes_chromaprint import compress_chromaprint, fake_raw_print

    full = fake_raw_print("full song", 950)
    _root, a, b = files
    monkeypatch.setattr(chromaprint_utils, "missing_tools", lambda *a, **k: [])
    monkeypatch.setattr(
        chromaprint_utils, "fingerprint_fpcalc",
        lambda path, **kw: compress_chromaprint(full if str(path).endswith(".flac") else full[:240]),
    )
    ws = make_workspace("similarity")
    report = _run(ws, qtbot, a, b)
    assert "VERDICT: NOT A DUPLICATE" in report, report
    assert float(report.split("Overlap-only distance:")[1].split()[0]) == 0.0
