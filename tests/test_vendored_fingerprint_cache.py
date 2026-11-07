"""Library Sync's vendored fingerprint cache / generator never serve a stale print.

* ``fingerprint_cache.get_fingerprint`` was wrapped in ``@lru_cache`` keyed on
  the path alone, so a file edited mid-process kept its old fingerprint even
  though the SQLite row is checked against (mtime, size).
* ``fingerprint_generator.compute_fingerprints_parallel`` reused a cached row
  whenever the mtime matched, ignoring a size change.
* ``compute_fingerprint_for_file`` passed pyacoustid's ``bytes`` through, which
  ``TrackRecord.signature()`` cannot split; it now returns text.
"""
from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

from library_sync_indexer_engine import fingerprint_cache as vcache
from library_sync_indexer_engine import fingerprint_generator as vgen


def _counter(*values):
    calls = []

    def compute(path):
        calls.append(path)
        return 1, values[min(len(calls), len(values)) - 1]

    return compute, calls


def test_get_fingerprint_sees_a_file_edited_mid_process(tmp_path):
    song = tmp_path / "song.flac"
    song.write_bytes(b"a" * 10)
    db = tmp_path / "Docs" / "fp.db"
    compute, calls = _counter("first", "second")

    assert vcache.get_fingerprint(str(song), str(db), compute) == "first"
    assert vcache.get_fingerprint(str(song), str(db), compute) == "first"
    assert len(calls) == 1          # unchanged file: served from cache

    song.write_bytes(b"b" * 20)     # edited: new size (and mtime)
    os.utime(song, (1_700_000_000, 1_700_000_000))
    assert vcache.get_fingerprint(str(song), str(db), compute) == "second"
    assert len(calls) == 2


def test_get_fingerprint_sees_a_same_size_edit_with_new_mtime(tmp_path):
    song = tmp_path / "song.flac"
    song.write_bytes(b"a" * 10)
    os.utime(song, (1_600_000_000, 1_600_000_000))
    db = tmp_path / "fp.db"
    compute, _calls = _counter("first", "second")
    assert vcache.get_fingerprint(str(song), str(db), compute) == "first"
    song.write_bytes(b"b" * 10)
    os.utime(song, (1_600_000_100, 1_600_000_100))
    assert vcache.get_fingerprint(str(song), str(db), compute) == "second"


def test_flush_cache_still_clears(tmp_path):
    song = tmp_path / "song.flac"
    song.write_bytes(b"a")
    db = tmp_path / "fp.db"
    compute, calls = _counter("first", "second")
    vcache.get_fingerprint(str(song), str(db), compute)
    vcache.flush_cache(str(db))
    assert vcache.get_fingerprint(str(song), str(db), compute) == "second"
    assert len(calls) == 2


def test_compute_fingerprint_for_file_returns_text(tmp_path, monkeypatch):
    song = tmp_path / "song.flac"
    song.write_bytes(b"x")
    monkeypatch.setattr(vgen.acoustid, "fingerprint_file", lambda *a, **k: (12.0, b"AQAAAA"), raising=False)
    path, duration, fp, err = vgen.compute_fingerprint_for_file((str(song), "", {}))
    assert err is None and fp == "AQAAAA" and isinstance(fp, str)


def test_parallel_reuse_requires_matching_size(tmp_path, monkeypatch):
    song = tmp_path / "song.flac"
    song.write_bytes(b"a" * 10)
    db = tmp_path / "fp.db"
    mtime = os.path.getmtime(song)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE fingerprints (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, duration INT, fingerprint TEXT)"
    )
    # A row written before the file grew, with the same mtime (e.g. a tag
    # editor that preserves timestamps).
    conn.execute("INSERT INTO fingerprints VALUES (?, ?, ?, ?, ?)", (str(song), mtime, 5, 1, "stale"))
    conn.commit()
    conn.close()

    monkeypatch.setattr(vgen, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(vgen, "compute_fingerprint_for_file", lambda item: (item[0], 2, "fresh", None))
    results = vgen.compute_fingerprints_parallel([str(song)], str(db), log_callback=lambda _m: None)
    assert results == [(str(song), 2, "fresh")]

    # ...and the row it wrote carries the size, so the next call reuses it.
    monkeypatch.setattr(vgen, "compute_fingerprint_for_file", lambda item: (item[0], 3, "recomputed", None))
    assert vgen.compute_fingerprints_parallel([str(song)], str(db), log_callback=lambda _m: None) == [
        (str(song), 2, "fresh")
    ]
