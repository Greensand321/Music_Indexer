"""The fingerprint cache must degrade to "cache miss", never break a scan.

Before:
* a corrupt cache file made ``ensure_fingerprint_cache`` raise
  ``DatabaseError: file is not a database`` — every Duplicates scan failed —
  and the cached-lookup helpers only caught ``OperationalError``;
* rows from before the mtime/size columns (NULL there) raised ``TypeError``
  from ``get_cached_fingerprint``/``get_cached_fingerprint_metadata``;
* one row sqlite could not bind killed the writer thread: every later write
  was lost and ``flush_fingerprint_writes(db)`` (no timeout) hung forever;
* a writer replaced by another cache path dropped writes still aimed at it.
"""
from __future__ import annotations

import glob
import os
import sqlite3
import threading

import pytest

import fingerprint_cache as fc


@pytest.fixture(autouse=True)
def _fresh_writer():
    fc.shutdown_fingerprint_writer()
    fc._memo_clear()
    yield
    fc.shutdown_fingerprint_writer()
    fc._memo_clear()


def _audio(tmp_path, name="song.mp3", data=b"audio"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _row(db, path):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT fingerprint FROM fingerprints WHERE path=?", (path,)).fetchone()
    finally:
        conn.close()


def test_corrupt_cache_is_set_aside_and_rebuilt(tmp_path):
    db = str(tmp_path / "Docs" / ".duplicate_fingerprints.db")
    os.makedirs(os.path.dirname(db))
    with open(db, "wb") as fh:
        fh.write(b"this is not an sqlite database" * 100)
    song = _audio(tmp_path)

    # Lookups on the damaged file are cache misses, not exceptions.
    assert fc.get_cached_fingerprint(song, db) is None
    assert fc.get_cached_fingerprint_metadata(song, db) == (None, None)

    fc.ensure_fingerprint_cache(db)  # raised DatabaseError

    set_aside = glob.glob(db + ".corrupt-*")
    assert len(set_aside) == 1
    with open(set_aside[0], "rb") as fh:
        assert fh.read().startswith(b"this is not an sqlite database")
    assert fc.store_fingerprint(song, db, None, "FP-1") is True
    fc.flush_fingerprint_writes(db, timeout=5)
    fc._memo_clear()
    assert fc.get_cached_fingerprint(song, db) == "FP-1"


def test_writer_opening_a_corrupt_cache_recovers(tmp_path):
    db = str(tmp_path / "fp.db")
    with open(db, "wb") as fh:
        fh.write(b"\x00garbage" * 512)
    song = _audio(tmp_path)
    assert fc.store_fingerprint(song, db, None, "FP-2") is True  # raised from _get_writer
    fc.flush_fingerprint_writes(db, timeout=5)
    assert _row(db, fc.ensure_long_path(song)) == ("FP-2",)


def test_rows_without_mtime_or_size_are_a_miss(tmp_path):
    db = str(tmp_path / "old.db")
    song = _audio(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE fingerprints (path TEXT PRIMARY KEY, fingerprint TEXT)")
    conn.execute("INSERT INTO fingerprints VALUES (?, ?)", (fc.ensure_long_path(song), "OLD"))
    conn.commit()
    conn.close()
    fc.ensure_fingerprint_cache(db)  # schema upgrade adds NULL mtime/size

    assert fc.get_cached_fingerprint(song, db) is None  # raised TypeError
    assert fc.get_cached_fingerprint_metadata(song, db) == (None, None)
    assert fc.get_fingerprint(song, db, lambda p: (None, "NEW")) == "NEW"


def test_one_bad_row_does_not_kill_the_writer(tmp_path):
    db = str(tmp_path / "fp.db")
    good_before, good_after = _audio(tmp_path, "a.mp3"), _audio(tmp_path, "b.mp3")
    assert fc.store_fingerprint(good_before, db, None, "GOOD-1")
    writer = fc._get_writer(db)
    writer.enqueue("/bad.mp3", 1.0, 1, None, object())  # sqlite cannot bind this
    fc.flush_fingerprint_writes(db, timeout=5)
    assert fc.store_fingerprint(good_after, db, None, "GOOD-2")

    done = threading.Event()

    def flush_without_timeout():
        fc.flush_fingerprint_writes(db)  # hung forever once the thread had died
        done.set()

    threading.Thread(target=flush_without_timeout, daemon=True).start()
    assert done.wait(5), "flush_fingerprint_writes hung"
    assert _row(db, fc.ensure_long_path(good_before)) == ("GOOD-1",)
    assert _row(db, fc.ensure_long_path(good_after)) == ("GOOD-2",)


def test_writes_to_a_replaced_writer_are_not_lost(tmp_path):
    db1, db2 = str(tmp_path / "one.db"), str(tmp_path / "two.db")
    song = _audio(tmp_path)
    old_writer = fc._get_writer(db1)  # e.g. a worker thread holds this one…
    fc._get_writer(db2)               # …while another scan switches the current cache
    old_writer.enqueue(fc.ensure_long_path(song), 1.0, 5, None, "LATE")
    assert _row(db1, fc.ensure_long_path(song)) == ("LATE",)


def test_cache_in_a_non_ascii_folder(tmp_path):
    db = str(tmp_path / "Müsik" / "キャッシュ" / "fp.db")
    song = _audio(tmp_path / "Bibliothèque", "café.flac")
    fc.ensure_fingerprint_cache(db)
    assert fc.store_fingerprint(song, db, None, "FP-é", tags={"title": "Café"})
    fc.flush_fingerprint_writes(db, timeout=5)
    fc._memo_clear()
    fp, meta = fc.get_cached_fingerprint_metadata(song, db)
    assert fp == "FP-é" and meta["tags"] == {"title": "Café"}


def test_modified_file_is_a_cache_miss(tmp_path):
    db = str(tmp_path / "fp.db")
    song = _audio(tmp_path)
    assert fc.store_fingerprint(song, db, None, "BEFORE")
    fc.flush_fingerprint_writes(db, timeout=5)
    with open(song, "ab") as fh:
        fh.write(b" retagged")
    assert fc.get_cached_fingerprint(song, db) is None
    assert fc.get_cached_fingerprint_metadata(song, db) == (None, None)
    assert fc.get_fingerprint(song, db, lambda p: (None, "AFTER")) == "AFTER"
