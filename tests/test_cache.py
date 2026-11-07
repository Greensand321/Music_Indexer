import os
import sqlite3

import pytest

import fingerprint_cache
from fingerprint_cache import (
    flush_cache,
    get_cached_fingerprint,
    get_fingerprint,
    store_fingerprint,
)
from cache_prewarmer import prewarm_cache


@pytest.fixture(autouse=True)
def _reset_cache_state():
    """Each test starts with no background writer and an empty in-process memo."""
    fingerprint_cache.shutdown_fingerprint_writer()
    fingerprint_cache._memo_clear()
    yield
    fingerprint_cache.shutdown_fingerprint_writer()
    fingerprint_cache._memo_clear()

def test_get_fingerprint_cache(tmp_path):
    db = tmp_path / "fp.db"
    path = tmp_path / "a.mp3"
    path.write_text("x")
    calls = []
    def compute(p):
        calls.append(p)
        return 1, "hash"
    fp1 = get_fingerprint(str(path), str(db), compute)
    assert fp1 == "hash"
    assert calls == [str(path)]
    fp2 = get_fingerprint(str(path), str(db), compute)
    assert fp2 == "hash"
    assert calls == [str(path)]
    flush_cache(str(db))
    fp3 = get_fingerprint(str(path), str(db), compute)
    assert calls == [str(path), str(path)]


def test_lru_avoids_db(tmp_path, monkeypatch):
    db = tmp_path / "fp.db"
    path = tmp_path / "a.mp3"
    path.write_text("x")

    def compute(_):
        return 1, "hash"

    calls = []
    orig_connect = sqlite3.connect

    def fake_connect(*args, **kwargs):
        calls.append(args[0])
        return orig_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", fake_connect)

    get_fingerprint(str(path), str(db), compute)
    assert len(calls) == 1
    get_fingerprint(str(path), str(db), compute)
    assert len(calls) == 1
    flush_cache(str(db))
    get_fingerprint(str(path), str(db), compute)
    assert len(calls) == 2


def test_prewarm_cache(tmp_path):
    db = tmp_path / "fp.db"
    p1 = tmp_path / "a.mp3"
    p2 = tmp_path / "b.mp3"
    p1.write_text("x")
    p2.write_text("y")

    def compute(_):
        return 1, "hash"

    t = prewarm_cache([str(p1), str(p2)], str(db), compute)
    t.join()

    conn = sqlite3.connect(str(db))
    count = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
    conn.close()
    assert count == 2


def test_long_path_support(tmp_path):
    db = tmp_path / "fp.db"
    base = tmp_path
    for i in range(6):
        base = base / ("d" * 50 + str(i))
    base.mkdir(parents=True)
    path = base / "a.mp3"
    path.write_text("x")

    def compute(_):
        return 1, "hash"

    fp = get_fingerprint(str(path), str(db), compute)
    assert fp == "hash"


def test_missing_file_returns_none(tmp_path):
    db = tmp_path / "fp.db"
    path = tmp_path / "a.mp3"
    path.write_text("x")
    path.unlink()

    def compute(_):
        raise AssertionError("compute should not be called")

    fp = get_fingerprint(str(path), str(db), compute)
    assert fp is None


def test_schema_less_db_file_is_an_empty_cache(tmp_path):
    # A DB file that exists but has no ``fingerprints`` table (e.g. created by
    # a bare sqlite3.connect) must read as a cache miss, not raise
    # "no such table".
    db = tmp_path / "fp.db"
    sqlite3.connect(str(db)).close()
    assert db.exists()
    path = tmp_path / "a.mp3"
    path.write_text("x")
    trace = {}

    assert get_cached_fingerprint(str(path), str(db), trace=trace) is None
    assert trace["source"] == "missing"

    calls = []

    def compute(p):
        calls.append(p)
        return 1, "hash"

    assert get_fingerprint(str(path), str(db), compute) == "hash"
    assert calls == [str(path)]
    fingerprint_cache.flush_fingerprint_writes(str(db))
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT path, fingerprint FROM fingerprints").fetchall()
    finally:
        conn.close()
    # On Windows the cache key is the \\?\ long-path form (kept so existing
    # caches stay valid); callers always get their own path back.
    from utils.path_helpers import ensure_long_path
    assert rows == [(ensure_long_path(str(path)), "hash")]


def test_changed_file_is_recomputed_not_served_from_memory(tmp_path):
    db = tmp_path / "fp.db"
    path = tmp_path / "a.mp3"
    path.write_text("x")
    results = iter(["first", "second"])

    def compute(_):
        return 1, next(results)

    assert get_fingerprint(str(path), str(db), compute) == "first"
    path.write_text("changed contents")  # size (and mtime) change
    assert get_fingerprint(str(path), str(db), compute) == "second"


def test_stored_fingerprint_is_visible_before_flush(tmp_path):
    db = tmp_path / "fp.db"
    path = tmp_path / "a.mp3"
    path.write_text("x")
    assert store_fingerprint(str(path), str(db), 1, "stored")

    def compute(_):
        raise AssertionError("compute should not be called")

    assert get_fingerprint(str(path), str(db), compute) == "stored"


def test_prewarm_cache_persists_before_thread_finishes(tmp_path):
    # Regression for the prewarm race: the worker thread used to finish while
    # its fingerprints were still queued on the batching writer.
    for attempt in range(20):
        db = tmp_path / f"fp{attempt}.db"
        paths = []
        for idx in range(5):
            p = tmp_path / f"t{attempt}_{idx}.mp3"
            p.write_text(str(idx))
            paths.append(str(p))
        t = prewarm_cache(paths, str(db), lambda _p: (1, "hash"))
        t.join()
        conn = sqlite3.connect(str(db))
        try:
            count = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
        finally:
            conn.close()
        assert count == 5
