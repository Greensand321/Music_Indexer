import os
import sqlite3
from fingerprint_cache import get_fingerprint, flush_cache
from cache_prewarmer import prewarm_cache

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

