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

