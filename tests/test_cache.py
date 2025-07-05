import os
from fingerprint_cache import get_fingerprint, flush_cache

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

