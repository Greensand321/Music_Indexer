"""Library Sync keeps decoded fingerprints in its cache (area: improve).

Building the candidate index decodes every library print (pure Python,
~1.5 ms each: ~30 s for 20k tracks) on every Build Plan. The decoded items
are now stored in the vendored fingerprint cache's ``decoded`` column, next
to the compressed print, and reused while the file's path, mtime and size
match. These tests pin down that the new path gives exactly the old results,
that old caches migrate, and that a changed file or print is re-decoded.
"""
from __future__ import annotations

import random
import sqlite3

import pytest

import library_sync
from _helpers.dupes_chromaprint import compress_chromaprint, fake_fingerprint
from library_sync_indexer_engine import fingerprint_cache as vcache
from library_sync_indexer_engine import near_duplicate_detector as nd
from library_sync_types import TrackRecord


def _walk(n, seed):
    r = random.Random(seed)
    cur = r.getrandbits(32)
    out = []
    for _ in range(n):
        for _ in range(r.choice((3, 4, 5, 6, 7))):
            cur ^= 1 << r.randrange(32)
        out.append(cur)
    return out


def _rec(path, fp, length=200.0, ext=".mp3", mtime=1.0, size=10):
    return TrackRecord(
        track_id=path, path=path, normalized_path=path, ext=ext, bitrate=128000,
        size=size, mtime=mtime, fingerprint=fp, tags={}, duration=120, audio_length=length,
    )


@pytest.fixture
def corpus():
    library = []
    raws = []
    for i in range(60):
        raw = _walk(400 + i, f"lib{i}")
        raws.append(raw)
        library.append(_rec(f"/lib/{i:03d}.mp3", compress_chromaprint(raw), length=150.0 + i))
    # odd ones: plain integer prints, fpcalc's empty print, no print, a silent print
    library.append(_rec("/lib/plain.mp3", "1 2 3 4 5 6 7 8 9", length=99.0))
    library.append(_rec("/lib/empty.mp3", "AQAAAA", length=99.0))
    library.append(_rec("/lib/none.mp3", None, length=99.0))
    library.append(_rec("/lib/silent.mp3", compress_chromaprint([7] * 200), length=99.0))
    incoming = []
    for j, k in enumerate((3, 17, 42)):
        raw = list(raws[k])
        r = random.Random(j)
        for _ in range(len(raw) // 4):
            raw[r.randrange(len(raw))] ^= 1 << r.randrange(32)
        incoming.append(_rec(f"/inc/dup{j}.flac", compress_chromaprint(raw), 150.0 + k, ".flac"))
    incoming.append(_rec("/inc/new.flac", compress_chromaprint(_walk(500, "new")), 170.0, ".flac"))
    incoming.append(_rec("/inc/plain.flac", "1 2 3 4 5 6 7 9 9", 99.0, ".flac"))
    return library, incoming


def _write_rows(db, records, *, old_schema=False):
    conn = sqlite3.connect(db)
    if old_schema:
        conn.execute(
            "CREATE TABLE fingerprints (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, "
            "duration INT, fingerprint TEXT)"
        )
    else:
        library_sync._ensure_db(str(db)).close()
    conn.executemany(
        "INSERT OR REPLACE INTO fingerprints (path, mtime, size, duration, fingerprint) VALUES (?,?,?,?,?)",
        [(r.path, r.mtime, r.size, r.duration, r.fingerprint) for r in records],
    )
    conn.commit()
    conn.close()


def _build(library, db=None):
    nd._decode_cached.cache_clear()
    store = library_sync._DecodedPrintStore(str(db)) if db else None
    try:
        return library_sync._CandidateIndex(library, decoded_store=store), store
    finally:
        if store:
            store.close()


def _summary(results):
    return [
        (m.incoming.path, m.status, m.existing.path if m.existing else None, m.distance,
         [(c.distance, c.track.path) for c in m.candidates])
        for m in results
    ]


def _match(library, incoming, index):
    return _summary(library_sync._match_tracks(
        incoming, library, library_sync.DEFAULT_FP_THRESHOLDS, library_sync.FORMAT_PRIORITY,
        mixed_codec_threshold_boost=0.03, index=index,
    ))


def _same_index(a, b):
    assert a._postings == b._postings
    assert a._windows == b._windows
    assert a._by_text == b._by_text
    assert a._chromaprint_pool._indices == b._chromaprint_pool._indices
    assert a._plain_pool._indices == b._plain_pool._indices


def _decoded_rows(db):
    with sqlite3.connect(db) as conn:
        return dict(conn.execute("SELECT path, decoded FROM fingerprints"))


def test_old_and_new_paths_give_identical_indexes_and_matches(corpus, tmp_path):
    library, incoming = corpus
    db = tmp_path / "fp.db"
    _write_rows(db, library)
    old, _ = _build(library)
    cold, s1 = _build(library, db)        # decodes and stores
    warm, s2 = _build(library, db)        # reads every print back
    _same_index(old, cold)
    _same_index(old, warm)
    assert s1.hits == 0 and s1.decoded > 0
    assert s2.decoded == 0 and s2.hits == s1.decoded
    expected = _match(library, incoming, old)
    assert _match(library, incoming, warm) == expected
    assert [s for _p, s, *_ in expected].count(library_sync.MatchStatus.NEW) >= 1
    assert sum(1 for _p, _s, e, *_ in expected if e) >= 3


def test_warm_run_does_not_decode_the_library(corpus, tmp_path, monkeypatch):
    library, _incoming = corpus
    db = tmp_path / "fp.db"
    _write_rows(db, library)
    _build(library, db)
    calls = []
    real = nd._decoded_packed
    monkeypatch.setattr(nd, "_decoded_packed", lambda text: calls.append(text) or real(text))
    _build(library, db)
    assert calls == []


def test_old_schema_is_migrated_and_rows_decoded_lazily(corpus, tmp_path):
    library, _incoming = corpus
    db = tmp_path / "old.db"
    _write_rows(db, library, old_schema=True)
    with sqlite3.connect(db) as conn:
        assert "decoded" not in {r[1] for r in conn.execute("PRAGMA table_info(fingerprints)")}
    _build(library, db)
    with sqlite3.connect(db) as conn:
        assert "decoded" in {r[1] for r in conn.execute("PRAGMA table_info(fingerprints)")}
        # nothing else changed: fingerprints, mtimes and sizes are as written
        rows = {p: (m, s, f) for p, m, s, f in conn.execute("SELECT path, mtime, size, fingerprint FROM fingerprints")}
    assert rows == {r.path: (r.mtime, r.size, r.fingerprint) for r in library}
    decoded = _decoded_rows(db)
    for rec in library[:60]:
        stored = vcache.unpack_decoded(rec.fingerprint, decoded[rec.path])
        assert stored == nd._decoded_packed(rec.fingerprint)
    # the plain integer print is "not Chromaprint" and recorded as such
    assert vcache.unpack_decoded("1 2 3 4 5 6 7 8 9", decoded["/lib/plain.mp3"]) is None
    assert decoded["/lib/none.mp3"] is None and decoded["/lib/empty.mp3"] is None


def test_changed_file_is_decoded_again_and_not_stored(corpus, tmp_path, monkeypatch):
    library, _incoming = corpus
    db = tmp_path / "fp.db"
    _write_rows(db, library)
    _build(library, db)
    before = _decoded_rows(db)
    # The file on disk changed since the cache row was written (the scan would
    # have re-fingerprinted it; here the record simply disagrees with the row).
    changed = list(library)
    changed[5] = _rec(library[5].path, library[5].fingerprint, mtime=2.0)
    calls = []
    real = nd._decoded_packed
    monkeypatch.setattr(nd, "_decoded_packed", lambda text: calls.append(text) or real(text))
    _index, store = _build(changed, db)
    assert calls == [library[5].fingerprint]
    assert store.hits == 61 and store.decoded == 1
    assert _decoded_rows(db) == before        # an invalid row is never written to


def test_new_fingerprint_for_the_same_row_is_not_paired_with_a_stale_decode(corpus, tmp_path):
    library, incoming = corpus
    db = tmp_path / "fp.db"
    _write_rows(db, library)
    _build(library, db)
    # Another statement rewrites the print in place (mtime/size unchanged):
    other = fake_fingerprint("something else entirely")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE fingerprints SET fingerprint=? WHERE path=?", (other, library[3].path))
    rewritten = list(library)
    rewritten[3] = _rec(library[3].path, other, length=library[3].audio_length)
    fresh, _ = _build(rewritten)
    via_cache, store = _build(rewritten, db)
    _same_index(fresh, via_cache)
    assert store.decoded == 1
    assert _match(rewritten, incoming, via_cache) == _match(rewritten, incoming, fresh)
    # and the corrected decode was written back
    stored = vcache.unpack_decoded(other, _decoded_rows(db)[library[3].path])
    assert stored == nd._decoded_packed(other)


def test_insert_or_replace_resets_the_decoded_column(tmp_path):
    db = tmp_path / "fp.db"
    fp = compress_chromaprint(_walk(300, "x"))
    rec = _rec("/lib/x.mp3", fp)
    _write_rows(db, [rec])
    _build([rec], db)
    assert _decoded_rows(db)[rec.path] is not None
    _write_rows(db, [rec])                    # what a re-fingerprint does
    assert _decoded_rows(db)[rec.path] is None


def test_malformed_blobs_are_ignored():
    fp = compress_chromaprint(_walk(100, "y"))
    good = vcache.pack_decoded(fp, nd._decoded_packed(fp))
    assert vcache.unpack_decoded(fp, good) == nd._decoded_packed(fp)
    for blob in (None, b"", b"\x02" + good[1:], good[:-1], "text", good[:5]):
        assert not vcache.is_stored(vcache.unpack_decoded(fp, blob)), blob


def test_unwritable_cache_only_costs_speed(corpus, tmp_path):
    library, incoming = corpus
    db = tmp_path / "fp.db"
    _write_rows(db, library)
    store = library_sync._DecodedPrintStore(str(db))
    store._conn.execute("PRAGMA query_only = ON")
    try:
        index = library_sync._CandidateIndex(library, decoded_store=store)
    finally:
        store.close()                         # the failed write is swallowed
    old, _ = _build(library)
    _same_index(old, index)


@pytest.mark.skipif(not __import__("library_factory").ffmpeg_available(), reason="needs ffmpeg")
def test_compare_libraries_stores_decodes_and_reuses_them(tmp_path, monkeypatch):
    from _helpers.syncmatch_library import build_sync_scenario, install_title_fingerprints

    install_title_fingerprints(monkeypatch)
    sc = build_sync_scenario(tmp_path / "case")
    db = tmp_path / "fp.db"

    def compare():
        return library_sync.compare_libraries(
            str(sc.library), str(sc.incoming), str(db), include_match_objects=True,
        )

    first = compare()
    lib_paths = set(first["existing"])
    stored = {p: b for p, b in _decoded_rows(db).items() if p in lib_paths}
    assert len(stored) == 73 and all(stored.values())

    calls = []
    real = nd._decoded_packed
    monkeypatch.setattr(nd, "_decoded_packed", lambda text: calls.append(text) or real(text))
    nd._decode_cached.cache_clear()
    built = []
    real_init = library_sync._CandidateIndex.__init__

    def spy(self, records, cancel_event=None, decoded_store=None):
        before = len(calls)
        real_init(self, records, cancel_event=cancel_event, decoded_store=decoded_store)
        built.append(len(calls) - before)

    monkeypatch.setattr(library_sync._CandidateIndex, "__init__", spy)
    second = compare()
    assert built == [0]                       # no library print decoded while building the index
    assert _summary(second["match_objects"]) == _summary(first["match_objects"])
