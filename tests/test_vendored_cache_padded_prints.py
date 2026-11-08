"""Library Sync's vendored cache stops trusting old silence-padded prints.

Library Sync's generator fingerprints through ``audio_norm.normalize_for_fp``
with a window (120 s by default). Until that stopped padding, every shorter
track was padded with silence up to the window, so the prints of *different*
short tracks agreed on the padding (unrelated 5-second clips ~0.012, 20 s
~0.05, 40 s ~0.1 — all under Library Sync's 0.2 threshold). Its cache kept
serving those prints. Their stored duration equals the window either way, so
only the print itself tells them apart.

Now (a port of the root cache fix): rows carry ``fp_scheme``; rows written now
are always trusted, and an old row (NULL scheme) is a cache miss only when
its print ends in a run of padding silence. Every reader of the vendored
cache applies it — ``library_sync._scan_folder``, the vendored generator's
own reuse check (which used to hand the padded print straight back) and
``fingerprint_cache.get_fingerprint`` — and every writer of a new print
clears the row's ``decoded`` blob in the same statement.

Also ported: a corrupt cache file is set aside and rebuilt instead of
failing every Library Sync scan, and an unreadable/unbindable row is a cache
miss or a skipped cache write, never a crashed scan.
"""
from __future__ import annotations

import glob
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

import library_sync
from _helpers.dupes_chromaprint import compress_chromaprint, fake_raw_print
from library_sync_indexer_engine import fingerprint_cache as vcache
from library_sync_indexer_engine import fingerprint_generator as vgen
from library_sync_indexer_engine import near_duplicate_detector as vnd

SILENCE = 627964279  # Chromaprint's sub-fingerprint for digital silence
WINDOW_ITEMS = 950    # ~120 s of sub-fingerprints


def _padded(seed: str, audio_items: int = 40) -> str:
    """What the old generator stored for a ~5 s track: audio, then padding."""
    return compress_chromaprint(fake_raw_print(seed, audio_items) + [SILENCE] * (WINDOW_ITEMS - audio_items))


def _unpadded(seed: str, items: int = WINDOW_ITEMS) -> str:
    return compress_chromaprint(fake_raw_print(seed, items))


def _ends_in_silence(seed: str) -> str:
    """A genuine track whose last ~6 s are digital silence (looks padded)."""
    return compress_chromaprint(fake_raw_print(seed, 300) + [SILENCE] * 50)


def _file(folder, name: str, data: bytes = b"not really audio") -> str:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _legacy_cache(db: str, rows, *, decoded_for=None) -> None:
    """A cache as the pre-fix code left it: no fp_scheme column.

    ``decoded_for`` maps a path to the print text its ``decoded`` blob was
    made from (the column existed before ``fp_scheme``).
    """
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE fingerprints (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, "
        "duration INT, fingerprint TEXT, decoded BLOB)"
    )
    for path, fp in rows:
        blob = None
        if decoded_for and path in decoded_for:
            text = decoded_for[path]
            blob = vcache.pack_decoded(text, vnd._decoded_packed(text))
        conn.execute(
            "INSERT INTO fingerprints VALUES (?, ?, ?, ?, ?, ?)",
            (path, os.path.getmtime(path), os.path.getsize(path), 120, fp, blob),
        )
    conn.commit()
    conn.close()


def _rows(db: str):
    conn = sqlite3.connect(db)
    try:
        return {
            path: (fp, scheme, blob)
            for path, fp, scheme, blob in conn.execute(
                "SELECT path, fingerprint, fp_scheme, decoded FROM fingerprints"
            )
        }
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _fresh_memo():
    vcache._memo_clear()
    vnd._decode_cached.cache_clear()
    yield
    vcache._memo_clear()


@pytest.fixture
def real_generator(monkeypatch):
    """The real vendored generator, minus the process pool and fpcalc.

    ``computed`` lists the files actually (re)fingerprinted; each gets a new,
    unpadded print.
    """
    computed: list[str] = []

    def compute(item):
        path = item[0]
        computed.append(path)
        return path, 120, _unpadded("fresh " + os.path.basename(path), 300), None

    monkeypatch.setattr(vgen, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(vgen, "compute_fingerprint_for_file", compute)
    return computed


SETTINGS = {"fingerprint_duration_ms": 120_000}


# ── detection ────────────────────────────────────────────────────────────────

def test_vendored_padding_check_matches_the_root():
    import near_duplicate_detector as root_nd

    for fp in (_padded("clip"), _unpadded("song"), _ends_in_silence("fade"),
               compress_chromaprint(fake_raw_print("x", 100) + [SILENCE] * 30), "1 2 3", None, "AQAAAA"):
        assert vnd.has_silence_padding(fp) == root_nd.has_silence_padding(fp)
    assert vnd.has_silence_padding(_padded("clip"))
    assert not vnd.has_silence_padding(_unpadded("song"))
    # Why it matters: two different padded clips look like one recording.
    assert vnd.fingerprint_distance(_padded("clip A"), _padded("clip B")) < 0.03


def test_trust_rules():
    assert not vcache.fingerprint_trusted(_padded("clip"), None)
    assert vcache.fingerprint_trusted(_padded("clip"), vcache.FP_SCHEME_UNPADDED)
    assert vcache.fingerprint_trusted(_unpadded("song"), None)
    assert vcache.fingerprint_trusted("1,2,3", None)
    assert vcache.fingerprint_trusted(None, None)


# ── Library Sync's scan (the path Build Plan takes) ─────────────────────────

def test_scan_refetches_only_old_padded_rows(tmp_path, real_generator):
    folder = tmp_path / "incoming"
    clip = _file(folder, "interlude.mp3", b"short clip")
    song = _file(folder, "song.mp3", b"full song")
    fade = _file(folder, "fade.mp3", b"song ending in silence")
    db = str(tmp_path / "Docs" / "sync.db")
    os.makedirs(os.path.dirname(db))
    _legacy_cache(db, [(clip, _padded("clip")), (song, _unpadded("song")), (fade, _ends_in_silence("fade"))])

    events = []
    records = library_sync._scan_folder(
        str(folder), db, log_callback=events.append, fingerprint_settings=SETTINGS,
    )
    fps = {os.path.basename(r.path): r.fingerprint for r in records}

    # The padded clip and the (indistinguishable) old row ending in silence
    # are fingerprinted again; the ordinary old row is reused untouched.
    assert sorted(real_generator) == sorted([clip, fade])
    assert fps["song.mp3"] == _unpadded("song")
    assert fps["interlude.mp3"] == _unpadded("fresh interlude.mp3", 300)
    assert not vnd.has_silence_padding(fps["interlude.mp3"])
    assert any('"silence_padding"' in e for e in events if isinstance(e, str))

    rows = _rows(db)
    assert rows[clip][:2] == (_unpadded("fresh interlude.mp3", 300), vcache.FP_SCHEME_UNPADDED)
    assert rows[song][:2] == (_unpadded("song"), None)          # not rewritten

    # Second scan: nothing is fingerprinted again — the rewritten rows are
    # trusted (the old one that really ended in silence too), the rest unchanged.
    real_generator.clear()
    library_sync._scan_folder(str(folder), db, fingerprint_settings=SETTINGS)
    assert real_generator == []


def test_rows_written_now_are_trusted_even_if_they_end_in_silence(tmp_path, real_generator):
    folder = tmp_path / "incoming"
    fade = _file(folder, "fade.mp3")
    db = str(tmp_path / "sync.db")
    conn = library_sync._ensure_db(db)
    vcache.store_fingerprint_row(conn, fade, os.path.getmtime(fade), os.path.getsize(fade), 120,
                                 _ends_in_silence("fade"))
    conn.commit()
    conn.close()
    [rec] = library_sync._scan_folder(str(folder), db, fingerprint_settings=SETTINGS)
    assert real_generator == []
    assert rec.fingerprint == _ends_in_silence("fade")


def test_generator_alone_does_not_reuse_a_padded_row(tmp_path, real_generator):
    """``compute_fingerprints_parallel`` has its own reuse check: it handed the
    padded print straight back even when the caller asked for a new one."""
    clip = _file(tmp_path / "in", "clip.flac")
    song = _file(tmp_path / "in", "song.flac", b"other")
    db = str(tmp_path / "sync.db")
    _legacy_cache(db, [(clip, _padded("clip")), (song, _unpadded("song"))])
    results = dict((p, fp) for p, _d, fp in vgen.compute_fingerprints_parallel(
        [clip, song], db, log_callback=lambda _m: None, fingerprint_duration_ms=120_000,
    ))
    assert real_generator == [clip]
    assert results[song] == _unpadded("song")
    assert not vnd.has_silence_padding(results[clip])
    assert _rows(db)[clip][1] == vcache.FP_SCHEME_UNPADDED


def test_get_fingerprint_recomputes_only_padded_old_rows(tmp_path):
    clip = _file(tmp_path, "clip.flac")
    song = _file(tmp_path, "song.flac", b"other")
    db = str(tmp_path / "sync.db")
    _legacy_cache(db, [(clip, _padded("clip")), (song, _unpadded("song"))])
    calls = []

    def compute(path):
        calls.append(path)
        return 30, _unpadded("fresh", 240)

    assert vcache.get_fingerprint(song, db, compute) == _unpadded("song")
    assert vcache.get_fingerprint(clip, db, compute) == _unpadded("fresh", 240)
    assert calls == [clip]
    vcache._memo_clear()
    assert vcache.get_fingerprint(clip, db, compute) == _unpadded("fresh", 240)
    assert calls == [clip]                                  # the new row is trusted
    assert _rows(db)[clip][1] == vcache.FP_SCHEME_UNPADDED


# ── the decoded blob follows the fingerprint ────────────────────────────────

def test_decoded_blob_is_cleared_with_the_refetched_print(tmp_path, real_generator):
    folder = tmp_path / "library"
    clip = _file(folder, "clip.mp3")
    song = _file(folder, "song.mp3", b"other")
    db = str(tmp_path / "sync.db")
    # Both rows already carry a decoded blob of their (old) print.
    _legacy_cache(db, [(clip, _padded("clip")), (song, _unpadded("song"))],
                  decoded_for={clip: _padded("clip"), song: _unpadded("song")})
    with sqlite3.connect(db) as conn:
        [(old_song_blob,)] = conn.execute("SELECT decoded FROM fingerprints WHERE path=?", (song,))
    assert old_song_blob

    records = library_sync._scan_folder(str(folder), db, fingerprint_settings=SETTINGS)
    rows = _rows(db)
    assert rows[clip][2] is None                 # cleared with the new print
    assert rows[song][2] == old_song_blob        # untouched row keeps its decode

    # The candidate index decodes the new print and stores *that* decode.
    store = library_sync._DecodedPrintStore(db)
    try:
        library_sync._CandidateIndex(records, decoded_store=store)
    finally:
        store.close()
    assert store.hits == 1 and store.decoded == 1
    new_fp, _scheme, blob = _rows(db)[clip]
    assert vcache.unpack_decoded(new_fp, blob) == vnd._decoded_packed(new_fp)
    assert not vcache.is_stored(vcache.unpack_decoded(_padded("clip"), blob))


def test_store_row_clears_a_stale_decode(tmp_path):
    db = str(tmp_path / "sync.db")
    conn = library_sync._ensure_db(db)
    old = _unpadded("old")
    conn.execute(
        "INSERT INTO fingerprints (path, mtime, size, duration, fingerprint, decoded) VALUES (?,?,?,?,?,?)",
        ("/x.mp3", 1.0, 1, 120, old, vcache.pack_decoded(old, vnd._decoded_packed(old))),
    )
    vcache.store_fingerprint_row(conn, "/x.mp3", 1.0, 1, 120, _unpadded("new"))
    conn.commit()
    conn.close()
    assert _rows(db)["/x.mp3"] == (_unpadded("new"), vcache.FP_SCHEME_UNPADDED, None)


# ── robustness ───────────────────────────────────────────────────────────────

def test_corrupt_cache_is_set_aside_and_rebuilt(tmp_path, real_generator):
    folder = tmp_path / "incoming"
    song = _file(folder, "song.mp3")
    db = str(tmp_path / "Docs" / "sync.db")
    os.makedirs(os.path.dirname(db))
    with open(db, "wb") as fh:
        fh.write(b"this is not an sqlite database" * 100)

    [rec] = library_sync._scan_folder(str(folder), db, fingerprint_settings=SETTINGS)  # raised DatabaseError

    assert rec.fingerprint == _unpadded("fresh song.mp3", 300)
    set_aside = glob.glob(db + ".corrupt-*")
    assert len(set_aside) == 1
    with open(set_aside[0], "rb") as fh:
        assert fh.read().startswith(b"this is not an sqlite database")
    assert _rows(db)[song][1] == vcache.FP_SCHEME_UNPADDED       # a working cache again


def test_get_fingerprint_on_a_corrupt_cache(tmp_path):
    song = _file(tmp_path, "song.flac")
    db = str(tmp_path / "sync.db")
    with open(db, "wb") as fh:
        fh.write(b"\x00garbage" * 512)
    assert vcache.get_fingerprint(song, db, lambda _p: (1, "FP")) == "FP"
    assert len(glob.glob(db + ".corrupt-*")) == 1


def test_rows_without_mtime_or_size_are_a_cache_miss(tmp_path, real_generator):
    """Rows from before the mtime/size columns: ``abs(None - mtime)`` raised."""
    folder = tmp_path / "incoming"
    song = _file(folder, "song.mp3")
    db = str(tmp_path / "sync.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE fingerprints (path TEXT PRIMARY KEY, duration INT, fingerprint TEXT)")
    conn.execute("INSERT INTO fingerprints VALUES (?, ?, ?)", (song, 120, _unpadded("old")))
    conn.commit()
    conn.close()
    [rec] = library_sync._scan_folder(str(folder), db, fingerprint_settings=SETTINGS)
    assert real_generator == [song]
    assert rec.fingerprint == _unpadded("fresh song.mp3", 300)


def test_an_unbindable_row_does_not_stop_the_scan(tmp_path, monkeypatch):
    """A value sqlite cannot bind used to raise out of the scan."""
    folder = tmp_path / "incoming"
    a = _file(folder, "a.mp3")
    b = _file(folder, "b.mp3", b"other")

    def compute(item):
        path = item[0]
        duration = object() if path == a else 120     # not bindable
        return path, duration, _unpadded(path, 100), None

    monkeypatch.setattr(vgen, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(vgen, "compute_fingerprint_for_file", compute)
    db = str(tmp_path / "sync.db")
    records = library_sync._scan_folder(str(folder), db, fingerprint_settings=SETTINGS)
    assert {r.path: r.fingerprint for r in records} == {a: _unpadded(a, 100), b: _unpadded(b, 100)}
    assert set(_rows(db)) == {b}
