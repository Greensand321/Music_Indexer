"""Cached fingerprints made with the old silence padding are not reused.

``audio_norm.normalize_for_fp`` used to pad tracks shorter than the
fingerprint window with silence, so the prints of *different* short tracks
agreed on the padding (two unrelated 5-second clips ~0.012, under the 0.02
exact threshold). New prints are not padded, but rows already cached were —
and the caches kept serving them until the file changed. Rows written now
carry ``fp_scheme``; an old row is distrusted only when its print ends in a
run of padding silence, so only those short tracks are fingerprinted again.
"""
from __future__ import annotations

import os
import sqlite3
import struct
import random

import pytest

import fingerprint_cache as fc
import near_duplicate_detector as nd
from _helpers.dupes_chromaprint import compress_chromaprint, fake_raw_print
from _helpers.dupes_plans import write

SILENCE = 627964279  # Chromaprint's sub-fingerprint for digital silence


def _padded(seed: str, audio_items: int = 40, total: int = 950) -> str:
    return compress_chromaprint(fake_raw_print(seed, audio_items) + [SILENCE] * (total - audio_items))


@pytest.fixture(autouse=True)
def _fresh_writer():
    fc.shutdown_fingerprint_writer()
    fc._memo_clear()
    yield
    fc.shutdown_fingerprint_writer()
    fc._memo_clear()


def test_padding_is_recognised_and_ordinary_prints_are_not():
    assert nd.has_silence_padding(_padded("clip"))
    assert not nd.has_silence_padding(compress_chromaprint(fake_raw_print("song", 950)))
    assert not nd.has_silence_padding(compress_chromaprint(fake_raw_print("x", 100) + [SILENCE] * 30))
    assert not nd.has_silence_padding("1 2 3 4") and not nd.has_silence_padding(None)
    # Two different padded clips really do look like the same recording.
    assert nd.fingerprint_distance(_padded("clip A"), _padded("clip B")) < 0.03


def test_padding_from_real_chromaprint_is_recognised():
    try:
        import chromaprint

        fingerprinter = chromaprint.Fingerprinter()
    except Exception as exc:  # libchromaprint missing (e.g. Windows CI)
        pytest.skip(f"libchromaprint unavailable: {exc}")
    rate = 44100
    rnd = random.Random(3)
    clip = b"".join(struct.pack("<h", rnd.randint(-9000, 9000)) for _ in range(rate * 5))
    for padding_seconds, expected in ((115, True), (0, False)):
        fingerprinter = chromaprint.Fingerprinter()
        fingerprinter.start(rate, 1)
        fingerprinter.feed(clip + b"\x00\x00" * (rate * padding_seconds))
        fp = fingerprinter.finish()
        fp = fp.decode("ascii") if isinstance(fp, bytes) else fp
        assert nd.has_silence_padding(fp) is expected


def _old_cache(db: str, rows):
    """A cache as the pre-fix code left it: no fp_scheme column."""
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE fingerprints (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, duration INT, "
        "fingerprint TEXT, ext TEXT, bitrate INT, sample_rate INT, bit_depth INT, channels INT, "
        "codec TEXT, container TEXT, tags_json TEXT, artwork_json TEXT, normalized_artist TEXT, "
        "normalized_title TEXT, normalized_album TEXT)"
    )
    for path, fp in rows:
        key = fc.ensure_long_path(path)
        conn.execute(
            "INSERT INTO fingerprints (path, mtime, size, fingerprint, ext, tags_json) VALUES (?, ?, ?, ?, ?, ?)",
            (key, os.path.getmtime(key), os.path.getsize(key), fp, os.path.splitext(path)[1], "{}"),
        )
    conn.commit()
    conn.close()


def test_old_padded_rows_are_stale_and_other_old_rows_stay_valid(tmp_path):
    db = str(tmp_path / "fp.db")
    short = str(write(tmp_path / "Intro.mp3"))
    long_track = str(write(tmp_path / "Song.mp3"))
    song_fp = compress_chromaprint(fake_raw_print("song", 950))
    _old_cache(db, [(short, _padded("intro")), (long_track, song_fp)])

    assert fc.get_cached_fingerprint(short, db) is None
    assert fc.get_cached_fingerprint_metadata(short, db) == (None, None)
    assert fc.get_cached_fingerprint(long_track, db) == song_fp  # read-only DB, column absent: no error
    assert fc.get_cached_fingerprint_metadata(long_track, db)[0] == song_fp

    calls = []

    def recompute(path):
        calls.append(path)
        # Even a fresh print that happens to end in silence is trusted once
        # it was written with the current scheme (no endless re-fingerprinting).
        return None, _padded("intro, unpadded but silent at the end")

    fc.ensure_fingerprint_cache(db)
    new_fp = fc.get_fingerprint(short, db, recompute)
    assert calls == [short]
    assert fc.get_fingerprint(long_track, db, recompute) == song_fp
    assert calls == [short]
    fc.flush_fingerprint_writes(db, timeout=5)
    fc._memo_clear()
    assert fc.get_cached_fingerprint(short, db) == new_fp
    assert fc.get_fingerprint(short, db, recompute) == new_fp
    assert calls == [short]


def test_duplicates_scan_refingerprints_padded_short_tracks(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import threading

    import chromaprint_utils
    from duplicate_consolidation import build_consolidation_plan
    from gui.workspaces.duplicates import _gather_tracks_for_plan

    intro_a = str(write(tmp_path / "Artist" / "Album A" / "01 - Intro.mp3", b"intro a"))
    intro_b = str(write(tmp_path / "Artist" / "Album B" / "01 - Intro.mp3", b"intro b"))
    db = str(tmp_path / "Docs" / ".duplicate_fingerprints.db")
    os.makedirs(os.path.dirname(db))
    _old_cache(db, [(intro_a, _padded("intro a")), (intro_b, _padded("intro b"))])
    fresh = {
        intro_a: compress_chromaprint(fake_raw_print("intro a", 40)),
        intro_b: compress_chromaprint(fake_raw_print("intro b", 40)),
    }
    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", lambda path, **_kw: fresh[path])

    tracks = _gather_tracks_for_plan(str(tmp_path), db, lambda m: None, lambda p: None, threading.Event())

    assert {t["path"]: t["fingerprint"] for t in tracks} == fresh
    for t in tracks:
        t["tags"] = {"artist": "Artist", "title": "Intro"}
    assert build_consolidation_plan(tracks).groups == []


def test_fingerprint_generator_does_not_reuse_padded_rows(tmp_path):
    import fingerprint_generator as fg

    db = str(tmp_path / "fp.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE fingerprints (path TEXT PRIMARY KEY, mtime REAL, duration INT, fingerprint TEXT)")
    conn.execute("INSERT INTO fingerprints VALUES ('/lib/intro.mp3', 1.0, 120, ?)", (_padded("intro"),))
    song_fp = compress_chromaprint(fake_raw_print("song", 950))
    conn.execute("INSERT INTO fingerprints VALUES ('/lib/song.mp3', 1.0, 120, ?)", (song_fp,))
    fg._ensure_scheme_column(conn)
    entries = fg._load_cached_entries(conn)
    conn.close()
    assert "/lib/intro.mp3" not in entries
    assert entries["/lib/song.mp3"] == (1.0, 120, song_fp)
