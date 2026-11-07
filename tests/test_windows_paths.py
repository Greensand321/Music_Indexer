"""Windows path behaviour, checked on any OS.

The owner runs AlphaDEX on Windows; CI's Windows job found playlist rewrites
that switched an entry's ``/`` to ``\\``, ``os.path.relpath`` crashing across
drives, mixed separators in the Indexer preview, and ``\\\\?\\`` long paths
leaking out of the fingerprint cache/generators.

Pure path logic runs here under Windows semantics by swapping ``os.path`` for
``ntpath`` (plus ``os.sep`` / ``os.altsep`` / ``os.name``) for the duration of
one call. Anything touching the file system instead simulates the Windows
behaviour (a ``ValueError`` from ``relpath``, a long path that differs from the
caller's string) with real POSIX paths.
"""
from __future__ import annotations

import contextlib
import io
import ntpath
import os

import pytest

from utils import path_helpers


@pytest.fixture
def windows(monkeypatch):
    """``with windows():`` runs its body with Windows path semantics."""

    @contextlib.contextmanager
    def ctx():
        with monkeypatch.context() as m:
            m.setattr(os, "path", ntpath)
            m.setattr(os, "sep", "\\")
            m.setattr(os, "altsep", "/")
            m.setattr(os, "name", "nt")
            yield

    return ctx


# ── long-path prefix ─────────────────────────────────────────────────────────

def test_strip_ext_prefix_strips_the_real_prefix(windows):
    long = "\\\\?\\C:\\Music\\a.flac"
    with windows():
        # It compared against a raw string with one backslash too many, so it
        # never matched and the prefix reached fpcalc and the Opus mirror log.
        assert path_helpers.strip_ext_prefix(long) == "C:\\Music\\a.flac"
        assert path_helpers.ensure_long_path("C:\\Music\\a.flac") == long
        assert path_helpers.ensure_long_path(long) == long
        assert path_helpers.strip_ext_prefix(path_helpers.ensure_long_path("C:/Music/a.flac")) == "C:\\Music\\a.flac"
    if os.name != "nt":
        assert path_helpers.strip_ext_prefix(long) == long  # only ever stripped on Windows


# ── relpath across drives / playlist entries ────────────────────────────────

def test_safe_relpath_falls_back_to_absolute_across_drives(windows):
    with windows():
        with pytest.raises(ValueError):
            os.path.relpath("D:\\Music\\a.mp3", "C:\\Playlists")
        assert path_helpers.safe_relpath("D:\\Music\\a.mp3", "C:\\Playlists") == "D:\\Music\\a.mp3"
        assert path_helpers.safe_relpath("C:\\Music\\a.mp3", "C:\\Music\\Playlists") == "..\\a.mp3"


@pytest.mark.parametrize("like, target, expected", [
    ("../Old/a.mp3", "C:\\Music\\New\\a.mp3", "../New/a.mp3"),          # "/" stays "/"
    ("..\\Old\\a.mp3", "C:\\Music\\New\\a.mp3", "..\\New\\a.mp3"),       # "\\" stays "\\"
    ("C:/Music/Old/a.mp3", "C:\\Music\\New\\a.mp3", "C:/Music/New/a.mp3"),  # absolute stays absolute
    ("C:\\Music\\Old\\a.mp3", "C:\\Music\\New\\a.mp3", "C:\\Music\\New\\a.mp3"),
    ("../Old/a.mp3", "D:\\Moved\\a.mp3", "D:/Moved/a.mp3"),             # other drive: absolute
])
def test_playlist_entry_keeps_the_entry_form(windows, like, target, expected):
    with windows():
        assert path_helpers.playlist_entry(target, "C:\\Music\\Playlists", like=like) == expected


@pytest.mark.skipif(os.name == "nt", reason="checks POSIX semantics: there \\ is a file-name character")
def test_playlist_entry_is_unchanged_off_windows():
    # "\\" is an ordinary file-name character on POSIX: never a separator.
    assert path_helpers.playlist_entry("/m/New/a.mp3", "/m/Playlists", like="..\\Old\\a.mp3") == "../New/a.mp3"
    assert path_helpers.playlist_entry("/m/New/a.mp3", "/m/Playlists", like="/m/Old/a.mp3") == "/m/New/a.mp3"


def test_playlist_rewrite_keeps_each_entry_style_on_windows(windows):
    import playlist_generator as pg

    text = (
        "#EXTM3U\n"
        "../Artist/Album/01 One (1).mp3\n"
        "..\\Artist\\Album\\02 Two (1).mp3\r\n"
        "C:/Music/Artist/Album/03 Three (1).mp3\n"
        "../Artist/Album/04 Four (1).mp3\n"
        "../Artist/Album/05 Untouched.mp3\n"
    )
    with windows():
        def key(p):
            return ntpath.normcase(ntpath.normpath(p))

        album = "C:\\Music\\Artist\\Album\\"
        mapping = {
            key(album + "01 One (1).mp3"): album + "01 One.mp3",
            key(album + "02 Two (1).mp3"): album + "02 Two.mp3",
            key(album + "03 Three (1).mp3"): album + "03 Three.mp3",
            key(album + "04 Four (1).mp3"): "D:\\Elsewhere\\04 Four.mp3",
        }
        new_text, changed = pg._rewrite_playlist_lines(text, "C:\\Music\\Playlists", mapping)
    assert changed == 4
    assert new_text == (
        "#EXTM3U\n"
        "../Artist/Album/01 One.mp3\n"
        "..\\Artist\\Album\\02 Two.mp3\r\n"
        "C:/Music/Artist/Album/03 Three.mp3\n"
        "D:/Elsewhere/04 Four.mp3\n"
        "../Artist/Album/05 Untouched.mp3\n"
    )


def test_duplicate_executor_playlist_rewrite_keeps_style_on_windows(windows):
    import duplicate_consolidation_executor as dce

    lines = ["#EXTM3U", "../Artist/Album/Song.mp3", "..\\Artist\\Album\\Other.mp3", "../Artist/Album/Far.mp3"]
    with windows():
        mapping = {
            "C:\\Music\\Artist\\Album\\Song.mp3": "C:\\Music\\Artist\\Album\\Song.flac",
            "C:\\Music\\Artist\\Album\\Other.mp3": "C:\\Music\\Artist\\Album\\Other.flac",
            "C:\\Music\\Artist\\Album\\Far.mp3": "D:\\Kept\\Far.flac",
        }
        replaced, new_lines = dce._rewrite_playlist_lines(lines, mapping, playlist_dir="C:\\Music\\Playlists")
    assert replaced == 3
    assert new_lines == ["#EXTM3U", "../Artist/Album/Song.flac", "..\\Artist\\Album\\Other.flac", "D:/Kept/Far.flac"]


def test_new_playlist_on_another_drive_gets_absolute_entries(windows, monkeypatch):
    import playlist_generator as pg

    with windows():
        assert pg._relative_lines(["D:\\Music\\a.mp3", "E:\\Out\\b.mp3"], "E:\\Out\\mix.m3u") == [
            "D:\\Music\\a.mp3", "b.mp3",
        ]


# ── Indexer preview: one separator style ────────────────────────────────────

def test_indexer_preview_paths_use_forward_slashes_on_windows(windows):
    import music_indexer_api as mia

    with windows():
        rel = mia._rel("C:\\Music\\Trash\\Basalt\\Granite\\cover.jpg", "C:\\Music")
        folder = mia._rel("C:\\Music\\Aurora Lines\\Dawn Signals", "C:\\Music")
        other_drive = mia._rel("D:\\x.jpg", "C:\\Music")
    assert rel == "Trash/Basalt/Granite/cover.jpg"
    assert folder + "/" == "Aurora Lines/Dawn Signals/"
    assert other_drive == "D:\\x.jpg"


# ── update_genres CLI: relpath across drives ─────────────────────────────────

def test_genre_cli_logs_the_absolute_path_across_drives(tmp_path, monkeypatch):
    import update_genres

    def relpath(path, start=os.curdir):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")  # what ntpath raises

    monkeypatch.setattr(update_genres, "extract_genre_list", lambda _p: ["Rock", "Pop"])
    monkeypatch.setattr(os.path, "relpath", relpath)
    log = io.StringIO()
    track = str(tmp_path / "multi.flac")
    update_genres.process_file(track, log)
    assert log.getvalue().startswith(f"{track}  → skipped")


# ── long paths stay at the OS boundary ───────────────────────────────────────

def _long_form(path: str) -> str:
    """Stand-in for ``\\\\?\\C:\\...``: another spelling of the same file."""
    return os.path.join(os.path.dirname(path), ".", os.path.basename(path))


def test_fingerprint_cache_hands_compute_the_callers_path(tmp_path, monkeypatch):
    import sqlite3

    import fingerprint_cache as fc

    monkeypatch.setattr(fc, "ensure_long_path", _long_form)
    fc.shutdown_fingerprint_writer()
    fc._memo_clear()
    song = tmp_path / "a.mp3"
    song.write_bytes(b"x")
    db = tmp_path / "Docs" / "fp.db"
    calls = []

    def compute(p):
        calls.append(p)
        return 1, "hash"

    try:
        assert fc.get_fingerprint(str(song), str(db), compute) == "hash"
        assert calls == [str(song)]
        fc.flush_fingerprint_writes(str(db))
        fc._memo_clear()
        # The key is the long form, as before: existing caches stay valid.
        assert fc.get_cached_fingerprint(str(song), str(db)) == "hash"
        assert fc.get_fingerprint(str(song), str(db), compute) == "hash"
        assert calls == [str(song)]
        with sqlite3.connect(db) as conn:
            assert [r[0] for r in conn.execute("SELECT path FROM fingerprints")] == [_long_form(str(song))]
    finally:
        fc.shutdown_fingerprint_writer()
        fc._memo_clear()


@pytest.mark.parametrize("module", ["fingerprint_generator", "library_sync_indexer_engine.indexer_engine.fingerprint_generator"])
def test_fingerprint_generators_return_the_callers_path(tmp_path, monkeypatch, module):
    import importlib

    gen = importlib.import_module(module)
    song = tmp_path / "song.flac"
    song.write_bytes(b"x")
    opened = []

    def fingerprint_file(target=None, **_kw):
        opened.append(target)
        return 12.0, "AQAAAA"

    monkeypatch.setattr(gen, "ensure_long_path", _long_form)
    monkeypatch.setattr(gen.acoustid, "fingerprint_file", fingerprint_file, raising=False)
    path, duration, fp, err = gen.compute_fingerprint_for_file((str(song), "", {}))
    assert err is None and fp == "AQAAAA"
    assert opened == [_long_form(str(song))]     # the file is opened by its long form ...
    assert path == str(song)                     # ... but the caller's path comes back
