"""Regression tests for playlist_generator / playlist_engine (TOOL-03, PL-10,
PL-11, PL-12, PL-14, PL-28)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import playlist_engine
import playlist_generator as pg


def _touch(root: Path, rel: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0")
    return str(p)


# ── TOOL-03: update_playlists finds the library root ─────────────────────


def test_rename_inside_one_album_updates_library_playlists(tmp_path):
    old = _touch(tmp_path, "Artist/Album/01 Song (1).mp3")
    other = _touch(tmp_path, "Artist/Album/02 Other.mp3")
    playlists = tmp_path / "Playlists"
    playlists.mkdir()
    pl = playlists / "mix.m3u"
    pl.write_text("../Artist/Album/01 Song (1).mp3\n../Artist/Album/02 Other.mp3\n", encoding="utf-8")

    new = str(tmp_path / "Artist/Album/01 Song.mp3")
    os.rename(old, new)
    # Common path of the change is the album folder, which has no Playlists/.
    count = pg.update_playlists({old: new})

    assert count == 1
    assert pl.read_text(encoding="utf-8").splitlines() == [
        os.path.join("..", "Artist", "Album", "01 Song.mp3"),
        os.path.join("..", "Artist", "Album", "02 Other.mp3"),
    ]
    assert os.path.exists(other)
    # No folder playlist is created as a side effect of the rename.
    assert sorted(os.listdir(playlists)) == ["mix.m3u"]


def test_explicit_library_root_is_used(tmp_path):
    lib = tmp_path / "lib"
    old = _touch(lib, "A/B/x (1).mp3")
    (lib / "Playlists").mkdir()
    pl = lib / "Playlists" / "p.m3u"
    pl.write_text("../A/B/x (1).mp3\n", encoding="utf-8")
    new = str(lib / "A/B/x.mp3")
    os.rename(old, new)
    assert pg.update_playlists({old: new}, library_root=str(lib)) == 1
    assert "x.mp3" in pl.read_text(encoding="utf-8")


def test_update_without_playlists_folder_reports_zero(tmp_path):
    a = _touch(tmp_path, "A/a.mp3")
    logs = []
    assert pg.update_playlists({a: a}, log_callback=logs.append) == 0
    assert logs


# ── PL-12: folder playlists skip reserved folders ────────────────────────


def test_folder_plan_is_per_album_and_skips_reserved_folders(tmp_path):
    for rel in (
        "Artist A/Album 1/01 a.mp3", "Artist A/Album 1/02 b.mp3", "Artist A/Album 2/01 c.flac",
        "Quarantine/dupe.mp3", "Trash/x.mp3", "Not Sorted/n.mp3", "Docs/sample.wav",
        "Manual Review/m.mp3", "Playlists/old.mp3",
    ):
        _touch(tmp_path, rel)
    plan = pg.plan_folder_playlists(str(tmp_path))
    names = sorted(os.path.basename(p.path) for p in plan)
    assert names == ["Artist A_Album 1.m3u", "Artist A_Album 2.m3u"]
    album1 = next(p for p in plan if p.path.endswith("Album 1.m3u"))
    assert album1.entries == 2
    assert album1.lines == sorted(album1.lines)
    assert not (tmp_path / "Playlists" / "Artist A_Album 1.m3u").exists(), "planning must not write"


# ── PL-11: writing is exactly what was planned, with backups ─────────────


def test_write_plan_backs_up_overwritten_files_and_logs(tmp_path):
    track = _touch(tmp_path, "A/B/t.mp3")
    out = tmp_path / "Playlists" / "autodj.m3u"
    out.parent.mkdir()
    out.write_text("old content\n", encoding="utf-8")

    item = pg.plan_playlist([track], str(out))
    assert item.exists and item.action == "overwrite"
    result = pg.write_playlist_plan([item], log_dir=str(tmp_path / "Docs"))

    assert result["written"] == [str(out)]
    assert len(result["backups"]) == 1
    assert Path(result["backups"][0]).read_text(encoding="utf-8") == "old content\n"
    assert out.read_text(encoding="utf-8").splitlines() == item.lines
    log = (tmp_path / "Docs" / "playlist_writes.log").read_text(encoding="utf-8")
    assert "BACKUP" in log and "WRITE" in log


def test_new_file_needs_no_backup(tmp_path):
    track = _touch(tmp_path, "A/t.mp3")
    item = pg.plan_playlist([track], str(tmp_path / "Playlists" / "new.m3u"))
    assert item.action == "new"
    result = pg.write_playlist_plan([item])
    assert result["backups"] == [] and len(result["written"]) == 1


# ── PL-10: repair actually repairs the chosen playlists ──────────────────


def test_repair_plan_fixes_user_selected_playlist_anywhere(tmp_path):
    lib = tmp_path / "lib"
    _touch(lib, "Artist A/Album 1/01 a.mp3")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pl = elsewhere / "mine.m3u"
    pl.write_text(
        "#EXTM3U\n"
        f"{lib}/Artist A/Old Album/01 a.mp3\n"
        f"{lib}/gone/zzz qqq.mp3\n",
        encoding="utf-8",
    )
    plan = pg.plan_playlist_repair([str(pl)], str(lib))
    assert len(plan) == 1
    item = plan[0]
    assert item.changes == [
        (f"{lib}/Artist A/Old Album/01 a.mp3", str(lib / "Artist A/Album 1/01 a.mp3"))
    ]
    assert item.unresolved == [f"{lib}/gone/zzz qqq.mp3"]
    assert item.lines[0] == "#EXTM3U"
    # Nothing written by planning.
    assert "Old Album" in pl.read_text(encoding="utf-8")

    pg.write_playlist_plan([item])
    text = pl.read_text(encoding="utf-8")
    assert "Album 1/01 a.mp3" in text and "Old Album" not in text


@pytest.mark.parametrize("prefer_opus,missing,expected", [
    (True, "song.flac", "song.opus"),
    (False, "song.opus", "song.flac"),
])
def test_repair_codec_swap_follows_prefer_opus(tmp_path, prefer_opus, missing, expected):
    lib = tmp_path
    _touch(lib, "A/song.opus")
    _touch(lib, "A/song.flac")
    (lib / "A" / missing).unlink()
    (lib / "Playlists").mkdir()
    pl = lib / "Playlists" / "p.m3u"
    pl.write_text(f"../A/{missing}\n", encoding="utf-8")
    item = pg.plan_playlist_repair([str(pl)], str(lib), prefer_opus=prefer_opus)[0]
    assert item.changes == [(f"../A/{missing}", os.path.join("..", "A", expected))]


def test_repair_plan_reports_unreadable_playlist(tmp_path):
    item = pg.plan_playlist_repair([str(tmp_path / "nope.m3u")], str(tmp_path))[0]
    assert item.error


# ── PL-28 / PL-14: tempo cache and ranges ────────────────────────────────


@pytest.fixture
def fake_tempo(monkeypatch):
    values = {}

    def compute(path, engine="librosa"):
        return values[os.path.basename(path)]

    monkeypatch.setattr(playlist_engine, "librosa", object())
    monkeypatch.setattr(playlist_engine, "compute_tempo_energy", compute)
    return values


def test_corrupt_tempo_cache_is_rebuilt_not_fatal(tmp_path, fake_tempo):
    t = _touch(tmp_path, "A/a.mp3")
    fake_tempo["a.mp3"] = (100.0, 0.2)
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / "tempo_energy.npy").write_bytes(b"\x93NUMPY garbage")
    logs = []
    result = playlist_engine.bucket_by_tempo_energy([t], str(tmp_path), log_callback=logs.append)
    assert result["buckets"] == {("medium", "medium"): [t]}
    assert any("Ignoring unreadable tempo/energy cache" in m for m in logs)
    # Rebuilt cache is readable without pickle.
    with np.load(tmp_path / "Docs" / "tempo_energy.npy", allow_pickle=False) as data:
        assert list(data["paths"]) == [os.path.abspath(t)]


def test_legacy_pickled_tempo_cache_is_not_unpickled(tmp_path, fake_tempo):
    t = _touch(tmp_path, "A/a.mp3")
    fake_tempo["a.mp3"] = (80.0, 0.05)
    (tmp_path / "Docs").mkdir()
    np.save(tmp_path / "Docs" / "tempo_energy.npy",
            {os.path.abspath(t): {"engine": "librosa", "tempo": 200.0, "rms": 0.9}},
            allow_pickle=True)
    result = playlist_engine.bucket_by_tempo_energy([t], str(tmp_path))
    # The pickled values (200 BPM / 0.9) were not used.
    assert result["buckets"] == {("slow", "low"): [t]}


def test_tempo_cache_round_trip_skips_reanalysis(tmp_path, fake_tempo, monkeypatch):
    t = _touch(tmp_path, "A/a.mp3")
    fake_tempo["a.mp3"] = (130.0, 0.4)
    playlist_engine.bucket_by_tempo_energy([t], str(tmp_path))
    fake_tempo.clear()  # would KeyError if re-analysed
    result = playlist_engine.bucket_by_tempo_energy([t], str(tmp_path))
    assert result["buckets"] == {("fast", "high"): [t]}


def test_tempo_and_energy_ranges_filter_tracks(tmp_path, fake_tempo):
    a = _touch(tmp_path, "A/a.mp3")
    b = _touch(tmp_path, "A/b.mp3")
    c = _touch(tmp_path, "A/c.mp3")
    fake_tempo.update({"a.mp3": (70.0, 0.2), "b.mp3": (125.0, 0.2), "c.mp3": (125.0, 0.8)})
    result = playlist_engine.bucket_by_tempo_energy(
        [a, b, c], str(tmp_path), tempo_range=(100, 180), energy_range=(0.0, 0.5)
    )
    assert result["buckets"] == {("fast", "medium"): [b]}
    assert result["excluded"] == 2


def test_bucketing_writes_no_playlists(tmp_path, fake_tempo):
    a = _touch(tmp_path, "A/a.mp3")
    fake_tempo["a.mp3"] = (100.0, 0.2)
    result = playlist_engine.bucket_by_tempo_energy([a], str(tmp_path))
    assert not list((tmp_path).glob("Playlists/*.m3u"))
    assert result["playlist_paths"][("medium", "medium")].endswith("medium_medium.m3u")
