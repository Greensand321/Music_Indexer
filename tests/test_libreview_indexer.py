"""Indexer execute, second review round (area: libreview).

* Execute no longer rewrites artist tags it never previewed (5a).
* Windows case handling: a case-only rename shown in the preview really
  happens, and two destinations that differ only in case never both pass the
  preview (5d).
* Sweeps never touch OS / sync-tool metadata, files in the library root, or
  folders that never held music; Explorer rewriting desktop.ini no longer
  makes the plan look stale (5e).
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest

import music_indexer_api as api
from library_factory import TrackSpec, build_library, build_standard_library
from utils.audio_metadata_reader import read_tags


def _tree(root: Path) -> dict[str, str]:
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            out[p.relative_to(root).as_posix()] = hashlib.sha1(p.read_bytes()).hexdigest()
    return out


def _rel(plan_map: dict, root: Path) -> dict[str, str]:
    return {Path(s).relative_to(root).as_posix(): Path(d).relative_to(root).as_posix()
            for s, d in plan_map.items()}


@pytest.fixture
def windows_case(monkeypatch):
    """Emulate Windows path comparison: normcase lower-cases (separators are
    left alone so the test runs on any OS)."""
    monkeypatch.setattr(os.path, "normcase", lambda p: p.lower())


# ── 5a. No unpreviewed tag writes ────────────────────────────────────────────

def test_execute_never_rewrites_artist_tags(tmp_path):
    lib = tmp_path / "lib"
    specs = [TrackSpec("Aurora Lines", f"Song {i}", "Dawn", i, "2019", "Pop", freq=300 + i * 10)
             for i in range(1, 4)]
    specs.append(TrackSpec("AURORA LINES feat. Bob", "Odd One", "Dawn", 9, "2019", "Pop", freq=900))
    build_library(lib, specs)
    odd = lib / "AURORA LINES feat. Bob" / "Dawn" / "09 - Odd One.mp3"
    assert odd.exists()
    odd_bytes = odd.read_bytes()
    out = lib / "Docs" / "MusicIndex.html"
    plan = api.run_full_indexer(str(lib), str(out), dry_run_only=True)["plan"]
    logs = []
    summary = api.apply_indexer_plan(plan, log_callback=logs.append, create_playlists=False)
    assert summary["errors"] == []
    dst = Path(plan.moves[str(odd)])
    assert read_tags(str(dst)).get("artist") == "AURORA LINES feat. Bob"
    assert dst.read_bytes() == odd_bytes, "execute moves files; it never edits them"
    assert not any("artist tag" in line for line in logs)


# ── 5d. Windows case ─────────────────────────────────────────────────────────

def test_case_only_rename_in_the_preview_is_executed(tmp_path, windows_case):
    lib = tmp_path / "lib"
    wrong_case = "By Year/2019/aurora lines_01_song 1.mp3"
    build_library(lib, [TrackSpec("Aurora Lines", "Song 1", "Dawn", 1, "2019", "Pop", relpath=wrong_case)])
    plan = api.build_indexer_plan(str(lib))
    expected = "By Year/2019/Aurora Lines_01_Song 1.mp3"
    assert _rel(plan.moves, lib) == {wrong_case: expected}           # what the preview shows…
    assert _rel(plan.file_moves, lib) == {wrong_case: expected}      # …is what execute does
    assert plan.counts()["moves"] == 1 and plan.counts()["unchanged"] == 0
    summary = api.apply_indexer_plan(plan, create_playlists=False)
    assert summary["errors"] == [] and summary["moved"] == 1
    assert list(_tree(lib / "By Year")) == ["2019/Aurora Lines_01_Song 1.mp3"]


def test_destinations_that_differ_only_in_case_never_collide(tmp_path, windows_case):
    lib = tmp_path / "lib"
    build_library(lib, [
        TrackSpec("Aurora Lines", "Song", "Dawn", 1, "2019", "Pop", freq=330),
        TrackSpec("AURORA LINES", "SONG", "Other", 1, "2019", "Pop", freq=660, relpath="b/x.mp3"),
    ])
    plan = api.build_indexer_plan(str(lib))
    dests = list(plan.moves.values())
    assert len({d.lower() for d in dests}) == 2, f"case-only collision planned: {dests}"
    assert any(Path(d).stem.endswith("(1)") for d in dests)
    summary = api.apply_indexer_plan(plan, create_playlists=False)
    assert summary["errors"] == [] and summary["moved"] == 2
    assert all(Path(d).exists() for d in dests)


def test_rerun_on_an_organised_library_still_changes_nothing_with_windows_case(tmp_path, windows_case):
    lib = tmp_path / "lib"
    build_library(lib, [TrackSpec("Aurora Lines", "Song", "Dawn", 1, "2019", "Pop")])
    api.apply_indexer_plan(api.build_indexer_plan(str(lib)), create_playlists=False)
    again = api.build_indexer_plan(str(lib))
    assert again.file_moves == {} and again.leftovers == {}


# ── 5e. Sweeps ───────────────────────────────────────────────────────────────

METADATA = ("desktop.ini", "Thumbs.db", ".DS_Store", "._cover.jpg")


@pytest.fixture
def sweep_library(tmp_path):
    lib = tmp_path / "lib"
    build_standard_library(lib)
    extras = {
        # user files at the library root
        "README.txt": b"my notes", "folder.jpg": b"img", "desktop.ini": b"[.ShellClassInfo]",
        # sync-tool folder (with an old file version inside) and a folder that never held music
        ".stfolder/marker": b"", ".stversions/old take.mp3": b"not really audio",
        "Artwork/scan1.png": b"png",
        # a sub-folder without music inside an album that is emptied
        "Aurora Lines/Dawn Signals/Scans/back.png": b"scan",
    }
    for name in METADATA:      # OS metadata inside an album folder that the plan empties
        extras[f"Aurora Lines/Dawn Signals/{name}"] = b"meta"
    for rel, data in extras.items():
        p = lib / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return lib


def test_sweep_leaves_root_metadata_and_never_music_folders_alone(sweep_library):
    lib = sweep_library
    plan = api.build_indexer_plan(str(lib))
    swept = _rel(plan.leftovers, lib)
    # Still swept, as before: leftovers of album folders the plan empties.
    assert swept == {
        "Aurora Lines/Dawn Signals/cover.jpg": "Trash/Aurora Lines/Dawn Signals/cover.jpg",
        "Basalt/Granite/notes.txt": "Docs/Basalt/Granite/notes.txt",
    }
    moved_audio = {Path(s).relative_to(lib).parts[0] for s in plan.moves}
    assert ".stversions" not in moved_audio
    empty = {Path(p).relative_to(lib).as_posix() for p in plan.empty_dirs}
    assert "Aurora Lines/Dawn Signals" not in empty and "Aurora Lines" not in empty
    assert "Artwork" not in empty and ".stfolder" not in empty and "Basalt/Granite" in empty

    before = _tree(lib)
    summary = api.apply_indexer_plan(plan, create_playlists=False)
    assert summary["errors"] == []
    after = _tree(lib)
    kept = ["README.txt", "folder.jpg", "desktop.ini", ".stfolder/marker", ".stversions/old take.mp3",
            "Artwork/scan1.png", "Aurora Lines/Dawn Signals/Scans/back.png"]
    kept += [f"Aurora Lines/Dawn Signals/{name}" for name in METADATA]
    for rel in kept:
        assert after.get(rel) == before[rel], f"{rel} must stay where it is"


def test_explorer_rewriting_metadata_does_not_make_the_plan_stale(sweep_library):
    lib = sweep_library
    plan = api.build_indexer_plan(str(lib))
    time.sleep(0.01)
    for rel in ("desktop.ini", "Aurora Lines/Dawn Signals/desktop.ini", "Aurora Lines/Dawn Signals/Thumbs.db"):
        (lib / rel).write_bytes(b"[.ShellClassInfo]\nIconResource=x")
    (lib / "Basalt" / "Granite" / "Thumbs.db").write_bytes(b"new thumbnail cache")
    assert api.check_plan_freshness(plan) == []
    summary = api.apply_indexer_plan(plan, create_playlists=False)
    assert summary["errors"] == []
    # The folder Explorer wrote into is not removed (it is not empty) — and
    # its Thumbs.db is not swept.
    assert (lib / "Basalt" / "Granite" / "Thumbs.db").exists()


@pytest.mark.parametrize("name", ["desktop.ini", "Thumbs.db", ".DS_Store", "._01 - Track.mp3.jpg",
                                  "Icon\r", "song.!sync", ".syncthing.cover.jpg.tmp"])
def test_os_metadata_names(name):
    assert api.is_os_metadata_file(name)


@pytest.mark.parametrize("name", ["cover.jpg", "folder.jpg", "notes.txt", "booklet.pdf"])
def test_ordinary_files_are_not_metadata(name):
    assert not api.is_os_metadata_file(name)
