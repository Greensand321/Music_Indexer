"""Opus library mirror: safety, change detection, orphans (listening backends audit).

* writes never leave the mirror: a destination folder that is a link out of
  the mirror (or into the source) is refused, even with ``overwrite``; a
  destination inside the source, or a source inside the destination, is
  refused up front;
* one folder the destination drive cannot create no longer aborts the run;
* long track names are mirrored (the temporary name used to be too long);
* folders whose names differ only in case cannot race onto one file;
* a mirror file whose source changed is reported as stale (and replaced only
  with ``update_changed``);
* files whose source was deleted or renamed are listed as orphans, never
  deleted by a run; :func:`remove_mirror_orphans` deletes a reviewed list and
  refuses anything outside the mirror;
* temporary files of an interrupted run are cleaned up.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from utils import opus_library_mirror as mirror

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _write(path: Path, data: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _bump_mtime(path: Path, seconds: float = 100.0) -> None:
    st = path.stat()
    os.utime(path, (st.st_atime + seconds, st.st_mtime + seconds))


def _symlink_or_skip(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")


def _case_sensitive(folder: Path) -> bool:
    folder.mkdir(parents=True, exist_ok=True)
    probe = folder / "CaseProbe"
    probe.write_bytes(b"")
    try:
        return not (folder / "caseprobe").exists()
    finally:
        probe.unlink()


# ── writes stay inside the mirror ─────────────────────────────────────────


def test_overwrite_never_writes_through_a_link_out_of_the_mirror(tmp_path):
    src = tmp_path / "src"
    _write(src / "Album" / "song.mp3", b"new")
    precious = _write(tmp_path / "elsewhere" / "song.mp3", b"precious")
    (tmp_path / "dst").mkdir()
    _symlink_or_skip(tmp_path / "elsewhere", tmp_path / "dst" / "Album")

    summary = mirror.mirror_library(str(src), str(tmp_path / "dst"), True, max_workers=2)

    assert precious.read_bytes() == b"precious", "a file outside the mirror was overwritten"
    assert summary["copied"] == 0 and summary["errors"] == 1
    (path, reason), = summary["error_files"]
    assert path == str(src / "Album" / "song.mp3")
    assert "outside the mirror" in reason


def test_overwrite_never_writes_through_a_link_into_the_source(tmp_path):
    src = tmp_path / "src"
    original = _write(src / "Album" / "song.mp3", b"original")
    (tmp_path / "dst").mkdir()
    _symlink_or_skip(src / "Album", tmp_path / "dst" / "Album")

    summary = mirror.mirror_library(str(src), str(tmp_path / "dst"), True, max_workers=2)

    assert original.read_bytes() == b"original"
    assert sorted(p.name for p in (src / "Album").iterdir()) == ["song.mp3"]
    assert summary["errors"] == 1 and summary["copied"] == 0


@pytest.mark.parametrize("layout", ["same", "dest_inside_source", "source_inside_dest"])
def test_overlapping_folders_are_refused_before_anything_is_written(tmp_path, layout):
    if layout == "same":
        src = dst = tmp_path / "Music"
    elif layout == "dest_inside_source":
        src, dst = tmp_path / "Music", tmp_path / "Music" / "Mirror"
    else:
        src, dst = tmp_path / "Mirror" / "Music", tmp_path / "Mirror"
    _write(src / "Album" / "song.flac", b"not-a-real-flac")
    before = sorted(str(p) for p in tmp_path.rglob("*"))

    with pytest.raises(ValueError):
        mirror.mirror_library(str(src), str(dst), True, max_workers=2)

    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_a_folder_the_destination_cannot_create_fails_alone(tmp_path, monkeypatch):
    """E.g. "AC/DC: Live?" on a FAT/exFAT card: that folder fails, the run goes on."""
    src = tmp_path / "src"
    _write(src / "Bad: Name" / "a.mp3")
    _write(src / "Good" / "b.mp3")
    dst = tmp_path / "dst"
    real_makedirs = os.makedirs

    def makedirs(path, *args, **kwargs):
        if "Bad: Name" in str(path):
            raise OSError(22, "Invalid argument", str(path))
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(mirror.os, "makedirs", makedirs)
    summary = mirror.mirror_library(str(src), str(dst), False, max_workers=2)

    assert (dst / "Good" / "b.mp3").read_bytes() == b"data"
    assert summary["copied"] == 1 and summary["errors"] == 1
    (path, reason), = summary["error_files"]
    assert path == str(src / "Bad: Name" / "a.mp3")
    assert "Could not create destination folder" in reason


def test_long_track_names_are_mirrored(tmp_path):
    src = tmp_path / "src"
    name = "x" * 230 + ".mp3"  # legal; name + temporary marker was not
    _write(src / "Album" / name, b"long")

    summary = mirror.mirror_library(str(src), str(tmp_path / "dst"), False, max_workers=2)

    assert summary["errors"] == 0, summary["error_files"]
    assert (tmp_path / "dst" / "Album" / name).read_bytes() == b"long"
    assert not [p for p in (tmp_path / "dst").rglob("*") if ".alphadex-partial" in p.name]


def test_folders_differing_only_in_case_cannot_race_onto_one_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    if not _case_sensitive(src):
        pytest.skip("needs a case-sensitive file system to hold both folders")
    _write(src / "Artist" / "song.mp3", b"upper")
    _write(src / "artist" / "song.mp3", b"lower")

    summary = mirror.mirror_library(str(src), str(tmp_path / "dst"), False, max_workers=4)

    # Deterministic: sorted walk order, "Artist" < "artist".
    assert summary["copied"] == 1 and summary["skipped"] == 1
    (skipped, reason), = summary["skipped_files"]
    assert skipped == str(src / "artist" / "song.mp3")
    assert "Same destination name" in reason
    assert (tmp_path / "dst" / "Artist" / "song.mp3").read_bytes() == b"upper"


# ── change detection ──────────────────────────────────────────────────────


def test_changed_source_is_reported_stale_and_kept_by_default(tmp_path):
    src = tmp_path / "src"
    song = _write(src / "song.mp3", b"v1")
    dst = tmp_path / "dst"
    mirror.mirror_library(str(src), str(dst), False)
    song.write_bytes(b"version two")
    _bump_mtime(song)

    summary = mirror.mirror_library(str(src), str(dst), False)

    assert (dst / "song.mp3").read_bytes() == b"v1"
    assert summary["stale"] == 1 and summary["updated"] == 0
    (path, reason), = summary["skipped_files"]
    assert path == str(song) and "older than the source" in reason


def test_update_changed_replaces_only_stale_files(tmp_path):
    src = tmp_path / "src"
    changed = _write(src / "changed.mp3", b"v1")
    _write(src / "same.mp3", b"same")
    dst = tmp_path / "dst"
    mirror.mirror_library(str(src), str(dst), False)
    (dst / "same.mp3").write_bytes(b"same")  # untouched content, newer mtime
    changed.write_bytes(b"version two")
    _bump_mtime(changed)

    summary = mirror.mirror_library(str(src), str(dst), False, update_changed=True)

    assert (dst / "changed.mp3").read_bytes() == b"version two"
    assert summary["copied"] == 1 and summary["updated"] == 1 and summary["stale"] == 1
    assert [p for p, _ in summary["skipped_files"]] == [str(src / "same.mp3")]


@needs_ffmpeg
def test_retagged_flac_is_reconverted_with_update_changed(tmp_path):
    from library_factory import TrackSpec, build_library
    from mutagen.flac import FLAC
    from mutagen.oggopus import OggOpus

    build_library(tmp_path / "src", [TrackSpec("A", "Old Title", "B", 1, ext=".flac", relpath="t.flac")])
    flac_path = tmp_path / "src" / "t.flac"
    dst = tmp_path / "dst"
    assert mirror.mirror_library(str(tmp_path / "src"), str(dst), False)["converted"] == 1
    audio = FLAC(str(flac_path))
    audio["title"] = "New Title"
    audio.save()
    _bump_mtime(flac_path)

    kept = mirror.mirror_library(str(tmp_path / "src"), str(dst), False)
    assert kept["stale"] == 1 and kept["converted"] == 0
    assert OggOpus(str(dst / "t.opus"))["title"] == ["Old Title"]

    updated = mirror.mirror_library(str(tmp_path / "src"), str(dst), False, update_changed=True)
    assert updated["converted"] == 1 and updated["updated"] == 1
    assert OggOpus(str(dst / "t.opus"))["title"] == ["New Title"]


# ── orphans ───────────────────────────────────────────────────────────────


def test_deleted_and_renamed_sources_leave_listed_orphans_not_deletions(tmp_path):
    src = tmp_path / "src"
    gone = _write(src / "Album" / "gone.mp3")
    old = _write(src / "Album" / "old name.mp3")
    _write(src / "Album" / "kept.mp3")
    dst = tmp_path / "dst"
    mirror.mirror_library(str(src), str(dst), False)
    gone.unlink()
    old.rename(src / "Album" / "new name.mp3")

    summary = mirror.mirror_library(str(src), str(dst), False)

    assert summary["orphan_scan"] == "done"
    assert sorted(summary["orphan_files"]) == sorted([
        str(dst / "Album" / "gone.mp3"), str(dst / "Album" / "old name.mp3"),
    ])
    # Listed, never deleted by a run.
    assert (dst / "Album" / "gone.mp3").exists() and (dst / "Album" / "old name.mp3").exists()
    assert (dst / "Album" / "new name.mp3").exists()


def test_converted_flac_and_reports_are_not_orphans(tmp_path):
    src = tmp_path / "src"
    _write(src / "song.flac", b"x")
    dst = tmp_path / "dst"
    _write(dst / "song.opus", b"mirrored earlier")
    _write(dst / "Docs" / "opus_library_mirror_reports" / "r.html", b"<html/>")
    _write(dst / "Docs" / "notes.txt", b"mine")

    summary = mirror.mirror_library(str(src), str(dst), False, exclude_folders=())

    assert summary["orphan_files"] == [str(dst / "Docs" / "notes.txt")]


def test_cancelled_run_does_not_report_orphans(tmp_path):
    import threading

    src = tmp_path / "src"
    _write(src / "a.mp3")
    dst = tmp_path / "dst"
    _write(dst / "stray.mp3")
    stop = threading.Event()
    stop.set()
    summary = mirror.mirror_library(str(src), str(dst), False, cancel_event=stop)
    assert summary["orphan_files"] == [] and summary["orphan_scan"].startswith("skipped")


def test_interrupted_run_leftovers_are_removed_and_redone(tmp_path):
    src = tmp_path / "src"
    _write(src / "song.mp3", b"full")
    dst = tmp_path / "dst"
    other_pid = os.getpid() + 1
    leftover = _write(dst / f".song.alphadex-partial-{other_pid}-123.mp3", b"half")
    mine = _write(dst / "song.alphadex-partial notes.txt", b"not ours")

    summary = mirror.mirror_library(str(src), str(dst), False)

    assert not leftover.exists()
    assert summary["partials_removed"] == 1
    assert mine.exists(), "only our own temporary names are removed"
    assert (dst / "song.mp3").read_bytes() == b"full"


def test_remove_mirror_orphans_deletes_only_reviewed_files_inside_the_mirror(tmp_path):
    src = tmp_path / "src"
    _write(src / "Album" / "kept.mp3")
    gone = _write(src / "Old" / "gone.mp3")
    dst = tmp_path / "dst"
    mirror.mirror_library(str(src), str(dst), False)
    gone.unlink()
    gone.parent.rmdir()
    orphans = mirror.mirror_library(str(src), str(dst), False)["orphan_files"]
    assert orphans == [str(dst / "Old" / "gone.mp3")]

    outside = _write(tmp_path / "outside.mp3", b"precious")
    library_file = src / "Album" / "kept.mp3"
    sneaky = str(dst / ".." / "outside.mp3")
    result = mirror.remove_mirror_orphans(
        str(src), str(dst), [*orphans, str(outside), sneaky, str(library_file)]
    )

    assert result["removed"] == orphans
    assert not (dst / "Old").exists(), "the emptied folder is removed"
    assert dst.is_dir() and (dst / "Album" / "kept.mp3").exists()
    assert outside.read_bytes() == b"precious" and library_file.exists()
    reasons = dict(result["refused"])
    assert reasons[str(outside)] == "outside the mirror"
    assert reasons[os.path.abspath(sneaky)] == "outside the mirror"
    assert reasons[str(library_file)] == "outside the mirror"
    log = Path(result["log_path"])
    assert log.parent == dst / "Docs" / "opus_library_mirror_reports"
    assert f"removed\t{orphans[0]}" in log.read_text(encoding="utf-8")


def test_remove_mirror_orphans_refuses_links_and_restored_sources(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write(src / "a.mp3")
    mirror.mirror_library(str(src), str(dst), False)
    stray = _write(dst / "back.opus", b"opus")
    _write(src / "back.flac", b"flac")  # its source came back since the scan
    elsewhere = _write(tmp_path / "elsewhere" / "x.mp3", b"precious")
    try:
        os.symlink(elsewhere, dst / "link.mp3")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")

    result = mirror.remove_mirror_orphans(str(src), str(dst), [str(stray), str(dst / "link.mp3")])

    assert result["removed"] == []
    reasons = dict(result["refused"])
    assert reasons[str(stray)] == "its source file exists again"
    assert reasons[str(dst / "link.mp3")] == "not a regular file"
    assert stray.exists() and elsewhere.read_bytes() == b"precious"


def test_remove_mirror_orphans_refuses_overlapping_folders(tmp_path):
    lib = tmp_path / "Music"
    victim = _write(lib / "a.mp3")
    with pytest.raises(ValueError):
        mirror.remove_mirror_orphans(str(lib), str(lib), [str(victim)])
    with pytest.raises(ValueError):
        mirror.remove_mirror_orphans(str(lib / "sub"), str(lib), [str(victim)])
    assert victim.exists()


# ── ffmpeg, tags, cover ───────────────────────────────────────────────────


def test_missing_ffmpeg_is_reported_once_and_copies_still_run(tmp_path, monkeypatch):
    src = tmp_path / "src"
    for i in range(5):
        _write(src / f"{i}.flac", b"flac")
    _write(src / "keep.mp3", b"mp3")
    monkeypatch.setattr(mirror.shutil, "which", lambda _name: None)
    errors: list[str] = []

    summary = mirror.mirror_library(str(src), str(tmp_path / "dst"), False, error_callback=errors.append)

    assert summary["copied"] == 1 and summary["errors"] == 5
    assert len(errors) == 1 and "FFmpeg was not found" in errors[0]
    assert (tmp_path / "dst" / "keep.mp3").read_bytes() == b"mp3"


@needs_ffmpeg
def test_conversion_keeps_multi_value_tags_and_front_cover(tmp_path):
    from library_factory import TrackSpec, build_library
    from mutagen.flac import FLAC, Picture
    from mutagen.oggopus import OggOpus
    from utils.opus_metadata_reader import read_opus_metadata

    build_library(tmp_path / "src", [TrackSpec("A", "T", "B", 1, ext=".flac", relpath="t.flac")])
    path = tmp_path / "src" / "t.flac"
    audio = FLAC(str(path))
    audio["genre"] = ["Rock", "Pop"]
    audio["ARTISTS"] = ["A", "B"]
    back, front = Picture(), Picture()
    back.type, back.mime, back.data = 4, "image/png", b"\x89PNG-back"
    front.type, front.mime, front.data = 3, "image/jpeg", b"\xff\xd8front"
    audio.add_picture(back)
    audio.add_picture(front)
    audio.save()

    summary = mirror.mirror_library(str(tmp_path / "src"), str(tmp_path / "dst"), False)

    assert summary["converted"] == 1, summary["error_files"]
    opus = OggOpus(str(tmp_path / "dst" / "t.opus"))
    assert opus["genre"] == ["Rock", "Pop"]
    assert opus["artists"] == ["A", "B"]
    _tags, covers, error = read_opus_metadata(str(tmp_path / "dst" / "t.opus"))
    assert error is None and covers == [b"\xff\xd8front"]


# ── playlists ─────────────────────────────────────────────────────────────


def test_mirrored_playlists_point_at_the_converted_opus(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _write(src / "Album" / "01 Song.flac", b"flac")
    _write(src / "Album" / "02 Other.mp3", b"mp3")
    _write(src / "Album" / "03 Both.flac", b"flac")
    _write(src / "Album" / "03 Both.opus", b"an opus that wins the name")
    playlist = (
        "#EXTM3U\r\n"
        "#EXTINF:1,Song\r\n"
        "../Album/01 Song.flac\r\n"
        "..\\Album\\01 Song.FLAC\r\n"
        "../Album/02 Other.mp3\r\n"
        "../Album/03 Both.flac\r\n"
        "../Album/missing.flac\r\n"
        "/somewhere/else/x.flac\r\n"
        "../../outside.flac\r\n"
        "http://radio.example/stream.flac\r\n"
    ).encode("utf-8")
    _write(src / "Playlists" / "mix.m3u8", playlist)
    latin = "../Album/01 Song.flac\nCaf\xe9.mp3\n".encode("cp1252")
    _write(src / "Playlists" / "old.m3u", latin)
    monkeypatch.setattr(mirror.shutil, "which", lambda _name: None)  # conversions just fail

    dst = tmp_path / "dst"
    summary = mirror.mirror_library(str(src), str(dst), False)

    out = (dst / "Playlists" / "mix.m3u8").read_bytes().decode("utf-8").split("\r\n")
    assert out == [
        "#EXTM3U",
        "#EXTINF:1,Song",
        "../Album/01 Song.opus",
        "..\\Album\\01 Song.opus",     # separators and stem kept, only the extension changes
        "../Album/02 Other.mp3",
        "../Album/03 Both.flac",      # the mirror's 03 Both.opus is a copy, not this FLAC
        "../Album/missing.flac",
        "/somewhere/else/x.flac",
        "../../outside.flac",
        "http://radio.example/stream.flac",
        "",
    ]
    assert (dst / "Playlists" / "old.m3u").read_bytes() == "../Album/01 Song.opus\nCaf\xe9.mp3\n".encode("cp1252")
    # Same modification time as the source, so the next run sees it as current.
    assert os.path.getmtime(dst / "Playlists" / "mix.m3u8") == pytest.approx(
        os.path.getmtime(src / "Playlists" / "mix.m3u8"), abs=0.01
    )
    again = mirror.mirror_library(str(src), str(dst), False, update_changed=True)
    assert again["stale"] == 0 and again["copied"] == 0
    assert summary["orphan_files"] == []


def test_report_lists_orphans_and_stale_counts(tmp_path):
    src = tmp_path / "src"
    song = _write(src / "song.mp3", b"v1")
    gone = _write(src / "<gone>.mp3")
    dst = tmp_path / "dst"
    mirror.mirror_library(str(src), str(dst), False)
    gone.unlink()
    song.write_bytes(b"v2 longer")
    _bump_mtime(song)
    summary = mirror.mirror_library(str(src), str(dst), False)

    report = tmp_path / "report.html"
    mirror.write_mirror_report(str(report), summary)
    text = report.read_text(encoding="utf-8")
    assert "Orphaned files in the mirror: 1" in text
    assert "&lt;gone&gt;.mp3" in text and "<gone>" not in text
    assert "older than their changed source: 1 (0 updated)" in text
