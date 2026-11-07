"""Opus mirror name clashes (writers review, finding 6a).

``Song.flac`` (converted to ``Song.opus``) and ``Song.opus`` (copied) in one
folder both targeted ``Song.opus``; whichever worker finished last won. The
plan now decides up front: the file copied as-is wins, the other is skipped
and reported.
"""
from __future__ import annotations

import os

import pytest

from utils import opus_library_mirror as mirror


def _case_sensitive(folder) -> bool:
    """Whether ``folder``'s file system tells ``Probe`` and ``probe`` apart
    (Windows and macOS defaults don't)."""
    folder.mkdir(parents=True, exist_ok=True)
    probe = folder / "CaseProbe"
    probe.write_bytes(b"")
    try:
        return not (folder / "caseprobe").exists()
    finally:
        probe.unlink()


def test_flac_and_opus_with_the_same_stem_prefer_the_existing_opus(tmp_path):
    src = tmp_path / "src" / "Album"
    src.mkdir(parents=True)
    (src / "Song.flac").write_bytes(b"not-a-real-flac")
    (src / "Song.opus").write_bytes(b"the-real-opus")
    dst = tmp_path / "dst"

    summary = mirror.mirror_library(str(tmp_path / "src"), str(dst), overwrite=False, max_workers=4)

    assert (dst / "Album" / "Song.opus").read_bytes() == b"the-real-opus"
    assert summary["copied"] == 1 and summary["converted"] == 0
    # The FLAC was never handed to ffmpeg (it would have failed here) ...
    assert summary["errors"] == 0, summary["error_files"]
    # ... and is reported as skipped, with the reason.
    skipped = dict(summary["skipped_files"])
    assert str(src / "Song.flac") in skipped
    assert "Song.opus" in skipped[str(src / "Song.flac")]


def test_names_differing_only_in_case_are_mirrored_once(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    if not _case_sensitive(src):
        pytest.skip("needs a case-sensitive file system (see the listing test below)")
    (src / "track.mp3").write_bytes(b"lower")
    (src / "Track.mp3").write_bytes(b"upper")
    dst = tmp_path / "dst"

    summary = mirror.mirror_library(str(src), str(dst), overwrite=False, max_workers=4)

    # Deterministic: sorted order, "Track.mp3" < "track.mp3".
    assert sorted(p.name for p in dst.iterdir()) == ["Track.mp3"]
    assert (dst / "Track.mp3").read_bytes() == b"upper"
    assert summary["copied"] == 1 and summary["skipped"] == 1
    assert [p for p, _ in summary["skipped_files"]] == [str(src / "track.mp3")]


def test_case_only_clash_in_the_listing_is_detected_on_any_file_system(tmp_path, monkeypatch):
    """The clash detection itself, on Windows too: the source *lists* both
    names (as a case-sensitive NAS share or Linux drive would), whatever the
    local file system can store."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "Track.mp3").write_bytes(b"upper")
    dst = tmp_path / "dst"
    real_walk = os.walk

    def walk(top, *args, **kwargs):
        for root, dirs, files in real_walk(top, *args, **kwargs):
            if os.path.normcase(root) == os.path.normcase(str(src)):
                files = [*files, "track.mp3"]
            yield root, dirs, files

    monkeypatch.setattr(mirror.os, "walk", walk)
    summary = mirror.mirror_library(str(src), str(dst), overwrite=False, max_workers=4)

    assert sorted(p.name for p in dst.iterdir()) == ["Track.mp3"]
    assert (dst / "Track.mp3").read_bytes() == b"upper"
    assert summary["copied"] == 1 and summary["skipped"] == 1
    (skipped, reason), = summary["skipped_files"]
    assert skipped == str(src / "track.mp3")
    assert "Same destination name" in reason
