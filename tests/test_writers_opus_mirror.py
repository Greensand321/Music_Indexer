"""Opus mirror name clashes (writers review, finding 6a).

``Song.flac`` (converted to ``Song.opus``) and ``Song.opus`` (copied) in one
folder both targeted ``Song.opus``; whichever worker finished last won. The
plan now decides up front: the file copied as-is wins, the other is skipped
and reported.
"""
from __future__ import annotations

from utils import opus_library_mirror as mirror


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
    (src / "track.mp3").write_bytes(b"lower")
    (src / "Track.mp3").write_bytes(b"upper")
    dst = tmp_path / "dst"

    summary = mirror.mirror_library(str(src), str(dst), overwrite=False, max_workers=4)

    # Deterministic: sorted order, "Track.mp3" < "track.mp3".
    assert sorted(p.name for p in dst.iterdir()) == ["Track.mp3"]
    assert (dst / "Track.mp3").read_bytes() == b"upper"
    assert summary["copied"] == 1 and summary["skipped"] == 1
    assert [p for p, _ in summary["skipped_files"]] == [str(src / "track.mp3")]
