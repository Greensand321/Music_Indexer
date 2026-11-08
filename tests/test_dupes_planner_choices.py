"""Which copy the planner keeps, and which groups it forms.

Before:
* a copy with readable cover art outranked every copy without — an MP3 with a
  cover beat the FLAC without one, so the lossless master was the "loser";
* between lossless copies the bitrate decided, so an uncompressed WAV always
  beat the same audio as FLAC (the docs and ``config.FORMAT_PRIORITY`` say
  FLAC > WAV);
* a symlink/hard link and its target were grouped as exact duplicates; when
  the link won, the real file was quarantined or deleted;
* every group with an undated copy was flagged "Placeholder metadata" (the
  normalised "Unknown" year matched the placeholder token "unknown"), and
  every artwork-less copy was reported as "Artwork present but unreadable".
"""
from __future__ import annotations

import io
import os
import random

import pytest

import near_duplicate_detector as nd
from duplicate_consolidation import build_consolidation_plan
from _helpers.dupes_chromaprint import real_fingerprints
from _helpers.dupes_plans import write

TAGS = {"artist": "Artist", "title": "Song", "album": "Album", "date": "2001"}


def _track(path, ext, *, bitrate, sample_rate=44100, bit_depth=16, channels=2, art=None, tags=None, fp="1 2 3 4 5 6 7 8"):
    raw = {
        "path": str(path), "fingerprint": fp, "ext": ext,
        "bitrate": bitrate, "sample_rate": sample_rate, "bit_depth": bit_depth, "channels": channels,
        "tags": dict(tags if tags is not None else TAGS),
    }
    if art is not None:
        raw["artwork"] = [{"bytes": art, "status": "ok"}]
    return raw


def _jpeg() -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "orange").save(buf, "JPEG")
    return buf.getvalue()


def test_lossless_without_art_beats_lossy_with_art(tmp_path):
    flac = write(tmp_path / "A" / "Song.flac")
    mp3 = write(tmp_path / "B" / "Song.mp3")
    plan = build_consolidation_plan([
        _track(flac, ".flac", bitrate=900),
        _track(mp3, ".mp3", bitrate=320, art=_jpeg()),
    ])
    (group,) = plan.groups
    assert group.winner_path == str(flac)
    assert group.losers == [str(mp3)]


def test_flac_beats_wav_at_the_same_resolution(tmp_path):
    flac = write(tmp_path / "A" / "Song.flac")
    wav = write(tmp_path / "B" / "Song.wav")
    plan = build_consolidation_plan([
        _track(wav, ".wav", bitrate=1411),
        _track(flac, ".flac", bitrate=880),
    ])
    (group,) = plan.groups
    assert group.winner_path == str(flac)
    assert any("Preferred lossless format" in r for r in group.winner_quality["reasons"])


def test_higher_resolution_lossless_still_wins(tmp_path):
    flac = write(tmp_path / "A" / "Song.flac")
    wav = write(tmp_path / "B" / "Song.wav")
    plan = build_consolidation_plan([
        _track(flac, ".flac", bitrate=880),
        _track(wav, ".wav", bitrate=4608, sample_rate=96000, bit_depth=24),
    ])
    assert plan.groups[0].winner_path == str(wav)


def test_higher_bitrate_lossy_wins(tmp_path):
    hi = write(tmp_path / "A" / "Song.mp3")
    lo = write(tmp_path / "B" / "Song.mp3")
    plan = build_consolidation_plan([_track(lo, ".mp3", bitrate=128), _track(hi, ".mp3", bitrate=320)])
    assert plan.groups[0].winner_path == str(hi)


def test_winner_and_groups_do_not_depend_on_input_order(tmp_path):
    paths = [write(tmp_path / name / "Song.flac") for name in ("b", "a", "c", "d")]
    tracks = [_track(p, ".flac", bitrate=900) for p in paths]
    expected = build_consolidation_plan(list(tracks)).to_dict()["groups"]
    rng = random.Random(7)
    for _ in range(5):
        shuffled = list(tracks)
        rng.shuffle(shuffled)
        groups = build_consolidation_plan(shuffled).to_dict()["groups"]
        assert [(g["winner_path"], g["losers"]) for g in groups] == [(g["winner_path"], g["losers"]) for g in expected]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_symlink_and_its_target_are_not_duplicates(tmp_path):
    real = write(tmp_path / "Real" / "Song.mp3")
    link = tmp_path / "Linked" / "Song.flac"  # the link even looks like the better copy
    link.parent.mkdir()
    link.symlink_to(real)
    logs = []
    plan = build_consolidation_plan(
        [_track(real, ".mp3", bitrate=320), _track(link, ".flac", bitrate=900)],
        log_callback=logs.append,
    )
    assert plan.groups == []
    assert any("Same file reached through two paths" in m for m in logs)


def test_hard_links_are_one_file(tmp_path):
    first = write(tmp_path / "A" / "Song.mp3")
    second = tmp_path / "B" / "Song.mp3"
    second.parent.mkdir()
    try:
        os.link(first, second)
    except OSError:
        pytest.skip("hard links unsupported here")
    plan = build_consolidation_plan([_track(first, ".mp3", bitrate=320), _track(second, ".mp3", bitrate=320)])
    assert plan.groups == []


def test_undated_tracks_are_not_flagged_as_placeholder_metadata(tmp_path):
    undated = {"artist": "Artist", "title": "Song", "album": "Album"}
    a = write(tmp_path / "A" / "Song.flac")
    b = write(tmp_path / "B" / "Song.mp3")
    plan = build_consolidation_plan([
        _track(a, ".flac", bitrate=900, tags=undated),
        _track(b, ".mp3", bitrate=320, tags=undated),
    ])
    (group,) = plan.groups
    assert not any("Placeholder" in flag for flag in group.review_flags)
    assert not plan.placeholders_present
    assert not any("Placeholder" in flag for flag in plan.review_flags)


def test_real_placeholder_metadata_is_still_flagged(tmp_path):
    tags = {"artist": "Unknown Artist", "title": "Song", "album": "Album"}
    a = write(tmp_path / "A" / "Song.flac")
    b = write(tmp_path / "B" / "Song.mp3")
    plan = build_consolidation_plan([_track(a, ".flac", bitrate=900, tags=tags), _track(b, ".mp3", bitrate=320, tags=tags)])
    assert any("Placeholder" in flag for flag in plan.groups[0].review_flags)


def test_files_without_artwork_are_not_reported_as_unreadable(tmp_path):
    import shutil

    from library_factory import encode_clip, ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("needs ffmpeg for real (cover-less) audio files")
    a, b = tmp_path / "A" / "Song.flac", tmp_path / "B" / "Song.mp3"
    for path in (a, b):
        path.parent.mkdir(parents=True)
        shutil.copyfile(encode_clip(path.suffix, 2.0, 440.0, tmp_path / "clips"), path)
    plan = build_consolidation_plan([_track(a, ".flac", bitrate=900), _track(b, ".mp3", bitrate=320)])
    (group,) = plan.groups
    flags = " | ".join(group.review_flags + plan.review_flags)
    assert "unreadable" not in flags and "extraction failed" not in flags
    assert "No artwork candidates available" in flags


# ── grouping semantics (pins) ────────────────────────────────────────────────

def test_chained_matches_do_not_merge_into_one_group(tmp_path):
    # A~B and B~C are within the threshold, A~C is not: complete linkage keeps C out.
    base = list(range(1, 11))
    fps = {
        "A": base,
        "B": base[:9] + [99],
        "C": base[:8] + [98, 99],
    }
    paths = {name: write(tmp_path / name / "Song.mp3") for name in fps}
    tracks = [
        _track(paths[n], ".mp3", bitrate=320 - i, fp=" ".join(map(str, fp)))
        for i, (n, fp) in enumerate(fps.items())
    ]
    plan = build_consolidation_plan(tracks, exact_duplicate_threshold=0.0, near_duplicate_threshold=0.1)
    assert [sorted([g.winner_path, *g.losers]) for g in plan.groups] == [sorted([str(paths["A"]), str(paths["B"])])]


def test_threshold_boundary_is_inclusive(tmp_path):
    a = write(tmp_path / "A" / "Song.mp3")
    b = write(tmp_path / "B" / "Song.mp3")
    tracks = [
        _track(a, ".mp3", bitrate=320, fp="1 2 3 4 5 6 7 8 9 10"),
        _track(b, ".mp3", bitrate=128, fp="1 2 3 4 5 6 7 8 9 11"),
    ]
    assert len(build_consolidation_plan(tracks, exact_duplicate_threshold=0.0, near_duplicate_threshold=0.1).groups) == 1
    assert build_consolidation_plan(tracks, exact_duplicate_threshold=0.0, near_duplicate_threshold=0.09).groups == []


def test_distance_is_symmetric_on_real_prints():
    fps = real_fingerprints()
    names = sorted(fps)
    for left in names:
        for right in names:
            assert nd.fingerprint_distance(fps[left], fps[right]) == nd.fingerprint_distance(fps[right], fps[left])
