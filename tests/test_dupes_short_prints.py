"""A short clip is not a duplicate of the full song it was cut from.

fpcalc fingerprints the first 120 s, so a 60-second clip's print is a prefix of
the full song's. The distance only compared the overlap, scored the pair 0.0
("exact duplicate"), and the planner grouped them — keeping the clip whenever
it was the lossless file, so the full song became the quarantined loser.
"""
from __future__ import annotations

import pytest

import near_duplicate_detector as nd
from config import EXACT_DUPLICATE_THRESHOLD, MIXED_CODEC_THRESHOLD_BOOST, NEAR_DUPLICATE_THRESHOLD
from duplicate_consolidation import build_consolidation_plan
from _helpers.dupes_chromaprint import compress_chromaprint, fake_raw_print, flip_bits
from _helpers.dupes_plans import write

FULL = fake_raw_print("full song", 950)  # ~118 s of items


@pytest.mark.parametrize("clip_items", [40, 240, 475])
def test_a_prefix_clip_is_far_from_the_full_print(clip_items):
    distance = nd.raw_fingerprint_distance(FULL, FULL[:clip_items], count_unmatched=True)
    assert distance > NEAR_DUPLICATE_THRESHOLD + MIXED_CODEC_THRESHOLD_BOOST, distance
    # Symmetric.
    assert nd.raw_fingerprint_distance(FULL[:clip_items], FULL, count_unmatched=True) == distance


def test_a_slightly_trimmed_copy_is_still_an_exact_duplicate():
    # A few items shorter (silence trimming, a different pregap) stays at 0.
    assert nd.raw_fingerprint_distance(FULL, FULL[5:], count_unmatched=True) == 0.0
    assert nd.raw_fingerprint_distance(FULL, FULL[:-6], count_unmatched=True) == 0.0
    lossy = flip_bits(FULL, 100)
    assert nd.raw_fingerprint_distance(FULL, lossy[3:], count_unmatched=True) <= EXACT_DUPLICATE_THRESHOLD


def test_a_copy_missing_its_last_tenth_is_only_a_near_duplicate():
    distance = nd.raw_fingerprint_distance(FULL, FULL[:850], count_unmatched=True)
    assert EXACT_DUPLICATE_THRESHOLD < distance <= NEAR_DUPLICATE_THRESHOLD


def test_the_planner_does_not_group_a_clip_with_its_song(tmp_path):
    tags = {"artist": "Artist", "title": "Song", "album": "Album"}
    song = write(tmp_path / "A" / "Song.mp3")
    clip = write(tmp_path / "B" / "Song.flac")
    tracks = [
        {"path": str(song), "fingerprint": compress_chromaprint(FULL), "ext": ".mp3", "tags": dict(tags)},
        {"path": str(clip), "fingerprint": compress_chromaprint(FULL[:240]), "ext": ".flac", "tags": dict(tags)},
    ]
    plan = build_consolidation_plan(
        tracks,
        exact_duplicate_threshold=EXACT_DUPLICATE_THRESHOLD,
        near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD,
        mixed_codec_threshold_boost=MIXED_CODEC_THRESHOLD_BOOST,
    )
    assert plan.groups == []


def test_duplicate_distance_on_compressed_prints_and_default_stays_overlap_only():
    song, clip = compress_chromaprint(FULL), compress_chromaprint(FULL[:240])
    assert nd.duplicate_fingerprint_distance(song, clip) > NEAR_DUPLICATE_THRESHOLD + MIXED_CODEC_THRESHOLD_BOOST
    # The plain distance (Similarity page, vendored Library Sync parity) is unchanged.
    assert nd.fingerprint_distance(song, clip) == 0.0
