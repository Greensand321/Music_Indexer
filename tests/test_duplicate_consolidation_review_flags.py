import os
import threading

from duplicate_consolidation import build_consolidation_plan


def test_remix_keywords_trigger_review_flag(tmp_path):
    tracks = [
        {
            "path": os.path.join(tmp_path, "Song (Remix).flac"),
            "fingerprint": "fp-keyword-1",
            "ext": ".flac",
            "bitrate": 1000,
            "sample_rate": 48000,
            "bit_depth": 24,
            "tags": {"title": "Song (Remix)"},
        },
        {
            "path": os.path.join(tmp_path, "Song.flac"),
            "fingerprint": "fp-keyword-1",
            "ext": ".flac",
            "bitrate": 900,
            "sample_rate": 44100,
            "bit_depth": 16,
            "tags": {"title": "Song"},
        },
    ]

    plan = build_consolidation_plan(tracks)
    assert plan.groups
    assert any("remix" in flag.lower() or "speed" in flag.lower() for flag in plan.groups[0].review_flags)


def test_missing_artwork_sets_global_review_flag(tmp_path):
    tracks = [
        {
            "path": os.path.join(tmp_path, "Album", "Track A.flac"),
            "fingerprint": "fp-art-1",
            "ext": ".flac",
            "bitrate": 800,
            "sample_rate": 44100,
            "bit_depth": 16,
            "tags": {"title": "Track A", "album": "Album"},
        },
        {
            "path": os.path.join(tmp_path, "Album", "Track A (copy).flac"),
            "fingerprint": "fp-art-1",
            "ext": ".flac",
            "bitrate": 700,
            "sample_rate": 44100,
            "bit_depth": 16,
            "tags": {"title": "Track A", "album": "Album"},
        },
    ]

    plan = build_consolidation_plan(tracks, cancel_event=threading.Event())
    assert plan.review_flags
    assert any("No artwork available" in flag for flag in plan.review_flags)


def test_cancellation_sets_review_flag(tmp_path):
    tracks = [
        {"path": os.path.join(tmp_path, f"Track {idx}.flac"), "fingerprint": "fp-cancel", "ext": ".flac"}
        for idx in range(3)
    ]
    cancel = threading.Event()
    cancel.set()

    plan = build_consolidation_plan(tracks, cancel_event=cancel)

    assert "Cancelled before normalization." in plan.review_flags
    assert not plan.groups  # cancelled before processing any tracks


def test_multiple_singles_with_conflicting_artwork_require_review(tmp_path):
    tracks = [
        {
            "path": os.path.join(tmp_path, "Singles", "Track A.flac"),
            "fingerprint": "fp-art-ambiguous",
            "ext": ".flac",
            "bitrate": 256,
            "sample_rate": 44100,
            "bit_depth": 16,
            "tags": {"album_type": "single", "cover_hash": "cover-a"},
        },
        {
            "path": os.path.join(tmp_path, "Singles", "Track A copy.flac"),
            "fingerprint": "fp-art-ambiguous",
            "ext": ".flac",
            "bitrate": 256,
            "sample_rate": 44100,
            "bit_depth": 16,
            "tags": {"album_type": "single", "cover_hash": "cover-b"},
        },
    ]

    plan = build_consolidation_plan(tracks)
    group = plan.groups[0]

    assert any("artwork" in flag.lower() for flag in group.review_flags)
    assert any("Artwork selection ambiguous" in flag for flag in plan.review_flags)
