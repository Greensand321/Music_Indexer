import hashlib
import os

import duplicate_consolidation
from duplicate_consolidation import build_consolidation_plan


def test_artwork_loading_deferred_until_cluster(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_read_metadata(path: str, *, include_cover: bool = False, audio=None):
        calls.append((path, include_cover))
        tags = {}
        payloads = [f"art:{os.path.basename(path)}".encode()] if include_cover else []
        return tags, payloads, None, "fake"

    monkeypatch.setattr(duplicate_consolidation, "read_metadata", fake_read_metadata)
    monkeypatch.setattr(duplicate_consolidation, "read_sidecar_artwork_bytes", lambda _path: None)

    clustered_a = os.path.join(tmp_path, "Album", "Song A.flac")
    clustered_b = os.path.join(tmp_path, "Album", "Song A.mp3")
    singleton = os.path.join(tmp_path, "Album", "Song B.flac")
    tracks = [
        {
            "path": clustered_a,
            "fingerprint": "1 2 3 4 5 6 7 8",
            "ext": ".flac",
            "bitrate": 1000,
            "sample_rate": 48000,
            "bit_depth": 24,
            "tags": {"title": "Song A"},
        },
        {
            "path": clustered_b,
            "fingerprint": "1 2 3 4 5 6 7 8",
            "ext": ".mp3",
            "bitrate": 320,
            "sample_rate": 44100,
            "bit_depth": 0,
            "tags": {"title": "Song A"},
        },
        {
            "path": singleton,
            "fingerprint": "9 9 9 9 9 9 9 9",
            "ext": ".flac",
            "bitrate": 900,
            "sample_rate": 44100,
            "bit_depth": 16,
            "tags": {"title": "Song B"},
        },
    ]

    plan = build_consolidation_plan(tracks)

    assert len(plan.groups) == 1
    assert plan.groups[0].artwork_candidates
    assert any(candidate.bytes for candidate in plan.groups[0].artwork_candidates)

    initial_flags = [flag for _path, flag in calls[: len(tracks)]]
    assert initial_flags == [False, False, False]

    include_cover_paths = {path for path, flag in calls if flag}
    assert include_cover_paths == {clustered_a, clustered_b}


def test_normalize_track_keeps_raw_artwork_bytes(tmp_path) -> None:
    payload = b"raw-art"
    track_path = os.path.join(tmp_path, "Album", "Song C.flac")
    raw = {
        "path": track_path,
        "fingerprint": "fp-raw-art",
        "ext": ".flac",
        "bitrate": 1000,
        "sample_rate": 48000,
        "bit_depth": 24,
        "tags": {"title": "Song C"},
        "artwork": [{"bytes": payload, "status": "ok"}],
    }

    normalized = duplicate_consolidation._normalize_track(raw, quick_state=True)

    assert normalized.artwork
    candidate = normalized.artwork[0]
    assert candidate.bytes == payload
    assert candidate.size == len(payload)
    assert candidate.hash == hashlib.sha256(payload).hexdigest()


def test_artwork_status_fields_remain_after_deferral(monkeypatch, tmp_path) -> None:
    def fake_read_metadata(path: str, *, include_cover: bool = False, audio=None):
        tags = {}
        payloads = []
        return tags, payloads, None, "fake"

    monkeypatch.setattr(duplicate_consolidation, "read_metadata", fake_read_metadata)
    monkeypatch.setattr(duplicate_consolidation, "read_sidecar_artwork_bytes", lambda _path: None)

    clustered_a = os.path.join(tmp_path, "Album", "Song D.flac")
    clustered_b = os.path.join(tmp_path, "Album", "Song D.mp3")
    tracks = [
        {
            "path": clustered_a,
            "fingerprint": "1 1 1 1",
            "ext": ".flac",
            "bitrate": 1000,
            "sample_rate": 48000,
            "bit_depth": 24,
            "tags": {"title": "Song D"},
        },
        {
            "path": clustered_b,
            "fingerprint": "1 1 1 1",
            "ext": ".mp3",
            "bitrate": 320,
            "sample_rate": 44100,
            "bit_depth": 0,
            "tags": {"title": "Song D"},
        },
    ]

    plan = build_consolidation_plan(tracks)
    group = plan.groups[0]

    assert clustered_a in group.artwork_unknown_reasons
    assert "art missing/unreadable" in group.artwork_unknown_reasons[clustered_a]
    assert clustered_b in group.artwork_unknown_reasons
    assert "art missing/unreadable" in group.artwork_unknown_reasons[clustered_b]
