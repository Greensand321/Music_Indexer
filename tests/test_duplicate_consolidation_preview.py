import json
import hashlib
import os

from duplicate_consolidation import (
    build_consolidation_plan,
    export_consolidation_preview,
)


def _sample_tracks(root: str) -> list[dict[str, object]]:
    return [
        {
            "path": os.path.join(root, "Album", "Song A.flac"),
            "fingerprint": "fp-test-001",
            "ext": ".flac",
            "bitrate": 1000,
            "sample_rate": 48000,
            "bit_depth": 24,
            "tags": {"title": "Song A", "album": "Album A", "album_type": "album", "cover_hash": "cover-a"},
        },
        {
            "path": os.path.join(root, "Album", "Song A.mp3"),
            "fingerprint": "fp-test-001",
            "ext": ".mp3",
            "bitrate": 320,
            "sample_rate": 44100,
            "bit_depth": 0,
            "tags": {
                "title": "Song A",
                "album": "Album A",
                "albumartist": "Various",
                "track": 1,
                "discnumber": 1,
            },
        },
    ]


def _artwork_payload(seed: str) -> dict[str, object]:
    payload = seed.encode()
    return {
        "hash": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "width": 600,
        "height": 600,
        "status": "ok",
        "bytes": payload,
    }


def test_plan_contains_metadata_and_quality_summary(tmp_path) -> None:
    plan = build_consolidation_plan(_sample_tracks(str(tmp_path)))
    assert plan.groups
    group = plan.groups[0]

    assert group.metadata_changes  # winner tags should be merged
    assert group.winner_quality["reasons"]  # rationale present
    assert group.playlist_impact.entries == 0
    assert group.winner_current_tags  # real tags captured
    assert group.current_tags[group.winner_path]["title"] == "Song A"
    assert group.chosen_artwork_source  # artwork decision captured even if unchanged
    assert group.tag_source is not None


def test_preview_export(tmp_path) -> None:
    plan = build_consolidation_plan(_sample_tracks(str(tmp_path)))
    json_path = tmp_path / "preview.json"

    export_consolidation_preview(plan, str(json_path))

    data = json.loads(json_path.read_text())
    assert data["summary"]["groups"] == len(plan.groups)
    assert data["summary"]["review_required"] == plan.review_required_count


def test_album_winner_gets_single_artwork_applied(tmp_path) -> None:
    album_path = os.path.join(tmp_path, "Album", "Song A.flac")
    single_path = os.path.join(tmp_path, "Singles", "Song A.mp3")
    tracks = [
        {
            "path": album_path,
            "fingerprint": "fp-art-apply",
            "ext": ".flac",
            "bitrate": 1000,
            "sample_rate": 48000,
            "bit_depth": 24,
            "tags": {"title": "Song A", "album": "Album A", "track": 1, "album_type": "album"},
        },
        {
            "path": single_path,
            "fingerprint": "fp-art-apply",
            "ext": ".mp3",
            "bitrate": 320,
            "sample_rate": 44100,
            "bit_depth": 0,
            "tags": {"title": "Song A", "album_type": "single"},
            "artwork": [_artwork_payload("single-art")],
        },
    ]

    plan = build_consolidation_plan(tracks)
    group = plan.groups[0]

    assert group.artwork_status == "apply"
    assert group.artwork
    assert group.artwork[0].source == single_path
    assert group.artwork[0].target == album_path
    assert group.chosen_artwork_source.get("context") == "single"
    assert any("single artwork" in ev.lower() for ev in group.artwork_evidence)


def test_single_winner_keeps_artwork_even_with_album_cover(tmp_path) -> None:
    single_path = os.path.join(tmp_path, "Singles", "Song B.flac")
    album_path = os.path.join(tmp_path, "Album", "Song B.mp3")
    tracks = [
        {
            "path": single_path,
            "fingerprint": "fp-single-winner",
            "ext": ".flac",
            "bitrate": 1100,
            "sample_rate": 48000,
            "bit_depth": 24,
            "tags": {"title": "Song B", "album_type": "single"},
            "artwork": [_artwork_payload("single-winner")],
        },
        {
            "path": album_path,
            "fingerprint": "fp-single-winner",
            "ext": ".mp3",
            "bitrate": 256,
            "sample_rate": 44100,
            "bit_depth": 0,
            "tags": {"title": "Song B", "album": "Album B", "album_type": "album"},
            "artwork": [_artwork_payload("album-art")],
        },
    ]

    plan = build_consolidation_plan(tracks)
    group = plan.groups[0]

    assert group.winner_path == single_path
    assert group.artwork_status != "apply"
    assert not group.artwork  # no cross-application when single wins
    assert any("single-context winner" in ev.lower() for ev in group.artwork_evidence)


def test_metadata_normalization_prefers_album_source(tmp_path) -> None:
    single_path = os.path.join(tmp_path, "Singles", "Song C.flac")
    album_path = os.path.join(tmp_path, "Album", "Song C.mp3")
    tracks = [
        {
            "path": single_path,
            "fingerprint": "fp-tag-source",
            "ext": ".flac",
            "bitrate": 900,
            "sample_rate": 48000,
            "bit_depth": 24,
            "tags": {"title": "Song C", "album_type": "single"},
        },
        {
            "path": album_path,
            "fingerprint": "fp-tag-source",
            "ext": ".mp3",
            "bitrate": 320,
            "sample_rate": 44100,
            "bit_depth": 0,
            "tags": {"title": "Song C", "album": "Album C", "albumartist": "Artist C", "track": 1, "year": 2020, "album_type": "album"},
        },
    ]

    plan = build_consolidation_plan(tracks)
    group = plan.groups[0]

    assert group.tag_source == album_path
    assert group.planned_winner_tags["album"] == "Album C"
    assert group.planned_winner_tags["albumartist"] == "Artist C"
    assert group.planned_winner_tags["year"] == 2020
    assert "album" in group.metadata_changes
