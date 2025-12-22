import json
import os

from duplicate_consolidation import (
    build_consolidation_plan,
    render_consolidation_preview,
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


def test_plan_contains_metadata_and_quality_summary(tmp_path) -> None:
    plan = build_consolidation_plan(_sample_tracks(str(tmp_path)))
    assert plan.groups
    group = plan.groups[0]

    assert group.metadata_changes  # winner tags should be merged
    assert group.winner_quality["reasons"]  # rationale present
    assert group.playlist_impact.entries == len(group.losers)
    assert group.winner_current_tags  # real tags captured
    assert group.current_tags[group.winner_path]["title"] == "Song A"
    assert group.chosen_artwork_source  # artwork decision captured even if unchanged
    assert group.tag_source is not None


def test_preview_render_and_export(tmp_path) -> None:
    plan = build_consolidation_plan(_sample_tracks(str(tmp_path)))
    html_path = tmp_path / "preview.html"
    json_path = tmp_path / "preview.json"

    render_consolidation_preview(plan, str(html_path))
    export_consolidation_preview(plan, str(json_path))

    html = html_path.read_text()
    assert "Duplicate Consolidation Preview" in html
    assert plan.groups[0].winner_path in html

    data = json.loads(json_path.read_text())
    assert data["summary"]["groups"] == len(plan.groups)
    assert data["summary"]["review_required"] == plan.review_required_count
