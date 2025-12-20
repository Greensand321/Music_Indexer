import csv
import json
import sys
import types
from datetime import datetime, timezone

# Stub heavy dependencies before importing modules under test
acoustid_stub = types.ModuleType("acoustid")
acoustid_stub.fingerprint_file = lambda path: (0, "stub-fp")
sys.modules.setdefault("acoustid", acoustid_stub)

silence_stub = types.ModuleType("pydub.silence")
silence_stub.detect_nonsilent = lambda *args, **kwargs: []

audio_segment = type("AudioSegment", (), {"from_file": staticmethod(lambda path: None)})
pydub_stub = types.ModuleType("pydub")
pydub_stub.AudioSegment = audio_segment
pydub_stub.silence = silence_stub

sys.modules.setdefault("pydub", pydub_stub)
sys.modules.setdefault("pydub.silence", silence_stub)

from library_sync import MatchResult, MatchStatus, TrackRecord
from library_sync_review_report import (
    DEFAULT_REPORT_VERSION,
    calculate_scan_config_hash,
    export_report,
)
from library_sync_review_state import ReviewStateStore


def _track(track_id: str, path: str, ext: str = ".mp3", bitrate: int = 192) -> TrackRecord:
    return TrackRecord(
        track_id=track_id,
        path=path,
        normalized_path=path,
        ext=ext,
        bitrate=bitrate,
        size=1,
        mtime=0.0,
        fingerprint="fp",
        tags={},
        duration=60,
    )


def _match(
    incoming_id: str,
    existing_id: str | None,
    status: MatchStatus = MatchStatus.COLLISION,
    incoming_score: int = 10,
    existing_score: int | None = 5,
) -> MatchResult:
    incoming = _track(incoming_id, f"/incoming/{incoming_id}")
    existing = _track(existing_id, f"/library/{existing_id}") if existing_id else None
    return MatchResult(
        incoming=incoming,
        existing=existing,
        status=status,
        distance=0.1 if existing else None,
        threshold_used=0.3,
        near_miss_margin=0.05,
        confidence=0.8 if existing else 0.0,
        candidates=[],
        quality_label="Potential Upgrade"
        if existing_score is not None and incoming_score > existing_score
        else "Keep Existing",
        incoming_score=incoming_score,
        existing_score=existing_score,
    )


def test_export_report_json_includes_required_fields(tmp_path) -> None:
    store = ReviewStateStore()
    results = [
        _match("inc-new", None, status=MatchStatus.NEW, incoming_score=7, existing_score=None),
        _match("inc-collision", "lib-b", status=MatchStatus.COLLISION, incoming_score=9, existing_score=5),
    ]
    store.flag_for_copy(results[0].incoming.track_id)
    store.flag_for_replace(results[1])
    store.set_note(results[0].incoming.track_id, "verify tags")

    scan_config = {"global_threshold": 0.3, "per_format_overrides": {"flac": 0.2}}
    ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
    captured_summary: dict[str, int] = {}

    def preview(summary: dict[str, int]) -> bool:
        captured_summary.update(summary)
        return True

    path = tmp_path / "report.json"
    export_report(
        str(path),
        results,
        store,
        scan_config,
        fmt="json",
        report_version=2,
        timestamp=ts,
        preview_handler=preview,
    )

    data = json.loads(path.read_text())
    assert data["schema_version"] == 2
    assert data["scan_config_hash"] == calculate_scan_config_hash(scan_config)
    assert len(data["items"]) == 2

    first = data["items"][0]
    assert first["incoming_path"].endswith("inc-new")
    assert first["existing_path"] is None
    assert first["user_flag"] == "copy"
    assert first["notes"] == "verify tags"
    assert first["timestamp"] == ts.isoformat()
    assert captured_summary["flagged_copy"] == 1
    assert captured_summary["flagged_replace"] == 1


def test_export_report_csv_and_summary_counts(tmp_path) -> None:
    store = ReviewStateStore()
    results = [
        _match("inc-new", None, status=MatchStatus.NEW, incoming_score=3, existing_score=None),
        _match("inc-collision", "lib-1", status=MatchStatus.COLLISION, incoming_score=5, existing_score=2),
        _match(
            "inc-low",
            "lib-low",
            status=MatchStatus.LOW_CONFIDENCE,
            incoming_score=4,
            existing_score=6,
        ),
    ]
    store.flag_for_copy(results[2].incoming.track_id)
    store.flag_for_replace(results[1])

    scan_config = {"default": 0.3}
    ts = datetime(2024, 1, 3, tzinfo=timezone.utc)
    previews: list[dict[str, int]] = []

    def preview(summary: dict[str, int]) -> bool:
        previews.append(summary)
        return True

    path = tmp_path / "report.csv"
    export_report(
        str(path),
        results,
        store,
        scan_config,
        fmt="csv",
        timestamp=ts,
        preview_handler=preview,
    )

    rows = list(csv.DictReader(path.read_text().splitlines()))
    assert len(rows) == len(results)
    assert rows[0]["schema_version"] == str(DEFAULT_REPORT_VERSION)
    assert rows[1]["existing_path"].endswith("lib-1")
    assert rows[2]["user_flag"] == "copy"
    assert rows[0]["status"] == MatchStatus.NEW.value
    assert previews[0]["collisions"] == 1
    assert previews[0]["low_confidence"] == 1
    assert previews[0]["flagged_replace"] == 1


def test_export_report_aborts_when_preview_rejected(tmp_path) -> None:
    store = ReviewStateStore()
    results = [_match("inc-new", None, status=MatchStatus.NEW, existing_score=None)]
    path = tmp_path / "rejected.json"

    def reject(summary: dict[str, int]) -> bool:
        return False

    before = set(tmp_path.iterdir())
    returned_path = export_report(str(path), results, store, {}, preview_handler=reject)
    after = set(tmp_path.iterdir())

    assert returned_path == ""
    assert not path.exists()
    assert before == after
