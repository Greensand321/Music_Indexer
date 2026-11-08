"""The exported Library Sync HTML report has no fake action buttons (area: improve).

It used to render "Replace All" / "Replace with Better" / "Cancel" buttons
that no script wired to anything. A static file cannot safely drive file
operations, so the report now says how to replace instead.
"""
from __future__ import annotations

from _helpers.module_isolation import make_acoustid_stub, stub_if_unavailable

stub_if_unavailable({"acoustid": make_acoustid_stub()})

import library_sync  # noqa: E402
from library_sync_review_report import REPLACE_HINT, export_review_report_html  # noqa: E402
from library_sync_review_state import ReviewStateStore  # noqa: E402
from library_sync_types import MatchResult, MatchStatus, TrackRecord  # noqa: E402


def _track(track_id, path):
    return TrackRecord(
        track_id=track_id, path=path, normalized_path=path, ext=".mp3", bitrate=192,
        size=1, mtime=0.0, fingerprint="fp", tags={}, duration=60,
    )


def _result():
    return MatchResult(
        incoming=_track("inc", "/incoming/a.flac"), existing=_track("lib", "/library/a.mp3"),
        status=MatchStatus.COLLISION, distance=0.01, threshold_used=0.3, near_miss_margin=0.05,
        confidence=0.9, candidates=[], quality_label="Potential Upgrade",
        incoming_score=10, existing_score=5,
    )


def test_report_has_no_buttons_and_explains_how_to_replace(tmp_path):
    res = _result()
    store = ReviewStateStore()
    store.flag_for_replace(res)
    out = tmp_path / "report.html"
    export_review_report_html(
        str(out), [res], library_root="/library", incoming_root="/incoming",
        thresholds={"default": 0.3}, mixed_codec_threshold_boost=0.03, format_priority={".flac": 3},
        flags=store,
    )
    text = out.read_text(encoding="utf-8")
    assert "<button" not in text
    assert "data-action" not in text
    assert "Replace All" not in text and "Replace with Better" not in text
    assert REPLACE_HINT == "To replace, flag tracks in AlphaDEX → Library Sync and build a plan"
    assert "To replace, flag tracks in AlphaDEX → Library Sync and build a plan" in text
    # The review content itself is unchanged.
    assert "<tr><th>Flagged Replace</th><td>1</td></tr>" in text
    assert "<td>Replace</td>" in text


def test_dead_replacement_plan_api_is_gone():
    # It had no caller and backed only the fake buttons; replacing is done by
    # flagging tracks and building a normal plan (compute_library_sync_plan).
    assert not hasattr(library_sync, "build_review_replacement_plan")
    assert not hasattr(library_sync, "ReviewAction")
