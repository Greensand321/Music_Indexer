import sys
import types

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
from library_sync_review_state import (
    ReviewStateStore,
    filter_collisions_only,
    quality_delta,
    sort_by_quality_delta,
)


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
        quality_label="Potential Upgrade" if existing_score is not None and incoming_score > existing_score else "Keep Existing",
        incoming_score=incoming_score,
        existing_score=existing_score,
    )


def test_collisions_filter_excludes_new_rows() -> None:
    results = [
        _match("inc-new", None, status=MatchStatus.NEW, existing_score=None),
        _match("inc-match", "lib-a", status=MatchStatus.COLLISION),
        _match("inc-exact", "lib-b", status=MatchStatus.EXACT_MATCH),
    ]
    filtered = filter_collisions_only(results)
    assert len(filtered) == 2
    assert all(res.existing is not None for res in filtered)


def test_quality_delta_sort_orders_by_upgrade() -> None:
    upgrade = _match("inc-upgrade", "lib-low", incoming_score=12, existing_score=4)
    downgrade = _match("inc-downgrade", "lib-high", incoming_score=5, existing_score=15)
    unmatched = _match("inc-new", None, status=MatchStatus.NEW, existing_score=None)
    sorted_results = sort_by_quality_delta([unmatched, downgrade, upgrade])
    assert sorted_results[0].incoming.track_id == upgrade.incoming.track_id
    assert sorted_results[-1].incoming.track_id == unmatched.incoming.track_id
    assert quality_delta(upgrade) == 8
    assert quality_delta(downgrade) == -10
    assert quality_delta(unmatched) is None


def test_replace_flags_rebind_when_best_match_changes() -> None:
    store = ReviewStateStore()
    first_result = _match("inc-1", "lib-1")
    store.flag_for_replace(first_result)

    updated_results = [
        _match("inc-1", "lib-2", status=MatchStatus.COLLISION),
        _match("inc-2", None, status=MatchStatus.NEW, existing_score=None),
    ]
    warnings = store.reconcile_best_matches(updated_results)

    assert store.replace_target("inc-1") == "lib-2"
    assert len(warnings) == 1
    assert "lib-1" in warnings[0] and "lib-2" in warnings[0]


def test_replace_flags_clear_when_best_match_missing() -> None:
    store = ReviewStateStore()
    first_result = _match("inc-1", "lib-1")
    store.flag_for_replace(first_result)

    warnings = store.reconcile_best_matches([_match("inc-1", None, status=MatchStatus.NEW, existing_score=None)])

    assert store.replace_target("inc-1") is None
    assert len(warnings) == 1
    assert "Cleared replace flag" in warnings[0]


def test_flags_survive_filtering_and_sorting() -> None:
    store = ReviewStateStore()
    flagged_copy = _match("inc-copy", "lib-copy")
    flagged_replace = _match("inc-replace", "lib-replace")

    store.flag_for_copy(flagged_copy.incoming.track_id)
    store.flag_for_replace(flagged_replace)

    filtered = filter_collisions_only([flagged_copy, flagged_replace])
    sorted_results = sort_by_quality_delta(filtered)

    assert len(sorted_results) == 2
    assert store.is_copy_flagged(flagged_copy.incoming.track_id)
    assert store.replace_target(flagged_replace.incoming.track_id) == flagged_replace.existing.track_id
