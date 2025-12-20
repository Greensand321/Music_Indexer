import pytest

import library_sync
from library_sync import MatchStatus, PerformanceProfile, TrackRecord, compute_quality_score


def _record(name: str, fp: str, ext: str = ".flac", bitrate: int | None = 320000) -> TrackRecord:
    return TrackRecord(
        track_id=name,
        path=f"/music/{name}{ext}",
        normalized_path=f"/music/{name}{ext}",
        ext=ext,
        bitrate=bitrate or 0,
        size=1024,
        mtime=0.0,
        fingerprint=fp,
        tags={},
        duration=120,
    )


def test_classification_rules_and_best_match(monkeypatch):
    dist_map = {
        ("lib-exact", "lib-exact"): 0.0,
        ("sig-exact", "lib-exact"): 0.0,
        ("sig-close", "lib-close"): 0.2,
        ("sig-close", "alt-close"): 0.5,
        ("sig-near", "lib-near"): 0.34,
        ("sig-new", "lib-new"): 0.9,
    }

    def fake_distance(fp1, fp2):
        return dist_map.get((fp1, fp2), 1.0)

    monkeypatch.setattr(library_sync, "fingerprint_distance", fake_distance)

    lib = [
        _record("lib-exact", "lib-exact"),
        _record("lib-close", "lib-close"),
        _record("lib-near", "lib-near"),
        _record("alt-close", "alt-close"),
        _record("lib-new", "lib-new"),
    ]
    incoming = [
        _record("inc-exact", "lib-exact"),
        _record("inc-close", "sig-close"),
        _record("inc-near", "sig-near"),
        _record("inc-new", "sig-new"),
    ]

    profile = PerformanceProfile()
    matches = library_sync._match_tracks(
        incoming, lib, {".flac": 0.3, "default": 0.3}, {".flac": 3}, profiler=profile
    )

    statuses = [m.status for m in matches]
    assert statuses == [
        MatchStatus.EXACT_MATCH,
        MatchStatus.COLLISION,
        MatchStatus.LOW_CONFIDENCE,
        MatchStatus.NEW,
    ]
    assert matches[1].existing.track_id == "lib-close"
    assert matches[2].near_miss_margin == pytest.approx(0.075)
    assert profile.signature_shortlists >= 1
    assert len(profile.shortlist_sizes) == len(incoming)


def test_threshold_precedence_and_margin_floor(monkeypatch):
    dist_map = {("inc-wav", "lib-wav"): 0.025}
    monkeypatch.setattr(library_sync, "fingerprint_distance", lambda a, b: dist_map.get((a, b), 1.0))

    matches = library_sync._match_tracks(
        [_record("inc-wav", "inc-wav", ext=".wav")],
        [_record("lib-wav", "lib-wav", ext=".wav")],
        {".wav": 0.01, "default": 0.3},
        {".wav": 2},
    )

    assert matches[0].threshold_used == 0.01
    assert matches[0].near_miss_margin == pytest.approx(0.02)
    assert matches[0].status == MatchStatus.LOW_CONFIDENCE


def test_quality_score_fallback_paths(monkeypatch):
    fmt_priority = {".flac": 3, ".mp3": 1}
    monkeypatch.setattr(library_sync, "fingerprint_distance", lambda *_a, **_k: 0.2)

    match = library_sync._match_tracks(
        [_record("upgrade", "inc", ext=".flac", bitrate=None)],
        [_record("existing", "lib", ext=".mp3", bitrate=None)],
        {".flac": 0.3, "default": 0.3},
        fmt_priority,
    )[0]

    assert compute_quality_score({"ext": ".flac", "bitrate": None}, fmt_priority) == 3
    assert compute_quality_score({"ext": ".mp3", "bitrate": None}, fmt_priority) == 1
    assert match.quality_label == "Potential Upgrade"
    assert match.incoming_score == 3
    assert match.existing_score == 1
