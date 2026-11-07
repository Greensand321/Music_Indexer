"""Library Sync end to end on a mixed-codec library (area: syncmatch).

scan (``compare_libraries``) → plan (``compute_library_sync_plan`` from the
match results) → ``execute_plan``, on real tagged files. The library has 73
tracks in five codecs; the incoming folder holds a FLAC whose MP3 copy is in
the middle of the library, a same-codec duplicate of the *last* library
track, a 0.3 s-shifted copy and one genuinely new track (see
``_helpers/syncmatch_library.py``). Before the shortlist fix the three
duplicates all came out NEW and were copied into the library again.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import library_sync
from _helpers.syncmatch_library import build_sync_scenario, install_title_fingerprints
from library_factory import ffmpeg_available
from library_sync import ExecutionDecision
from library_sync_types import MatchStatus

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="needs ffmpeg to build MP3/FLAC/M4A/Ogg/Opus files")


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    install_title_fingerprints(monkeypatch)
    return build_sync_scenario(tmp_path / "case")


def _compare(sc, db):
    return library_sync.compare_libraries(
        str(sc.library), str(sc.incoming), str(db), include_match_objects=True, include_profile=True,
    )


def _by_name(result):
    return {Path(m.incoming.path).name: m for m in result["match_objects"]}


def _assert_classification(result, sc):
    m = _by_name(result)
    assert set(m) == {"song_a.flac", "late.mp3", "shifted.mp3", "brand_new.flac"}

    flac = m["song_a.flac"]
    assert flac.status == MatchStatus.COLLISION, (flac.status, flac.distance)
    assert Path(flac.existing.path) == sc.lib_song_a
    assert flac.quality_label == "Potential Upgrade"
    assert flac.distance < 0.01

    late = m["late.mp3"]
    assert late.status == MatchStatus.COLLISION, (late.status, late.distance)
    assert Path(late.existing.path) == sc.lib_late
    assert late.quality_label == "Keep Existing"

    shifted = m["shifted.mp3"]
    assert shifted.status == MatchStatus.COLLISION, (shifted.status, shifted.distance)
    assert Path(shifted.existing.path) == sc.lib_song_a
    assert 0.05 < shifted.distance < 0.1

    new = m["brand_new.flac"]
    assert new.status == MatchStatus.NEW, (new.status, new.distance)

    assert library_sync.summarize_matches(result["match_objects"]) == {
        "incoming": 4, "new": 1, "matched": 3, "upgrades": 1, "low_confidence": 0,
    }


def _audio(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    }


def test_scan_classifies_every_incoming_track(scenario, tmp_path):
    result = _compare(scenario, tmp_path / "fp.db")
    assert len(result["existing"]) == 73
    _assert_classification(result, scenario)
    # Every incoming track was decided from a handful of exact comparisons,
    # not by comparing it with the whole library.
    assert max(result["profile"]["shortlist_sizes"]) <= 5


def test_duplicates_are_not_transferred_at_execute(scenario, tmp_path):
    result = _compare(scenario, tmp_path / "fp.db")
    _assert_classification(result, scenario)
    plan = library_sync.compute_library_sync_plan(
        str(scenario.library), str(scenario.incoming), transfer_mode="move",
        match_results=result["match_objects"],
    )
    decisions = {Path(i.source).name: i.decision for i in plan.items}
    assert decisions == {
        "song_a.flac": ExecutionDecision.REVIEW_REQUIRED,      # possible upgrade: flag Replace to use it
        "late.mp3": ExecutionDecision.SKIP_DUPLICATE,
        "shifted.mp3": ExecutionDecision.SKIP_DUPLICATE,
        "brand_new.flac": ExecutionDecision.COPY,
    }

    library_before = _audio(scenario.library)
    new_bytes = scenario.brand_new.read_bytes()
    library_sync.execute_plan(plan, create_playlist=False)

    library_after = _audio(scenario.library)
    added = {k: v for k, v in library_after.items() if k not in library_before}
    assert list(added.values()) == [new_bytes], sorted(added)
    assert {k: library_after[k] for k in library_before} == library_before   # nothing replaced
    # The duplicates stay where they were; only the new track moved.
    for path in (scenario.song_a_flac, scenario.late_copy, scenario.shifted):
        assert path.exists(), path
    assert not scenario.brand_new.exists()


def test_old_cache_with_blob_fingerprints_still_matches(scenario, tmp_path):
    """Older caches hold pyacoustid's ``bytes`` (a BLOB): reading one crashed
    ``TrackRecord.signature()``; the records must carry text."""
    db = tmp_path / "fp.db"
    _compare(scenario, db)                       # fills the cache
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT path, fingerprint FROM fingerprints").fetchall()
        conn.executemany(
            "UPDATE fingerprints SET fingerprint=? WHERE path=?",
            [(sqlite3.Binary(fp.encode("ascii")), path) for path, fp in rows],
        )
        assert conn.execute("SELECT typeof(fingerprint) FROM fingerprints LIMIT 1").fetchone() == ("blob",)

    result = _compare(scenario, db)
    assert result["profile"]["cache_hits"] == 77 and result["profile"]["fingerprints_computed"] == 0
    for rec in result["library_record_objects"] + result["incoming_record_objects"]:
        assert isinstance(rec.fingerprint, str)
        assert rec.signature()
    json.dumps(result["library_records"])        # reports serialise the records
    _assert_classification(result, scenario)


def test_recompute_reuses_the_index_and_keeps_the_classification(scenario, tmp_path, monkeypatch):
    result = _compare(scenario, tmp_path / "fp.db")
    decoded = []
    real = library_sync._print_info
    monkeypatch.setattr(library_sync, "_print_info", lambda fp: decoded.append(fp) or real(fp))

    again = library_sync.rematch_scan_result(result, thresholds={"default": 0.2})
    assert again["candidate_index"] is result["candidate_index"]
    assert len(decoded) == 4                     # the incoming prints only, not the library's 73
    _assert_classification(again, scenario)
