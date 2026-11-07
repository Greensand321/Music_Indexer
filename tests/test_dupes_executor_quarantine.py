"""Executor safety: quarantining losers must never overwrite another file.

The Qt Duplicates page runs the executor with ``quarantine_flatten=True``
(every loser lands directly in ``Quarantine/``). Losers that share a file name
— "01 - Intro.mp3" from two different albums, or a loser whose name matches a
file quarantined by an earlier run — used to be moved onto the same
destination, and ``shutil.move`` silently replaced the earlier file.
"""
from __future__ import annotations

import os
import threading

import pytest

from duplicate_consolidation import ConsolidationPlan, GroupPlan, PlaylistImpact
from duplicate_consolidation_executor import ExecutionConfig, execute_consolidation_plan


def _snapshot(path):
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "mtime": int(stat.st_mtime)}


def _group(group_id, winner, loser, disposition="quarantine"):
    return GroupPlan(
        group_id=group_id,
        winner_path=str(winner),
        losers=[str(loser)],
        planned_winner_tags={},
        winner_current_tags={},
        current_tags={str(winner): {}, str(loser): {}},
        metadata_changes={},
        winner_quality={"reasons": ["test"]},
        artwork=[],
        artwork_candidates=[],
        chosen_artwork_source={},
        artwork_status="unchanged",
        artwork_variant_id=1,
        artwork_variant_total=1,
        artwork_variant_label="Single artwork variant",
        artwork_unknown_tracks=[],
        artwork_unknown_reasons={},
        loser_disposition={str(loser): disposition},
        playlist_rewrites={},
        playlist_impact=PlaylistImpact(playlists=0, entries=0),
        review_flags=[],
        context_summary={"album": [], "single": [], "unknown": [str(winner)]},
        context_evidence={str(winner): [], str(loser): []},
        tag_source=str(winner),
        placeholders_present=False,
        tag_source_reason="winner",
        tag_source_evidence=[],
        track_quality={str(winner): {}, str(loser): {}},
        group_confidence="High",
        group_match_type="Exact",
        grouping_metadata_key=("artist", "title"),
        grouping_thresholds={"exact": 0.02, "near": 0.1},
        grouping_decisions=[],
        artwork_evidence=[],
        fingerprint_distances={},
        library_state={str(winner): _snapshot(winner), str(loser): _snapshot(loser)},
    )


def _library(tmp_path, albums=("AlbumA", "AlbumB", "AlbumC")):
    groups = []
    payloads = {}
    for idx, album in enumerate(albums):
        folder = tmp_path / "Artist" / album
        folder.mkdir(parents=True)
        winner = folder / "01 - Intro.flac"
        loser = folder / "01 - Intro.mp3"
        winner.write_bytes(f"winner {album}".encode())
        loser.write_bytes(f"loser {album}".encode())
        payloads[str(loser)] = loser.read_bytes()
        groups.append(_group(f"g{idx}", winner, loser))
    return ConsolidationPlan(groups=groups), payloads


def _config(tmp_path, **overrides):
    cfg = dict(
        library_root=str(tmp_path),
        reports_dir=str(tmp_path / "Docs" / "duplicate_execution_reports"),
        playlists_dir=str(tmp_path / "Playlists"),
        quarantine_dir=str(tmp_path / "Quarantine"),
        quarantine_flatten=True,
        apply_metadata=False,
        apply_artwork=False,
        cancel_event=threading.Event(),
        log_callback=lambda _m: None,
    )
    cfg.update(overrides)
    return ExecutionConfig(**cfg)


@pytest.mark.parametrize("flatten", [True, False])
def test_same_named_losers_never_overwrite_each_other(tmp_path, flatten):
    plan, payloads = _library(tmp_path)
    result = execute_consolidation_plan(plan, _config(tmp_path, quarantine_flatten=flatten))

    assert result.success, [a.detail for a in result.actions if a.status in {"failed", "blocked"}]
    quarantined = [p for p in (tmp_path / "Quarantine").rglob("*") if p.is_file()]
    assert len(quarantined) == len(payloads)
    # Every loser's bytes survive, and the report says where each one went.
    assert sorted(p.read_bytes() for p in quarantined) == sorted(payloads.values())
    for loser, payload in payloads.items():
        dest = result.quarantine_index[loser]
        assert os.path.isfile(dest) and open(dest, "rb").read() == payload


def test_existing_quarantine_file_is_not_replaced(tmp_path):
    plan, payloads = _library(tmp_path, albums=("AlbumA",))
    earlier = tmp_path / "Quarantine" / "01 - Intro.mp3"
    earlier.parent.mkdir(parents=True)
    earlier.write_bytes(b"quarantined by an earlier run")

    result = execute_consolidation_plan(plan, _config(tmp_path))

    assert result.success
    assert earlier.read_bytes() == b"quarantined by an earlier run"
    (loser,) = payloads
    dest = result.quarantine_index[loser]
    assert dest != str(earlier)
    assert open(dest, "rb").read() == payloads[loser]


def test_unique_destination_suffixes_before_extension(tmp_path):
    from duplicate_consolidation_executor import _unique_destination

    taken = {str(tmp_path / "a.mp3")}
    (tmp_path / "a (1).mp3").write_text("x")
    dest = _unique_destination(str(tmp_path / "a.mp3"), taken)
    assert dest == str(tmp_path / "a (2).mp3")
    assert dest in taken
