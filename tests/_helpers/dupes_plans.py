"""Builders for hand-made consolidation plans in the duplicate-executor tests.

``group(winner, losers)`` makes a ``GroupPlan`` whose snapshot is the files'
real current state (so the executor's drift checks pass), ``plan(*groups)``
wraps them, and ``config(root)`` is an ``ExecutionConfig`` that writes its
reports under ``root/Docs`` and quarantines into ``root/Quarantine``.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, Mapping

from duplicate_consolidation import (
    ArtworkDirective,
    ConsolidationPlan,
    GroupPlan,
    PlaylistImpact,
    _capture_library_state,
)
from duplicate_consolidation_executor import ExecutionConfig


def snapshot(path: str | os.PathLike) -> dict:
    return dict(_capture_library_state(str(path)))


def group(
    winner: str | os.PathLike,
    losers: Iterable[str | os.PathLike],
    *,
    group_id: str | None = None,
    disposition: str = "quarantine",
    planned_tags: Mapping[str, object] | None = None,
    metadata_changes: Mapping[str, Mapping[str, object]] | None = None,
    artwork: Iterable[ArtworkDirective] = (),
    playlist_rewrites: Mapping[str, str] | None = None,
) -> GroupPlan:
    winner = str(winner)
    losers = [str(p) for p in losers]
    paths = [winner, *losers]
    return GroupPlan(
        group_id=group_id or f"g-{os.path.basename(winner)}",
        winner_path=winner,
        losers=losers,
        planned_winner_tags=dict(planned_tags or {}),
        winner_current_tags={},
        current_tags={p: {} for p in paths},
        metadata_changes={k: dict(v) for k, v in (metadata_changes or {}).items()},
        winner_quality={"reasons": ["test"]},
        artwork=list(artwork),
        artwork_candidates=[],
        chosen_artwork_source={},
        artwork_status="unchanged",
        artwork_variant_id=1,
        artwork_variant_total=1,
        artwork_variant_label="Single artwork variant",
        artwork_unknown_tracks=[],
        artwork_unknown_reasons={},
        loser_disposition={p: disposition for p in losers},
        playlist_rewrites=dict(playlist_rewrites or {}),
        playlist_impact=PlaylistImpact(playlists=0, entries=0),
        review_flags=[],
        context_summary={"album": [], "single": [], "unknown": [winner]},
        context_evidence={p: [] for p in paths},
        tag_source=winner,
        placeholders_present=False,
        tag_source_reason="winner",
        tag_source_evidence=[],
        track_quality={p: {} for p in paths},
        group_confidence="High",
        group_match_type="Exact",
        grouping_metadata_key=("artist", "title"),
        grouping_thresholds={"exact": 0.02, "near": 0.1},
        grouping_decisions=[],
        artwork_evidence=[],
        fingerprint_distances={},
        library_state={p: snapshot(p) for p in paths if os.path.lexists(p)},
    )


def plan(*groups: GroupPlan) -> ConsolidationPlan:
    return ConsolidationPlan(groups=list(groups))


def config(root: str | os.PathLike, **overrides) -> ExecutionConfig:
    root = Path(root)
    values = dict(
        library_root=str(root),
        reports_dir=str(root / "Docs" / "duplicate_execution_reports"),
        playlists_dir=str(root / "Playlists"),
        quarantine_dir=str(root / "Quarantine"),
        quarantine_flatten=True,
        apply_metadata=False,
        apply_artwork=False,
        cancel_event=threading.Event(),
        log_callback=lambda _m: None,
    )
    values.update(overrides)
    return ExecutionConfig(**values)


def write(path: Path, data: bytes = b"audio") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def cleanup_actions(result) -> dict:
    """``{loser: action}`` for every per-file loser_cleanup record."""
    return {a.target: a for a in result.actions if a.step == "loser_cleanup" and a.target != "execution"}


def quarantine_files(root: Path) -> list[Path]:
    q = Path(root) / "Quarantine"
    return sorted(p for p in q.rglob("*") if p.is_file()) if q.exists() else []
