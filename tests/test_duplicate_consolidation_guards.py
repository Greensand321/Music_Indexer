import os
import threading

from duplicate_consolidation import PlaylistImpact, ConsolidationPlan, GroupPlan
from duplicate_consolidation_executor import ExecutionConfig, execute_consolidation_plan


def _make_group(tmp_path, *, review_flags=None, disposition="quarantine"):
    winner = tmp_path / "winner.flac"
    loser = tmp_path / "loser.mp3"
    winner.write_text("winner")
    loser.write_text("loser")
    return GroupPlan(
        group_id="g1",
        winner_path=str(winner),
        losers=[str(loser)],
        planned_winner_tags={"title": "Winner"},
        metadata_changes={"title": {"from": "old", "to": "Winner"}},
        winner_quality={"reasons": ["test"]},
        artwork=[],
        loser_disposition={str(loser): disposition},
        playlist_rewrites={str(loser): str(winner)},
        playlist_impact=PlaylistImpact(playlists=1, entries=1),
        review_flags=review_flags or [],
        context_summary={"album": [], "single": [], "unknown": [str(winner)]},
    )


def _config(tmp_path, **overrides):
    defaults = dict(
        library_root=str(tmp_path),
        reports_dir=str(tmp_path / "reports"),
        playlists_dir=str(tmp_path / "Playlists"),
        quarantine_dir=str(tmp_path / "Quarantine"),
        cancel_event=threading.Event(),
        log_callback=lambda _m: None,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def test_execution_blocks_review_required_groups(tmp_path):
    plan = ConsolidationPlan(groups=[_make_group(tmp_path, review_flags=["manual review required"])])
    result = execute_consolidation_plan(plan, _config(tmp_path))

    assert result.success is False
    assert any(action.status == "blocked" for action in result.actions)
    assert (tmp_path / "loser.mp3").exists()


def test_execution_enforces_operation_limit(tmp_path):
    plan = ConsolidationPlan(groups=[_make_group(tmp_path, review_flags=[])])
    cfg = _config(tmp_path, operation_limit=1, confirm_operation_overage=False, allow_review_required=True)

    result = execute_consolidation_plan(plan, cfg)

    assert result.success is False
    assert any(action.status == "blocked" and "Planned operations" in action.detail for action in result.actions)
    assert (tmp_path / "loser.mp3").exists()


def test_execution_requires_deletion_confirmation(tmp_path):
    plan = ConsolidationPlan(groups=[_make_group(tmp_path, disposition="delete", review_flags=[])])
    cfg = _config(tmp_path, allow_review_required=True)

    result = execute_consolidation_plan(plan, cfg)

    assert result.success is False
    assert (tmp_path / "loser.mp3").exists()
    assert any("Deletion requested" in action.detail for action in result.actions)


def test_dry_run_execute_skips_loser_cleanup(tmp_path):
    plan = ConsolidationPlan(groups=[_make_group(tmp_path, disposition="delete", review_flags=[])])
    cfg = _config(
        tmp_path,
        allow_review_required=True,
        allow_deletion=False,
        dry_run_execute=True,
    )

    result = execute_consolidation_plan(plan, cfg)

    assert result.success is True
    assert result.quarantine_index[str(tmp_path / "loser.mp3")] == "retained"
    assert (tmp_path / "loser.mp3").exists()
