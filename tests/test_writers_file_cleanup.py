"""File Cleanup regressions from the writers review (findings 6b, 6c).

6b. ``os.path.normcase`` doesn't fold case on macOS (or exFAT cards mounted
    anywhere), so two targets differing only in case were not flagged; and
    the "target exists?" check and ``os.rename`` had a gap in which a new
    file could be overwritten on POSIX.
6c. ``undo_cleanup`` existed but nothing could reach it: the backend now
    plans an undo (a preview) and executes exactly that plan, logging each
    restore into the original cleanup log.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

import file_cleanup as fc


def _touch(p: Path, data: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# ── 6b: case-insensitive collisions ──────────────────────────────────────


def test_two_targets_differing_only_in_case_are_collisions(tmp_path):
    _touch(tmp_path / "A" / "Song (1).mp3")
    _touch(tmp_path / "A" / "song (2).mp3")
    plan = fc.build_cleanup_plan(str(tmp_path))
    assert {os.path.basename(i.dst) for i in plan.items} == {"Song.mp3", "song.mp3"}
    assert plan.renames == []
    assert all(i.note == "several files would get this name" for i in plan.collisions)


def test_existing_file_differing_only_in_case_is_a_collision(tmp_path):
    _touch(tmp_path / "A" / "SONG.mp3")
    _touch(tmp_path / "A" / "Song (1).mp3")
    plan = fc.build_cleanup_plan(str(tmp_path))
    assert len(plan.items) == 1 and plan.items[0].status == fc.STATUS_COLLISION
    assert plan.items[0].note == "a file with the new name already exists"


# ── 6b: no check-then-rename gap ─────────────────────────────────────────


@pytest.mark.skipif(os.name == "nt", reason="POSIX rename semantics")
def test_target_created_after_the_check_is_never_overwritten(tmp_path, monkeypatch):
    src = _touch(tmp_path / "A" / "Song (1).mp3", b"copy")
    plan = fc.build_cleanup_plan(str(tmp_path))
    assert len(plan.renames) == 1
    dst = Path(plan.renames[0].dst)

    # Another program creates the target between the executor's check and
    # its rename: make the check miss it.
    real_lexists = os.path.lexists
    monkeypatch.setattr(fc.os.path, "lexists",
                        lambda p: False if os.fspath(p) == str(dst) else real_lexists(p))
    _touch(dst, b"new file from elsewhere")

    result = fc.execute_cleanup_plan(plan)

    assert dst.read_bytes() == b"new file from elsewhere"
    assert src.read_bytes() == b"copy"
    assert result.renamed == {}
    assert result.failed == [(str(src), "skipped: target appeared after the preview")]


def test_rename_without_hard_links_falls_back_to_a_rechecked_rename(tmp_path, monkeypatch):
    def no_links(*_a, **_k):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(fc.os, "link", no_links)
    src = _touch(tmp_path / "a (1).mp3", b"data")
    fc.rename_no_overwrite(str(src), str(tmp_path / "a.mp3"))
    assert (tmp_path / "a.mp3").read_bytes() == b"data" and not src.exists()

    other = _touch(tmp_path / "b (1).mp3", b"b")
    _touch(tmp_path / "b.mp3", b"keep")
    with pytest.raises(fc.TargetExists):
        fc.rename_no_overwrite(str(other), str(tmp_path / "b.mp3"))
    assert (tmp_path / "b.mp3").read_bytes() == b"keep" and other.exists()


def test_successful_rename_leaves_a_single_name(tmp_path):
    src = _touch(tmp_path / "x (1).flac", b"d")
    fc.rename_no_overwrite(str(src), str(tmp_path / "x.flac"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.flac"]
    assert os.stat(tmp_path / "x.flac").st_nlink == 1


# ── 6c: undo is planned, previewed, executed and logged ──────────────────


def _cleaned_library(tmp_path: Path):
    lib = tmp_path / "Music"
    _touch(lib / "Band" / "LP" / "Song (1).mp3")
    _touch(lib / "Band" / "LP" / "Other - Copy.mp3")
    plan = fc.build_cleanup_plan(str(lib))
    result = fc.execute_cleanup_plan(plan)
    assert len(result.renamed) == 2
    return lib, result


def test_undo_plan_previews_and_execute_restores_exactly_it(tmp_path):
    lib, cleanup = _cleaned_library(tmp_path)
    log = fc.latest_undoable_log(str(lib))
    assert log == cleanup.log_path

    plan = fc.plan_undo(log)
    assert isinstance(plan, fc.UndoPlan) and plan.log_path == log
    assert {(os.path.basename(i.src), os.path.basename(i.dst)) for i in plan.renames} == {
        ("Song.mp3", "Song (1).mp3"), ("Other.mp3", "Other - Copy.mp3"),
    }
    assert (lib / "Band" / "LP" / "Song.mp3").exists(), "planning must not rename"

    result = fc.execute_undo(plan)
    assert sorted(p.name for p in (lib / "Band" / "LP").iterdir()) == [
        "Other - Copy.mp3", "Song (1).mp3",
    ]
    # cleaned -> original, ready for update_playlists
    assert result.renamed == {i.src: i.dst for i in plan.renames}
    records = [json.loads(l) for l in Path(log).read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in records if r["type"] == "undo" and r["outcome"] == "restored") == 2
    # Nothing left to undo in that log.
    assert fc.latest_undoable_log(str(lib)) is None


def test_undo_skips_what_it_cannot_restore_and_retries_it_later(tmp_path):
    lib, _cleanup = _cleaned_library(tmp_path)
    log = fc.latest_undoable_log(str(lib))
    _touch(lib / "Band" / "LP" / "Song (1).mp3", b"a new file took the old name")

    plan = fc.plan_undo(log)
    assert [os.path.basename(i.dst) for i in plan.renames] == ["Other - Copy.mp3"]
    assert [(os.path.basename(i.dst), i.note) for i in plan.collisions] == [
        ("Song (1).mp3", "the original name is taken"),
    ]
    fc.execute_undo(plan)
    assert (lib / "Band" / "LP" / "Song (1).mp3").read_bytes() == b"a new file took the old name"
    assert (lib / "Band" / "LP" / "Song.mp3").exists()

    # Once the name is free again, the same log still offers the rest.
    os.remove(lib / "Band" / "LP" / "Song (1).mp3")
    assert fc.latest_undoable_log(str(lib)) == log
    again = fc.plan_undo(log)
    assert [os.path.basename(i.dst) for i in again.renames] == ["Song (1).mp3"]


def test_latest_undoable_log_walks_back_through_older_cleanups(tmp_path):
    lib = tmp_path / "Music"
    _touch(lib / "A" / "one (1).mp3")
    first = fc.execute_cleanup_plan(fc.build_cleanup_plan(str(lib)))
    _touch(lib / "A" / "two (1).mp3")
    second = fc.execute_cleanup_plan(fc.build_cleanup_plan(str(lib)))
    os.utime(first.log_path, (1_000_000, 1_000_000))
    assert fc.latest_undoable_log(str(lib)) == second.log_path
    fc.execute_undo(fc.plan_undo(second.log_path))
    assert fc.latest_undoable_log(str(lib)) == first.log_path


def test_no_cleanup_log_means_nothing_to_undo(tmp_path):
    assert fc.latest_undoable_log(str(tmp_path)) is None
