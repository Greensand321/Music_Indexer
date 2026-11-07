"""Executor step 6 (quarantine/delete losers): what runs is what the report says.

Before: every loser was submitted to a thread pool up front, and a cancel or a
failed move only stopped the loop that *recorded* results — the pool still
moved (or deleted) every remaining loser, none of them in the report or the
quarantine index, and the summary undercounted. Losers were also removed
without checking that their winner still existed (or was not the very same
file through a link), moves could overwrite a quarantined file whose name only
differed in case/Unicode form on Windows/macOS, and a cross-device move that
could not delete the original left a second copy in Quarantine/.
"""
from __future__ import annotations

import errno
import os
import unicodedata
from pathlib import Path

import pytest

import duplicate_consolidation_executor as ex
from duplicate_consolidation_executor import execute_consolidation_plan
from _helpers.dupes_plans import cleanup_actions, config, group, plan, quarantine_files, write


def _library(root: Path, count: int):
    groups, losers = [], []
    for idx in range(count):
        folder = root / "Artist" / f"Album{idx:02d}"
        winner = write(folder / "Song.flac", b"winner %d" % idx)
        loser = write(folder / "Song.mp3", b"loser %d" % idx)
        groups.append(group(winner, [loser], group_id=f"g{idx:02d}"))
        losers.append(loser)
    return plan(*groups), losers


def _assert_report_matches_disk(result, losers):
    """Every loser is either still in place, or gone with a verified record."""
    records = cleanup_actions(result)
    for loser in losers:
        record = records.get(str(loser))
        assert record is not None, f"no report entry for {loser}"
        if loser.exists():
            assert record.status != "success", (loser, record.detail)
            assert str(loser) not in result.quarantine_index or result.quarantine_index[str(loser)] == "retained"
        else:
            assert record.status == "success", (loser, record.status, record.detail)
            dest = result.quarantine_index[str(loser)]
            assert dest == "deleted" or os.path.isfile(dest)


def test_cancel_stops_further_moves_and_reports_each_one(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_default_max_workers", lambda: 1)
    the_plan, losers = _library(tmp_path, 6)
    cfg = config(tmp_path)
    real_replace = os.replace
    moves = []

    def replace_then_cancel(src, dst, *a, **kw):
        if str(src).endswith("Song.mp3"):
            moves.append(src)
            cfg.cancel_event.set()  # the user presses Cancel during the first move
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", replace_then_cancel)
    monkeypatch.setattr(os, "rename", replace_then_cancel)

    result = execute_consolidation_plan(the_plan, cfg)

    assert not result.success
    assert len(moves) == 1
    assert sum(1 for p in losers if not p.exists()) == 1
    _assert_report_matches_disk(result, losers)
    skipped = [a for a in cleanup_actions(result).values() if a.metadata.get("skip_reason") == "not_attempted"]
    assert len(skipped) == 5
    assert any(a.status == "cancelled" for a in result.actions)


def test_a_failed_move_stops_the_run_and_every_completed_move_is_reported(tmp_path, monkeypatch):
    the_plan, losers = _library(tmp_path, 12)
    locked = str(losers[3])
    real_replace, real_rename = os.replace, os.rename

    def locked_file(real):
        def inner(src, dst, *a, **kw):
            if os.fspath(src) == locked:
                raise PermissionError(errno.EACCES, "file is open in another program", src)
            return real(src, dst, *a, **kw)
        return inner

    monkeypatch.setattr(os, "replace", locked_file(real_replace))
    monkeypatch.setattr(os, "rename", locked_file(real_rename))

    result = execute_consolidation_plan(the_plan, config(tmp_path))

    assert not result.success
    assert Path(locked).read_bytes() == b"loser 3"
    _assert_report_matches_disk(result, losers)
    # No empty placeholder (or stray copy) was left behind for the failed move.
    moved = [p for p in losers if not p.exists()]
    assert len(quarantine_files(tmp_path)) == len(moved)
    assert all(p.stat().st_size > 0 for p in quarantine_files(tmp_path))
    assert cleanup_actions(result)[locked].status == "failed"


def test_loser_is_kept_when_its_winner_vanished_after_preflight(tmp_path, monkeypatch):
    winner = write(tmp_path / "A" / "Song.flac", b"winner")
    loser = write(tmp_path / "A" / "Song.mp3", b"loser")
    the_plan = plan(group(winner, [loser], disposition="delete"))
    real_capture = ex._capture_state_with_cache

    def capture_then_lose_winner(path):
        state = real_capture(path)
        if path == str(winner):
            os.remove(path)  # another program removes the winner right after the check
        return state

    monkeypatch.setattr(ex, "_capture_state_with_cache", capture_then_lose_winner)
    result = execute_consolidation_plan(
        the_plan, config(tmp_path, allow_deletion=True, confirm_deletion=True)
    )

    assert loser.read_bytes() == b"loser"
    record = cleanup_actions(result)[str(loser)]
    assert record.status == "failed"
    assert record.metadata["skip_reason"] == "winner_missing"
    assert not result.success


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
@pytest.mark.parametrize("disposition", ["quarantine", "delete"])
def test_winner_that_is_a_link_to_the_loser_never_loses_its_target(tmp_path, disposition):
    real = write(tmp_path / "Other" / "Song.flac", b"the only copy")
    alias = tmp_path / "Album" / "Song.flac"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(real)
    the_plan = plan(group(alias, [real], disposition=disposition))

    result = execute_consolidation_plan(
        the_plan, config(tmp_path, allow_deletion=True, confirm_deletion=True)
    )

    assert real.read_bytes() == b"the only copy"
    assert alias.read_bytes() == b"the only copy"
    assert cleanup_actions(result)[str(real)].metadata["skip_reason"] == "same_file"


def test_hard_linked_loser_is_left_alone(tmp_path):
    winner = write(tmp_path / "A" / "Song.flac", b"x")
    loser = tmp_path / "B" / "Song.flac"
    loser.parent.mkdir()
    try:
        os.link(winner, loser)
    except OSError:
        pytest.skip("hard links unsupported here")
    result = execute_consolidation_plan(plan(group(winner, [loser])), config(tmp_path))
    assert loser.exists()
    assert cleanup_actions(result)[str(loser)].metadata["skip_reason"] == "same_file"


def test_loser_changed_after_preflight_is_not_removed(tmp_path, monkeypatch):
    winner = write(tmp_path / "A" / "Song.flac", b"winner")
    loser = write(tmp_path / "A" / "Song.mp3", b"loser")
    the_plan = plan(group(winner, [loser]))
    real_capture = ex._capture_state_with_cache

    def capture_then_edit(path):
        state = real_capture(path)
        if path == str(loser):
            loser.write_bytes(b"retagged by a tag editor meanwhile")
        return state

    monkeypatch.setattr(ex, "_capture_state_with_cache", capture_then_edit)
    result = execute_consolidation_plan(the_plan, config(tmp_path))

    assert loser.read_bytes() == b"retagged by a tag editor meanwhile"
    assert cleanup_actions(result)[str(loser)].metadata["skip_reason"] == "changed_since_preview"


def test_overlapping_groups_are_refused_before_anything_moves(tmp_path):
    a = write(tmp_path / "a.flac")
    b = write(tmp_path / "b.mp3")
    c = write(tmp_path / "c.mp3")
    # b is a loser of g1 *and* the winner of g2: cleaning g1 would remove g2's winner.
    the_plan = plan(group(a, [b], group_id="g1"), group(b, [c], group_id="g2"))

    result = execute_consolidation_plan(the_plan, config(tmp_path))

    assert not result.success
    assert a.exists() and b.exists() and c.exists()
    assert any(x.step == "preflight" and x.status == "blocked" and "overlap" in x.detail for x in result.actions)


def test_case_and_unicode_variants_get_distinct_quarantine_names(tmp_path):
    nfc = unicodedata.normalize("NFC", "Café.mp3")
    nfd = unicodedata.normalize("NFD", "Café.mp3")
    names = [nfc, nfd, "CAFÉ.mp3"]
    groups, losers = [], []
    for idx, name in enumerate(names):
        folder = tmp_path / f"Album{idx}"
        winner = write(folder / "Cafe.flac", b"w%d" % idx)
        loser = write(folder / name, b"loser %d" % idx)
        groups.append(group(winner, [loser], group_id=f"g{idx}"))
        losers.append(loser)

    result = execute_consolidation_plan(plan(*groups), config(tmp_path))

    assert result.success, [a.detail for a in result.actions if a.status in {"failed", "blocked"}]
    dests = [result.quarantine_index[str(p)] for p in losers]
    # On Windows/macOS these three names are one file; they must never share a destination.
    folded = {unicodedata.normalize("NFC", os.path.basename(d)).casefold() for d in dests}
    assert len(folded) == 3, dests
    assert sorted(Path(d).read_bytes() for d in dests) == [b"loser 0", b"loser 1", b"loser 2"]


def test_destination_helpers_respect_case_insensitive_names(tmp_path):
    first = str(tmp_path / "Song.mp3")
    assert ex._unique_destination(str(tmp_path / "SONG.mp3"), {first}) == str(tmp_path / "SONG (1).mp3")
    taken: set[str] = set()
    assert ex._reserve_destination(first, taken) == first
    assert ex._reserve_destination(str(tmp_path / "song.MP3"), taken) == str(tmp_path / "song (1).MP3")
    assert os.path.getsize(first) == 0  # the claim is a real (empty) placeholder file


def test_cross_device_move_that_cannot_delete_the_original_leaves_one_copy(tmp_path, monkeypatch):
    winner = write(tmp_path / "A" / "Song.flac", b"winner")
    loser = write(tmp_path / "A" / "Song.mp3", b"loser bytes")
    real_replace, real_rename, real_remove, real_unlink = os.replace, os.rename, os.remove, os.unlink

    def cross_device(real):
        def inner(src, dst, *a, **kw):
            if os.fspath(src) == str(loser):
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real(src, dst, *a, **kw)
        return inner

    def locked(real):
        def inner(path, *a, **kw):
            if os.fspath(path) == str(loser):
                raise PermissionError(errno.EACCES, "in use", path)
            return real(path, *a, **kw)
        return inner

    monkeypatch.setattr(os, "replace", cross_device(real_replace))
    monkeypatch.setattr(os, "rename", cross_device(real_rename))
    monkeypatch.setattr(os, "remove", locked(real_remove))
    monkeypatch.setattr(os, "unlink", locked(real_unlink))

    result = execute_consolidation_plan(plan(group(winner, [loser])), config(tmp_path))

    assert loser.read_bytes() == b"loser bytes"
    assert quarantine_files(tmp_path) == []  # no second copy, no placeholder, no .part file
    record = cleanup_actions(result)[str(loser)]
    assert record.status == "failed"
    assert "could not be deleted" in record.detail
    assert not result.success


def test_cross_device_copy_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    winner = write(tmp_path / "A" / "Song.flac", b"winner")
    loser = write(tmp_path / "A" / "Song.mp3", b"loser bytes")
    real_replace = os.replace

    def cross_device(src, dst, *a, **kw):
        if os.fspath(src) == str(loser):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(src, dst, *a, **kw)

    def disk_full(src, dst, *a, **kw):
        with open(dst, "wb") as fh:
            fh.write(b"half")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "replace", cross_device)
    monkeypatch.setattr(ex.shutil, "copy2", disk_full)

    result = execute_consolidation_plan(plan(group(winner, [loser])), config(tmp_path))

    assert loser.read_bytes() == b"loser bytes"
    assert quarantine_files(tmp_path) == []
    assert cleanup_actions(result)[str(loser)].status == "failed"


def test_cross_device_move_succeeds(tmp_path, monkeypatch):
    winner = write(tmp_path / "A" / "Song.flac", b"winner")
    loser = write(tmp_path / "A" / "Song.mp3", b"loser bytes")
    real_replace = os.replace

    def cross_device(src, dst, *a, **kw):
        if os.fspath(src) == str(loser):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", cross_device)
    result = execute_consolidation_plan(plan(group(winner, [loser])), config(tmp_path))

    assert result.success
    assert not loser.exists()
    assert Path(result.quarantine_index[str(loser)]).read_bytes() == b"loser bytes"
    assert len(quarantine_files(tmp_path)) == 1


def test_metadata_failure_stops_further_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_default_max_workers", lambda: 1)
    if ex.MutagenFile is None:
        pytest.skip("mutagen unavailable")
    groups, losers = [], []
    for idx in range(4):
        winner = write(tmp_path / f"A{idx}" / "Song.flac", b"w")
        loser = write(tmp_path / f"A{idx}" / "Song.mp3", b"l")
        groups.append(group(
            winner, [loser], group_id=f"g{idx}",
            planned_tags={"album": "X"}, metadata_changes={"album": {"from": None, "to": "X"}},
        ))
        losers.append(loser)
    calls = []

    def failing_write(target, tags, *a, **kw):
        calls.append(target)
        return False, "Failed to save metadata tags.", []

    monkeypatch.setattr(ex, "_apply_metadata", failing_write)
    result = execute_consolidation_plan(plan(*groups), config(tmp_path, apply_metadata=True))

    assert len(calls) == 1  # was: every queued write still ran after the first failure
    assert all(p.exists() for p in losers)
    not_attempted = [a for a in result.actions if a.step == "metadata" and a.metadata.get("skip_reason") == "not_attempted"]
    assert len(not_attempted) == 3


def test_winner_rename_only_takes_back_a_removed_losers_name(tmp_path):
    folder = tmp_path / "Album"
    w1 = write(folder / "Song (1).flac", b"winner")
    l1 = write(folder / "Song.flac", b"loser")
    w2 = write(folder / "Live (1986).flac", b"winner 2")
    l2 = write(folder / "Live.mp3", b"loser 2")
    w3 = write(folder / "Intro (2).flac", b"winner 3")
    l3 = write(tmp_path / "Other" / "Intro (2).mp3", b"loser 3")

    result = execute_consolidation_plan(
        plan(group(w1, [l1], group_id="g1"), group(w2, [l2], group_id="g2"), group(w3, [l3], group_id="g3")),
        config(tmp_path),
    )

    assert result.success, [a.detail for a in result.actions if a.status in {"failed", "blocked"}]
    assert (folder / "Song.flac").read_bytes() == b"winner"  # the copy-number suffix went
    assert w2.exists() and not (folder / "Live.flac").exists()  # a year is not a copy number
    assert w3.exists() and not (folder / "Intro.flac").exists()  # no loser held "Intro.flac"


def test_winner_rename_is_skipped_when_the_loser_stays(tmp_path):
    folder = tmp_path / "Album"
    winner = write(folder / "Song (1).flac", b"winner")
    loser = write(folder / "Song.flac", b"loser")
    result = execute_consolidation_plan(plan(group(winner, [loser], disposition="retain")), config(tmp_path))
    assert winner.exists() and loser.read_bytes() == b"loser"
    assert not [a for a in result.actions if a.step == "filename_cleanup"]


def test_preview_shows_the_winner_rename(tmp_path):
    from duplicate_consolidation import export_consolidation_preview_html, planned_winner_rename

    folder = tmp_path / "Album"
    winner = write(folder / "Song (1).flac")
    loser = write(folder / "Song.flac")
    the_group = group(winner, [loser])
    assert planned_winner_rename(the_group) == str(loser)
    out = tmp_path / "preview.html"
    export_consolidation_preview_html(plan(the_group), str(out))
    assert "Then renamed to" in out.read_text(encoding="utf-8")
