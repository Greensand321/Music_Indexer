"""Library Sync execute, second review round (area: libreview).

Every transfer step is reversible (a failed Replace on Windows used to strand
the incoming file inside the library), nothing partial ever sits under a real
track name, a plan whose files changed is not executed as if nothing
happened, Windows paths keep their letter case, a new track whose name is
taken by a different song is added under a ``(n)`` name, an incoming
``Music/`` folder is not nested, and every run keeps its own audit.

fpcalc is not available in CI, so fingerprints are faked from each file's
title tag (same song = exact match, different songs never match).
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

import library_sync
from library_factory import TrackSpec, build_library
from library_sync import ExecutionDecision
from library_sync_review_state import ReviewStateStore
from utils.audio_metadata_reader import read_tags

D = ExecutionDecision


def _fp(title: str) -> str:
    seed = int(hashlib.sha1(title.encode()).hexdigest()[:6], 16)
    return ",".join(str(seed + i) for i in range(16))


@pytest.fixture(autouse=True)
def fake_fp(monkeypatch):
    def fake(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kw):
        return [(p, 1, _fp(str(read_tags(p).get("title") or Path(p).stem))) for p in paths]

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", fake)


NORTHBOUND = dict(artist="Aurora Lines", title="Northbound", album="Dawn Signals", track=2,
                  year="2019", genre="Electronic")
OLD_REL = "Aurora Lines/Dawn Signals/02 - Northbound.mp3"


def _tree(root: Path) -> dict[str, str]:
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            out[p.relative_to(root).as_posix()] = hashlib.sha1(p.read_bytes()).hexdigest()
    return out


def _music(tree: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in tree.items() if not k.startswith(("Docs/", "Playlists/"))}


def _scan(lib: Path, inc: Path, tmp_path: Path):
    return library_sync.compare_libraries(
        str(lib), str(inc), str(tmp_path / "fp.db"), include_match_objects=True
    )["match_objects"]


def _plan(lib, inc, tmp_path, *, mode="move", replace=(), copy=()):
    matches = _scan(lib, inc, tmp_path)
    store = ReviewStateStore()
    for m in matches:
        name = Path(m.incoming.path).name
        if name in replace:
            store.flag_for_replace(m)
        if name in copy:
            store.flag_for_copy(m.incoming.track_id)
    return library_sync.compute_library_sync_plan(
        str(lib), str(inc), transfer_mode=mode, match_results=matches, review_flags=store,
    )


@pytest.fixture
def same_ext_replace(tmp_path):
    """Library MP3 + an incoming MP3 of the same song flagged Replace (an
    in-place swap of the library file)."""
    lib, inc = tmp_path / "lib", tmp_path / "inc"
    build_library(lib, [TrackSpec(**NORTHBOUND, ext=".mp3", freq=349)])
    build_library(inc, [TrackSpec(**NORTHBOUND, ext=".mp3", freq=351, seconds=2.0,
                                  relpath="drop/northbound.mp3")])
    return lib, inc, lib / OLD_REL, inc / "drop" / "northbound.mp3"


@pytest.fixture
def upgrade_replace(tmp_path):
    """Library MP3 + an incoming FLAC of the same song flagged Replace (the
    FLAC takes the MP3's slot; the MP3 is retired to the backups)."""
    lib, inc = tmp_path / "lib", tmp_path / "inc"
    build_library(lib, [TrackSpec(**NORTHBOUND, ext=".mp3", freq=349)])
    build_library(inc, [TrackSpec(**NORTHBOUND, ext=".flac", freq=349,
                                  relpath="drop/northbound.flac")])
    return lib, inc, lib / OLD_REL, inc / "drop" / "northbound.flac"


@pytest.fixture
def new_track(tmp_path):
    lib, inc = tmp_path / "lib", tmp_path / "inc"
    build_library(lib, [TrackSpec(**NORTHBOUND, ext=".mp3", freq=349)])
    build_library(inc, [TrackSpec("Zephyr Quay", "Harbour Lights", "Moorings", 1, "2024", "Pop",
                                  ext=".flac", freq=700, seconds=3, relpath="drop/new.flac")])
    return lib, inc, inc / "drop" / "new.flac"


def _no_temp_files(root: Path) -> bool:
    return not [p for p in root.rglob("*") if ".alphadex-" in p.name]


# ── 1. A failed Replace is undone ────────────────────────────────────────────

def _lock(monkeypatch, locked: Path, fn_name: str = "replace"):
    """Make ``os.<fn_name>`` fail with a Windows sharing violation whenever it
    targets (as source or destination) ``locked`` — a file open in a player."""
    real = getattr(os, fn_name)
    calls = []

    def fake(a, b, *args, **kw):
        calls.append((a, b))
        if os.path.abspath(a) == str(locked) or os.path.abspath(b) == str(locked):
            raise PermissionError(13, "The process cannot access the file because it is being "
                                      "used by another process", str(locked))
        return real(a, b, *args, **kw)

    monkeypatch.setattr(library_sync.os, fn_name, fake)
    return calls


@pytest.mark.parametrize("mode", ["move", "copy"])
def test_locked_replace_target_leaves_everything_as_it_was(same_ext_replace, tmp_path, monkeypatch, mode):
    lib, inc, old, incoming = same_ext_replace
    plan = _plan(lib, inc, tmp_path, mode=mode, replace={"northbound.mp3"})
    assert [i.decision for i in plan.items] == [D.REPLACE]
    lib_before, inc_before = _music(_tree(lib)), _tree(inc)

    _lock(monkeypatch, old)
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)

    assert _music(_tree(lib)) == lib_before, "the library file must be untouched"
    assert _tree(inc) == inc_before, "the incoming file must be back where it was"
    assert _no_temp_files(lib)
    assert summary["replaced"] == 0 and summary["moved"] == 0 and summary["copied"] == 0
    assert summary["backups"] == [] and summary["leftovers"] == []
    [error] = summary["errors"]
    assert "could not be replaced" in error and "library file was not changed" in error
    if mode == "move":
        assert f"put back at {incoming}" in error
    audit = json.loads(Path(summary["audit_path"]).read_text())
    [item] = audit["items"]
    assert item["status"] == "failed" and "leftover_path" not in item


def test_backup_is_made_before_the_library_file_is_displaced(same_ext_replace, tmp_path, monkeypatch):
    lib, inc, old, incoming = same_ext_replace
    old_bytes, new_bytes = old.read_bytes(), incoming.read_bytes()
    plan = _plan(lib, inc, tmp_path, replace={"northbound.mp3"})
    real = os.replace
    seen = []

    def spy(a, b, *args, **kw):
        if os.path.abspath(b) == str(old):
            backups = list((lib / "Docs" / "Backups" / "LibrarySync").rglob("*.mp3"))
            seen.append([p.read_bytes() for p in backups])
        return real(a, b, *args, **kw)

    monkeypatch.setattr(library_sync.os, "replace", spy)
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert seen == [[old_bytes]], "a complete backup must exist before the swap"
    assert summary["replaced"] == 1 and summary["moved"] == 1 and summary["errors"] == []
    assert old.read_bytes() == new_bytes and not incoming.exists()
    assert [Path(b).read_bytes() for b in summary["backups"]] == [old_bytes]


def test_library_file_lost_by_a_failed_replace_is_restored_from_backup(same_ext_replace, tmp_path, monkeypatch):
    lib, inc, old, incoming = same_ext_replace
    old_bytes = old.read_bytes()
    plan = _plan(lib, inc, tmp_path, mode="copy", replace={"northbound.mp3"})
    real = os.replace

    def broken(a, b, *args, **kw):
        if os.path.abspath(b) == str(old):
            os.unlink(b)            # e.g. a network share that drops the target first
            raise OSError(errno.EIO, "I/O error", b)
        return real(a, b, *args, **kw)

    monkeypatch.setattr(library_sync.os, "replace", broken)
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert old.read_bytes() == old_bytes
    assert "restored from its backup" in summary["errors"][0]
    assert summary["replaced"] == 0 and _no_temp_files(lib)
    assert [Path(b).read_bytes() for b in summary["backups"]] == [old_bytes]   # kept


def test_failed_staging_copy_keeps_the_library_file_and_no_spare_backup(same_ext_replace, tmp_path, monkeypatch):
    lib, inc, old, incoming = same_ext_replace
    plan = _plan(lib, inc, tmp_path, mode="copy", replace={"northbound.mp3"})
    lib_before = _music(_tree(lib))
    real = shutil.copyfileobj

    def fail_for_incoming(fsrc, fdst, length=0):
        if os.path.abspath(fsrc.name) == str(incoming):
            fdst.write(fsrc.read(100))
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(fsrc, fdst, length)

    monkeypatch.setattr(library_sync.shutil, "copyfileobj", fail_for_incoming)
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert _music(_tree(lib)) == lib_before and _no_temp_files(lib)
    assert summary["replaced"] == 0 and summary["backups"] == []
    assert not list((lib / "Docs" / "Backups").rglob("*.mp3")), "the unused backup is removed"
    assert "partial copy was removed" in summary["errors"][0]


@pytest.mark.parametrize("mode", ["move", "copy"])
def test_upgrade_is_undone_when_the_old_file_cannot_be_retired(upgrade_replace, tmp_path, monkeypatch, mode):
    lib, inc, old, incoming = upgrade_replace
    plan = _plan(lib, inc, tmp_path, mode=mode, replace={"northbound.flac"})
    [item] = plan.items
    assert item.decision == D.REPLACE and item.destination.endswith("02 - Northbound.flac")
    lib_before, inc_before = _music(_tree(lib)), _tree(inc)

    _lock(monkeypatch, old, "rename")
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)

    assert _music(_tree(lib)) == lib_before, "no FLAC next to the MP3, MP3 untouched"
    assert _tree(inc) == inc_before
    assert summary["replaced"] == 0 and summary["backups"] == []
    assert "replace was undone" in summary["errors"][0]


# ── 2. Nothing partial under a real name ─────────────────────────────────────

def _half_copy(monkeypatch):
    def half(fsrc, fdst, length=0):
        data = fsrc.read()
        fdst.write(data[: len(data) // 2])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(library_sync.shutil, "copyfileobj", half)


def test_copy_failing_half_way_leaves_no_partial_file(new_track, tmp_path, monkeypatch):
    lib, inc, incoming = new_track
    plan = _plan(lib, inc, tmp_path, mode="copy")
    [item] = plan.items
    assert item.decision == D.COPY
    lib_before, inc_before = _music(_tree(lib)), _tree(inc)

    _half_copy(monkeypatch)
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)

    assert _music(_tree(lib)) == lib_before
    assert not Path(item.destination).exists() and _no_temp_files(lib)
    assert not Path(item.destination).parent.exists(), "the folder made for it is removed again"
    assert _tree(inc) == inc_before
    assert summary["copied"] == 0 and "partial copy was removed" in summary["errors"][0]


def test_move_to_another_drive_copies_first_and_deletes_the_source_last(new_track, tmp_path, monkeypatch):
    lib, inc, incoming = new_track
    data = incoming.read_bytes()
    plan = _plan(lib, inc, tmp_path, mode="move")
    [item] = plan.items
    monkeypatch.setattr(library_sync, "_same_device", lambda *_a: False)

    _half_copy(monkeypatch)
    failed = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert incoming.read_bytes() == data, "the original survives a failed cross-drive move"
    assert not Path(item.destination).exists() and _no_temp_files(lib)
    assert failed["moved"] == 0 and failed["errors"]

    monkeypatch.undo()
    monkeypatch.setattr(library_sync, "_same_device", lambda *_a: False)
    seen_source = []
    real_rename = library_sync._rename_no_clobber

    def spy(a, b):
        seen_source.append(incoming.exists())
        real_rename(a, b)

    monkeypatch.setattr(library_sync, "_rename_no_clobber", spy)
    ok = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert seen_source == [True], "the source still exists when the copy is renamed into place"
    assert ok["moved"] == 1 and ok["errors"] == []
    assert Path(item.destination).read_bytes() == data and not incoming.exists()


def test_cross_drive_move_whose_source_cannot_be_deleted_is_reported_as_a_copy(new_track, tmp_path, monkeypatch):
    lib, inc, incoming = new_track
    plan = _plan(lib, inc, tmp_path, mode="move")
    [item] = plan.items
    monkeypatch.setattr(library_sync, "_same_device", lambda *_a: False)
    real_unlink = os.unlink

    def locked(path, *a, **k):
        if os.path.abspath(path) == str(incoming):
            raise PermissionError(13, "in use", path)
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(library_sync.os, "unlink", locked)
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert Path(item.destination).exists() and incoming.exists()
    assert summary["moved"] == 0 and summary["copied"] == 1
    assert "original could not be removed" in summary["errors"][0]


# ── 3. The library changed after Build Plan ──────────────────────────────────

def test_replace_whose_target_moved_is_not_turned_into_a_copy(upgrade_replace, tmp_path):
    lib, inc, old, incoming = upgrade_replace
    plan = _plan(lib, inc, tmp_path, replace={"northbound.flac"})
    # The Indexer reorganises the library before Execute.
    moved = lib / "By Year" / "2019" / "Aurora Lines_02_Northbound.mp3"
    moved.parent.mkdir(parents=True)
    os.rename(old, moved)
    before = _music(_tree(lib))

    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)

    assert summary["replaced"] == 0 and summary["moved"] == 0 and summary["copied"] == 0
    assert _music(_tree(lib)) == before and incoming.exists()
    [stale] = summary["stale"]
    assert "library changed since the plan was built" in stale["reason"] and "is gone" in stale["reason"]
    assert summary["errors"], "a stale item is a problem the user must see"
    audit = json.loads(Path(summary["audit_path"]).read_text())
    assert audit["items"][0]["status"] == "skipped" and audit["items"][0]["stale"] is True


def test_replace_whose_target_was_edited_is_skipped(same_ext_replace, tmp_path):
    lib, inc, old, incoming = same_ext_replace
    plan = _plan(lib, inc, tmp_path, replace={"northbound.mp3"})
    old.write_bytes(old.read_bytes() + b"retagged")
    edited = old.read_bytes()
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert old.read_bytes() == edited and incoming.exists()
    assert summary["replaced"] == 0 and "changed" in summary["stale"][0]["reason"]


def test_incoming_file_changed_after_the_plan_is_skipped(new_track, tmp_path):
    lib, inc, incoming = new_track
    plan = _plan(lib, inc, tmp_path)
    incoming.write_bytes(incoming.read_bytes() + b"x")
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert summary["moved"] == 0 and "incoming file new.flac changed" in summary["stale"][0]["reason"]
    assert not Path(plan.items[0].destination).exists()


def test_unchanged_plan_is_not_stale(upgrade_replace, tmp_path):
    lib, inc, old, incoming = upgrade_replace
    plan = _plan(lib, inc, tmp_path, replace={"northbound.flac"})
    assert set(plan.snapshot) == {str(incoming), str(old), str(old.with_suffix(".flac"))}
    assert plan.snapshot[str(old.with_suffix(".flac"))] is None      # must not exist yet
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert summary["stale"] == [] and summary["errors"] == [] and summary["replaced"] == 1


# ── 4. Windows: paths keep their letter case ─────────────────────────────────

def test_windows_normcase_never_reaches_the_file_system(tmp_path, monkeypatch):
    lib, inc = tmp_path / "Lib", tmp_path / "Inc"
    build_library(lib, [TrackSpec(**NORTHBOUND, ext=".mp3", freq=349)])
    build_library(inc, [
        TrackSpec("Zephyr Quay", "Harbour Lights", "Moorings", 1, "2024", "Pop", ext=".flac",
                  freq=700, relpath="Drop/New.flac"),
        TrackSpec(**NORTHBOUND, ext=".flac", freq=349, relpath="Drop/Northbound.flac"),
    ])
    matches = _scan(lib, inc, tmp_path)
    store = ReviewStateStore()
    store.flag_for_replace(next(m for m in matches if m.incoming.path.endswith("Northbound.flac")))
    # Emulate Windows: normcase lower-cases (separators untouched so it runs here).
    monkeypatch.setattr(os.path, "normcase", lambda p: p.lower())
    plan = library_sync.compute_library_sync_plan(
        str(lib), str(inc), transfer_mode="move", match_results=matches, review_flags=store,
    )
    by_name = {Path(i.source).name: i for i in plan.items}
    assert set(by_name) == {"New.flac", "Northbound.flac"}, "sources keep their case"
    assert by_name["New.flac"].decision == D.COPY
    assert by_name["New.flac"].destination == str(lib / "By Year" / "2024" / "Zephyr Quay_01_Harbour Lights.flac")
    assert by_name["Northbound.flac"].decision == D.REPLACE
    assert by_name["Northbound.flac"].destination == str(lib / "Aurora Lines" / "Dawn Signals" / "02 - Northbound.flac")
    assert plan.library_root == str(lib) and plan.incoming_root == str(inc)
    assert all(t == str(lib / OLD_REL) for t in plan.replacement_targets.values())

    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert summary["errors"] == [] and summary["moved"] == 2 and summary["replaced"] == 1
    names = set(_music(_tree(lib)))
    assert names == {"By Year/2024/Zephyr Quay_01_Harbour Lights.flac",
                     "Aurora Lines/Dawn Signals/02 - Northbound.flac"}
    backups = [Path(b) for b in summary["backups"]]
    assert [b.name for b in backups] == ["02 - Northbound.mp3"]
    assert "Backups/LibrarySync" in backups[0].as_posix() and "Aurora Lines/Dawn Signals" in backups[0].as_posix()


# ── 5b. A different song already has the routed name ─────────────────────────

def test_new_track_whose_name_is_taken_by_a_different_song_is_added_as_n(tmp_path):
    lib, inc = tmp_path / "lib", tmp_path / "inc"
    build_library(lib, [TrackSpec(**NORTHBOUND, ext=".mp3", freq=349)])
    taken = lib / "By Year" / "2020" / "Kilo Mode_01_Live Wire.mp3"
    build_library(lib, [TrackSpec("Kilo Mode", "Live Wire (Other Recording)", "X", 1, "2020", "Rock",
                                  ext=".mp3", freq=1200, relpath=taken.relative_to(lib).as_posix())])
    build_library(inc, [TrackSpec("Kilo Mode", "Live Wire", "Alive", 1, "2020", "Rock", ext=".mp3",
                                  freq=900, relpath="drop/live.mp3")])
    taken_bytes = taken.read_bytes()
    plan = _plan(lib, inc, tmp_path, mode="copy")
    [item] = plan.items
    assert item.decision == D.COPY, "a new song is never skipped as 'already in the library'"
    assert item.destination == str(taken.with_name("Kilo Mode_01_Live Wire (1).mp3"))
    assert "already taken by a different file" in item.reason
    assert plan.counts()["copy"] == 1 and plan.counts()["skip"] == 0

    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert summary["copied"] == 1 and taken.read_bytes() == taken_bytes
    assert Path(item.destination).read_bytes() == (inc / "drop" / "live.mp3").read_bytes()


def test_identical_file_at_the_routed_name_is_still_a_duplicate(new_track, tmp_path):
    lib, inc, incoming = new_track
    first = _plan(lib, inc, tmp_path, mode="copy")
    dst = Path(first.items[0].destination)
    dst.parent.mkdir(parents=True)
    shutil.copyfile(incoming, dst)
    # Same bytes at the destination, but (with fingerprints faked by title)
    # also matched; force the "new" path by clearing the library match.
    matches = _scan(lib, inc, tmp_path)
    for m in matches:
        m.existing = None
        m.status = library_sync.MatchStatus.NEW
    plan = library_sync.compute_library_sync_plan(str(lib), str(inc), transfer_mode="copy",
                                                  match_results=matches, review_flags=ReviewStateStore())
    assert [i.decision for i in plan.items] == [D.SKIP_DUPLICATE]


# ── 5c. An incoming Music/ folder ────────────────────────────────────────────

@pytest.mark.parametrize("library_has_music", [True, False])
def test_incoming_music_folder_is_not_nested(tmp_path, library_has_music):
    lib, inc = tmp_path / "lib", tmp_path / "inc"
    rel = "Music/By Year/2019/Aurora Lines_02_Northbound.mp3" if library_has_music else OLD_REL
    build_library(lib, [TrackSpec(**NORTHBOUND, ext=".mp3", freq=349, relpath=rel)])
    build_library(inc, [TrackSpec("Zephyr Quay", "Harbour Lights", "Moorings", 1, "2024", "Pop",
                                  ext=".flac", freq=700, relpath="Music/new.flac")])
    plan = _plan(lib, inc, tmp_path)
    [item] = plan.items
    base = lib / "Music" if library_has_music else lib
    assert item.destination == str(base / "By Year" / "2024" / "Zephyr Quay_01_Harbour Lights.flac")
    assert "Music" + os.sep + "Music" not in item.destination


# ── 5f. One audit per run ────────────────────────────────────────────────────

def test_every_run_keeps_its_own_audit_and_report(tmp_path):
    lib, inc = tmp_path / "lib", tmp_path / "inc"
    (lib / "Music").mkdir(parents=True)
    inc.mkdir()
    dst = lib / "Music" / "a.mp3"
    runs = []
    for version in ("v1", "v2"):
        dst.write_text(f"library {version}")
        src = inc / "a.mp3"
        src.write_text(f"incoming {version}")
        plan = library_sync.LibrarySyncPlan(
            library_root=str(lib), incoming_root=str(inc), destination_root=str(lib / "Music"),
            moves={str(src): str(dst)}, tag_index={}, decision_log=[], allowed_replacements={str(dst)},
        )
        runs.append(library_sync.execute_plan(plan, create_playlist=False))
    first, second = runs
    assert first["run_id"] != second["run_id"]
    assert first["audit_path"] != second["audit_path"] and first["report_path"] != second["report_path"]
    for summary, version in zip(runs, ("v1", "v2")):
        audit = json.loads(Path(summary["audit_path"]).read_text())
        assert audit["run_id"] == summary["run_id"]
        assert audit["backup_folder"] == summary["backup_folder"]
        assert Path(audit["backup_folder"]).name == summary["run_id"]
        [backup] = summary["backups"]
        assert Path(backup).is_relative_to(audit["backup_folder"])
        assert Path(backup).read_text() == f"library {version}"
        assert Path(summary["report_path"]).exists()
    docs = lib / "Docs"
    assert (docs / "LibrarySyncAudit.json").read_text() == Path(second["audit_path"]).read_text()
    assert (docs / "LibrarySyncExecuted.html").read_text() == Path(second["report_path"]).read_text()
    assert second["latest_audit_path"] == str(docs / "LibrarySyncAudit.json")
    assert len(list(docs.glob("LibrarySyncAudit_*.json"))) == 2
