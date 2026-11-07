"""Indexer backend: one plan, previewed and executed verbatim (area: library).

Covers LIB-01 (non-audio sweep overwrote files), LIB-02 (reserved folders
scanned), LIB-13 (Manual Review routing dead), LIB-14 (execute recomputed the
plan), LIB-15 (missing library created), LIB-22 (no execution log), plus the
Opus-swept-to-Trash and re-run-adds-"(1)" regressions found while fixing them.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
from collections import Counter
from pathlib import Path

import pytest

import music_indexer_api as api


def _tree(root: Path) -> dict[str, str]:
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            out[p.relative_to(root).as_posix()] = hashlib.sha1(p.read_bytes()).hexdigest()
    return out


@pytest.fixture
def messy(synthetic_library):
    root = synthetic_library.root
    track = synthetic_library.find("First Light")
    for folder, name in (("Quarantine", "dupe_loser.mp3"), ("Trash", "old.mp3"),
                         ("Manual Review", "pending.mp3"), ("Not Sorted", "mine.mp3")):
        (root / folder).mkdir(exist_ok=True)
        shutil.copyfile(track, root / folder / name)
    (root / "Trash" / "cover.jpg").write_bytes(b"already-in-trash")
    (root / "Docs").mkdir(exist_ok=True)
    (root / "Docs" / "keep.txt").write_text("user doc")
    (root / "Basalt" / "Granite" / "cover.jpg").write_bytes(b"basalt-cover")
    return synthetic_library


def test_reserved_folders_are_never_planned(messy):
    plan = api.build_indexer_plan(str(messy.root))
    touched = list(plan.moves) + list(plan.leftovers)
    for path in touched:
        top = Path(path).relative_to(messy.root).parts[0]
        assert top not in ("Quarantine", "Trash", "Manual Review", "Not Sorted", "Docs", "Playlists")
    assert not any(Path(d).name in ("Quarantine", "Trash", "Docs") for d in plan.empty_dirs)


def test_untagged_file_goes_to_manual_review_and_named_file_is_parsed(messy, tmp_path):
    root = messy.root
    # Same audio as an untagged file, but with a parseable "Artist - Title" name.
    shutil.copyfile(root / "untitled_track.mp3", root / "Nova Drift - Static Bloom.mp3")
    plan = api.build_indexer_plan(str(root))
    review = {Path(p).name for p in plan.manual_review}
    assert review == {"untitled_track.mp3"}
    named_dst = plan.moves[str(root / "Nova Drift - Static Bloom.mp3")]
    assert Path(named_dst).name == "Nova Drift_00_Static Bloom.mp3"
    assert Path(plan.moves[str(root / "Echo Park - Loose Wire.mp3")]).name == "Echo Park_00_Loose Wire.mp3"


def test_preview_has_no_side_effects(synthetic_library):
    root = synthetic_library.root
    before = _tree(root)
    dirs_before = sorted(p for p, _d, _f in os.walk(root))
    html = root / "Docs" / "MusicIndex.html"
    result = api.run_full_indexer(str(root), str(html), dry_run_only=True)
    after = _tree(root)
    assert set(after) - set(before) == {"Docs/MusicIndex.html", "Docs/indexer_log.txt"}
    assert all(after[k] == v for k, v in before.items())
    new_dirs = sorted(set(p for p, _d, _f in os.walk(root)) - set(dirs_before))
    assert new_dirs == [str(root / "Docs")]          # no Manual Review/, Not Sorted/
    assert result["plan"].counts()["manual_review"] == 1


def test_leftovers_keep_structure_and_never_overwrite(messy):
    root = messy.root
    before = _tree(root)
    plan = api.build_indexer_plan(str(root))
    rel = {Path(s).relative_to(root).as_posix(): Path(d).relative_to(root).as_posix()
           for s, d in plan.leftovers.items()}
    assert rel["Aurora Lines/Dawn Signals/cover.jpg"] == "Trash/Aurora Lines/Dawn Signals/cover.jpg"
    assert rel["Basalt/Granite/cover.jpg"] == "Trash/Basalt/Granite/cover.jpg"
    assert rel["Basalt/Granite/notes.txt"] == "Docs/Basalt/Granite/notes.txt"
    assert not any(s.endswith(".opus") for s in rel), "Opus audio must never be swept"
    summary = api.apply_indexer_plan(plan, create_playlists=False)
    assert summary["errors"] == []
    after = _tree(root)
    assert after["Trash/cover.jpg"] == before["Trash/cover.jpg"]
    assert after["Trash/Basalt/Granite/cover.jpg"] == before["Basalt/Granite/cover.jpg"]
    assert after["Trash/Aurora Lines/Dawn Signals/cover.jpg"] == before["Aurora Lines/Dawn Signals/cover.jpg"]
    lost = Counter(v for k, v in before.items() if not k.startswith("Playlists/")) - Counter(after.values())
    assert not lost, "file contents disappeared"


def test_leftover_name_clash_gets_a_suffix(messy):
    root = messy.root
    (root / "Trash" / "Basalt" / "Granite").mkdir(parents=True)
    (root / "Trash" / "Basalt" / "Granite" / "cover.jpg").write_bytes(b"older trash")
    plan = api.build_indexer_plan(str(root))
    assert plan.leftovers[str(root / "Basalt" / "Granite" / "cover.jpg")].endswith("cover (1).jpg")


def test_execute_runs_the_previewed_plan_once(synthetic_library, monkeypatch):
    calls = []
    real = api.compute_moves_and_tag_index

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(api, "compute_moves_and_tag_index", spy)
    root = synthetic_library.root
    result = api.run_full_indexer(str(root), str(root / "Docs" / "MusicIndex.html"),
                                  dry_run_only=False, create_playlists=False)
    assert len(calls) == 1, "preview + execute must share one plan"
    plan = result["plan"]
    for src, dst in plan.file_moves.items():
        assert not os.path.exists(src) and os.path.exists(dst)


def test_stale_plan_is_refused_and_nothing_moves(synthetic_library):
    root = synthetic_library.root
    plan = api.build_indexer_plan(str(root))
    (root / "Echo Park - Loose Wire.mp3").write_bytes(b"edited after the preview")
    before = _tree(root)
    with pytest.raises(api.StalePlanError) as exc:
        api.apply_indexer_plan(plan, create_playlists=False)
    assert "modified: Echo Park - Loose Wire.mp3" in exc.value.problems
    assert _tree(root) == before


def test_new_audio_after_preview_makes_the_plan_stale(synthetic_library):
    root = synthetic_library.root
    plan = api.build_indexer_plan(str(root))
    shutil.copyfile(synthetic_library.find("Magma"), root / "late.flac")
    with pytest.raises(api.StalePlanError):
        api.apply_indexer_plan(plan, create_playlists=False)


def test_existing_destination_is_never_overwritten(messy):
    root = messy.root
    plan = api.build_indexer_plan(str(root))
    # Something lands in Trash/ (reserved, not part of the freshness check)
    # exactly where a leftover is planned to go.
    target = Path(plan.leftovers[str(root / "Basalt" / "Granite" / "notes.txt")])
    src_cover = root / "Aurora Lines" / "Dawn Signals" / "cover.jpg"
    clash = Path(plan.leftovers[str(src_cover)])
    clash.parent.mkdir(parents=True, exist_ok=True)
    clash.write_bytes(b"arrived meanwhile")
    summary = api.apply_indexer_plan(plan, create_playlists=False)
    assert clash.read_bytes() == b"arrived meanwhile"
    assert src_cover.exists(), "the clashing leftover stays where it was"
    assert any("not overwritten" in e for e in summary["errors"])
    assert target.exists()


def test_execute_writes_an_audit_of_every_action(messy):
    root = messy.root
    plan = api.build_indexer_plan(str(root))
    summary = api.apply_indexer_plan(plan, create_playlists=False)
    import json
    payload = json.loads(Path(summary["audit_path"]).read_text())
    kinds = Counter(a["action"] for a in payload["actions"] if a["status"] == "done")
    assert kinds["move"] == len(plan.file_moves) == summary["moved"]
    assert kinds["move_leftover"] == len(plan.leftovers)
    assert kinds["remove_empty_folder"] == len(summary["removed_dirs"]) > 0
    assert Path(summary["log_path"]).read_text().count("[done] move:") == summary["moved"]
    for removed in summary["removed_dirs"]:
        assert removed in plan.empty_dirs


def test_cancel_mid_execute_stops_and_logs(synthetic_library):
    root = synthetic_library.root
    plan = api.build_indexer_plan(str(root))
    event = threading.Event()
    seen = []

    def progress(current, total, path, phase):
        seen.append(path)
        if len(seen) == 2:
            event.set()

    summary = api.apply_indexer_plan(plan, progress_callback=progress, create_playlists=False,
                                     cancel_event=event)
    assert summary["cancelled"] and summary["moved"] == 2
    assert len(summary["removed_dirs"]) == 0
    assert "CANCELLED" in Path(summary["log_path"]).read_text()


def test_rerun_on_an_organised_library_changes_nothing(synthetic_library):
    root = synthetic_library.root
    api.run_full_indexer(str(root), str(root / "Docs" / "MusicIndex.html"), create_playlists=False)
    plan = api.build_indexer_plan(str(root))
    assert plan.file_moves == {}, f"re-run would rename: {list(plan.file_moves.values())[:3]}"
    assert plan.leftovers == {}


def test_missing_library_is_not_created(tmp_path):
    missing = tmp_path / "typo" / "library"
    with pytest.raises(FileNotFoundError):
        api.run_full_indexer(str(missing), str(missing / "Docs" / "MusicIndex.html"), dry_run_only=True)
    assert not missing.exists() and not missing.parent.exists()


def test_global_cancel_flag_still_honoured(synthetic_library):
    import indexer_control
    indexer_control.cancel_event.set()
    try:
        with pytest.raises(indexer_control.IndexCancelled):
            api.build_indexer_plan(str(synthetic_library.root))
    finally:
        indexer_control.cancel_event.clear()
