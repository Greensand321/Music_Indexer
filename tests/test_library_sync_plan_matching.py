"""Library Sync backend: the reviewed matches + flags *are* the plan, and
execute runs exactly that plan (area: library).

Covers LIB-03 (Copy flag lost at execute), LIB-04 (Replace had no effect),
LIB-05 (duplicates copied in), LIB-06 (reserved folders scanned as
"existing"), LIB-15 (missing / nested folders), LIB-19 (HTML report without
flags), LIB-23 (flags dropped on re-scan) and the backup-overwrite and
re-match (LIB-17 Recompute) paths. fpcalc is not available, so fingerprints
are faked from each file's title tag.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import library_sync
from library_factory import TrackSpec, build_library
from library_sync_review_state import ReviewStateStore
from library_sync_types import MatchStatus
from utils.audio_metadata_reader import read_tags


def _fp(title: str) -> str:
    seed = int(hashlib.sha1(title.encode()).hexdigest()[:6], 16)
    return ",".join(str(seed + i) for i in range(16))


@pytest.fixture
def fake_fp(monkeypatch):
    calls = []

    def fake(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kw):
        calls.append(list(paths))
        return [(p, 1, _fp(str(read_tags(p).get("title") or Path(p).stem))) for p in paths]

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", fake)
    return calls


@pytest.fixture
def sync_env(synthetic_library, tmp_path, fake_fp):
    lib = synthetic_library.root
    incoming = tmp_path / "incoming"
    inc = build_library(incoming, [
        TrackSpec("Zephyr Quay", "Harbour Lights", "Moorings", 1, "2024", "Pop", ext=".flac",
                  freq=700, relpath="drop/new_track.flac"),
        TrackSpec("Echo Park", "Loose Wire", None, None, None, None, ext=".mp3", freq=622,
                  relpath="drop/loose_wire_copy.mp3"),
        TrackSpec("Aurora Lines", "Northbound", "Dawn Signals", 2, "2019", "Electronic",
                  ext=".flac", freq=349, relpath="drop/northbound.flac"),
    ])
    paths = {spec.title: p for p, spec in inc.tracks.items()}
    # A quarantined copy of the *new* track: if Quarantine/ were scanned the
    # new track would wrongly match it (LIB-06).
    (lib / "Quarantine").mkdir()
    shutil.copyfile(paths["Harbour Lights"], lib / "Quarantine" / "loser.flac")
    return lib, incoming, paths


def _compare(lib, incoming, tmp_path):
    return library_sync.compare_libraries(
        str(lib), str(incoming), str(tmp_path / "fp.db"), include_match_objects=True
    )


def _by_title(result):
    return {read_tags(m.incoming.path).get("title"): m for m in result["match_objects"]}


def test_reserved_folders_are_not_existing_tracks(sync_env, tmp_path):
    lib, incoming, _paths = sync_env
    result = _compare(lib, incoming, tmp_path)
    assert not any("Quarantine" in p for p in result["existing"])
    matches = _by_title(result)
    assert matches["Harbour Lights"].status == MatchStatus.NEW
    assert matches["Loose Wire"].status == MatchStatus.EXACT_MATCH
    assert matches["Northbound"].quality_label == "Potential Upgrade"


def test_plan_follows_matches_and_flags(sync_env, tmp_path):
    lib, incoming, paths = sync_env
    result = _compare(lib, incoming, tmp_path)
    matches = _by_title(result)
    flags = ReviewStateStore()
    flags.flag_for_copy(matches["Harbour Lights"].incoming.track_id)
    flags.flag_for_replace(matches["Northbound"])

    plan = library_sync.compute_library_sync_plan(
        str(lib), str(incoming), transfer_mode="move",
        match_results=result["match_objects"], review_flags=flags,
    )
    items = {Path(i.source).name: i for i in plan.items}
    new, dupe, up = items["new_track.flac"], items["loose_wire_copy.mp3"], items["northbound.flac"]

    assert new.decision == library_sync.ExecutionDecision.COPY and new.action == "copy"   # LIB-03
    assert dupe.decision == library_sync.ExecutionDecision.SKIP_DUPLICATE                 # LIB-05
    assert up.decision == library_sync.ExecutionDecision.REPLACE                          # LIB-04
    old_mp3 = library_sync._normalize_path(str(lib / "Aurora Lines" / "Dawn Signals" / "02 - Northbound.mp3"))
    assert up.destination == old_mp3[:-4] + ".flac"
    assert plan.replacement_targets[up.source] == old_mp3
    assert plan.copy_only == {new.source}
    assert plan.allowed_replacements == {up.destination}
    assert plan.counts() == {"total": 3, "copy": 1, "move": 0, "replace": 1, "skip": 1, "review": 0}


def test_unflagged_upgrade_and_low_confidence_need_review(sync_env, tmp_path):
    lib, incoming, _paths = sync_env
    result = _compare(lib, incoming, tmp_path)
    plan = library_sync.compute_library_sync_plan(
        str(lib), str(incoming), match_results=result["match_objects"], review_flags=ReviewStateStore(),
    )
    items = {Path(i.source).name: i for i in plan.items}
    assert items["northbound.flac"].decision == library_sync.ExecutionDecision.REVIEW_REQUIRED
    assert "flag Replace" in items["northbound.flac"].reason
    assert items["new_track.flac"].decision == library_sync.ExecutionDecision.COPY
    assert items["new_track.flac"].action == "move"


def test_execute_runs_the_previewed_items(sync_env, tmp_path):
    lib, incoming, paths = sync_env
    result = _compare(lib, incoming, tmp_path)
    matches = _by_title(result)
    flags = ReviewStateStore()
    flags.flag_for_copy(matches["Harbour Lights"].incoming.track_id)
    flags.flag_for_replace(matches["Northbound"])
    plan = library_sync.build_library_sync_preview(
        str(lib), str(incoming), str(tmp_path / "preview.html"), transfer_mode="move",
        match_results=result["match_objects"], review_flags=flags,
    )
    html = (tmp_path / "preview.html").read_text(encoding="utf-8")
    assert "move — originals are moved" in html and "Flagged Replace" in html

    old_mp3 = lib / "Aurora Lines" / "Dawn Signals" / "02 - Northbound.mp3"
    old_bytes = old_mp3.read_bytes()
    up_bytes = paths["Northbound"].read_bytes()
    new_bytes = paths["Harbour Lights"].read_bytes()
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)

    assert summary["errors"] == [] and not summary["cancelled"]
    assert summary["copied"] == 1 and summary["moved"] == 1 and summary["replaced"] == 1
    # Copy flag honoured in move mode: the original stays (LIB-03).
    assert paths["Harbour Lights"].read_bytes() == new_bytes
    new_dst = Path(next(i.destination for i in plan.items if i.source.endswith("new_track.flac")))
    assert new_dst.read_bytes() == new_bytes
    # The duplicate was not copied in (LIB-05).
    assert paths["Loose Wire"].exists()
    assert not any(p.name.startswith("Echo Park_") for p in lib.rglob("*.mp3") if "By Year" in str(p))
    # Replace swapped the FLAC into the MP3's slot; the MP3 went to the backups.
    assert not old_mp3.exists()
    assert old_mp3.with_suffix(".flac").read_bytes() == up_bytes
    assert not paths["Northbound"].exists()          # move mode
    backups = [Path(b) for b in summary["backups"]]
    assert any(b.read_bytes() == old_bytes for b in backups)
    assert all("Docs/Backups/LibrarySync/" in b.as_posix() for b in backups)
    audit = json.loads(Path(summary["audit_path"]).read_text())
    assert {i["decision"] for i in audit["items"]} == {"COPY", "SKIP_DUPLICATE", "REPLACE"}
    assert audit["plan"]["transfer_mode"] == "move"


def test_execute_never_overwrites_a_file_that_appeared_after_the_preview(sync_env, tmp_path):
    lib, incoming, paths = sync_env
    result = _compare(lib, incoming, tmp_path)
    plan = library_sync.compute_library_sync_plan(
        str(lib), str(incoming), transfer_mode="copy",
        match_results=result["match_objects"], review_flags=ReviewStateStore(),
    )
    new_item = next(i for i in plan.items if i.source.endswith("new_track.flac"))
    Path(new_item.destination).parent.mkdir(parents=True, exist_ok=True)
    Path(new_item.destination).write_bytes(b"arrived meanwhile")
    summary = library_sync.execute_library_sync_plan(plan, create_playlist=False)
    assert Path(new_item.destination).read_bytes() == b"arrived meanwhile"
    assert summary["copied"] == 0
    assert any("replacement not allowed" in s["reason"] for s in summary["skipped"])


def test_previewed_plan_ignores_allow_replacements(tmp_path):
    lib = tmp_path / "lib"
    inc = tmp_path / "inc"
    (lib / "Music").mkdir(parents=True)
    inc.mkdir()
    src = inc / "a.mp3"
    src.write_text("new")
    dst = lib / "Music" / "a.mp3"
    dst.write_text("old")
    plan = library_sync.LibrarySyncPlan(
        library_root=str(lib), incoming_root=str(inc), destination_root=str(lib / "Music"),
        moves={str(src): str(dst)}, tag_index={}, decision_log=[],
    )
    plan.items = library_sync._compute_plan_items(plan.moves, destination_root=str(lib / "Music"))
    library_sync.execute_plan(plan, allow_replacements=True, create_playlist=False)
    assert dst.read_text() == "old", "a previewed SKIP must not turn into a replace at execute"


def test_backups_of_separate_runs_never_overwrite_each_other(tmp_path):
    lib = tmp_path / "lib"
    inc = tmp_path / "inc"
    (lib / "Music").mkdir(parents=True)
    inc.mkdir()
    dst = lib / "Music" / "a.mp3"
    backups = []
    for version in ("v1", "v2", "v3"):
        dst.write_text(f"library {version}")
        src = inc / "a.mp3"
        src.write_text(f"incoming {version}")
        plan = library_sync.LibrarySyncPlan(
            library_root=str(lib), incoming_root=str(inc), destination_root=str(lib / "Music"),
            moves={str(src): str(dst)}, tag_index={}, decision_log=[],
            allowed_replacements={str(dst)},
        )
        summary = library_sync.execute_plan(plan, create_playlist=False)
        backups.extend(summary["backups"])
    contents = sorted(Path(b).read_text() for b in backups)
    assert contents == ["library v1", "library v2", "library v3"]


@pytest.mark.parametrize("case", ["missing_library", "missing_incoming", "same", "nested"])
def test_folder_validation(tmp_path, case):
    lib = tmp_path / "lib"
    lib.mkdir()
    inc = {"missing_library": tmp_path / "inc", "missing_incoming": tmp_path / "nope",
           "same": lib, "nested": lib / "Incoming"}[case]
    if case in ("missing_library", "nested"):
        inc.mkdir()
    library = tmp_path / "typo" if case == "missing_library" else lib
    with pytest.raises(ValueError):
        library_sync.compare_libraries(str(library), str(inc), str(tmp_path / "fp.db"))
    assert not (tmp_path / "typo").exists()


def test_rematch_reuses_fingerprints(sync_env, tmp_path, fake_fp):
    lib, incoming, _paths = sync_env
    result = _compare(lib, incoming, tmp_path)
    fingerprinted = sum(len(c) for c in fake_fp)
    again = library_sync.rematch_scan_result(result, thresholds={"default": 0.05})
    assert sum(len(c) for c in fake_fp) == fingerprinted, "re-match must not fingerprint again"
    assert len(again["match_objects"]) == len(result["match_objects"])
    assert {m.status for m in again["match_objects"]} == {m.status for m in result["match_objects"]}
    # Also from the plain-dict form of a result.
    plain = {k: v for k, v in result.items() if not k.endswith("_record_objects")}
    assert len(library_sync.rematch_scan_result(plain)["match_objects"]) == 3


def test_flags_carry_over_a_rescan(sync_env, tmp_path):
    lib, incoming, paths = sync_env
    result = _compare(lib, incoming, tmp_path)
    m = _by_title(result)
    store = ReviewStateStore()
    store.flag_for_copy(m["Harbour Lights"].incoming.track_id)
    store.flag_for_replace(m["Northbound"])
    store.set_note(m["Loose Wire"].incoming.track_id, "check this")
    assert store.carry_over(_compare(lib, incoming, tmp_path)["match_objects"]) == []
    assert store.is_copy_flagged(m["Harbour Lights"].incoming.track_id)
    assert store.replace_target(m["Northbound"].incoming.track_id)

    paths["Harbour Lights"].unlink()
    (lib / "Aurora Lines" / "Dawn Signals" / "02 - Northbound.mp3").unlink()
    warnings = store.carry_over(_compare(lib, incoming, tmp_path)["match_objects"])
    assert len(warnings) == 2
    assert not store.flags.copy and not store.flags.replace
    assert store.note_for(m["Loose Wire"].incoming.track_id) == "check this"


def test_html_review_report_includes_flags_and_notes(sync_env, tmp_path):
    from library_sync_review_report import export_review_report_html

    lib, incoming, _paths = sync_env
    result = _compare(lib, incoming, tmp_path)
    m = _by_title(result)
    store = ReviewStateStore()
    store.flag_for_replace(m["Northbound"])
    store.set_note(m["Northbound"].incoming.track_id, "NOTE-XYZ")
    out = tmp_path / "report.html"
    export_review_report_html(
        str(out), result["match_objects"], library_root=str(lib), incoming_root=str(incoming),
        thresholds={"default": 0.3}, mixed_codec_threshold_boost=0.03, format_priority={},
        flags=store,
    )
    text = out.read_text()
    assert "NOTE-XYZ" in text
    assert "<tr><th>Flagged Replace</th><td>1</td></tr>" in text
    assert "<td>Replace</td>" in text
