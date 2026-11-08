"""File Cleanup artwork sidecars and damaged undo logs; validator inputs.

* Renaming ``Song (1).mp3`` to ``Song.mp3`` left its ``Song (1).mp3.artwork``
  behind, so the cleaned track lost its cover (the readers look for
  ``Song.mp3.artwork``). The sidecar is now its own row of the plan, renamed
  right after (and only if) its track, logged, and undone with it.
* A damaged cleanup log (not UTF-8, non-object lines, records without
  names) raised out of "Undo Last Cleanup".
* ``validate_soundvault_structure`` raised on ``None`` and judged ``""``
  against the working directory.
"""
from __future__ import annotations

import json
import os

import file_cleanup as fc
from utils.audio_metadata_reader import read_sidecar_artwork_bytes
from validator import validate_soundvault_structure


def _write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_artwork_sidecar_follows_its_track_and_undo_restores_both(tmp_path):
    lib = tmp_path / "lib"
    _write(lib / "A" / "Song (1).mp3", b"audio")
    _write(lib / "A" / "Song (1).mp3.artwork", b"cover")

    plan = fc.build_cleanup_plan(str(lib))
    rows = [(os.path.basename(i.src), os.path.basename(i.dst), i.status) for i in plan.items]
    assert rows == [
        ("Song (1).mp3", "Song.mp3", "rename"),
        ("Song (1).mp3.artwork", "Song.mp3.artwork", "rename"),
    ]

    result = fc.execute_cleanup_plan(plan)
    assert result.failed == []
    assert read_sidecar_artwork_bytes(str(lib / "A" / "Song.mp3")) == b"cover"
    records = [json.loads(line) for line in open(result.log_path, encoding="utf-8")]
    sidecar = [r for r in records if r.get("new", "").endswith(".artwork")]
    assert sidecar and sidecar[0]["sidecar_of"] == str(lib / "A" / "Song (1).mp3")

    undo = fc.plan_undo(result.log_path)
    assert [os.path.basename(i.src) for i in undo.items] == ["Song.mp3", "Song.mp3.artwork"]
    fc.execute_undo(undo)
    assert (lib / "A" / "Song (1).mp3").read_bytes() == b"audio"
    assert (lib / "A" / "Song (1).mp3.artwork").read_bytes() == b"cover"


def test_sidecar_is_not_renamed_when_its_track_is_not(tmp_path):
    lib = tmp_path / "lib"
    _write(lib / "A" / "Song (1).mp3")
    _write(lib / "A" / "Song (1).mp3.jpg", b"cover")
    plan = fc.build_cleanup_plan(str(lib))
    # "Song.mp3" appears after the preview: the track is skipped, so the
    # cover must not move onto the other file's name.
    _write(lib / "A" / "Song.mp3", b"someone else")

    result = fc.execute_cleanup_plan(plan)

    assert (lib / "A" / "Song (1).mp3.jpg").exists()
    assert not (lib / "A" / "Song.mp3.jpg").exists()
    assert dict(result.failed)[str(lib / "A" / "Song (1).mp3.jpg")] == "skipped: its track was not renamed"


def test_sidecar_with_a_taken_name_is_a_collision_row(tmp_path):
    lib = tmp_path / "lib"
    _write(lib / "A" / "Song (1).mp3")
    _write(lib / "A" / "Song (1).mp3.artwork", b"new cover")
    _write(lib / "A" / "Song.mp3.artwork", b"stale cover")
    plan = fc.build_cleanup_plan(str(lib))
    side = [i for i in plan.items if i.src.endswith(".artwork")]
    assert len(side) == 1 and side[0].status == fc.STATUS_COLLISION
    fc.execute_cleanup_plan(plan)
    assert (lib / "A" / "Song.mp3.artwork").read_bytes() == b"stale cover"


def test_a_damaged_cleanup_log_never_breaks_undo(tmp_path):
    lib = tmp_path / "lib"
    _write(lib / "A" / "Song.mp3")
    docs = lib / "Docs"
    docs.mkdir()
    good = {"type": "rename", "old": str(lib / "A" / "Song (1).mp3"), "new": str(lib / "A" / "Song.mp3")}
    lines = [
        json.dumps({"type": "header", "library_root": str(lib)}).encode(),
        b"[1, 2, 3]",
        b"42",
        json.dumps({"type": "rename", "old": "only-old"}).encode(),
        json.dumps(good).encode(),
        b'{"type": "rename", "old": "caf\xe9"',  # cut short, and not UTF-8
    ]
    log = docs / "file_cleanup_20260101_000000.jsonl"
    log.write_bytes(b"\n".join(lines) + b"\n")
    _write(docs / "file_cleanup_20250101_000000.jsonl", b"\xff\xfe\x00garbage")

    assert fc.latest_undoable_log(str(lib)) == str(log)
    plan = fc.plan_undo(str(log))
    assert [(i.src, i.dst) for i in plan.renames] == [(good["new"], good["old"])]


def test_validator_handles_empty_missing_and_none(tmp_path, monkeypatch):
    (tmp_path / "By Artist").mkdir()
    monkeypatch.chdir(tmp_path)
    # "" used to be checked against the working directory (and pass here).
    for value in ("", None, "   "):
        valid, errors = validate_soundvault_structure(value)
        assert valid is False and errors == ["No library folder was selected."]
    valid, errors = validate_soundvault_structure(str(tmp_path / "missing"))
    assert valid is False and "does not exist" in errors[0]
    assert validate_soundvault_structure(str(tmp_path)) == (True, [])
