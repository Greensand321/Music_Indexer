"""File Cleanup backend (TOOL-01 / TOOL-02): conservative rules, preview-first,
Docs/ action log, undo, reserved folders skipped."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import file_cleanup as fc


# ── TOOL-01: the patterns must not eat real titles ────────────────────────────

@pytest.mark.parametrize("stem", [
    "Carbon Copy",                 # capitalised word "Copy" is a real title
    "Xerocopy",                    # no separator before "copy"
    "Live at Wembley (1986)",      # a year, not a copy counter
    "Symphony No. 5 (1808)",
    "Song (100)",                  # counters stop at 99
    "01 - Copy",                   # track 01 titled "Copy"
    "(1)", "Copy", "copy",         # nothing would remain
    "Song(1)",                     # no space: not what OS file managers write
    "Requiem (Live)",
    "Copycat",
    "Hard Copy - Remix",
])
def test_real_titles_are_left_alone(stem):
    assert fc.clean_stem(stem) is None


@pytest.mark.parametrize("stem, expected", [
    ("Song (1)", "Song"),
    ("Song (12)", "Song"),
    ("Song copy", "Song"),            # macOS Finder
    ("Song copy 2", "Song"),
    ("Song - Copy", "Song"),          # Windows Explorer
    ("Song - Copy (3)", "Song"),
    ("03 - Northbound (1)", "03 - Northbound"),
])
def test_os_artifacts_are_stripped(stem, expected):
    assert fc.clean_stem(stem)[0] == expected


def _touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"audio")
    return p


# ── TOOL-02: plan first, nothing touched ──────────────────────────────────────

def test_plan_changes_nothing_and_skips_reserved_folders(tmp_path):
    lib = tmp_path / "Music"
    keep = [
        "A/B/Carbon Copy.mp3", "A/B/Live at Wembley (1986).mp3", "A/B/Xerocopy.mp3",
        "Quarantine/Loser (1).mp3", "Trash/Old copy.mp3", "Not Sorted/X (2).mp3",
        "Manual Review/Y (1).flac", "Docs/Z (1).mp3", "Playlists/P (1).mp3",
        "A/B/cover (1).jpg",
    ]
    for rel in keep:
        _touch(lib, rel)
    _touch(lib, "A/B/Song (1).mp3")
    before = sorted(p.relative_to(lib).as_posix() for p in lib.rglob("*") if p.is_file())

    plan = fc.build_cleanup_plan(str(lib))

    after = sorted(p.relative_to(lib).as_posix() for p in lib.rglob("*") if p.is_file())
    assert after == before, "building the plan must not touch the disk"
    assert [(Path(i.src).name, Path(i.dst).name) for i in plan.renames] == [("Song (1).mp3", "Song.mp3")]
    assert not list((lib / "Docs").glob("file_cleanup_*")), "no log until something executes"


def test_collisions_are_flagged_not_renamed(tmp_path):
    lib = tmp_path / "Music"
    _touch(lib, "A/Song.mp3")
    _touch(lib, "A/Song (1).mp3")          # target exists
    _touch(lib, "B/Tune (1).mp3")
    _touch(lib, "B/Tune (2).mp3")          # both would become Tune.mp3
    plan = fc.build_cleanup_plan(str(lib))
    assert plan.renames == []
    assert len(plan.collisions) == 3
    result = fc.execute_cleanup_plan(plan)
    assert result.renamed == {}
    assert (lib / "A/Song (1).mp3").exists() and (lib / "B/Tune (2).mp3").exists()


def test_title_tag_protects_a_real_suffix(tmp_path):
    lib = tmp_path / "Music"
    real = _touch(lib, "A/Love Song (2).mp3")      # title tag really is "Love Song (2)"
    dup = _touch(lib, "A/Other (1).mp3")           # title tag "Other": an OS artifact
    titles = {str(real): "Love Song (2)", str(dup): "Other"}
    plan = fc.build_cleanup_plan(str(lib), read_title=lambda p: titles.get(p))
    assert [Path(i.src).name for i in plan.renames] == ["Other (1).mp3"]
    assert plan.kept_by_title == 1


def test_execute_runs_exactly_the_plan_and_logs_for_undo(tmp_path):
    lib = tmp_path / "Music"
    _touch(lib, "A/One (1).mp3")
    _touch(lib, "A/Two copy.flac")
    plan = fc.build_cleanup_plan(str(lib))
    # A file that appears after the preview is NOT renamed: only the plan runs.
    _touch(lib, "A/Three (1).mp3")

    result = fc.execute_cleanup_plan(plan)

    assert sorted(Path(p).name for p in result.renamed.values()) == ["One.mp3", "Two.flac"]
    assert (lib / "A/Three (1).mp3").exists()
    log = Path(result.log_path)
    assert log.parent == lib / "Docs" and log.name.startswith("file_cleanup_")
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert records[0]["type"] == "header"
    assert {Path(r["old"]).name for r in records if r["type"] == "rename"} == {"One (1).mp3", "Two copy.flac"}

    restored, problems = fc.undo_cleanup(str(log))
    assert restored == 2 and problems == []
    assert (lib / "A/One (1).mp3").exists() and (lib / "A/Two copy.flac").exists()


def test_execute_never_overwrites_a_target_created_after_preview(tmp_path):
    lib = tmp_path / "Music"
    _touch(lib, "A/One (1).mp3")
    plan = fc.build_cleanup_plan(str(lib))
    late = lib / "A/One.mp3"
    late.write_bytes(b"precious")
    result = fc.execute_cleanup_plan(plan)
    assert result.renamed == {}
    assert late.read_bytes() == b"precious"
    assert (lib / "A/One (1).mp3").exists()


def test_execute_cancel_stops_between_renames(tmp_path):
    lib = tmp_path / "Music"
    for n in range(5):
        _touch(lib, f"A/T{n} (1).mp3")
    plan = fc.build_cleanup_plan(str(lib))
    stop = threading.Event()
    result = fc.execute_cleanup_plan(plan, cancel_event=stop, on_item=lambda *_: stop.set())
    assert result.cancelled
    assert len(result.renamed) == 1
    assert "cancelled" in Path(result.log_path).read_text(encoding="utf-8")


def test_plan_cancel(tmp_path):
    lib = tmp_path / "Music"
    _touch(lib, "A/T (1).mp3")
    stop = threading.Event()
    stop.set()
    plan = fc.build_cleanup_plan(str(lib), cancel_event=stop)
    assert plan.cancelled and plan.items == []


def test_real_tagged_titles_from_the_factory(tmp_path):
    """End to end with real tags: a track genuinely titled "Copy (2)" survives."""
    from library_factory import TrackSpec, build_library
    from utils.audio_metadata_reader import read_tags

    lib = build_library(tmp_path / "Music", [
        TrackSpec("Band", "Copy (2)", "LP", 1, relpath="Band/LP/Copy (2).mp3"),
        TrackSpec("Band", "Fault Line", "LP", 2, relpath="Band/LP/Fault Line (1).mp3"),
    ])
    plan = fc.build_cleanup_plan(str(lib.root), read_title=lambda p: read_tags(p).get("title"))
    assert [Path(i.src).name for i in plan.renames] == ["Fault Line (1).mp3"]
