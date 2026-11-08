"""Undo for Tag Fixer and Genre Normalizer writes (area: improve).

``controllers.tag_revert`` turns a ``Docs/tag_changes_<tool>_<ts>.jsonl``
change log back into a write, preview-first: list → plan → apply exactly the
plan, logged as a ``revert`` change log that can itself be reverted.

End to end on real files of every format (WAV with ID3 included): the
writes come from the real Tag Fixer apply path and the real
``apply_genre_changes``; multi-value tags are set and read back with mutagen
directly (``tests/_helpers/meta_tags.py``) so the app's own reader/writer
can't hide a bug.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from _helpers.meta_tags import ALL_FORMATS, build_genre_library, get_field, get_genres, needs_ffmpeg, set_genres
from controllers import tag_revert
from controllers.normalize_controller import GenreChange, apply_genre_changes
from library_factory import _real_mutagen
from plugins.base import SourceStatus
from utils.tag_change_log import TagChangeLog, read_change_log


class FakeSource:
    def __init__(self, answers, name="FakeID"):
        self.name = name
        self.answers = answers

    def preflight(self):
        return SourceStatus(self.name, True, "test", "")

    def identify(self, path):
        from utils.audio_metadata_reader import read_tags

        return dict(self.answers.get(read_tags(path).get("title"), {}))


def set_field(path: Path, field: str, values: list[str]) -> None:
    """Write ``values`` as a real multi-value text tag, with mutagen only."""
    m = _real_mutagen()
    ext = path.suffix.lower()
    if ext == ".mp3":
        audio = m["easyid3"].EasyID3(str(path))
        audio[field] = values
        audio.save(str(path))
    elif ext == ".m4a":
        key = {"artist": "\xa9ART", "title": "\xa9nam", "album": "\xa9alb"}[field]
        audio = m["mp4"].MP4(str(path))
        audio[key] = values
        audio.save()
    elif ext == ".wav":
        frame_id = {"artist": "TPE1", "title": "TIT2", "album": "TALB"}[field]
        audio = m["wave"].WAVE(str(path))
        if audio.tags is None:
            audio.add_tags()
        audio.tags.setall(frame_id, [getattr(m["id3"], frame_id)(encoding=3, text=values)])
        audio.save()
    else:
        cls = {".flac": m["flac"].FLAC, ".ogg": m["oggvorbis"].OggVorbis,
               ".opus": m["oggopus"].OggOpus}[ext]
        audio = cls(str(path))
        audio[field] = values
        audio.save()


def _records(tag_fixer, root):
    db = os.path.join(str(root), "Docs", ".soundvault.db")
    tag_fixer.init_db(db)
    conn = sqlite3.connect(db)
    try:
        return db, tag_fixer.build_file_records(
            str(root), db_conn=conn, show_all=True, log_callback=lambda _m: None
        )
    finally:
        conn.close()


def _genre_apply(root, changes, tool="genre_normalize"):
    with TagChangeLog(str(root), tool) as log:
        result = apply_genre_changes(changes, change_log=log)
    assert not result.failed and not result.skipped, (result.failed, result.skipped)
    return log.path


def _write_entries(root, name, entries):
    docs = Path(root) / "Docs"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / name
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return str(path)


# ── Genre Normalizer: multi-value genres, every format ──────────────────────


@needs_ffmpeg
@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_genre_write_is_reverted_and_the_revert_can_be_reverted(tmp_path, ext):
    files = build_genre_library(tmp_path, [
        (f"multi{ext}", "Artist", "Multi", ["Rock", "Pop", "Indie"]),
        (f"empty{ext}", "Artist", "Empty", []),
    ])
    multi, empty = files[f"multi{ext}"], files[f"empty{ext}"]
    log_path = _genre_apply(tmp_path, [
        GenreChange(str(multi), ["Rock", "Pop", "Indie"], ["Rock"]),
        GenreChange(str(empty), [], ["Jazz", "Blues"]),
    ])
    assert get_genres(multi) == ["Rock"]
    assert get_genres(empty) == ["Jazz", "Blues"]

    plan = tag_revert.plan_revert(log_path)
    assert plan.counts()["files"] == 2 and not plan.conflicts and not plan.missing
    by_path = {item.path: item for item in plan.items}
    assert by_path[str(multi)].fields["genre"].restore == ["Rock", "Pop", "Indie"]
    assert by_path[str(empty)].fields["genre"].restore == []          # = remove the tag again
    # planning wrote nothing
    assert get_genres(multi) == ["Rock"]

    result = tag_revert.apply_revert(plan)
    assert sorted(result.reverted) == sorted([str(multi), str(empty)])
    assert get_genres(multi) == ["Rock", "Pop", "Indie"]              # every value, in order
    assert get_genres(empty) == []
    # the other tags are untouched
    assert get_field(multi, "artist") == ["Artist"] and get_field(multi, "title") == ["Multi"]

    entries = read_change_log(result.log_path)
    assert os.path.basename(result.log_path).startswith("tag_changes_revert_")
    assert {e["status"] for e in entries} == {"written"}
    assert {e["reverts"] for e in entries} == {os.path.basename(log_path)}

    # Revert the revert: back to what the normalizer wrote.
    again = tag_revert.apply_revert(tag_revert.plan_revert(result.log_path))
    assert len(again.reverted) == 2
    assert get_genres(multi) == ["Rock"]
    assert get_genres(empty) == ["Jazz", "Blues"]


# ── Tag Fixer: multi-value artist + appended genres, every format ───────────


@needs_ffmpeg
@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_tag_fixer_write_is_reverted(tmp_path, monkeypatch, ext):
    import tag_fixer
    from controllers import tagfix_controller

    files = build_genre_library(tmp_path, [(f"a{ext}", "Artist A", "Song", ["Rock", "Pop"])])
    path = files[f"a{ext}"]
    set_field(path, "artist", ["Artist A", "Artist B"])
    monkeypatch.setattr(tag_fixer, "PLUGINS", [FakeSource({
        "Song": {"artist": "New Artist", "title": "Song", "album": "New Album",
                 "genres": ["Jazz"], "score": 1.0},
    })])
    db, records = _records(tag_fixer, tmp_path)
    result = tagfix_controller.apply_selected_fields(
        [(records[0], ["artist", "album", "genres"])], records, db, log_callback=lambda _m: None,
    )
    assert result.applied == [str(path)]
    assert get_field(path, "artist") == ["New Artist"]
    assert get_genres(path) == ["Rock", "Pop", "Jazz"]

    [info] = [i for i in tag_revert.list_undoable_logs(str(tmp_path)) if i.tool == "tagfixer"]
    assert info.path == result.log_path and info.files == 1
    plan = tag_revert.plan_revert(info.path)
    assert set(plan.items[0].fields) == {"artist", "album", "genres"}
    done = tag_revert.apply_revert(plan)
    assert done.reverted == [str(path)]
    assert get_field(path, "artist") == ["Artist A", "Artist B"]
    assert get_field(path, "album") == ["Album"]
    assert get_genres(path) == ["Rock", "Pop"]
    assert get_field(path, "title") == ["Song"]

    infos = tag_revert.list_undoable_logs(str(tmp_path))
    assert infos[0].path == done.log_path and infos[0].reverts == os.path.basename(info.path)
    assert [i for i in infos if i.path == info.path][0].reverted_by == [done.log_path]


# ── Safety: conflicts, missing files, stale plans, already reverted ─────────


@needs_ffmpeg
def test_changed_and_moved_files_are_reported_and_left_alone(tmp_path):
    files = build_genre_library(tmp_path, [
        ("keep.flac", "A", "Keep", ["Rock", "Pop"]),
        ("edited.mp3", "A", "Edited", ["Rock", "Pop"]),
        ("moved.ogg", "A", "Moved", ["Rock", "Pop"]),
        ("wav.wav", "A", "Wav", ["Rock", "Pop"]),
    ])
    log_path = _genre_apply(tmp_path, [GenreChange(str(p), ["Rock", "Pop"], ["Rock"]) for p in files.values()])
    set_genres(files["edited.mp3"], ["Metal"])                    # someone changed it since
    moved_to = tmp_path / "elsewhere.ogg"
    os.replace(files["moved.ogg"], moved_to)

    plan = tag_revert.plan_revert(log_path)
    assert [i.path for i in plan.items] == [str(files["keep.flac"]), str(files["wav.wav"])]
    assert plan.missing == [str(files["moved.ogg"])]
    [conflict] = plan.conflicts
    assert conflict.path == str(files["edited.mp3"])
    assert conflict.fields == {"genre": (["Rock"], ["Metal"])}

    result = tag_revert.apply_revert(plan)
    assert get_genres(files["keep.flac"]) == ["Rock", "Pop"]
    assert get_genres(files["wav.wav"]) == ["Rock", "Pop"]
    assert get_genres(files["edited.mp3"]) == ["Metal"]            # never overwritten
    assert get_genres(moved_to) == ["Rock"]
    assert len(result.reverted) == 2

    # Planning again: the two reverted files are already reverted; nothing to write.
    second = tag_revert.plan_revert(log_path)
    assert sorted(second.already_reverted) == sorted([str(files["keep.flac"]), str(files["wav.wav"])])
    assert not second.items
    before = set((tmp_path / "Docs").iterdir())
    assert tag_revert.apply_revert(second).log_path == ""
    assert set((tmp_path / "Docs").iterdir()) == before           # no empty revert log


@needs_ffmpeg
def test_a_file_changed_between_preview_and_apply_is_skipped(tmp_path):
    files = build_genre_library(tmp_path, [("a.flac", "A", "A", ["Rock", "Pop"]), ("b.mp3", "A", "B", ["Rock", "Pop"])])
    log_path = _genre_apply(tmp_path, [GenreChange(str(p), ["Rock", "Pop"], ["Rock"]) for p in files.values()])
    plan = tag_revert.plan_revert(log_path)
    assert len(plan.items) == 2
    set_genres(files["b.mp3"], ["Rock", "Ska"])                    # after the revert preview
    result = tag_revert.apply_revert(plan)
    assert result.reverted == [str(files["a.flac"])]
    assert result.skipped == [(str(files["b.mp3"]), "genre changed since the preview")]
    assert get_genres(files["b.mp3"]) == ["Rock", "Ska"]
    statuses = {e["path"]: e["status"] for e in read_change_log(result.log_path)}
    assert statuses == {str(files["a.flac"]): "written", str(files["b.mp3"]): "skipped"}


def test_failed_write_is_reported_and_logged(tmp_path):
    target = tmp_path / "x.flac"
    target.write_bytes(b"not really audio")
    log_path = _write_entries(tmp_path, "tag_changes_genre_fill_20260101-000000.jsonl", [
        {"path": str(target), "status": "written", "changes": {"genre": {"before": [], "after": ["Jazz"]}}},
    ])
    reader = lambda path, fields: {"genre": ["Jazz"]}

    def boom(path, values):
        raise RuntimeError("disk full")

    plan = tag_revert.plan_revert(log_path, reader=reader)
    result = tag_revert.apply_revert(plan, reader=reader, writer=boom)
    assert result.failed == [(str(target), "disk full")]
    [entry] = read_change_log(result.log_path)
    assert entry["status"] == "failed" and entry["error"] == "disk full"


def test_unreadable_file_is_reported_not_raised(tmp_path):
    target = tmp_path / "broken.flac"
    target.write_bytes(b"garbage")
    log_path = _write_entries(tmp_path, "tag_changes_genre_fill_20260101-000000.jsonl", [
        {"path": str(target), "status": "written", "changes": {"genre": {"before": [], "after": ["Jazz"]}}},
    ])
    plan = tag_revert.plan_revert(log_path)
    assert [p for p, _why in plan.unreadable] == [str(target)] and not plan.items


# ── Listing and log parsing ──────────────────────────────────────────────────


def test_list_is_newest_first_and_skips_logs_that_wrote_nothing(tmp_path):
    f = str(tmp_path / "f.mp3")
    w = {"path": f, "status": "written", "changes": {"genre": {"before": ["A"], "after": ["B"]}}}
    s = {"path": f, "status": "skipped", "changes": {"genre": {"before": ["A"], "after": ["B"]}}}
    _write_entries(tmp_path, "tag_changes_genre_fill_20260102-090000.jsonl", [w, w])
    _write_entries(tmp_path, "tag_changes_tagfixer_20260103-090000.jsonl", [w])
    _write_entries(tmp_path, "tag_changes_tagfixer_20260103-090000_2.jsonl", [w, s])
    _write_entries(tmp_path, "tag_changes_genre_normalize_20260104-090000.jsonl", [s])   # wrote nothing
    (tmp_path / "Docs" / "unrelated.jsonl").write_text("{}\n")
    infos = tag_revert.list_undoable_logs(str(tmp_path))
    assert [i.name for i in infos] == [
        "tag_changes_tagfixer_20260103-090000_2.jsonl",
        "tag_changes_tagfixer_20260103-090000.jsonl",
        "tag_changes_genre_fill_20260102-090000.jsonl",
    ]
    assert [i.tool for i in infos] == ["tagfixer", "tagfixer", "genre_fill"]
    assert (infos[0].written, infos[0].skipped, infos[0].files) == (1, 1, 1)
    assert infos[2].written == 2 and infos[2].files == 1
    assert tag_revert.list_undoable_logs(str(tmp_path / "nowhere")) == []


def test_torn_last_line_and_repeated_writes(tmp_path):
    f = tmp_path / "f.mp3"
    f.write_bytes(b"x")
    entries = [
        {"path": str(f), "status": "written", "changes": {"genre": {"before": ["Orig"], "after": ["One"]}}},
        {"path": str(f), "status": "failed", "changes": {"genre": {"before": ["One"], "after": ["Nope"]}}},
        {"path": str(f), "status": "written", "changes": {"genre": {"before": ["One"], "after": ["Two"]}}},
    ]
    log_path = _write_entries(tmp_path, "tag_changes_genre_fill_20260101-000000.jsonl", entries)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write('{"path": "torn')                                 # crash mid-write
    plan = tag_revert.plan_revert(log_path, reader=lambda p, fields: {"genre": ["Two"]})
    assert plan.bad_lines == 1 and plan.not_written == 1
    [item] = plan.items
    assert item.fields["genre"].current == ["Two"]
    assert item.fields["genre"].restore == ["Orig"]                # before the *first* write


def test_cancel_stops_planning_and_applying(tmp_path):
    import threading

    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.mp3"
        p.write_bytes(b"x")
        paths.append(p)
    log_path = _write_entries(tmp_path, "tag_changes_genre_fill_20260101-000000.jsonl", [
        {"path": str(p), "status": "written", "changes": {"genre": {"before": [], "after": ["J"]}}} for p in paths
    ])
    reader = lambda p, fields: {"genre": ["J"]}
    cancel = threading.Event()
    plan = tag_revert.plan_revert(
        log_path, reader=reader, cancel_event=cancel,
        progress_callback=lambda i, _t: cancel.set() if i == 1 else None,
    )
    assert plan.cancelled and len(plan.items) == 1
    full = tag_revert.plan_revert(log_path, reader=reader)
    written = []
    cancel2 = threading.Event()
    result = tag_revert.apply_revert(
        full, reader=reader, writer=lambda p, v: written.append((p, v)), cancel_event=cancel2,
        progress_callback=lambda i, _t: cancel2.set() if i == 2 else None,
    )
    assert result.cancelled and len(written) == 2
    assert written[0] == (str(paths[0]), {"genre": None})          # [] before -> remove the field


# ── tag_writer: None removes a field ─────────────────────────────────────────


@needs_ffmpeg
@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_writer_none_removes_the_field(tmp_path, ext):
    from utils.tag_writer import write_fields

    path = build_genre_library(tmp_path, [(f"a{ext}", "Artist", "Song", ["Rock", "Pop"])])[f"a{ext}"]
    write_fields(str(path), {"genre": None, "album": None})
    assert get_genres(path) == []
    assert get_field(path, "album") == []
    assert get_field(path, "artist") == ["Artist"]
    write_fields(str(path), {"genre": None})                       # removing an absent field is fine
    assert get_genres(path) == []
