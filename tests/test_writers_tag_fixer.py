"""Tag Fixer regressions from the writers review (findings 4 and 5).

4. ``read_tags`` returns only the first value of a field, and the Tag Fixer
   writes the whole field: a file tagged ``["Artist A", "Artist B"]`` was
   previewed as "Artist A → New Artist", lost "Artist B" on apply, and the
   change log recorded only "Artist A".
5. Genres were merged from every lookup result before the score check, so a
   low-confidence match's genres were written along with a confident match.

Real files of every format, multi-values set with mutagen directly.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from _helpers.meta_tags import ALL_FORMATS, build_genre_library, get_field, needs_ffmpeg
from library_factory import _real_mutagen
from plugins.base import SourceStatus


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


# ── Finding 4: every value of a multi-value field ─────────────────────────


@needs_ffmpeg
@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_multi_value_artist_is_previewed_logged_and_replaced_faithfully(tmp_path, monkeypatch, ext):
    import tag_fixer
    from controllers import tagfix_controller
    from utils.tag_change_log import read_change_log

    files = build_genre_library(tmp_path, [(f"a{ext}", "Artist A", "Song", [])])
    path = files[f"a{ext}"]
    set_field(path, "artist", ["Artist A", "Artist B"])
    assert get_field(path, "artist") == ["Artist A", "Artist B"]

    monkeypatch.setattr(tag_fixer, "PLUGINS", [FakeSource({
        "Song": {"artist": "New Artist", "title": "Song", "score": 1.0},
    })])
    db, records = _records(tag_fixer, tmp_path)
    rec = records[0]

    # The preview sees both values, and says the proposal replaces both.
    assert rec.old_artist == ["Artist A", "Artist B"]
    assert tag_fixer.planned_changes(rec) == {
        "artist": (["Artist A", "Artist B"], "New Artist"),
    }
    # The DB row stays plain text.
    conn = sqlite3.connect(db)
    (stored,) = conn.execute("SELECT old_artist FROM files").fetchone()
    conn.close()
    assert stored == "Artist A; Artist B"

    result = tagfix_controller.apply_selected_fields(
        [(rec, ["artist"])], records, db, log_callback=lambda _m: None
    )
    assert result.applied == [str(path)]
    assert get_field(path, "artist") == ["New Artist"]
    entry = read_change_log(result.log_path)[0]
    assert entry["changes"] == {
        "artist": {"before": ["Artist A", "Artist B"], "after": "New Artist"},
    }


@needs_ffmpeg
@pytest.mark.parametrize("ext", (".flac", ".mp3", ".wav"))
def test_a_value_added_since_the_preview_makes_the_file_stale(tmp_path, monkeypatch, ext):
    """The "changed since preview" check compares the full lists: adding a
    second artist after the scan must not be overwritten blind."""
    import tag_fixer

    files = build_genre_library(tmp_path, [(f"a{ext}", "Artist A", "Song", [])])
    path = files[f"a{ext}"]
    monkeypatch.setattr(tag_fixer, "PLUGINS", [FakeSource({
        "Song": {"artist": "New Artist", "score": 1.0},
    })])
    _db, records = _records(tag_fixer, tmp_path)
    rec = records[0]
    assert rec.old_artist == "Artist A"
    set_field(path, "artist", ["Artist A", "Artist B"])

    result = tag_fixer.apply_tag_changes([(rec, ["artist"])], log_callback=lambda _m: None)
    assert result.skipped == [str(path)] and not result.applied
    assert get_field(path, "artist") == ["Artist A", "Artist B"]


@needs_ffmpeg
def test_multi_value_title_and_album_are_read_in_full(tmp_path, monkeypatch):
    import tag_fixer

    files = build_genre_library(tmp_path, [("a.flac", "Art", "Song", [])])
    path = files["a.flac"]
    set_field(path, "album", ["Disc One", "Bonus"])
    monkeypatch.setattr(tag_fixer, "PLUGINS", [FakeSource({
        "Song": {"album": "Disc One", "score": 1.0},
    })])
    _db, records = _records(tag_fixer, tmp_path)
    rec = records[0]
    assert rec.old_title == "Song"
    assert rec.old_album == ["Disc One", "Bonus"]
    # "Disc One" alone would drop "Bonus": that is a change, shown as one.
    assert tag_fixer.planned_changes(rec) == {"album": (["Disc One", "Bonus"], "Disc One")}


def test_joined_proposal_equal_to_the_current_values_is_no_change():
    from tag_fixer import FileRecord, planned_changes

    rec = FileRecord(path="x.flac", status="new", old_artist=["A", "B"], new_artist="A; B")
    assert planned_changes(rec) == {}


# ── Finding 5: low-confidence genres are not merged ──────────────────────


def test_genres_only_from_results_at_the_proposal_threshold(tmp_path, monkeypatch):
    import tag_fixer

    build_genre_library(tmp_path, [("Art/t.wav", "Art", "T", [])])  # WAV: no ffmpeg needed

    class Source:
        def __init__(self, name, result):
            self.name, self.result = name, result

        def preflight(self):
            return SourceStatus(self.name, True, "test", "")

        def identify(self, _path):
            return dict(self.result)

    monkeypatch.setattr(tag_fixer, "PLUGINS", [
        Source("Good", {"artist": "Right", "score": 0.95, "genres": ["Rock"]}),
        Source("Weak", {"artist": "Wrong", "score": 0.30, "genres": ["Polka"]}),
    ])
    monkeypatch.setattr(tag_fixer, "is_remix", lambda _p: False)
    _db, records = _records(tag_fixer, tmp_path)
    rec = records[0]
    assert rec.new_artist == "Right"
    assert rec.new_genres == ["Rock"]


def test_low_confidence_genres_alone_propose_nothing(tmp_path, monkeypatch):
    import tag_fixer

    build_genre_library(tmp_path, [("Art/t.wav", "Art", "T", [])])  # WAV: no ffmpeg needed

    class Weak:
        name = "Weak"

        def preflight(self):
            return SourceStatus(self.name, True, "test", "")

        def identify(self, _path):
            return {"artist": "Wrong", "score": 0.30, "genres": ["Polka"]}

    monkeypatch.setattr(tag_fixer, "PLUGINS", [Weak()])
    monkeypatch.setattr(tag_fixer, "is_remix", lambda _p: False)
    _db, records = _records(tag_fixer, tmp_path)
    assert records[0].new_genres == [] and records[0].status == "unmatched"
