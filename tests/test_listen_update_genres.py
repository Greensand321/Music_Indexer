"""update_genres command line: preview first, never write blind (listening audit).

The CLI ``main()`` used to write each file the moment its MusicBrainz lookup
returned. ``run_cli`` now looks everything up, lists the changes, asks, and
writes exactly the listed changes (skipping a file changed since the
preview). MusicBrainz is never contacted: the lookup is injected.
"""
from __future__ import annotations

import io
import json

import pytest

from _helpers.meta_tags import build_genre_library, get_genres, needs_ffmpeg


def _lookup(genres_by_title):
    from controllers.normalize_controller import FillLookup
    from utils.audio_metadata_reader import read_tags

    def lookup(path):
        title = read_tags(path).get("title")
        if title in genres_by_title:
            return FillLookup(genres=list(genres_by_title[title]), status="matched", detail=title)
        return FillLookup(status="no_match", detail="no match")

    return lookup


@pytest.fixture
def library(tmp_path):
    files = build_genre_library(tmp_path, [
        ("one.flac", "Art", "One", ["Rock"]),
        ("two.flac", "Art", "Two", []),
        ("multi.flac", "Art", "Multi", ["Rock", "Pop"]),
    ])
    return tmp_path, files


PLAN = {"plan_kwargs": {"lookup": _lookup({"One": ["electronic"], "Two": ["jazz"], "Multi": ["x"]}),
                        "rate_limit": 0}}


@needs_ffmpeg
def test_cli_lists_the_plan_and_writes_nothing_without_a_yes(library):
    import update_genres

    root, files = library
    out: list[str] = []
    asked: list[str] = []
    written = update_genres.run_cli(
        str(root), ask=lambda prompt: asked.append(prompt) or "n", out=out.append, **PLAN
    )

    assert written == 0
    assert asked and "2 file(s)" in asked[0]
    text = "\n".join(out)
    assert "one.flac: [Rock] → [Rock, electronic]" in text
    assert "two.flac: [] → [jazz]" in text
    assert get_genres(files["one.flac"]) == ["Rock"]
    assert get_genres(files["two.flac"]) == []
    log = (root / "Docs" / "genre_update_log.txt").read_text(encoding="utf-8")
    assert "planned: one.flac" in log and "nothing was written" in log


@needs_ffmpeg
def test_cli_writes_exactly_the_confirmed_plan_and_logs_it(library):
    import update_genres

    root, files = library
    written = update_genres.run_cli(str(root), ask=lambda _p: "y", out=lambda _m: None, **PLAN)

    assert written == 2
    assert get_genres(files["one.flac"]) == ["Rock", "electronic"]
    assert get_genres(files["two.flac"]) == ["jazz"]
    assert get_genres(files["multi.flac"]) == ["Rock", "Pop"]
    logs = list((root / "Docs").glob("tag_changes_genre_fill_cli_*.jsonl"))
    assert len(logs) == 1
    records = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines() if line]
    assert sorted(r["status"] for r in records if "status" in r) == ["written", "written"]


@needs_ffmpeg
def test_a_file_changed_after_the_preview_is_skipped_not_overwritten(library):
    from _helpers.meta_tags import set_genres
    import update_genres

    root, files = library

    def ask(_prompt):
        set_genres(files["one.flac"], ["Metal"])  # edited while the plan was on screen
        return "yes"

    written = update_genres.run_cli(str(root), ask=ask, out=lambda _m: None, **PLAN)

    assert written == 1
    assert get_genres(files["one.flac"]) == ["Metal"]
    assert get_genres(files["two.flac"]) == ["jazz"]
    log = (root / "Docs" / "genre_update_log.txt").read_text(encoding="utf-8")
    assert "skipped (genres changed since the preview): one.flac" in log


def test_process_file_never_writes_a_file_whose_genres_cannot_be_read(tmp_path, monkeypatch):
    import update_genres
    from utils import audio_metadata_reader

    track = tmp_path / "broken.flac"
    track.write_bytes(b"fLaC\x00\x00")
    writes: list[str] = []
    monkeypatch.setattr(update_genres, "extract_artist_title", lambda _p: ("Art", "Song"))
    monkeypatch.setattr(update_genres, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(update_genres.musicbrainzngs, "set_useragent", lambda *a, **k: None)
    monkeypatch.setattr(
        update_genres.musicbrainzngs, "search_recordings",
        lambda **k: {"recording-list": [{"tag-list": [{"name": "jazz", "count": "3"}]}]},
    )
    monkeypatch.setattr(update_genres, "write_genre_tag", lambda p, g: writes.append(p))
    monkeypatch.setattr(update_genres, "update_genre_tag", lambda p, g: writes.append(p) or True)
    log = io.StringIO()

    update_genres.process_file(str(track), log)

    assert writes == []
    assert "could not read its tags" in log.getvalue()
    assert audio_metadata_reader.read_genres_checked(str(track))[1].startswith("read failed")


def test_process_file_logs_why_a_write_failed(tmp_path, monkeypatch):
    import update_genres
    from utils.tag_writer import TagWriteError

    def locked(_path, _values):
        raise TagWriteError("could not save ([Errno 13] the file is in use)")

    monkeypatch.setattr(update_genres, "extract_genre_list", lambda _p: ["Rock"])
    monkeypatch.setattr(update_genres, "extract_artist_title", lambda _p: ("Art", "Song"))
    monkeypatch.setattr(update_genres, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(update_genres.musicbrainzngs, "set_useragent", lambda *a, **k: None)
    monkeypatch.setattr(
        update_genres.musicbrainzngs, "search_recordings",
        lambda **k: {"recording-list": [{"tag-list": [{"name": "jazz", "count": "3"}]}]},
    )
    monkeypatch.setattr(update_genres, "write_fields", locked)
    log = io.StringIO()

    update_genres.process_file(str(tmp_path / "a.flac"), log)

    assert "failed to write genres [Rock, jazz]: could not save" in log.getvalue()
    assert "the file is in use" in log.getvalue()
    with pytest.raises(TagWriteError, match="in use"):
        update_genres.write_genre_tag(str(tmp_path / "a.flac"), ["x"])
    assert update_genres.update_genre_tag(str(tmp_path / "a.flac"), ["x"]) is False
