"""Multi-value genres survive every genre tool (META-02/04/05/13/14).

The tag reader used to return only the *first* genre value, so every tool
that rewrote genres silently deleted the others. These tests build real
FLAC / MP3 / Opus / M4A / Ogg / WAV files with several genre values and check
what is on disk afterwards, read back with mutagen directly.
"""
from __future__ import annotations

import os
import sqlite3
import stat

import pytest

from _helpers.meta_tags import ALL_FORMATS, build_genre_library, get_field, get_genres, needs_ffmpeg, set_genres

pytestmark = needs_ffmpeg


# ── reader ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_read_genres_returns_every_value(tmp_path, ext):
    from utils.audio_metadata_reader import read_genres, read_genres_checked, read_tags

    files = build_genre_library(tmp_path, [(f"a{ext}", "A", "T", ["Rock", "Pop", "Jazz"])])
    path = str(files[f"a{ext}"])
    assert read_genres(path) == ["Rock", "Pop", "Jazz"]
    assert read_genres_checked(path) == (["Rock", "Pop", "Jazz"], None)
    # Backward compatible: read_tags keeps its single-string "genre" and gains no keys.
    tags = read_tags(path)
    assert tags["genre"] == "Rock"
    assert "genres" not in tags


def test_read_genres_untagged_and_corrupt(tmp_path):
    from utils.audio_metadata_reader import read_genres_checked

    files = build_genre_library(tmp_path, [("u.flac", None, None, [])])
    assert read_genres_checked(str(files["u.flac"])) == ([], None)
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all")
    genres, error = read_genres_checked(str(bad))
    assert genres == []
    # mutagen either refuses the file or finds nothing; never raises
    assert error is None or error.startswith("read failed")


# ── Normalize with a mapping (META-02, META-14, META-13) ─────────────────


@pytest.mark.parametrize("ext", [".flac", ".mp3", ".opus", ".m4a", ".wav"])
def test_normalize_keeps_unmapped_genres_of_multi_genre_files(tmp_path, ext):
    """Mapping "rock" must not delete the file's other genre values."""
    from controllers import normalize_controller as nc

    files = build_genre_library(tmp_path, [(f"a{ext}", "A", "T", ["rock", "Pop", "90s"])])
    path = str(files[f"a{ext}"])
    mapping = {"rock": ["Classic Rock"], "90s": ["invalid"]}

    raw = nc.scan_raw_genres(str(tmp_path))
    assert {"rock", "Pop", "90s"} <= set(raw)

    plan = nc.plan_genre_normalization(str(tmp_path), mapping)
    assert [c.path for c in plan.changes] == [path]
    change = plan.changes[0]
    # The preview shows every value, and what will be dropped (only "90s").
    assert change.before == ["rock", "Pop", "90s"]
    assert change.after == ["Classic Rock", "Pop"]

    result = nc.apply_genre_changes(plan.changes)
    assert result.applied == [path] and not result.failed
    assert get_genres(path) == ["Classic Rock", "Pop"]


def test_normalize_maps_each_part_of_a_combined_string():
    """META-14: the scan lists split parts, so the mapping must apply to parts."""
    from controllers.normalize_controller import normalize_genres

    mapping = {"hip hop": ["Hip-Hop"], "rap": ["Rap"], "90s": ["invalid"]}
    assert normalize_genres(["hip hop/rap"], mapping) == ["Hip-Hop", "Rap"]
    assert normalize_genres(["hip hop; 90s"], mapping) == ["Hip-Hop"]
    # A whole-string mapping still wins over its parts.
    assert normalize_genres(["hip hop/rap"], {"hip hop/rap": ["Hip-Hop"]}) == ["Hip-Hop"]
    # Unmapped parts fall through unchanged.
    assert normalize_genres(["Indie/rap"], mapping) == ["Indie", "Rap"]


def test_normalize_writes_wav_id3_genres(tmp_path):
    """META-13: WAV with ID3 tags used to always fail (and reappear every preview)."""
    from controllers import normalize_controller as nc

    files = build_genre_library(tmp_path, [("w.wav", "A", "T", ["hip hop", "Jazz"])])
    plan = nc.plan_genre_normalization(str(tmp_path), {"hip hop": ["Hip-Hop"]})
    assert len(plan.changes) == 1 and not plan.blocked
    result = nc.apply_genre_changes(plan.changes)
    assert result.applied == [str(files["w.wav"])]
    assert get_genres(files["w.wav"]) == ["Hip-Hop", "Jazz"]
    # A second preview finds nothing left to do.
    assert nc.plan_genre_normalization(str(tmp_path), {"hip hop": ["Hip-Hop"]}).changes == []


def test_preview_flags_read_only_files_instead_of_failing_later(tmp_path):
    """META-13: files that can't be written are shown up front, not planned."""
    from controllers import normalize_controller as nc

    files = build_genre_library(tmp_path, [
        ("ok.flac", "A", "T1", ["rock"]),
        ("ro.flac", "A", "T2", ["rock"]),
    ])
    ro = files["ro.flac"]
    os.chmod(ro, stat.S_IRUSR | stat.S_IRGRP)
    try:
        plan = nc.plan_genre_normalization(str(tmp_path), {"rock": ["Rock"]})
        assert [c.path for c in plan.changes] == [str(files["ok.flac"])]
        assert [(p, "read-only" in why) for p, why in plan.blocked] == [(str(ro), True)]
        assert plan.unchanged == 0
    finally:
        os.chmod(ro, stat.S_IRUSR | stat.S_IWUSR)


def test_apply_skips_a_file_changed_since_the_preview(tmp_path):
    """What executes must be what was previewed — never overwrite blind."""
    from controllers import normalize_controller as nc

    files = build_genre_library(tmp_path, [("a.flac", "A", "T", ["rock"])])
    plan = nc.plan_genre_normalization(str(tmp_path), {"rock": ["Rock"]})
    set_genres(files["a.flac"], ["rock", "Added Later"])  # someone edits it meanwhile
    result = nc.apply_genre_changes(plan.changes)
    assert result.applied == [] and result.skipped == [str(files["a.flac"])]
    assert get_genres(files["a.flac"]) == ["rock", "Added Later"]


def test_apply_writes_a_before_after_change_log(tmp_path):
    """META-23: every write leaves a Docs/ record of its previous value."""
    from controllers import normalize_controller as nc
    from utils.tag_change_log import TagChangeLog, read_change_log

    files = build_genre_library(tmp_path, [("a.mp3", "A", "T", ["rock", "Pop"])])
    plan = nc.plan_genre_normalization(str(tmp_path), {"rock": ["Rock"]})
    with TagChangeLog(str(tmp_path), "genre_normalize") as log:
        result = nc.apply_genre_changes(plan.changes, change_log=log)
    assert result.log_path == log.path
    assert os.path.dirname(log.path) == str(tmp_path / "Docs")
    entries = read_change_log(log.path)
    assert entries == [{
        "time": entries[0]["time"], "tool": "genre_normalize",
        "path": str(files["a.mp3"]), "status": "written",
        "changes": {"genre": {"before": ["rock", "Pop"], "after": ["Rock", "Pop"]}},
    }]


# ── Tag Fixer genre write (META-05, META-13) ──────────────────────────────


class _FakeSource:
    """A lookup source keyed by title; no network."""

    name = "FakeID"

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def preflight(self):
        from plugins.base import SourceStatus

        return SourceStatus(self.name, True, "test")

    def identify(self, path):
        self.calls.append(path)
        from utils.audio_metadata_reader import read_tags

        return dict(self.answers.get(read_tags(path).get("title"), {}))


def _scan(tag_fixer, root, monkeypatch, answers):
    source = _FakeSource(answers)
    monkeypatch.setattr(tag_fixer, "PLUGINS", [source])
    db_path = os.path.join(str(root), "Docs", ".soundvault.db")
    tag_fixer.init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        records = tag_fixer.build_file_records(
            str(root), db_conn=conn, show_all=True, log_callback=lambda _m: None
        )
    finally:
        conn.close()
    return {os.path.basename(str(r.path)): r for r in records}, source


@pytest.mark.parametrize("ext", [".flac", ".mp3", ".opus", ".m4a", ".wav"])
def test_tag_fixer_adds_genres_without_dropping_or_joining(tmp_path, monkeypatch, ext):
    import tag_fixer

    files = build_genre_library(tmp_path, [(f"m{ext}", "Art", "Multi", ["Rock", "Pop"])])
    recs, _ = _scan(tag_fixer, tmp_path, monkeypatch,
                    {"Multi": {"genres": ["pop", "Electronic"], "score": 1.0}})
    rec = recs[f"m{ext}"]
    assert rec.old_genres == ["Rock", "Pop"]
    assert tag_fixer.planned_changes(rec) == {
        "genres": (["Rock", "Pop"], ["Rock", "Pop", "Electronic"])
    }
    assert tag_fixer.update_tags(str(rec.path), rec, ["genres"], lambda _m: None)
    # A real multi-value list — not one joined "Rock; Pop; Electronic" string.
    assert get_genres(files[f"m{ext}"]) == ["Rock", "Pop", "Electronic"]


def test_tag_fixer_never_touches_genres_when_lookup_found_none(tmp_path, monkeypatch):
    import tag_fixer

    files = build_genre_library(tmp_path, [
        ("m.flac", "Art", "Multi", ["Rock", "Pop"]),
        ("u.flac", "Art", "Nothing", ["Rock", "Pop"]),
    ])
    recs, _ = _scan(tag_fixer, tmp_path, monkeypatch,
                    {"Multi": {"artist": "Fixed Artist", "score": 1.0}})
    assert recs["u.flac"].status == "unmatched"
    assert tag_fixer.planned_changes(recs["u.flac"]) == {}
    assert tag_fixer.planned_changes(recs["m.flac"]) == {"artist": ("Art", "Fixed Artist")}
    n = tag_fixer.apply_tag_proposals(
        list(recs.values()), fields=["artist", "title", "album", "genres"],
        log_callback=lambda _m: None,
    )
    assert n == 1
    assert get_genres(files["m.flac"]) == ["Rock", "Pop"]
    assert get_genres(files["u.flac"]) == ["Rock", "Pop"]
    assert get_field(files["m.flac"], "artist") == ["Fixed Artist"]
    assert get_field(files["u.flac"], "artist") == ["Art"]


def test_tag_fixer_wav_with_id3_does_not_abort_the_batch(tmp_path, monkeypatch):
    """META-13: the WAV used to raise out of update_tags and stop the batch."""
    import tag_fixer

    files = build_genre_library(tmp_path, [
        ("a_w.wav", "Art", "Wav", ["hip hop"]),
        ("b_f.flac", "Art", "Flac", ["Rock"]),
    ])
    answers = {
        "Wav": {"artist": "New Wav Artist", "genres": ["Jazz"], "score": 1.0},
        "Flac": {"artist": "New Flac Artist", "score": 1.0},
    }
    recs, _ = _scan(tag_fixer, tmp_path, monkeypatch, answers)
    ordered = [recs["a_w.wav"], recs["b_f.flac"]]
    result = tag_fixer.apply_tag_changes(
        [(r, ["artist", "genres"]) for r in ordered], log_callback=lambda _m: None
    )
    assert sorted(result.applied) == sorted(str(p) for p in files.values())
    assert get_field(files["a_w.wav"], "artist") == ["New Wav Artist"]
    assert get_genres(files["a_w.wav"]) == ["hip hop", "Jazz"]
    assert get_field(files["b_f.flac"], "artist") == ["New Flac Artist"]


def test_tag_fixer_failed_write_is_reported_and_batch_continues(tmp_path, monkeypatch):
    import tag_fixer

    files = build_genre_library(tmp_path, [
        ("a.flac", "Art", "One", []),
        ("b.flac", "Art", "Two", []),
    ])
    recs, _ = _scan(tag_fixer, tmp_path, monkeypatch, {
        "One": {"artist": "X", "score": 1.0}, "Two": {"artist": "Y", "score": 1.0},
    })
    real_assign = tag_fixer.assign_fields

    def flaky(audio, values):
        if str(getattr(audio, "filename", "")).endswith("a.flac"):
            raise OSError("disk says no")
        return real_assign(audio, values)

    monkeypatch.setattr(tag_fixer, "assign_fields", flaky)
    result = tag_fixer.apply_tag_changes(
        [(recs["a.flac"], ["artist"]), (recs["b.flac"], ["artist"])], log_callback=lambda _m: None
    )
    assert result.failed == [str(files["a.flac"])]
    assert result.applied == [str(files["b.flac"])]
    assert get_field(files["a.flac"], "artist") == ["Art"]
