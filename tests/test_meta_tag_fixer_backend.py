"""Tag Fixer backend regressions (META-06/10/11/15/16/18/19/21/23).

No network: lookup sources are fakes, and the AcoustID / MusicBrainz client
modules are replaced with stubs where their call sites are under test.
"""
from __future__ import annotations

import json
import os
import sqlite3
import types

import pytest

from _helpers.meta_tags import build_genre_library, get_field, needs_ffmpeg
from plugins.base import SourceStatus


class FakeSource:
    def __init__(self, answers=None, name="FakeID", active=True, problem="", fail=None, delay=None):
        self.name = name
        self.answers = answers or {}
        self.active = active
        self.problem = problem
        self.fail = fail
        self.calls = []
        self.delay = delay

    def preflight(self):
        return SourceStatus(self.name, self.active, "test", self.problem)

    def identify(self, path):
        self.calls.append(path)
        if self.delay:
            self.delay(path)
        if self.fail:
            raise RuntimeError(self.fail)
        from utils.audio_metadata_reader import read_tags

        return dict(self.answers.get(read_tags(path).get("title"), {}))


def _db(root):
    import tag_fixer

    path = os.path.join(str(root), "Docs", ".soundvault.db")
    tag_fixer.init_db(path)
    return path


def _build(tag_fixer, root, **kwargs):
    conn = sqlite3.connect(_db(root))
    try:
        return tag_fixer.build_file_records(
            str(root), db_conn=conn, show_all=True, log_callback=kwargs.pop("log", lambda _m: None), **kwargs
        )
    finally:
        conn.close()


# ── META-16: reserved folders ─────────────────────────────────────────────


def test_find_files_skips_reserved_folders_but_keeps_manual_review(tmp_path):
    import tag_fixer

    for rel in ("Artist/a.mp3", "Quarantine/q.flac", "Not Sorted/ns.flac", "Trash/t.mp3",
                "Docs/d.mp3", "Playlists/p.mp3", "Manual Review/mr.mp3",
                "Artist/Docs/nested_ok.mp3", "quarantine/case.flac"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    found = sorted(os.path.relpath(f, tmp_path) for f in tag_fixer.find_files(str(tmp_path)))
    assert found == sorted([
        os.path.join("Artist", "a.mp3"),
        os.path.join("Artist", "Docs", "nested_ok.mp3"),
        os.path.join("Manual Review", "mr.mp3"),
    ])
    from controllers.tagfix_controller import discover_files

    assert sorted(os.path.relpath(f, tmp_path) for f in discover_files(str(tmp_path))) == found


def test_one_extension_set_for_every_tag_tool():
    """META-25: the Fill tab skipped .opus while the other tools handled it."""
    import tag_fixer
    import update_genres

    assert ".opus" in update_genres.SUPPORTED_EXTS
    assert update_genres.SUPPORTED_EXTS == tag_fixer.SUPPORTED_EXTS


# ── META-06: preview == what apply writes ────────────────────────────────


def test_planned_changes_never_proposes_blanks_or_no_ops():
    from tag_fixer import FileRecord, planned_changes

    rec = FileRecord(path="x.mp3", status="new", old_artist="A", new_artist=None,
                     old_title="T", new_title="T", old_album="Al", new_album="",
                     old_genres=["Rock"], new_genres=["rock"])
    assert planned_changes(rec) == {}
    rec.new_artist = "  B  "
    rec.new_genres = ["Pop"]
    assert planned_changes(rec) == {"artist": ("A", "B"), "genres": (["Rock"], ["Rock", "Pop"])}
    assert planned_changes(rec, ["genres"]) == {"genres": (["Rock"], ["Rock", "Pop"])}


@needs_ffmpeg
def test_unmatched_and_no_diff_statuses(tmp_path, monkeypatch):
    import tag_fixer

    build_genre_library(tmp_path, [
        ("a.flac", "Art", "Hit", []),
        ("b.flac", "Art", "Same", []),
        ("c.flac", "Art", "Miss", []),
    ])
    source = FakeSource({"Hit": {"artist": "New", "score": 1.0},
                         "Same": {"artist": "Art", "title": "Same", "score": 1.0}})
    monkeypatch.setattr(tag_fixer, "PLUGINS", [source])
    diag = tag_fixer.ScanDiagnostics()
    recs = {os.path.basename(str(r.path)): r for r in _build(tag_fixer, tmp_path, diagnostics=diag)}
    assert recs["a.flac"].status == "new"
    assert recs["b.flac"].status == "no_diff"
    assert recs["c.flac"].status == "unmatched" and recs["c.flac"].note == "no match found"
    assert (diag.matched, diag.unmatched) == (2, 1)


# ── META-18 / META-19: say why nothing matched ───────────────────────────


@needs_ffmpeg
def test_inactive_source_is_skipped_and_the_reason_reported(tmp_path, monkeypatch):
    import tag_fixer

    build_genre_library(tmp_path, [("a.flac", "Art", "Hit", [])])
    off = FakeSource({"Hit": {"artist": "X", "score": 1.0}}, name="AcoustID", active=False,
                     problem="fpcalc (Chromaprint) was not found")
    monkeypatch.setattr(tag_fixer, "PLUGINS", [off])
    logs = []
    diag = tag_fixer.ScanDiagnostics()
    recs = _build(tag_fixer, tmp_path, diagnostics=diag, log=logs.append)
    assert off.calls == []  # really skipped, not "will be skipped" then run anyway
    assert recs[0].status == "unmatched"
    assert "fpcalc" in diag.unmatched_reason()
    assert any("fpcalc" in line for line in logs)


@needs_ffmpeg
def test_lookup_errors_are_counted_not_swallowed(tmp_path, monkeypatch):
    import tag_fixer

    build_genre_library(tmp_path, [("a.flac", "Art", "One", []), ("b.flac", "Art", "Two", [])])
    monkeypatch.setattr(tag_fixer, "PLUGINS", [FakeSource(fail="network error (offline?)")])
    diag = tag_fixer.ScanDiagnostics()
    recs = _build(tag_fixer, tmp_path, diagnostics=diag)
    assert all(r.status == "unmatched" for r in recs)
    assert all("network error" in r.note for r in recs)
    assert diag.errors == {"FakeID: network error (offline?)": 2}
    assert "network error" in diag.unmatched_reason()


def test_acoustid_preflight_reports_missing_fpcalc_and_offline(monkeypatch):
    from plugins.acoustid_plugin import AcoustIDService

    monkeypatch.setattr(AcoustIDService, "fingerprinting_available", staticmethod(lambda: False))
    monkeypatch.setattr(AcoustIDService, "check_connection", staticmethod(lambda: pytest.fail("network!")))
    status = AcoustIDService().preflight()
    assert not status.active and "fpcalc" in status.problem

    monkeypatch.setattr(AcoustIDService, "fingerprinting_available", staticmethod(lambda: True))
    monkeypatch.setattr(AcoustIDService, "check_connection", staticmethod(lambda: False))
    status = AcoustIDService().preflight()
    assert not status.active and "offline" in status.problem

    monkeypatch.setattr(AcoustIDService, "check_connection", staticmethod(lambda: True))
    assert AcoustIDService().preflight().active


def test_empty_api_key_means_built_in_key(monkeypatch):
    """Clearing the key in Settings saved "" — which beat the built-in default."""
    import config
    import tag_fixer
    from plugins import acoustid_plugin

    config.save_config({"metadata_api_key": "  "})
    assert acoustid_plugin.acoustid_api_key() == (tag_fixer.ACOUSTID_API_KEY, "built-in key")
    config.save_config({"metadata_api_key": "mykey"})
    assert acoustid_plugin.acoustid_api_key() == ("mykey", "key from Settings")

    seen = []
    monkeypatch.setattr(acoustid_plugin, "query_metadata", lambda svc, key, f: seen.append((svc, key)) or {})
    acoustid_plugin.AcoustIDService().identify("x.mp3")
    assert seen == [("AcoustID", "mykey")]


def test_acoustid_identify_raises_a_readable_reason(monkeypatch):
    from plugins import acoustid_plugin

    def boom(*_a):
        raise ConnectionError("Failed to establish a new connection")

    monkeypatch.setattr(acoustid_plugin, "query_metadata", boom)
    with pytest.raises(RuntimeError, match="network error"):
        acoustid_plugin.AcoustIDService().identify("x.mp3")


def test_lookup_errors_never_leak_the_api_key(monkeypatch, tmp_path):
    """A requests error message embeds the URL — including ``api_key=…``."""
    import plugins.lastfm as lastfm

    monkeypatch.setenv("LASTFM_API_KEY", "SECRET123")
    monkeypatch.setattr(lastfm, "read_tags", lambda _p: {"artist": "A", "title": "T"})

    def fail(*_a, **_k):
        raise ValueError(
            "HTTPConnectionPool(host='ws.audioscrobbler.com'): bad url "
            "/2.0/?method=track.getInfo&api_key=SECRET123&artist=A"
        )

    monkeypatch.setattr(lastfm.requests, "get", fail)
    with pytest.raises(RuntimeError) as info:
        lastfm.LastfmPlugin().identify(str(tmp_path / "x.mp3"))
    assert "SECRET123" not in str(info.value)


def test_settings_service_selection_is_read(monkeypatch):
    """META-19: Settings -> Service was saved but never read."""
    import config
    from plugins.lastfm import LastfmPlugin
    from plugins.musicbrainz_plugin import MusicBrainzSearchPlugin

    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    monkeypatch.setattr("plugins.lastfm.API_KEY", None)
    config.save_config({"metadata_service": "AcoustID"})
    assert not MusicBrainzSearchPlugin().preflight().active
    assert LastfmPlugin().preflight() == SourceStatus("Last.fm", False, "", "")

    config.save_config({"metadata_service": "MusicBrainz"})
    assert MusicBrainzSearchPlugin().preflight().active

    config.save_config({"metadata_service": "Last.fm"})
    status = LastfmPlugin().preflight()
    assert not status.active and "LASTFM_API_KEY" in status.problem
    monkeypatch.setenv("LASTFM_API_KEY", "k")
    assert LastfmPlugin().preflight().active


def test_musicbrainz_source_is_discovered_by_the_tag_fixer():
    import tag_fixer

    names = [type(p).__name__ for p in tag_fixer.PLUGINS]
    assert "MusicBrainzSearchPlugin" in names and "AcoustIDService" in names


# ── META-15: MusicBrainz user agent ───────────────────────────────────────


def test_acoustid_enrichment_sets_the_user_agent_and_reports_errors(monkeypatch):
    import config
    import metadata_service

    calls = []
    mb = types.ModuleType("musicbrainzngs")

    def set_useragent(app, version, contact=None):
        calls.append(("ua", app, version, contact))

    def get_recording_by_id(rid, includes=()):
        if not any(c[0] == "ua" for c in calls):
            raise RuntimeError("UsageError: set a user agent first")
        return {"recording": {"releases": [{"title": "The Album"}], "tag-list": [{"name": "rock"}]}}

    mb.set_useragent = set_useragent
    mb.get_recording_by_id = get_recording_by_id
    ac = types.ModuleType("acoustid")
    ac.match = lambda key, path: iter([(0.97, "rid-1", "Song", "Artist")])
    monkeypatch.setitem(__import__("sys").modules, "musicbrainzngs", mb)
    monkeypatch.setitem(__import__("sys").modules, "acoustid", ac)

    config.save_config({"musicbrainz_useragent": {"app": "", "version": "", "contact": "me@x"}})
    result = metadata_service.query_acoustid("key", "song.mp3")
    assert calls == [("ua", "AlphaDEX", "1.0", "me@x")]
    assert result["album"] == "The Album" and result["genres"] == ["rock"]
    assert "enrichment_error" not in result

    mb.get_recording_by_id = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503"))
    result = metadata_service.query_acoustid("key", "song.mp3")
    assert result["artist"] == "Artist" and "503" in result["enrichment_error"]


# ── META-10 / META-11: cancel and progress ───────────────────────────────


@needs_ffmpeg
def test_cancel_is_honoured_per_file_and_progress_is_a_fraction(tmp_path, monkeypatch):
    import tag_fixer

    build_genre_library(tmp_path, [(f"t{i}.flac", "Art", f"T{i}", []) for i in range(6)])
    source = FakeSource()
    monkeypatch.setattr(tag_fixer, "PLUGINS", [source])
    seen = []
    diag = tag_fixer.ScanDiagnostics()
    recs = _build(tag_fixer, tmp_path, progress_detail=lambda i, t: seen.append((i, t)),
                  should_cancel=lambda: len(source.calls) >= 2, diagnostics=diag)
    assert len(source.calls) == 2 and len(recs) == 2
    assert diag.cancelled
    assert seen == [(1, 6), (2, 6)]


# ── META-21 / META-23: only successes are "applied"; changes are logged ──


@needs_ffmpeg
def test_failed_writes_are_not_marked_applied_and_every_change_is_logged(tmp_path, monkeypatch):
    import stat

    import tag_fixer
    from controllers import tagfix_controller
    from utils.tag_change_log import read_change_log

    files = build_genre_library(tmp_path, [
        ("ok.flac", "Art", "Ok", ["Rock", "Pop"]),
        ("ro.flac", "Art", "Ro", []),
    ])
    monkeypatch.setattr(tag_fixer, "PLUGINS", [FakeSource({
        "Ok": {"artist": "New", "genres": ["Jazz"], "score": 1.0},
        "Ro": {"artist": "New", "score": 1.0},
    })])
    db_path = _db(tmp_path)
    records = tagfix_controller.gather_records(str(tmp_path), db_path, True, None)
    by_name = {os.path.basename(str(r.path)): r for r in records}
    ro = files["ro.flac"]
    os.chmod(ro, stat.S_IRUSR | stat.S_IRGRP)
    try:
        real = tag_fixer.MutagenFile

        def opener(path, easy=False):
            # The suite may run as root, which ignores permission bits; make
            # the save fail the way it does for a normal user.
            audio = real(path, easy=easy)
            if str(path).endswith("ro.flac"):
                def refuse():
                    raise PermissionError("read-only file")
                audio.save = refuse
            return audio

        monkeypatch.setattr(tag_fixer, "MutagenFile", opener)
        result = tagfix_controller.apply_selected_fields(
            [(by_name["ok.flac"], ["artist", "genres"]), (by_name["ro.flac"], ["artist"])],
            records, db_path, log_callback=lambda _m: None,
        )
    finally:
        os.chmod(ro, stat.S_IRUSR | stat.S_IWUSR)
    assert result.applied == [str(files["ok.flac"])]
    assert result.failed == [str(ro)]
    conn = sqlite3.connect(db_path)
    statuses = dict(conn.execute("SELECT path, status FROM files"))
    conn.close()
    assert statuses[str(files["ok.flac"])] == "applied"
    assert statuses[str(ro)] != "applied"

    entries = read_change_log(result.log_path)
    assert os.path.dirname(result.log_path) == str(tmp_path / "Docs")
    by_path = {e["path"]: e for e in entries}
    assert by_path[str(files["ok.flac"])]["status"] == "written"
    assert by_path[str(files["ok.flac"])]["changes"] == {
        "artist": {"before": "Art", "after": "New"},
        "genres": {"before": ["Rock", "Pop"], "after": ["Rock", "Pop", "Jazz"]},
    }
    assert by_path[str(ro)]["status"] == "failed" and "read-only" in by_path[str(ro)]["error"]
    assert get_field(ro, "artist") == ["Art"]


@needs_ffmpeg
def test_apply_skips_a_file_edited_since_the_scan(tmp_path, monkeypatch):
    import tag_fixer
    from _helpers.meta_tags import _real_mutagen

    files = build_genre_library(tmp_path, [("a.flac", "Art", "One", [])])
    monkeypatch.setattr(tag_fixer, "PLUGINS", [FakeSource({"One": {"artist": "New", "score": 1.0}})])
    rec = _build(tag_fixer, tmp_path)[0]
    audio = _real_mutagen()["flac"].FLAC(str(files["a.flac"]))
    audio["artist"] = ["Edited By Hand"]
    audio.save()
    result = tag_fixer.apply_tag_changes([(rec, ["artist"])], log_callback=lambda _m: None)
    assert result.skipped == [str(files["a.flac"])] and not result.applied
    assert get_field(files["a.flac"], "artist") == ["Edited By Hand"]


def test_fix_tags_cli_only_marks_successful_writes(tmp_path, monkeypatch):
    """The CLI path had the same "every selected file is applied" bug."""
    import tag_fixer

    (tmp_path / "x.mp3").write_bytes(b"not really audio")
    monkeypatch.setattr(tag_fixer, "PLUGINS", [])
    summary = tag_fixer.fix_tags(str(tmp_path), log_callback=lambda _m: None)
    assert summary == {"processed": 1, "updated": 0}


def test_write_guard_allows_one_writer():
    from controllers import tag_write_guard as g

    assert g.acquire("A")
    try:
        assert not g.acquire("B")
        assert g.current_owner() == "A"
        assert "A is writing tags" in g.busy_message("B")
        g.release("B")  # not the owner: no effect
        assert g.current_owner() == "A"
    finally:
        g.release("A")
    assert g.current_owner() is None
