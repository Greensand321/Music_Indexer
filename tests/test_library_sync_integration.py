import csv
import os
import sys
import threading
import types

import pytest

import library_sync
from fingerprint_cache import flush_cache
from library_sync import MatchStatus, PerformanceProfile
from library_sync_review_report import ReportFormat, export_report
from library_sync_review_state import ReviewStateStore
import music_indexer_api


def _ensure_fp_module() -> types.ModuleType:
    if "fingerprint_generator" not in sys.modules:
        sys.modules["fingerprint_generator"] = types.ModuleType("fingerprint_generator")
    return sys.modules["fingerprint_generator"]


def _setup_media_stubs(monkeypatch, fp_map, corrupt=None, missing=None, cancel_after=None):
    corrupt = corrupt or set()
    missing = missing or set()
    fp_mod = _ensure_fp_module()

    def fake_compute(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kwargs):
        for idx, path in enumerate(paths):
            if cancel_after is not None and idx >= cancel_after:
                if cancel_event:
                    cancel_event.set()
                break
            fp = fp_map.get(os.path.basename(path), "0 0")
            yield path, 120, fp

    monkeypatch.setattr(fp_mod, "compute_fingerprints_parallel", fake_compute)

    def fake_get_tags(path):
        name = os.path.basename(path)
        if name in missing:
            return {"artist": None, "title": None, "album": None, "year": None, "track": None, "genre": None}
        return {"artist": "Artist", "title": name, "album": "Album", "year": None, "track": None, "genre": None}

    monkeypatch.setattr(music_indexer_api, "get_tags", fake_get_tags)

    class DummyInfo:
        def __init__(self):
            self.bitrate = 256000
            self.sample_rate = 44100

    def fake_mutagen(path, easy=False):
        if os.path.basename(path) in corrupt:
            raise IOError("corrupt file")
        return types.SimpleNamespace(info=DummyInfo())

    monkeypatch.setattr(music_indexer_api, "MutagenFile", fake_mutagen)


def test_partial_scan_and_cancellation_preserves_results(tmp_path, monkeypatch):
    fp_map = {"lib1.flac": "1 1 1", "inc1.flac": "1 1 1", "inc2.flac": "2 2 2"}
    _setup_media_stubs(monkeypatch, fp_map, cancel_after=1)

    library = tmp_path / "library"
    incoming = tmp_path / "incoming"
    library.mkdir()
    incoming.mkdir()
    (library / "lib1.flac").write_text("a")
    (library / "lib2.flac").write_text("b")
    (incoming / "inc1.flac").write_text("c")
    (incoming / "inc2.flac").write_text("d")

    db_path = tmp_path / "fp.db"
    cancel_event = threading.Event()
    profile = PerformanceProfile()
    res = library_sync.compare_libraries(
        str(library),
        str(incoming),
        str(db_path),
        cancel_event=cancel_event,
        include_profile=True,
        profiler=profile,
    )

    assert res["partial"] is True
    assert len(res["library_records"]) == 2
    assert len(res["incoming_records"]) in (0, 2)
    if res["matches"]:
        assert res["matches"][0]["status"] in {MatchStatus.NEW.value, MatchStatus.COLLISION.value}
    assert profile.progress_updates > 0
    flush_cache(str(db_path))


def test_rescan_threshold_change_uses_cache_and_profile(tmp_path, monkeypatch):
    fp_map = {"song.mp3": "1 1 1 2", "song.wav": "1 1 1 1"}
    _setup_media_stubs(monkeypatch, fp_map)

    library = tmp_path / "library"
    incoming = tmp_path / "incoming"
    library.mkdir()
    incoming.mkdir()
    (library / "song.mp3").write_text("lib")
    (incoming / "song.wav").write_text("inc")
    db_path = tmp_path / "fp.db"

    profile1 = PerformanceProfile()
    res_strict = library_sync.compare_libraries(
        str(library),
        str(incoming),
        str(db_path),
        thresholds={".wav": 0.2, ".mp3": 0.2},
        include_profile=True,
        profiler=profile1,
    )

    profile2 = PerformanceProfile()
    res_relaxed = library_sync.compare_libraries(
        str(library),
        str(incoming),
        str(db_path),
        thresholds={".wav": 0.5, ".mp3": 0.5},
        include_profile=True,
        profiler=profile2,
    )

    first_match = res_strict["matches"][0]
    second_match = res_relaxed["matches"][0]
    assert first_match["status"] in {MatchStatus.LOW_CONFIDENCE.value, MatchStatus.NEW.value}
    assert second_match["status"] == MatchStatus.COLLISION.value
    assert profile1.fingerprints_computed >= 2
    assert profile2.cache_hits >= 2
    assert profile2.fingerprints_computed == 0
    assert len(profile1.shortlist_sizes) == 1
    assert profile1.progress_updates >= 2
    flush_cache(str(db_path))


def test_export_report_covers_flags_and_notes(tmp_path, monkeypatch):
    fp_map = {
        "dup.flac": "9 9 9",
        "dup (1).flac": "9 9 9",
        "song.mp3": "1 2 3 4",
        "song.mp3.incoming": "4 3 2 1",
        "missing.wav": "5 5 5 5",
        "corrupt.flac": "8 8 8 8",
    }
    corrupt = {"corrupt.flac"}
    missing = {"missing.wav"}
    _setup_media_stubs(monkeypatch, fp_map, corrupt=corrupt, missing=missing)

    library = tmp_path / "library"
    incoming = tmp_path / "incoming"
    library.mkdir()
    incoming.mkdir()
    (library / "dup.flac").write_text("dup")
    (library / "song.mp3").write_text("song-old")
    (library / "corrupt.flac").write_text("bad")
    (incoming / "dup (1).flac").write_text("dup-inc")
    (incoming / "song.mp3").write_text("song-new")
    (incoming / "missing.wav").write_text("missing")
    (incoming / "notes.txt").write_text("skip me")
    (incoming / "song.mp3.incoming").write_text("alt content")

    db_path = tmp_path / "fp.db"
    lib_records = library_sync._scan_folder(str(library), str(db_path), cancel_event=threading.Event())
    inc_records = library_sync._scan_folder(str(incoming), str(db_path), cancel_event=threading.Event())

    matches = library_sync._match_tracks(
        inc_records, lib_records, {".wav": 0.3, ".mp3": 0.3, ".flac": 0.3, "default": 0.3}, {".flac": 3, ".mp3": 1}
    )
    assert len(matches) == 3

    flags = ReviewStateStore()
    for res in matches:
        if res.status in (MatchStatus.EXACT_MATCH, MatchStatus.COLLISION):
            flags.flag_for_replace(res)
            flags.set_note(res.incoming.track_id, "duplicate handled")
            break
    for res in matches:
        if res.status == MatchStatus.NEW:
            flags.flag_for_copy(res.incoming.track_id)
            flags.set_note(res.incoming.track_id, "new track")
            break

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    export_report(
        str(json_path),
        matches,
        flags,
        {"global_threshold": 0.3, "per_format_overrides": {".flac": 0.3}},
        fmt=ReportFormat.JSON.value,
    )
    export_report(
        str(csv_path),
        matches,
        flags,
        {"global_threshold": 0.3, "per_format_overrides": {".flac": 0.3}},
        fmt=ReportFormat.CSV.value,
    )

    with open(json_path, "r", encoding="utf-8") as f:
        payload = f.read()
    assert "schema_version" in payload
    assert any(row.incoming.path.endswith("dup (1).flac") for row in matches)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(matches)
    assert set(rows[0].keys()) >= set(
        [
            "schema_version",
            "incoming_path",
            "existing_path",
            "status",
            "distance",
            "threshold_used",
            "incoming_quality_score",
            "existing_quality_score",
            "quality_delta",
            "user_flag",
            "notes",
            "timestamp",
            "scan_config_hash",
        ]
    )
    flush_cache(str(db_path))
