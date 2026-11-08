"""Previews, caches and audit logs are written whole or not at all, and say the truth.

Before:
* ``export_consolidation_preview`` truncated ``Docs/duplicate_preview.json``
  first and then streamed into it, so a failure mid-write left half a file
  (Tools → Pair Review reads it); the metadata cache had the same shape, and
  a tag value json could not encode crashed the whole plan build after all
  the work had been done;
* a metadata cache that was not valid UTF-8 crashed the plan build;
* a failure writing the executor's JSON audit log was swallowed silently;
* the fuzzy near-duplicate POC reported only the distances of the links that
  chained a group together, never the far pair it contained.
"""
from __future__ import annotations

import json

import pytest

import duplicate_consolidation as dc
import duplicate_consolidation_executor as ex
import fingerprint_cache as fc
import near_duplicate_detector as nd
from _helpers.dupes_plans import config, group, plan, write

TAGS = {"artist": "Artist", "title": "Song", "album": "Album"}


def test_failed_preview_export_keeps_the_previous_preview(tmp_path):
    out = tmp_path / "duplicate_preview.json"
    out.write_text('{"previous": true}', encoding="utf-8")
    a, b = write(tmp_path / "a.flac"), write(tmp_path / "b.mp3")
    with pytest.raises(TypeError):  # fails while encoding, i.e. "mid-write"
        dc.export_consolidation_preview(
            plan(group(a, [b])), str(out), preview_settings={"note": b"bytes json cannot encode"}
        )
    assert json.loads(out.read_text(encoding="utf-8")) == {"previous": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.flac", "b.mp3", "duplicate_preview.json"]


def test_preview_export_writes_complete_json(tmp_path):
    out = tmp_path / "duplicate_preview.json"
    a, b = write(tmp_path / "a.flac"), write(tmp_path / "b.mp3")
    dc.export_consolidation_preview(plan(group(a, [b])), str(out))
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["groups"] == 1


def _tracks(tmp_path):
    a, b = write(tmp_path / "A" / "Song.flac"), write(tmp_path / "B" / "Song.mp3")
    scan = {"scan_roots": [str(tmp_path)]}
    return [
        {"path": str(a), "fingerprint": "1 2 3 4", "ext": ".flac", "tags": dict(TAGS), "discovery": scan},
        {"path": str(b), "fingerprint": "1 2 3 4", "ext": ".mp3", "tags": dict(TAGS), "discovery": scan},
    ]


def test_metadata_cache_that_is_not_utf8_is_ignored(tmp_path):
    tracks = _tracks(tmp_path)
    cache = tmp_path / "Docs" / dc.METADATA_CACHE_FILENAME
    cache.parent.mkdir()
    cache.write_bytes(b'{"version": 1, "entries": {"\xff\xfe": 1}}')
    the_plan = dc.build_consolidation_plan(tracks)  # raised UnicodeDecodeError
    assert len(the_plan.groups) == 1
    assert json.loads(cache.read_text(encoding="utf-8"))["version"] == dc.METADATA_CACHE_VERSION


def test_unencodable_metadata_cache_value_does_not_crash_the_plan(tmp_path, monkeypatch):
    tracks = _tracks(tmp_path)
    real_entry = dc._metadata_cache_entry

    def entry_with_bytes(track):
        entry = real_entry(track)
        if entry:
            entry["tags"]["comment"] = b"\x00raw bytes"
        return entry

    monkeypatch.setattr(dc, "_metadata_cache_entry", entry_with_bytes)
    the_plan = dc.build_consolidation_plan(tracks)  # raised TypeError after all the work
    assert len(the_plan.groups) == 1
    cache = tmp_path / "Docs" / dc.METADATA_CACHE_FILENAME
    assert json.loads(cache.read_text(encoding="utf-8"))["entries"]


def test_audit_json_failure_is_reported(tmp_path, monkeypatch):
    a, b = write(tmp_path / "A" / "Song.flac"), write(tmp_path / "A" / "Song.mp3")
    logs = []

    def disk_full(path, payload):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ex, "_atomic_write_json", disk_full)
    ex.execute_consolidation_plan(plan(group(a, [b])), config(tmp_path, log_callback=logs.append))
    assert any("Report write failed" in line and "execution_report.json" in line for line in logs)


def test_fingerprint_cache_stores_tags_json_cannot_encode(tmp_path):
    fc.shutdown_fingerprint_writer()
    song = write(tmp_path / "song.mp3")
    db = str(tmp_path / "fp.db")
    try:
        assert fc.store_fingerprint(str(song), db, None, "FP", tags={"comment": b"\x01bytes"}) is True
        fc.flush_fingerprint_writes(db, timeout=5)
        fc._memo_clear()
        fp, meta = fc.get_cached_fingerprint_metadata(str(song), db)
        assert fp == "FP" and "bytes" in meta["tags"]["comment"]
    finally:
        fc.shutdown_fingerprint_writer()


def test_fuzzy_groups_report_their_true_max_distance():
    base = list(range(1, 11))
    infos = {
        "/lib/A.mp3": {"fp": " ".join(map(str, base)), "title": "Song", "artist": "X", "album": "Al", "ext": ".mp3"},
        "/lib/B.mp3": {"fp": " ".join(map(str, base[:9] + [99])), "title": "Song", "artist": "X", "album": "Al", "ext": ".mp3"},
        "/lib/C.mp3": {"fp": " ".join(map(str, base[:8] + [98, 99])), "title": "Song", "artist": "X", "album": "Al", "ext": ".mp3"},
    }
    result = nd.find_near_duplicates(infos, {".mp3": 1}, 0.1, max_workers=1)
    (grp,) = result.review_groups
    assert grp.max_distance == pytest.approx(0.2)  # A↔C, not the 0.1 links
    assert "chained match" in grp.reason
