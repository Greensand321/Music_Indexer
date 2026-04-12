import json
import os

from duplicate_consolidation import (
    _capture_library_state,
    _metadata_cache_key,
    build_consolidation_plan,
)


def _track_payload(
    path: str,
    *,
    tags: dict[str, object] | None,
    audio_props: dict[str, object] | None,
    artwork: list[dict[str, object]] | None = None,
):
    payload = {
        "path": path,
        "fingerprint": "1 2 3 4 5",
        "ext": ".flac",
        "bitrate": 0,
        "sample_rate": 0,
        "bit_depth": 0,
        "channels": 0,
        "codec": "",
        "container": "",
        "tags": tags or {},
        "artwork": list(artwork or []),
        "discovery": {"scan_roots": [str(path).rsplit("/", 1)[0]]},
    }
    if audio_props:
        payload.update(audio_props)
    return payload


def _write_cache(cache_path, entries):
    cache_path.write_text(json.dumps({"version": 1, "entries": entries}, indent=2))


def _cache_entry(path: str, state: dict[str, object], *, tags: dict[str, object], audio_props: dict[str, object]):
    return {
        "path": path,
        "size": state.get("size"),
        "mtime": state.get("mtime"),
        "ext": ".flac",
        "tags": tags,
        "audio_props": audio_props,
        "artwork": [{"hash": "abc", "size": 1}],
    }


def test_metadata_cache_preloads_for_unchanged_files(monkeypatch, tmp_path):
    file_a = tmp_path / "a.flac"
    file_b = tmp_path / "b.flac"
    file_a.write_text("audio-a")
    file_b.write_text("audio-b")

    docs_dir = tmp_path / "Docs"
    docs_dir.mkdir()

    state_a = _capture_library_state(str(file_a), quick=True)
    state_b = _capture_library_state(str(file_b), quick=True)
    key_a = _metadata_cache_key(str(file_a), state_a)
    key_b = _metadata_cache_key(str(file_b), state_b)

    tags = {"artist": "Artist", "title": "Song", "album": "Album"}
    audio_props = {
        "bitrate": 320000,
        "sample_rate": 44100,
        "bit_depth": 16,
        "channels": 2,
        "codec": "FLAC",
        "container": "FLAC",
    }

    cache_entries = {
        key_a: _cache_entry(str(file_a), state_a, tags=tags, audio_props=audio_props),
        key_b: _cache_entry(str(file_b), state_b, tags=tags, audio_props=audio_props),
    }
    cache_path = docs_dir / ".duplicate_metadata_cache.json"
    _write_cache(cache_path, cache_entries)

    def _fail_read(*_args, **_kwargs):
        raise AssertionError("metadata read should be skipped")

    monkeypatch.setattr("duplicate_consolidation._read_tags_and_artwork", _fail_read)
    monkeypatch.setattr("duplicate_consolidation._load_artwork_for_track", lambda _t: None)

    track_a = _track_payload(str(file_a), tags=None, audio_props=None)
    track_b = _track_payload(str(file_b), tags=None, audio_props=None)

    plan = build_consolidation_plan([track_a, track_b])
    assert plan.groups
    assert plan.groups[0].winner_current_tags.get("title") == "Song"


def test_metadata_cache_invalidates_on_change(monkeypatch, tmp_path):
    file_a = tmp_path / "a.flac"
    file_a.write_text("audio-a")

    docs_dir = tmp_path / "Docs"
    docs_dir.mkdir()

    state_a = _capture_library_state(str(file_a), quick=True)
    key_a = _metadata_cache_key(str(file_a), state_a)
    cache_entries = {
        key_a: _cache_entry(
            str(file_a),
            state_a,
            tags={"artist": "Cached", "title": "Old"},
            audio_props={
                "bitrate": 128000,
                "sample_rate": 44100,
                "bit_depth": 16,
                "channels": 2,
                "codec": "MP3",
                "container": "MP3",
            },
        )
    }
    cache_path = docs_dir / ".duplicate_metadata_cache.json"
    _write_cache(cache_path, cache_entries)

    file_a.write_text("audio-a-updated")
    updated_mtime = int(state_a.get("mtime") or 0) + 5
    os.utime(file_a, (updated_mtime, updated_mtime))

    calls = {"count": 0}

    def _fake_read(path, provided):
        calls["count"] += 1
        return (
            {"artist": "Fresh", "title": "New"},
            [],
            None,
            None,
            {
                "bitrate": 256000,
                "sample_rate": 44100,
                "bit_depth": 16,
                "channels": 2,
                "codec": "FLAC",
                "container": "FLAC",
            },
            {
                "reader_hint": "test",
                "cover_count": 0,
                "mp4_covr_missing": False,
                "sidecar_used": False,
                "cover_deferred": True,
                "source": "file",
            },
        )

    monkeypatch.setattr("duplicate_consolidation._read_tags_and_artwork", _fake_read)
    monkeypatch.setattr("duplicate_consolidation._load_artwork_for_track", lambda _t: None)

    track_a = _track_payload(str(file_a), tags=None, audio_props=None)
    build_consolidation_plan([track_a])

    assert calls["count"] == 1


def test_metadata_cache_output_parity(monkeypatch, tmp_path):
    file_a = tmp_path / "a.flac"
    file_b = tmp_path / "b.flac"
    file_a.write_text("audio-a")
    file_b.write_text("audio-b")

    docs_dir = tmp_path / "Docs"
    docs_dir.mkdir()

    tags = {"artist": "Artist", "title": "Song", "album": "Album"}
    audio_props = {
        "bitrate": 320000,
        "sample_rate": 44100,
        "bit_depth": 16,
        "channels": 2,
        "codec": "FLAC",
        "container": "FLAC",
    }

    state_a = _capture_library_state(str(file_a), quick=True)
    state_b = _capture_library_state(str(file_b), quick=True)
    cache_entries = {
        _metadata_cache_key(str(file_a), state_a): _cache_entry(
            str(file_a), state_a, tags=tags, audio_props=audio_props
        ),
        _metadata_cache_key(str(file_b), state_b): _cache_entry(
            str(file_b), state_b, tags=tags, audio_props=audio_props
        ),
    }
    cache_path = docs_dir / ".duplicate_metadata_cache.json"
    _write_cache(cache_path, cache_entries)

    monkeypatch.setattr("duplicate_consolidation._load_artwork_for_track", lambda _t: None)

    cached_plan = build_consolidation_plan(
        [
            _track_payload(str(file_a), tags=None, audio_props=None),
            _track_payload(str(file_b), tags=None, audio_props=None),
        ]
    )

    cache_path.unlink()

    def _fail_read(*_args, **_kwargs):
        raise AssertionError("metadata read should be skipped")

    monkeypatch.setattr("duplicate_consolidation._read_tags_and_artwork", _fail_read)

    direct_plan = build_consolidation_plan(
        [
            _track_payload(
                str(file_a),
                tags=tags,
                audio_props=audio_props,
                artwork=[{"hash": "abc", "size": 1}],
            ),
            _track_payload(
                str(file_b),
                tags=tags,
                audio_props=audio_props,
                artwork=[{"hash": "abc", "size": 1}],
            ),
        ]
    )

    def _digest(plan):
        groups = []
        for group in plan.groups:
            groups.append(
                {
                    "winner": group.winner_path,
                    "losers": sorted(group.losers),
                    "planned_winner_tags": group.planned_winner_tags,
                    "winner_current_tags": group.winner_current_tags,
                    "group_match_type": group.group_match_type,
                }
            )
        return groups

    assert _digest(cached_plan) == _digest(direct_plan)
