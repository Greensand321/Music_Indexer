import hashlib

import duplicate_consolidation
from duplicate_consolidation import build_consolidation_plan


def test_normalize_track_uses_preprovided_metadata(monkeypatch, tmp_path) -> None:
    def fail_read(*_args, **_kwargs):
        raise AssertionError("_read_tags_and_artwork should not be called")

    monkeypatch.setattr(duplicate_consolidation, "_read_tags_and_artwork", fail_read)

    track_path = tmp_path / "Album" / "Song A.flac"
    raw = {
        "path": str(track_path),
        "fingerprint": "1 2 3",
        "ext": ".flac",
        "bitrate": 320000,
        "sample_rate": 44100,
        "channels": 2,
        "bit_depth": 16,
        "codec": "FLAC",
        "container": "FLAC",
        "tags": {
            "artist": "Artist",
            "title": "Title",
            "tracknumber": "4/10",
            "discnumber": "1/2",
            "date": "2020-01-01",
        },
        "artwork": [{"bytes": b"art", "status": "ok"}],
    }

    normalized = duplicate_consolidation._normalize_track(raw, quick_state=True)

    assert normalized.current_tags["track"] == 4
    assert normalized.current_tags["disc"] == 1
    assert normalized.current_tags["year"] == "2020"
    assert normalized.trace["metadata_read"]["source"] == "provided"
    assert normalized.trace["album_art"]["source"] == "provided"
    assert normalized.artwork
    assert normalized.artwork[0].hash == hashlib.sha256(b"art").hexdigest()


def test_normalize_track_falls_back_when_incomplete(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_read(path: str, _provided_tags):
        calls.append(path)
        return (
            {"artist": "Fallback", "title": "Fallback"},
            [],
            None,
            "deferred",
            {
                "bitrate": 256000,
                "sample_rate": 44100,
                "bit_depth": 16,
                "channels": 2,
                "codec": "MP3",
                "container": "MP3",
            },
            {
                "reader_hint": "fake",
                "cover_count": None,
                "mp4_covr_missing": False,
                "sidecar_used": False,
                "cover_deferred": True,
                "source": "file",
            },
        )

    monkeypatch.setattr(duplicate_consolidation, "_read_tags_and_artwork", fake_read)

    track_path = tmp_path / "Song B.flac"
    raw = {
        "path": str(track_path),
        "fingerprint": "1 2 3",
        "ext": ".flac",
        "bitrate": 320000,
        "sample_rate": 44100,
        "channels": 2,
        "tags": {"title": "Title Only"},
    }

    duplicate_consolidation._normalize_track(raw, quick_state=True)

    assert calls == [str(track_path)]


def test_duplicate_detection_consistent_with_preprovided_metadata(monkeypatch, tmp_path) -> None:
    def fake_read(_path: str, _provided_tags):
        return (
            {"artist": "Artist", "title": "Title"},
            [],
            None,
            "deferred",
            {
                "bitrate": 320000,
                "sample_rate": 44100,
                "bit_depth": 16,
                "channels": 2,
                "codec": "FLAC",
                "container": "FLAC",
            },
            {
                "reader_hint": "fake",
                "cover_count": None,
                "mp4_covr_missing": False,
                "sidecar_used": False,
                "cover_deferred": True,
                "source": "file",
            },
        )

    monkeypatch.setattr(duplicate_consolidation, "_read_tags_and_artwork", fake_read)

    path_a = tmp_path / "Song A.flac"
    path_b = tmp_path / "Song B.flac"

    tracks_preprovided = [
        {
            "path": str(path_a),
            "fingerprint": "1 2 3 4",
            "ext": ".flac",
            "bitrate": 320000,
            "sample_rate": 44100,
            "channels": 2,
            "bit_depth": 16,
            "codec": "FLAC",
            "container": "FLAC",
            "tags": {"artist": "Artist", "title": "Title"},
        },
        {
            "path": str(path_b),
            "fingerprint": "1 2 3 4",
            "ext": ".flac",
            "bitrate": 320000,
            "sample_rate": 44100,
            "channels": 2,
            "bit_depth": 16,
            "codec": "FLAC",
            "container": "FLAC",
            "tags": {"artist": "Artist", "title": "Title"},
        },
    ]

    tracks_fallback = [
        {
            "path": str(path_a),
            "fingerprint": "1 2 3 4",
            "ext": ".flac",
            "bitrate": 320000,
            "sample_rate": 44100,
            "channels": 2,
            "tags": {"title": "Title"},
        },
        {
            "path": str(path_b),
            "fingerprint": "1 2 3 4",
            "ext": ".flac",
            "bitrate": 320000,
            "sample_rate": 44100,
            "channels": 2,
            "tags": {"title": "Title"},
        },
    ]

    plan_preprovided = build_consolidation_plan(tracks_preprovided)
    plan_fallback = build_consolidation_plan(tracks_fallback)

    assert len(plan_preprovided.groups) == 1
    assert len(plan_fallback.groups) == 1
    assert plan_preprovided.groups[0].winner_path == plan_fallback.groups[0].winner_path
    assert set(plan_preprovided.groups[0].losers) == set(plan_fallback.groups[0].losers)
