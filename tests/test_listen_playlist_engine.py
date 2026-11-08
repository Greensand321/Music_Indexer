"""playlist_engine edge cases (listening backends audit).

Tempo/energy bucketing: NaN analysis results, duplicates, empty input.
Auto-DJ / more-like-this: an unreadable track no longer aborts the run, a
track listed twice is used once, non-finite or mismatched vectors are left
out, Cancel works during the analysis, the result is deterministic and the
chain always terminates with every track used at most once.
"""
from __future__ import annotations

import os
import random
import threading

import numpy as np
import pytest

import playlist_engine
from indexer_control import IndexCancelled


def _touch(root, rel: str) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


@pytest.fixture
def fake_tempo(monkeypatch):
    values: dict[str, tuple] = {}
    calls: list[str] = []

    def compute(path, engine="librosa"):
        calls.append(os.path.basename(path))
        return values[os.path.basename(path)]

    monkeypatch.setattr(playlist_engine, "librosa", object())
    monkeypatch.setattr(playlist_engine, "compute_tempo_energy", compute)
    return values, calls


# ── tempo / energy buckets ────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [(float("nan"), 0.2), (100.0, float("nan")), (float("inf"), 0.2), (None, 0.2)])
def test_non_finite_analysis_goes_in_no_bucket_and_is_not_cached(tmp_path, fake_tempo, bad):
    values, _calls = fake_tempo
    a = _touch(tmp_path, "A/a.mp3")
    b = _touch(tmp_path, "A/b.mp3")
    values.update({"a.mp3": bad, "b.mp3": (100.0, 0.2)})

    result = playlist_engine.bucket_by_tempo_energy([a, b], str(tmp_path))

    # NaN compares false with everything, so it used to land in fast/high.
    assert result["buckets"] == {("medium", "medium"): [b]}
    assert result["failed"] == [a]
    with np.load(tmp_path / "Docs" / "tempo_energy.npy", allow_pickle=False) as data:
        assert list(data["paths"]) == [os.path.abspath(b)]


def test_a_cached_nan_entry_is_analysed_again(tmp_path, fake_tempo):
    values, calls = fake_tempo
    a = _touch(tmp_path, "A/a.mp3")
    values["a.mp3"] = (float("nan"), 0.2)
    playlist_engine.bucket_by_tempo_energy([a], str(tmp_path))
    # An older version cached the NaN; simulate that cache.
    playlist_engine._save_tempo_cache(
        str(tmp_path / "Docs" / "tempo_energy.npy"),
        {os.path.abspath(a): {"engine": "librosa", "tempo": float("nan"), "rms": 0.2}},
    )
    values["a.mp3"] = (130.0, 0.4)
    calls.clear()
    result = playlist_engine.bucket_by_tempo_energy([a], str(tmp_path))
    assert calls == ["a.mp3"]
    assert result["buckets"] == {("fast", "high"): [a]}


def test_a_track_listed_twice_is_bucketed_once(tmp_path, fake_tempo):
    values, calls = fake_tempo
    a = _touch(tmp_path, "A/a.mp3")
    values["a.mp3"] = (100.0, 0.2)
    result = playlist_engine.bucket_by_tempo_energy([a, a, os.path.join(str(tmp_path), "A", ".", "a.mp3")], str(tmp_path))
    assert result["buckets"] == {("medium", "medium"): [a]}
    assert calls == ["a.mp3"]
    assert result["processed"] == 3 and result["total"] == 3


def test_empty_input_and_identical_values(tmp_path, fake_tempo):
    values, _calls = fake_tempo
    empty = playlist_engine.bucket_by_tempo_energy([], str(tmp_path))
    assert empty["buckets"] == {} and empty["failed"] == [] and empty["processed"] == 0
    tracks = [_touch(tmp_path, f"A/{i}.mp3") for i in range(3)]
    values.update({f"{i}.mp3": (120.0, 0.3) for i in range(3)})
    result = playlist_engine.bucket_by_tempo_energy(tracks, str(tmp_path))
    assert result["buckets"] == {("fast", "high"): tracks}


# ── Auto-DJ / more like this ──────────────────────────────────────────────


@pytest.fixture
def fake_features(monkeypatch):
    """Features from a table; "broken" raises like an undecodable file."""
    import clustered_playlists

    table: dict[str, object] = {}
    calls: list[str] = []

    def extract(path, log_callback=None, engine="librosa"):
        calls.append(path)
        value = table[path]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(clustered_playlists, "extract_audio_features", extract)
    return table, calls


def test_an_unreadable_track_is_skipped_not_fatal(fake_features):
    table, _calls = fake_features
    table.update({
        "a": [0.0, 0.0], "b": [1.0, 1.0], "broken": RuntimeError("cannot decode"),
        "nan": [float("nan"), 0.0], "short": [0.5], "c": [2.0, 2.0],
    })
    logs: list[str] = []
    order = playlist_engine.autodj_playlist(
        "a", ["a", "b", "broken", "nan", "short", "c"], n=10, log_callback=logs.append
    )
    assert order == ["a", "b", "c"]
    assert any("broken" in m and "cannot decode" in m for m in logs)
    assert any("nan" in m for m in logs) and any("short" in m for m in logs)
    assert playlist_engine.more_like_this("a", ["broken", "c", "b"], n=5) == ["b", "c"]


def test_an_unreadable_seed_is_one_clear_error(fake_features):
    table, _calls = fake_features
    table.update({"seed": OSError("gone"), "b": [1.0]})
    with pytest.raises(RuntimeError, match="seed track"):
        playlist_engine.autodj_playlist("seed", ["seed", "b"], n=5)


def test_each_track_is_used_once_and_the_chain_terminates(fake_features):
    table, _calls = fake_features
    rng = random.Random(7)
    names = [f"t{i}" for i in range(40)]
    for name in names:
        table[name] = [rng.random(), rng.random(), rng.random()]
    listed = names + names[:10]  # duplicates in the input
    order = playlist_engine.autodj_playlist("t0", listed, n=1000)
    assert sorted(order) == sorted(names)
    assert order[0] == "t0"
    # Deterministic.
    assert playlist_engine.autodj_playlist("t0", listed, n=1000) == order
    assert playlist_engine.autodj_playlist("t0", listed, n=5) == order[:5]
    assert playlist_engine.autodj_playlist("t0", listed, n=0) == []


def test_ties_go_to_the_track_listed_first(fake_features):
    table, _calls = fake_features
    table.update({"s": [0.0], "x": [1.0], "y": [-1.0], "z": [1.0]})
    assert playlist_engine.autodj_playlist("s", ["s", "x", "y", "z"], n=2) == ["s", "x"]
    assert playlist_engine.autodj_playlist("s", ["s", "y", "x", "z"], n=2) == ["s", "y"]


def test_cancel_stops_the_analysis_of_a_large_library(monkeypatch):
    import clustered_playlists

    stop = threading.Event()
    calls: list[str] = []

    def extract(path, log_callback=None, engine="librosa"):
        calls.append(path)
        if len(calls) == 10:
            stop.set()  # the user clicks Cancel while track 10 is analysed
        return [float(len(calls))]

    monkeypatch.setattr(clustered_playlists, "extract_audio_features", extract)
    with pytest.raises(IndexCancelled):
        playlist_engine.autodj_playlist("t0", [f"t{i}" for i in range(200)], n=20, cancel_event=stop)
    # It used to analyse all 200 tracks before the first cancel check.
    assert len(calls) == 10


def test_a_callers_empty_feature_cache_is_filled_in_place(fake_features):
    table, calls = fake_features
    table.update({"a": [0.0], "b": [1.0]})
    cache: dict = {}
    playlist_engine.autodj_playlist("a", ["a", "b"], n=2, feature_cache=cache)
    assert set(cache) == {"a", "b"}
    calls.clear()
    playlist_engine.autodj_playlist("a", ["a", "b"], n=2, feature_cache=cache)
    assert calls == [], "the second run reuses the caller's cache"
