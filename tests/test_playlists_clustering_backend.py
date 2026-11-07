"""Regression tests for clustered_playlists (PL-04, PL-09, PL-16, PL-17, PL-28 and
the index-alignment rule from CLAUDE.md).

librosa, hdbscan and umap are not needed: feature extraction is faked by
monkeypatching ``extract_audio_features`` and the embeddings are replaced by
cheap deterministic ones.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import clustered_playlists as cp

REPO = Path(__file__).resolve().parents[1]


def _tracks(root: Path, n: int) -> list[str]:
    out = []
    for i in range(n):
        p = root / f"Artist {i % 3}" / f"track_{i:03d}.mp3"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0")
        out.append(str(p))
    return out


@pytest.fixture
def fake_engine(monkeypatch):
    """Fake the heavy parts: librosa present, extraction returns index-derived vectors."""
    calls: list[str] = []
    failing: set[str] = set()

    def fake_extract(path, log_callback=None, engine="librosa", features=None):
        calls.append(path)
        if path in failing:
            raise RuntimeError("decode error")
        idx = int(Path(path).stem.split("_")[1])
        dim = cp.feature_vector_length(features)
        return np.full(dim, float(idx % 4) + idx * 0.001, dtype=np.float32)

    monkeypatch.setattr(cp, "librosa", object())
    monkeypatch.setattr(cp, "extract_audio_features", fake_extract)
    monkeypatch.setattr(cp, "compute_2d_embedding", lambda X, log=None: np.asarray(X)[:, :2].copy())
    monkeypatch.setattr(cp, "compute_3d_embedding", lambda X, log=None: np.asarray(X)[:, :3].copy())
    # The 3-D HTML generator is exercised in its own tests.
    import cluster_graph_3d
    monkeypatch.setattr(cluster_graph_3d, "generate_cluster_graph_html_from_data", lambda *a, **k: None)
    # Process pools can't see monkeypatched functions; a thread pool has the
    # same API (including shutdown(cancel_futures=...)).
    monkeypatch.setattr(cp, "ProcessPoolExecutor", ThreadPoolExecutor)
    return calls, failing


def _run(root, tracks, **kw):
    kw.setdefault("method", "kmeans")
    kw.setdefault("params", {"n_clusters": 2})
    kw.setdefault("engine", "serial")
    return cp.generate_clustered_playlists(tracks, str(root), **kw)


# ── PL-17: hdbscan is optional ───────────────────────────────────────────


def test_module_imports_without_hdbscan():
    """K-Means clustering and Auto-DJ must not need hdbscan installed."""
    code = (
        "import sys; sys.modules['hdbscan'] = None\n"
        "import clustered_playlists, playlist_engine\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_hdbscan_missing_gives_a_friendly_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "hdbscan", None)
    with pytest.raises(RuntimeError, match="HDBSCAN clustering needs the optional 'hdbscan'"):
        cp.cluster_tracks(np.zeros((6, 2)), "hdbscan")


def test_kmeans_runs_without_hdbscan(monkeypatch):
    monkeypatch.setitem(sys.modules, "hdbscan", None)
    X = np.array([[0, 0], [0, 0.1], [5, 5], [5, 5.1]], dtype=float)
    labels = cp.cluster_tracks(X, "kmeans", n_clusters=2)
    assert len(set(labels)) == 2


# ── PL-04: method spelling ───────────────────────────────────────────────


@pytest.mark.parametrize("spelling", ["kmeans", "K-Means", "k-means", "KMEANS"])
def test_kmeans_spellings_run_kmeans(spelling):
    assert cp.normalize_method(spelling) == "kmeans"


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown clustering method"):
        cp.cluster_tracks(np.zeros((4, 2)), "dbscan")


# ── PL-16: failures are not cached as zero vectors ───────────────────────


def test_failed_tracks_are_excluded_and_not_cached(tmp_path, fake_engine):
    calls, failing = fake_engine
    tracks = _tracks(tmp_path, 8)
    failing.add(tracks[3])

    result = _run(tmp_path, tracks)
    assert tracks[3] not in result["tracks"]
    assert tracks[3] in result["failed"]
    assert len(result["tracks"]) == len(result["labels"]) == len(result["X"]) == 7

    payload = json.loads((tmp_path / "Docs" / "cluster_info.json").read_text())
    assert tracks[3] not in payload["tracks"]
    assert len(payload["tracks"]) == len(payload["labels"]) == len(payload["X_2d"]) == 7

    # Second run: the failure is retried (not served from cache as zeros).
    failing.clear()
    calls.clear()
    result2 = _run(tmp_path, tracks)
    assert calls == [tracks[3]]
    assert tracks[3] in result2["tracks"]


def test_all_failures_raise(tmp_path, fake_engine):
    _calls, failing = fake_engine
    tracks = _tracks(tmp_path, 3)
    failing.update(tracks)
    with pytest.raises(RuntimeError, match="failed for every track"):
        _run(tmp_path, tracks)
    assert not (tmp_path / "Docs" / "cluster_info.json").exists()


# ── PL-09: cancellation reaches the extraction loop ──────────────────────


@pytest.mark.parametrize("engine", ["serial", "parallel"])
def test_cancel_during_extraction_writes_no_artefacts(tmp_path, fake_engine, monkeypatch, engine):
    calls, _failing = fake_engine
    tracks = _tracks(tmp_path, 40)
    cancel = threading.Event()
    real = cp.extract_audio_features

    def slow_extract(path, *a, **k):
        if len(calls) >= 5:
            cancel.set()
        return real(path, *a, **k)

    monkeypatch.setattr(cp, "extract_audio_features", slow_extract)
    with pytest.raises(cp.ClusteringCancelled):
        _run(tmp_path, tracks, engine=engine, cancel_event=cancel)

    assert len(calls) < len(tracks), "extraction kept going after cancel"
    assert not (tmp_path / "Docs" / "cluster_info.json").exists()
    # Work done before the cancel is kept, so the next run resumes.
    cache = cp._load_feature_cache(str(tmp_path / "Docs" / cp.feature_cache_name(None)))
    assert 0 < len(cache) < len(tracks)


def test_progress_callback_reports_per_file(tmp_path, fake_engine):
    tracks = _tracks(tmp_path, 5)
    seen = []
    _run(tmp_path, tracks, progress_callback=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (5, 5)
    assert len(seen) >= 5


# ── PL-28: caches load without pickle and survive corruption ─────────────


def test_feature_cache_round_trips_without_pickle(tmp_path):
    f = tmp_path / "features.npy"
    cache = {"/a.mp3": np.arange(3, dtype=np.float32), "/b.mp3": np.ones(3, dtype=np.float32)}
    cp._save_feature_cache(str(f), cache)
    loaded = cp._load_feature_cache(str(f))
    assert set(loaded) == set(cache)
    np.testing.assert_array_equal(loaded["/a.mp3"], cache["/a.mp3"])
    # Readable with pickle disabled.
    with np.load(f, allow_pickle=False) as data:
        assert list(data["paths"]) == ["/a.mp3", "/b.mp3"]


def test_legacy_pickled_cache_is_ignored_not_unpickled(tmp_path):
    f = tmp_path / "features.npy"
    np.save(f, {"/a.mp3": np.zeros(3)}, allow_pickle=True)
    logs = []
    assert cp._load_feature_cache(str(f), logs.append) == {}
    assert any("Ignoring unreadable feature cache" in m for m in logs)


def test_corrupt_cache_is_rebuilt(tmp_path, fake_engine):
    tracks = _tracks(tmp_path, 4)
    docs = tmp_path / "Docs"
    docs.mkdir()
    (docs / cp.feature_cache_name(None)).write_bytes(b"garbage, not numpy")
    result = _run(tmp_path, tracks)
    assert len(result["tracks"]) == 4
    assert len(cp._load_feature_cache(str(docs / cp.feature_cache_name(None)))) == 4


# ── CLAUDE.md: payload arrays stay index-aligned when downsampling ───────


def test_payload_downsampling_keeps_arrays_aligned():
    n = 12_000
    X = np.arange(n, dtype=float).reshape(-1, 1) * np.ones((1, 4))
    labels = np.arange(n) % 7
    tracks = [f"/m/{i}.mp3" for i in range(n)]
    payload = cp.build_cluster_payload(X, X[:, :2], X[:, :3], labels, tracks, {})
    assert payload["X_downsampled"] is True
    assert payload["X_total_points"] == n
    kept = payload["X_indices"]
    assert len(kept) == cp.MAX_VISUALIZATION_POINTS
    assert len(payload["X"]) == len(payload["X_2d"]) == len(payload["X_3d"]) == len(
        payload["labels"]
    ) == len(payload["tracks"]) == len(kept)
    for pos in (0, 1, 2500, len(kept) - 1):
        i = kept[pos]
        assert payload["tracks"][pos] == f"/m/{i}.mp3"
        assert payload["labels"][pos] == i % 7
        assert payload["X_2d"][pos][0] == float(i)
        assert payload["X_3d"][pos][2] == float(i)


def test_payload_between_limit_and_twice_limit_really_downsamples():
    """The old step-based subsetting kept all 7,000 points while logging that it downsampled."""
    n = 7_000
    X = np.zeros((n, 3))
    payload = cp.build_cluster_payload(X, X[:, :2], X, np.zeros(n), [str(i) for i in range(n)], {})
    assert len(payload["tracks"]) == cp.MAX_VISUALIZATION_POINTS


def test_payload_below_limit_is_untouched():
    X = np.zeros((10, 3))
    payload = cp.build_cluster_payload(X, X[:, :2], X, np.zeros(10), [str(i) for i in range(10)], {})
    assert payload["X_downsampled"] is False
    assert payload["X_indices"] is None
    assert len(payload["tracks"]) == 10


def test_payload_rejects_misaligned_input():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="index-aligned"):
        cp.build_cluster_payload(X, X[:9, :2], X, np.zeros(10), [str(i) for i in range(10)], {})


def test_written_payload_loads_through_the_graph_data_layer(tmp_path, fake_engine):
    import cluster_graph_data as cgd

    tracks = _tracks(tmp_path, 9)
    _run(tmp_path, tracks, params={"n_clusters": 3})
    data = cgd.load_cluster_data(tmp_path)
    xy, labels, out_tracks = cgd.graph_points(data)
    assert len(xy) == len(labels) == len(out_tracks) == 9
    assert not os.path.exists(tmp_path / "Docs" / "cluster_info.json.tmp")
