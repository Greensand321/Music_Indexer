import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from playlist_engine_kmeans import KMeansPlaylistEngine
from playlist_engine_hdbscan import HDBSCANPlaylistEngine


def _stub_feature_loader(mapping):
    def _load(path, log_callback):
        return np.asarray(mapping[path], dtype=float)

    return _load


def _read_playlist(outfile: Path) -> set[str]:
    return set(outfile.read_text().splitlines())


def test_kmeans_engine_regression(tmp_path):
    tracks = [tmp_path / f"track{i}.mp3" for i in range(3)]
    for p in tracks:
        p.write_text("x")

    feats = {
        str(tracks[0]): [0.0, 0.0],
        str(tracks[1]): [0.1, 0.1],
        str(tracks[2]): [4.0, 4.2],
    }

    engine = KMeansPlaylistEngine()
    logs: list[str] = []
    X = engine.generate(
        [str(t) for t in tracks],
        str(tmp_path),
        {"n_clusters": 2},
        logs.append,
        feature_loader=_stub_feature_loader(feats),
    )

    assert X.shape == (3, 2)
    playlists_dir = tmp_path / "Playlists"
    outputs = sorted(playlists_dir.glob("kmeans_cluster_*.m3u"))
    assert len(outputs) == 2

    groups = [_read_playlist(p) for p in outputs]
    expected = {
        {os.path.relpath(tracks[0], playlists_dir), os.path.relpath(tracks[1], playlists_dir)},
        {os.path.relpath(tracks[2], playlists_dir)},
    }
    assert set(map(frozenset, groups)) == set(map(frozenset, expected))
    assert any("Clustering" in line for line in logs)


def test_hdbscan_engine_independent(monkeypatch, tmp_path):
    tracks = [tmp_path / f"song{i}.mp3" for i in range(4)]
    for p in tracks:
        p.write_text("x")

    feats = {
        str(tracks[0]): [0.0, 0.0],
        str(tracks[1]): [0.2, 0.1],
        str(tracks[2]): [5.0, 5.1],
        str(tracks[3]): [5.2, 5.0],
    }

    captured_kwargs = {}

    class DummyHDB:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def fit_predict(self, X):  # noqa: N802  - mimic real signature
            return np.array([0, 0, -1, 1])

    monkeypatch.setattr("playlist_engine_hdbscan.hdbscan.HDBSCAN", DummyHDB)

    engine = HDBSCANPlaylistEngine()
    logs: list[str] = []
    X = engine.generate(
        [str(t) for t in tracks],
        str(tmp_path),
        {"min_cluster_size": 10, "min_samples": 50, "cluster_selection_epsilon": 0.5},
        logs.append,
        feature_loader=_stub_feature_loader(feats),
    )

    assert X.shape == (4, 2)
    assert captured_kwargs == {"min_cluster_size": 10, "min_samples": 10, "cluster_selection_epsilon": 0.2}

    playlists_dir = tmp_path / "Playlists"
    outputs = sorted(playlists_dir.glob("hdbscan_cluster_*.m3u"))
    assert len(outputs) == 2
    groups = [_read_playlist(p) for p in outputs]
    expected = {
        {os.path.relpath(tracks[0], playlists_dir), os.path.relpath(tracks[1], playlists_dir)},
        {os.path.relpath(tracks[3], playlists_dir)},
    }
    assert set(map(frozenset, groups)) == set(map(frozenset, expected))
    assert any("Running clustering" in line for line in logs)
