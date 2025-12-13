"""KMeans-based playlist clustering engine."""

from __future__ import annotations

import os
from typing import Callable

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None
try:  # pragma: no cover - optional dependency
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

from playlist_engine_shared import build_feature_matrix


class KMeansPlaylistEngine:
    method = "kmeans"

    def default_params(self) -> dict:
        return {"n_clusters": 5}

    def cluster_only(
        self, feature_matrix: np.ndarray, params: dict, log_callback: Callable[[str], None] | None = None
    ) -> np.ndarray:
        if log_callback is None:
            log_callback = lambda msg: None

        if np is None:
            raise RuntimeError("numpy is required for KMeans clustering")
        if KMeans is None:
            raise RuntimeError("scikit-learn is required for KMeans clustering")

        n_clusters = int(params.get("n_clusters", params.get("num", 5) or 5))
        log_callback(f"⚙ Clustering {len(feature_matrix)} tracks into {n_clusters} groups …")
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(feature_matrix)
        log_callback("✓ Clustering complete")
        return labels

    def generate(
        self,
        tracks: list[str],
        root_path: str,
        params: dict,
        log_callback: Callable[[str], None],
        *,
        engine: str = "librosa",
        feature_loader=None,
    ) -> np.ndarray:
        log_callback(f"⚙ Extracting audio features with {engine}…")
        if np is None:
            raise RuntimeError("numpy is required for KMeans clustering")
        if KMeans is None:
            raise RuntimeError("scikit-learn is required for KMeans clustering")
        X = build_feature_matrix(tracks, root_path, log_callback, feature_loader=feature_loader)
        labels = self.cluster_only(X, params, log_callback)
        self._write_playlists(tracks, labels, root_path, log_callback)
        return X

    def _write_playlists(
        self,
        tracks: list[str],
        labels: np.ndarray,
        root_path: str,
        log_callback: Callable[[str], None],
    ) -> None:
        playlists_dir = os.path.join(root_path, "Playlists")
        os.makedirs(playlists_dir, exist_ok=True)

        for cluster_id in sorted(set(labels)):
            if cluster_id < 0:
                continue
            playlist = [tracks[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
            outfile = os.path.join(playlists_dir, f"{self.method}_cluster_{cluster_id}.m3u")
            try:
                with open(outfile, "w", encoding="utf-8") as pf:
                    for p in playlist:
                        pf.write(os.path.relpath(p, playlists_dir) + "\n")
                log_callback(f"→ Writing clustered playlist: {outfile}")
            except Exception as e:
                log_callback(f"✗ Failed to write {outfile}: {e}")

