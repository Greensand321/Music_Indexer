"""HDBSCAN-based playlist clustering engine."""

from __future__ import annotations

import os
from typing import Callable

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None
try:  # pragma: no cover - optional dependency
    import hdbscan
except Exception:
    hdbscan = None

from playlist_engine_shared import build_feature_matrix


def _clamp_min_cluster_size(value: int) -> int:
    return max(5, min(value, 500))


def _clamp_min_samples(value: int, min_cluster_size: int) -> int:
    clamped = max(1, min(value, 20))
    return min(clamped, min_cluster_size)


def _clamp_epsilon(value: float) -> float:
    return max(0.0, min(value, 0.2))


class HDBSCANPlaylistEngine:
    method = "hdbscan"

    def default_params(self) -> dict:
        return {"min_cluster_size": 25}

    def _parse_params(self, params: dict, log_callback: Callable[[str], None]) -> dict:
        min_cluster_size_raw = int(params.get("min_cluster_size", params.get("num", 25) or 25))
        min_cluster_size = _clamp_min_cluster_size(min_cluster_size_raw)
        if min_cluster_size != min_cluster_size_raw:
            log_callback(f"• Clamped min_cluster_size from {min_cluster_size_raw} to {min_cluster_size}")

        extras: dict = {}
        if "min_samples" in params:
            raw_ms = int(params["min_samples"])
            clamped_ms = _clamp_min_samples(raw_ms, min_cluster_size)
            if raw_ms != clamped_ms:
                log_callback(f"• Clamped min_samples from {raw_ms} to {clamped_ms}")
            extras["min_samples"] = clamped_ms
        if "cluster_selection_epsilon" in params:
            raw_eps = float(params["cluster_selection_epsilon"])
            clamped_eps = _clamp_epsilon(raw_eps)
            if raw_eps != clamped_eps:
                log_callback(
                    f"• Clamped cluster_selection_epsilon from {raw_eps} to {clamped_eps}"
                )
            extras["cluster_selection_epsilon"] = clamped_eps

        return {"min_cluster_size": min_cluster_size, **extras}

    def cluster_only(
        self, feature_matrix: np.ndarray, params: dict, log_callback: Callable[[str], None] | None = None
    ) -> np.ndarray:
        if log_callback is None:
            log_callback = lambda msg: None

        if np is None:
            raise RuntimeError("numpy is required for HDBSCAN clustering")
        if hdbscan is None:
            raise RuntimeError("hdbscan is required for clustering")
        parsed = self._parse_params(params, log_callback)
        log_callback("⚙ Running clustering algorithm …")
        labels = hdbscan.HDBSCAN(**parsed).fit_predict(feature_matrix)
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
            raise RuntimeError("numpy is required for HDBSCAN clustering")
        if hdbscan is None:
            raise RuntimeError("hdbscan is required for clustering")
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

