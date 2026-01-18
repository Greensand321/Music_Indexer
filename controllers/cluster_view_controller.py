"""Caching helpers for interactive clustering views."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class ClusterComputationManager:
    """Cache PCA projections and clustering labels for interactive panels."""

    def __init__(self, tracks: list[str], features: list, log_callback: Callable[[str], None]):
        self.tracks = tracks
        self.features = features
        self.log_callback = log_callback

        self._X: np.ndarray | None = None
        self._X_key: str | None = None
        self._pca_cache: dict[tuple[str, tuple], np.ndarray] = {}
        self._labels_cache: dict[tuple[str, str, tuple], np.ndarray] = {}
        self._lock = threading.Lock()

    # ─── Dataset Helpers ──────────────────────────────────────────────────
    def _materialize_features(self) -> np.ndarray:
        if self._X is not None:
            return self._X

        start = time.perf_counter()
        self._X = np.vstack(self.features)
        duration = (time.perf_counter() - start) * 1000
        logger.info("[perf] materialize features in %.1f ms", duration)
        return self._X

    def dataset_key(self) -> str:
        if self._X_key:
            return self._X_key

        X = self._materialize_features().astype(np.float32, copy=False)
        start = time.perf_counter()
        digest = hashlib.sha1(X.tobytes()).hexdigest()
        duration = (time.perf_counter() - start) * 1000
        self._X_key = digest
        logger.info("[perf] dataset fingerprint in %.1f ms", duration)
        return digest

    # ─── PCA Caching ──────────────────────────────────────────────────────
    def get_projection(self, pca_params: tuple = (2,)) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, X_2d) using PCA with caching."""

        key = (self.dataset_key(), pca_params)
        with self._lock:
            if key in self._pca_cache:
                logger.info("[perf] PCA cache hit for key=%s", key)
                return self._materialize_features(), self._pca_cache[key]

        start = time.perf_counter()
        X = self._materialize_features()
        pca = PCA(n_components=pca_params[0])
        X2 = pca.fit_transform(X)
        duration = (time.perf_counter() - start) * 1000
        logger.info("[perf] PCA compute in %.1f ms", duration)

        with self._lock:
            self._pca_cache[key] = X2
        return X, X2

    # ─── Clustering Cache ─────────────────────────────────────────────────
    def get_cached_labels(self, algo: str, params: dict) -> np.ndarray | None:
        key = (self.dataset_key(), algo, tuple(sorted(params.items())))
        with self._lock:
            labels = self._labels_cache.get(key)
        if labels is not None:
            logger.info("[perf] clustering cache hit for %s params=%s", algo, params)
        return labels

    def compute_labels(self, algo: str, params: dict, cluster_func: Callable) -> np.ndarray:
        key = (self.dataset_key(), algo, tuple(sorted(params.items())))
        with self._lock:
            if key in self._labels_cache:
                logger.info("[perf] clustering cache hit for %s params=%s", algo, params)
                return self._labels_cache[key]

        start = time.perf_counter()
        labels = cluster_func(self._materialize_features(), params)
        duration = (time.perf_counter() - start) * 1000
        logger.info("[perf] clustering compute for %s in %.1f ms", algo, duration)

        with self._lock:
            self._labels_cache[key] = labels
        return labels
