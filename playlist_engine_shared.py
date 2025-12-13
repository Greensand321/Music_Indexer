"""Shared, read-only helpers for playlist clustering engines."""

from __future__ import annotations

import os
from typing import Callable

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None
try:  # pragma: no cover - optional dependency
    import librosa
except Exception:
    librosa = None


def ensure_1d(a):
    """Return ``a`` flattened to 1-D for feature stacking."""

    a = np.asarray(a)
    if a.ndim > 1:
        a = a.ravel()
    return a


def extract_audio_features(file_path: str, log_callback=None) -> np.ndarray:
    """Return a simple feature vector for ``file_path`` using librosa."""

    if log_callback is None:
        log_callback = lambda msg: None

    if librosa is None or np is None:
        raise RuntimeError("librosa and numpy are required for feature extraction")

    y, sr = librosa.load(file_path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    log_callback(f"   · MFCC shape for {os.path.basename(file_path)}: {mfcc.shape}")
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mean_mfcc = ensure_1d(np.mean(mfcc, axis=1))
    std_mfcc = ensure_1d(np.std(mfcc, axis=1))
    tempo_arr = ensure_1d(np.array([tempo], dtype=np.float32))

    vec = np.hstack([mean_mfcc, std_mfcc, tempo_arr]).astype(np.float32)
    if vec.shape[0] != 27:
        raise RuntimeError(f"Feature vector has wrong length {vec.shape[0]}, expected 27")
    return vec


def build_feature_matrix(
    tracks: list[str],
    root_path: str,
    log_callback: Callable[[str], None],
    *,
    feature_loader: Callable[[str, Callable[[str], None]], np.ndarray] | None = None,
) -> np.ndarray:
    """Return standardized feature matrix for ``tracks`` using ``feature_loader``.

    ``feature_loader`` defaults to :func:`extract_audio_features` and can be
    overridden in tests to avoid heavyweight audio processing.
    """

    if feature_loader is None:
        feature_loader = extract_audio_features

    if np is None:
        raise RuntimeError("numpy is required for feature extraction")

    docs = os.path.join(root_path, "Docs")
    os.makedirs(docs, exist_ok=True)
    cache_file = os.path.join(docs, "features.npy")

    try:
        cache = dict(np.load(cache_file, allow_pickle=True).item())
        log_callback(f"→ Loaded {len(cache)} cached feature vectors")
    except FileNotFoundError:
        cache = {}
        log_callback("→ No feature cache found; extracting all tracks")

    feats = []
    updated = False
    for idx, path in enumerate(tracks, 1):
        if path in cache:
            log_callback(f"• Using cached features for {os.path.basename(path)}")
        else:
            log_callback(f"• Extracting features {idx}/{len(tracks)}")
            try:
                cache[path] = feature_loader(path, log_callback)
            except Exception as e:
                log_callback(f"! Failed features for {path}: {e}")
                cache[path] = np.zeros(27, dtype=np.float32)
            updated = True
        feats.append(cache[path])

    if updated:
        np.save(cache_file, cache)
        log_callback(f"✓ Saved feature cache ({len(cache)} entries) to {cache_file}")

    from sklearn.preprocessing import StandardScaler

    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)
    return X
