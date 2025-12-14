import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan
from playlist_generator import DEFAULT_EXTS


def _ensure_1d(a):
    """Return ``a`` flattened to 1-D for feature stacking."""
    a = np.asarray(a)
    if a.ndim > 1:
        a = a.ravel()
    return a


def extract_audio_features(file_path: str, log_callback=None) -> np.ndarray:
    """Return a simple feature vector for ``file_path`` using librosa."""
    if log_callback is None:
        log_callback = lambda msg: None

    y, sr = librosa.load(file_path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Log raw MFCC shape for debugging
    log_callback(
        f"   \u00b7 MFCC shape for {os.path.basename(file_path)}: {mfcc.shape}"
    )
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mean_mfcc = _ensure_1d(np.mean(mfcc, axis=1))
    std_mfcc = _ensure_1d(np.std(mfcc, axis=1))
    tempo_arr = _ensure_1d(np.array([tempo], dtype=np.float32))

    vec = np.hstack([mean_mfcc, std_mfcc, tempo_arr]).astype(np.float32)

    # Validate length and ensure we have exactly 27 values
    if vec.shape[0] != 27:
        raise RuntimeError(
            f"Feature vector has wrong length {vec.shape[0]}, expected 27"
        )
    return vec


def cluster_tracks(
    feature_matrix: np.ndarray,
    method: str = "kmeans",
    log_callback=None,
    **kwargs,
) -> np.ndarray:
    """Cluster a matrix of feature vectors using KMeans or HDBSCAN."""
    if log_callback is None:
        log_callback = lambda msg: None

    if method == "kmeans":
        n_clusters = int(kwargs.get("n_clusters", 5))
        log_callback(
            f"⚙ Clustering {len(feature_matrix)} tracks into {n_clusters} groups …"
        )
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(
            feature_matrix
        )
    else:
        min_cluster_size = int(kwargs.get("min_cluster_size", 5))
        extra: dict = {}
        if "min_samples" in kwargs:
            extra["min_samples"] = int(kwargs["min_samples"])
        if "cluster_selection_epsilon" in kwargs:
            extra["cluster_selection_epsilon"] = float(
                kwargs["cluster_selection_epsilon"]
            )
        log_callback("⚙ Running clustering algorithm …")
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, **extra
        ).fit_predict(feature_matrix)

    log_callback("✓ Clustering complete")
    return labels


def generate_clustered_playlists(
    tracks,
    root_path: str,
    method: str,
    params: dict,
    log_callback=None,
    engine: str = "librosa",
) -> None:
    """Create clustered playlists for the given tracks."""
    if log_callback is None:
        log_callback = lambda msg: None

    # ------------------------------------------------------------------
    # Feature cache setup
    # ------------------------------------------------------------------
    docs = os.path.join(root_path, "Docs")
    os.makedirs(docs, exist_ok=True)
    cache_file = os.path.join(docs, "features.npy")

    try:
        cache = dict(np.load(cache_file, allow_pickle=True).item())
        log_callback(f"→ Loaded {len(cache)} cached feature vectors")
    except FileNotFoundError:
        cache = {}
        log_callback("→ No feature cache found; extracting all tracks")

    log_callback(f"⚙ Extracting audio features with {engine}…")

    feats = []
    updated = False
    for idx, path in enumerate(tracks, 1):
        if path in cache:
            log_callback(f"• Using cached features for {os.path.basename(path)}")
        else:
            log_callback(f"• Extracting features {idx}/{len(tracks)}")
            try:
                cache[path] = extract_audio_features(path, log_callback)
            except Exception as e:
                log_callback(f"! Failed features for {path}: {e}")
                cache[path] = np.zeros(27, dtype=np.float32)
            updated = True
        feats.append(cache[path])

    if updated:
        np.save(cache_file, cache)
        log_callback(f"✓ Saved feature cache ({len(cache)} entries) to {cache_file}")

    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)

    labels = cluster_tracks(X, method, log_callback=log_callback, **params)
    log_callback(
        f"✓ Clustering complete: found {len(set([l for l in labels if l >= 0]) )} clusters"
    )

    playlists_dir = os.path.join(root_path, "Playlists")
    os.makedirs(playlists_dir, exist_ok=True)

    for cluster_id in sorted(set(labels)):
        if cluster_id < 0:
            continue
        playlist = [tracks[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        outfile = os.path.join(playlists_dir, f"{method}_cluster_{cluster_id}.m3u")
        try:
            with open(outfile, "w", encoding="utf-8") as pf:
                for p in playlist:
                    pf.write(os.path.relpath(p, playlists_dir) + "\n")
            log_callback(f"→ Writing clustered playlist: {outfile}")
        except Exception as e:
            log_callback(f"\u2717 Failed to write {outfile}: {e}")

    log_callback("✓ Clustered playlist generation finished")
    return X
