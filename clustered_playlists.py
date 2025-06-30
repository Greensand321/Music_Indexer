import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan
from playlist_generator import DEFAULT_EXTS


def extract_features(file_path: str) -> np.ndarray:
    """Return a feature vector for the given audio file."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    vec = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        [tempo],
    ])
    return vec.astype(np.float32)


def cluster_tracks(feature_matrix: np.ndarray, method: str = "kmeans", **kwargs) -> np.ndarray:
    """Cluster a matrix of feature vectors using KMeans or HDBSCAN."""
    if method == "kmeans":
        n_clusters = int(kwargs.get("n_clusters", 5))
        km = KMeans(n_clusters=n_clusters)
        labels = km.fit_predict(feature_matrix)
    else:
        min_cluster_size = int(kwargs.get("min_cluster_size", 5))
        db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = db.fit_predict(feature_matrix)
    return labels


def generate_clustered_playlists(tracks, root_path: str, method: str, params: dict, log_callback=None) -> None:
    """Create clustered playlists for the given tracks."""
    if log_callback is None:
        log_callback = lambda msg: None

    feats = []
    for idx, path in enumerate(tracks, 1):
        log_callback(f"\u2022 Extracting features {idx}/{len(tracks)}")
        try:
            feats.append(extract_features(path))
        except Exception as e:
            log_callback(f"! Failed features for {path}: {e}")
            feats.append(np.zeros(27, dtype=np.float32))

    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)
    labels = cluster_tracks(X, method, **params)

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
            log_callback(f"\u2192 Writing clustered playlist: {outfile}")
        except Exception as e:
            log_callback(f"\u2717 Failed to write {outfile}: {e}")
