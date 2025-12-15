import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Literal

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:  # pragma: no cover - defensive for broken environments
        return False


if _module_available("numpy"):
    import numpy as np  # type: ignore
else:  # pragma: no cover - optional dependency
    np = None  # type: ignore

if _module_available("librosa"):
    import librosa  # type: ignore
else:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore

if _module_available("essentia"):
    import essentia  # type: ignore
    from essentia.standard import MonoLoader, MusicExtractor
else:  # pragma: no cover - optional dependency
    essentia = None  # type: ignore
    MonoLoader = MusicExtractor = None  # type: ignore

AudioFeatureEngine = Literal["librosa", "essentia"]


def _ensure_1d(a):
    """Return ``a`` flattened to 1-D for feature stacking."""
    if np is None:
        raise RuntimeError("numpy required for audio feature extraction")
    a = np.asarray(a)
    if a.ndim > 1:
        a = a.ravel()
    return a


def extract_audio_features(
    file_path: str, log_callback=None, engine: AudioFeatureEngine = "librosa"
) -> "np.ndarray":
    """Return a simple feature vector for ``file_path`` using the requested engine."""

    if log_callback is None:
        log_callback = lambda msg: None

    if np is None:
        raise RuntimeError("numpy required for audio feature extraction")

    if engine == "librosa":
        return _extract_audio_features_librosa(file_path, log_callback)
    if engine == "essentia":
        return _extract_audio_features_essentia(file_path, log_callback)
    raise ValueError(f"Unknown feature extraction engine: {engine}")


def _extract_audio_features_librosa(file_path: str, log_callback) -> "np.ndarray":
    if librosa is None:
        raise RuntimeError("librosa required for feature extraction (engine='librosa')")

    y, sr = librosa.load(file_path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    log_callback(
        f"   \u00b7 MFCC shape for {os.path.basename(file_path)}: {mfcc.shape}"
    )
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mean_mfcc = _ensure_1d(np.mean(mfcc, axis=1))
    std_mfcc = _ensure_1d(np.std(mfcc, axis=1))
    tempo_arr = _ensure_1d(np.array([tempo], dtype=np.float32))

    return _assemble_feature_vector(mean_mfcc, std_mfcc, tempo_arr)


def _extract_audio_features_essentia(file_path: str, log_callback) -> "np.ndarray":
    if (
        essentia is None
        or MonoLoader is None
        or MusicExtractor is None
    ):
        raise RuntimeError("Essentia required for feature extraction (engine='essentia')")

    extractor = MusicExtractor(
        lowlevelStats=["mean", "stdev"],
        rhythmStats=["mean"],
        numberMfccCoefficients=13,
    )
    features = extractor(file_path)

    mfcc_mean = _ensure_1d(np.asarray(features["lowlevel.mfcc.mean"], dtype=np.float32))
    mfcc_std = _ensure_1d(np.asarray(features["lowlevel.mfcc.stdev"], dtype=np.float32))
    tempo_arr = _ensure_1d(np.array([features["rhythm.bpm"]], dtype=np.float32))

    log_callback(
        f"   \u00b7 Essentia MFCC lengths for {os.path.basename(file_path)}: {mfcc_mean.shape}"
    )

    return _assemble_feature_vector(mfcc_mean, mfcc_std, tempo_arr)


def _assemble_feature_vector(mean_mfcc, std_mfcc, tempo_arr) -> "np.ndarray":
    vec = np.hstack([mean_mfcc, std_mfcc, tempo_arr]).astype(np.float32)

    if vec.shape[0] != 27:
        raise RuntimeError(
            f"Feature vector has wrong length {vec.shape[0]}, expected 27"
        )
    return vec


def _extract_worker(path: str, engine: AudioFeatureEngine) -> tuple[str, "np.ndarray", str | None]:
    """Process-pool friendly wrapper around :func:`extract_audio_features`."""

    try:
        return path, extract_audio_features(path, engine=engine), None
    except Exception as exc:  # pragma: no cover - defensive logging path
        zeros = np.zeros(27, dtype=np.float32) if np is not None else None
        return path, zeros, str(exc)


def _extract_features_parallel(tracks, cache, log_callback, engine: AudioFeatureEngine):
    """Fill ``cache`` for ``tracks`` using a bounded process pool."""

    feats: list["np.ndarray"] = []
    missing = [p for p in tracks if p not in cache]

    if missing:
        workers = max(1, min(multiprocessing.cpu_count(), 4))
        log_callback(
            f"⚙ Parallel feature extraction for {len(missing)} files ({workers} workers)"
        )
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_map = {
                ex.submit(_extract_worker, path, engine): path for path in missing
            }
            for done, fut in enumerate(as_completed(future_map), 1):
                path, vec, err = fut.result()
                if err:
                    log_callback(f"! Failed features for {path}: {err}")
                else:
                    log_callback(
                        f"• Extracted features {done}/{len(missing)}: {os.path.basename(path)}"
                    )
                cache[path] = vec

    for path in tracks:
        if path in cache:
            log_callback(f"• Using cached features for {os.path.basename(path)}")
        feats.append(cache[path])

    return feats, bool(missing)


def _extract_features_serial(tracks, cache, log_callback, engine: AudioFeatureEngine):
    """Sequential feature extraction (original behavior)."""

    feats: list["np.ndarray"] = []
    updated = False
    for idx, path in enumerate(tracks, 1):
        if path in cache:
            log_callback(f"• Using cached features for {os.path.basename(path)}")
        else:
            log_callback(f"• Extracting features {idx}/{len(tracks)}")
            try:
                cache[path] = extract_audio_features(path, log_callback, engine=engine)
            except Exception as e:  # pragma: no cover - defensive path
                log_callback(f"! Failed features for {path}: {e}")
                cache[path] = np.zeros(27, dtype=np.float32)
            updated = True
        feats.append(cache[path])

    return feats, updated


def cluster_tracks(
    feature_matrix: "np.ndarray",
    method: str = "kmeans",
    log_callback=None,
    engine: str = "serial",
    **kwargs,
) -> "np.ndarray":
    """Cluster a matrix of feature vectors using KMeans or HDBSCAN."""
    if log_callback is None:
        log_callback = lambda msg: None

    if method == "kmeans":
        n_clusters = int(kwargs.get("n_clusters", 5))
        log_callback(
            f"⚙ Clustering {len(feature_matrix)} tracks into {n_clusters} groups …"
        )
        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(
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
        if engine == "parallel":
            extra["core_dist_n_jobs"] = max(1, multiprocessing.cpu_count() - 1)
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
    engine: str = "serial",
    feature_engine: AudioFeatureEngine = "librosa",
) -> None:
    """Create clustered playlists for the given tracks."""
    if log_callback is None:
        log_callback = lambda msg: None

    if np is None:
        raise RuntimeError(
            "numpy required for clustered playlist feature extraction and clustering"
        )
    if feature_engine == "librosa" and librosa is None:
        raise RuntimeError(
            "librosa required for feature extraction (engine='librosa')"
        )
    if feature_engine == "essentia" and (
        essentia is None or MonoLoader is None or MusicExtractor is None
    ):
        raise RuntimeError(
            "Essentia required for feature extraction (engine='essentia')"
        )

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

    engine_mode = "serial" if engine in (None, "librosa", "serial") else "parallel"

    log_callback(
        f"⚙ Extracting audio features with {engine_mode} engine ({feature_engine}) …"
    )

    if engine_mode == "parallel":
        feats, updated = _extract_features_parallel(
            tracks, cache, log_callback, feature_engine
        )
    else:
        feats, updated = _extract_features_serial(
            tracks, cache, log_callback, feature_engine
        )

    if updated:
        np.save(cache_file, cache)
        log_callback(f"✓ Saved feature cache ({len(cache)} entries) to {cache_file}")

    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)

    labels = cluster_tracks(
        X, method, log_callback=log_callback, engine=engine_mode, **params
    )
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
    return feats
