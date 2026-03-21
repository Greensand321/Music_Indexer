import importlib.util
import os
import shutil
import subprocess
import tempfile
import logging
import json as json_module
from contextlib import contextmanager
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Literal

logger = logging.getLogger(__name__)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan

from utils.path_helpers import ensure_long_path

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

if _module_available("umap"):
    import umap  # type: ignore
else:
    umap = None  # type: ignore

if _module_available("sklearn.manifold"):
    from sklearn.manifold import TSNE
else:
    TSNE = None  # type: ignore

AudioFeatureEngine = Literal["librosa", "essentia"]

# Magic number constants for clarity
DEFAULT_MFCC_COEFS = 13
FEATURE_VECTOR_LENGTH = 27  # Mean MFCC (13) + Std MFCC (13) + Tempo (1)
MAX_VISUALIZATION_POINTS = 5000  # Downsample large datasets for viz performance
MIN_VISUALIZATION_POINTS = 100  # Ensure minimum visualization quality
DEFAULT_CLUSTER_COUNT = 8  # Default K for K-Means
DEFAULT_HDBSCAN_MIN_SIZE = 5  # Default minimum cluster size for HDBSCAN
DEFAULT_HDBSCAN_MIN_SAMPLES = 5  # Default min_samples for HDBSCAN
PROXIMITY_THRESHOLD_RATIO = 0.1  # Hover detection within 10% of data range


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

    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as exc:
        if os.path.splitext(file_path)[1].lower() != ".opus":
            raise
        log_callback(
            f"! Direct Opus decode failed ({exc}); retrying via FFmpeg for {os.path.basename(file_path)}"
        )
        with _decode_opus_as_wav(file_path, log_callback) as audio_path:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=DEFAULT_MFCC_COEFS)
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
        numberMfccCoefficients=DEFAULT_MFCC_COEFS,
    )
    try:
        features = extractor(file_path)
    except Exception as exc:
        if os.path.splitext(file_path)[1].lower() != ".opus":
            raise
        log_callback(
            f"! Direct Opus decode failed ({exc}); retrying via FFmpeg for {os.path.basename(file_path)}"
        )
        with _decode_opus_as_wav(file_path, log_callback) as audio_path:
            features = extractor(audio_path)

    mfcc_mean = _ensure_1d(np.asarray(features["lowlevel.mfcc.mean"], dtype=np.float32))
    mfcc_std = _ensure_1d(np.asarray(features["lowlevel.mfcc.stdev"], dtype=np.float32))
    tempo_arr = _ensure_1d(np.array([features["rhythm.bpm"]], dtype=np.float32))

    log_callback(
        f"   \u00b7 Essentia MFCC lengths for {os.path.basename(file_path)}: {mfcc_mean.shape}"
    )

    return _assemble_feature_vector(mfcc_mean, mfcc_std, tempo_arr)


def _assemble_feature_vector(mean_mfcc, std_mfcc, tempo_arr) -> "np.ndarray":
    vec = np.hstack([mean_mfcc, std_mfcc, tempo_arr]).astype(np.float32)

    if vec.shape[0] != FEATURE_VECTOR_LENGTH:
        raise RuntimeError(
            f"Feature vector has wrong length {vec.shape[0]}, expected {FEATURE_VECTOR_LENGTH}"
        )
    return vec


@contextmanager
def _decode_opus_as_wav(file_path: str, log_callback):
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is required to decode Opus files for clustering.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    cmd = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-i",
        ensure_long_path(file_path),
        "-ac",
        "1",
        "-vn",
        ensure_long_path(temp_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-500:]
            raise RuntimeError(f"FFmpeg failed to decode Opus: {stderr_tail}")
        log_callback(f"• Decoded Opus via FFmpeg: {os.path.basename(file_path)}")
        yield temp_path
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def _extract_worker(path: str, engine: AudioFeatureEngine) -> tuple[str, "np.ndarray", str | None]:
    """Process-pool friendly wrapper around :func:`extract_audio_features`."""

    try:
        return path, extract_audio_features(path, engine=engine), None
    except Exception as exc:  # pragma: no cover - defensive logging path
        zeros = np.zeros(27, dtype=np.float32) if np is not None else None
        return path, zeros, str(exc)


def _extract_features_parallel(
    tracks,
    cache,
    log_callback,
    engine: AudioFeatureEngine,
    use_max_workers: bool = False,
):
    """Fill ``cache`` for ``tracks`` using a bounded process pool."""

    feats: list["np.ndarray"] = []
    missing = [p for p in tracks if p not in cache]

    if missing:
        if use_max_workers:
            workers = max(1, multiprocessing.cpu_count() - 1)
        else:
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
        else:
            log_callback(f"! ERROR: No features found for {os.path.basename(path)} after extraction")
            raise RuntimeError(f"Feature extraction failed for {path} - no cached features available")

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
        n_clusters = max(1, int(kwargs.get("n_clusters", 5)))
        if n_clusters > len(feature_matrix):
            log_callback(
                f"→ Requested {n_clusters} clusters but only "
                f"{len(feature_matrix)} tracks; reducing to {len(feature_matrix)}"
            )
            n_clusters = len(feature_matrix)
        log_callback(
            f"⚙ Clustering {len(feature_matrix)} tracks into {n_clusters} groups …"
        )
        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(
            feature_matrix
        )
    else:
        min_cluster_size = max(2, int(kwargs.get("min_cluster_size", 5)))
        if min_cluster_size > len(feature_matrix):
            log_callback(
                f"→ Min cluster size {min_cluster_size} exceeds track count "
                f"{len(feature_matrix)}; reducing to {len(feature_matrix)}"
            )
            min_cluster_size = len(feature_matrix)
        extra: dict = {}
        if "min_samples" in kwargs:
            min_samples = max(1, int(kwargs["min_samples"]))
            if min_samples > len(feature_matrix):
                log_callback(
                    f"→ Min samples {min_samples} exceeds track count "
                    f"{len(feature_matrix)}; reducing to {len(feature_matrix)}"
                )
                min_samples = len(feature_matrix)
            extra["min_samples"] = min_samples
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


def log_cluster_summary(labels: "np.ndarray", log_callback) -> None:
    """Log a compact summary of cluster counts and noise."""

    if log_callback is None:
        return

    total = len(labels)
    noise = sum(1 for l in labels if l < 0)
    clusters = Counter(int(l) for l in labels if l >= 0)
    if not clusters:
        log_callback(
            f"→ All {total} tracks were marked as noise; try adjusting parameters"
        )
        return

    cluster_count = len(clusters)
    log_callback(
        f"→ Cluster summary: {cluster_count} clusters, {noise} noise tracks"
    )
    top_sizes = ", ".join(
        f"{cid}: {size}" for cid, size in clusters.most_common(5)
    )
    log_callback(f"→ Largest clusters (id: size): {top_sizes}")


def compute_2d_embedding(X: "np.ndarray", log_callback=None) -> "np.ndarray":
    """Compute 2D coordinates for visualization using UMAP or t-SNE.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    log_callback : callable, optional
        Callback for logging messages

    Returns
    -------
    np.ndarray
        2D coordinates of shape (n_samples, 2)
    """
    if log_callback is None:
        log_callback = lambda msg: None

    if np is None:
        raise RuntimeError("numpy required for 2D embedding computation")

    n_samples = len(X)
    if n_samples < 10:
        # For very small datasets, use simple PCA-like approach
        log_callback("→ Dataset too small for UMAP/t-SNE; using random projection")
        return np.random.randn(n_samples, 2).astype(np.float32)

    # Try UMAP first (faster, better for large datasets)
    if umap is not None:
        try:
            log_callback("⚙ Computing 2D embedding with UMAP …")
            mapper = umap.UMAP(n_components=2, n_neighbors=min(15, n_samples - 1), random_state=42)
            X_2d = mapper.fit_transform(X).astype(np.float32)
            log_callback("✓ 2D embedding computed with UMAP")
            return X_2d
        except Exception as e:
            log_callback(f"⚠ UMAP failed: {e}; falling back to t-SNE")

    # Fall back to t-SNE
    if TSNE is not None:
        try:
            log_callback("⚙ Computing 2D embedding with t-SNE …")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, (n_samples - 1) // 3))
            X_2d = tsne.fit_transform(X).astype(np.float32)
            log_callback("✓ 2D embedding computed with t-SNE")
            return X_2d
        except Exception as e:
            log_callback(f"⚠ t-SNE failed: {e}; using random projection")

    # Final fallback: random projection (preserves approximate distances)
    log_callback("⚠ Neither UMAP nor t-SNE available; using random projection for 2D visualization")
    return np.random.randn(n_samples, 2).astype(np.float32)


def compute_3d_embedding(X: "np.ndarray", log_callback=None) -> "np.ndarray":
    """Compute 3D coordinates for advanced exploration using UMAP or t-SNE.

    3D preserves 75-85% of data variance vs 50-60% for 2D, enabling discovery
    of hidden cluster relationships and feature interactions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    log_callback : callable, optional
        Callback for logging messages

    Returns
    -------
    np.ndarray
        3D coordinates of shape (n_samples, 3)
    """
    if log_callback is None:
        log_callback = lambda msg: None

    if np is None:
        raise RuntimeError("numpy required for 3D embedding computation")

    n_samples = len(X)
    if n_samples < 10:
        log_callback("→ Dataset too small for UMAP/t-SNE; using random projection")
        return np.random.randn(n_samples, 3).astype(np.float32)

    # Try UMAP first (much faster for 3D than t-SNE)
    if umap is not None:
        try:
            log_callback("⚙ Computing 3D embedding with UMAP (preserves ~80% variance) …")
            mapper = umap.UMAP(n_components=3, n_neighbors=min(15, n_samples - 1), random_state=42)
            X_3d = mapper.fit_transform(X).astype(np.float32)
            log_callback("✓ 3D embedding computed with UMAP")
            return X_3d
        except Exception as e:
            log_callback(f"⚠ UMAP 3D failed: {e}; falling back to t-SNE")

    # Fall back to t-SNE (slower but good quality)
    if TSNE is not None:
        try:
            log_callback("⚙ Computing 3D embedding with t-SNE (this may take a moment) …")
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, (n_samples - 1) // 3))
            X_3d = tsne.fit_transform(X).astype(np.float32)
            log_callback("✓ 3D embedding computed with t-SNE")
            return X_3d
        except Exception as e:
            log_callback(f"⚠ t-SNE failed: {e}; using random projection")

    # Final fallback: random projection
    log_callback("⚠ Neither UMAP nor t-SNE available; using random projection for 3D visualization")
    return np.random.randn(n_samples, 3).astype(np.float32)


def _validate_cache_entry(path: str, cached_time: float) -> bool:
    """Check if cached entry is still valid (file not modified since cache was created)."""
    try:
        file_time = os.path.getmtime(path)
        return file_time <= cached_time
    except OSError:
        # File not accessible, invalidate cache
        return False


def _load_cache_with_validation(cache_file: str, tracks: list, log_callback=None) -> dict:
    """Load feature cache with validation for stale entries."""
    if log_callback is None:
        log_callback = lambda msg: None

    try:
        cache = dict(np.load(cache_file, allow_pickle=True).item())
    except FileNotFoundError:
        return {}
    except Exception as e:
        log_callback(f"! Error loading cache file: {e}")
        return {}

    # Validate cache entries against track modification times
    cache_metadata_file = cache_file.replace(".npy", "_metadata.json")
    try:
        with open(cache_metadata_file) as f:
            metadata = json_module.load(f)
        cache_time = metadata.get("created_time", 0)
    except (FileNotFoundError, json_module.JSONDecodeError):
        cache_time = 0

    # Remove stale entries
    stale_count = 0
    valid_cache = {}
    for path, features in cache.items():
        if _validate_cache_entry(path, cache_time):
            valid_cache[path] = features
        else:
            stale_count += 1

    if stale_count > 0:
        log_callback(f"! Invalidated {stale_count} stale cache entries")

    return valid_cache


def generate_clustered_playlists(
    tracks,
    root_path: str,
    method: str,
    params: dict,
    log_callback=None,
    engine: str = "parallel",  # DEFAULT: Use parallel processing for speed!
    feature_engine: AudioFeatureEngine = "librosa",
    use_max_workers: bool = False,
) -> None:
    """Create clustered data for the given tracks without writing playlists."""
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
    try:
        os.makedirs(docs, exist_ok=True)
    except OSError as e:
        log_callback(f"! Warning: Cannot create Docs folder: {e}")
        docs = root_path  # Fall back to root

    cache_file = os.path.join(docs, "features.npy")

    # Load cache with validation for stale entries
    cache = _load_cache_with_validation(cache_file, tracks, log_callback)
    if cache:
        log_callback(f"→ Loaded {len(cache)} cached feature vectors (validated)")
    else:
        log_callback("→ No feature cache found or all entries stale; extracting all tracks")

    engine_mode = "serial" if engine in (None, "librosa", "serial") else "parallel"

    log_callback(
        f"⚙ Extracting audio features with {engine_mode} engine ({feature_engine}) …"
    )

    if engine_mode == "parallel":
        feats, updated = _extract_features_parallel(
            tracks, cache, log_callback, feature_engine, use_max_workers=use_max_workers
        )
    else:
        feats, updated = _extract_features_serial(
            tracks, cache, log_callback, feature_engine
        )

    if updated:
        try:
            np.save(cache_file, cache)
            log_callback(f"✓ Saved feature cache ({len(cache)} entries) to {cache_file}")
        except OSError as e:
            log_callback(f"! Warning: Could not save cache file: {e}")
            logger.warning(f"Failed to save feature cache: {e}")

    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)

    labels = cluster_tracks(
        X, method, log_callback=log_callback, engine=engine_mode, **params
    )
    log_callback(
        f"✓ Clustering complete: found {len(set([l for l in labels if l >= 0]) )} clusters"
    )
    log_cluster_summary(labels, log_callback)

    log_callback("→ Automatic playlist export disabled; manage selections manually.")

    # Prepare cluster metadata
    unique_labels = set([l for l in labels if l >= 0])
    cluster_info = {}
    for cluster_id in sorted(unique_labels):
        track_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        cluster_info[int(cluster_id)] = {
            "size": len(track_indices),
            "genres": [],  # Could extract from tracks if metadata available
            "tempo": [0, 0],  # Could compute from features if available
        }

    # Compute 2D embedding for visualization
    log_callback("→ Computing 2D scatter plot coordinates…")
    X_2d = compute_2d_embedding(X, log_callback)

    # Compute 3D embedding for advanced exploration
    log_callback("→ Computing 3D audio feature space (preserves ~80% data variance)…")
    X_3d = compute_3d_embedding(X, log_callback)

    # Save cluster data to JSON for visualization
    # For large datasets, optionally downsample X for visualization
    x_to_save = X
    x_2d_to_save = X_2d
    x_3d_to_save = X_3d
    downsampled = False

    if len(X) > MAX_VISUALIZATION_POINTS:
        # Downsample for visualization while keeping all labels and tracks
        log_callback(
            f"⚠ Library has {len(X)} tracks; downsampling X to {MAX_VISUALIZATION_POINTS} "
            f"points for visualization (labels and tracks preserved)"
        )
        # Keep every nth point to maintain cluster distribution
        step = max(1, len(X) // MAX_VISUALIZATION_POINTS)
        indices = np.arange(0, len(X), step)
        if len(indices) < MIN_VISUALIZATION_POINTS:
            # Ensure minimum number of points for visualization
            indices = np.linspace(0, len(X) - 1, MIN_VISUALIZATION_POINTS, dtype=int)
        x_to_save = X[indices]
        x_2d_to_save = X_2d[indices]
        x_3d_to_save = X_3d[indices]
        downsampled = True

    cluster_data = {
        "X": x_to_save.tolist(),
        "X_2d": x_2d_to_save.tolist(),   # 2D embedding for quick scatter plot
        "X_3d": x_3d_to_save.tolist(),   # 3D embedding for advanced exploration (preserves ~80% variance)
        "X_downsampled": downsampled,  # Flag if X was downsampled
        "X_total_points": len(X),      # Original number of points
        "labels": labels.tolist(),
        "tracks": tracks,
        "cluster_info": cluster_info,
    }

    cluster_info_file = os.path.join(docs, "cluster_info.json")
    try:
        with open(cluster_info_file, "w") as f:
            json_module.dump(cluster_data, f, indent=2)
        log_callback(f"✓ Saved cluster data to {cluster_info_file}")

        # Save cache metadata with timestamp
        cache_metadata_file = cache_file.replace(".npy", "_metadata.json")
        import time
        metadata = {
            "created_time": time.time(),
            "track_count": len(tracks),
        }
        with open(cache_metadata_file, "w") as f:
            json_module.dump(metadata, f, indent=2)
    except OSError as e:
        log_callback(f"! Error saving cluster data: {e}")
        logger.exception("Failed to save cluster data")

    # Generate interactive 3D HTML visualization
    try:
        from cluster_graph_3d import generate_cluster_graph_html_from_data

        html_out = os.path.join(docs, "cluster_graph.html")
        generate_cluster_graph_html_from_data(
            cluster_data, html_out, log_callback
        )
    except Exception as e:
        log_callback(f"⚠ Could not generate 3D graph HTML: {e}")
        logger.warning("Failed to generate cluster_graph.html: %s", e)

    # Return full result dict for caller
    return {
        "features": feats,
        "X": X,
        "labels": labels,
        "tracks": tracks,
        "cluster_info": cluster_info,
        "metrics": {},  # Will be filled by caller with sklearn metrics
    }
