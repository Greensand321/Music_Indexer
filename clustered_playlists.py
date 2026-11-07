import importlib.util
import os
import shutil
import subprocess
import tempfile
import logging
import json as json_module
from contextlib import contextmanager
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait as _wait_futures
import multiprocessing
import time
from typing import Literal

logger = logging.getLogger(__name__)

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# hdbscan is imported lazily, inside the HDBSCAN branch of cluster_tracks():
# importing it here made K-Means clustering and Auto-DJ (which only needs
# extract_audio_features) fail with "No module named 'hdbscan'" on installs
# that never use HDBSCAN.

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
FEATURE_VECTOR_LENGTH = 27  # Legacy default: mean MFCC (13) + std MFCC (13) + tempo (1)

# ── Selectable audio features ────────────────────────────────────────────────
# The clustering wizard lets the user pick which qualities of the sound to group
# on. Blocks are always assembled in this order so a vector's meaning is fixed by
# the selection alone, never by dict iteration order.
FEATURE_ORDER = ("mfcc", "tempo", "chroma", "spectral", "energy", "onset_rate")

#: How many numbers each block contributes to the feature vector.
FEATURE_DIMENSIONS = {
    "mfcc": 2 * DEFAULT_MFCC_COEFS,  # mean + standard deviation per coefficient
    "tempo": 1,                      # beats per minute
    "chroma": 12,                    # mean energy per pitch class (harmonic content)
    "spectral": 2,                   # spectral centroid mean + std (brightness)
    "energy": 2,                     # RMS loudness mean + std
    "onset_rate": 1,                 # note onsets per second (percussive density)
}

#: Feature scalers the clustering wizard can choose between. Robust scaling uses
#: the median and interquartile range, so a handful of outlier tracks pull the
#: rest of the library around far less than with standard scaling.
SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

#: What callers get when they do not ask for anything specific. Matches the
#: historical 27-dimension vector so existing caches and callers (Auto-DJ, the
#: older clustering entry points) keep working unchanged.
DEFAULT_FEATURES = {name: name in ("mfcc", "tempo") for name in FEATURE_ORDER}


def normalize_feature_selection(features: dict | None = None) -> dict:
    """Return a complete, validated ``{feature: bool}`` selection.

    ``None`` yields the legacy MFCC+tempo set. Unknown keys are ignored rather
    than raising, so a wizard that grows a checkbox before the engine supports
    it degrades to "not selected" instead of crashing a clustering run.
    """
    if features is None:
        return dict(DEFAULT_FEATURES)
    selection = {name: bool(features.get(name, False)) for name in FEATURE_ORDER}
    if not any(selection.values()):
        raise ValueError(
            "at least one audio feature must be selected for clustering"
        )
    return selection


def feature_vector_length(features: dict | None = None) -> int:
    """Return the length of the vector produced by *features*."""
    selection = normalize_feature_selection(features)
    return sum(FEATURE_DIMENSIONS[name] for name in FEATURE_ORDER if selection[name])


def feature_selection_key(features: dict | None = None) -> str:
    """Return a short, stable identifier for a feature selection."""
    selection = normalize_feature_selection(features)
    return "-".join(name for name in FEATURE_ORDER if selection[name])


def feature_cache_name(
    features: dict | None = None, feature_engine: str = "librosa"
) -> str:
    """Return the cache filename for a given selection and extraction engine.

    Cached vectors are only interchangeable with vectors built from the *same*
    features by the *same* engine — mixing them would stack arrays of different
    widths (or, worse, the same width with different meanings). Keying the file
    name on both means switching selections re-extracts once and then reuses
    each set thereafter, instead of invalidating everything on every toggle.

    The legacy MFCC+tempo/librosa combination keeps the original
    ``features.npy`` name so existing caches are still picked up.
    """
    key = feature_selection_key(features)
    if key == feature_selection_key(None) and feature_engine == "librosa":
        return "features.npy"
    return f"features_{feature_engine}_{key}.npy"
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
    file_path: str,
    log_callback=None,
    engine: AudioFeatureEngine = "librosa",
    features: dict | None = None,
) -> "np.ndarray":
    """Return the feature vector for ``file_path`` using the requested engine.

    ``features`` selects which blocks to compute (see :data:`FEATURE_ORDER`);
    omitting it keeps the historical MFCC+tempo vector. Only the selected
    blocks are computed, so unticking an expensive one \u2014 onset detection in
    particular \u2014 genuinely saves the work rather than just hiding the result.
    """

    if log_callback is None:
        log_callback = lambda msg: None

    if np is None:
        raise RuntimeError("numpy required for audio feature extraction")

    selection = normalize_feature_selection(features)

    if engine == "librosa":
        return _extract_audio_features_librosa(file_path, log_callback, selection)
    if engine == "essentia":
        return _extract_audio_features_essentia(file_path, log_callback, selection)
    raise ValueError(f"Unknown feature extraction engine: {engine}")


def _extract_audio_features_librosa(
    file_path: str, log_callback, selection: dict
) -> "np.ndarray":
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

    blocks: dict[str, "np.ndarray"] = {}

    if selection["mfcc"]:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=DEFAULT_MFCC_COEFS)
        log_callback(
            f"   \u00b7 MFCC shape for {os.path.basename(file_path)}: {mfcc.shape}"
        )
        blocks["mfcc"] = np.hstack(
            [_ensure_1d(np.mean(mfcc, axis=1)), _ensure_1d(np.std(mfcc, axis=1))]
        )

    if selection["tempo"]:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        blocks["tempo"] = _ensure_1d(np.array([tempo], dtype=np.float32))

    if selection["chroma"]:
        # chroma_stft rather than chroma_cqt: both describe pitch-class energy,
        # but the STFT variant is markedly faster and does not fail on very
        # short or unusual inputs, which matters when sweeping a whole library.
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        blocks["chroma"] = _ensure_1d(np.mean(chroma, axis=1))

    if selection["spectral"]:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        blocks["spectral"] = np.array(
            [float(np.mean(centroid)), float(np.std(centroid))], dtype=np.float32
        )

    if selection["energy"]:
        rms = librosa.feature.rms(y=y)
        blocks["energy"] = np.array(
            [float(np.mean(rms)), float(np.std(rms))], dtype=np.float32
        )

    if selection["onset_rate"]:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration = len(y) / float(sr) if sr else 0.0
        rate = (len(onsets) / duration) if duration > 0 else 0.0
        blocks["onset_rate"] = np.array([rate], dtype=np.float32)

    return _assemble_feature_vector(blocks, selection)


def _essentia_descriptor(features, key: str, size: int) -> "np.ndarray":
    """Pull *key* from an Essentia result, failing loudly if it is absent.

    Descriptor names drift between Essentia versions. Zero-filling a missing one
    would quietly cluster on a constant \u2014 worse than stopping \u2014 so this reports
    exactly which descriptor is missing and points at the engine that does have
    it.
    """
    try:
        value = features[key]
    except Exception as exc:  # noqa: BLE001 - Essentia raises its own key errors
        raise RuntimeError(
            f"Essentia did not provide '{key}'. This build may not expose that "
            "descriptor; switch the feature engine to librosa, or deselect the "
            "feature that needs it."
        ) from exc
    arr = _ensure_1d(np.asarray(value, dtype=np.float32))
    if arr.shape[0] != size:
        raise RuntimeError(
            f"Essentia descriptor '{key}' has length {arr.shape[0]}, expected {size}"
        )
    return arr


def _extract_audio_features_essentia(
    file_path: str, log_callback, selection: dict
) -> "np.ndarray":
    if (
        essentia is None
        or MonoLoader is None
        or MusicExtractor is None
    ):
        raise RuntimeError("Essentia required for feature extraction (engine='essentia')")

    extractor = MusicExtractor(
        lowlevelStats=["mean", "stdev"],
        rhythmStats=["mean"],
        tonalStats=["mean"],
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

    blocks: dict[str, "np.ndarray"] = {}

    if selection["mfcc"]:
        mfcc_mean = _essentia_descriptor(features, "lowlevel.mfcc.mean", DEFAULT_MFCC_COEFS)
        mfcc_std = _essentia_descriptor(features, "lowlevel.mfcc.stdev", DEFAULT_MFCC_COEFS)
        log_callback(
            f"   \u00b7 Essentia MFCC lengths for {os.path.basename(file_path)}: {mfcc_mean.shape}"
        )
        blocks["mfcc"] = np.hstack([mfcc_mean, mfcc_std])

    if selection["tempo"]:
        blocks["tempo"] = _essentia_descriptor(features, "rhythm.bpm", 1)

    if selection["chroma"]:
        # HPCP is Essentia's pitch-class profile \u2014 the counterpart to librosa's
        # chroma.
        blocks["chroma"] = _essentia_descriptor(features, "tonal.hpcp.mean", 12)

    if selection["spectral"]:
        blocks["spectral"] = np.hstack([
            _essentia_descriptor(features, "lowlevel.spectral_centroid.mean", 1),
            _essentia_descriptor(features, "lowlevel.spectral_centroid.stdev", 1),
        ])

    if selection["energy"]:
        blocks["energy"] = np.hstack([
            _essentia_descriptor(features, "lowlevel.spectral_energy.mean", 1),
            _essentia_descriptor(features, "lowlevel.spectral_energy.stdev", 1),
        ])

    if selection["onset_rate"]:
        blocks["onset_rate"] = _essentia_descriptor(features, "rhythm.onset_rate", 1)

    return _assemble_feature_vector(blocks, selection)


def _assemble_feature_vector(blocks: dict, selection: dict) -> "np.ndarray":
    """Stack the computed blocks in :data:`FEATURE_ORDER` and check the width."""
    parts = [blocks[name] for name in FEATURE_ORDER if selection.get(name)]
    vec = np.hstack(parts).astype(np.float32)

    expected = feature_vector_length(selection)
    if vec.shape[0] != expected:
        raise RuntimeError(
            f"Feature vector has wrong length {vec.shape[0]}, expected {expected} "
            f"for features [{feature_selection_key(selection)}]"
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


class ClusteringCancelled(Exception):
    """Raised when a clustering run is cancelled through its ``cancel_event``.

    Nothing is written to ``Docs/cluster_info.json`` / ``cluster_graph.html``
    for a cancelled run; feature vectors extracted before the cancel are kept
    in the cache so the next run resumes instead of starting over.
    """


def _is_cancelled(cancel_event) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _extract_worker(
    path: str, engine: AudioFeatureEngine, features: dict | None = None
) -> tuple[str, "np.ndarray | None", str | None]:
    """Process-pool friendly wrapper around :func:`extract_audio_features`.

    A failure returns ``(path, None, error)``. It used to return a zero vector,
    which was then cached and clustered forever as a fake "silent" track.
    """

    try:
        return path, extract_audio_features(path, engine=engine, features=features), None
    except Exception as exc:  # pragma: no cover - defensive logging path
        return path, None, str(exc)


def _collect_extracted(tracks, cache, failed: dict, log_callback):
    """Return ``(kept_tracks, feats)`` for the tracks that have a vector.

    Failed tracks are left out of both lists together, so the feature matrix
    stays index-aligned with the track list it is clustered against.
    """
    kept: list[str] = []
    feats: list["np.ndarray"] = []
    for path in tracks:
        if path in cache:
            kept.append(path)
            feats.append(cache[path])
        elif path not in failed:
            failed[path] = "no features available after extraction"
            log_callback(f"! No features found for {os.path.basename(path)}; leaving it out")
    return kept, feats


def _extract_features_parallel(
    tracks,
    cache,
    log_callback,
    engine: AudioFeatureEngine,
    use_max_workers: bool = False,
    features: dict | None = None,
    cancel_event=None,
    progress_callback=None,
):
    """Fill ``cache`` for ``tracks`` using a bounded process pool.

    Returns ``(kept_tracks, feats, updated, failed)``. ``cancel_event`` is
    polled while waiting on the pool; on cancel, queued work is dropped
    (``cancel_futures``), files already being decoded are allowed to finish,
    and :class:`ClusteringCancelled` is raised.
    """

    failed: dict[str, str] = {}
    missing = [p for p in tracks if p not in cache]
    cached_count = len(tracks) - len(missing)
    if cached_count:
        log_callback(f"• Using cached features for {cached_count} track(s)")
    if progress_callback is not None:
        progress_callback(cached_count, len(tracks))

    updated = False
    if missing:
        if use_max_workers:
            workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            workers = max(1, min(multiprocessing.cpu_count(), 4))
        log_callback(
            f"⚙ Parallel feature extraction for {len(missing)} files ({workers} workers)"
        )
        ex = ProcessPoolExecutor(max_workers=workers)
        cancelled = False
        try:
            pending = {
                ex.submit(_extract_worker, path, engine, features)
                for path in missing
            }
            done_count = 0
            while pending:
                if _is_cancelled(cancel_event):
                    cancelled = True
                    # Keep results that are already in, so a re-run resumes.
                    done = {f for f in pending if f.done() and not f.cancelled()}
                    pending = set()
                else:
                    done, pending = _wait_futures(
                        pending, timeout=0.25, return_when=FIRST_COMPLETED
                    )
                for fut in done:
                    path, vec, err = fut.result()
                    done_count += 1
                    if err or vec is None:
                        failed[path] = err or "unknown error"
                        log_callback(f"! Failed features for {path}: {err}")
                    else:
                        cache[path] = vec
                        updated = True
                        log_callback(
                            f"• Extracted features {done_count}/{len(missing)}: {os.path.basename(path)}"
                        )
                    if progress_callback is not None:
                        progress_callback(cached_count + done_count, len(tracks))
        finally:
            # cancel_futures drops everything still queued; wait=True only
            # waits for the handful of files the workers are decoding right
            # now, so no child process is orphaned.
            ex.shutdown(wait=True, cancel_futures=cancelled)
        if cancelled:
            raise ClusteringCancelled()

    kept, feats = _collect_extracted(tracks, cache, failed, log_callback)
    return kept, feats, updated, failed


def _extract_features_serial(
    tracks,
    cache,
    log_callback,
    engine: AudioFeatureEngine,
    features: dict | None = None,
    cancel_event=None,
    progress_callback=None,
):
    """Sequential feature extraction. Returns ``(kept, feats, updated, failed)``."""

    failed: dict[str, str] = {}
    updated = False
    for idx, path in enumerate(tracks, 1):
        if _is_cancelled(cancel_event):
            raise ClusteringCancelled()
        if path in cache:
            log_callback(f"• Using cached features for {os.path.basename(path)}")
        else:
            log_callback(f"• Extracting features {idx}/{len(tracks)}")
            try:
                cache[path] = extract_audio_features(
                    path, log_callback, engine=engine, features=features
                )
                updated = True
            except Exception as e:  # pragma: no cover - defensive path
                # Not cached: a transient failure (locked file, missing
                # ffmpeg) must be retried next run, not frozen in as zeros.
                failed[path] = str(e)
                log_callback(f"! Failed features for {path}: {e}")
        if progress_callback is not None:
            progress_callback(idx, len(tracks))

    kept, feats = _collect_extracted(tracks, cache, failed, log_callback)
    return kept, feats, updated, failed


#: Accepted spellings of the clustering methods (the wizard once sent
#: "k-means", which silently ran HDBSCAN instead).
_METHOD_ALIASES = {
    "kmeans": "kmeans",
    "k-means": "kmeans",
    "k_means": "kmeans",
    "hdbscan": "hdbscan",
}


def normalize_method(method: str) -> str:
    """Return ``"kmeans"`` or ``"hdbscan"``; raise ``ValueError`` otherwise."""
    key = str(method or "").strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unknown clustering method {method!r}; expected 'kmeans' or 'hdbscan'"
        )
    return _METHOD_ALIASES[key]


def _import_hdbscan():
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HDBSCAN clustering needs the optional 'hdbscan' package "
            "(pip install hdbscan). K-Means works without it."
        ) from exc
    return hdbscan


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

    method = normalize_method(method)
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
        hdbscan = _import_hdbscan()
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


def _load_feature_cache(cache_file: str, log_callback=None) -> dict:
    """Read a feature cache written by :func:`_save_feature_cache`.

    The cache lives in the library's ``Docs/`` folder, so it is loaded with
    ``allow_pickle=False``: the old format was a pickled dict, and unpickling a
    file from a library folder can run arbitrary code. Any unreadable,
    corrupt or legacy-format file is ignored (logged) and rebuilt, rather than
    failing every run until the user deletes it.
    """
    if log_callback is None:
        log_callback = lambda msg: None
    if not os.path.exists(cache_file):
        return {}
    try:
        data = np.load(cache_file, allow_pickle=False)
        if not hasattr(data, "files"):
            raise ValueError("legacy (pickled) cache format")
        try:
            paths = [str(p) for p in data["paths"]]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        finally:
            data.close()
        if vectors.ndim != 2 or len(vectors) != len(paths):
            raise ValueError("paths and vectors are not aligned")
        return {path: vectors[i] for i, path in enumerate(paths)}
    except Exception as e:  # noqa: BLE001 - any bad cache means "re-extract"
        log_callback(f"! Ignoring unreadable feature cache ({e}); it will be rebuilt")
        return {}


def _save_feature_cache(cache_file: str, cache: dict) -> None:
    """Write ``cache`` (path -> vector) as a pickle-free ``.npz`` payload.

    The file keeps its historical name (``features*.npy``); ``np.load`` tells
    the formats apart by content. Written to a temp file and renamed, so a
    crash mid-write cannot leave a truncated cache behind.
    """
    paths = list(cache.keys())
    if paths:
        vectors = np.vstack([np.asarray(cache[p], dtype=np.float32) for p in paths])
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)
    tmp = cache_file + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, paths=np.array(paths, dtype=str), vectors=vectors)
    os.replace(tmp, cache_file)


def _load_cache_with_validation(cache_file: str, tracks: list, log_callback=None) -> dict:
    """Load feature cache with validation for stale entries."""
    if log_callback is None:
        log_callback = lambda msg: None

    cache = _load_feature_cache(cache_file, log_callback)
    if not cache:
        return {}

    # Validate cache entries against track modification times
    cache_metadata_file = cache_file.replace(".npy", "_metadata.json")
    try:
        with open(cache_metadata_file, encoding="utf-8") as f:
            metadata = json_module.load(f)
        cache_time = float(metadata.get("created_time", 0))
    except (OSError, ValueError, TypeError, AttributeError):
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


def _write_cache_metadata(cache_file: str, created_time: float, track_count: int) -> None:
    cache_metadata_file = cache_file.replace(".npy", "_metadata.json")
    with open(cache_metadata_file, "w", encoding="utf-8") as f:
        json_module.dump(
            {"created_time": created_time, "track_count": track_count}, f, indent=2
        )


def visualization_indices(n_points: int, max_points: int = None) -> "np.ndarray | None":
    """Indices of the points to keep for the graph, or ``None`` to keep all.

    Evenly spaced across the whole range (so every cluster keeps its share),
    exactly ``max_points`` long. The old "every n-th point" step kept all
    points for anything below 2x the limit while logging that it had
    downsampled.
    """
    if max_points is None:
        max_points = MAX_VISUALIZATION_POINTS
    if n_points <= max_points:
        return None
    keep = max(max_points, MIN_VISUALIZATION_POINTS)
    return np.unique(np.linspace(0, n_points - 1, keep).astype(int))


def build_cluster_payload(X, X_2d, X_3d, labels, tracks, cluster_info, log_callback=None) -> dict:
    """Assemble the ``Docs/cluster_info.json`` payload.

    Every per-point array (``X``, ``X_2d``, ``X_3d``, ``labels``, ``tracks``)
    is subset with the *same* indices when downsampling, so point i of each
    array still describes the same track (see CLAUDE.md).
    """
    if log_callback is None:
        log_callback = lambda msg: None
    labels = np.asarray(labels)
    n = len(X)
    if not (len(X_2d) == len(X_3d) == len(labels) == len(tracks) == n):
        raise ValueError(
            "cluster arrays are not index-aligned "
            f"(X={n}, X_2d={len(X_2d)}, X_3d={len(X_3d)}, labels={len(labels)}, tracks={len(tracks)})"
        )
    indices = visualization_indices(n)
    if indices is not None:
        log_callback(
            f"⚠ Library has {n} tracks; downsampling to {len(indices)} "
            f"points for visualization"
        )
        X, X_2d, X_3d = X[indices], X_2d[indices], X_3d[indices]
        labels = labels[indices]
        tracks = [tracks[i] for i in indices]
    return {
        "X": np.asarray(X).tolist(),
        "X_2d": np.asarray(X_2d).tolist(),   # 2D embedding for quick scatter plot
        "X_3d": np.asarray(X_3d).tolist(),   # 3D embedding for advanced exploration
        "X_downsampled": indices is not None,
        "X_total_points": n,
        "X_indices": [int(i) for i in indices] if indices is not None else None,
        "labels": labels.tolist(),
        "tracks": list(tracks),
        "cluster_info": cluster_info,
    }


def generate_clustered_playlists(
    tracks,
    root_path: str,
    method: str,
    params: dict,
    log_callback=None,
    engine: str = "parallel",  # DEFAULT: Use parallel processing for speed!
    feature_engine: AudioFeatureEngine = "librosa",
    use_max_workers: bool = False,
    features: dict | None = None,
    normalization: str = "standard",
    cancel_event=None,
    progress_callback=None,
) -> dict:
    """Cluster ``tracks`` and write the graph artefacts. Writes **no playlists**.

    Despite the historical name, this only writes analysis artefacts: the
    feature cache, ``Docs/cluster_info.json`` and ``Docs/cluster_graph.html``.
    Cluster playlists are exported by hand from the Music Graph (select a
    cluster, then "Export playlist…").

    ``features`` picks which audio qualities to cluster on (see
    :data:`FEATURE_ORDER`); ``None`` keeps the legacy MFCC+tempo behaviour.
    ``normalization`` selects the scaler applied before clustering — see
    :data:`SCALERS`.

    ``cancel_event`` (a ``threading.Event``) is polled throughout feature
    extraction and between the later stages; when set,
    :class:`ClusteringCancelled` is raised and no ``cluster_info.json`` /
    ``cluster_graph.html`` is written. ``progress_callback(done, total)``
    reports per-file extraction progress.

    Tracks whose features cannot be extracted are left out (and listed in the
    result's ``failed`` mapping) rather than clustered as zero vectors; the
    returned ``tracks``/``X``/``labels`` are index-aligned with each other.
    """
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
    method = normalize_method(method)
    if method == "hdbscan":
        _import_hdbscan()  # fail before a long extraction, not after it

    run_started = time.time()

    # ------------------------------------------------------------------
    # Feature cache setup
    # ------------------------------------------------------------------
    docs = os.path.join(root_path, "Docs")
    try:
        os.makedirs(docs, exist_ok=True)
    except OSError as e:
        log_callback(f"! Warning: Cannot create Docs folder: {e}")
        docs = root_path  # Fall back to root

    selection = normalize_feature_selection(features)
    # Vectors are only interchangeable within one (selection, engine) pair, so
    # each pair gets its own cache file rather than one file that would have to
    # be thrown away whenever the user changed a checkbox.
    cache_file = os.path.join(docs, feature_cache_name(selection, feature_engine))

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
    log_callback(
        f"→ Features: {feature_selection_key(selection).replace('-', ', ')} "
        f"({feature_vector_length(selection)} dimensions)"
    )

    def _save_cache() -> None:
        try:
            _save_feature_cache(cache_file, cache)
            _write_cache_metadata(cache_file, run_started, len(cache))
            log_callback(f"✓ Saved feature cache ({len(cache)} entries) to {cache_file}")
        except OSError as e:
            log_callback(f"! Warning: Could not save cache file: {e}")
            logger.warning(f"Failed to save feature cache: {e}")

    extract_kwargs = dict(
        features=selection, cancel_event=cancel_event, progress_callback=progress_callback
    )
    if engine_mode == "parallel":
        extract = _extract_features_parallel
        extract_kwargs["use_max_workers"] = use_max_workers
    else:
        extract = _extract_features_serial
    cache_len_before = len(cache)
    try:
        kept_tracks, feats, updated, failed = extract(
            tracks, cache, log_callback, feature_engine, **extract_kwargs
        )
    except ClusteringCancelled:
        # Keep what was extracted so the next run resumes; write nothing else.
        if len(cache) != cache_len_before:
            _save_cache()
        log_callback("! Clustering cancelled; no cluster data was written.")
        raise

    if updated:
        _save_cache()

    if failed:
        log_callback(
            f"⚠ {len(failed)} track(s) could not be analysed and were left out "
            "(they will be retried next run)"
        )
    if not feats:
        first = next(iter(failed.values()), "no tracks")
        raise RuntimeError(
            f"Feature extraction failed for every track (first error: {first})"
        )
    if _is_cancelled(cancel_event):
        raise ClusteringCancelled()
    tracks = kept_tracks

    X = np.vstack(feats)
    scaler_cls = SCALERS.get(normalization)
    if scaler_cls is None:
        log_callback(
            f"! Unknown normalization {normalization!r}; falling back to 'standard'"
        )
        scaler_cls = SCALERS["standard"]
    # Scaling matters more now that features are selectable: tempo is in the
    # hundreds while chroma sits in [0, 1], so without it whichever block has
    # the largest raw numbers would dominate the distances.
    log_callback(f"→ Normalizing features ({normalization})")
    X = scaler_cls().fit_transform(X)

    labels = cluster_tracks(
        X, method, log_callback=log_callback, engine=engine_mode, **params
    )
    log_callback(
        f"✓ Clustering complete: found {len(set([l for l in labels if l >= 0]) )} clusters"
    )
    log_cluster_summary(labels, log_callback)

    log_callback(
        "→ Cluster playlists are not written automatically; export a cluster "
        "from the Music Graph (select it, then Export playlist…)."
    )

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

    if _is_cancelled(cancel_event):
        raise ClusteringCancelled()

    # Compute 2D embedding for visualization
    log_callback("→ Computing 2D scatter plot coordinates…")
    X_2d = compute_2d_embedding(X, log_callback)

    # Compute 3D embedding for advanced exploration
    log_callback("→ Computing 3D audio feature space (preserves ~80% data variance)…")
    X_3d = compute_3d_embedding(X, log_callback)

    if _is_cancelled(cancel_event):
        raise ClusteringCancelled()

    # Save cluster data to JSON for visualization (downsampled for very large
    # libraries, but always index-aligned — see build_cluster_payload).
    cluster_data = build_cluster_payload(
        X, X_2d, X_3d, labels, tracks, cluster_info, log_callback
    )

    cluster_info_file = os.path.join(docs, "cluster_info.json")
    try:
        tmp = cluster_info_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json_module.dump(cluster_data, f, indent=2)
        os.replace(tmp, cluster_info_file)
        log_callback(f"✓ Saved cluster data to {cluster_info_file}")
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
        "failed": failed,
        "metrics": {},  # Will be filled by caller with sklearn metrics
    }
