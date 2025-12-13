import os
import math
from typing import Callable, Iterable, Sequence

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy optional
    np = None
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover - librosa optional
    librosa = None

from playlist_generator import write_playlist


DEFAULT_TEMPO_BUCKETS: list[tuple[float | None, float | None, str]] = [
    (None, 90.0, "slow"),
    (90.0, 120.0, "medium"),
    (120.0, None, "fast"),
]
DEFAULT_ENERGY_BUCKETS: list[tuple[float | None, float | None, str]] = [
    (None, 0.1, "low"),
    (0.1, 0.3, "medium"),
    (0.3, None, "high"),
]


def _fmt_val(val: float | None, decimals: int = 0) -> str:
    if val is None:
        return ""
    if decimals:
        return f"{val:.{decimals}f}".rstrip("0").rstrip(".")
    return str(int(val))


def _format_range_name(lower: float | None, upper: float | None, decimals: int = 0) -> str:
    lo = _fmt_val(lower, decimals) if lower is not None else "0"
    if upper is None:
        return f"{lo}+"
    hi = _fmt_val(upper, decimals)
    return f"{lo}-{hi}"


def parse_range_spec(spec: str) -> list[tuple[float | None, float | None]]:
    ranges: list[tuple[float | None, float | None]] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if part.endswith("+"):
            lower = float(part[:-1])
            upper = None
        elif "-" in part:
            lower_s, upper_s = part.split("-", 1)
            lower = float(lower_s) if lower_s else None
            upper = float(upper_s) if upper_s else None
        else:
            raise ValueError(f"Invalid range segment: '{part}'")
        if upper is not None and lower is not None and upper <= lower:
            raise ValueError(f"Upper bound must be greater than lower bound: '{part}'")
        ranges.append((lower, upper))
    if not ranges:
        raise ValueError("Provide at least one tempo range")
    return ranges


def parse_thresholds(spec: str) -> list[float]:
    vals = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        vals.append(float(part))
    vals = sorted(set(vals))
    return vals


def _tempo_buckets_from_ranges(ranges: Iterable[tuple[float | None, float | None]] | None):
    if not ranges:
        return DEFAULT_TEMPO_BUCKETS
    buckets = []
    for lower, upper in ranges:
        buckets.append((lower, upper, _format_range_name(lower, upper)))
    return buckets


def _energy_buckets_from_thresholds(thresholds: Iterable[float] | None):
    if not thresholds:
        return DEFAULT_ENERGY_BUCKETS
    sorted_thr = sorted(thresholds)
    buckets = []
    prev: float | None = None
    for thr in sorted_thr:
        buckets.append((prev, thr, _format_range_name(prev, thr, decimals=2)))
        prev = thr
    buckets.append((prev, None, _format_range_name(prev, None, decimals=2)))
    return buckets


def categorize_tempo(bpm: float, buckets=None) -> str:
    buckets = buckets or DEFAULT_TEMPO_BUCKETS
    for lower, upper, name in buckets:
        if (lower is None or bpm >= lower) and (upper is None or bpm < upper):
            return name
    return "unknown"


def categorize_energy(rms: float, buckets=None) -> str:
    buckets = buckets or DEFAULT_ENERGY_BUCKETS
    for lower, upper, name in buckets:
        if (lower is None or rms >= lower) and (upper is None or rms < upper):
            return name
    return "unknown"


def compute_tempo_energy(path: str) -> tuple[float, float]:
    if librosa is None or np is None:
        raise RuntimeError("librosa/numpy required for tempo analysis")
    y, sr = librosa.load(path, mono=True, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    return tempo, rms


def bucket_by_tempo_energy(
    library_path: str,
    tempo_ranges: Sequence[tuple[float | None, float | None]] | None = None,
    energy_thresholds: Sequence[float] | None = None,
    output_dir: str | None = None,
    folder_filter: dict | None = None,
    log_callback: Callable[[str], None] | None = None,
    feature_provider: Callable[[str], tuple[float, float]] | None = None,
) -> dict:
    if log_callback is None:
        log_callback = lambda m: None
    from controllers.cluster_controller import gather_tracks

    feature_provider = feature_provider or compute_tempo_energy
    if feature_provider is compute_tempo_energy and (librosa is None or np is None):
        raise RuntimeError("librosa/numpy required for bucket generation")

    tracks = gather_tracks(library_path, folder_filter)
    if not tracks:
        raise ValueError("No audio files found in the library.")

    tempo_buckets = _tempo_buckets_from_ranges(tempo_ranges)
    energy_buckets = _energy_buckets_from_thresholds(energy_thresholds)

    assignments: dict[str, dict[str, list[str]]] = {
        "tempo": {name: [] for *_rest, name in tempo_buckets},
        "energy": {name: [] for *_rest, name in energy_buckets},
    }

    log_callback(f"Found {len(tracks)} audio files; analyzing tempo and energy")
    for path in tracks:
        try:
            tempo, rms = feature_provider(path)
        except Exception as e:
            log_callback(f"! Failed analysis for {path}: {e}")
            continue
        tempo_label = categorize_tempo(tempo, tempo_buckets)
        energy_label = categorize_energy(rms, energy_buckets)
        assignments["tempo"].setdefault(tempo_label, []).append(path)
        assignments["energy"].setdefault(energy_label, []).append(path)
        log_callback(
            f"• {os.path.basename(path)} → tempo={tempo_label} ({tempo:.2f} BPM), "
            f"energy={energy_label} ({rms:.3f})"
        )

    playlists_dir = output_dir or os.path.join(library_path, "Playlists")
    os.makedirs(playlists_dir, exist_ok=True)

    results: dict[str, dict[str, str]] = {"tempo": {}, "energy": {}}
    for label, items in assignments["tempo"].items():
        if not items:
            continue
        outfile = os.path.join(playlists_dir, f"tempo_{label}.m3u")
        write_playlist(items, outfile)
        results["tempo"][label] = outfile
        log_callback(f"→ Wrote {outfile}")

    for label, items in assignments["energy"].items():
        if not items:
            continue
        outfile = os.path.join(playlists_dir, f"energy_{label}.m3u")
        write_playlist(items, outfile)
        results["energy"][label] = outfile
        log_callback(f"→ Wrote {outfile}")

    return results


def _get_feat(path: str, cache: dict, log_callback):
    if path not in cache:
        from clustered_playlists import extract_audio_features
        cache[path] = extract_audio_features(path, log_callback)
    return cache[path]


def _dist(a, b) -> float:
    if np is not None:
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def more_like_this(ref_track: str, tracks: list[str], n: int = 10, feature_cache=None, log_callback=None) -> list[str]:
    if log_callback is None:
        log_callback = lambda m: None
    feature_cache = feature_cache or {}
    ref_vec = _get_feat(ref_track, feature_cache, log_callback)
    others = [t for t in tracks if t != ref_track]
    dist = []
    for t in others:
        vec = _get_feat(t, feature_cache, log_callback)
        d = _dist(ref_vec, vec)
        dist.append((d, t))
    dist.sort()
    return [t for _d, t in dist[:n]]


def autodj_playlist(start_track: str, tracks: list[str], n: int = 20, feature_cache=None, log_callback=None) -> list[str]:
    if log_callback is None:
        log_callback = lambda m: None
    feature_cache = feature_cache or {}
    order = [start_track]
    remaining = [t for t in tracks if t != start_track]
    while remaining and len(order) < n:
        cur_feat = _get_feat(order[-1], feature_cache, log_callback)
        next_track = min(
            remaining,
            key=lambda t: _dist(cur_feat, _get_feat(t, feature_cache, log_callback)),
        )
        order.append(next_track)
        remaining.remove(next_track)
    return order
