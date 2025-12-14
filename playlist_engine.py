import os
import math
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy optional
    np = None
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover - librosa optional
    librosa = None

from playlist_generator import write_playlist, DEFAULT_EXTS


def categorize_tempo(bpm: float) -> str:
    if bpm < 90:
        return "slow"
    if bpm < 120:
        return "medium"
    return "fast"


def categorize_energy(rms: float) -> str:
    if rms < 0.1:
        return "low"
    if rms < 0.3:
        return "medium"
    return "high"


def compute_tempo_energy(path: str) -> tuple[float, float]:
    if librosa is None or np is None:
        raise RuntimeError("librosa/numpy required for tempo analysis")
    y, sr = librosa.load(path, mono=True, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    return tempo, rms


def bucket_by_tempo_energy(
    tracks: list[str],
    root_path: str,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
) -> dict:
    if log_callback is None:
        log_callback = lambda m: None
    if progress_callback is None:
        progress_callback = lambda _count: None
    if librosa is None or np is None:
        raise RuntimeError(
            "Tempo/Energy buckets require numpy and librosa. Install with `pip install -r requirements.txt`."
        )
    playlists_dir = os.path.join(root_path, "Playlists")
    os.makedirs(playlists_dir, exist_ok=True)
    buckets: dict[tuple[str, str], list[str]] = {}
    processed = 0
    for path in tracks:
        if cancel_event and cancel_event.is_set():
            log_callback("! Bucket generation cancelled by user.")
            break
        try:
            tempo, rms = compute_tempo_energy(path)
            tb = categorize_tempo(tempo)
            eb = categorize_energy(rms)
            buckets.setdefault((tb, eb), []).append(path)
            log_callback(f"• {os.path.basename(path)} → {tb}/{eb}")
        except Exception as e:
            log_callback(f"! Failed analysis for {path}: {e}")
        processed += 1
        progress_callback(processed)
    out_paths = {}
    for (tb, eb), items in buckets.items():
        outfile = os.path.join(playlists_dir, f"{tb}_{eb}.m3u")
        write_playlist(items, outfile)
        out_paths[(tb, eb)] = outfile
        log_callback(f"→ Wrote {outfile}")
    stats = {
        (tb, eb): {"count": len(items), "playlist": out_paths[(tb, eb)]}
        for (tb, eb), items in buckets.items()
    }
    return {
        "buckets": stats,
        "processed": processed,
        "total": len(tracks),
        "cancelled": bool(cancel_event and cancel_event.is_set()),
    }


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
