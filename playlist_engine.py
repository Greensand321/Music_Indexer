import importlib.util
import os
import math
import re
from typing import Callable, Iterable, Literal

try:
    _mutagen_available = importlib.util.find_spec("mutagen") is not None
except ValueError:  # pragma: no cover - defensive for broken environments
    _mutagen_available = False
if _mutagen_available:
    from mutagen import File as MutagenFile
    from mutagen.easyid3 import EasyID3
    from mutagen.flac import FLAC
else:  # pragma: no cover - optional dependency
    class _DummyAudio(dict):
        def get(self, key, default=None):
            return []

    def MutagenFile(*_a, **_k):
        return None

    class EasyID3(_DummyAudio):
        pass

    class FLAC(_DummyAudio):
        pass

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy optional
    np = None
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover - librosa optional
    librosa = None

try:
    _essentia_available = importlib.util.find_spec("essentia") is not None
except ValueError:  # pragma: no cover - defensive for broken environments
    _essentia_available = False
if _essentia_available:
    import essentia  # type: ignore
    from essentia.standard import MonoLoader, RhythmExtractor2013
else:  # pragma: no cover - optional dependency
    essentia = None
    MonoLoader = RhythmExtractor2013 = None  # type: ignore

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


TempoEngine = Literal["librosa", "essentia"]


def compute_tempo_energy(path: str, engine: TempoEngine = "librosa") -> tuple[float, float]:
    if engine == "librosa":
        return _compute_tempo_energy_librosa(path)
    if engine == "essentia":
        return _compute_tempo_energy_essentia(path)
    raise ValueError(f"Unknown tempo engine: {engine}")


def _compute_tempo_energy_librosa(path: str) -> tuple[float, float]:
    if librosa is None or np is None:
        raise RuntimeError("librosa/numpy required for tempo analysis (engine='librosa')")
    y, sr = librosa.load(path, mono=True, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    return tempo, rms


def _compute_tempo_energy_essentia(path: str) -> tuple[float, float]:
    if essentia is None or np is None or MonoLoader is None or RhythmExtractor2013 is None:
        raise RuntimeError("Essentia/numpy required for tempo analysis (engine='essentia')")
    audio = np.asarray(MonoLoader(filename=path)(), dtype=np.float32)
    tempo, *_ = RhythmExtractor2013(method="multifeature")(audio)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return float(tempo), rms


def bucket_by_tempo_energy(
    tracks: list[str],
    root_path: str,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
    engine: TempoEngine = "librosa",
) -> dict:
    if log_callback is None:
        log_callback = lambda m: None
    if progress_callback is None:
        progress_callback = lambda _count: None
    if engine == "librosa" and (librosa is None or np is None):
        raise RuntimeError(
            "Tempo/Energy buckets require numpy and librosa. Install with `pip install -r requirements.txt`."
        )
    if engine == "essentia" and (essentia is None or np is None):
        raise RuntimeError(
            "Tempo/Energy buckets require numpy and Essentia when engine='essentia'. Install Essentia to continue."
        )
    playlists_dir = os.path.join(root_path, "Playlists")
    os.makedirs(playlists_dir, exist_ok=True)

    # Cached results setup (mirrors clustering tools for faster re-runs)
    docs_dir = os.path.join(root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    cache_file = os.path.join(docs_dir, "tempo_energy.npy")
    try:
        cache = dict(np.load(cache_file, allow_pickle=True).item())
        log_callback(f"→ Loaded {len(cache)} cached tempo/energy entries")
    except FileNotFoundError:
        cache = {}
        log_callback("→ No tempo/energy cache found; analyzing all tracks")

    buckets: dict[tuple[str, str], list[str]] = {}
    processed = 0
    updated_cache = False
    for path in tracks:
        if cancel_event and cancel_event.is_set():
            log_callback("! Bucket generation cancelled by user.")
            break
        cached = cache.get(path)
        try:
            if cached and cached.get("engine") == engine:
                tempo = float(cached["tempo"])
                rms = float(cached["rms"])
                log_callback(f"• Using cached tempo/energy for {os.path.basename(path)}")
            else:
                tempo, rms = compute_tempo_energy(path, engine=engine)
                cache[path] = {"engine": engine, "tempo": tempo, "rms": rms}
                updated_cache = True
            tb = categorize_tempo(tempo)
            eb = categorize_energy(rms)
            buckets.setdefault((tb, eb), []).append(path)
            log_callback(f"• {os.path.basename(path)} → {tb}/{eb}")
        except Exception as e:
            log_callback(f"! Failed analysis for {path}: {e}")
        processed += 1
        progress_callback(processed)

    if updated_cache:
        np.save(cache_file, cache)
        log_callback(f"✓ Saved tempo/energy cache ({len(cache)} entries) to {cache_file}")
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
        "buckets": buckets,
        "playlist_paths": out_paths,
        "stats": stats,
        "processed": processed,
        "total": len(tracks),
        "cancelled": bool(cancel_event and cancel_event.is_set()),
    }


def _split_genres(genres: Iterable[str]) -> list[str]:
    parts: list[str] = []
    for raw in genres:
        for chunk in re.split(r"[;,/|\\]", raw):
            cleaned = chunk.strip()
            if cleaned:
                parts.append(cleaned)
    return parts


def extract_genres(path: str) -> list[str]:
    ext = os.path.splitext(path)[1].lower()
    tags: list[str] = []
    if ext == ".mp3":
        audio = EasyID3(path)
        tags = [t for t in audio.get("genre", []) if isinstance(t, str)]
    elif ext == ".flac":
        audio = FLAC(path)
        tags = [t for t in audio.get("genre", []) if isinstance(t, str)]
    else:
        audio = MutagenFile(path, easy=True)
        if audio and audio.tags:
            tags = [t for t in audio.tags.get("genre", []) if isinstance(t, str)]
    return _split_genres(tags)


def _sanitize_genre(genre: str, used: set[str]) -> str:
    safe = re.sub(r"[^\w\- ]+", "_", genre).strip() or "Unknown"
    candidate = safe.replace(" ", "_")
    if candidate not in used:
        used.add(candidate)
        return candidate
    counter = 2
    while f"{candidate}_{counter}" in used:
        counter += 1
    final = f"{candidate}_{counter}"
    used.add(final)
    return final


def sort_tracks_by_genre(
    tracks: list[str],
    root_path: str,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
    genre_reader: Callable[[str], list[str]] | None = None,
    export: bool = True,
    selected_genres: set[str] | list[str] | None = None,
) -> dict:
    """Group tracks by genre and optionally write playlists.

    If ``export`` is False, this function will only analyze genres and return
    stats without writing any playlists. Pass ``selected_genres`` to export a
    subset of genres when ``export`` is True.
    """

    if log_callback is None:
        log_callback = lambda m: None
    if progress_callback is None:
        progress_callback = lambda _count: None
    reader = genre_reader or extract_genres

    playlists_dir = os.path.join(root_path, "Playlists", "Genres")
    if export:
        os.makedirs(playlists_dir, exist_ok=True)

    buckets: dict[str, list[str]] = {}
    processed = 0
    for path in tracks:
        if cancel_event and cancel_event.is_set():
            log_callback("! Genre sorting cancelled by user.")
            break
        try:
            genres = reader(path) or ["Unknown"]
        except Exception as exc:
            log_callback(f"! Failed to read genres from {path}: {exc}")
            genres = ["Unknown"]
        for g in genres:
            buckets.setdefault(g, []).append(path)
        log_callback(f"• {os.path.basename(path)} → {', '.join(genres)}")
        processed += 1
        progress_callback(processed)

    used_names: set[str] = set()
    stats = {}
    planned_paths: dict[str, str] = {}
    selected = set(selected_genres) if selected_genres else None
    for genre, items in sorted(buckets.items(), key=lambda kv: kv[0].lower()):
        fname = _sanitize_genre(genre, used_names) + ".m3u"
        out_path = os.path.join(playlists_dir, fname)
        planned_paths[genre] = out_path
        should_export = export and (selected is None or genre in selected)
        if should_export:
            write_playlist(items, out_path)
            log_callback(f"→ Wrote {out_path}")
        stats[genre] = {
            "count": len(items),
            "playlist": out_path,
            "exported": should_export,
        }
        if export and not should_export:
            log_callback(f"• Skipped exporting {genre}")

    return {
        "genres": stats,
        "processed": processed,
        "total": len(tracks),
        "cancelled": bool(cancel_event and cancel_event.is_set()),
        "buckets": buckets,
        "playlist_paths": planned_paths,
    }


def export_genre_playlists(
    buckets: dict[str, list[str]],
    root_path: str,
    selected_genres: set[str] | list[str] | None = None,
    log_callback=None,
    planned_paths: dict[str, str] | None = None,
):
    """Write playlists for the provided genre buckets.

    Parameters
    ----------
    buckets : dict
        Mapping of genre -> list of track paths.
    root_path : str
        Root music library path used to resolve playlist directory.
    selected_genres : set | list | None
        Optional subset of genres to export. If None, all genres are exported.
    log_callback : callable | None
        Callback for writing log messages.
    planned_paths : dict | None
        Optional mapping of genre -> playlist path generated earlier. When
        provided, the same filenames are reused to keep the preview stable.
    """

    if log_callback is None:
        log_callback = lambda m: None

    playlists_dir = os.path.join(root_path, "Playlists", "Genres")
    os.makedirs(playlists_dir, exist_ok=True)

    selected = set(selected_genres) if selected_genres else None
    used_names: set[str] = set()
    stats: dict[str, dict] = {}

    for genre, items in sorted(buckets.items(), key=lambda kv: kv[0].lower()):
        if planned_paths and genre in planned_paths:
            out_path = planned_paths[genre]
            # Ensure later genres do not reuse the same base name
            used_names.add(os.path.splitext(os.path.basename(out_path))[0])
        else:
            fname = _sanitize_genre(genre, used_names) + ".m3u"
            out_path = os.path.join(playlists_dir, fname)

        should_export = selected is None or genre in selected
        if should_export:
            write_playlist(items, out_path)
            log_callback(f"→ Wrote {out_path}")
        else:
            log_callback(f"• Skipped exporting {genre}")

        stats[genre] = {
            "count": len(items),
            "playlist": out_path,
            "exported": should_export,
        }

    return {"genres": stats, "playlists_dir": playlists_dir}


def _get_feat(path: str, cache: dict, log_callback, engine: TempoEngine = "librosa"):
    if path not in cache:
        from clustered_playlists import extract_audio_features

        cache[path] = extract_audio_features(path, log_callback, engine=engine)
    return cache[path]


def _dist(a, b) -> float:
    if np is not None:
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def more_like_this(
    ref_track: str,
    tracks: list[str],
    n: int = 10,
    feature_cache=None,
    log_callback=None,
    engine: TempoEngine = "librosa",
) -> list[str]:
    if log_callback is None:
        log_callback = lambda m: None
    feature_cache = feature_cache or {}
    ref_vec = _get_feat(ref_track, feature_cache, log_callback, engine)
    others = [t for t in tracks if t != ref_track]
    dist = []
    for t in others:
        vec = _get_feat(t, feature_cache, log_callback, engine)
        d = _dist(ref_vec, vec)
        dist.append((d, t))
    dist.sort()
    return [t for _d, t in dist[:n]]


def autodj_playlist(
    start_track: str,
    tracks: list[str],
    n: int = 20,
    feature_cache=None,
    log_callback=None,
    engine: TempoEngine = "librosa",
) -> list[str]:
    if log_callback is None:
        log_callback = lambda m: None
    feature_cache = feature_cache or {}
    order = [start_track]
    remaining = [t for t in tracks if t != start_track]
    while remaining and len(order) < n:
        cur_feat = _get_feat(order[-1], feature_cache, log_callback, engine)
        next_track = min(
            remaining,
            key=lambda t: _dist(
                cur_feat, _get_feat(t, feature_cache, log_callback, engine)
            ),
        )
        order.append(next_track)
        remaining.remove(next_track)
    return order
