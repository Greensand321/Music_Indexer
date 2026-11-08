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
from indexer_control import IndexCancelled
from utils.audio_metadata_reader import read_tags


def _finite(value) -> float | None:
    """``value`` as a finite float, or ``None`` (missing, NaN, inf, junk)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
    # Newer librosa returns tempo as a 1-element array.
    return float(np.atleast_1d(tempo)[0]), rms


def _compute_tempo_energy_essentia(path: str) -> tuple[float, float]:
    if essentia is None or np is None or MonoLoader is None or RhythmExtractor2013 is None:
        raise RuntimeError("Essentia/numpy required for tempo analysis (engine='essentia')")
    audio = np.asarray(MonoLoader(filename=path)(), dtype=np.float32)
    tempo, *_ = RhythmExtractor2013(method="multifeature")(audio)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return float(tempo), rms


def _load_tempo_cache(cache_file: str, log_callback) -> dict:
    """Read ``Docs/tempo_energy.npy`` without unpickling it.

    The file sits in the library folder, so it is loaded with
    ``allow_pickle=False`` (the old pickled-dict format is ignored and
    rebuilt). A corrupt or unreadable file is logged and rebuilt instead of
    failing every Tempo run until the user deletes it.
    """
    if not os.path.exists(cache_file):
        log_callback("→ No tempo/energy cache found; analyzing all tracks")
        return {}
    try:
        data = np.load(cache_file, allow_pickle=False)
        if not hasattr(data, "files"):
            raise ValueError("legacy (pickled) cache format")
        try:
            paths = [str(p) for p in data["paths"]]
            engines = [str(e) for e in data["engines"]]
            tempos = [float(t) for t in data["tempos"]]
            rms = [float(r) for r in data["rms"]]
        finally:
            data.close()
        if not (len(paths) == len(engines) == len(tempos) == len(rms)):
            raise ValueError("cache arrays are not aligned")
    except Exception as exc:  # noqa: BLE001 - any bad cache means "re-analyse"
        log_callback(f"! Ignoring unreadable tempo/energy cache ({exc}); it will be rebuilt")
        return {}
    cache = {
        p: {"engine": e, "tempo": t, "rms": r}
        for p, e, t, r in zip(paths, engines, tempos, rms)
    }
    log_callback(f"→ Loaded {len(cache)} cached tempo/energy entries")
    return cache


def _save_tempo_cache(cache_file: str, cache: dict) -> None:
    paths = list(cache)
    tmp = cache_file + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            paths=np.array(paths, dtype=str),
            engines=np.array([str(cache[p].get("engine", "")) for p in paths], dtype=str),
            tempos=np.array([float(cache[p]["tempo"]) for p in paths], dtype=np.float64),
            rms=np.array([float(cache[p]["rms"]) for p in paths], dtype=np.float64),
        )
    os.replace(tmp, cache_file)


def _in_range(value: float, bounds) -> bool:
    if not bounds:
        return True
    low, high = bounds
    return float(low) <= float(value) <= float(high)


def bucket_by_tempo_energy(
    tracks: list[str],
    root_path: str,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
    engine: TempoEngine = "librosa",
    tempo_range: tuple[float, float] | None = None,
    energy_range: tuple[float, float] | None = None,
) -> dict:
    """Analyse tempo/energy and group tracks into ``slow|medium|fast x low|medium|high``.

    Writes no playlists (only the analysis cache under ``Docs/``); the
    returned ``playlist_paths`` are where the caller *may* write them.
    ``tempo_range`` (BPM) and ``energy_range`` (mean RMS, 0-1) drop tracks
    outside the given inclusive bounds; their count is returned as
    ``excluded``. A track listed twice is analysed and bucketed once. A track
    whose analysis fails or yields no finite tempo/energy (NaN used to land
    in "fast/high", since every comparison with NaN is false) is put in no
    bucket, is not cached, and is listed in ``failed``.
    """
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

    # Cached results setup (mirrors clustering tools for faster re-runs)
    docs_dir = os.path.join(root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    cache_file = os.path.join(docs_dir, "tempo_energy.npy")
    cache = _load_tempo_cache(cache_file, log_callback)

    # Normalize cache keys to absolute paths so repeated runs can reuse results
    # even if callers pass in different path representations.
    normalized_cache: dict[str, dict] = {}
    for key, value in cache.items():
        abs_key = os.path.abspath(os.fspath(key))
        normalized_cache[abs_key] = value
    cache_normalized_changed = cache != normalized_cache
    cache = normalized_cache

    buckets: dict[tuple[str, str], list[str]] = {}
    processed = 0
    excluded = 0
    failed: list[str] = []
    updated_cache = cache_normalized_changed

    normalized_tracks: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for path in tracks:
        try:
            path_str = os.fspath(path)
        except TypeError:
            # Handle unexpected iterable inputs (e.g., lists from GUI selections)
            if isinstance(path, (list, tuple)) and path:
                path_str = os.fspath(path[0])
            else:
                log_callback(f"! Skipping unsupported path value: {path!r}")
                processed += 1
                progress_callback(processed)
                continue
        abs_path = os.path.abspath(path_str)
        if os.path.normcase(abs_path) in seen_keys:
            processed += 1
            progress_callback(processed)
            continue
        seen_keys.add(os.path.normcase(abs_path))
        normalized_tracks.append((path_str, abs_path))

    def _cached_ok(entry) -> bool:
        return bool(entry) and entry.get("engine") == engine and (
            _finite(entry.get("tempo")) is not None and _finite(entry.get("rms")) is not None
        )

    all_cached = bool(normalized_tracks) and all(
        _cached_ok(cache.get(abs_path)) for _, abs_path in normalized_tracks
    )
    if all_cached:
        log_callback(
            f"→ All {len(normalized_tracks)} tracks already cached with engine='{engine}'. Skipping re-analysis."
        )

    for path_str, cache_key in normalized_tracks:
        if cancel_event and cancel_event.is_set():
            log_callback("! Bucket generation cancelled by user.")
            break
        cached = cache.get(cache_key)
        try:
            if _cached_ok(cached):
                tempo = float(cached["tempo"])
                rms = float(cached["rms"])
                if not all_cached:
                    log_callback(f"• Using cached tempo/energy for {os.path.basename(path_str)}")
            else:
                raw_tempo, raw_rms = compute_tempo_energy(path_str, engine=engine)
                tempo, rms = _finite(raw_tempo), _finite(raw_rms)
                if tempo is None or rms is None:
                    if cache.pop(cache_key, None) is not None:
                        updated_cache = True
                    failed.append(path_str)
                    log_callback(
                        f"! Failed analysis for {path_str}: no usable tempo/energy "
                        f"(got {raw_tempo!r} BPM, energy {raw_rms!r})"
                    )
                    processed += 1
                    progress_callback(processed)
                    continue
                cache[cache_key] = {"engine": engine, "tempo": tempo, "rms": rms}
                updated_cache = True
            if not (_in_range(tempo, tempo_range) and _in_range(rms, energy_range)):
                excluded += 1
                log_callback(
                    f"• {os.path.basename(path_str)} skipped "
                    f"({float(tempo):.0f} BPM, energy {float(rms):.2f} outside the chosen range)"
                )
                processed += 1
                progress_callback(processed)
                continue
            tb = categorize_tempo(tempo)
            eb = categorize_energy(rms)
            buckets.setdefault((tb, eb), []).append(path_str)
            log_callback(f"• {os.path.basename(path_str)} → {tb}/{eb}")
        except Exception as e:
            failed.append(path_str)
            log_callback(f"! Failed analysis for {path_str}: {e}")
        processed += 1
        progress_callback(processed)

    if updated_cache:
        try:
            _save_tempo_cache(cache_file, cache)
            log_callback(f"✓ Saved tempo/energy cache ({len(cache)} entries) to {cache_file}")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            log_callback(f"! Could not save tempo/energy cache: {exc}")
    out_paths = {}
    for (tb, eb), items in buckets.items():
        outfile = os.path.join(playlists_dir, f"{tb}_{eb}.m3u")
        out_paths[(tb, eb)] = outfile
        log_callback(
            f"→ Prepared {len(items)} track(s) for {tb}/{eb} (playlist not yet saved)"
        )
    stats = {
        (tb, eb): {
            "count": len(items),
            "playlist": out_paths[(tb, eb)],
            "tracks": list(items),
        }
        for (tb, eb), items in buckets.items()
    }

    return {
        "buckets": buckets,
        "playlist_paths": out_paths,
        "stats": stats,
        "processed": processed,
        "excluded": excluded,
        "failed": failed,
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
    raw = read_tags(path).get("genre")
    if raw in (None, ""):
        return []
    if isinstance(raw, (list, tuple)):
        values = [str(v) for v in raw if isinstance(v, str)]
    else:
        values = [str(raw)]
    return _split_genres(values)


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


def _usable_vector(vec) -> list[float] | None:
    """``vec`` as a list of finite floats, or ``None`` if it cannot be compared."""
    if vec is None:
        return None
    try:
        values = [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]
    except (TypeError, ValueError):
        return None
    if not values or not all(math.isfinite(x) for x in values):
        return None
    return values


def _track_key(path) -> str:
    """Identity of a track for de-duplication (the same file listed twice)."""
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except TypeError:
        return repr(path)


def _candidate_vectors(
    ref_track: str,
    tracks: Iterable[str],
    feature_cache: dict,
    log_callback,
    engine: TempoEngine,
    cancel_event=None,
    progress=None,
) -> tuple[list[float], list[str], list[list[float]]]:
    """The reference vector, then every other usable track and its vector.

    Each track appears once (the first time it is listed). A track whose
    features cannot be extracted, are not finite or have another length than
    the reference's is left out with a log line — one unreadable file used to
    abort the whole Auto-DJ run. Raises ``RuntimeError`` when the reference
    itself cannot be analysed, and ``IndexCancelled`` when cancelled (checked
    before every track, so Cancel works during the analysis of a large
    library, not only between picks).
    """
    try:
        ref_vec = _usable_vector(_get_feat(ref_track, feature_cache, log_callback, engine))
    except IndexCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as one clear error
        raise RuntimeError(
            f"Could not analyse the seed track {os.path.basename(str(ref_track))}: {exc}"
        ) from exc
    if ref_vec is None:
        raise RuntimeError(
            f"Could not analyse the seed track {os.path.basename(str(ref_track))}: "
            "no usable audio features"
        )
    seen = {_track_key(ref_track)}
    unique: list[str] = []
    for track in tracks:
        key = _track_key(track)
        if key in seen:
            continue
        seen.add(key)
        unique.append(track)

    names: list[str] = []
    vectors: list[list[float]] = []
    for index, track in enumerate(unique, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise IndexCancelled()
        try:
            vec = _usable_vector(_get_feat(track, feature_cache, log_callback, engine))
            problem = None if vec is not None else "no usable audio features"
        except IndexCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            vec, problem = None, str(exc) or type(exc).__name__
        if vec is not None and len(vec) != len(ref_vec):
            vec, problem = None, "its features do not match the seed's"
        if vec is None:
            log_callback(f"! Skipping {os.path.basename(str(track))}: {problem}")
        else:
            names.append(track)
            vectors.append(vec)
        if progress is not None:
            progress(index, len(unique))
    return ref_vec, names, vectors


def _distances(cur: list[float], vectors: list[list[float]]) -> list[float]:
    if np is not None and vectors:
        return np.linalg.norm(
            np.asarray(vectors, dtype=np.float64) - np.asarray(cur, dtype=np.float64), axis=1
        ).tolist()
    return [_dist(cur, v) for v in vectors]


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
    """The ``n`` tracks closest to ``ref_track`` (nearest first; ties keep
    input order). Tracks that cannot be analysed are skipped."""
    if log_callback is None:
        log_callback = lambda m: None
    if feature_cache is None:
        feature_cache = {}
    ref_vec, names, vectors = _candidate_vectors(
        ref_track, tracks, feature_cache, log_callback, engine
    )
    dist = sorted(zip(_distances(ref_vec, vectors), range(len(names))))
    return [names[i] for _d, i in dist[: max(0, n)]]


def autodj_playlist(
    start_track: str,
    tracks: list[str],
    n: int = 20,
    feature_cache=None,
    log_callback=None,
    engine: TempoEngine = "librosa",
    progress_callback=None,
    cancel_event=None,
) -> list[str]:
    """Greedy nearest-neighbour chain of up to ``n`` tracks from ``start_track``.

    Every track is used at most once (a track listed twice is one track),
    the result is deterministic (ties go to the track listed first), and it
    always terminates. Tracks whose features cannot be extracted are skipped
    and logged; only an unreadable seed is an error. ``feature_cache`` (a
    dict, filled in place — an empty one too) avoids re-extracting features.
    """
    if log_callback is None:
        log_callback = lambda m: None
    if progress_callback is None:
        progress_callback = lambda _c, _t, _m=None: None
    if feature_cache is None:
        feature_cache = {}
    if n <= 0:
        return []
    if cancel_event is not None and cancel_event.is_set():
        raise IndexCancelled()

    def _analysis_progress(done: int, count: int) -> None:
        if done == count or done % 25 == 0:
            progress_callback(1, min(n, count + 1), f"Analysed {done} of {count} tracks")

    cur, names, vectors = _candidate_vectors(
        start_track, tracks, feature_cache, log_callback, engine,
        cancel_event=cancel_event, progress=_analysis_progress,
    )
    order = [start_track]
    total = min(n, len(names) + 1)
    progress_callback(1, total, "Seed track selected")
    remaining = list(range(len(names)))
    while remaining and len(order) < n:
        if cancel_event is not None and cancel_event.is_set():
            raise IndexCancelled()
        dists = _distances(cur, [vectors[i] for i in remaining])
        best = min(range(len(remaining)), key=lambda j: (dists[j], j))
        pick = remaining.pop(best)
        order.append(names[pick])
        cur = vectors[pick]
        progress_callback(len(order), total, f"Added track {len(order)} of {total}")
    return order
