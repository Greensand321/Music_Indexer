"""Helpers for generating clustered playlists."""

import os
from playlist_generator import DEFAULT_EXTS
from playlist_engine_kmeans import KMeansPlaylistEngine
from playlist_engine_hdbscan import HDBSCANPlaylistEngine


def gather_tracks(library_path: str, folder_filter: dict | None = None) -> list[str]:
    """Return list of audio file paths under ``library_path`` respecting ``folder_filter``."""

    music_root = (
        os.path.join(library_path, "Music")
        if os.path.isdir(os.path.join(library_path, "Music"))
        else library_path
    )

    include: list[str] = []
    exclude: list[str] = []
    if folder_filter:
        include = [os.path.abspath(p) for p in folder_filter.get("include", [])]
        exclude = [os.path.abspath(p) for p in folder_filter.get("exclude", [])]

    def iter_dirs() -> list[str]:
        if include:
            for root in include:
                if os.path.isdir(root):
                    yield from [root]
        else:
            yield music_root

    tracks_set: set[str] = set()
    for start in iter_dirs():
        for dirpath, dirs, files in os.walk(start):
            abs_dir = os.path.abspath(dirpath)
            if any(abs_dir.startswith(e) for e in exclude):
                dirs[:] = []
                continue
            for fname in files:
                if os.path.splitext(fname)[1].lower() in DEFAULT_EXTS:
                    tracks_set.add(os.path.join(dirpath, fname))

    return sorted(tracks_set)


def cluster_library(
    library_path: str,
    method: str,
    cluster_params: dict,
    log_callback,
    folder_filter: dict | None = None,
    engine: str = "librosa",
    feature_loader=None,
) -> tuple[list[str], list]:
    """Generate clustered playlists for ``library_path`` and return features."""

    engine_impl = HDBSCANPlaylistEngine() if method == "hdbscan" else KMeansPlaylistEngine()

    tracks = gather_tracks(library_path, folder_filter)
    log_path = os.path.join(library_path, f"{method}_log.txt")

    def log(msg: str) -> None:
        log_callback(msg)
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception:
            pass

    log(f"Found {len(tracks)} audio files")
    feats = engine_impl.generate(
        tracks,
        library_path,
        cluster_params,
        log,
        engine=engine,
        feature_loader=feature_loader,
    )
    return tracks, feats
