"""Helpers for generating clustered playlists."""

import os
from playlist_generator import DEFAULT_EXTS
from clustered_playlists import generate_clustered_playlists


def gather_tracks(library_path: str) -> list[str]:
    """Return list of audio file paths under ``library_path``."""
    music_root = (
        os.path.join(library_path, "Music")
        if os.path.isdir(os.path.join(library_path, "Music"))
        else library_path
    )
    tracks: list[str] = []
    for dirpath, _, files in os.walk(music_root):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in DEFAULT_EXTS:
                tracks.append(os.path.join(dirpath, fname))
    return tracks


def cluster_library(library_path: str, method: str, num: int, log_callback) -> None:
    """Generate clustered playlists for ``library_path``."""
    tracks = gather_tracks(library_path)
    log_path = os.path.join(library_path, f"{method}_log.txt")

    def log(msg: str) -> None:
        log_callback(msg)
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception:
            pass

    log(f"Found {len(tracks)} audio files")
    params = {"n_clusters": num} if method == "kmeans" else {"min_cluster_size": num}
    generate_clustered_playlists(tracks, library_path, method, params, log)
