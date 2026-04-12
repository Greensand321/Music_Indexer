import os
import json
from typing import Tuple, Dict, Callable, Iterable
from controllers.playlist_controller import save_playlist as _save
from validator import validate_soundvault_structure

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "last_path.txt")
CONFIG_FILE = os.path.normpath(CONFIG_FILE)


def load_last_path() -> str:
    """Return the previously chosen folder path if available."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def save_last_path(path: str) -> None:
    """Persist the given path for next launch."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(path)
    except Exception:
        pass


def count_audio_files(root: str, progress_callback: Callable[[int], None] | None = None) -> int:
    """Return the number of audio files under ``root``."""
    if progress_callback is None:
        def progress_callback(_c: int) -> None:
            pass
    exts = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
    count = 0
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                count += 1
                progress_callback(count)
    return count


def open_library(folder_path: str, progress_callback: Callable[[int], None] | None = None) -> Dict[str, object]:
    """Handle library selection and return basic info."""
    if not folder_path:
        raise ValueError("folder_path is required")
    info: Dict[str, object] = {
        "path": folder_path,
        "name": os.path.basename(folder_path) or folder_path,
    }
    info["song_count"] = count_audio_files(folder_path, progress_callback)
    valid, errors = validate_soundvault_structure(folder_path)
    info["is_valid"] = valid
    info["errors"] = errors
    return info


def save_playlist(library_path: str, name: str, tracks: Iterable[str]) -> str:
    """Save a playlist inside ``library_path`` and update auto playlists."""

    playlists_dir = os.path.join(library_path, "Playlists")
    os.makedirs(playlists_dir, exist_ok=True)
    out_path = os.path.join(playlists_dir, f"{name}.m3u")
    return _save(tracks, out_path)
