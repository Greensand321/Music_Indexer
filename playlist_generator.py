"""Playlist generation utilities with edge-case handling.

DEFAULT_EXTS defines which audio file extensions are included when building
playlists. Existing playlists are overwritten by default; pass
``overwrite=False`` to skip files that already exist.
"""

import os, hashlib
from collections import defaultdict

# Default extensions for playlist generation
DEFAULT_EXTS = {".mp3", ".flac", ".wav", ".aac", ".m4a"}


def _sanitize_name(rel_path, existing):
    """Sanitize a relative path for playlist filename.

    Replaces path separators with underscores and appends a short hash if
    the sanitized name already exists in ``existing``.
    """
    base = rel_path.replace(os.sep, "_") or "root"
    if base not in existing:
        return base
    h = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:6]
    return f"{base}_{h}"


def generate_playlists(
    moves,
    root_path,
    output_dir=None,
    valid_exts=None,
    overwrite=True,
    log_callback=None,
):
    """Generate M3U playlists based on move mapping.

    Parameters
    ----------
    moves : dict
        Mapping of original file paths to their new locations.
    root_path : str
        Root of the music library.
    output_dir : str or None
        Directory to write playlists into. Defaults to ``root_path/Playlists``.
    valid_exts : set or None
        Extensions to include. Defaults to ``DEFAULT_EXTS``.
    overwrite : bool
        Whether to overwrite existing playlists.
    log_callback : callable
        Function for logging messages.
    """
    if log_callback is None:
        log_callback = lambda msg: None
    valid_exts = {e.lower() for e in (valid_exts or DEFAULT_EXTS)}

    playlists_dir = output_dir or os.path.join(root_path, "Playlists")
    os.makedirs(playlists_dir, exist_ok=True)

    dir_map = defaultdict(list)
    for old, new in moves.items():
        ext = os.path.splitext(new)[1].lower()
        if ext in valid_exts:
            dir_map[os.path.dirname(old)].append(new)

    used = set()
    for old_dir, files in dir_map.items():
        if not files:
            log_callback(f"\u26A0 Skipping empty folder: {old_dir}")
            continue

        rel = os.path.relpath(old_dir, root_path)
        name = _sanitize_name(rel, used)
        used.add(name)

        playlist_file = os.path.join(playlists_dir, f"{name}.m3u")

        rel_files = {os.path.relpath(p, playlists_dir) for p in files}

        if os.path.exists(playlist_file):
            if not overwrite:
                try:
                    with open(playlist_file, "r", encoding="utf-8") as f:
                        existing = {line.strip() for line in f if line.strip()}
                except Exception as e:
                    log_callback(f"\u2717 Failed to read {playlist_file}: {e}")
                    existing = set()
                rel_files.update(existing)
            log_callback(f"\u2192 Writing playlist: {playlist_file}")
        else:
            log_callback(f"\u2192 Writing playlist: {playlist_file}")

        try:
            with open(playlist_file, "w", encoding="utf-8") as f:
                for p in sorted(rel_files):
                    f.write(p + "\n")
        except Exception as e:
            log_callback(f"\u2717 Failed to write {playlist_file}: {e}")


def write_playlist(tracks, outfile):
    """Write a simple M3U playlist from ``tracks`` to ``outfile``."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    try:
        with open(outfile, "w", encoding="utf-8") as f:
            for p in tracks:
                f.write(os.path.relpath(p, os.path.dirname(outfile)) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write playlist {outfile}: {e}")


def update_playlists(new_tracks):
    """Placeholder for playlist update hook used by Library Sync."""
    # The real implementation would update any smart playlists
    # that should include ``new_tracks``. For now we simply call
    # ``generate_playlists`` treating each parent folder as a playlist.
    root = os.path.commonpath(new_tracks) if new_tracks else None
    if not root:
        return
    moves = {p: p for p in new_tracks}
    generate_playlists(moves, root, overwrite=False, log_callback=lambda m: None)

