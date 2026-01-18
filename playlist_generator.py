"""Playlist generation utilities with edge-case handling.

DEFAULT_EXTS defines which audio file extensions are included when building
playlists. Existing playlists are overwritten by default; pass
``overwrite=False`` to skip files that already exist.
"""

import os, hashlib, difflib, re
from collections import defaultdict
from crash_watcher import record_event
from crash_logger import watcher

# Default extensions for playlist generation
DEFAULT_EXTS = {".mp3", ".flac", ".wav", ".aac", ".m4a", ".opus"}
SKIP_SCAN_DIRS = {"trash", "docs", "not sorted", "playlists"}


_COPY_SUFFIX_RE = re.compile(r"(?:\s*-\s*copy|\s+copy)(?:\s*\d+)?$", re.IGNORECASE)
_NUMBER_SUFFIX_RE = re.compile(r"\s*[\(\[\{]\s*\d+\s*[\)\]\}]\s*$")


def _normalize_track_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0].lower()
    base = base.replace("_", " ").replace(".", " ")
    base = _COPY_SUFFIX_RE.sub("", base)
    base = _NUMBER_SUFFIX_RE.sub("", base)
    return re.sub(r"\s+", " ", base).strip()


def _tokenize_name(name: str) -> list[str]:
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", name) if tok]
    return [tok for tok in tokens if not tok.isdigit()]


def _build_audio_index(root: str, valid_exts: set[str]) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    name_index: dict[str, list[str]] = defaultdict(list)
    token_index: dict[str, set[str]] = defaultdict(set)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_parts = set(os.path.relpath(dirpath, root).split(os.sep))
        if rel_parts & SKIP_SCAN_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_SCAN_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts:
                continue
            full_path = os.path.join(dirpath, fname)
            norm_name = _normalize_track_name(full_path)
            if not norm_name:
                continue
            name_index[norm_name].append(full_path)
            for token in _tokenize_name(norm_name):
                token_index[token].add(full_path)
    return name_index, token_index


def _pick_best_match(
    missing_path: str,
    name_index: dict[str, list[str]],
    token_index: dict[str, set[str]],
) -> str | None:
    norm_missing = _normalize_track_name(missing_path)
    if not norm_missing:
        return None
    tokens = set(_tokenize_name(norm_missing))
    candidates = list(name_index.get(norm_missing, []))
    if not candidates and tokens:
        candidate_set: set[str] = set()
        for token in tokens:
            candidate_set.update(token_index.get(token, set()))
        candidates = list(candidate_set)
    if not candidates:
        return None

    orig_dir = os.path.dirname(missing_path)
    orig_ext = os.path.splitext(missing_path)[1].lower()

    def score(path: str) -> tuple[float, float]:
        cand_norm = _normalize_track_name(path)
        cand_tokens = set(_tokenize_name(cand_norm))
        overlap = len(tokens & cand_tokens) / len(tokens) if tokens else 0.0
        ratio = difflib.SequenceMatcher(None, norm_missing, cand_norm).ratio()
        bonus = 0.0
        if os.path.dirname(path) == orig_dir:
            bonus += 0.2
        if orig_ext and os.path.splitext(path)[1].lower() == orig_ext:
            bonus += 0.1
        return overlap + ratio + bonus, overlap

    best_path = None
    best_score = 0.0
    best_overlap = 0.0
    for path in candidates:
        total_score, overlap = score(path)
        if total_score > best_score:
            best_score = total_score
            best_overlap = overlap
            best_path = path

    if best_path is None:
        return None
    if best_overlap < 0.4 and best_score < 1.0:
        return None
    return best_path
DEFAULT_EXTS = {".mp3", ".flac", ".wav", ".aac", ".m4a", ".ogg", ".opus"}


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


@watcher.traced
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
    record_event(f"playlist_generator: generating playlists in {playlists_dir}")

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
    record_event("playlist_generator: playlist generation complete")


def write_playlist(tracks, outfile):
    """Write a simple M3U playlist from ``tracks`` to ``outfile``."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    try:
        with open(outfile, "w", encoding="utf-8") as f:
            for p in tracks:
                f.write(os.path.relpath(p, os.path.dirname(outfile)) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write playlist {outfile}: {e}")


@watcher.traced
def update_playlists(changes):
    """Update ``.m3u`` playlists based on moved or deleted tracks.

    ``changes`` may be either an iterable of new track paths or a mapping of
    ``old_path -> new_path``. A ``None`` value means the old path was removed
    without a replacement.
    """

    if not changes:
        return

    if isinstance(changes, dict):
        move_map = dict(changes)
    else:
        move_map = {p: p for p in changes}

    normalized_map = {
        os.path.normcase(os.path.normpath(old)): new for old, new in move_map.items()
    }

    all_paths = list(move_map.keys()) + [p for p in move_map.values() if p]
    root = os.path.commonpath(all_paths) if all_paths else None
    if not root:
        return

    playlists_dir = os.path.join(root, "Playlists")
    if not os.path.isdir(playlists_dir):
        return
    record_event("playlist_generator: updating existing playlists")

    name_index = None
    token_index = None
    valid_exts = {e.lower() for e in DEFAULT_EXTS}

    for dirpath, _dirs, files in os.walk(playlists_dir):
        for fname in files:
            if not fname.lower().endswith((".m3u", ".m3u8")):
                continue
            pl_path = os.path.join(dirpath, fname)
            try:
                with open(pl_path, "r", encoding="utf-8") as f:
                    lines = [ln.rstrip("\n") for ln in f]
            except Exception:
                continue

            changed = False
            new_lines = []
            for ln in lines:
                if ln.startswith("#"):
                    new_lines.append(ln)
                    continue
                abs_line = os.path.normcase(os.path.normpath(os.path.join(dirpath, ln)))
                if abs_line in normalized_map:
                    new = normalized_map[abs_line]
                    if new is None:
                        changed = True
                        continue
                    rel_new = os.path.relpath(new, dirpath)
                    if rel_new != ln:
                        changed = True
                    new_lines.append(rel_new)
                else:
                    if not os.path.exists(abs_line):
                        if name_index is None or token_index is None:
                            name_index, token_index = _build_audio_index(root, valid_exts)
                        repaired = _pick_best_match(abs_line, name_index, token_index)
                        if repaired:
                            rel_repaired = os.path.relpath(repaired, dirpath)
                            new_lines.append(rel_repaired)
                            changed = True
                            continue
                    new_lines.append(ln)

            if changed:
                try:
                    with open(pl_path, "w", encoding="utf-8") as f:
                        for l in new_lines:
                            f.write(l + "\n")
                except Exception:
                    pass

    new_tracks = [p for p in move_map.values() if p]
    if new_tracks:
        generate_playlists(
            {p: p for p in new_tracks},
            root,
            overwrite=False,
            log_callback=lambda m: None,
        )
    record_event("playlist_generator: playlist update complete")
