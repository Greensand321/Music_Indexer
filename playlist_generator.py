"""Playlist generation utilities with edge-case handling.

DEFAULT_EXTS defines which audio file extensions are included when building
playlists. Existing playlists are overwritten by default; pass
``overwrite=False`` to skip files that already exist.

Preview-first API (used by the Qt Playlists workspace): the ``plan_*``
functions compute exactly what would be written (one :class:`PlannedPlaylist`
per file, with its final lines) without touching the disk, and
:func:`write_playlist_plan` writes exactly those lines, backing up any file it
overwrites first.
"""

import os, hashlib, difflib, re, shutil, time
from collections import defaultdict
from dataclasses import dataclass, field
from crash_watcher import record_event
from crash_logger import watcher

# Default extensions for playlist generation
DEFAULT_EXTS = {".mp3", ".flac", ".wav", ".aac", ".m4a", ".opus"}
SKIP_SCAN_DIRS = {"trash", "docs", "not sorted", "playlists", "quarantine"}


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
    only_existing=False,
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
    only_existing : bool
        Only touch folder playlists that already exist (never create new
        ones). ``update_playlists`` uses this so that renaming a few files
        does not create a playlist for every folder they live in.
    """
    if log_callback is None:
        log_callback = lambda msg: None
    valid_exts = {e.lower() for e in (valid_exts or DEFAULT_EXTS)}

    playlists_dir = output_dir or os.path.join(root_path, "Playlists")
    if only_existing and not os.path.isdir(playlists_dir):
        return
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
        if only_existing and not os.path.exists(playlist_file):
            continue

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


def _find_library_root(paths) -> str | None:
    """Best guess at the library root for ``paths``: the nearest ancestor of
    their common path that has a ``Playlists`` folder.

    The common path alone is the album folder whenever every change is inside
    one album (the usual case for a rename), and ``<album>/Playlists`` never
    exists — which is why ``update_playlists`` used to do nothing there.
    """
    try:
        common = os.path.commonpath([os.path.abspath(p) for p in paths])
    except ValueError:  # different drives, or empty
        return None
    candidate = common if os.path.isdir(common) else os.path.dirname(common)
    while candidate:
        if os.path.isdir(os.path.join(candidate, "Playlists")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent
    return None


@watcher.traced
def update_playlists(changes, library_root=None, log_callback=None):
    """Update ``.m3u`` playlists based on moved or deleted tracks.

    ``changes`` may be either an iterable of new track paths or a mapping of
    ``old_path -> new_path``. A ``None`` value means the old path was removed
    without a replacement.

    ``library_root`` is the library whose ``Playlists/`` folder is updated.
    Pass it whenever you know it. When omitted, the nearest ancestor of the
    changed paths that has a ``Playlists`` folder is used (the old code looked
    only at the common path itself, so changes confined to one album silently
    updated nothing).

    Returns the number of playlist files rewritten (0 when nothing needed
    changing or no ``Playlists`` folder was found), so callers can report what
    really happened.
    """
    if log_callback is None:
        log_callback = lambda msg: None

    if not changes:
        return 0

    if isinstance(changes, dict):
        move_map = dict(changes)
    else:
        move_map = {p: p for p in changes}

    normalized_map = {
        os.path.normcase(os.path.normpath(old)): new for old, new in move_map.items()
    }

    all_paths = list(move_map.keys()) + [p for p in move_map.values() if p]
    root = library_root or (_find_library_root(all_paths) if all_paths else None)
    if not root:
        log_callback("• No Playlists folder found for the changed tracks; nothing to update")
        return 0

    playlists_dir = os.path.join(root, "Playlists")
    if not os.path.isdir(playlists_dir):
        log_callback(f"• {playlists_dir} does not exist; nothing to update")
        return 0
    record_event("playlist_generator: updating existing playlists")

    name_index = None
    token_index = None
    valid_exts = {e.lower() for e in DEFAULT_EXTS}
    rewritten = 0

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
                    rewritten += 1
                    log_callback(f"✓ Updated playlist {pl_path}")
                except Exception as exc:
                    log_callback(f"! Could not update {pl_path}: {exc}")

    new_tracks = [p for p in move_map.values() if p]
    if new_tracks:
        # Merge moved tracks into folder playlists that already exist; never
        # create new playlists as a side effect of a rename.
        generate_playlists(
            {p: p for p in new_tracks},
            root,
            overwrite=False,
            log_callback=lambda m: None,
            only_existing=True,
        )
    record_event("playlist_generator: playlist update complete")
    return rewritten


# ── Preview-first planning ────────────────────────────────────────────────


@dataclass
class PlannedPlaylist:
    """One playlist file that *would* be written, with its exact contents."""

    path: str
    lines: list = field(default_factory=list)
    exists: bool = False            # an existing file would be overwritten
    entries: int = 0                # track entries (non-comment lines)
    changes: list = field(default_factory=list)     # repair: (old_line, new_line)
    unresolved: list = field(default_factory=list)  # repair: entries still missing
    error: str = ""                 # repair: the playlist could not be read

    @property
    def action(self) -> str:
        return "overwrite" if self.exists else "new"


def _relative_lines(tracks, outfile) -> list:
    base = os.path.dirname(os.path.abspath(outfile))
    return [os.path.relpath(p, base) for p in tracks]


def plan_playlist(tracks, outfile) -> PlannedPlaylist:
    """Plan writing ``tracks`` to ``outfile`` (same format as :func:`write_playlist`)."""
    tracks = list(tracks)
    return PlannedPlaylist(
        path=os.path.abspath(outfile),
        lines=_relative_lines(tracks, outfile),
        exists=os.path.exists(outfile),
        entries=len(tracks),
    )


def plan_folder_playlists(library_root, output_dir=None, valid_exts=None) -> list:
    """Plan one playlist per library folder that directly contains audio files.

    That is one playlist per *album folder* (``Artist/Album`` ->
    ``Artist_Album.m3u``), named exactly as :func:`generate_playlists` names
    them. The reserved top-level folders (Playlists, Docs, Trash, Quarantine,
    Not Sorted, Manual Review) are skipped.
    """
    from utils.reserved_folders import walk_library

    valid_exts = {e.lower() for e in (valid_exts or DEFAULT_EXTS)}
    playlists_dir = output_dir or os.path.join(library_root, "Playlists")
    by_dir = defaultdict(list)
    for dirpath, dirnames, filenames in walk_library(library_root):
        dirnames.sort()
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                by_dir[dirpath].append(os.path.join(dirpath, fname))

    used = set()
    plan = []
    for folder in sorted(by_dir):
        rel = os.path.relpath(folder, library_root)
        name = _sanitize_name(rel, used)
        used.add(name)
        outfile = os.path.join(playlists_dir, f"{name}.m3u")
        item = plan_playlist(by_dir[folder], outfile)
        item.lines = sorted(item.lines)  # generate_playlists writes them sorted
        plan.append(item)
    return plan


def _resolve_entry(entry, playlist_dir):
    is_abs = os.path.isabs(entry)
    path = entry if is_abs else os.path.join(playlist_dir, entry)
    return os.path.normpath(path), is_abs


def plan_playlist_repair(playlist_paths, library_root, prefer_opus=False, log_callback=None) -> list:
    """Plan repairs for ``playlist_paths`` (any folder) against ``library_root``.

    For every entry whose file no longer exists:

    1. a same-name sibling in the other codec is tried first (``.flac`` ->
       ``.opus`` when ``prefer_opus``, otherwise ``.opus`` -> ``.flac``);
    2. otherwise the library is searched by normalised file name
       (:func:`_pick_best_match`).

    Entries that cannot be resolved are kept unchanged and listed in
    ``unresolved``. Nothing is written; pass the items that have ``changes``
    to :func:`write_playlist_plan`.
    """
    if log_callback is None:
        log_callback = lambda msg: None
    valid_exts = {e.lower() for e in DEFAULT_EXTS}
    name_index = token_index = None
    plan = []
    for pl_path in playlist_paths:
        pl_path = os.path.abspath(pl_path)
        pl_dir = os.path.dirname(pl_path)
        item = PlannedPlaylist(path=pl_path, exists=os.path.exists(pl_path))
        try:
            with open(pl_path, "r", encoding="utf-8-sig") as fh:
                raw_lines = [ln.rstrip("\r\n") for ln in fh]
        except OSError as exc:
            item.error = str(exc)
            log_callback(f"! Could not read {pl_path}: {exc}")
            plan.append(item)
            continue

        for raw in raw_lines:
            entry = raw.strip()
            if not entry or entry.startswith("#"):
                item.lines.append(raw)
                continue
            item.entries += 1
            target, was_abs = _resolve_entry(entry, pl_dir)
            if os.path.exists(target):
                item.lines.append(raw)
                continue

            found = None
            stem, ext = os.path.splitext(target)
            ext = ext.lower()
            if ext == ".flac" and prefer_opus and os.path.exists(stem + ".opus"):
                found = stem + ".opus"
            elif ext == ".opus" and not prefer_opus and os.path.exists(stem + ".flac"):
                found = stem + ".flac"
            if found is None:
                if name_index is None:
                    name_index, token_index = _build_audio_index(library_root, valid_exts)
                found = _pick_best_match(target, name_index, token_index)

            if found:
                new_line = found if was_abs else os.path.relpath(found, pl_dir)
                item.lines.append(new_line)
                item.changes.append((entry, new_line))
            else:
                item.lines.append(raw)
                item.unresolved.append(entry)
        plan.append(item)
    return plan


def _backup_path(path: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = f"{path}.{stamp}.bak"
    n = 2
    while os.path.exists(candidate):
        candidate = f"{path}.{stamp}-{n}.bak"
        n += 1
    return candidate


def write_playlist_plan(plan, backup=True, log_callback=None, log_dir=None) -> dict:
    """Write each :class:`PlannedPlaylist` in ``plan`` exactly as planned.

    Any file that exists at write time is first copied to
    ``<name>.<timestamp>.bak`` beside it (when ``backup``). With ``log_dir``
    (normally ``<library>/Docs``) every write and backup is appended to
    ``playlist_writes.log`` there.

    Returns ``{"written": [...], "backups": [...], "errors": [(path, msg)]}``.
    """
    if log_callback is None:
        log_callback = lambda msg: None
    written, backups, errors, log_lines = [], [], [], []
    for item in plan:
        path = item.path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if backup and os.path.exists(path):
                bak = _backup_path(path)
                shutil.copy2(path, bak)
                backups.append(bak)
                log_lines.append(f"BACKUP {path} -> {bak}")
                log_callback(f"• Backed up {os.path.basename(path)} → {os.path.basename(bak)}")
            with open(path, "w", encoding="utf-8") as fh:
                for line in item.lines:
                    fh.write(line + "\n")
            written.append(path)
            log_lines.append(f"WRITE {path} ({item.entries} entries)")
            log_callback(
                f"✓ Wrote {item.entries} entr{'y' if item.entries == 1 else 'ies'} → {path}"
            )
        except OSError as exc:
            errors.append((path, str(exc)))
            log_lines.append(f"ERROR {path}: {exc}")
            log_callback(f"! Could not write {path}: {exc}")
    if log_dir and log_lines:
        try:
            os.makedirs(log_dir, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(log_dir, "playlist_writes.log"), "a", encoding="utf-8") as fh:
                for line in log_lines:
                    fh.write(f"{stamp} {line}\n")
        except OSError as exc:
            log_callback(f"! Could not append to the playlist write log: {exc}")
    return {"written": written, "backups": backups, "errors": errors}
