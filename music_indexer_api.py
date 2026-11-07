# music_indexer_api.py

import html as _html
import json
import os
import re
import shutil  # used for relocating special folders
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from dry_run_coordinator import DryRunCoordinator
from utils.reserved_folders import RESERVED_FOLDERS

def _verify_dependencies() -> None:
    """Raise RuntimeError if the real mutagen library is missing."""
    try:
        import mutagen  # type: ignore
        if getattr(mutagen.File, "__name__", "") == "<lambda>":
            raise ImportError
    except Exception:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Missing required library: mutagen. Install it via `pip install -r requirements.txt`."
        )
try:
    from mutagen import File as MutagenFile
except Exception:  # pragma: no cover - optional dependency
    def MutagenFile(*_a, **_k):
        return None
from utils.audio_metadata_reader import read_tags
from indexer_control import IndexCancelled, check_cancelled, cancel_event

# ─── CONFIGURATION ─────────────────────────────────────────────────────
COMMON_ARTIST_THRESHOLD = 10
REMIX_FOLDER_THRESHOLD  = 3
# Audio formats the indexer routes. ``.opus`` is part of the app's audio
# contract (see CLAUDE.md); leaving it out used to get Opus tracks swept into
# Trash/ as "non-audio leftovers".
SUPPORTED_EXTS          = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
# Audio-like files the non-audio leftover sweep must never touch, even when the
# indexer cannot route them (they simply stay where they are).
NEVER_SWEPT_EXTS = SUPPORTED_EXTS | {
    ".wma", ".aif", ".aiff", ".aifc", ".ape", ".alac", ".wv", ".mpc", ".tta",
    ".dsf", ".dff", ".mka", ".m4b", ".m4p", ".mp2", ".mp4", ".ac3", ".dts",
    ".amr", ".oga", ".spx", ".caf", ".webm", ".mid", ".midi",
}
# Leftovers with these extensions go to Docs/, everything else to Trash/.
DOC_EXTS = {".txt", ".html", ".db"}
# Folder names skipped at *any* depth (historic indexer rule).
_ANY_DEPTH_SKIP = {"not sorted", "playlists"}

# Files and folders the OS, file browsers and sync tools keep inside music
# folders. They are never swept to Trash/ (Explorer and Finder recreate and
# rewrite them all the time, which also made every preview look stale) and
# the folders are never scanned (old file versions and deleted files live in
# some of them). Compared case-insensitively.
_OS_METADATA_FILES = frozenset({
    "desktop.ini", "thumbs.db", "ehthumbs.db", "ehthumbs_vista.db", ".ds_store",
    ".localized", ".directory", "icon\r", ".picasa.ini", "picasa.ini", ".nomedia",
    ".stignore", ".dropbox", ".dropbox.attr",
})
_OS_METADATA_PREFIXES = ("._", ".syncthing.", "~$")
_OS_METADATA_SUFFIXES = (".!sync", ".!qb", ".crdownload", ".part")
_OS_METADATA_DIRS = frozenset({
    ".stfolder", ".stversions", ".sync", ".dropbox.cache", "@eadir", "#recycle",
    ".spotlight-v100", ".trashes", ".fseventsd", ".temporaryitems",
    ".documentrevisions-v100", "$recycle.bin", "system volume information",
    ".tmp.drivedownload", ".tmp.driveupload", ".git",
})


def is_os_metadata_file(name: str) -> bool:
    """True for OS / file-browser / sync-tool bookkeeping files (desktop.ini,
    Thumbs.db, .DS_Store, ``._`` AppleDouble files, sync partials…)."""
    low = name.casefold()
    return (
        low in _OS_METADATA_FILES
        or low.startswith(_OS_METADATA_PREFIXES)
        or low.endswith(_OS_METADATA_SUFFIXES)
    )


def is_os_metadata_dir(name: str) -> bool:
    """True for OS / sync-tool folders (.stfolder, @eaDir, $RECYCLE.BIN…)."""
    return name.casefold() in _OS_METADATA_DIRS


# ─── Scan helpers ──────────────────────────────────────────────────────

def _path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _same_path(a: str, b: str) -> bool:
    return _path_key(a) == _path_key(b)


def _exact_path(path: str) -> str:
    """Absolute, normalised ``path`` with its letter case kept."""
    return os.path.normpath(os.path.abspath(path))


def _real_moves(moves: dict) -> dict:
    """The moves that change a file's path or name — including a rename that
    only changes letter case, which a case-insensitive comparison misses."""
    return {s: d for s, d in moves.items() if _exact_path(s) != _exact_path(d)}


def _music_root_for(root_path: str) -> str:
    candidate = os.path.join(root_path, "Music")
    return candidate if os.path.isdir(candidate) else root_path


def _iter_scan_dirs(library_root: str, scan_root: str | None = None):
    """``os.walk`` over ``scan_root`` that never enters the reserved folders.

    The reserved top-level folders (``utils.reserved_folders.RESERVED_FOLDERS``:
    Not Sorted, Playlists, Manual Review, Docs, Trash, Quarantine) are pruned
    at the top of the library *and* at the top of ``scan_root`` (the
    ``Music/`` sub-folder when the library has one). ``Not Sorted`` and
    ``Playlists`` are additionally skipped at any depth, as before, and so
    are OS / sync-tool folders (``is_os_metadata_dir``). Directory and file
    names are sorted so plans are deterministic.
    """
    scan_root = scan_root or library_root
    tops = {_path_key(library_root), _path_key(scan_root)}
    reserved = {name.casefold() for name in RESERVED_FOLDERS}
    for dirpath, dirnames, filenames in os.walk(scan_root):
        rel_parts = {p.lower() for p in os.path.relpath(dirpath, library_root).split(os.sep)}
        if _ANY_DEPTH_SKIP & rel_parts:
            dirnames[:] = []
            continue
        if _path_key(dirpath) in tops:
            dirnames[:] = [d for d in dirnames if d.casefold() not in reserved]
        dirnames[:] = [d for d in dirnames if not is_os_metadata_dir(d)]
        dirnames.sort()
        filenames.sort()
        yield dirpath, dirnames, filenames


_INDEXER_FILENAME = re.compile(r"^(?P<artist>.+?)_(?P<track>\d{1,3})_(?P<title>.+)$")


def _artist_title_from_filename(path: str) -> tuple[str, str]:
    """Best-effort ``(artist, title)`` from a file name, or ``("", "")``.

    Understands ``Artist - Title`` and the indexer's own ``Artist_NN_Title``.
    A leading track number (``01 - Title``) is not an artist.
    """
    stem = os.path.splitext(os.path.basename(path))[0].strip()
    match = _INDEXER_FILENAME.match(stem)
    if match:
        return match.group("artist").strip(), match.group("title").strip()
    if " - " in stem:
        artist, title = (part.strip() for part in stem.split(" - ", 1))
        if artist and title and not artist.isdigit():
            return artist, title
    return "", ""

# ─── Helper: build primary_counts for the entire vault ─────────────────────
def _select_canonical_primary(casing_counts: dict[str, dict[str, int]]) -> dict[str, str]:
    canonical = {}
    for normalized, variants in casing_counts.items():
        if not variants:
            continue
        ordered = sorted(
            variants.items(),
            key=lambda item: (-item[1], item[0].casefold(), item[0]),
        )
        canonical[normalized] = ordered[0][0]
    return canonical


def build_primary_counts(root_path, progress_callback=None, phase="A", log_callback=None,
                         library_root=None):
    """
    Walk the entire vault (By Artist, By Year, Incoming, etc.) and
    return a dict mapping each lowercase-normalized artist → total file count,
    plus a canonical casing map for each normalized artist.
    Falls back to filename if metadata “artist” is missing.

    ``progress_callback`` will be invoked every 100 files with
    ``(count, 0, path)`` allowing callers to show progress without
    a separate pre-scan.

    The reserved folders (Quarantine/, Trash/, Docs/, …) are not counted;
    ``library_root`` is the library top when ``root_path`` is its ``Music/``
    sub-folder.
    """
    if progress_callback is None:
        def progress_callback(_c, _t, _m, _p):
            pass
    if log_callback is None:
        def log_callback(_m):
            pass

    counts = {}
    casing_counts = defaultdict(lambda: defaultdict(int))
    idx = 0
    for dirpath, _, files in _iter_scan_dirs(library_root or root_path, root_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue

            path = os.path.join(dirpath, fname)
            tags = get_tags(path)

            idx += 1
            if idx % 100 == 0:
                progress_callback(idx, 0, os.path.relpath(path, root_path), phase)

            # Normalize artist identically to Phase 4
            raw = _normalize_tag_value(tags.get("artist"), "artist", log_callback)

            if not raw:
                name_only = os.path.splitext(fname)[0]
                if "_" in name_only:
                    raw = name_only.split("_", 1)[0]
                elif " - " in name_only:
                    raw = name_only.split(" - ", 1)[0]
                else:
                    raw = name_only

            primary, _ = extract_primary_and_collabs(raw)
            p_lower = primary.lower()
            counts[p_lower] = counts.get(p_lower, 0) + 1
            casing_counts[p_lower][primary] += 1

    return counts, _select_canonical_primary(casing_counts)

# ─── Shared Utility Functions ─────────────────────────────────────────
def _coerce_to_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            try:
                part = "" if item is None else str(item)
            except Exception:
                part = ""
            if part:
                parts.append(part)
        return ", ".join(parts)
    try:
        return str(value)
    except Exception:
        return ""


def sanitize(name: object) -> str:
    """Normalize values for filesystem path segments and HTML/report labels.

    Contract:
    - Accepts any object (including None, numbers, lists).
    - Coerces to string deterministically.
    - Removes filesystem-invalid characters and control chars.
    - Returns "Unknown" when the result is empty.
    - Does not HTML-escape to avoid double-escaping upstream content.
    """
    invalid = r'<>:"/\\|?*'
    raw = _coerce_to_string(name) or "Unknown"
    normalized = unicodedata.normalize("NFKC", raw)
    normalized = re.sub(r"[\x00-\x1f\x7f]", "", normalized)
    cleaned = "".join(c for c in normalized if c not in invalid).strip()
    return cleaned or "Unknown"



def _normalize_tag_value(value: object, field: str, log_callback=None) -> str:
    text = _coerce_to_string(value).strip()
    if not text:
        if log_callback and value not in (None, "", [], (), set()):
            log_callback(f"   ! {field} tag value {value!r} invalid; treating as Unknown")
        return ""
    if log_callback and not isinstance(value, str):
        log_callback(f"   ! {field} tag value {value!r} coerced to '{text}'")
    return text


def _sanitize_tag_value(value: object, field: str, log_callback=None) -> str:
    normalized = _normalize_tag_value(value, field, log_callback)
    sanitized = sanitize(normalized) if normalized else "Unknown"
    if log_callback and normalized and sanitized == "Unknown":
        log_callback(f"   ! {field} tag '{normalized}' sanitized to 'Unknown'")
    return sanitized


def _ensure_unique_destination(
    path: str,
    moves: dict,
    log_callback=None,
    *,
    source_path: str | None = None,
    decision_log: list[str] | None = None,
    taken: set | None = None,
) -> str:
    """Resolve destination collisions deterministically by suffixing names.

    Destinations are compared by ``_path_key`` — case-insensitively where the
    file system is (Windows) — so two tracks whose names differ only in case
    never get "different" destinations that collide at execute. ``taken`` is
    the caller's set of keys of the destinations planned so far (computed
    from ``moves`` when omitted).
    """
    if taken is None:
        taken = {_path_key(v) for v in moves.values()}
    if _path_key(path) not in taken and not os.path.exists(path):
        return path
    if (
        source_path is not None
        and _same_path(path, source_path)
        and _path_key(path) not in taken
    ):
        # The file already sits at its destination (re-running the indexer on
        # an organised library), possibly under a differently-cased name: it
        # is its own "collision", keep the name.
        return path
    base, ext = os.path.splitext(path)
    counter = 1
    candidate = f"{base} ({counter}){ext}"
    while _path_key(candidate) in taken or os.path.exists(candidate):
        counter += 1
        candidate = f"{base} ({counter}){ext}"
    if log_callback:
        source_label = f" for '{os.path.basename(source_path)}'" if source_path else ""
        log_callback(
            f"  ! Destination collision{source_label}; renamed to '{os.path.basename(candidate)}'"
        )
    if decision_log is not None:
        source_label = f" for '{os.path.basename(source_path)}'" if source_path else ""
        decision_log.append(
            f"  ! Destination collision{source_label}; renamed to '{os.path.basename(candidate)}'"
        )
    return candidate


def _cleanup_trailing_numeric_suffixes(
    moves: dict[str, str],
    tag_index: dict[str, dict],
    log_callback=None,
) -> None:
    if log_callback is None:
        def log_callback(_msg):
            pass

    occupied = {_path_key(v) for v in moves.values()}
    suffix_pattern = re.compile(r"^(?P<base>.+) \((?P<num>\d+)\)$")
    for old_path, new_path in list(moves.items()):
        dir_name, filename = os.path.split(new_path)
        stem, ext = os.path.splitext(filename)
        match = suffix_pattern.match(stem)
        if not match:
            continue
        candidate_filename = f"{match.group('base')}{ext}"
        candidate_path = os.path.join(dir_name, candidate_filename)
        if candidate_path == new_path:
            continue
        if _path_key(candidate_path) in occupied or (
            os.path.exists(candidate_path) and not _same_path(candidate_path, old_path)
        ):
            continue
        occupied.discard(_path_key(new_path))
        occupied.add(_path_key(candidate_path))
        moves[old_path] = candidate_path
        if new_path in tag_index:
            tag_index[candidate_path] = tag_index.pop(new_path)
        log_callback(
            f"   • Removed redundant suffix: '{os.path.basename(new_path)}' → "
            f"'{os.path.basename(candidate_path)}'"
        )


def _preview_trailing_suffix_cleanup(
    moves: dict[str, str],
    tag_index: dict[str, dict],
) -> tuple[dict[str, str], dict[str, dict]]:
    preview_moves = dict(moves)
    preview_tag_index = dict(tag_index)
    if not preview_moves:
        return preview_moves, preview_tag_index
    occupied = {_path_key(v) for v in preview_moves.values()}
    suffix_pattern = re.compile(r"^(?P<base>.+) \((?P<num>\d+)\)$")
    for old_path, new_path in list(preview_moves.items()):
        dir_name, filename = os.path.split(new_path)
        stem, ext = os.path.splitext(filename)
        match = suffix_pattern.match(stem)
        if not match:
            continue
        candidate_filename = f"{match.group('base')}{ext}"
        candidate_path = os.path.join(dir_name, candidate_filename)
        if candidate_path == new_path:
            continue
        if _path_key(candidate_path) in occupied or (
            os.path.exists(candidate_path) and not _same_path(candidate_path, old_path)
        ):
            continue
        occupied.discard(_path_key(new_path))
        occupied.add(_path_key(candidate_path))
        preview_moves[old_path] = candidate_path
        if new_path in preview_tag_index:
            preview_tag_index[candidate_path] = preview_tag_index.pop(new_path)
    return preview_moves, preview_tag_index

def collapse_repeats(name: str) -> str:
    """
    If name is something like 'DROELOEDROELOE', collapse to 'DROELOE'.
    For strings shorter than 4 characters, return unchanged.
    """
    if not name or len(name) < 4:
        return name or "Unknown"
    for length in range(1, len(name)//2 + 1):
        if len(name) % length == 0:
            chunk = name[:length]
            if chunk * (len(name)//length) == name:
                return chunk
    return name

def is_repeated(name: str) -> bool:
    """
    Detect if the entire artist tag is a repeated substring,
    e.g. 'DROELOEDROELOE'. For strings shorter than 4, return False.
    """
    if not name or len(name) < 4:
        return False
    for length in range(1, len(name)//2 + 1):
        if len(name) % length == 0:
            chunk = name[:length]
            if chunk * (len(name)//length) == name:
                return True
    return False

def _normalize_disc_value(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, tuple):
        raw_value = raw_value[0] if raw_value else None
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    first = text.split("/", 1)[0].strip()
    if not first:
        return None
    if first.isdigit():
        return str(int(first))
    return first


def _apply_part_of_set(
    base_folder: str,
    album: str,
    disc_value: str | None,
    is_multi_disc: bool,
    decision_log: list[str],
) -> str:
    if is_multi_disc and disc_value and album and os.path.basename(base_folder) == sanitize(album):
        if disc_value.lower().startswith("disc "):
            disc_folder_label = disc_value
        elif disc_value.isdigit():
            disc_folder_label = f"Disc {int(disc_value)}"
        else:
            disc_folder_label = f"Disc {disc_value}"
        part_folder = sanitize(disc_folder_label)
        decision_log.append(
            f"  → Multi-disc album; placed under subfolder '{part_folder}'"
        )
        return os.path.join(base_folder, part_folder)
    return base_folder


def get_tags(path: str):
    """
    Read basic tags using Mutagen (artist, title, album, year, track, genre, disc_value).
    Return a dict with those fields (or None if missing).
    """
    tags = read_tags(path)
    artist = tags.get("artist")
    title = tags.get("title")
    album = tags.get("album")
    year = tags.get("year")
    track = tags.get("track")
    genre = tags.get("genre")
    raw_disc = tags.get("discnumber") or tags.get("disc")
    disc_value = _normalize_disc_value(raw_disc)
    return {
        "artist": artist,
        "title": title,
        "album": album,
        "year": year,
        "track": track,
        "genre": genre,
        "disc_value": disc_value,
    }

def extract_primary_and_collabs(raw_artist: str):
    """
    Use collapse_repeats on raw_artist. Then:
      - If raw_artist contains "/" → split on "/" → first part is primary, rest are collabs.
      - Else look for separators: " feat.", " ft.", " & ", " x ", ", ", ";"
        First segment is primary, remainder split by commas/& is collabs.
      - Else split on uppercase boundary, treat first as primary and rest as a single collab string.
      - Otherwise, primary = raw_artist, collabs = [].
    """
    raw_artist = _coerce_to_string(raw_artist)
    if not raw_artist:
        return ("Unknown", [])
    text = raw_artist.strip()
    text = collapse_repeats(text)

    if "/" in text:
        parts = [collapse_repeats(p.strip()) for p in text.split("/") if p.strip()]
        return (parts[0], parts[1:])

    lowered = text.lower()
    separators = [" feat.", " ft.", " & ", " x ", ", ", ";"]
    for sep in separators:
        idx = lowered.find(sep)
        if idx != -1:
            primary = collapse_repeats(text[:idx].strip())
            rest = text[idx + len(sep):].strip()
            subparts = [collapse_repeats(p.strip()) for p in re.split(r"\s*&\s*|,\s*", rest) if p.strip()]
            return (primary, subparts)

    parts = re.split(r'(?<=[a-z])(?=[A-Z])', text)
    if len(parts) > 1:
        primary = collapse_repeats(parts[0].strip())
        collabs = [" & ".join(collapse_repeats(p.strip()) for p in parts[1:])]
        return (primary, collabs)

    return (text, [])


# (The execute-time artist-tag rewrite that used to live here was removed:
# the Indexer only moves and renames files — tag edits are the Tag Fixer's,
# which previews them first. See docs/features/indexer.md.)


def derive_tags_from_path(filepath: str, music_root: str):
    """
    If a file lives in subfolders under music_root, gather those folder names
    as “inherited tags.” E.g. /…/By Artist/ArtistName/AlbumX/Track.mp3 → {"ArtistName", "AlbumX"}.
    """
    rel = os.path.relpath(os.path.dirname(filepath), music_root)
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    return set(parts)



# ─── A. HELPER: COMPUTE MOVES & TAG INDEX ───────────────────────────────

def compute_moves_and_tag_index(
    root_path,
    log_callback=None,
    progress_callback=None,
    dry_run=False,
    enable_phase_c=False,
    flush_cache=False,
    max_workers=None,
    coord=None,
):
    """
    1) Determine MUSIC_ROOT: if root_path/Music exists, use that; otherwise root_path itself.
    2) Scan for all audio files under MUSIC_ROOT.
    3) Read metadata into `songs` dict, build:
       - album_counts: how many genuine (non-remix) tracks per (artist, album)
       - remix_counts: how many remix‐tagged tracks per (artist, album)
    4) Phase 4: For each entry in `songs`, decide new folder & filename,
       with special‐case for small “(Remixes)” albums.
    Returns:
      - moves: { old_path: new_path, ... }
      - tag_index: { new_path: { "leftover_tags": [...], "old_paths": [...] }, ... }
      - decision_log: list of strings explaining each track’s decision

    ``enable_phase_c``, ``flush_cache``, ``max_workers``, and ``coord`` are
    retained for backward compatibility but are ignored.
    """
    _verify_dependencies()
    if log_callback is None:
        def log_callback(msg):
            pass
    if progress_callback is None:
        def progress_callback(current, total, message, phase):
            pass

    check_cancelled()

    # ─── 1) Determine MUSIC_ROOT ──────────────────────────────────────────────
    MUSIC_ROOT = _music_root_for(root_path)

    # --- Phase 0: Pre-scan entire vault (under MUSIC_ROOT) ---
    global_counts, canonical_primary_map = build_primary_counts(
        MUSIC_ROOT,
        progress_callback,
        phase="A",
        log_callback=log_callback,
        library_root=root_path,
    )
    log_callback(f"   → Pre-scan: found {len(global_counts)} unique artists")
    log_callback(f"   → Music root: {MUSIC_ROOT}")

    # --- Phase 1: Scan for audio files ---
    # The reserved folders (Quarantine/, Trash/, Manual Review/, Docs/, …) are
    # never scanned: re-filing quarantined duplicate losers back into the
    # library silently undid the Duplicate Finder.
    log_callback("1/4: Discovering audio files…")
    all_audio = []
    idx = 0
    for dirpath, _, files in _iter_scan_dirs(root_path, MUSIC_ROOT):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                full = os.path.join(dirpath, fname)
                all_audio.append(full)
                idx += 1
                progress_callback(idx, 0, f"Scanning {os.path.relpath(full, root_path)}", "A")
    total_audio = idx
    log_callback(f"   → Found {total_audio} audio files.")

    # Planned, not created: computing a plan must not touch the library.
    review_root = os.path.join(root_path, "Manual Review")

    # ─── Phase 2: Read metadata & build counters ─────────────────────────────────
    log_callback("2/4: Reading metadata and building counters…")
    songs = {}
    album_counts   = defaultdict(int)
    remix_counts   = defaultdict(int)
    album_disc_values = defaultdict(set)
    total_files = len(all_audio)
    progress_callback(0, total_files, "Reading metadata", "B")

    for idx, fullpath in enumerate(all_audio, start=1):
        if idx % 50 == 0 or idx == total_files:
            log_callback(f"   • Reading metadata {idx}/{total_files}")
        progress_callback(idx, total_files, f"Metadata {os.path.relpath(fullpath, root_path)}", "B")

        data = get_tags(fullpath)
        tag_artist = _normalize_tag_value(data.get("artist"), "artist", log_callback)
        tag_title = _normalize_tag_value(data.get("title"), "title", log_callback)
        name_artist, name_title = ("", "")
        if not (tag_artist and tag_title):
            name_artist, name_title = _artist_title_from_filename(fullpath)
        raw_artist = tag_artist or name_artist
        title = tag_title or name_title
        # Decided *before* the bare-filename fallbacks below, which always
        # succeed (that is why Manual Review routing never happened).
        missing_core = not raw_artist or not title
        stem = os.path.splitext(os.path.basename(fullpath))[0]
        if not raw_artist:
            raw_artist = stem
        raw_artist = collapse_repeats(raw_artist)
        if not title:
            title = stem
        album = _normalize_tag_value(data.get("album"), "album", log_callback)
        year = _normalize_tag_value(data.get("year"), "year", log_callback)
        genre = _normalize_tag_value(data.get("genre"), "genre", log_callback)
        track = data.get("track")
        disc_value = _normalize_tag_value(data.get("disc_value"), "disc_value", log_callback)

        # 1) Primary & collabs (preserve original case)
        primary, collabs = extract_primary_and_collabs(raw_artist)
        p_lower = primary.lower()
        canonical_primary = canonical_primary_map.get(p_lower, primary)

        # 2) album_counts for genuine (non-remix) tracks under each (artist, album)
        if album and "remix" not in title.lower() and album.strip().lower() != title.strip().lower():
            album_counts[(p_lower, album.lower())] += 1

        # 3) remix_counts for tracks whose album tag contains "remix"
        if album and "remix" in album.lower():
            remix_counts[(p_lower, album.lower())] += 1

        # 4) track distinct disc values per (artist, album)
        if album and disc_value:
            album_disc_values[(p_lower, album.lower())].add(disc_value)

        folder_tags = derive_tags_from_path(fullpath, MUSIC_ROOT)

        songs[fullpath] = {
            "raw_artist": raw_artist,
            "primary":    primary,
            "canonical_primary": canonical_primary,
            "collabs":    collabs,
            "title":      title,
            "album":      album,
            "year":       year,
            "genre":      genre,
            "track":      track,
            "disc_value": disc_value,
            "folder_tags": folder_tags,
            "missing_core": missing_core,
        }

    log_callback(f"   → Collected metadata for {total_files} files.")

    # ─── Phase 4: Determine destination for each file (with logging) ─────────────
    log_callback("3/4: Determining destination paths for each file…")
    tag_index = {}
    moves = {}
    # Keys (``_path_key``) of every destination planned so far: collisions
    # are decided case-insensitively where the file system is, and in O(1).
    taken = set()
    decision_log = []
    index = 0
    total = len(songs)

    # Precompute any necessary lookups from songs data
    # (album_counts, remix_counts, etc. were built in Phase 3)
    multi_disc_albums = {
        key for key, disc_values in album_disc_values.items() if len(disc_values) >= 2
    }

    for old_path, info in songs.items():
        index += 1
        if index % 50 == 0 or index == total:
            log_callback(f"   • Determining destination {index}/{total}")

        raw_artist = info["raw_artist"]
        canonical_primary = info.get("canonical_primary")
        title = info["title"] or ""
        album = info["album"] or ""
        year = info["year"] or ""
        track = info["track"]
        disc_value = info.get("disc_value")
        folders    = info["folder_tags"]
        year_label = _sanitize_tag_value(year, "year", log_callback)
        if info.get("missing_core"):
            candidate = os.path.join(review_root, os.path.basename(old_path))
            root_c, ext_c = os.path.splitext(candidate)
            idx_dup = 1
            while os.path.exists(candidate) or _path_key(candidate) in taken:
                candidate = f"{root_c} ({idx_dup}){ext_c}"
                idx_dup += 1
            moves[old_path] = candidate
            taken.add(_path_key(candidate))
            tag_index[candidate] = {
                "leftover_tags": [],
                "old_paths": sorted(folders),
                "canonical_primary": canonical_primary,
            }
            decision_log.append(
                f"  → Missing metadata, placed under Manual Review/{os.path.basename(candidate)}"
            )
            continue

        # Normalize again for consistent global lookup
        primary_norm, _ = extract_primary_and_collabs(raw_artist)
        p_lower = primary_norm.lower()

        # Lookup how many total tracks this artist has across the entire vault
        count_now = global_counts.get(p_lower, 0)
        decision_log.append(
            f"Song: {raw_artist} – {title} ({album}) → global_counts[{raw_artist}] = {count_now}"
        )

        # ─── 4.1) EARLY BAIL-OUT: “Rare” artists go straight to By Year ─────────────────
        if count_now < COMMON_ARTIST_THRESHOLD:
            base_folder = os.path.join(MUSIC_ROOT, "By Year", year_label)
            decision_log.append(
                f"  Early-exit: Count {count_now} < {COMMON_ARTIST_THRESHOLD} → group under By Year/{year_label}"
            )

            # Build the new filename as usual (rename invalid artists or use sanitized tags)
            if "/" in raw_artist or is_repeated(raw_artist):
                new_filename = os.path.basename(old_path)
                decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping filename '{new_filename}'")
            else:
                ext = os.path.splitext(old_path)[1].lower()
                filename_artist = _sanitize_tag_value(raw_artist, "artist", log_callback)
                track_str = f"{track:02d}" if track is not None else "00"
                title_str = _sanitize_tag_value(title, "title", log_callback)
                new_filename = f"{filename_artist}_{track_str}_{title_str}{ext}"
                decision_log.append(f"  Renaming to '{new_filename}' (using raw artist)")

            new_path = os.path.join(base_folder, new_filename)
            new_path = _ensure_unique_destination(
                new_path,
                moves,
                log_callback,
                source_path=old_path,
                decision_log=decision_log,
                taken=taken,
            )
            moves[old_path] = new_path
            taken.add(_path_key(new_path))
            tag_index[new_path] = {
                "leftover_tags": [],
                "old_paths": sorted(folders),
                "canonical_primary": canonical_primary,
            }
            decision_log.append(
                f"  → Final: '{os.path.relpath(new_path, MUSIC_ROOT)}'"
            )
            # Skip all other logic (no remix or album/singles checks)
            continue

        # At this point, count_now ≥ COMMON_ARTIST_THRESHOLD
        # ─── 4.2) SPECIAL CASE: Album ends with “(Remixes)” ──────────────────────────
        if album and album.strip().lower().endswith("(remixes)"):
            rcount = remix_counts.get((p_lower, album.lower()), 0)
            decision_log.append(
                f"  Album '{album}' has {rcount} remix-tagged tracks (threshold={REMIX_FOLDER_THRESHOLD})"
            )

            # Only force into By Artist if both remix-count AND artist-count thresholds are met
            if rcount >= REMIX_FOLDER_THRESHOLD and count_now >= COMMON_ARTIST_THRESHOLD:
                # Promote primary to remixer (first part of raw_artist before “/”)
                main_artist = raw_artist.split("/", 1)[0]
                main_artist = canonical_primary_map.get(main_artist.lower(), main_artist)
                p_lower = main_artist.lower()
                decision_log.append(
                    f"  → Enough remixes ({rcount} ≥ {REMIX_FOLDER_THRESHOLD}) "
                    f"AND count_now ({count_now}) ≥ {COMMON_ARTIST_THRESHOLD}; force primary = '{main_artist}'"
                )
                folder_primary = main_artist
                if canonical_primary and canonical_primary.casefold() == main_artist.casefold():
                    folder_primary = canonical_primary
                artist_folder = os.path.join(
                    MUSIC_ROOT, "By Artist", _sanitize_tag_value(folder_primary, "artist", log_callback)
                )
                base_folder = os.path.join(artist_folder, _sanitize_tag_value(album, "album", log_callback))
                is_multi_disc = (p_lower, album.lower()) in multi_disc_albums if album else False
                base_folder = _apply_part_of_set(base_folder, album, disc_value, is_multi_disc, decision_log)

                # Build filename (skip normal album logic)
                basename = os.path.basename(old_path)
                if "/" in raw_artist or is_repeated(raw_artist):
                    new_filename = basename
                    decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping '{basename}'")
                else:
                    ext = os.path.splitext(old_path)[1].lower()
                    filename_artist = _sanitize_tag_value(raw_artist, "artist", log_callback)
                    track_str = f"{track:02d}" if track is not None else "00"
                    title_str = _sanitize_tag_value(title, "title", log_callback)
                    new_filename = f"{filename_artist}_{track_str}_{title_str}{ext}"
                    decision_log.append(f"  Renaming to '{new_filename}' (using raw artist)")

                new_path = os.path.join(base_folder, new_filename)
                new_path = _ensure_unique_destination(
                    new_path,
                    moves,
                    log_callback,
                    source_path=old_path,
                    decision_log=decision_log,
                    taken=taken,
                )
                moves[old_path] = new_path
                taken.add(_path_key(new_path))
                tag_index[new_path] = {
                    "leftover_tags": [],
                    "old_paths": sorted(folders),
                    "canonical_primary": canonical_primary,
                }
                decision_log.append(
                    f"  → (Remixes folder) placed under By Artist/{main_artist}/{album} "
                    f"→ Final: '{os.path.relpath(new_path, MUSIC_ROOT)}'"
                )
                continue
            else:
                decision_log.append(
                    f"  → Either only {rcount} remixes (< {REMIX_FOLDER_THRESHOLD}) "
                    f"or count_now ({count_now}) < {COMMON_ARTIST_THRESHOLD}; "
                    f"fall back to album/singles logic"
                )

        # ─── 4.3) BROAD COLLABORATOR RANKING (unchanged) ────────────────────────────
        all_candidates = [info["primary"]] + info["collabs"]
        counts = {artist: global_counts.get(artist.lower(), 0) for artist in all_candidates}
        best_artist = max(counts, key=lambda a: counts[a])
        decision_log.append(f"  Candidates: {all_candidates}, counts: {counts}")
        primary = canonical_primary_map.get(best_artist.lower(), best_artist)
        p_lower = primary.lower()
        decision_log.append(f"  → After ranking, primary = '{primary}'")

        # Build “By Artist/<primary>” base path
        folder_primary = primary
        if canonical_primary and canonical_primary.casefold() == primary.casefold():
            folder_primary = canonical_primary
        artist_folder = os.path.join(
            MUSIC_ROOT, "By Artist", _sanitize_tag_value(folder_primary, "artist", log_callback)
        )

        # ─── 4.4) COMMON ARTIST: decide between “Album” vs. “Singles” ───────────────
        # (At this point, count_now ≥ COMMON_ARTIST_THRESHOLD is guaranteed.)
        if not album or album.strip().lower() == title.strip().lower():
            # No album tag (or album == title) → put into “<Artist> - Singles”
            primary_label = _sanitize_tag_value(folder_primary, "artist", log_callback)
            base_folder = os.path.join(artist_folder, f"{primary_label} - Singles")
            decision_log.append(
                f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, no album or album=title "
                f"→ group under '{primary} - Singles'"
            )
        else:
            c2 = album_counts.get((p_lower, album.lower()), 0)
            if c2 > 3:
                # Enough tracks in this album → put into “<Artist>/<Album>”
                base_folder = os.path.join(
                    artist_folder, _sanitize_tag_value(album, "album", log_callback)
                )
                decision_log.append(
                    f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, album_count={c2} > 3 "
                    f"→ group under Album '{album}'"
                )
            else:
                # Few tracks in album → treat as a “Singles” release
                primary_label = _sanitize_tag_value(folder_primary, "artist", log_callback)
                base_folder = os.path.join(artist_folder, f"{primary_label} - Singles")
                decision_log.append(
                    f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, but "
                    f"album_count={c2} ≤ 3 → group under '{primary} - Singles'"
                )

        is_multi_disc = (p_lower, album.lower()) in multi_disc_albums if album else False
        base_folder = _apply_part_of_set(base_folder, album, disc_value, is_multi_disc, decision_log)

        # Build filename for all “common” tracks
        basename = os.path.basename(old_path)
        if "/" in raw_artist or is_repeated(raw_artist):
            new_filename = basename
            decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping '{basename}'")
        else:
            ext = os.path.splitext(old_path)[1].lower()
            filename_artist = _sanitize_tag_value(raw_artist, "artist", log_callback)
            track_str = f"{track:02d}" if track is not None else "00"
            title_str = _sanitize_tag_value(title, "title", log_callback)
            new_filename = f"{filename_artist}_{track_str}_{title_str}{ext}"
            decision_log.append(f"  Renaming to '{new_filename}' (using raw artist)")

        new_path = os.path.join(base_folder, new_filename)
        new_path = _ensure_unique_destination(
            new_path,
            moves,
            log_callback,
            source_path=old_path,
            decision_log=decision_log,
            taken=taken,
        )
        moves[old_path] = new_path
        taken.add(_path_key(new_path))
        decision_log.append(
            f"  → Final: '{os.path.relpath(new_path, MUSIC_ROOT)}'"
        )

        # ─── 4.4) Build “leftover_tags” for HTML preview (optional) ───────────────
        leftover = set(folders)
        leftover.discard(primary)
        if album:
            leftover.discard(album)
        if is_multi_disc and disc_value:
            if disc_value.lower().startswith("disc "):
                disc_folder_label = disc_value
            elif disc_value.isdigit():
                disc_folder_label = f"Disc {int(disc_value)}"
            else:
                disc_folder_label = f"Disc {disc_value}"
            leftover.discard(disc_folder_label)
        if year:
            leftover.discard(year)
        if genre:
            leftover.add(genre)
        sanitized_leftover = sorted(
            {_sanitize_tag_value(tag, "tag", log_callback) for tag in leftover}
        )

        tag_index[new_path] = {
            "leftover_tags": sanitized_leftover,
            "old_paths": sorted(folders),
            "canonical_primary": canonical_primary,
        }

    log_callback("   → Destination paths determined for all files.")
    return moves, tag_index, decision_log


# ─── B. BUILD DRY-RUN HTML ─────────────────────────────────────────────

def render_dry_run_html_from_plan(
    root_path,
    output_html_path,
    moves,
    tag_index,
    plan_items=None,
    *,
    heading_text="Phase A – Exact Metadata",
    title_prefix="Music Index (Dry Run)",
    coord: DryRunCoordinator | None = None,
):
    """Render a dry-run preview from an already computed move plan."""
    coord = coord or DryRunCoordinator()

    def _decision_annotation(decision: str, action: str | None = None):
        normalized_action = (action or "").lower()
        action_label = "(copy)" if normalized_action == "copy" else "(move)"
        mapping = {
            "SKIP_DUPLICATE": ("(duplicate)", "decision-skip"),
            "SKIP_KEEP_EXISTING": ("(kept existing)", "decision-skip"),
            "REVIEW_REQUIRED": ("(review required)", "decision-review"),
            "REPLACE": ("(replace)", "decision-replace"),
            "COPY": (action_label, "decision-copy" if normalized_action == "copy" else "decision-move"),
        }
        return mapping.get(decision.upper())

    def _render_legend(decisions, action_map=None):
        if not decisions:
            return ""
        legend_lines = ["<div class=\"legend\"><strong>Legend:</strong>"]
        descriptions = {
            "SKIP_DUPLICATE": "Identical file already exists; will be skipped.",
            "SKIP_KEEP_EXISTING": "A different file is already present; keeping existing.",
            "REVIEW_REQUIRED": "Needs manual approval before moving.",
            "REPLACE": "Existing file will be replaced.",
            "COPY": "Incoming file will be transferred.",
        }
        for key in ["SKIP_DUPLICATE", "SKIP_KEEP_EXISTING", "REVIEW_REQUIRED", "REPLACE", "COPY"]:
            if key not in decisions:
                continue
            action = None
            if action_map and key in action_map and action_map[key]:
                action = sorted(action_map[key])[0]
            annotation = _decision_annotation(key, action or "move")
            if not annotation:
                continue
            label, css = annotation
            desc = descriptions.get(key, "")
            legend_lines.append(
                f"<div><span class=\"decision-label {css}\">{sanitize(label)}</span> "
                f"<span class=\"legend-note\">{sanitize(desc)}</span></div>"
            )
        legend_lines.append("</div>")
        return "\n".join(legend_lines)

    def build_exact_metadata_section() -> str:
        lines = [f"<h2>{heading_text}</h2>"]
        decision_map = {}
        decision_keys = set()
        decision_actions = {}
        source_map = {}
        if plan_items:
            for item in plan_items:
                dst = item.get("destination")
                src = item.get("source")
                if dst and src:
                    source_map[dst] = src
                decision = (item.get("decision") or "").upper()
                annotation = _decision_annotation(decision, item.get("action"))
                if dst and annotation:
                    decision_map[dst] = annotation
                    decision_keys.add(decision)
                    action_val = (item.get("action") or "").lower()
                    if action_val:
                        decision_actions.setdefault(decision, set()).add(action_val)

        legend_html = _render_legend(decision_keys, decision_actions if decision_actions else None)
        if legend_html:
            lines.append(legend_html)
        lines.append("<pre>")
        lines.append(f"<span class=\"folder\">{sanitize(os.path.basename(root_path))}/</span>")
        tree_nodes = set()
        target_paths = [item.get("destination") for item in plan_items if item.get("destination")] if plan_items else list(moves.values())
        for new_path in target_paths:
            parts = os.path.relpath(new_path, root_path).split(os.sep)
            for i in range(1, len(parts) + 1):
                subtree = os.path.join(root_path, *parts[:i])
                tree_nodes.add(subtree)
        for node in sorted(tree_nodes, key=lambda p: os.path.relpath(p, root_path)):
            rel = os.path.relpath(node, root_path)
            depth = rel.count(os.sep)
            indent = "    " * depth
            _, ext = os.path.splitext(node)
            if os.path.isdir(node) or ext.lower() not in SUPPORTED_EXTS:
                lines.append(f"{indent}<span class=\"folder\">{sanitize(os.path.basename(node))}/</span>")
            else:
                fname = os.path.basename(node)
                leftover = tag_index.get(node, {}).get("leftover_tags", [])
                tags_str = ", ".join(leftover) if leftover else ""
                annotation = decision_map.get(node)
                classes = ["song"]
                label = ""
                if annotation:
                    label_text, css_class = annotation
                    classes.append(css_class)
                    label = f" <span class=\"decision-label {css_class}\">{sanitize(label_text)}</span>"
                source_label = ""
                src_path = source_map.get(node)
                if src_path:
                    source_label = f" <span class=\"source\">(from {sanitize(os.path.basename(src_path))})</span>"
                line = f"{indent}<span class=\"{' '.join(classes)}\">- {sanitize(fname)}{label}{source_label}</span>"
                if tags_str:
                    line += f"  <span class=\"tags\">[{sanitize(tags_str)}]</span>"
                lines.append(line)
        lines.append("</pre>")
        return "\n".join(lines)

    sec_a = build_exact_metadata_section()
    coord.set_html_section('A', sec_a)

    html_body = coord.assemble_final_report()

    with open(output_html_path, "w", encoding="utf-8") as out:
        out.write(
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  "
            f"<title>{title_prefix} – {sanitize(os.path.basename(root_path))}</title>\n  "
            "<style>\n    body { background:#2e3440; color:#d8dee9; font-family:'Courier New', monospace; }\n    pre  { font-size:14px; }\n    .folder { color:#81a1c1; }\n    .song   { color:#a3be8c; }\n    .tags   { color:#88c0d0; font-size:12px; margin-left:1em; }\n    .legend { margin:0 0 8px 0; font-size:13px; color:#e5e9f0; }\n    .legend-note { color:#cfd6e3; font-size:12px; }\n    .decision-label { color:#b5bcc9; font-size:12px; margin-left:0.5em; }\n    .decision-skip { color:#c0c8d8; opacity:0.7; }\n    .decision-review { color:#ebcb8b; }\n    .decision-replace { color:#bf616a; }\n    .decision-copy { color:#8fbcbb; }\n    .decision-move { color:#88c0d0; }\n    .source { color:#cfd6e3; font-size:12px; margin-left:0.5em; }\n  </style>\n</head>\n<body>\n"
        )
        out.write(html_body)
        if html_body and not html_body.endswith("\n"):
            out.write("\n")
        out.write("</body>\n</html>\n")
    return output_html_path


# ─── B2. THE INDEXER PLAN (compute once, preview it, execute *that*) ─────
#
# Preview and execute used to recompute the move plan independently (three
# times per execute), so what ran was not what had been reviewed, and the
# non-audio sweep / folder removals were never previewed at all. Now a single
# ``IndexerPlan`` is computed, rendered as the preview, and executed verbatim;
# execution refuses a plan whose library changed since it was computed.


class StalePlanError(RuntimeError):
    """The library changed since the plan was previewed; nothing was moved."""

    def __init__(self, problems):
        self.problems = list(problems)
        sample = "; ".join(self.problems[:3])
        more = f" (+{len(self.problems) - 3} more)" if len(self.problems) > 3 else ""
        super().__init__(
            f"The library changed since the preview ({len(self.problems)} difference(s): "
            f"{sample}{more}). Nothing was moved — run the preview again."
        )


@dataclass
class IndexerPlan:
    """Everything an Indexer execute will do, computed once at preview time."""

    root_path: str
    music_root: str
    moves: dict                     # audio source -> destination (may be equal)
    tag_index: dict
    decision_log: list
    leftovers: dict = field(default_factory=dict)    # non-audio source -> Trash/ or Docs/ path
    empty_dirs: list = field(default_factory=list)   # folders removed once emptied (deepest first)
    snapshot: dict = field(default_factory=dict)     # source -> (size, mtime_ns) at plan time
    created_at: str = ""

    @property
    def file_moves(self) -> dict:
        """Moves that actually relocate or rename a file.

        A case-only rename (``aurora lines.mp3`` → ``Aurora Lines.mp3``) is a
        move: the preview shows the new name, so execute must apply it
        (``_safe_move`` renames in place on a case-insensitive disk).
        """
        return _real_moves(self.moves)

    @property
    def manual_review(self) -> list:
        review = _path_key(os.path.join(self.root_path, "Manual Review")) + os.sep
        return [d for d in self.moves.values() if _path_key(d).startswith(review)]

    def counts(self) -> dict:
        docs = _path_key(os.path.join(self.root_path, "Docs")) + os.sep
        to_docs = sum(1 for d in self.leftovers.values() if _path_key(d).startswith(docs))
        return {
            "audio_files": len(self.moves),
            "moves": len(self.file_moves),
            "unchanged": len(self.moves) - len(self.file_moves),
            "manual_review": len(self.manual_review),
            "leftovers": len(self.leftovers),
            "leftovers_to_docs": to_docs,
            "leftovers_to_trash": len(self.leftovers) - to_docs,
            "empty_dirs": len(self.empty_dirs),
        }

    def to_dict(self) -> dict:
        return {
            "root_path": self.root_path,
            "music_root": self.music_root,
            "created_at": self.created_at,
            "counts": self.counts(),
            "moves": dict(self.moves),
            "leftovers": dict(self.leftovers),
            "empty_dirs": list(self.empty_dirs),
        }


def _cancel_requested(event=None) -> bool:
    """True when the caller's event or the process-global indexer cancel flag
    (``indexer_control.cancel_event``, still used by legacy callers) is set."""
    return bool(event is not None and event.is_set()) or cancel_event.is_set()


def _raise_if_cancelled(event=None) -> None:
    if _cancel_requested(event):
        raise IndexCancelled()


def _unique_path(path: str, taken: set) -> str:
    """``path``, or ``name (n).ext`` if it exists on disk or is already planned."""
    if _path_key(path) not in taken and not os.path.lexists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        candidate = f"{base} ({n}){ext}"
        if _path_key(candidate) not in taken and not os.path.lexists(candidate):
            return candidate
        n += 1


def _plan_leftovers(root_path: str, music_root: str, moves: dict, exclude=()) -> dict:
    """Non-audio files left behind in folders the plan empties of music.

    A folder is swept only when all of these hold:

    * it is not the library root (nor its ``Music/`` folder) — files sitting
      there are the user's own;
    * it, or a folder inside it, held audio when the plan was made — a folder
      that never contained music (``Artwork/``, ``Scans/``) is the user's own;
    * neither it nor any folder inside it holds audio after the plan (a
      cover.jpg next to, or above, an album that stays put is not a leftover).

    OS / file-browser / sync-tool files (desktop.ini, Thumbs.db, .DS_Store,
    ``._*``, …; ``is_os_metadata_file``) are never swept, and their folders
    are never entered. Each swept file keeps its folder structure under
    ``Trash/`` or ``Docs/`` and gets a ``(n)`` suffix rather than replacing
    anything already there.
    """
    docs_dir = os.path.join(root_path, "Docs")
    trash_dir = os.path.join(root_path, "Trash")
    stop = {_path_key(root_path), _path_key(music_root)}
    moving = {_path_key(s) for s in _real_moves(moves)}
    excluded = {_path_key(p) for p in exclude if p}
    entries = list(_iter_scan_dirs(root_path, music_root))

    def _mark(dir_key: str, into: set) -> None:
        # dir_key and its parents, up to (not including) the library root.
        while dir_key and dir_key not in into and dir_key not in stop:
            into.add(dir_key)
            parent = os.path.dirname(dir_key)
            if parent == dir_key:
                break
            dir_key = parent

    held_audio: set = set()
    audio_after: set = set()
    for dirpath, _dirnames, filenames in entries:
        dir_key = _path_key(dirpath)
        for name in filenames:
            if os.path.splitext(name)[1].lower() in NEVER_SWEPT_EXTS:
                _mark(dir_key, held_audio)
                if _path_key(os.path.join(dirpath, name)) not in moving:
                    _mark(dir_key, audio_after)
    for dest in moves.values():
        _mark(os.path.dirname(_path_key(dest)), audio_after)

    taken: set = set()
    leftovers: dict = {}
    for dirpath, _dirnames, filenames in entries:
        dir_key = _path_key(dirpath)
        if dir_key in stop or dir_key not in held_audio or dir_key in audio_after:
            continue
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in NEVER_SWEPT_EXTS or is_os_metadata_file(fname) or _path_key(full) in excluded:
                continue
            target_root = docs_dir if ext in DOC_EXTS else trash_dir
            dest = _unique_path(os.path.join(target_root, os.path.relpath(full, music_root)), taken)
            taken.add(_path_key(dest))
            leftovers[full] = dest
    return leftovers


def _predict_empty_dirs(root_path: str, music_root: str, moves: dict, leftovers: dict) -> list:
    """Folders that will be empty once the plan's files have left (deepest first)."""
    leaving = {_path_key(s) for s in _real_moves(moves)}
    leaving |= {_path_key(s) for s in leftovers}
    occupied: set = set()
    stop = {_path_key(root_path), _path_key(music_root)}
    for dest in list(moves.values()) + list(leftovers.values()):
        parent = os.path.dirname(_path_key(dest))
        while parent and parent not in occupied and parent not in stop:
            occupied.add(parent)
            nxt = os.path.dirname(parent)
            if nxt == parent:
                break
            parent = nxt
    entries = list(_iter_scan_dirs(root_path, music_root))
    will_empty: dict = {}
    ordered: list = []
    for dirpath, dirnames, filenames in reversed(entries):
        key = _path_key(dirpath)
        if key in stop:
            continue
        empty = (
            key not in occupied
            and all(_path_key(os.path.join(dirpath, f)) in leaving for f in filenames)
            and all(will_empty.get(_path_key(os.path.join(dirpath, d)), False) for d in dirnames)
        )
        will_empty[key] = empty
        if empty:
            ordered.append(dirpath)
    ordered.sort(key=lambda p: p.count(os.sep), reverse=True)
    return ordered


def _snapshot(paths) -> dict:
    snap = {}
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            continue
        snap[path] = (st.st_size, st.st_mtime_ns)
    return snap


def build_indexer_plan(
    root_path,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
    exclude=(),
) -> IndexerPlan:
    """Compute the complete Indexer plan (no file is touched).

    ``cancel_event`` (a ``threading.Event``) is honoured at every progress/log
    callback, as is the process-global ``indexer_control.cancel_event``.
    ``exclude`` lists files that must never be swept (e.g. the report being
    written).
    """
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Library folder not found: {root_path}")
    if log_callback is None:
        def log_callback(_msg):
            pass

    def _log(msg):
        _raise_if_cancelled(cancel_event)
        log_callback(msg)

    def _progress(current, total, message, phase="A"):
        _raise_if_cancelled(cancel_event)
        if progress_callback is not None:
            progress_callback(current, total, message, phase)

    _raise_if_cancelled(cancel_event)
    moves, tag_index, decision_log = compute_moves_and_tag_index(
        root_path, _log, _progress, dry_run=True, coord=None,
    )
    _raise_if_cancelled(cancel_event)
    moves, tag_index = _preview_trailing_suffix_cleanup(moves, tag_index)
    music_root = _music_root_for(root_path)
    leftovers = _plan_leftovers(root_path, music_root, moves, exclude=exclude)
    empty_dirs = _predict_empty_dirs(root_path, music_root, moves, leftovers)
    plan = IndexerPlan(
        root_path=root_path,
        music_root=music_root,
        moves=moves,
        tag_index=tag_index,
        decision_log=list(decision_log),
        leftovers=leftovers,
        empty_dirs=empty_dirs,
        snapshot=_snapshot(list(moves) + list(leftovers)),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    c = plan.counts()
    log_callback(
        f"   → Plan: {c['moves']} move/rename(s), {c['unchanged']} already in place, "
        f"{c['manual_review']} to Manual Review, {c['leftovers']} non-audio leftover(s), "
        f"{c['empty_dirs']} folder(s) to remove."
    )
    return plan


def _rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def _render_plan_extras_html(plan: IndexerPlan) -> str:
    """The parts of an execute the folder tree does not show."""
    esc = _html.escape
    root = plan.root_path
    c = plan.counts()
    lines = ["<h2>Other changes Execute will make</h2>", "<ul>"]
    lines.append(
        f"<li>{c['moves']} audio file(s) moved or renamed; {c['unchanged']} already in place; "
        f"{c['manual_review']} sent to Manual Review/ (missing artist or title).</li>"
    )
    lines.append(
        f"<li>{c['leftovers']} non-audio file(s) moved out of emptied folders "
        f"({c['leftovers_to_trash']} to Trash/, {c['leftovers_to_docs']} to Docs/), "
        "keeping their folder structure. Nothing is overwritten.</li>"
    )
    lines.append(
        "<li>Left where they are: files directly in the library folder, folders that never "
        "held music, and system / sync-tool files (desktop.ini, Thumbs.db, .DS_Store, "
        "._* files, .stfolder/ …).</li>"
    )
    lines.append(f"<li>{c['empty_dirs']} folder(s) removed once empty.</li>")
    lines.append("<li>Files are only moved or renamed: no tag inside any file is changed.</li>")
    lines.append(
        "<li>Existing playlists in Playlists/ that reference moved tracks are updated; "
        "with “Create playlists” on, one playlist per source folder is (re)written.</li>"
    )
    lines.append("<li>Reserved folders (" + ", ".join(f"{esc(n)}/" for n in RESERVED_FOLDERS)
                 + ") are not touched.</li>")
    lines.append("</ul>")
    if plan.leftovers:
        lines.append("<h3>Non-audio files</h3>")
        lines.append("<pre>")
        for src, dst in sorted(plan.leftovers.items()):
            lines.append(f"{esc(_rel(src, root))}  →  {esc(_rel(dst, root))}")
        lines.append("</pre>")
    if plan.empty_dirs:
        lines.append("<h3>Folders removed once empty</h3>")
        lines.append("<pre>")
        for path in plan.empty_dirs:
            lines.append(f"{esc(_rel(path, root))}/")
        lines.append("</pre>")
    return "\n".join(lines)


def write_indexer_preview(plan: IndexerPlan, output_html_path, log_callback=None, log_path=None):
    """Write the preview of ``plan``: the HTML report and, optionally, the
    decision log text file. Writes nothing else."""
    if log_callback is None:
        def log_callback(_msg):
            pass
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write("Indexing Decision Log\n")
            lf.write("======================\n")
            lf.write(f"Library root: {plan.root_path}\n")
            lf.write(f"Generated on: {datetime.now()}\n\n")
            for line in plan.decision_log:
                lf.write(line + "\n")
            if plan.leftovers:
                lf.write("\nNon-audio leftovers (moved on Execute)\n")
                for src, dst in sorted(plan.leftovers.items()):
                    lf.write(f"  {src} -> {dst}\n")
            if plan.empty_dirs:
                lf.write("\nFolders removed once empty (on Execute)\n")
                for path in plan.empty_dirs:
                    lf.write(f"  {path}\n")
        log_callback(f"✓ Detailed log written to: {log_path}")

    log_callback("4/4: Writing dry-run HTML…")
    out_dir = os.path.dirname(output_html_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    coord = DryRunCoordinator()
    coord.set_html_section("B", _render_plan_extras_html(plan))
    render_dry_run_html_from_plan(
        plan.root_path,
        output_html_path,
        plan.moves,
        plan.tag_index,
        coord=coord,
    )
    log_callback(f"✓ Dry-run HTML written to: {output_html_path}")
    return output_html_path


def check_plan_freshness(plan: IndexerPlan) -> list:
    """Differences between the library now and when ``plan`` was computed."""
    problems = []
    for path, sig in plan.snapshot.items():
        if is_os_metadata_file(os.path.basename(path)):
            continue    # Explorer / Finder rewrite these just by opening a folder
        try:
            st = os.stat(path)
        except OSError:
            problems.append(f"missing: {_rel(path, plan.root_path)}")
            continue
        if (st.st_size, st.st_mtime_ns) != tuple(sig):
            problems.append(f"modified: {_rel(path, plan.root_path)}")
    known = {_path_key(p) for p in plan.moves}
    if os.path.isdir(plan.music_root):
        for dirpath, _dirs, files in _iter_scan_dirs(plan.root_path, plan.music_root):
            for fname in files:
                if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTS:
                    continue
                full = os.path.join(dirpath, fname)
                if _path_key(full) not in known:
                    problems.append(f"new: {_rel(full, plan.root_path)}")
    else:
        problems.append(f"missing: {plan.music_root}")
    return problems


def _safe_move(src: str, dst: str) -> str | None:
    """Move ``src`` to ``dst`` without ever overwriting. Returns an error or None."""
    if not os.path.lexists(src):
        return "source missing"
    if os.path.lexists(dst):
        try:
            case_only = os.path.samefile(src, dst)
        except OSError:
            case_only = False
        if not case_only:
            return "destination already exists; not overwritten"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.lexists(dst):  # case-only rename on a case-insensitive disk
            os.rename(src, dst)
        else:
            shutil.move(src, dst)
    except Exception as exc:  # pragma: no cover - OS interaction
        return str(exc)
    return None


def _write_execution_audit(plan: IndexerPlan, actions: list, summary: dict, started: datetime):
    """Record every executed action under Docs/ (JSON + readable text)."""
    docs_dir = os.path.join(plan.root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    stamp = started.strftime("%Y%m%d_%H%M%S")
    json_path = _unique_path(os.path.join(docs_dir, f"indexer_executed_{stamp}.json"), set())
    txt_path = os.path.splitext(json_path)[0] + ".txt"
    payload = {
        "library_root": plan.root_path,
        "plan_created_at": plan.created_at,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {k: v for k, v in summary.items() if k not in ("audit_path", "log_path")},
        "actions": actions,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("Indexer execution log\n=====================\n")
        fh.write(f"Library root: {plan.root_path}\n")
        fh.write(f"Started: {payload['started_at']}   Finished: {payload['finished_at']}\n")
        fh.write(
            f"Moved: {summary['moved']}   Leftovers moved: {summary['leftovers_moved']}   "
            f"Folders removed: {len(summary['removed_dirs'])}   Errors: {len(summary['errors'])}"
            f"{'   CANCELLED' if summary.get('cancelled') else ''}\n\n"
        )
        for act in actions:
            status = act.get("status", "")
            src = act.get("source", "")
            dst = act.get("destination", "")
            reason = f"  ({act['reason']})" if act.get("reason") else ""
            arrow = f" -> {dst}" if dst else ""
            fh.write(f"[{status}] {act.get('action', '')}: {src}{arrow}{reason}\n")
    return json_path, txt_path


def apply_indexer_plan(
    plan: IndexerPlan,
    log_callback=None,
    progress_callback=None,
    create_playlists: bool = True,
    cancel_event=None,
) -> dict:
    """Execute exactly ``plan`` and log every action to Docs/.

    Refuses (``StalePlanError``, nothing moved) when the library changed since
    the plan was computed. Never overwrites a file. Cancellation stops at the
    next file; what was done so far is still logged and playlists are updated
    for it. Returns a summary dict (``moved``, ``errors``, ``leftovers_moved``,
    ``removed_dirs``, ``cancelled``, ``audit_path``, ``log_path`` …).
    """
    if log_callback is None:
        def log_callback(_msg):
            pass
    if progress_callback is None:
        def progress_callback(_c, _t, _p, _ph):
            pass

    _raise_if_cancelled(cancel_event)
    problems = check_plan_freshness(plan)
    if problems:
        raise StalePlanError(problems)

    root_path = plan.root_path
    os.makedirs(os.path.join(root_path, "Not Sorted"), exist_ok=True)
    started = datetime.now()
    summary = {
        "moved": 0,
        "unchanged": len(plan.moves) - len(plan.file_moves),
        "leftovers_moved": 0,
        "removed_dirs": [],
        "errors": [],
        "skipped": [],
        "cancelled": False,
    }
    actions: list = []
    executed_moves: dict = {}
    file_moves = sorted(plan.file_moves.items())
    total = len(file_moves) + len(plan.leftovers)
    step = 0

    def _record(kind, src, dst, error):
        status = "failed" if error else "done"
        actions.append({"action": kind, "source": src, "destination": dst,
                        "status": status, "reason": error or ""})
        if error:
            msg = f"{kind} {src} → {dst}: {error}"
            summary["errors"].append(msg)
            summary["skipped"].append({"source": src, "destination": dst, "reason": error})
            log_callback(f"   ! {msg}")

    try:
        log_callback(f"Executing previewed plan: {len(file_moves)} move(s)…")
        for src, dst in file_moves:
            if _cancel_requested(cancel_event):
                summary["cancelled"] = True
                break
            step += 1
            progress_callback(step, total, src, "C")
            if step % 50 == 0 or step == len(file_moves):
                log_callback(f"   • Moving file {step}/{len(file_moves)}")
            error = _safe_move(src, dst)
            _record("move", src, dst, error)
            if error is None:
                executed_moves[src] = dst
                summary["moved"] += 1
                # Execute only moves and renames files: tags are never
                # rewritten (that was never previewed; the Tag Fixer does it).

        if executed_moves:
            try:
                from playlist_generator import update_playlists
                log_callback("   • Updating existing playlists for renamed tracks…")
                update_playlists(executed_moves)
            except ImportError:
                log_callback("! playlist_generator.py missing; skipping playlist updates.")
            except Exception as e:
                log_callback(f"! Playlist update error: {e}")

        if not summary["cancelled"]:
            for src, dst in sorted(plan.leftovers.items()):
                if _cancel_requested(cancel_event):
                    summary["cancelled"] = True
                    break
                step += 1
                progress_callback(step, total, src, "C")
                error = _safe_move(src, dst)
                _record("move_leftover", src, dst, error)
                if error is None:
                    summary["leftovers_moved"] += 1
                    log_callback(f"Moved leftover: {_rel(src, root_path)} → {_rel(dst, root_path)}")

        if not summary["cancelled"]:
            for dirpath in plan.empty_dirs:
                try:
                    if os.path.isdir(dirpath) and not os.listdir(dirpath):
                        os.rmdir(dirpath)
                        summary["removed_dirs"].append(dirpath)
                        actions.append({"action": "remove_empty_folder", "source": dirpath,
                                        "destination": "", "status": "done", "reason": ""})
                        log_callback(f"Removed empty folder: {dirpath}")
                except OSError:
                    pass

        if create_playlists and not summary["cancelled"]:
            # ─── Generate playlists for what actually moved ──────────────
            try:
                from playlist_generator import generate_playlists
                log_callback("4/4: Generating playlists…")

                def plog(msg):
                    log_callback(msg)

                generate_playlists(
                    executed_moves,
                    root_path,
                    output_dir=None,        # default: root_path/Playlists
                    valid_exts=None,        # default extensions
                    overwrite=True,
                    log_callback=plog,
                )
                log_callback("✓ Robust playlists created.")
            except ImportError:
                log_callback("! playlist_generator.py missing; skipping playlists.")
            except Exception as e:
                log_callback(f"! Playlist generation error: {e}")
        elif not create_playlists:
            log_callback("• Playlist creation disabled")
        if summary["cancelled"]:
            log_callback("✘ Execution cancelled; the actions done so far are logged in Docs/.")
    finally:
        try:
            audit_path, log_path = _write_execution_audit(plan, actions, summary, started)
            summary["audit_path"] = audit_path
            summary["log_path"] = log_path
            log_callback(f"✓ Execution log written to: {log_path}")
        except Exception as exc:  # pragma: no cover - OS interaction
            log_callback(f"! Failed to write the execution log: {exc}")
    return summary


def build_dry_run_html(
    root_path,
    output_html_path,
    log_callback=None,
    enable_phase_c=False,
    flush_cache=False,
    max_workers=None,
):
    """
    Compute the Indexer plan and write it as a dry-run HTML to
    ``output_html_path``. Does NOT move any files.

    ``enable_phase_c``, ``flush_cache`` and ``max_workers`` are accepted for
    backward compatibility and ignored.
    """
    plan = build_indexer_plan(root_path, log_callback, None, exclude=[output_html_path])
    write_indexer_preview(plan, output_html_path, log_callback)
    return {"moved": 0, "html": output_html_path, "dry_run": True, "plan": plan}


# ─── C. APPLY MOVES ─────────────────────────────────────────────────────

def apply_indexer_moves(
    root_path,
    log_callback=None,
    progress_callback=None,
    create_playlists: bool = True,
    enable_phase_c: bool = False,
    flush_cache: bool = False,
    max_workers: int | None = None,
    cancel_event=None,
):
    """Compute the plan and execute it immediately (legacy entry point).

    Prefer ``build_indexer_plan`` → preview → ``apply_indexer_plan`` so the
    executed plan is the reviewed one. Returns the ``apply_indexer_plan``
    summary (``{"moved": <count>, "errors": [...], ...}``).
    """
    plan = build_indexer_plan(root_path, log_callback, progress_callback, cancel_event)
    return apply_indexer_plan(
        plan,
        log_callback,
        progress_callback,
        create_playlists=create_playlists,
        cancel_event=cancel_event,
    )


# ─── D. HIGH-LEVEL “RUN FULL INDEXER” ───────────────────────────────────

def run_full_indexer(
    root_path,
    output_html_path,
    dry_run_only: bool = False,
    log_callback=None,
    progress_callback=None,
    enable_phase_c: bool = False,
    flush_cache: bool = False,
    max_workers: int | None = None,
    create_playlists: bool = True,
    cancel_event=None,
):
    """
    1) Compute the Indexer plan once (``build_indexer_plan``).
    2) Write ``Docs/indexer_log.txt`` and the dry-run HTML from that plan.
    3) If ``dry_run_only`` is False, execute *that same plan*
       (``apply_indexer_plan``). Playlist generation can be skipped by passing
       ``create_playlists=False``.
    Returns ``{"moved", "html", "dry_run", "plan"}`` plus, after an execute,
    ``"summary"``, ``"errors"`` and ``"cancelled"``.

    ``enable_phase_c``, ``flush_cache`` and ``max_workers`` are accepted for
    backward compatibility and ignored.
    """
    if log_callback is None:
        def log_callback(msg): pass

    plan = build_indexer_plan(
        root_path, log_callback, progress_callback, cancel_event, exclude=[output_html_path]
    )
    write_indexer_preview(
        plan,
        output_html_path,
        log_callback,
        log_path=os.path.join(root_path, "Docs", "indexer_log.txt"),
    )

    if dry_run_only:
        return {"moved": 0, "html": output_html_path, "dry_run": True, "plan": plan}

    summary = apply_indexer_plan(
        plan,
        log_callback,
        progress_callback,
        create_playlists=create_playlists,
        cancel_event=cancel_event,
    )
    return {
        "moved": summary["moved"],
        "html": output_html_path,
        "dry_run": False,
        "plan": plan,
        "summary": summary,
        "errors": summary["errors"],
        "cancelled": summary["cancelled"],
    }


# ─── E. DUPLICATE DETECTION HELPER ─────────────────────────────────────

def find_duplicates(root_path, log_callback=None):
    """Return list of (original_path, duplicate_path) that would be marked as
    duplicates by the indexer.

    Duplicate detection and keep/winner ranking are intentionally excluded
    from the indexer. The indexer treats every file independently and relies
    on deterministic routing plus collision handling (suffixing/buckets).
    """
    if log_callback:
        log_callback("Duplicate detection is disabled in the indexer.")
    return []


def main(argv: List[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SoundVault Music Indexer")
    parser.add_argument("root", help="Library root path")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--enable-phase-c", action="store_true", help="(Ignored) kept for backward compatibility")
    parser.add_argument("--flush-cache", action="store_true", help="(Ignored) kept for backward compatibility")
    parser.add_argument("--max-workers", type=int, default=None, help="(Ignored) kept for backward compatibility")
    parser.add_argument("--no-playlists", action="store_true", help="Skip playlist creation")
    args = parser.parse_args(argv)

    output = os.path.join(args.root, "Docs", "MusicIndex.html")
    if args.dry_run:
        build_dry_run_html(
            args.root,
            output,
            print,
            enable_phase_c=args.enable_phase_c,
            flush_cache=args.flush_cache,
            max_workers=args.max_workers,
        )
    else:
        run_full_indexer(
            args.root,
            output,
            dry_run_only=False,
            log_callback=print,
            progress_callback=None,
            enable_phase_c=args.enable_phase_c,
            flush_cache=args.flush_cache,
            max_workers=args.max_workers,
            create_playlists=not args.no_playlists,
        )


if __name__ == "__main__":
    from crash_logger import install as install_crash_logger

    install_crash_logger()
    main()
