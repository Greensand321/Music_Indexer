# music_indexer_api.py

import os
import re
import shutil  # used for relocating special folders
import hashlib
import base64
from collections import defaultdict
from typing import Dict, List
from dry_run_coordinator import DryRunCoordinator
from config import MIXED_CODEC_THRESHOLD_BOOST, load_config

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
from utils.audio_metadata_reader import read_metadata, read_tags
from indexer_control import check_cancelled, cancel_event

# ─── CONFIGURATION ─────────────────────────────────────────────────────
COMMON_ARTIST_THRESHOLD = 10
REMIX_FOLDER_THRESHOLD  = 3
SUPPORTED_EXTS          = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
DEFAULT_FUZZY_FP_THRESHOLD = 0.1  # allowed fingerprint difference ratio
DEDUP_KEEP_DELTA_THRESHOLD = 50  # score difference to auto-delete duplicates
FINGERPRINT_TIMEOUT = 30          # seconds before fingerprint worker is timed out

# ─── Helper: build primary_counts for the entire vault ─────────────────────
def build_primary_counts(root_path, progress_callback=None, phase="A"):
    """
    Walk the entire vault (By Artist, By Year, Incoming, etc.) and
    return a dict mapping each lowercase-normalized artist → total file count.
    Falls back to filename if metadata “artist” is missing.

    ``progress_callback`` will be invoked every 100 files with
    ``(count, 0, path)`` allowing callers to show progress without
    a separate pre-scan.
    """
    if progress_callback is None:
        def progress_callback(_c, _t, _m, _p):
            pass

    counts = {}
    idx = 0
    for dirpath, _, files in os.walk(root_path):
        rel_dir = os.path.relpath(dirpath, root_path)
        if {"not sorted", "playlists"} & {p.lower() for p in rel_dir.split(os.sep)}:
            continue
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
            raw = (tags.get("artist") or "").strip()

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

    return counts

# ─── Shared Utility Functions ─────────────────────────────────────────
def sanitize(name: str) -> str:
    invalid = r'<>:"/\\|?*'
    return "".join(c for c in (name or "Unknown") if c not in invalid).strip() or "Unknown"

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

def _normalize_part_of_set(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    first = text.split("/", 1)[0].strip()
    if not first:
        return None
    if first.isdigit():
        return f"Part {int(first)}"
    return first


def _apply_part_of_set(
    base_folder: str,
    album: str,
    part_of_set: str | None,
    decision_log: list[str],
) -> str:
    if part_of_set and album and os.path.basename(base_folder) == sanitize(album):
        part_folder = sanitize(part_of_set)
        decision_log.append(
            f"  → Part of set '{part_of_set}' detected; placed under subfolder '{part_folder}'"
        )
        return os.path.join(base_folder, part_folder)
    return base_folder


def get_tags(path: str):
    """
    Read basic tags using Mutagen (artist, title, album, year, track, genre, part_of_set).
    Return a dict with those fields (or None if missing).
    """
    tags = read_tags(path)
    artist = tags.get("artist")
    title = tags.get("title")
    album = tags.get("album")
    year = tags.get("year")
    track = tags.get("track")
    genre = tags.get("genre")
    raw_part_of_set = tags.get("discnumber") or tags.get("disc")
    part_of_set = _normalize_part_of_set(raw_part_of_set)
    return {
        "artist": artist,
        "title": title,
        "album": album,
        "year": year,
        "track": track,
        "genre": genre,
        "part_of_set": part_of_set,
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

def derive_tags_from_path(filepath: str, music_root: str):
    """
    If a file lives in subfolders under music_root, gather those folder names
    as “inherited tags.” E.g. /…/By Artist/ArtistName/AlbumX/Track.mp3 → {"ArtistName", "AlbumX"}.
    """
    rel = os.path.relpath(os.path.dirname(filepath), music_root)
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    return set(parts)


def _parse_fp(fp: str) -> tuple[str, list[int] | bytes] | None:
    """Parse fingerprint string into integer list or decoded bytes."""
    text = fp.strip()
    if not text:
        return None
    try:
        arr = [int(x) for x in text.replace(',', ' ').split()]
        if arr:
            return ("ints", arr)
    except Exception:
        pass
    try:
        data = base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))
        if data:
            return ("bytes", data)
    except Exception:
        pass
    return None


def _hamming_ints(a: list[int], b: list[int]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = sum(x != y for x, y in zip(a[:n], b[:n]))
    return diff / n


def _hamming_bytes(a: bytes, b: bytes) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = 0
    for i in range(n):
        diff += (a[i] ^ b[i]).bit_count()
    return diff / (n * 8)


def fingerprint_distance(fp1: str | None, fp2: str | None) -> float:
    """Return normalized Hamming distance between two fingerprint strings."""
    if not fp1 or not fp2:
        return 1.0
    if fp1 == fp2:
        return 0.0
    p1 = _parse_fp(fp1)
    p2 = _parse_fp(fp2)
    if not p1 or not p2 or p1[0] != p2[0]:
        return 1.0
    kind, a = p1
    _, b = p2
    if kind == "ints":
        return _hamming_ints(a, b)  # type: ignore[arg-type]
    else:
        return _hamming_bytes(a, b)  # type: ignore[arg-type]


def _keep_score(path: str, info: dict, ext_priority: dict) -> float:
    """Compute keep score for a file based on extension, metadata and filename."""
    ext = os.path.splitext(path)[1].lower()
    pri = ext_priority.get(ext, 99)
    ext_score = 1000.0 / (pri + 1)
    meta_score = info.get("meta_count", 0) * 10
    fname_score = len(os.path.splitext(os.path.basename(path))[0])
    return ext_score + meta_score + fname_score




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
    3) Deduplicate tracks by fingerprint only and delete lower-priority duplicates.
       If ``dry_run`` is True, deduplication actions are simulated only and no
       files are modified.
    4) Read metadata into `songs` dict, build:
       - album_counts: how many genuine (non-remix) tracks per (artist, album)
       - remix_counts: how many remix‐tagged tracks per (artist, album)
       - cover_counts: how many files share each embedded cover
    5) Phase 4: For each entry in `songs`, decide new folder & filename,
       with special‐case for small “(Remixes)” albums.
    Returns:
      - moves: { old_path: new_path, ... }
      - tag_index: { new_path: { "leftover_tags": [...], "old_paths": [...] }, ... }
      - decision_log: list of strings explaining each track’s decision

    ``coord`` can be provided to collect additional diagnostic data during
    near-duplicate detection.
    """
    _verify_dependencies()
    if log_callback is None:
        def log_callback(msg):
            pass
    if progress_callback is None:
        def progress_callback(current, total, message, phase):
            pass

    cfg = load_config()
    fuzzy_fp_threshold = float(cfg.get("fuzzy_fp_threshold", DEFAULT_FUZZY_FP_THRESHOLD))
    check_cancelled()

    # ─── 1) Determine MUSIC_ROOT ──────────────────────────────────────────────
    MUSIC_ROOT = os.path.join(root_path, "Music") \
        if os.path.isdir(os.path.join(root_path, "Music")) \
        else root_path

    SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}

    docs_dir = os.path.join(root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    db_path = os.path.join(docs_dir, ".soundvault.db")
    if flush_cache:
        from fingerprint_cache import flush_cache as _flush
        _flush(db_path)

    # --- Phase 0: Pre-scan entire vault (under MUSIC_ROOT) ---
    global_counts = build_primary_counts(MUSIC_ROOT, progress_callback, phase="A")
    log_callback(f"   → Pre-scan: found {len(global_counts)} unique artists")
    log_callback(f"   → DEBUG: MUSIC_ROOT = {MUSIC_ROOT}")
    log_callback(f"   → DEBUG: droeloe count = {global_counts.get('droeloe', 0)}")

    # --- Phase 1: Scan for audio files ---
    log_callback("1/6: Discovering audio files…")
    all_audio = []
    idx = 0
    for dirpath, _, files in os.walk(MUSIC_ROOT):
        rel_dir = os.path.relpath(dirpath, root_path)
        if {"not sorted", "playlists"} & {p.lower() for p in rel_dir.split(os.sep)}:
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                full = os.path.join(dirpath, fname)
                if {"not sorted", "playlists"} & {p.lower() for p in os.path.relpath(full, root_path).split(os.sep)}:
                    continue
                all_audio.append(full)
                idx += 1
                progress_callback(idx, 0, f"Scanning {os.path.relpath(full, root_path)}", "A")
    total_audio = idx
    log_callback(f"   → Found {total_audio} audio files.")

    # ─── Phase 2: Deduplicate using fingerprints ─────────────────────────────
    log_callback("2/6: Deduplicating by fingerprint…")
    EXT_PRIORITY = {".flac": 0, ".m4a": 1, ".aac": 1, ".mp3": 2, ".wav": 3, ".ogg": 4}

    path_fps: Dict[str, str | None] = {}
    from fingerprint_cache import get_fingerprint

    def _compute(path: str) -> tuple[int | None, str | None]:
        try:
            import acoustid
            return acoustid.fingerprint_file(path)
        except Exception:
            return None, None

    progress_callback(0, len(all_audio), "Fingerprinting", "A")
    for idx, p in enumerate(all_audio, start=1):
        fp_val = get_fingerprint(p, db_path, _compute)
        path_fps[p] = fp_val
        progress_callback(idx, len(all_audio), os.path.relpath(p, root_path), "A")
        if idx % 50 == 0 or idx == len(all_audio):
            log_callback(f"   • Fingerprinting {idx}/{len(all_audio)}")

    file_infos: Dict[str, Dict[str, str | None]] = {}
    for idx, fullpath in enumerate(all_audio, start=1):
        if idx % 50 == 0 or idx == total_audio:
            log_callback(f"   • Processing file {idx}/{total_audio} for dedupe")
        data = get_tags(fullpath)
        raw_artist = data["artist"] or os.path.splitext(os.path.basename(fullpath))[0]
        raw_artist = collapse_repeats(raw_artist)
        title = data["title"] or os.path.splitext(os.path.basename(fullpath))[0]
        album = data["album"] or ""
        meta_count = sum(
            1
            for t in [data.get("artist"), data.get("title"), data.get("album"), data.get("track"), data.get("year"), data.get("genre")]
            if t
        )
        primary, _ = extract_primary_and_collabs(raw_artist)
        fp = path_fps.get(fullpath)
        if isinstance(fp, (bytes, bytearray)):
            try:
                fp = fp.decode("utf-8")
            except Exception:
                fp = fp.decode("latin1", errors="ignore")
        ext = os.path.splitext(fullpath)[1].lower()
        file_infos[fullpath] = {
            "primary": primary.lower(),
            "title": title.lower(),
            "album": album.lower(),
            "fp": fp,
            "meta_count": meta_count,
            "ext": ext,
        }

    to_delete: Dict[str, str] = {}
    fp_groups: Dict[str, List[str]] = defaultdict(list)
    for path, info in file_infos.items():
        if info["fp"]:
            fp_groups[info["fp"]].append(path)

    kept_files = set(file_infos.keys())
    review_root = os.path.join(root_path, "Manual Review")
    os.makedirs(review_root, exist_ok=True)

    for fp, paths in fp_groups.items():
        if len(paths) <= 1:
            continue
        scored = sorted(paths, key=lambda p: _keep_score(p, file_infos[p], EXT_PRIORITY), reverse=True)
        if len(scored) < 2:
            continue
        delta = _keep_score(scored[0], file_infos[scored[0]], EXT_PRIORITY) - _keep_score(scored[1], file_infos[scored[1]], EXT_PRIORITY)
        if delta >= DEDUP_KEEP_DELTA_THRESHOLD:
            for loser in scored[1:]:
                if loser not in to_delete:
                    to_delete[loser] = "Exact FP match"
                    kept_files.discard(loser)
        else:
            meta_counts = {p: file_infos[p].get("meta_count", 0) for p in scored}
            max_meta = max(meta_counts.values())
            winner = None
            for p in scored:
                if meta_counts[p] == max_meta:
                    winner = p
                    break
            for p in scored:
                if p == winner:
                    continue
                if p not in to_delete:
                    to_delete[p] = "Exact FP match"
                    kept_files.discard(p)
            log_callback(
                f"Ambiguous duplicates for fingerprint {fp} auto-resolved—kept {os.path.basename(winner)}."
            )

    if coord is not None:
        coord.add_exact_dupes([p for p, r in to_delete.items() if r == "Exact FP match"])


    # --- Near-duplicate detection -------------------------------------------------
    from near_duplicate_detector import find_near_duplicates
    log_callback("   • Detecting near-duplicates…")
    cfg = load_config()
    mixed_codec_boost = float(cfg.get("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST))
    near_dupes = find_near_duplicates(
        {p: file_infos[p] for p in kept_files},
        EXT_PRIORITY,
        fuzzy_fp_threshold,
        log_callback,
        enable_phase_c,
        coord,
        max_workers,
        mixed_codec_boost=mixed_codec_boost,
    )
    if near_dupes.review_required:
        log_callback(
            f"   ! Near-duplicate candidates require review ({len(near_dupes.review_groups)} group(s)); "
            "no automatic merges will be applied."
        )
        for group in near_dupes.review_groups:
            log_callback(
                f"     - Review {os.path.basename(group.winner)} vs {len(group.losers)} candidates "
                f"(max fp distance {group.max_distance:.3f})"
            )
    for loser, reason in near_dupes.items():
        if loser not in to_delete:
            to_delete[loser] = reason
            kept_files.discard(loser)


    trash_dir = os.path.join(root_path, "Trash")
    os.makedirs(trash_dir, exist_ok=True)
    for loser, reason in to_delete.items():
        if dry_run:
            log_callback(
                f"   ? (dry-run) Would move duplicate {loser} to Trash ({reason})"
            )
        else:
            try:
                dest = os.path.join(trash_dir, os.path.basename(loser))
                base, ext = os.path.splitext(os.path.basename(loser))
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(trash_dir, f"{base}_{counter}{ext}")
                    counter += 1
                shutil.move(loser, dest)
                log_callback(
                    f"   - Moved duplicate to Trash ({reason}): {loser}"
                )
            except Exception as e:
                log_callback(f"   ! Failed to move {loser} to Trash: {e}")

    # ─── Phase 3: Read metadata & build counters ─────────────────────────────────
    log_callback("3/6: Reading metadata and building counters…")
    songs = {}
    album_counts   = defaultdict(int)
    remix_counts   = defaultdict(int)
    cover_counts   = defaultdict(int)
    total_kept = len(kept_files)
    progress_callback(0, total_kept, "Reading metadata", "B")

    for idx, fullpath in enumerate(kept_files, start=1):
        if idx % 50 == 0 or idx == total_kept:
            log_callback(f"   • Reading metadata {idx}/{total_kept}")
        progress_callback(idx, total_kept, f"Metadata {os.path.relpath(fullpath, root_path)}", "B")

        data = get_tags(fullpath)
        raw_artist = data["artist"] or os.path.splitext(os.path.basename(fullpath))[0]
        raw_artist = collapse_repeats(raw_artist)
        title   = data["title"] or os.path.splitext(os.path.basename(fullpath))[0]
        album   = data["album"]
        year    = data["year"]
        genre   = data["genre"]
        track   = data["track"]
        part_of_set = data["part_of_set"]

        # 1) Primary & collabs (preserve original case)
        primary, collabs = extract_primary_and_collabs(raw_artist)
        p_lower = primary.lower()

        # 2) album_counts for genuine (non-remix) tracks under each (artist, album)
        if album and "remix" not in title.lower() and album.strip().lower() != title.strip().lower():
            album_counts[(p_lower, album.lower())] += 1

        # 3) remix_counts for tracks whose album tag contains "remix"
        if album and "remix" in album.lower():
            remix_counts[(p_lower, album.lower())] += 1

        # 4) EXTRACT EMBEDDED COVER DATA → compute SHA-1 and store first 10 hex digits
        cover_hash = None
        cover_payloads = []
        try:
            _tags, cover_payloads, _error, _reader = read_metadata(fullpath, include_cover=True)
        except Exception:
            cover_payloads = []
        if cover_payloads:
            sha1 = hashlib.sha1(cover_payloads[0]).hexdigest()
            cover_hash = sha1[:10]
            cover_counts[cover_hash] += 1

        folder_tags = derive_tags_from_path(fullpath, MUSIC_ROOT)

        songs[fullpath] = {
            "raw_artist": raw_artist,
            "primary":    primary,
            "collabs":    collabs,
            "title":      title,
            "album":      album,
            "year":       year,
            "genre":      genre,
            "track":      track,
            "part_of_set": part_of_set,
            "cover_hash": cover_hash,
            "folder_tags": folder_tags,
            "missing_core": not data.get("artist") or not data.get("title"),
        }

    log_callback(f"   → Collected metadata for {total_kept} files.")

    # ─── Phase 4: Determine destination for each file (with logging) ─────────────
    log_callback("4/6: Determining destination paths for each file…")
    tag_index = {}
    moves = {}
    decision_log = []
    index = 0
    total = len(songs)

    # Precompute any necessary lookups from songs data
    # (album_counts, remix_counts, etc. were built in Phase 3)

    for old_path, info in songs.items():
        index += 1
        if index % 50 == 0 or index == total:
            log_callback(f"   • Determining destination {index}/{total}")

        raw_artist = info["raw_artist"]
        title      = info["title"] or ""
        album      = info["album"] or ""
        year       = info["year"] or "Unknown"
        track      = info["track"]
        part_of_set = info.get("part_of_set")
        folders    = info["folder_tags"]
        if info.get("missing_core"):
            candidate = os.path.join(review_root, os.path.basename(old_path))
            root_c, ext_c = os.path.splitext(candidate)
            idx_dup = 1
            while os.path.exists(candidate) or candidate in moves.values():
                candidate = f"{root_c} ({idx_dup}){ext_c}"
                idx_dup += 1
            moves[old_path] = candidate
            decision_log.append(
                f"  → Missing metadata, placed under Manual Review/{os.path.basename(candidate)}"
            )
            continue

        # Normalize again for consistent global lookup
        primary_norm, _ = extract_primary_and_collabs(raw_artist)
        p_lower = primary_norm.lower()

        # Lookup how many total tracks this artist has across the entire vault
        count_now = global_counts.get(p_lower, 0)
        decision_log.append(f"Song: {raw_artist} – {title} ({album}) → global_counts[{raw_artist}] = {count_now}")

        # ─── 4.1) EARLY BAIL-OUT: “Rare” artists go straight to By Year ─────────────────
        if count_now < COMMON_ARTIST_THRESHOLD:
            base_folder = os.path.join(MUSIC_ROOT, "By Year", sanitize(year))
            decision_log.append(
                f"  Early-exit: Count {count_now} < {COMMON_ARTIST_THRESHOLD} → group under By Year/{year}"
            )

            # Build the new filename as usual (rename invalid artists or use sanitized tags)
            if "/" in raw_artist or is_repeated(raw_artist):
                new_filename = os.path.basename(old_path)
                decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping filename '{new_filename}'")
            else:
                ext = os.path.splitext(old_path)[1].lower()
                filename_artist = sanitize(raw_artist)
                track_str = f"{track:02d}" if track is not None else "00"
                title_str = sanitize(title)
                new_filename = f"{filename_artist}_{track_str}_{title_str}{ext}"
                decision_log.append(f"  Renaming to '{new_filename}' (using raw artist)")

            new_path = os.path.join(base_folder, new_filename)
            moves[old_path] = new_path
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
                p_lower = main_artist.lower()
                decision_log.append(
                    f"  → Enough remixes ({rcount} ≥ {REMIX_FOLDER_THRESHOLD}) "
                    f"AND count_now ({count_now}) ≥ {COMMON_ARTIST_THRESHOLD}; force primary = '{main_artist}'"
                )
                artist_folder = os.path.join(MUSIC_ROOT, "By Artist", sanitize(main_artist))
                base_folder = os.path.join(artist_folder, sanitize(album))
                base_folder = _apply_part_of_set(base_folder, album, part_of_set, decision_log)

                # Build filename (skip normal album logic)
                basename = os.path.basename(old_path)
                if "/" in raw_artist or is_repeated(raw_artist):
                    new_filename = basename
                    decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping '{basename}'")
                else:
                    ext = os.path.splitext(old_path)[1].lower()
                    filename_artist = sanitize(raw_artist)
                    track_str = f"{track:02d}" if track is not None else "00"
                    title_str = sanitize(title)
                    new_filename = f"{filename_artist}_{track_str}_{title_str}{ext}"
                    decision_log.append(f"  Renaming to '{new_filename}' (using raw artist)")

                new_path = os.path.join(base_folder, new_filename)
                moves[old_path] = new_path
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
        primary = best_artist
        p_lower = primary.lower()
        decision_log.append(f"  → After ranking, primary = '{primary}'")

        # Build “By Artist/<primary>” base path
        artist_folder = os.path.join(MUSIC_ROOT, "By Artist", sanitize(primary))

        # ─── 4.4) COMMON ARTIST: decide between “Album” vs. “Singles” ───────────────
        # (At this point, count_now ≥ COMMON_ARTIST_THRESHOLD is guaranteed.)
        if not album or album.strip().lower() == title.strip().lower():
            # No album tag (or album == title) → put into “<Artist> - Singles”
            base_folder = os.path.join(artist_folder, f"{sanitize(primary)} - Singles")
            decision_log.append(
                f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, no album or album=title "
                f"→ group under '{primary} - Singles'"
            )
        else:
            c2 = album_counts.get((p_lower, album.lower()), 0)
            if c2 > 3:
                # Enough tracks in this album → put into “<Artist>/<Album>”
                base_folder = os.path.join(artist_folder, sanitize(album))
                decision_log.append(
                    f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, album_count={c2} > 3 "
                    f"→ group under Album '{album}'"
                )
            else:
                # Few tracks in album → treat as a “Singles” release
                base_folder = os.path.join(artist_folder, f"{sanitize(primary)} - Singles")
                decision_log.append(
                    f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, but "
                    f"album_count={c2} ≤ 3 → group under '{primary} - Singles'"
                )

        base_folder = _apply_part_of_set(base_folder, album, part_of_set, decision_log)

        # Build filename for all “common” tracks
        basename = os.path.basename(old_path)
        if "/" in raw_artist or is_repeated(raw_artist):
            new_filename = basename
            decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping '{basename}'")
        else:
            ext = os.path.splitext(old_path)[1].lower()
            filename_artist = sanitize(raw_artist)
            track_str = f"{track:02d}" if track is not None else "00"
            title_str = sanitize(title)
            new_filename = f"{filename_artist}_{track_str}_{title_str}{ext}"
            decision_log.append(f"  Renaming to '{new_filename}' (using raw artist)")

        new_path = os.path.join(base_folder, new_filename)
        moves[old_path] = new_path
        decision_log.append(
            f"  → Final: '{os.path.relpath(new_path, MUSIC_ROOT)}'"
        )

        # ─── 4.4) Build “leftover_tags” for HTML preview (optional) ───────────────
        leftover = set(folders)
        leftover.discard(primary)
        if album:
            leftover.discard(album)
        if part_of_set:
            leftover.discard(part_of_set)
        leftover.discard(year)
        if genre:
            leftover.add(genre)

        tag_index[new_path] = {
            "leftover_tags": sorted(leftover),
            "old_paths": sorted(folders)
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
            if os.path.isdir(node) or ext.lower() not in {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}:
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


def build_dry_run_html(
    root_path,
    output_html_path,
    log_callback=None,
    enable_phase_c=False,
    flush_cache=False,
    max_workers=None,
): 
    """
    1) Call compute_moves_and_tag_index() to get (moves, tag_index, decision_log).
    2) Write a dry-run HTML to output_html_path showing the proposed tree.
    Does NOT move any files.
    """
    if log_callback is None:
        def log_callback(msg): pass

    coord = DryRunCoordinator()
    moves, tag_index, _ = compute_moves_and_tag_index(
        root_path,
        log_callback,
        None,
        dry_run=True,
        enable_phase_c=enable_phase_c,
        flush_cache=flush_cache,
        max_workers=max_workers,
        coord=coord,
    )

    log_callback("5/6: Writing dry-run HTML…")

    render_dry_run_html_from_plan(root_path, output_html_path, moves, tag_index, coord=coord)
    log_callback(f"✓ Dry-run HTML written to: {output_html_path}")
    return {"moved": 0, "html": output_html_path, "dry_run": True}


# ─── C. APPLY MOVES ─────────────────────────────────────────────────────

def apply_indexer_moves(
    root_path,
    log_callback=None,
    progress_callback=None,
    create_playlists: bool = True,
    enable_phase_c: bool = False,
    flush_cache: bool = False,
    max_workers: int | None = None,
):
    """
    1) Call compute_moves_and_tag_index() to get (moves, tag_index, decision_log).
    2) Move/rename each file in `moves`.
    3) Move any leftover non-audio or album cover images into Trash or into the correct folder.
    4) Remove empty directories in two passes: first those that held moved
       files, then a library-wide sweep for any remaining empty folders.
    Returns summary: {"moved": <count>, "errors": [<error strings>]}.
    Set ``create_playlists`` to ``False`` to skip playlist generation at the end.
    """
    if log_callback is None:
        def log_callback(msg):
            pass
    if progress_callback is None:
        def progress_callback(current, total, path, phase):
            pass

    check_cancelled()
    cfg = load_config()
    trim_silence = bool(cfg.get("trim_silence", False))

    # Ensure Not Sorted directory exists for user exclusions
    not_sorted_dir = os.path.join(root_path, "Not Sorted")
    os.makedirs(not_sorted_dir, exist_ok=True)

    # Determine where your actual music lives
    music_root = os.path.join(root_path, "Music")
    if not os.path.isdir(music_root):
        music_root = root_path


    # ─── Phase 0.5: Compute and cache fingerprints ─────────────────
    docs_dir = os.path.join(root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    db_path = os.path.join(docs_dir, ".soundvault.db")
    from fingerprint_generator import compute_fingerprints_parallel
    compute_fingerprints_parallel(
        root_path,
        db_path,
        log_callback,
        progress_callback,
        max_workers,
        trim_silence,
        phase="A",
        cancel_event=cancel_event,
    )

    moves, _, _ = compute_moves_and_tag_index(
        root_path,
        log_callback,
        progress_callback,
        dry_run=False,
        enable_phase_c=enable_phase_c,
        flush_cache=flush_cache,
        max_workers=max_workers,
        coord=None,
    )



    total_moves = len(moves)
    check_cancelled()

    MUSIC_ROOT = os.path.join(root_path, "Music") \
        if os.path.isdir(os.path.join(root_path, "Music")) \
        else root_path
    SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
    IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif"}

    summary = {"moved": 0, "errors": []}

    # Build a mapping: old_dir → set of target_dirs for audio files in that folder
    olddir_to_newdirs = defaultdict(set)
    for old_path, new_path in moves.items():
        old_dir = os.path.dirname(old_path)
        new_dir = os.path.dirname(new_path)
        olddir_to_newdirs[old_dir].add(new_dir)

    # Phase 5: Move audio files
    for idx, (old_path, new_path) in enumerate(moves.items(), start=1):
        check_cancelled()
        if progress_callback:
            progress_callback(idx, total_moves, old_path, "C")
        if idx % 50 == 0 or idx == total_moves:
            log_callback(f"   • Moving file {idx}/{total_moves}")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        try:
            if os.path.abspath(old_path) != os.path.abspath(new_path):
                shutil.move(old_path, new_path)
                summary["moved"] += 1
        except Exception as e:
            err = f"Failed to move {old_path} → {new_path}: {e}"
            summary["errors"].append(err)
            log_callback(f"   ! {err}")

    # Phase 6: Handle non-audio leftovers…
    docs_dir = os.path.join(root_path, "Docs")
    trash_dir = os.path.join(root_path, "Trash")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(trash_dir, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(MUSIC_ROOT, topdown=False):
        check_cancelled()
        # 1) Don’t recurse back into Trash/, Docs/, Playlists, or Not Sorted/
        dirnames[:] = [d for d in dirnames if d.lower() not in ("trash", "docs", "not sorted", "playlists")]
        if {"not sorted", "playlists"} & {p.lower() for p in os.path.relpath(dirpath, root_path).split(os.sep)}:
            continue

        for fname in filenames:
            check_cancelled()
            full = os.path.join(dirpath, fname)
            ext  = os.path.splitext(fname)[1].lower()

            # 2) Skip files already moved or valid audio
            if full in moves or ext in SUPPORTED_EXTS:
                continue

            target = docs_dir if ext in (".txt", ".html", ".db") else trash_dir
            if target is docs_dir:
                log_callback(f"Moving doc to Docs/: {full}")
            else:
                log_callback(f"Moving to Trash/: {full}")
            try:
                shutil.move(full, os.path.join(target, fname))
            except Exception as e:
                log_callback(f"   ! Failed to move leftover {full}: {e}")

    # Phase 6b: Remove empty directories that held moved files
    skip_names = {"trash", "docs", "not sorted", "playlists"}
    touched_dirs = sorted(olddir_to_newdirs.keys(), key=lambda p: p.count(os.sep), reverse=True)
    for base in touched_dirs:
        check_cancelled()
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=False):
            check_cancelled()
            rel = os.path.relpath(dirpath, root_path)
            first = rel.split(os.sep, 1)[0].lower() if rel != "." else ""
            if first in skip_names:
                continue
            if os.path.normpath(dirpath) in (root_path, MUSIC_ROOT):
                continue
            if not os.listdir(dirpath):
                try:
                    os.rmdir(dirpath)
                    log_callback(f"Removed empty folder: {dirpath}")
                except OSError:
                    pass

    # Phase 6c: Second pass to remove any remaining empty directories
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        check_cancelled()
        dirnames[:] = [d for d in dirnames if d.lower() not in skip_names]
        if os.path.normpath(dirpath) in (root_path, MUSIC_ROOT):
            continue
        rel = os.path.relpath(dirpath, root_path)
        first = rel.split(os.sep, 1)[0].lower() if rel != "." else ""
        if first in skip_names:
            continue
        if not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
                log_callback(f"Removed empty folder: {dirpath}")
            except OSError:
                pass

    if create_playlists:
        # ─── Phase 7: Generate Playlists with edge-case handling ────────────
        try:
            from playlist_generator import generate_playlists
            log_callback("7/7: Generating safe playlists…")

            def plog(msg):
                log_callback(msg)
                if progress_callback and msg.startswith("→ Writing playlist:"):
                    playlist_file = msg.split(":", 1)[1].strip()
                    progress_callback(total_moves, total_moves, playlist_file, "C")

            generate_playlists(
                moves,
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
    else:
        log_callback("• Playlist creation disabled")

    return summary


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
):
    """
    1) Call compute_moves_and_tag_index() to get (moves, tag_index, decision_log).
    2) Write a detailed log file `indexer_log.txt` under root_path.
    3) Write dry-run HTML via build_dry_run_html().
    4) If ``dry_run_only`` is False, call ``apply_indexer_moves()`` to move files.
       Playlist generation can be skipped by passing ``create_playlists=False``.
    Returns summary: {"moved": <count>, "html": <path>, "dry_run": <True/False>}.
    """
    if log_callback is None:
        def log_callback(msg): pass

    # Ensure Not Sorted directory exists for user exclusions
    not_sorted_dir = os.path.join(root_path, "Not Sorted")
    os.makedirs(not_sorted_dir, exist_ok=True)

    # Phase 1–4: Generate moves, tags, and collect decision log
    moves, tag_index, decision_log = compute_moves_and_tag_index(
        root_path,
        log_callback,
        progress_callback,
        dry_run=dry_run_only,
        enable_phase_c=enable_phase_c,
        flush_cache=flush_cache,
        max_workers=max_workers,
        coord=None,
    )

    # Write the detailed log file
    docs_dir = os.path.join(root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    log_path = os.path.join(docs_dir, "indexer_log.txt")
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("Indexing Decision Log\n")
        lf.write("======================\n")
        lf.write(f"Library root: {root_path}\n")
        lf.write(f"Generated on: {__import__('datetime').datetime.now()}\n\n")
        for line in decision_log:
            lf.write(line + "\n")
    log_callback(f"✓ Detailed log written to: {log_path}")

    # Build dry-run HTML
    log_callback("5/6: Writing dry-run HTML…")
    build_dry_run_html(
        root_path,
        output_html_path,
        log_callback,
        enable_phase_c=enable_phase_c,
        flush_cache=flush_cache,
        max_workers=max_workers,
    )

    if not dry_run_only:
        # Phase 5–6: Actually move audio files and cover images
        actual_summary = apply_indexer_moves(
            root_path,
            log_callback,
            progress_callback,
            create_playlists=create_playlists,
            enable_phase_c=enable_phase_c,
            flush_cache=flush_cache,
            max_workers=max_workers,
        )
        summary = {"moved": actual_summary["moved"], "html": output_html_path, "dry_run": False}
        return summary
    else:
        return {"moved": 0, "html": output_html_path, "dry_run": True}


# ─── E. DUPLICATE DETECTION HELPER ─────────────────────────────────────

def find_duplicates(root_path, log_callback=None):
    """Return list of (original_path, duplicate_path) that would be marked as
    duplicates by the indexer."""

    moves, _, _ = compute_moves_and_tag_index(
        root_path,
        log_callback,
        None,
        dry_run=True,
        enable_phase_c=False,
        flush_cache=False,
        max_workers=None,
        coord=None,
    )
    dup_indicator = os.path.join("Duplicates", "")
    return [
        (old, new)
        for old, new in moves.items()
        if dup_indicator in new
    ]


def main(argv: List[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SoundVault Music Indexer")
    parser.add_argument("root", help="Library root path")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--enable-phase-c", action="store_true", help="Enable cross-album scan")
    parser.add_argument("--flush-cache", action="store_true", help="Flush fingerprint cache")
    parser.add_argument("--max-workers", type=int, default=None, help="Fingerprint worker count")
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
