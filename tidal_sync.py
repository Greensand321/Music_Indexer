import os
import re
import shutil
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Callable

from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, ID3NoHeaderError
import acoustid
import logging

from music_indexer_api import fingerprint_distance, get_tags, sanitize
from fingerprint_cache import get_fingerprint


AUDIO_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
FP_PREFIX_LEN = 16

# Debug configuration
debug: bool = False
_logger = logging.getLogger(__name__)
_logger.propagate = False
_logger.addHandler(logging.NullHandler())
_file_handler: Optional[logging.Handler] = None


def set_debug(enabled: bool, log_root: str | None = None) -> None:
    """Enable or disable verbose debug logging."""
    global debug, _file_handler
    debug = enabled
    if not enabled:
        if _file_handler:
            _logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
        return

    _logger.setLevel(logging.DEBUG)
    if log_root:
        os.makedirs(log_root, exist_ok=True)
        log_path = os.path.join(log_root, "tidal_sync_debug.log")
        if _file_handler:
            _logger.removeHandler(_file_handler)
            _file_handler.close()
        _file_handler = logging.FileHandler(log_path, encoding="utf-8")
        _file_handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(_file_handler)


def _dlog(msg: str, log_callback: Callable[[str], None] | None = None) -> None:
    if not debug:
        return
    if log_callback:
        log_callback(msg)
    _logger.debug(msg)


def _read_tags(path: str) -> Dict[str, str | None]:
    try:
        audio = MutagenFile(path, easy=True)
        if not audio or not audio.tags:
            return {"artist": None, "title": None, "album": None}
        tags = audio.tags
        artist = tags.get("artist", [None])[0]
        title = tags.get("title", [None])[0]
        album = tags.get("album", [None])[0]
        return {"artist": artist, "title": title, "album": album}
    except Exception:
        return {"artist": None, "title": None, "album": None}


SUBPAR_DELIM = " \u2013 "


def _read_artist_title(path: str) -> Tuple[str, str]:
    """Return best-effort artist and title from ``path`` using multiple sources."""
    artist = None
    title = None

    try:
        id3 = ID3(path)
        for frame in ("TPE1", "TPE2"):
            if not artist:
                try:
                    if id3.get(frame):
                        artist = id3[frame].text[0]
                except (KeyError, UnicodeDecodeError):
                    pass
        if not title:
            try:
                if id3.get("TIT2"):
                    title = id3["TIT2"].text[0]
            except (KeyError, UnicodeDecodeError):
                pass
        for key in list(id3.keys()):
            if key.startswith("TXXX"):
                lower = key.lower()
                if "artist" in lower and not artist:
                    try:
                        artist = id3[key].text[0]
                    except (KeyError, UnicodeDecodeError):
                        pass
                if "title" in lower and not title:
                    try:
                        title = id3[key].text[0]
                    except (KeyError, UnicodeDecodeError):
                        pass
    except ID3NoHeaderError:
        pass
    except Exception:
        pass

    try:
        mp3 = MP3(path)
        tags = mp3.tags or {}
        if not artist:
            try:
                val = tags.get("artist")
                artist = val[0] if isinstance(val, list) else val
            except (KeyError, UnicodeDecodeError, AttributeError):
                pass
        if not title:
            try:
                val = tags.get("title")
                title = val[0] if isinstance(val, list) else val
            except (KeyError, UnicodeDecodeError, AttributeError):
                pass
    except Exception:
        pass

    if not artist or not title:
        base = os.path.splitext(os.path.basename(path))[0]
        parts = base.split(" \u2013 ", 1)
        if not artist:
            artist = parts[0]
        if not title:
            title = parts[1] if len(parts) > 1 else parts[0]

    return artist or "Unknown", title or "Unknown"


def _rename_with_sanitize(path: str, library_root: str) -> str:
    """Rename ``path`` to a sanitized ``Artist_XX_Title.ext`` pattern.

    If either artist or title metadata is missing, move the file into the
    ``Manual Review`` folder at the library root, preserving the original
    filename. Duplicate names in that folder get ``(1)``, ``(2)``, … appended.

    Returns the new path (or original if rename failed)."""
    tags = get_tags(path)
    artist = tags.get("artist")
    title = tags.get("title")

    # If key metadata is missing, move to Manual Review without renaming
    if not artist or not title:
        review_root = os.path.join(library_root, "Manual Review")
        os.makedirs(review_root, exist_ok=True)
        basename = os.path.basename(path)
        candidate = os.path.join(review_root, basename)
        root, ext_only = os.path.splitext(candidate)
        idx = 1
        while os.path.exists(candidate):
            candidate = f"{root} ({idx}){ext_only}"
            idx += 1
        try:
            os.rename(path, candidate)
            return candidate
        except Exception:
            return path

    artist = sanitize(artist)
    title = sanitize(title)
    track = tags.get("track")
    track_str = f"{track:02d}" if track is not None else "00"
    ext = os.path.splitext(path)[1].lower()
    base = f"{artist}_{track_str}_{title}{ext}"
    dirpath = os.path.dirname(path)
    candidate = os.path.join(dirpath, base)

    root, ext_only = os.path.splitext(candidate)
    idx = 1
    while os.path.exists(candidate) and os.path.abspath(candidate) != os.path.abspath(path):
        candidate = f"{root}_{idx}{ext_only}"
        idx += 1

    if candidate == path:
        return path

    try:
        os.rename(path, candidate)
        return candidate
    except Exception:
        return path


def scan_library_quality(library_root: str, outfile: str) -> int:
    """Scan ``library_root`` for non-FLAC files and write two lists.

    Any flagged file is renamed immediately using the ``Artist_XX_Title.ext``
    pattern. ``outfile`` is used as the base name for two outputs::

        <base>_full.txt    Artist \u2013 Title \u2013 Album \u2013 Path
        <base>_simple.txt  Artist \u2013 Title
    """
    items: List[Tuple[str, str, str, str]] = []
    for dirpath, _, files in os.walk(library_root):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTS and ext != ".flac":
                path = os.path.join(dirpath, fname)
                new_path = _rename_with_sanitize(path, library_root)
                tags = _read_tags(new_path)
                artist = tags.get("artist")
                title = tags.get("title")
                album = tags.get("album") or ""
                if not artist or not title:
                    a2, t2 = _read_artist_title(new_path)
                    artist = artist or a2
                    title = title or t2
                items.append((artist or "Unknown", title or "Unknown", album, new_path))

    base = os.path.splitext(outfile)[0]
    full_path = base + "_full.txt"
    simple_path = base + "_simple.txt"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as ffull, open(simple_path, "w", encoding="utf-8") as fsimple:
        for artist, title, album, path in items:
            ffull.write(f"{artist}{SUBPAR_DELIM}{title}{SUBPAR_DELIM}{album}{SUBPAR_DELIM}{path}\n")
            fsimple.write(f"{artist}{SUBPAR_DELIM}{title}\n")
    return len(items)


def load_subpar_list(path: str, db_path: str | None = None) -> List[Dict[str, str]]:
    """Read a txt list produced by :func:`scan_library_quality`.

    Supports both legacy ``Artist – Title – Album – FullPath`` lines and
    simplified ``Artist – Title`` entries. If a full path is present, a
    fingerprint is computed using :func:`fingerprint_cache.get_fingerprint` and
    cached under ``db_path``. Relative paths are prefixed with the configured
    ``library_root``.
    """
    from config import load_config
    import logging

    cfg = load_config()
    root = cfg.get("library_root", "")
    if not os.path.exists(path) and path.endswith("subpar_full.txt"):
        simple = path.replace("subpar_full.txt", "subpar_simple.txt")
        if os.path.exists(simple):
            logging.warning("subpar_full.txt not found; using simple list")
            path = simple

    out: List[Dict[str, str]] = []
    if db_path is None:
        db_path = os.path.join(os.path.dirname(path), "fp.db")

    def _compute_fp(p: str) -> tuple[int | None, str | None]:
        try:
            return acoustid.fingerprint_file(p)
        except Exception:
            return None, None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.split(SUBPAR_DELIM)
            if len(parts) == 4:
                artist, title, album, fpath = parts
                fpath = fpath or ""
                if fpath and not os.path.isabs(fpath):
                    fpath = os.path.join(root, fpath)
                fp = get_fingerprint(fpath, db_path, _compute_fp) if fpath else None
                out.append(
                    {
                        "artist": artist,
                        "title": title,
                        "album": album,
                        "path": fpath,
                        "fingerprint": fp,
                        "fp_prefix": fp[:FP_PREFIX_LEN] if fp else None,
                    }
                )
                continue

            parts = re.split(r"\s+[–-]\s+", text, maxsplit=1)
            if len(parts) == 2:
                artist, title = parts
                out.append(
                    {
                        "artist": artist,
                        "title": title,
                        "album": None,
                        "path": None,
                        "fingerprint": None,
                        "fp_prefix": None,
                    }
                )
    return out


def scan_downloads(
    folder: str,
    log_callback: Callable[[str], None] | None = None,
    max_workers: int = 1,
) -> List[Dict[str, str]]:
    """Return metadata and cached fingerprints for all audio files under ``folder``.

    ``max_workers`` controls the number of threads used for fingerprinting.
    A value of ``1`` preserves the previous serial behaviour.
    """
    items: List[Dict[str, str]] = []

    if max_workers <= 1:
        for dirpath, _, files in os.walk(folder):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in AUDIO_EXTS:
                    path = os.path.join(dirpath, fname)
                    items.append(_scan_one(path, log_callback))
        return items

    futures: List[Tuple[str, 'concurrent.futures.Future[Dict[str, str]]']] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exc:
        for dirpath, _, files in os.walk(folder):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in AUDIO_EXTS:
                    path = os.path.join(dirpath, fname)
                    fut = exc.submit(_scan_one, path, log_callback)
                    futures.append((path, fut))

        for _path, fut in futures:
            items.append(fut.result())

    return items


def _scan_one(path: str, log_callback: Callable[[str], None] | None = None) -> Dict[str, str]:
    """Read tags and fingerprint a single file."""
    tags = _read_tags(path)
    fp = _fingerprint(path, log_callback)
    return {
        "artist": tags.get("artist"),
        "title": tags.get("title"),
        "album": tags.get("album"),
        "path": path,
        "fingerprint": fp,
        "fp_prefix": fp[:FP_PREFIX_LEN] if fp else None,
    }


def _fingerprint(path: str, log_callback: Callable[[str], None] | None = None) -> str | None:
    from config import load_config
    from audio_norm import normalize_for_fp

    cfg = load_config()
    offset = cfg.get("fingerprint_offset_ms", 0)
    duration = cfg.get("fingerprint_duration_ms", 120_000)
    allow = cfg.get("allow_mismatched_edits", True)

    try:
        buf = normalize_for_fp(path, offset, duration, allow, log_callback)
        _, fp = acoustid.fingerprint_file(fileobj=buf)
        _dlog(
            f"DEBUG: Fingerprinting file: {path}; fp prefix={fp[:FP_PREFIX_LEN]!r}",
            log_callback,
        )
        return fp
    except Exception as exc:
        msg = f"Failed to fingerprint {path}: {exc}"
        _dlog(f"ERROR: {msg}", log_callback)
        logging.error(msg)
        return None


def _normalize_title(text: str) -> str:
    """Return lowercase alphanumeric title without track numbers."""
    if not text:
        return ""
    text = re.sub(r'^\d+\s*-\s*', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', '', text)
    return text.lower()


def _fuzzy_key(item: Dict[str, str], use_filename: bool = False) -> Tuple[str, str, str]:
    """Return a fuzzy key based on artist, title or filename, and album."""
    artist = (item.get("artist") or "").lower()
    title = item.get("title") or ""
    if use_filename:
        title = os.path.splitext(os.path.basename(item.get("path") or ""))[0]
    title = _normalize_title(title)
    album = (item.get("album") or "").lower()
    return artist, title, album


def _find_best_fp_match(
    orig_fp: Optional[str],
    cands: List[Dict[str, str]],
    thresholds: Dict[str, float],
    fp_prefix_map: Dict[str, List[Dict[str, str]]] | None,
    orig_path: str = "",
    log_callback: Callable[[str], None] | None = None,
    return_candidates: bool = False,
) -> Tuple[Optional[Dict[str, str]], float, bool, Optional[List[Dict[str, str]]]]:
    """Return best candidate by fingerprint distance.

    Returns (candidate, distance, ambiguous[, candidates]). Candidate is ``None`` if
    no distance is below the threshold for the candidate's extension. If ``return_candidates`` is ``True`` and
    ``ambiguous`` is ``True``, a list of close candidate dicts is returned as the
    fourth element; otherwise ``None`` is returned in that position.
    """
    if fp_prefix_map and orig_fp:
        prefix = orig_fp[:FP_PREFIX_LEN]
        allowed = {id(c) for c in fp_prefix_map.get(prefix, [])}
        if allowed:
            cands = [c for c in cands if id(c) in allowed]

    best: Optional[Dict[str, str]] = None
    best_dist = 1.0
    distances: List[float] = []
    for c in cands:
        dist = fingerprint_distance(orig_fp, c.get("fingerprint"))
        distances.append(dist)
        ext = os.path.splitext(c.get("path") or "")[1].lower()
        thr = thresholds.get(ext, thresholds.get("default", 0.3))
        _dlog(
            f"DEBUG: Distance between {orig_path} and {c.get('path')}: {dist:.4f}",
            log_callback,
        )
        _dlog(
            f"DEBUG: Threshold check ({thr:.4f}): {'passed' if dist < thr else 'failed'}",
            log_callback,
        )
        if dist < best_dist:
            best_dist = dist
            best = c
    if best is None:
        return (None, best_dist, False, None) if return_candidates else (None, best_dist, False)
    best_ext = os.path.splitext(best.get("path") or "")[1].lower()
    best_thr = thresholds.get(best_ext, thresholds.get("default", 0.3))
    if best_dist >= best_thr:
        return (None, best_dist, False, None) if return_candidates else (None, best_dist, False)

    ambiguous = sum(1 for d in distances if d <= best_dist + 0.05) > 1
    cand_list: Optional[List[Dict[str, str]]] = None
    if return_candidates and ambiguous:
        close = [
            c
            for c, d in sorted(zip(cands, distances), key=lambda t: t[1])
            if d <= best_dist + 0.05
        ]
        cand_list = close

    return (best, best_dist, ambiguous, cand_list) if return_candidates else (
        best,
        best_dist,
        ambiguous,
    )


def match_downloads(
    subpar: List[Dict[str, str]],
    downloads: List[Dict[str, str]],
    thresholds: Dict[str, float] | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    result_callback: Callable[[Dict[str, object]], None] | None = None,
) -> List[Dict[str, object]]:
    """Match subpar tracks with potential replacements in downloads."""
    if thresholds is None:
        thresholds = {"default": 0.3}

    matches: List[Dict[str, object]] = []
    dl_map: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    fuzzy_map: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    fp_map: Dict[str, List[Dict[str, str]]] = {}
    fp_prefix_map: Dict[str, List[Dict[str, str]]] = {}

    for item in downloads:
        key = (
            (item.get("artist") or "").lower(),
            (item.get("title") or "").lower(),
            (item.get("album") or "").lower(),
        )
        dl_map.setdefault(key, []).append(item)

        fp = item.get("fingerprint")
        if fp:
            fp_map.setdefault(fp, []).append(item)
            fp_prefix = item.get("fp_prefix")
            if fp_prefix:
                fp_prefix_map.setdefault(fp_prefix, []).append(item)

        for k in (_fuzzy_key(item, False), _fuzzy_key(item, True)):
            fuzzy_map.setdefault(k, []).append(item)

    if log_callback is None:
        log_callback = lambda msg: None
    if progress_callback is None:
        progress_callback = lambda idx: None

    for idx, sp in enumerate(subpar, start=1):
        key_exact = (
            (sp.get("artist") or "").lower(),
            (sp.get("title") or "").lower(),
            (sp.get("album") or "").lower(),
        )
        orig_fp = sp.get("fingerprint")
        if orig_fp is None and sp.get("path"):
            orig_fp = _fingerprint(sp["path"], log_callback)
        note = None
        method = "None"
        best = None
        best_dist = 1.0

        # exact fingerprint deduplication
        if orig_fp and len(fp_map.get(orig_fp, [])) == 1:
            best = fp_map[orig_fp][0]
            best_dist = 0.0
            method = "Fingerprint"

        cand_list = None

        if best is None:
            # fingerprint-first matching
            cand, best_dist, ambiguous, cand_list = _find_best_fp_match(
                orig_fp,
                downloads,
                thresholds,
                fp_prefix_map,
                sp.get("path", ""),
                log_callback,
                True,
            )
            if cand is not None:
                best = cand
                method = "Fingerprint"
                if ambiguous:
                    note = "Ambiguous – manual review"

        if best is None:
            _dlog(
                f"DEBUG: Looking up tag key: {key_exact!r}; candidates found: {len(dl_map.get(key_exact, []))}",
                log_callback,
            )
            cand = dl_map.get(key_exact)
            if cand:
                for candidate in cand:
                    _dlog(
                        f"DEBUG: Candidate path={candidate['path']}; artist={candidate.get('artist')!r}, title={candidate.get('title')!r}, album={candidate.get('album')!r}",
                        log_callback,
                    )
                best, best_dist, ambiguous, cand_list = _find_best_fp_match(
                    orig_fp,
                    cand,
                    thresholds,
                    fp_prefix_map,
                    sp.get("path", ""),
                    log_callback,
                    True,
                )
                if best is not None:
                    method = "Tag"
                    if ambiguous:
                        note = "Ambiguous – manual review"

        if best is None:
            cands: List[Dict[str, str]] = []
            for k in (_fuzzy_key(sp, False), _fuzzy_key(sp, True)):
                cands.extend(fuzzy_map.get(k, []))
            seen = set()
            unique_cands = []
            for c in cands:
                if id(c) not in seen:
                    unique_cands.append(c)
                    seen.add(id(c))

            if unique_cands:
                cand, best_dist, ambiguous, cand_list = _find_best_fp_match(
                    orig_fp,
                    unique_cands,
                    thresholds,
                    fp_prefix_map,
                    sp.get("path", ""),
                    log_callback,
                    True,
                )
                if cand is not None:
                    best = cand
                    method = "Fuzzy"
                    if ambiguous:
                        note = "Ambiguous – manual review"
                elif orig_fp is None and len(unique_cands) == 1:
                    best = unique_cands[0]
                    best_dist = 1.0
                    method = "Fuzzy"

        if orig_fp is None and note is None:
            note = "No fingerprint"

        score = None if best is None else 1 - best_dist
        cand_paths = [c.get("path") for c in cand_list] if cand_list else []
        _dlog(
            f"DEBUG: Best match: {best['path'] if best else None} with score={score if score is not None else float('nan'):.4f}",
            log_callback,
        )
        match = {
            "original": sp["path"],
            "download": None if best is None else best["path"],
            "score": score,
            "method": method,
            "tags": sp,
            "note": note,
            "candidates": cand_paths,
        }
        matches.append(match)
        if result_callback:
            result_callback(match)
        if progress_callback:
            progress_callback(idx)
    return matches


def replace_file(original: str, new_file: str) -> None:
    """Atomically replace ``original`` with ``new_file`` and backup the original."""
    backup_dir = os.path.join(os.path.dirname(original), "__backup__")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(original))
    tmp = original + ".tmp"
    shutil.copy2(new_file, tmp)
    if os.path.exists(original):
        shutil.move(original, backup_path)
    os.replace(tmp, original)


def restore_backups(root: str, backup_dirname: str = "__backup__") -> List[str]:
    """Move files from ``backup_dirname`` folders back to their original location."""
    restored: List[str] = []
    for dirpath, dirs, _files in os.walk(root):
        if backup_dirname not in dirs:
            continue
        bdir = os.path.join(dirpath, backup_dirname)
        for fname in os.listdir(bdir):
            src = os.path.join(bdir, fname)
            dest = os.path.join(dirpath, fname)
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(src, dest)
            restored.append(dest)
        try:
            os.rmdir(bdir)
        except OSError:
            pass
    return restored
