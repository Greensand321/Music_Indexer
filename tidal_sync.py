import os
import re
import shutil
from typing import List, Dict, Tuple, Optional

from mutagen import File as MutagenFile
import acoustid

from music_indexer_api import fingerprint_distance


AUDIO_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}


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


def scan_library_quality(library_root: str, outfile: str) -> int:
    """Scan ``library_root`` for non-FLAC files and write them to ``outfile``.

    The resulting file contains one line per track in the form::

        Artist \u2013 Title \u2013 Album \u2013 FullPath
    """
    items: List[Tuple[str, str, str, str]] = []
    for dirpath, _, files in os.walk(library_root):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTS and ext != ".flac":
                path = os.path.join(dirpath, fname)
                tags = _read_tags(path)
                items.append((tags.get("artist") or "", tags.get("title") or "", tags.get("album") or "", path))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for artist, title, album, path in items:
            f.write(
                f"{artist}{SUBPAR_DELIM}{title}{SUBPAR_DELIM}{album}{SUBPAR_DELIM}{path}\n"
            )
    return len(items)


def load_subpar_list(path: str) -> List[Dict[str, str]]:
    """Read a txt list produced by :func:`scan_library_quality`.

    Each line is formatted as ``Artist – Title – Album – FullPath``.
    """
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(SUBPAR_DELIM)
            if len(parts) != 4:
                continue
            artist, title, album, fpath = parts
            out.append({"artist": artist, "title": title, "album": album, "path": fpath})
    return out


def scan_downloads(folder: str) -> List[Dict[str, str]]:
    """Return metadata and cached fingerprints for all audio files under ``folder``."""
    items: List[Dict[str, str]] = []
    for dirpath, _, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTS:
                path = os.path.join(dirpath, fname)
                tags = _read_tags(path)
                fp = _fingerprint(path)
                items.append({
                    "artist": tags.get("artist"),
                    "title": tags.get("title"),
                    "album": tags.get("album"),
                    "path": path,
                    "fingerprint": fp,
                })
    return items


def _fingerprint(path: str) -> str | None:
    try:
        _, fp = acoustid.fingerprint_file(path)
        return fp
    except Exception:
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
    threshold: float,
) -> Tuple[Optional[Dict[str, str]], float, bool]:
    """Return best candidate by fingerprint distance.

    Returns (candidate, distance, ambiguous). Candidate is ``None`` if no
    distance below ``threshold``.
    """
    best: Optional[Dict[str, str]] = None
    best_dist = 1.0
    distances: List[float] = []
    for c in cands:
        dist = fingerprint_distance(orig_fp, c.get("fingerprint"))
        distances.append(dist)
        if dist < best_dist:
            best_dist = dist
            best = c
    if best is None or best_dist >= threshold:
        return None, best_dist, False
    ambiguous = sum(1 for d in distances if d <= best_dist + 0.05) > 1
    return best, best_dist, ambiguous


def match_downloads(
    subpar: List[Dict[str, str]],
    downloads: List[Dict[str, str]],
    threshold: float = 0.3,
) -> List[Dict[str, object]]:
    """Match subpar tracks with potential replacements in downloads."""
    matches: List[Dict[str, object]] = []
    dl_map: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    fuzzy_map: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    fp_map: Dict[str, List[Dict[str, str]]] = {}

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

        for k in (_fuzzy_key(item, False), _fuzzy_key(item, True)):
            fuzzy_map.setdefault(k, []).append(item)

    for sp in subpar:
        key_exact = (
            (sp.get("artist") or "").lower(),
            (sp.get("title") or "").lower(),
            (sp.get("album") or "").lower(),
        )
        orig_fp = _fingerprint(sp["path"])
        note = None
        method = "None"
        best = None
        best_dist = 1.0

        # exact fingerprint deduplication
        if orig_fp and len(fp_map.get(orig_fp, [])) == 1:
            best = fp_map[orig_fp][0]
            best_dist = 0.0
            method = "Fingerprint"

        if best is None:
            # fingerprint-first matching
            cand, best_dist, ambiguous = _find_best_fp_match(orig_fp, downloads, threshold)
            if cand is not None:
                best = cand
                method = "Fingerprint"
                if ambiguous:
                    note = "Ambiguous – manual review"

        if best is None:
            cand = dl_map.get(key_exact)
            if cand:
                best, best_dist, ambiguous = _find_best_fp_match(orig_fp, cand, threshold)
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
                cand, best_dist, ambiguous = _find_best_fp_match(orig_fp, unique_cands, threshold)
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
        matches.append(
            {
                "original": sp["path"],
                "download": None if best is None else best["path"],
                "score": score,
                "method": method,
                "tags": sp,
                "note": note,
            }
        )
    return matches


def replace_file(original: str, new_file: str) -> None:
    """Atomically replace ``original`` with ``new_file``."""
    tmp = original + ".tmp"
    shutil.copy2(new_file, tmp)
    os.replace(tmp, original)
