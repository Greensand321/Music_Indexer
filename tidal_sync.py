import os
import shutil
from typing import List, Dict, Tuple

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


def scan_library_quality(library_root: str, outfile: str) -> int:
    """Scan ``library_root`` for non-FLAC files and write them to ``outfile``.

    The output is a plain text file with one entry per line formatted as
    ``"Artist – Title"``.
    """
    items: List[Tuple[str, str]] = []
    for dirpath, _, files in os.walk(library_root):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTS and ext != ".flac":
                tags = _read_tags(os.path.join(dirpath, fname))
                items.append((tags.get("artist") or "", tags.get("title") or ""))
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for artist, title in items:
            f.write(f"{artist} – {title}\n")
    return len(items)


def load_subpar_list(path: str) -> List[Dict[str, str]]:
    """Read a list produced by :func:`scan_library_quality`.

    Each line should contain ``"Artist – Title"``.
    """
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "–" in line:
                artist, title = [p.strip() for p in line.split("–", 1)]
            elif "-" in line:
                artist, title = [p.strip() for p in line.split("-", 1)]
            else:
                continue
            out.append({"artist": artist, "title": title})
    return out


def scan_downloads(folder: str) -> List[Dict[str, str]]:
    """Return metadata for all audio files under ``folder``."""
    items: List[Dict[str, str]] = []
    for dirpath, _, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTS:
                path = os.path.join(dirpath, fname)
                tags = _read_tags(path)
                items.append({"artist": tags.get("artist"), "title": tags.get("title"), "album": tags.get("album"), "path": path})
    return items


def _fingerprint(path: str) -> str | None:
    try:
        _, fp = acoustid.fingerprint_file(path)
        return fp
    except Exception:
        return None


def match_downloads(subpar: List[Dict[str, str]], downloads: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Match subpar tracks with potential replacements in downloads."""
    matches: List[Dict[str, object]] = []
    dl_map: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for item in downloads:
        key = (
            (item.get("artist") or "").lower(),
            (item.get("title") or "").lower(),
        )
        dl_map.setdefault(key, []).append(item)

    for sp in subpar:
        key = (
            (sp.get("artist") or "").lower(),
            (sp.get("title") or "").lower(),
        )
        cand = dl_map.get(key)
        if not cand:
            matches.append({"original": None, "download": None, "score": None, "tags": sp})
            continue
        best = cand[0]
        matches.append({"original": None, "download": best["path"], "score": None, "tags": sp})
    return matches


def replace_file(original: str, new_file: str) -> None:
    """Atomically replace ``original`` with ``new_file``."""
    tmp = original + ".tmp"
    shutil.copy2(new_file, tmp)
    os.replace(tmp, original)
