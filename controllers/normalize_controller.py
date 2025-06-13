import os
import json
import re
from typing import Dict

PROMPT_TEMPLATE = """
I will provide a list of raw music genres (one per line). Your task is to group and map each raw genre into a canonical key in JSON format, for example:

{
  "rock & roll": ["Rock"],
  "future bass": ["Future Bass","Electronic"],
  "indie rock": ["Indie Rock","Rock"],
  "90s": ["invalid"]
}

Follow these guidelines:

• For each raw genre key, return an array of one or more canonical genre names as the value.  
• If a genre has clearly defined subgenres, list both the subgenre and its parent(s) (e.g. "future bass": ["Future Bass","Electronic"]).  
• Split and list merged terms separately (e.g. "hiphoprap": ["Hip-Hop","Rap"]).  
• Map non-genres to ["invalid"].  
• Ask clarifying questions if any terms are ambiguous.
"""


def load_mapping(folder: str) -> tuple[Dict[str, str], str]:
    """Return genre mapping dict and path."""
    path = os.path.join(folder, ".genre_mapping.json")
    mapping: Dict[str, str] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception:
            mapping = {}
    return mapping, path


def save_mapping(folder: str, mapping: Dict[str, str]) -> str:
    """Save mapping JSON and return path."""
    path = os.path.join(folder, ".genre_mapping.json")
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return path


def normalize_genres(genres: list[str], mapping: Dict[str, str]) -> list[str]:
    """Return list of genres normalized via mapping."""
    return [mapping.get(g, g) for g in genres]


def get_raw_genres(records):
    """
    Given a list of FileRecord objects, return a sorted, deduplicated list of
    all raw genre strings (from rec.old_genres and rec.new_genres).
    """
    raw_set = set()
    for rec in records:
        # include whatever fields you want—typically the pre-normalized tags:
        raw_set.update(rec.old_genres)
        # if you want newly suggested genres too:
        raw_set.update(rec.new_genres)
    return sorted(raw_set)


def scan_raw_genres(folder: str, progress_callback):
    """
    Walk the folder, read each file’s embedded 'genre' tags only,
    split on [],;,/ to separate combined entries, and return a
    sorted, deduplicated list of raw genres.
    """
    # discover_files comes from your tagfix controller
    from controllers.tagfix_controller import discover_files
    from mutagen import File as MutagenFile

    files = discover_files(folder)
    total = len(files)
    raw_set = set()
    progress_callback(0, total)
    for idx, path in enumerate(files, start=1):
        progress_callback(idx, total)
        audio = MutagenFile(path, easy=True)
        genres = audio.get("genre", []) or []
        for entry in genres:
            # split on semicolon, comma or slash
            parts = re.split(r'[;,/]', entry)
            for part in parts:
                part = part.strip()
                if part:
                    raw_set.add(part)
    return sorted(raw_set)
