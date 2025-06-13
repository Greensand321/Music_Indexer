import os
import json
from typing import Dict


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
