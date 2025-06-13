import os
import json
from typing import Dict

PROMPT_TEMPLATE = """
I will provide a list of raw music genres (one per line). Your task is to group and map each raw genre into a canonical key in JSON format, for example: {"rock & roll": "Rock", "hip hop": "Hip-Hop", …}.

Follow these guidelines:

If a genre has clearly defined subgenres, include both the subgenre and its parent genre(s). For instance, "future bass" should map to both "Future Bass" and "Electronic"; similarly, "indie rock" maps to both "Indie Rock" and "Rock". Keep the granularity and original details intact.

Ensure genres like "hiphoprap" are split and listed separately as "Hip-Hop" and "Rap".

If a provided term does not represent a valid music genre, list it under "invalid".

If you encounter ambiguous genre terms, please ask clarifying questions before proceeding.
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
