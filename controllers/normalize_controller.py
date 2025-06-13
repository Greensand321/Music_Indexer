import os
import json
from typing import Dict

PROMPT_TEMPLATE = """
I will provide a list of raw music genres (one per line). Your task is to group each variant under a controlled vocabulary by mapping each raw genre to an array of canonical genre names in JSON format. For example:

{
  "rock & roll": ["Rock"],
  "rock n roll": ["Rock"],
  "rock": ["Rock"],
  "indie rock": ["Indie Rock", "Rock"],
  "hip hop": ["Hip-Hop"],
  "90s": ["invalid"]
}

Follow these guidelines:

• Use arrays as values, so each raw genre key maps to one or more canonical genre names.  
• If a genre has clearly defined subgenres, include both the subgenre(s) and the parent genre(s) in the array—keep all granularity intact.  
• Split any merged or concatenated terms into separate genres (e.g. "hiphoprap" → ["Hip-Hop", "Rap"]).  
• Map any term that isn’t a valid music genre to ["invalid"].  
• If you encounter ambiguous terms, ask clarifying questions before proceeding.
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
