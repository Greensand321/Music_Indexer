import os
import json

CONFIG_PATH = os.path.expanduser("~/.soundvault_config.json")

# List of external metadata services supported by the application.
SUPPORTED_SERVICES = [
    "AcoustID",
    "Last.fm",
    "Spotify",
    "MusicBrainz",
    "Gracenote",
]

# Threshold for considering two tracks as near-duplicates when syncing
NEAR_DUPLICATE_THRESHOLD = 0.1

# File format quality priority used during Library Sync
FORMAT_PRIORITY = {".flac": 3, ".wav": 2, ".mp3": 1}


def load_config():
    """Load configuration from ``CONFIG_PATH``.

    Returns an empty dict if the file does not exist or can't be read.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    """Write ``cfg`` to ``CONFIG_PATH`` as JSON."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
