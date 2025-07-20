import os
import json

CONFIG_PATH = os.path.expanduser("~/.soundvault_config.json")

# Tuning parameters for Library Sync
NEAR_DUPLICATE_THRESHOLD = 0.05
# Higher value means higher quality. Can be overridden in config file.
FORMAT_PRIORITY = {".flac": 3, ".wav": 2, ".mp3": 1}

# List of external metadata services supported by the application.
SUPPORTED_SERVICES = [
    "AcoustID",
    "Last.fm",
    "Spotify",
    "MusicBrainz",
    "Gracenote",
]


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


def get_library_sync_config() -> dict:
    """Return Library Sync tuning options from config with defaults."""
    cfg = load_config()
    return {
        "near_duplicate_threshold": cfg.get("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD),
        "format_priority": cfg.get("format_priority", FORMAT_PRIORITY),
    }
