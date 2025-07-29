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

# Default fingerprint matching thresholds by file extension. The ``default`` key
# is used when a specific extension is not provided.
DEFAULT_FP_THRESHOLDS = {
    "default": 0.3,
    ".flac": 0.3,
    ".mp3": 0.3,
    ".aac": 0.3,
}

# Default threshold for simple duplicate detection
DEFAULT_DUP_THRESHOLD = 0.03
# Default prefix length used for duplicate grouping
DEFAULT_DUP_PREFIX_LEN = 16

# Default settings for fingerprint normalization
FP_OFFSET_MS = 0
FP_DURATION_MS = 120_000
ALLOW_MISMATCHED_EDITS = True


def load_config():
    """Load configuration from ``CONFIG_PATH``.

    Returns an empty dict if the file does not exist or can't be read.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "musicbrainz_useragent" not in cfg:
            cfg["musicbrainz_useragent"] = {
                "app": "",
                "version": "",
                "contact": "",
            }
        if "format_fp_thresholds" not in cfg:
            cfg["format_fp_thresholds"] = DEFAULT_FP_THRESHOLDS.copy()
        cfg.setdefault("duplicate_threshold", DEFAULT_DUP_THRESHOLD)
        cfg.setdefault("duplicate_prefix_len", DEFAULT_DUP_PREFIX_LEN)
        cfg.setdefault("fingerprint_offset_ms", FP_OFFSET_MS)
        cfg.setdefault("fingerprint_duration_ms", FP_DURATION_MS)
        cfg.setdefault("allow_mismatched_edits", ALLOW_MISMATCHED_EDITS)
        cfg.setdefault("library_root", "")
        return cfg
    except Exception:
        return {
            "fingerprint_offset_ms": FP_OFFSET_MS,
            "fingerprint_duration_ms": FP_DURATION_MS,
            "allow_mismatched_edits": ALLOW_MISMATCHED_EDITS,
            "library_root": "",
            "duplicate_threshold": DEFAULT_DUP_THRESHOLD,
            "duplicate_prefix_len": DEFAULT_DUP_PREFIX_LEN,
        }


def save_config(cfg: dict) -> None:
    """Write ``cfg`` to ``CONFIG_PATH`` as JSON."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
