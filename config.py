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
EXACT_DUPLICATE_THRESHOLD = 0.02
MIXED_CODEC_THRESHOLD_BOOST = 0.03
ARTWORK_VASTLY_DIFFERENT_THRESHOLD = 24
PREVIEW_ARTWORK_MAX_DIM = 192
PREVIEW_ARTWORK_QUALITY = 20

# File format quality priority used during Library Sync
FORMAT_PRIORITY = {".flac": 3, ".wav": 2, ".mp3": 1}

# Default fingerprint matching thresholds by file extension. The ``default`` key
# is used when a specific extension is not provided.
DEFAULT_FP_THRESHOLDS = {
    "default": 0.3,
    ".flac": 0.3,
    ".mp3": 0.35,
    ".m4a": 0.35,
    ".aac": 0.35,
}

# Default threshold for simple duplicate detection
DEFAULT_DUP_THRESHOLD = 0.03
# Default prefix length used for duplicate grouping
DEFAULT_DUP_PREFIX_LEN = 16

# Default settings for fingerprint normalization
FP_OFFSET_MS = 0
FP_DURATION_MS = 120_000
FP_TRIM_SILENCE = True
FP_SILENCE_THRESHOLD_DB = -50.0
FP_SILENCE_MIN_LEN_MS = 500
FP_TRIM_LEAD_MAX_MS = 1000
FP_TRIM_TRAIL_MAX_MS = 1000
FP_TRIM_PADDING_MS = 100
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
        cfg.setdefault("trim_silence", FP_TRIM_SILENCE)
        cfg.setdefault("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD)
        cfg.setdefault("exact_duplicate_threshold", EXACT_DUPLICATE_THRESHOLD)
        cfg.setdefault("fingerprint_offset_ms", FP_OFFSET_MS)
        cfg.setdefault("fingerprint_duration_ms", FP_DURATION_MS)
        cfg.setdefault("fingerprint_silence_threshold_db", FP_SILENCE_THRESHOLD_DB)
        cfg.setdefault("fingerprint_silence_min_len_ms", FP_SILENCE_MIN_LEN_MS)
        cfg.setdefault("fingerprint_trim_lead_max_ms", FP_TRIM_LEAD_MAX_MS)
        cfg.setdefault("fingerprint_trim_trail_max_ms", FP_TRIM_TRAIL_MAX_MS)
        cfg.setdefault("fingerprint_trim_padding_ms", FP_TRIM_PADDING_MS)
        cfg.setdefault("allow_mismatched_edits", ALLOW_MISMATCHED_EDITS)
        cfg.setdefault("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST)
        cfg.setdefault("artwork_vastly_different_threshold", ARTWORK_VASTLY_DIFFERENT_THRESHOLD)
        cfg.setdefault("duplicate_finder_show_artwork_variants", True)
        cfg.setdefault("duplicate_finder_debug_trace", False)
        cfg.setdefault("preview_artwork_max_dim", PREVIEW_ARTWORK_MAX_DIM)
        cfg.setdefault("preview_artwork_quality", PREVIEW_ARTWORK_QUALITY)
        cfg.setdefault("library_root", "")
        cfg.setdefault("use_library_sync_review", False)
        cfg.setdefault("library_sync_review", {})
        return cfg
    except Exception:
        return {
            "trim_silence": FP_TRIM_SILENCE,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "exact_duplicate_threshold": EXACT_DUPLICATE_THRESHOLD,
            "fingerprint_offset_ms": FP_OFFSET_MS,
            "fingerprint_duration_ms": FP_DURATION_MS,
            "fingerprint_silence_threshold_db": FP_SILENCE_THRESHOLD_DB,
            "fingerprint_silence_min_len_ms": FP_SILENCE_MIN_LEN_MS,
            "fingerprint_trim_lead_max_ms": FP_TRIM_LEAD_MAX_MS,
            "fingerprint_trim_trail_max_ms": FP_TRIM_TRAIL_MAX_MS,
            "fingerprint_trim_padding_ms": FP_TRIM_PADDING_MS,
            "allow_mismatched_edits": ALLOW_MISMATCHED_EDITS,
            "library_root": "",
            "duplicate_threshold": DEFAULT_DUP_THRESHOLD,
            "duplicate_prefix_len": DEFAULT_DUP_PREFIX_LEN,
            "use_library_sync_review": False,
            "library_sync_review": {},
            "mixed_codec_threshold_boost": MIXED_CODEC_THRESHOLD_BOOST,
            "artwork_vastly_different_threshold": ARTWORK_VASTLY_DIFFERENT_THRESHOLD,
            "duplicate_finder_show_artwork_variants": True,
            "duplicate_finder_debug_trace": False,
            "preview_artwork_max_dim": PREVIEW_ARTWORK_MAX_DIM,
            "preview_artwork_quality": PREVIEW_ARTWORK_QUALITY,
        }


def save_config(cfg: dict) -> None:
    """Write ``cfg`` to ``CONFIG_PATH`` as JSON."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
