from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time

_log = logging.getLogger(__name__)

# Why the last load_config() fell back to defaults (None = it did not).
_last_load_error: str | None = None
# Where save_config() last rescued an unreadable settings file (None = never).
_last_backup: str | None = None

CONFIG_PATH = os.path.expanduser("~/.soundvault_config.json")

# List of external metadata services supported by the application.
SUPPORTED_SERVICES = [
    "AcoustID",
    "MusicBrainz",
    "Last.fm",
    "Spotify",
    "Gracenote",
]

# Services listed above that are not yet implemented (stub functions only).
# The Settings UI shows these as disabled so users aren't misled.
UNAVAILABLE_SERVICES = {"Spotify", "Gracenote"}

# Threshold for considering two tracks as near-duplicates when syncing
NEAR_DUPLICATE_THRESHOLD = 0.1
EXACT_DUPLICATE_THRESHOLD = 0.02
MIXED_CODEC_THRESHOLD_BOOST = 0.03
ARTWORK_VASTLY_DIFFERENT_THRESHOLD = 24
PREVIEW_ARTWORK_MAX_DIM = 192
PREVIEW_ARTWORK_QUALITY = 20

# File format quality priority used during Library Sync
FORMAT_PRIORITY = {".flac": 3, ".wav": 2, ".mp3": 1}

# Default fingerprint matching thresholds by file extension (Library Sync).
# The ``default`` key is used when a specific extension is not provided.
#
# These mirror Library Sync's engine (``library_sync_indexer_engine`` config,
# which is what ``library_sync`` itself uses) so the pages that only import
# this module show the same values; tests/test_syncmatch_config.py keeps the
# two equal. The scale is the bit error rate of the *decoded* Chromaprint
# fingerprints: the same recording in any codec <= 0.008, a start shifted by
# 0.3-0.6 s 0.07-0.14, different recordings >= 0.43. The old 0.3 / 0.35 were
# tuned for the previous metric (Hamming distance of the compressed bytes).
DEFAULT_FP_THRESHOLDS = {
    "default": 0.2,
    ".flac": 0.2,
    ".mp3": 0.2,
    ".m4a": 0.2,
    ".aac": 0.2,
}
# Library Sync defaults of the old metric. A saved ``sync_threshold`` equal
# to one of them is migrated to the new default (see the Library Sync page).
LEGACY_SYNC_THRESHOLDS = (0.3, 0.35)
# Value of ``sync_threshold_scale`` once ``sync_threshold`` is on the scale above.
SYNC_THRESHOLD_SCALE = "chromaprint-ber"
# Distance from which different recordings start to match (they measure
# >= 0.43). Equal to ``library_sync.DIFFERENT_RECORDING_FLOOR``; a threshold
# at or above it treats unrelated tracks as duplicates.
SYNC_DIFFERENT_RECORDING_FLOOR = 0.4

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
FP_SUBPROCESS_TIMEOUT_SEC = 180


def _apply_defaults(cfg: dict) -> dict:
    """Fill in every key the app expects (existing values always win)."""
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
    cfg.setdefault("fingerprint_subprocess_timeout_sec", FP_SUBPROCESS_TIMEOUT_SEC)
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


def _read_config_file(path: str) -> dict:
    """Parse ``path``; raise if it is missing (FileNotFoundError) or unusable."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("the top-level JSON value is not an object")
    return cfg


def last_load_error() -> str | None:
    """Why the most recent ``load_config()`` fell back to defaults, or None.

    ``None`` also when the file simply does not exist yet (first run).
    """
    return _last_load_error


def load_config():
    """Load configuration from ``CONFIG_PATH``.

    Returns the defaults when the file does not exist or can't be read. An
    unreadable (truncated / corrupt) file is *not* lost: it is logged here,
    reported by :func:`last_load_error`, and backed up by
    :func:`save_config` before anything overwrites it.
    """
    global _last_load_error
    try:
        cfg = _read_config_file(CONFIG_PATH)
    except FileNotFoundError:
        _last_load_error = None
        return _apply_defaults({})
    except Exception as exc:  # corrupt JSON, wrong type, permissions, ...
        message = (
            f"Settings file {CONFIG_PATH} could not be read ({exc}); using defaults. "
            "It will be backed up before it is overwritten."
        )
        if message != _last_load_error:
            _log.warning(message)
        _last_load_error = message
        return _apply_defaults({})
    _last_load_error = None
    return _apply_defaults(cfg)


def _backup_unreadable_config(path: str) -> str | None:
    """Copy an existing-but-unreadable config aside; return the backup path."""
    global _last_backup
    if not os.path.exists(path):
        return None
    try:
        _read_config_file(path)
        return None  # readable: nothing to rescue
    except Exception:
        pass
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{path}.corrupt-{stamp}"
    n = 1
    while os.path.exists(backup):
        n += 1
        backup = f"{path}.corrupt-{stamp}-{n}"
    shutil.copy2(path, backup)
    _log.warning("Unreadable settings file backed up to %s before saving.", backup)
    _last_backup = backup
    return backup


def last_backup_path() -> str | None:
    """The backup ``save_config`` made of an unreadable settings file this
    session, or None."""
    return _last_backup


def save_config(cfg: dict) -> None:
    """Write ``cfg`` to ``CONFIG_PATH`` as JSON, atomically.

    The JSON is written to a temporary file in the same folder and moved
    over the old file with ``os.replace``, so a crash or full disk can never
    leave a half-written (truncated) settings file. If the current file
    exists but cannot be parsed, it is first copied to
    ``<name>.corrupt-<timestamp>`` so its contents (API keys, ...) survive.
    """
    path = CONFIG_PATH
    directory = os.path.dirname(os.path.abspath(path))
    _backup_unreadable_config(path)
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(path):
            try:
                shutil.copymode(path, tmp)
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
