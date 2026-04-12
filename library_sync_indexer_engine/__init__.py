from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent / "indexer_engine"
_ENGINE_DIR_STR = str(_ENGINE_DIR)
if _ENGINE_DIR_STR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR_STR)


def _load(name: str):
    return importlib.import_module(name)


music_indexer_api = _load("music_indexer_api")
dry_run_coordinator = _load("dry_run_coordinator")
config = _load("config")
indexer_control = _load("indexer_control")
fingerprint_cache = _load("fingerprint_cache")
fingerprint_generator = _load("fingerprint_generator")
near_duplicate_detector = _load("near_duplicate_detector")
playlist_generator = _load("playlist_generator")
crash_logger = _load("crash_logger")

__all__ = [
    "music_indexer_api",
    "dry_run_coordinator",
    "config",
    "indexer_control",
    "fingerprint_cache",
    "fingerprint_generator",
    "near_duplicate_detector",
    "playlist_generator",
    "crash_logger",
]
