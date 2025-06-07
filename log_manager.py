import json
import os


def get_log_path(library_root):
    """Return absolute path to the tagger log for ``library_root``."""
    return os.path.join(library_root, "docs", "tagger_log.json")


def load_log(library_root):
    """Load the tagger log for ``library_root``. Return empty dict if missing."""
    path = get_log_path(library_root)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def prune_missing_entries(log_dict, library_root):
    """Remove log entries for files that no longer exist."""
    removed = []
    for rel in list(log_dict.keys()):
        abs_path = os.path.join(library_root, rel)
        if not os.path.exists(abs_path):
            removed.append(rel)
            del log_dict[rel]
    return removed


def save_log(log_dict, library_root):
    """Write ``log_dict`` back to disk for ``library_root``."""
    path = get_log_path(library_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(log_dict, fh, indent=2)


def reset_log(library_root):
    """Delete the tagger log for ``library_root``."""
    path = get_log_path(library_root)
    if os.path.exists(path):
        os.remove(path)
