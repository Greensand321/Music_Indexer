import os


def ensure_long_path(path: str) -> str:
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith("\\\\?\\"):
            path = "\\\\?\\" + os.path.normpath(path)
    return path


def strip_long_path_prefix(path: str) -> str:
    """Return path without a Windows extended-length prefix."""
    if os.name == "nt" and path.startswith(r"\\?\\"):
        return path[4:]
    return path
