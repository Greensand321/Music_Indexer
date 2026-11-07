import os

#: Windows extended-length ("long path") prefix.
LONG_PATH_PREFIX = "\\\\?\\"


def ensure_long_path(path: str) -> str:
    """Return ``path`` in a form the Windows file APIs accept beyond 260 chars.

    Use it only at the point where the path is handed to the operating system
    (``open``, ``os.stat``, ``shutil.move``, mutagen, ...). Never display or
    compare the result: keep the caller's path for that, so the ``\\\\?\\``
    prefix doesn't leak into logs, playlists or the UI. (One deliberate
    exception: the fingerprint cache keys its rows by this long form, as it
    always has, so existing Windows caches stay valid; it hands the caller's
    own path back.) A no-op off Windows.
    """
    if os.name == "nt":
        path = os.path.abspath(path)
        if not path.startswith(LONG_PATH_PREFIX):
            path = LONG_PATH_PREFIX + os.path.normpath(path)
    return path


def strip_ext_prefix(path: str) -> str:
    """Return ``path`` without a Windows extended-length prefix."""
    if os.name == "nt" and path.startswith(LONG_PATH_PREFIX):
        return path[len(LONG_PATH_PREFIX):]
    return path


# Backwards compatibility -------------------------------------------------
strip_long_path_prefix = strip_ext_prefix


def safe_relpath(path: str, start: str = os.curdir) -> str:
    """``os.path.relpath`` that falls back to the absolute path.

    On Windows there is no relative path between two drives (``C:`` and
    ``D:``), and ``os.path.relpath`` raises ``ValueError``; the absolute path
    is the only correct answer then.
    """
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return os.path.abspath(path)


def match_separators(path: str, like: str) -> str:
    """Write ``path`` with the separator style of ``like``.

    On Windows both ``\\`` and ``/`` separate folders, and a playlist entry
    written with ``/`` (portable to phones and other systems) must stay that
    way when it is rewritten. Off Windows ``/`` is the only separator, so
    ``path`` is returned unchanged.
    """
    if os.altsep and os.altsep in like:
        return path.replace(os.sep, os.altsep)
    return path


def playlist_entry(target: str, playlist_dir: str, like: str | None = None) -> str:
    """The playlist entry pointing at ``target`` from a playlist in ``playlist_dir``.

    ``like`` is the entry being replaced: an absolute entry stays absolute, a
    relative one stays relative (absolute only when no relative path exists,
    i.e. another drive on Windows), and the separator style of ``like`` is
    kept. Without ``like`` the entry is relative with native separators.
    """
    if like is not None and os.path.isabs(like):
        entry = os.path.abspath(target)
    else:
        entry = safe_relpath(target, playlist_dir)
    return match_separators(entry, like) if like is not None else entry
