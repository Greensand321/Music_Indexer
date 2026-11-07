"""The library's reserved top-level folders — one definition for every workflow.

AlphaDEX keeps its own working folders at the top of a library::

    Not Sorted/     user-managed exclusion zone (the indexer never touches it)
    Playlists/      .m3u playlists
    Manual Review/  tracks missing required metadata
    Docs/           HTML reports and logs
    Trash/          non-audio leftovers moved out by the indexer
    Quarantine/     duplicate losers awaiting review

Scans that treat files as "the library" (indexing, duplicate finding, library
sync matching, tag/genre fixing, statistics, mirrors, exports) must skip them;
otherwise quarantined duplicate losers come back as library tracks, Docs/
reports get rewritten, and so on. Before this module each workflow kept its
own partial list (or none).

Only *top-level* folders are reserved: ``Artist/Album/Docs`` is ordinary
content. Matching is case-insensitive (Windows/macOS file systems are).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator

NOT_SORTED = "Not Sorted"
PLAYLISTS = "Playlists"
MANUAL_REVIEW = "Manual Review"
DOCS = "Docs"
TRASH = "Trash"
QUARANTINE = "Quarantine"

RESERVED_FOLDERS: tuple[str, ...] = (NOT_SORTED, PLAYLISTS, MANUAL_REVIEW, DOCS, TRASH, QUARANTINE)

# Folders whose contents are *not the user's music*: always skip when building
# a picture of "what is in the library". (Manual Review / Not Sorted hold real
# tracks the user may still want to browse or play.)
NON_LIBRARY_FOLDERS: tuple[str, ...] = (PLAYLISTS, DOCS, TRASH, QUARANTINE)


def _norm(names: Iterable[str]) -> frozenset[str]:
    return frozenset(n.casefold() for n in names)


def is_reserved_name(name: str, reserved: Iterable[str] = RESERVED_FOLDERS) -> bool:
    """True if a top-level folder called ``name`` is reserved."""
    return name.casefold() in _norm(reserved)


def reserved_component(
    path: str | os.PathLike, library_root: str | os.PathLike,
    reserved: Iterable[str] = RESERVED_FOLDERS,
) -> str | None:
    """The reserved top-level folder ``path`` lives in, or None.

    Paths outside ``library_root`` are never reserved.
    """
    try:
        rel = Path(os.path.abspath(path)).relative_to(os.path.abspath(library_root))
    except ValueError:
        return None
    if not rel.parts:
        return None
    first = rel.parts[0]
    return first if first.casefold() in _norm(reserved) else None


def is_reserved_path(
    path: str | os.PathLike, library_root: str | os.PathLike,
    reserved: Iterable[str] = RESERVED_FOLDERS,
) -> bool:
    return reserved_component(path, library_root, reserved) is not None


def prune_reserved_dirs(
    dirpath: str | os.PathLike, dirnames: list[str], library_root: str | os.PathLike,
    reserved: Iterable[str] = RESERVED_FOLDERS,
) -> None:
    """For use inside ``os.walk``: drop reserved folders from ``dirnames`` in place
    when ``dirpath`` is the library root, so the walk never descends into them."""
    if os.path.normcase(os.path.abspath(dirpath)) != os.path.normcase(os.path.abspath(library_root)):
        return
    names = _norm(reserved)
    dirnames[:] = [d for d in dirnames if d.casefold() not in names]


def walk_library(
    library_root: str | os.PathLike, reserved: Iterable[str] = RESERVED_FOLDERS,
) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk`` over the library that skips the reserved top-level folders."""
    reserved = tuple(reserved)
    for dirpath, dirnames, filenames in os.walk(library_root):
        prune_reserved_dirs(dirpath, dirnames, library_root, reserved)
        yield dirpath, dirnames, filenames


def iter_library_files(
    library_root: str | os.PathLike,
    extensions: Iterable[str] | None = None,
    reserved: Iterable[str] = RESERVED_FOLDERS,
) -> Iterator[Path]:
    """Every file under ``library_root`` outside the reserved folders,
    optionally filtered by (case-insensitive) extension such as ``".flac"``."""
    exts = {e.lower() for e in extensions} if extensions is not None else None
    for dirpath, _dirnames, filenames in walk_library(library_root, reserved):
        for name in filenames:
            if exts is None or os.path.splitext(name)[1].lower() in exts:
                yield Path(dirpath) / name
