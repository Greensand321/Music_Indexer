"""Install stub modules for one test without leaking them into the rest of the run.

Many test files used to do ``sys.modules['mutagen'] = stub`` at module level.
pytest imports every test module during collection, before any test runs, so
those stubs were in place for the *whole* session and whichever production
module happened to be imported next bound to them. Results then depended on
which files ran (or were collected) first.

``isolated_modules`` replaces that pattern::

    with isolated_modules({"mutagen": stub}, fresh=["music_indexer_api"]) as mods:
        mods["music_indexer_api"].build_dry_run_html(...)

On entry it installs ``stubs`` and re-imports each module in ``fresh`` so it
binds to them. On exit it puts every touched ``sys.modules`` entry back exactly
as it was, and drops any *project* module that was first imported inside the
block (it may have bound to a stub), so the next importer gets a clean copy.
Third-party modules first imported inside the block are left alone, since
C extensions such as numpy cannot be safely imported twice.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import types
from typing import Dict, Iterable, Iterator, Mapping

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_MISSING = object()


def _is_project_module(module: object) -> bool:
    path = getattr(module, "__file__", None)
    if not path:
        return False
    path = os.path.abspath(path)
    if not path.startswith(_REPO_ROOT + os.sep):
        return False
    rel = os.path.relpath(path, _REPO_ROOT)
    # Never treat virtualenvs or installed packages inside the repo as ours.
    return not any(part in {".venv", "venv", "site-packages"} for part in rel.split(os.sep))


@contextlib.contextmanager
def isolated_modules(
    stubs: Mapping[str, types.ModuleType] | None = None,
    fresh: Iterable[str] = (),
) -> Iterator[Dict[str, types.ModuleType]]:
    stubs = dict(stubs or {})
    fresh = list(fresh)
    touched = set(stubs) | set(fresh)
    saved = {name: sys.modules.get(name, _MISSING) for name in touched}
    before = set(sys.modules)
    try:
        for name in fresh:
            sys.modules.pop(name, None)
        sys.modules.update(stubs)
        yield {name: importlib.import_module(name) for name in fresh}
    finally:
        dropped = [
            name
            for name in set(sys.modules) - before - touched
            if _is_project_module(sys.modules.get(name))
        ]
        for name in dropped:
            sys.modules.pop(name, None)
            _set_parent_attr(name, _MISSING)
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
            # ``from pkg import sub`` and ``import pkg.sub as x`` read the
            # parent package's attribute, which importing ``fresh`` rebound.
            _set_parent_attr(name, module)


def _set_parent_attr(name: str, module: object) -> None:
    parent_name, _, attr = name.rpartition(".")
    if not parent_name:
        return
    parent = sys.modules.get(parent_name)
    if parent is None:
        return
    if module is _MISSING:
        if hasattr(parent, attr) and isinstance(getattr(parent, attr), types.ModuleType):
            try:
                delattr(parent, attr)
            except AttributeError:
                pass
    else:
        setattr(parent, attr, module)


def make_mutagen_stub(file_func=None) -> Dict[str, types.ModuleType]:
    """Return ``{"mutagen": ..., "mutagen.id3": ...}`` stub modules.

    ``file_func`` becomes ``mutagen.File``; by default it returns an object
    with no tags, i.e. "a readable file with nothing in it".
    """

    class DummyAudio:
        def __init__(self):
            self.tags = None
            self.pictures = []

    def File(*_a, **_k):  # noqa: N802 - mirrors mutagen.File
        return DummyAudio()

    mutagen_stub = types.ModuleType("mutagen")
    mutagen_stub.File = file_func or File
    id3_stub = types.ModuleType("mutagen.id3")
    id3_stub.ID3NoHeaderError = Exception
    mutagen_stub.id3 = id3_stub
    return {"mutagen": mutagen_stub, "mutagen.id3": id3_stub}


def stub_if_unavailable(stubs: Mapping[str, types.ModuleType]) -> None:
    """Install ``stubs`` only for optional dependencies that are not installed.

    Unlike ``sys.modules.setdefault(name, stub)`` (which shadows the *real*
    package for the rest of the session whenever this runs before anything
    else imported it), a stub is used here only when the real top-level
    package cannot be imported at all. It then stays installed, which is
    harmless: there is no real module for it to hide.
    """
    by_top: Dict[str, Dict[str, types.ModuleType]] = {}
    for name, module in stubs.items():
        by_top.setdefault(name.split(".", 1)[0], {})[name] = module
    for top, group in by_top.items():
        if top in sys.modules:
            continue
        try:
            importlib.import_module(top)
        except ImportError:
            for name, module in group.items():
                sys.modules.setdefault(name, module)


def make_acoustid_stub() -> types.ModuleType:
    acoustid_stub = types.ModuleType("acoustid")
    acoustid_stub.fingerprint_file = lambda *_a, **_k: (None, None)
    acoustid_stub.NoBackendError = Exception
    acoustid_stub.FingerprintGenerationError = Exception
    acoustid_stub.WebServiceError = Exception
    return acoustid_stub
