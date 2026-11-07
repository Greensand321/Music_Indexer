"""One tag-writing job at a time, across every metadata page.

The Tag Fixer's Apply, the Genre Normalizer's Fill and Normalize applies all
rewrite tags in place, often on the same files. Run concurrently they race:
the last save wins and the before/after logs no longer describe what
happened. Each apply takes this guard before it starts and releases it when
its worker finishes; a second apply is refused with a message naming the job
already writing.

Qt-free; meant to be used from the GUI thread (acquire on click, release in
the worker's finished slot), so a plain lock-protected owner name is enough.
"""
from __future__ import annotations

import threading
from typing import Optional

_lock = threading.Lock()
_owner: Optional[str] = None


def acquire(owner: str) -> bool:
    """Claim the right to write tags. False if another job holds it."""
    global _owner
    with _lock:
        if _owner is not None:
            return False
        _owner = owner
        return True


def release(owner: str) -> None:
    """Give the claim back (no-op if ``owner`` doesn't hold it)."""
    global _owner
    with _lock:
        if _owner == owner:
            _owner = None


def current_owner() -> Optional[str]:
    """Name of the job writing tags right now, or None."""
    with _lock:
        return _owner


def busy_message(wanted: str) -> str:
    owner = current_owner() or "another task"
    return (
        f"{owner} is writing tags right now. {wanted} can start when it has "
        "finished — running both at once could make them overwrite each other."
    )
