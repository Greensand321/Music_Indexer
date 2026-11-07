"""Helpers for the shell / startup tests (tests/ui/test_ui_shell_*.py)."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from ui_harness import process_events


def write_config(**values) -> None:
    import config

    cfg = config.load_config()
    cfg.update(values)
    config.save_config(cfg)


def make_album_tree(root, artists: int = 2, albums: int = 2, tracks: int = 2) -> str:
    """A tiny untagged library (folders + empty .mp3 files)."""
    root = str(root)
    for a in range(artists):
        for b in range(albums):
            d = os.path.join(root, f"Artist{a}", f"Album{b}")
            os.makedirs(d, exist_ok=True)
            for t in range(tracks):
                with open(os.path.join(d, f"{t:02d}.mp3"), "wb") as fh:
                    fh.write(b"\0" * 1000)
    return root


def wait_until(predicate, timeout_ms: int = 5000) -> bool:
    import time

    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if predicate():
            return True
        process_events(10)
    return bool(predicate())


def cta_button(landing) -> QtWidgets.QPushButton:
    return landing._cta.main_button


def finish_startup(startup, threads, timeout_ms: int = 10_000) -> None:
    """Let background work finish, then close and delete everything the
    startup created, so harness teardown sees no running thread or dialog."""
    assert threads.wait_all(timeout_ms) == []
    for w in (startup.window, startup.landing):
        if w is None:
            continue
        try:
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    assert threads.wait_all(timeout_ms) == []
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete.value)
    process_events(10)
