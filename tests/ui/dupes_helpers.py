"""Shared fixtures/helpers for the duplicate-detection GUI tests."""
from __future__ import annotations

import re
import threading
from pathlib import Path

from _helpers.dupes_chromaprint import fake_fingerprint

_TRACK_NO = re.compile(r"^\d+\s*-\s*")


def recording_key(path: str) -> str:
    """Same title (ignoring a leading track number) → same recording."""
    return _TRACK_NO.sub("", Path(path).stem).strip().lower()


class FakeFpcalc:
    """Stands in for ``chromaprint_utils.fingerprint_fpcalc`` (same signature:
    returns a compressed fingerprint *string*). Lossy copies get a few flipped
    bits, like a real transcode. ``gate`` lets a test hold the scan mid-run."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.gate: threading.Event | None = None
        self.started = threading.Event()

    def __call__(self, path: str, **_kwargs) -> str:
        self.calls.append(path)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(10)
        lossy = Path(path).suffix.lower() not in {".flac", ".wav"}
        return fake_fingerprint(recording_key(path), flips=12 if lossy else 0)


def install_fake_fpcalc(monkeypatch) -> FakeFpcalc:
    import chromaprint_utils
    import gui.workspaces.duplicates as dupes

    fake = FakeFpcalc()
    monkeypatch.setattr(chromaprint_utils, "fingerprint_fpcalc", fake)
    monkeypatch.setattr(dupes, "_missing_fingerprint_tools", lambda: [])
    return fake


def record_browser(monkeypatch) -> list[str]:
    opened: list[str] = []
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)
    return opened


def run_scan(ws, qtbot, timeout: int = 60_000) -> None:
    ws._scan_btn.click()
    qtbot.waitUntil(lambda: not ws._scan_running, timeout=timeout)


def run_execute(ws, qtbot, timeout: int = 60_000) -> None:
    ws._execute_btn.click()
    qtbot.waitUntil(lambda: not ws._exec_running, timeout=timeout)
