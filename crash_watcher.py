"""CrashWatcher module recording recent events and emitting crash reports."""

from __future__ import annotations

import atexit
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque

try:
    import tkinter as tk
    from tkinter import messagebox
    from tkinter.scrolledtext import ScrolledText
except Exception:  # pragma: no cover - tkinter may be unavailable in some envs
    tk = None  # type: ignore
    messagebox = None  # type: ignore
    ScrolledText = None  # type: ignore


class CrashWatcher(threading.Thread):
    """Background thread storing recent events in a circular buffer."""

    def __init__(self, max_events: int = 100):
        super().__init__(daemon=True)
        self._events: Deque[str] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record_event(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._events.append(f"{ts} - {msg}")

    def dump_events(self) -> str:
        with self._lock:
            return "\n".join(self._events)

    def run(self) -> None:  # pragma: no cover - thread performs no active work
        while True:
            time.sleep(1)


_watcher: CrashWatcher | None = None
clean_shutdown = False
_docs_dir: Path | None = None


def start(max_events: int = 100) -> None:
    """Start the global crash watcher thread and register exit handler."""
    global _watcher
    if _watcher is None:
        _watcher = CrashWatcher(max_events=max_events)
        _watcher.start()
        atexit.register(_handle_exit)


def record_event(msg: str) -> None:
    """Record an event in the crash buffer."""
    if _watcher is not None:
        _watcher.record_event(msg)


def mark_clean_shutdown() -> None:
    """Set the global clean shutdown flag."""
    global clean_shutdown
    clean_shutdown = True


def set_library_path(root: str | Path) -> None:
    """Configure the library root for crash reports."""
    global _docs_dir
    root_path = Path(root)
    _docs_dir = root_path / "Docs"
    _docs_dir.mkdir(parents=True, exist_ok=True)


def _handle_exit() -> None:
    if clean_shutdown or _watcher is None:
        return

    report_dir = _docs_dir or Path(__file__).resolve().parent / "docs"
    report_path = report_dir / "crash_report.txt"
    data = _watcher.dump_events()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(data, encoding="utf-8")

    if tk and ScrolledText:
        try:
            root = tk.Tk()
            root.title("Crash Report")
            text = ScrolledText(root, width=80, height=24)
            text.pack(fill="both", expand=True)
            text.insert("1.0", data)
            text.configure(state="disabled")
            tk.Button(root, text="Close", command=root.destroy).pack(pady=5)
            root.mainloop()
        except Exception:
            pass
    else:  # pragma: no cover - in headless environments
        print(f"Crash report written to {report_path}")
