"""Headless UI smoke test for the AlphaDEX Qt app.

Drives the real app under an offscreen/Xvfb display: builds every workspace,
visits every sub-tab, switches themes, and grabs each screen. Any Python
exception raised inside a Qt paint/event override is recorded — Qt swallows
those, which is how a broken tab label went unnoticed for so long.

Usage:  xvfb-run -a python3 scripts/ui_smoke.py [--out DIR] [--themes a,b]
Exit code 0 = clean, 1 = at least one error.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
from contextlib import redirect_stderr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

ERRORS: list[str] = []


def _hook(exc_type, exc, tb):
    ERRORS.append("".join(traceback.format_exception(exc_type, exc, tb)).strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--themes", default="midnight,pearl")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    a = ap.parse_args()

    from gui.compat import QtCore, QtWidgets

    sys.excepthook = _hook
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("AlphaDEX")

    from gui.main_window import AlphaDEXWindow, _WORKSPACE_MAP
    from gui.themes.manager import get_manager

    def pump(ms: int = 320) -> None:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(ms, loop.quit)
        loop.exec()

    if a.out:
        os.makedirs(a.out, exist_ok=True)

    win = AlphaDEXWindow()
    win.resize(a.width, a.height)
    win.show()
    pump(1200)

    shots = 0
    for theme in [t for t in a.themes.split(",") if t]:
        get_manager().apply(theme)
        pump(700)
        for key in _WORKSPACE_MAP:
            try:
                win._on_nav_changed(key)
                pump(520)
                tabs = win._workspaces[key].findChildren(QtWidgets.QTabWidget)
                indices = [(tw, i) for tw in tabs for i in range(tw.count())] or [(None, 0)]
                for tw, i in indices:
                    if tw is not None:
                        tw.setCurrentIndex(i)
                        pump(420)
                    pm = win.grab()
                    if pm.isNull() or pm.width() == 0:
                        ERRORS.append(f"{theme}/{key}[{i}]: grab() returned an empty pixmap")
                        continue
                    shots += 1
                    if a.out:
                        suffix = f"__{i}" if tw is not None else ""
                        pm.save(os.path.join(a.out, f"{theme}_{key}{suffix}.png"), "PNG")
            except Exception:
                ERRORS.append(f"{theme}/{key}: " + traceback.format_exc().strip())

    print(f"captured {shots} screens across {a.themes}")
    if ERRORS:
        print(f"\n!! {len(ERRORS)} ERROR(S)")
        for e in ERRORS[:40]:
            print("---")
            print(e[:1500])
        return 1
    print("OK — no exceptions raised during build, paint or theme switch")
    return 0


if __name__ == "__main__":
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            rc = main()
    finally:
        noise = buf.getvalue()
        # Qt prints swallowed override exceptions to stderr; surface them.
        for line in noise.splitlines():
            if "Error calling Python override" in line or "Traceback" in line:
                print("STDERR:", line[:300])
                rc = 1
    sys.exit(rc)
