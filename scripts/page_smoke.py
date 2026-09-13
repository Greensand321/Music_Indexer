"""Build ONE workspace headlessly and report any error. For per-page self-checks.

Usage:  xvfb-run -a python3 scripts/page_smoke.py <workspace_key> [--shot out.png]
        xvfb-run -a python3 scripts/page_smoke.py compression --themes midnight,pearl

Exit 0 = the page built, themed and painted cleanly. Exit 1 = it did not.
Catches exceptions raised inside Qt paint/event overrides, which Qt otherwise swallows.
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


def run(key: str, themes: list[str], shot: str, width: int, height: int) -> int:
    from gui.compat import QtCore, QtWidgets

    errors: list[str] = []
    sys.excepthook = lambda et, e, tb: errors.append(
        "".join(traceback.format_exception(et, e, tb)).strip()
    )

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("AlphaDEX")

    from gui.main_window import _WORKSPACE_MAP
    from gui.themes.manager import get_manager

    if key not in _WORKSPACE_MAP:
        print(f"unknown workspace '{key}'. choices: {', '.join(_WORKSPACE_MAP)}")
        return 1

    def pump(ms: int = 350) -> None:
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(ms, loop.quit)
        loop.exec()

    get_manager().load_persisted()

    host = QtWidgets.QMainWindow()
    host.resize(width, height)
    ws = _WORKSPACE_MAP[key](library_path="")
    host.setCentralWidget(ws)
    host.show()
    pump(900)

    widest = 0
    for theme in themes:
        try:
            get_manager().apply(theme)
            pump(650)
            for tw in ws.findChildren(QtWidgets.QTabWidget):
                for i in range(tw.count()):
                    tw.setCurrentIndex(i)
                    pump(320)
            inner = getattr(ws, "_inner", None)
            if inner is not None:
                widest = max(widest, inner.minimumSizeHint().width())
            pm = host.grab()
            if pm.isNull():
                errors.append(f"{theme}: grab() returned a null pixmap")
            elif shot:
                pm.save(shot if len(themes) == 1 else shot.replace(".png", f"_{theme}.png"), "PNG")
        except Exception:
            errors.append(f"{theme}: " + traceback.format_exc().strip())

    print(f"page '{key}' built across {', '.join(themes)}")
    if widest:
        verdict = "OK" if widest <= 900 else "TOO WIDE (content area can be ~540px)"
        print(f"content minimum width: {widest}px — {verdict}")
        if widest > 900:
            errors.append(f"content minimum width {widest}px will force horizontal scrolling")

    if errors:
        print(f"\n!! {len(errors)} ERROR(S)")
        for e in errors[:12]:
            print("---")
            print(e[:1200])
        return 1
    print("OK — no exceptions, no overflow")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--shot", default="")
    ap.add_argument("--themes", default="midnight,pearl")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=860)
    a = ap.parse_args()

    buf = io.StringIO()
    rc = 1
    try:
        with redirect_stderr(buf):
            rc = run(a.key, [t for t in a.themes.split(",") if t], a.shot, a.width, a.height)
    finally:
        for line in buf.getvalue().splitlines():
            if "Error calling Python override" in line or "Traceback" in line:
                print("STDERR:", line[:260])
                rc = 1
    sys.exit(rc)
