"""Render every AlphaDEX page and section in chosen themes to PNG + an HTML contact sheet.

The visual baseline for the GUI redesign: run it before and after a change
and compare the sheets side by side. Runs headless (Qt "offscreen"), in a
throw-away HOME so your real config is never touched, against a synthetic
tagged library by default.

    python scripts/gui_snapshots.py                       # midnight + pearl
    python scripts/gui_snapshots.py --themes all --out /tmp/shots
    python scripts/gui_snapshots.py --library "D:/Music" --themes midnight
    python scripts/gui_snapshots.py --pages indexer,duplicates/compare --size 1600x1000

Pages come from gui/navigation.py in sidebar order; a page with section tabs
gets one shot per section. ``--pages`` takes page keys (every section of the
page) or ``page/section`` keys (``--workspaces`` is an old alias).

Output: <out>/<theme>/<NN>_<page>[_<section>].png and <out>/index.html.

Notes: fonts render with the bundled Inter/JetBrains Mono, so offscreen shots
are close to (not pixel-identical with) a real desktop. Album art and other
library-dependent content appears only after background scans finish; the
script waits for all background jobs before each capture.
"""
from __future__ import annotations

import argparse
import html
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(REPO / "snapshots"),
                   help="output directory (default: ./snapshots, git-ignored)")
    p.add_argument("--themes", default="midnight,pearl",
                   help="comma-separated theme keys, or 'all'")
    p.add_argument("--pages", "--workspaces", dest="pages", default="all",
                   help="comma-separated page or page/section keys, or 'all'")
    p.add_argument("--size", default="1300x860", help="window size WxH")
    p.add_argument("--library", default="", help="library folder (default: a synthetic library)")
    p.add_argument("--scale", type=float, default=1.0, help="device pixel ratio for captures")
    p.add_argument("--settle-ms", type=int, default=250, help="event-loop time before each capture")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if args.scale != 1.0:
        os.environ["QT_SCALE_FACTOR"] = str(args.scale)

    # Isolate HOME before anything imports config (CONFIG_PATH is computed at import).
    home = Path(tempfile.mkdtemp(prefix="alphadex_snap_home_"))
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tests"))
    sys.path.insert(0, str(REPO / "tests" / "ui"))

    import pytest  # only for MonkeyPatch, used to install the dialog guard
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from gui.fonts import load_fonts
    load_fonts()

    from ui_harness import DialogGuard, process_events
    mp = pytest.MonkeyPatch()
    DialogGuard().install(mp)  # a stray modal must never hang the run

    library = args.library
    if not library:
        from library_factory import build_standard_library
        library = str(build_standard_library(home / "Music").root)

    from gui.main_window import AlphaDEXWindow
    from gui.navigation import PAGE_BY_KEY, PAGES
    from gui.themes.manager import get_manager
    from gui.themes.tokens import THEMES
    from gui.workers import job_registry

    themes = list(THEMES) if args.themes == "all" else [t.strip() for t in args.themes.split(",")]
    unknown = [t for t in themes if t not in THEMES]
    if unknown:
        print(f"Unknown theme(s): {unknown}. Available: {', '.join(THEMES)}", file=sys.stderr)
        return 2
    try:
        keys = _shot_keys(args.pages, PAGES, PAGE_BY_KEY)
    except KeyError as exc:
        print(f"Unknown page/section: {exc.args[0]}. Pages: "
              f"{', '.join(p.key for p in PAGES)}", file=sys.stderr)
        return 2
    w, h = (int(v) for v in args.size.lower().split("x"))

    window = AlphaDEXWindow()
    window.resize(w, h)
    window.show()
    window.set_library(library)
    job_registry().wait_all(30_000)
    process_events(args.settle_ms)

    out = Path(args.out)
    shots: dict[str, list[tuple[str, Path]]] = {}
    started = time.perf_counter()
    for theme in themes:
        get_manager().apply(theme)
        _settle(window, args.settle_ms, process_events)
        tdir = out / theme
        tdir.mkdir(parents=True, exist_ok=True)
        for i, key in enumerate(keys, 1):
            window._on_nav_changed(key)
            job_registry().wait_all(30_000)
            _settle(window, args.settle_ms, process_events)
            path = tdir / f"{i:02d}_{key.replace('/', '_')}.png"
            window.grab().save(str(path))
            shots.setdefault(theme, []).append((key, path))
            print(f"  {theme:<10} {key:<28} → {path.relative_to(out)}")

    _write_index(out, shots, library)
    print(f"Captured {sum(len(v) for v in shots.values())} screenshots in "
          f"{time.perf_counter() - started:.1f}s → {out / 'index.html'}")

    registry = job_registry()
    registry.cancel_all()
    registry.wait_all(10_000)
    window._closing = True
    window.close()
    mp.undo()
    return 0


def _shot_keys(spec: str, pages, page_by_key) -> list[str]:
    """Navigation keys to capture: one per page, or ``page/section`` for each
    section of a page with section tabs."""
    def expand(page_key: str) -> list[str]:
        page = page_by_key[page_key]
        if not page.has_sections:
            return [page.key]
        return [f"{page.key}/{s.key}" for s in page.sections]

    if spec == "all":
        return [k for page in pages for k in expand(page.key)]
    keys: list[str] = []
    for raw in (k.strip() for k in spec.split(",") if k.strip()):
        if "/" in raw:
            page_key, _, section_key = raw.partition("/")
            if page_key not in page_by_key or page_by_key[page_key].section(section_key) is None:
                raise KeyError(raw)
            keys.append(raw)
        elif raw in page_by_key:
            keys.extend(expand(raw))
        else:
            raise KeyError(raw)
    return keys


def _settle(window, settle_ms: int, process_events, timeout_ms: int = 5000) -> None:
    """Spin the event loop, then wait until no finite animation under the
    window is still running, so a capture never shows motion mid-flight
    (e.g. the sidebar pill's overshoot)."""
    import time as _time
    from PySide6 import QtCore

    process_events(settle_ms)
    deadline = _time.monotonic() + timeout_ms / 1000
    while _time.monotonic() < deadline:
        running = [
            a for a in window.findChildren(QtCore.QAbstractAnimation)
            if a.state() == QtCore.QAbstractAnimation.State.Running and a.loopCount() != -1
        ]
        if not running:
            return
        process_events(20)


def _write_index(out: Path, shots: dict, library: str) -> None:
    rows = []
    for theme, items in shots.items():
        cells = "".join(
            f'<figure><a href="{html.escape(p.relative_to(out).as_posix())}">'
            f'<img loading="lazy" src="{html.escape(p.relative_to(out).as_posix())}"></a>'
            f"<figcaption>{html.escape(key)}</figcaption></figure>"
            for key, p in items
        )
        rows.append(f"<h2>{html.escape(theme)}</h2><div class=grid>{cells}</div>")
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>AlphaDEX snapshots</title>"
        "<style>body{font:14px system-ui;margin:24px;background:#111;color:#ddd}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}"
        "img{width:100%;border:1px solid #333;border-radius:6px}figure{margin:0}"
        "figcaption{padding:4px 0;color:#aaa}</style>"
        f"<h1>AlphaDEX page snapshots</h1><p>Library: {html.escape(library)}</p>"
        + "".join(rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
