"""Home, Activity, Help and the Activity Log strip in the Liquid Glass look
(redesign R2): the "Spin something" albums, Library health tiles, Recent runs
rows, the Duplicates summary, Activity's job rows and progress, Help's links,
the log strip next to the Now Playing capsule, and every page fitting the
minimum window without horizontal scrolling."""
from __future__ import annotations

import datetime
import json
import os
import threading
import time

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from shell_helpers import wait_until
from ui_harness import process_events

from gui.compat import Signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cover(path: str, hue: int) -> None:
    img = QtGui.QImage(48, 48, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor.fromHsv(hue % 360, 200, 90 + (hue * 7) % 150))
    painter = QtGui.QPainter(img)
    painter.fillRect(8 + hue % 20, 10, 18, 24, QtGui.QColor.fromHsv((hue * 3) % 360, 255, 255))
    painter.end()
    assert img.save(path, "PNG")


def _track(path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(b"\0" * 64)


def art_library(root, albums: int = 4, *, extra=()):
    """Artist/Album folders, each with a distinct cover.png and two tracks.
    ``extra``: (relative folder, with tracks) pairs added the same way."""
    root = str(root)
    folders = [(f"Artist{i}/Album{i}", True) for i in range(albums)] + list(extra)
    for n, (rel, with_tracks) in enumerate(folders):
        d = os.path.join(root, *rel.split("/"))
        os.makedirs(d, exist_ok=True)
        _cover(os.path.join(d, "cover.png"), 23 * n + 5)
        if with_tracks:
            _track(os.path.join(d, "02 - B.mp3"))
            _track(os.path.join(d, "01 - A.mp3"))
    return root


def _docs(root, rel: str, body: str = "x", age_s: float = 0) -> str:
    path = os.path.join(root, "Docs", *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
    return path


def _home_ready(ws, threads, timeout=10_000) -> None:
    assert wait_until(lambda: ws.health() is not None and not ws.is_finding_albums(), timeout)
    assert threads.wait_all(timeout) == []
    process_events(10)


# ── Spin something: the album scan and tracks ────────────────────────────────

def test_album_tracks_reads_the_folder_in_track_order(tmp_path):
    from gui.workspaces.home import album_tracks

    lib = tmp_path / "Lib"
    album = lib / "Artist" / "Granite"
    album.mkdir(parents=True)
    for name in ("10 - J.mp3", "2 - B.mp3", "1 - A.flac", "notes.txt"):
        (album / name).write_bytes(b"x")
    tags = {"1 - A.flac": {"album": "Granite", "track": 3}}
    read = lambda p: tags.get(os.path.basename(p), {})          # noqa: E731
    got = [os.path.basename(p) for p in album_tracks(str(album), "Granite", str(lib), read_tags=read)]
    # Untagged tracks of an album folder belong to it; numbers sort naturally;
    # tagged track numbers win over names.
    assert got == ["1 - A.flac", "2 - B.mp3", "10 - J.mp3"]

    # A flat folder holding two albums: each gets only its own tracks.
    flat = lib / "Singles"
    flat.mkdir()
    for name in ("a.mp3", "b.mp3", "c.mp3"):
        (flat / name).write_bytes(b"x")
    tags = {"a.mp3": {"album": "One"}, "b.mp3": {"album": "Two"}, "c.mp3": {"album": "One", "track": 1}}
    one = [os.path.basename(p) for p in album_tracks(str(flat), "One", str(lib), read_tags=read)]
    assert one == ["c.mp3", "a.mp3"]
    # Loose untagged files at the library root are never an "album".
    (lib / "loose.mp3").write_bytes(b"x")
    assert album_tracks(str(lib), "Lib", str(lib), read_tags=read) == []


def test_album_scan_offers_albums_with_tracks_and_skips_reserved_folders(make_workspace, tmp_path, threads):
    lib = art_library(tmp_path / "Lib", albums=3, extra=[
        ("Quarantine/Run/Lost", True),        # reserved: never offered
        ("Artist9/Covers Only", False),        # a cover with nothing to play
    ])
    ws = make_workspace("home", library_path=lib)
    _home_ready(ws, threads)
    titles = sorted(a.title for a in ws.album_pool())
    assert titles == ["Album0", "Album1", "Album2"]
    for album in ws.album_pool():
        assert [os.path.basename(p) for p in album.tracks] == ["01 - A.mp3", "02 - B.mp3"]
        assert album.artist.startswith("Artist") and album.image is not None
    tiles = [t for t in ws._album_grid.tiles() if t.isVisibleTo(ws)]
    assert len(tiles) == 3 and all(t.album is not None for t in tiles)
    assert not ws._shuffle_btn.isEnabled(), "every album is already showing"


def test_tiles_show_placeholders_while_the_scan_runs(make_workspace, tmp_path, threads):
    lib = art_library(tmp_path / "Lib", albums=2)
    ws = make_workspace("home", library_path=lib)
    # Results arrive as queued signals: nothing is delivered yet.
    assert ws.is_finding_albums()
    tiles = [t for t in ws._album_grid.tiles() if t.isVisibleTo(ws)]
    assert len(tiles) == 6 and all(t.album is None and not t.isEnabled() for t in tiles)
    assert not ws._shuffle_btn.isEnabled()
    _home_ready(ws, threads)
    assert [t.album is not None for t in ws._album_grid.tiles() if t.isVisibleTo(ws)] == [True, True]


def test_clicking_an_album_plays_it(make_workspace, tmp_path, threads, qtbot):
    lib = art_library(tmp_path / "Lib", albums=2)
    ws = make_workspace("home", library_path=lib)
    _home_ready(ws, threads)
    played = []
    ws.play_tracks_requested.connect(lambda paths, label: played.append((paths, label)))
    tile = ws._album_grid.tiles()[0]
    qtbot.mouseClick(tile, QtCore.Qt.MouseButton.LeftButton)
    assert played == [(tile.album.tracks, tile.album.label)]
    assert tile.album.label == f"{tile.album.title} by {tile.album.artist}"
    assert tile.accessibleName().startswith("Play ")


def test_clicking_an_album_reaches_the_player(main_window, tmp_path, threads, monkeypatch):
    from gui.workspaces.player import PlayerWorkspace

    calls = []
    monkeypatch.setattr(PlayerWorkspace, "load_tracks_and_play",
                        lambda self, paths, label="": calls.append((list(paths), label)))
    lib = art_library(tmp_path / "Lib", albums=1)
    main_window.set_library(lib)
    home = main_window._workspaces["home"]
    assert wait_until(lambda: bool(home.shown_albums()), 10_000)
    home._album_grid.tiles()[0].click()
    assert calls and calls[0][0] == home.shown_albums()[0].tracks
    assert main_window.current_panel() is main_window._workspaces["player"]
    assert threads.wait_all(10_000) == []


def test_shuffle_shows_others_and_the_pool_is_kept_per_library(make_workspace, tmp_path, threads):
    import gui.workers as workers

    lib = art_library(tmp_path / "Lib", albums=9)
    other = art_library(tmp_path / "Other", albums=1)
    ws = make_workspace("home", library_path=lib)
    _home_ready(ws, threads)
    assert len(ws.album_pool()) == 9 and len(ws.shown_albums()) == 6
    assert ws._shuffle_btn.isEnabled()
    first = {a.key for a in ws.shown_albums()}
    ws._shuffle_btn.click()
    second = {a.key for a in ws.shown_albums()}
    assert len(second) == 6
    # The three not yet shown come first; the rest of the six fill up.
    assert {a.key for a in ws.album_pool()} - first <= second

    # Away and back: no new album scan, the same pool.
    started = []
    real = workers.start_worker

    def recording(thread, **kwargs):
        started.append(type(thread).__name__)
        return real(thread, **kwargs)

    workers.start_worker, saved = recording, workers.start_worker
    try:
        ws.set_library_path(other)
        _home_ready(ws, threads)
        assert [a.title for a in ws.album_pool()] == ["Album0"]
        ws.set_library_path(lib)
        _home_ready(ws, threads)
    finally:
        workers.start_worker = saved
    assert started.count("HomeAlbumScan") == 1, "the first library's albums were kept"
    assert len(ws.album_pool()) == 9 and len(ws.shown_albums()) == 6


def test_empty_states(make_workspace, tmp_path, threads):
    ws = make_workspace("home")
    assert ws._spin_empty.isVisibleTo(ws) and not ws._album_grid.isVisibleTo(ws)
    assert ws._spin_empty.title() == "No library chosen"
    assert ws._runs_empty.isVisibleTo(ws)

    bare = tmp_path / "Bare"
    (bare / "Artist" / "Album").mkdir(parents=True)
    _track(str(bare / "Artist" / "Album" / "01.mp3"))     # tracks, no cover art
    ws = make_workspace("home", library_path=bare)
    _home_ready(ws, threads)
    assert ws._spin_empty.isVisibleTo(ws) and ws._spin_empty.title() == "No albums to play yet"
    assert ws._runs_empty.isVisibleTo(ws) and ws._runs_empty.title() == "No runs yet"


# ── Library health, recent runs, duplicates ──────────────────────────────────

def test_library_counts_include_albums(tmp_path):
    from gui.library_stats import last_counts, library_counts, library_stats

    lib = art_library(tmp_path / "Lib", albums=3, extra=[("Quarantine/Run/Lost", True)])
    _track(os.path.join(lib, "Artist0", "loose.mp3"))         # under an artist: no album
    counts = library_counts(lib)
    assert (counts.tracks, counts.artists, counts.albums) == (7, 3, 3)
    assert last_counts(lib) is None or last_counts(lib) == counts
    assert library_stats(lib) == (7, counts.size_bytes, 3)
    assert last_counts(lib) == counts, "the full count is remembered for Home"


def test_health_tiles_layout_check_and_the_indexer_action(make_workspace, tmp_path, threads):
    lib = art_library(tmp_path / "Lib", albums=1)
    os.makedirs(os.path.join(lib, "Manual Review"))
    _track(os.path.join(lib, "Manual Review", "x.mp3"))
    ws = make_workspace("home", library_path=lib)
    _home_ready(ws, threads)
    nav = []
    ws.navigate_requested.connect(nav.append)

    mr = ws._folder_tiles["Manual Review"]
    assert mr.tile.value() == "1" and mr.tile.tone() == "warning"
    assert mr.open_btn.isVisibleTo(ws) and mr.open_btn.property("openPath").endswith("Manual Review")
    assert not mr.open_btn.icon().isNull() and mr.open_btn.accessibleName()
    assert not ws._folder_tiles["Quarantine"].open_btn.isVisibleTo(ws)

    ok, text = ws.layout_check()
    assert ok is False and text == "not organised by the Indexer yet"
    from gui.themes.manager import get_manager
    assert ws._layout_label.palette().color(QtGui.QPalette.ColorRole.WindowText).name() \
        == QtGui.QColor(get_manager().current.warning).name()
    ws._indexer_btn.click()
    assert nav == ["indexer"]

    os.makedirs(os.path.join(lib, "By Artist"))
    ws.refresh()
    _home_ready(ws, threads)
    assert ws.layout_check() == (True, "matches the Indexer's layout")
    assert not ws._indexer_btn.isVisibleTo(ws)
    get_manager().apply("pearl")
    assert ws._layout_label.palette().color(QtGui.QPalette.ColorRole.WindowText).name() \
        == QtGui.QColor(get_manager().current.success).name(), "follows the theme"


def test_stats_tiles_take_the_headers_numbers(make_workspace, tmp_path, threads):
    from gui.library_stats import library_stats

    lib = art_library(tmp_path / "Lib", albums=2)
    ws = make_workspace("home", library_path=lib)
    _home_ready(ws, threads)
    assert set(ws.library_stats_shown().values()) == {"–"}
    tracks, size, artists = library_stats(lib)
    ws.set_library_stats(tracks, size / 1024 ** 3, artists)
    assert ws.library_stats_shown() == {"tracks": "4", "size": "0 MB", "artists": "2", "albums": "2"}
    ws.set_library_stats(5, 162.44, 391)               # a count from elsewhere: no albums
    assert ws.library_stats_shown() == {"tracks": "5", "size": "162.4 GB", "artists": "391"}


def test_recent_run_rows(make_workspace, tmp_path, threads, monkeypatch):
    import gui.workspaces.home as home_mod

    opened = []
    monkeypatch.setattr(home_mod, "open_path", lambda p: opened.append(p) or True)
    lib = art_library(tmp_path / "Lib", albums=1)
    _docs(lib, "indexer_executed_20260101_000000.txt", age_s=3 * 86400)
    _docs(lib, "tag_changes_genre_normalize_20260101-000000.jsonl", age_s=60)
    _docs(lib, "tag_changes_genre_fill_20260102-000000_2.jsonl", age_s=30)
    ws = make_workspace("home", library_path=lib)
    _home_ready(ws, threads)
    rows = ws.run_rows()
    assert [r.title_label.text() for r in rows] == [
        "Tag changes · Fill genres", "Tag changes · Normalize genres", "Indexer run"]
    assert rows[0].detail_label.text().startswith("today ")
    assert rows[0].detail_label.toolTip() == rows[0].run.path
    rows[2].open_btn.click()
    assert opened == [rows[2].run.path]
    assert not ws._runs_empty.isVisibleTo(ws)


def test_relative_when():
    from gui.workspaces.home import relative_when

    now = datetime.datetime(2026, 11, 8, 15, 0)
    stamp = lambda **kw: (now - datetime.timedelta(**kw)).timestamp()   # noqa: E731
    assert relative_when(stamp(hours=1), now) == "today 14:00"
    assert relative_when(stamp(days=1), now) == "yesterday 15:00"
    assert relative_when(stamp(days=3), now) == "Thursday 15:00"
    assert relative_when(stamp(days=30), now) == "2026-10-09"


def _preview_json(groups: int, review: int, stamp: str, plan_size: int = 0) -> str:
    payload = {"generated_at": stamp,
               "summary": {"groups": groups, "review_required": review, "review_flags": ["a", "b"]},
               "plan": {"groups": [{"pad": "x" * 100} for _ in range(plan_size)]}}
    return json.dumps(payload, indent=2)


def test_last_duplicate_scan_reads_only_the_summary(tmp_path):
    from gui.workspaces.home import last_duplicate_scan

    lib = str(tmp_path / "Lib")
    os.makedirs(lib)
    assert last_duplicate_scan(lib) is None
    path = _docs(lib, "duplicate_preview.json",
                 _preview_json(18, 3, "2026-11-07T14:02:00", plan_size=50_000), age_s=100)
    assert os.path.getsize(path) > 4_000_000
    scan = last_duplicate_scan(lib)
    assert (scan.groups, scan.review_required, scan.preview_html, scan.cleaned_since) == (18, 3, "", False)
    assert datetime.datetime.fromtimestamp(scan.scanned_at) == datetime.datetime(2026, 11, 7, 14, 2)
    _docs(lib, "duplicate_preview.html", "<html>")
    _docs(lib, "duplicate_execution_reports/consolidation_run_1/reports/execution_report.html")
    scan = last_duplicate_scan(lib)
    assert scan.preview_html.endswith("duplicate_preview.html") and scan.cleaned_since
    _docs(lib, "duplicate_preview.json", "{ not json")
    assert last_duplicate_scan(lib) is None


def test_duplicates_card(make_workspace, tmp_path, threads):
    lib = art_library(tmp_path / "Lib", albums=1)
    ws = make_workspace("home", library_path=lib)
    _home_ready(ws, threads)
    nav = []
    ws.navigate_requested.connect(nav.append)
    assert "No duplicate scan yet" in ws._dupes_text.text()
    assert not ws._dupes_stats.isVisibleTo(ws) and not ws._dupes_open_btn.isVisibleTo(ws)
    ws._find_dupes_btn.click()
    assert nav == ["duplicates/find"]

    _docs(lib, "duplicate_preview.json", _preview_json(18, 3, datetime.datetime.now().isoformat()))
    html = _docs(lib, "duplicate_preview.html", "<html>")
    ws.refresh()
    _home_ready(ws, threads)
    assert ws._dupes_stats.isVisibleTo(ws)
    assert ws._dupes_stats.tile("groups").value() == "18"
    assert ws._dupes_stats.tile("review").value() == "3"
    assert ws._dupes_stats.tile("review").tone() == "warning"
    assert ws._dupes_text.text().startswith("From the last scan, today ")
    assert ws._dupes_open_btn.isVisibleTo(ws) and ws._dupes_open_btn.property("openPath") == html


def test_home_buttons_carry_glyphs(make_workspace):
    ws = make_workspace("home")
    for btn in [ws._shuffle_btn, ws._refresh_btn, ws._indexer_btn, *ws._action_buttons.values()]:
        assert not btn.icon().isNull(), btn.text()
    assert ws._refresh_btn.accessibleName() and ws._refresh_btn.toolTip()
    assert all("🔍" not in b.text() and "📄" not in b.text() for b in ws.findChildren(QtWidgets.QPushButton))


# ── Layout: two columns when wide, no horizontal scroll at 900 px ────────────

def test_pages_fit_the_minimum_window(main_window, tmp_path, threads):
    lib = art_library(tmp_path / "Lib", albums=7)
    _docs(lib, "indexer_executed_20260101_000000_with_a_rather_long_name_indeed.txt")
    main_window.set_library(lib)
    home = main_window._workspaces["home"]
    assert wait_until(lambda: home.health() is not None and not home.is_finding_albums(), 10_000)
    main_window.resize(1300, 860)
    process_events(30)
    assert home.columns() == 2
    main_window.resize(900, 600)
    for key in ("home", "activity", "help"):
        main_window._on_nav_changed(key)
        process_events(30)
        panel = main_window._workspaces[key]
        bar = panel._scroll_area.horizontalScrollBar()
        assert bar.maximum() == 0, f"{key} scrolls sideways at 900 px"
    assert home.columns() == 1
    tiles = [t for t in home._album_grid.tiles() if t.isVisibleTo(home)]
    assert len(tiles) == 6 and len({t.y() for t in tiles}) == 1, "six across, one row"
    assert len({t.width() for t in tiles}) == 1, "equal tiles"
    assert threads.wait_all(10_000) == []


# ── Activity ──────────────────────────────────────────────────────────────────

class _PercentWorker(QtCore.QThread):
    progress = Signal(int)

    def __init__(self):
        super().__init__()
        self.go = threading.Event()
        self.stop = threading.Event()

    def cancel(self):
        self.stop.set()

    def run(self):
        self.go.wait(5)
        self.progress.emit(40)
        self.stop.wait(10)


class _DoneTotalWorker(_PercentWorker):
    progress = Signal(int, int, str)

    def run(self):
        self.go.wait(5)
        self.progress.emit(3, 4, "scan")
        self.stop.wait(10)


class _OtherWorker(_PercentWorker):
    progress = Signal(int, int, int, int, int, int)   # compression's: not (done, total)

    def run(self):
        self.go.wait(5)
        self.progress.emit(10, 2, 0, 0, 0, 0)
        self.stop.wait(10)


@pytest.mark.parametrize("cls, expected", [
    (_PercentWorker, 0.4), (_DoneTotalWorker, 0.75), (_OtherWorker, None),
])
def test_activity_progress_bars_show_only_known_progress(main_window, threads, cls, expected):
    from gui.workers import start_worker

    main_window._on_nav_changed("activity")
    activity = main_window._workspaces["activity"]
    worker = cls()
    start_worker(worker, owner=main_window._workspaces["indexer"], name="Progress job")
    process_events(10)
    (row,) = [r for r in activity.job_rows() if r.job.name == "Progress job"]
    assert row.fraction is None
    worker.go.set()
    if expected is None:
        process_events(50)
        assert row.fraction is None
    else:
        assert wait_until(lambda: row.fraction is not None, 3000)
        assert row.fraction == pytest.approx(expected)
    worker.cancel()
    assert threads.wait_all(5000) == []
    assert wait_until(lambda: activity._jobs_empty.isVisible(), 3000)


def test_activity_lists_finished_jobs_newest_first(main_window, threads):
    from gui.workers import start_worker

    main_window._on_nav_changed("activity")
    activity = main_window._workspaces["activity"]
    for name in ("First", "Second"):
        worker = _PercentWorker()
        worker.go.set()
        start_worker(worker, owner=main_window._workspaces["indexer"], name=name)
        process_events(10)
        assert name in [r.job.name for r in activity.job_rows()]
        worker.stop.set()           # ends by itself: not a cancel
        assert threads.wait_all(5000) == []
        process_events(20)
    names = [f.name for f in activity.finished_jobs()]
    assert names.index("Second") < names.index("First")
    assert not any(f.stopped for f in activity.finished_jobs() if f.name in ("First", "Second"))
    assert activity._recent_caption.isVisibleTo(activity)
    assert activity._jobs_empty.isVisibleTo(activity)


# ── Help ──────────────────────────────────────────────────────────────────────

OLD_NAMES = ("Utilities", "Similarity", "Genre Normalizer", "Clustered", "Music Graph", "Compression")


def test_help_uses_current_page_names_and_no_emoji(make_workspace):
    from gui.widgets.link_button import LinkButton

    ws = make_workspace("help")
    texts = [w.text() for w in ws.findChildren(QtWidgets.QLabel)]
    texts += [b.text() for b in ws.findChildren(QtWidgets.QAbstractButton)]
    for text in texts:
        for old in OLD_NAMES:
            assert old not in text, f"old page name {old!r} in {text!r}"
        assert not any(ord(ch) >= 0x1F000 or 0x2600 <= ord(ch) < 0x2800 for ch in text), text
    assert all(isinstance(b, LinkButton) for b in ws._doc_buttons)


def test_help_links_paint_in_the_accent(make_workspace):
    from gui.themes.manager import get_manager

    ws = make_workspace("help")
    link = ws._doc_buttons[0]
    assert link.colour().name() == QtGui.QColor(get_manager().current.accent).name()
    get_manager().apply("pearl")
    assert link.colour().name() == QtGui.QColor(get_manager().current.accent).name()
    assert ws._issues_btn.toolTip().startswith("https://github.com/")


# ── The Activity Log strip and the Now Playing capsule ───────────────────────

@pytest.mark.parametrize("size", [(1300, 860), (900, 600)])
def test_the_capsule_never_covers_the_log_strip(main_window, size):
    bar = main_window._now_playing_bar
    bar.update_now_playing("Polar Night", "Aurora Lines", "Dawn Signals", QtGui.QPixmap())
    main_window.resize(*size)
    drawer = main_window._log_drawer
    central = main_window.centralWidget()
    for expanded in (False, True, False):
        if drawer.is_expanded() != expanded:
            drawer.toggle()
        process_events(30)
        assert bar.isVisible()
        strip = QtCore.QRect(drawer.mapTo(central, QtCore.QPoint(0, 0)), drawer.size())
        assert not bar.geometry().intersects(strip), (size, expanded)
        for control in (drawer._clear_btn, drawer._handle):
            rect = QtCore.QRect(control.mapTo(central, QtCore.QPoint(0, 0)), control.size())
            assert not bar.geometry().intersects(rect)
        assert drawer.geometry().bottom() < drawer.parentWidget().height() + 1, "inside the window"
        if expanded:
            assert drawer._text.height() >= 60, "a short window shrinks the log, never hides it"


def test_log_strip_status_dot_and_muted_text(qtbot):
    from gui.themes.manager import get_manager
    from gui.widgets.log_drawer import LogDrawer

    drawer = LogDrawer()
    qtbot.addWidget(drawer)
    drawer.set_status("Scanning…", "running")
    t = get_manager().current
    assert drawer.status_text() == "Scanning…", "no bullet character in the text"
    assert drawer._status_dot.colour().name() == QtGui.QColor(t.log_warn).name()
    assert drawer._status_chip.palette().color(QtGui.QPalette.ColorRole.WindowText).name() \
        == QtGui.QColor(t.text_secondary).name()
    assert drawer._toggle_icon.text() == "Activity Log"
