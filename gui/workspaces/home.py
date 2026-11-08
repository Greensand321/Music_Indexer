"""Home page: the library at a glance, and something to play.

* **Spin something** (the hero card): six random albums from the library as
  album tiles; clicking one plays it (``play_tracks_requested``, which the
  main window routes to the Player); "Shuffle again" shows six others. The
  albums come from the cover scanner the landing and the backdrop use
  (``gui.widgets.landing._CoverScan``), in a read-only JobRegistry job, kept
  per library so coming back to Home is instant. Each album's tracks are
  read from its folder (``album_tracks``): only albums with cover art *and*
  tracks are offered, never anything in the reserved folders.
* **Library health**: the numbers the main window's stats scan already has
  (tracks, size, artists, and albums from the same walk), how many tracks
  wait in ``Manual Review/``, ``Quarantine/`` and ``Not Sorted/`` (each with
  "Open folder"), and whether the folder has the Indexer's layout (the
  library Validator); when it has not, one action opens the Indexer.
* **Recent runs**: the newest report or log each workflow left in ``Docs/``
  (Indexer execute log, duplicate execution report, Library Sync report,
  tag change logs, file cleanup log), each with "Open".
* **Duplicates**: the last duplicate scan's own summary
  (``Docs/duplicate_preview.json``, written by the Duplicates page) when
  there is one, and "Find duplicates".
* **Quick actions** that navigate to the page that does the job.

Everything here is read-only and cheap: Home never walks the whole library
itself. ``scan_library_health`` only lists ``Docs/`` and walks the three
reserved folders above; the album scan stops after ``ALBUM_POOL`` covers.
Both run in read-only JobRegistry jobs (``ask_before_quit=False``), so they
never block the GUI and quitting cancels them silently.
"""
from __future__ import annotations

import datetime
import glob
import json
import math
import os
import random
import re
from dataclasses import dataclass, field

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.widgets.glass import ElidedLabel, GlassCard
from gui.widgets.landing import _CoverScan
from gui.workspaces.base import WorkspaceBase
from utils.reserved_folders import DOCS, MANUAL_REVIEW, NOT_SORTED, QUARANTINE, is_reserved_path

_AUDIO_EXTS = frozenset({".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"})

# Folder → what its tracks are waiting for.
ATTENTION_FOLDERS: tuple[tuple[str, str], ...] = (
    (MANUAL_REVIEW, "tracks missing the tags the Indexer needs"),
    (QUARANTINE, "duplicate losers awaiting your review"),
    (NOT_SORTED, "tracks you keep out of the Indexer's way"),
)

# (kind, glob patterns relative to the library, newest wins; one row per
# "group" — tag change logs get one row per tool).
RUN_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Indexer run", (f"{DOCS}/indexer_executed_*.txt",)),
    ("Duplicate cleanup", (
        f"{DOCS}/duplicate_execution_reports/consolidation_run_*/reports/execution_report.html",
    )),
    ("Library Sync run", (f"{DOCS}/LibrarySyncExecuted_*.html", f"{DOCS}/LibrarySyncExecuted.html",
                          f"{DOCS}/LibrarySyncAudit_*.json")),
    ("Tag changes", (f"{DOCS}/tag_changes_*.jsonl",)),
    ("File cleanup", (f"{DOCS}/file_cleanup_*.jsonl",)),
)

# ``tag_changes_<tool>_<YYYYMMDD-HHMMSS>[_n].jsonl`` (utils/tag_change_log.py)
_TAG_LOG = re.compile(r"^tag_changes_(?P<tool>.+)_(?P<stamp>\d{8}-\d{6})(?:_\d+)?\.jsonl$")
TAG_TOOL_NAMES = {
    "tagfixer": "Tag Fixer",
    "genre_fill": "Fill genres",
    "genre_normalize": "Normalize genres",
    "genre_fill_cli": "Genre update",
    "revert": "Undo changes",
}

# Where the Duplicates page leaves its last scan (gui/workspaces/duplicates.py).
DUPLICATE_PREVIEW_JSON = "duplicate_preview.json"
DUPLICATE_PREVIEW_HTML = "duplicate_preview.html"

SPIN_COUNT = 6       # album tiles in "Spin something"
ALBUM_POOL = 24      # albums a scan collects, so "Shuffle again" has others to show


# ── Library health ────────────────────────────────────────────────────────────

@dataclass
class RecentRun:
    kind: str          # "Indexer run", "Tag changes", …
    path: str
    mtime: float
    detail: str = ""   # e.g. the tool that wrote a tag change log

    @property
    def when(self) -> str:
        return datetime.datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")

    @property
    def title(self) -> str:
        """"Tag changes · Tag Fixer" (the tool's page name when known)."""
        if not self.detail:
            return self.kind
        return f"{self.kind} · {TAG_TOOL_NAMES.get(self.detail, self.detail)}"


@dataclass
class DuplicateScan:
    """The summary the Duplicates page saved with its last preview."""
    groups: int
    review_required: int
    scanned_at: float            # epoch seconds
    preview_json: str
    preview_html: str = ""       # "" when the HTML preview is missing
    cleaned_since: bool = False  # a cleanup report is newer than the scan


@dataclass
class LibraryHealth:
    root: str
    layout_ok: bool = False
    layout_messages: list[str] = field(default_factory=list)
    # folder name → audio files in it (None: the folder does not exist)
    folder_counts: dict[str, int | None] = field(default_factory=dict)
    recent_runs: list[RecentRun] = field(default_factory=list)
    duplicates: DuplicateScan | None = None


def _top_level_dir(root: str, name: str) -> str | None:
    """``root/<name>`` matched case-insensitively (as the reserved folders are)."""
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    for entry in entries:
        if entry.casefold() == name.casefold() and os.path.isdir(os.path.join(root, entry)):
            return os.path.join(root, entry)
    return None


def _count_audio(folder: str, should_stop) -> int | None:
    count = 0
    for _dirpath, _dirs, files in os.walk(folder):
        if should_stop():
            return None
        count += sum(1 for f in files if os.path.splitext(f)[1].lower() in _AUDIO_EXTS)
    return count


def _tag_tool(path: str) -> str:
    """``tag_changes_<tool>_<stamp>.jsonl`` → ``<tool>`` (tools may contain "_")."""
    name = os.path.basename(path)
    m = _TAG_LOG.match(name)
    if m:
        return m.group("tool")
    stem = name[len("tag_changes_"):]
    return stem.split("_", 1)[0] if "_" in stem else ""


def recent_runs(root: str) -> list[RecentRun]:
    """The newest log/report of each workflow under ``root/Docs``, newest first."""
    runs: list[RecentRun] = []
    for kind, patterns in RUN_SOURCES:
        found: list[str] = []
        for pattern in patterns:
            found.extend(glob.glob(os.path.join(glob.escape(root), pattern)))
            if found:
                break   # later patterns are fallbacks
        stamped = []
        for path in found:
            try:
                stamped.append((os.path.getmtime(path), path))
            except OSError:
                continue
        if not stamped:
            continue
        if kind == "Tag changes":
            newest_per_tool: dict[str, tuple[float, str]] = {}
            for mtime, path in stamped:
                tool = _tag_tool(path)
                if tool not in newest_per_tool or mtime > newest_per_tool[tool][0]:
                    newest_per_tool[tool] = (mtime, path)
            for tool, (mtime, path) in newest_per_tool.items():
                runs.append(RecentRun(kind, path, mtime, tool))
        else:
            mtime, path = max(stamped)
            runs.append(RecentRun(kind, path, mtime))
    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def _summary_from_head(head: str) -> tuple[int, int, str] | None:
    """(groups, review_required, generated_at) from the start of the preview
    JSON. ``export_consolidation_preview`` writes ``generated_at`` and
    ``summary`` before the (possibly huge) plan, so a few KB are enough."""
    start = head.find('"summary"')
    if start < 0:
        return None
    block = head[start:start + 600]
    groups = re.search(r'"groups"\s*:\s*(\d+)', block)
    if groups is None:
        return None
    review = re.search(r'"review_required"\s*:\s*(\d+)', block)
    stamp = re.search(r'"generated_at"\s*:\s*"([^"]+)"', head)
    return int(groups.group(1)), int(review.group(1)) if review else 0, stamp.group(1) if stamp else ""


def last_duplicate_scan(root: str) -> DuplicateScan | None:
    """The summary of the last duplicate scan the Duplicates page saved under
    ``Docs/`` (None when it never ran here, or the file is unreadable)."""
    docs = _top_level_dir(root, DOCS)
    if docs is None:
        return None
    json_path = os.path.join(docs, DUPLICATE_PREVIEW_JSON)
    try:
        mtime = os.path.getmtime(json_path)
        with open(json_path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(8192)
        parsed = _summary_from_head(head)
        if parsed is None and os.path.getsize(json_path) <= 4 * 1024 * 1024:
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            summary = data.get("summary") or {}
            parsed = (int(summary["groups"]), int(summary.get("review_required") or 0),
                      str(data.get("generated_at") or ""))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if parsed is None:
        return None
    groups, review, stamp = parsed
    scanned_at = mtime
    if stamp:
        try:
            scanned_at = datetime.datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            pass
    html_path = os.path.join(docs, DUPLICATE_PREVIEW_HTML)
    reports = glob.glob(os.path.join(glob.escape(docs), "duplicate_execution_reports",
                                     "consolidation_run_*", "reports", "execution_report.html"))
    newest_report = 0.0
    for report in reports:
        try:
            newest_report = max(newest_report, os.path.getmtime(report))
        except OSError:
            continue
    return DuplicateScan(groups, review, scanned_at, json_path,
                         html_path if os.path.isfile(html_path) else "",
                         newest_report > mtime)


def scan_library_health(root: str, should_stop=lambda: False) -> LibraryHealth | None:
    """Everything Home shows about ``root`` (None if stopped early)."""
    health = LibraryHealth(root=root)
    try:
        import validator
        health.layout_ok, health.layout_messages = validator.validate_soundvault_structure(root)
    except Exception as exc:  # noqa: BLE001 - one check must not hide the rest
        health.layout_ok, health.layout_messages = False, [f"Could not check the layout: {exc}"]
    for name, _why in ATTENTION_FOLDERS:
        if should_stop():
            return None
        folder = _top_level_dir(root, name)
        if folder is None:
            health.folder_counts[name] = None
            continue
        count = _count_audio(folder, should_stop)
        if count is None:
            return None
        health.folder_counts[name] = count
    if should_stop():
        return None
    health.recent_runs = recent_runs(root)
    try:
        health.duplicates = last_duplicate_scan(root)
    except Exception:  # noqa: BLE001 - a summary must not hide the rest
        health.duplicates = None
    return health


class HomeScanWorker(QtCore.QThread):
    # Not called ``finished``: that would shadow QThread.finished.
    health_ready = Signal(int, object)   # generation, LibraryHealth

    def __init__(self, root: str, generation: int) -> None:
        super().__init__()
        self.root = root
        self.generation = generation

    def run(self) -> None:
        try:
            result = scan_library_health(self.root, self.isInterruptionRequested)
        except Exception as exc:  # noqa: BLE001 - best effort
            result = LibraryHealth(root=self.root, layout_messages=[f"Could not read: {exc}"])
        if result is not None:
            self.health_ready.emit(self.generation, result)


def open_path(path: str) -> bool:
    """Open a file or folder with the system's default application."""
    return QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))


def relative_when(mtime: float, now: datetime.datetime | None = None) -> str:
    """``today 14:02`` · ``yesterday 09:10`` · ``Monday 14:02`` · ``2026-10-01``."""
    now = now or datetime.datetime.now()
    dt = datetime.datetime.fromtimestamp(mtime)
    days = (now.date() - dt.date()).days
    hm = dt.strftime("%H:%M")
    if days == 0:
        return f"today {hm}"
    if days == 1:
        return f"yesterday {hm}"
    if 1 < days < 7:
        return f"{dt.strftime('%A')} {hm}"
    return dt.strftime("%Y-%m-%d")


# ── Albums ("Spin something") ─────────────────────────────────────────────────

@dataclass
class AlbumPick:
    title: str
    artist: str
    folder: str
    tracks: list[str]
    image: QtGui.QImage | None = None

    @property
    def key(self) -> tuple[str, str]:
        return os.path.normcase(os.path.abspath(self.folder)), self.title.casefold()

    @property
    def label(self) -> str:
        """What the Player logs for the queue: "Granite by Basalt"."""
        return f"{self.title} by {self.artist}" if self.artist else self.title


def _read_tags(path: str) -> dict:
    try:
        from utils.audio_metadata_reader import read_metadata
        tags, _covers, _err, _hint = read_metadata(path, include_cover=False)
        return tags or {}
    except Exception:  # noqa: BLE001 - an unreadable file is just untagged
        return {}


def _as_int(value) -> int | None:
    try:
        return int(str(value).split("/", 1)[0])
    except (TypeError, ValueError):
        return None


def _natural(name: str) -> list:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", name)]


def album_tracks(folder: str, album: str, library_root: str, *, read_tags=_read_tags,
                 should_stop=lambda: False) -> list[str]:
    """The tracks of ``album`` in ``folder`` (not its subfolders), in disc and
    track order. Read-only.

    A track belongs to the album when its album tag matches; an untagged track
    does when the folder is named after the album (an album folder) and is not
    the library root. So a flat folder holding several albums yields each
    album's own tracks, and loose files at the root are never an "album"."""
    try:
        names = [e.name for e in os.scandir(folder)
                 if e.is_file(follow_symlinks=False)
                 and os.path.splitext(e.name)[1].lower() in _AUDIO_EXTS]
    except OSError:
        return []
    want = album.strip().casefold()
    if not names or not want:
        return []
    is_root = os.path.normcase(os.path.abspath(folder)) == os.path.normcase(os.path.abspath(library_root))
    folder_is_album = not is_root and os.path.basename(os.path.normpath(folder)).casefold() == want
    rows = []
    for name in names:
        if should_stop():
            return []
        path = os.path.join(folder, name)
        tags = read_tags(path)
        tagged = str(tags.get("album") or "").strip().casefold()
        if tagged == want or (not tagged and folder_is_album):
            disc = _as_int(tags.get("disc") or tags.get("discnumber")) or 0
            track = _as_int(tags.get("track") or tags.get("tracknumber"))
            rows.append((disc, track if track is not None else 1 << 20, _natural(name), path))
    rows.sort(key=lambda r: r[:3])
    return [r[3] for r in rows]


class HomeAlbumScan(_CoverScan, QtCore.QThread):
    """Collects up to ``ALBUM_POOL`` albums with covers and tracks, read-only.

    The cover search is the landing's and the backdrop's (``_CoverScan``),
    without their history and limited to the library proper (the reserved
    folders are skipped). Albums arrive one by one (``album_found``) so the
    tiles fill as they are found."""

    album_found = Signal(int, object)   # generation, AlbumPick
    scan_done = Signal(int, int)        # generation, albums found

    COVER_PX = 288   # baked once; tiles scale it down (about 140–200 px, ×DPR)

    def __init__(self, library_path: str, generation: int, count: int = ALBUM_POOL) -> None:
        QtCore.QThread.__init__(self)
        self._init_cover_scan(library_path, count, tile_w=self.COVER_PX, radius=0,
                              vignette=False, use_history=False)
        self.generation = generation
        self.setObjectName("HomeAlbumScan")

    def _ordered_dirs(self, history) -> list[str]:
        root = self._library
        return [d for d in super()._ordered_dirs(history) if not is_reserved_path(d, root)]

    def run(self) -> None:  # noqa: N802
        seen: set[tuple[str, str]] = set()
        found = 0

        def deliver(_index, image, folder, album, artist) -> None:
            nonlocal found
            if self._stopping():
                return   # also skips the scanner's repeat-fill after the search
            key = (os.path.normcase(os.path.abspath(folder)), (album or "").casefold())
            if key in seen:
                return
            seen.add(key)
            tracks = album_tracks(folder, album or "", self._library, should_stop=self._stopping)
            if not tracks:
                return   # a cover with nothing to play
            found += 1
            self.album_found.emit(self.generation, AlbumPick(
                album or os.path.basename(folder), artist or "", folder, tracks, image))

        try:
            self.scan(deliver)
        except Exception:  # noqa: BLE001 - a broken file must not end the page's scan badly
            import logging
            logging.getLogger(__name__).exception("Home album scan failed")
        if not self.isInterruptionRequested():
            self.scan_done.emit(self.generation, found)


# Quick actions: (button label, navigation key, tooltip, glyph). "Find
# duplicates" lives on the Duplicates card.
QUICK_ACTIONS: tuple[tuple[str, str, str, str], ...] = (
    ("Organize folders", "indexer", "Indexer: preview, then normalize folders and names", "folder"),
    ("Merge new music", "library_sync", "Library Sync: bring an incoming folder in", "download"),
    ("Clean up file names", "file_cleanup", "File Cleanup: remove ' (1)' / ' - Copy' suffixes", "file"),
    ("Fix tags", "metadata/tag_fixer", "Metadata: look tracks up and fix their tags", "music"),
    ("Explore the Sound Map", "sound_map", "Sound Map: analyse how tracks sound, then explore", "search"),
    ("Make a playlist", "playlists", "Playlists: rule-based playlist creation", "list"),
)


# ── Widgets ───────────────────────────────────────────────────────────────────

def _tokens():
    from gui.themes.manager import get_manager
    return get_manager().current


class _ReflowGrid(QtWidgets.QWidget):
    """Equal-width items in as many columns as fit (between ``min_cols`` and
    ``max_cols``), rows as needed. Laid out by hand, so the grid never asks
    for more width than one column: it reflows instead of scrolling."""

    EQUAL_WIDTHS = False

    def __init__(self, items: list[QtWidgets.QWidget], *, min_item: int, max_cols: int,
                 min_cols: int = 1, gap: int = 10, even: bool = False,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._items = list(items)
        self._min_item = min_item
        self._max_cols = max_cols
        self._min_cols = min_cols
        self._gap = gap
        self._even = even
        for item in self._items:
            item.setParent(self)
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                       QtWidgets.QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def items(self) -> list[QtWidgets.QWidget]:
        return list(self._items)

    def visible_items(self) -> list[QtWidgets.QWidget]:
        return [w for w in self._items if not w.isHidden()]

    def columns_for(self, width: int) -> int:
        n = max(1, len(self.visible_items()))
        fit = max(1, (width + self._gap) // (self._min_item + self._gap))
        cols = max(self._min_cols, min(self._max_cols, n, fit))
        if self._even and cols < n and n % cols:
            # Prefer a column count that fills every row (4 tiles: 4 or 2, never 3 + 1).
            for c in range(cols, 1, -1):
                if n % c == 0:
                    return max(self._min_cols, c)
        return cols

    def _item_width(self, width: int, cols: int) -> float:
        return max(1.0, float((width - (cols - 1) * self._gap) // cols))

    def _row_height(self, item_w: float) -> int:
        items = self.visible_items()
        return max((w.sizeHint().height() for w in items), default=0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        items = self.visible_items()
        if not items:
            return 0
        cols = self.columns_for(width)
        rows = math.ceil(len(items) / cols)
        return rows * self._row_height(self._item_width(width, cols)) + (rows - 1) * self._gap

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        n = max(1, len(self.visible_items()))
        cols = min(self._max_cols, n)
        w = cols * self._min_item + (cols - 1) * self._gap
        return QtCore.QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        cols = max(1, self._min_cols)
        w = cols * self._min_item + (cols - 1) * self._gap
        return QtCore.QSize(w, self.heightForWidth(w))

    def relayout(self) -> None:
        self.updateGeometry()
        self._place()

    def _place(self) -> None:
        width = self.width()
        items = self.visible_items()
        if not items or width <= 0:
            return
        cols = self.columns_for(width)
        item_w = int(self._item_width(width, cols))
        row_h = self._row_height(item_w)
        for i, item in enumerate(items):
            r, c = divmod(i, cols)
            if self.EQUAL_WIDTHS:
                # Whole pixels, the same for every item: equal tiles and baselines.
                x, w = c * (item_w + self._gap), item_w
            else:
                # Spread the spare pixels so the last column ends at the edge.
                x = c * (width + self._gap) // cols
                w = (c + 1) * (width + self._gap) // cols - self._gap - x
            item.setGeometry(x, r * (row_h + self._gap), w, row_h)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place()


class _FolderTile(QtWidgets.QWidget):
    """A ``StatTile`` for a reserved folder, with an "Open folder" glyph button."""

    def __init__(self, name: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from gui.widgets.glass_parts import StatTile
        from gui.widgets.glyphs import set_glyph

        self.folder_name = name
        self.tile = StatTile("–", f"in {name}", parent=self)
        self.tile.setObjectName(f"stat_{name.lower().replace(' ', '_')}")
        self.open_btn = QtWidgets.QPushButton(self)
        self.open_btn.setObjectName("homeFolderBtn")
        self.open_btn.setFlat(True)
        self.open_btn.setFixedSize(28, 28)
        self.open_btn.setToolTip(f"Open {name}/")
        self.open_btn.setAccessibleName(f"Open {name} folder")
        set_glyph(self.open_btn, "folder", role="muted", size=14)
        self.open_btn.hide()
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.tile.sizeHint()

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.tile.minimumSizeHint()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.tile.setGeometry(self.rect())
        self.open_btn.move(self.width() - self.open_btn.width() - 6, 6)
        self.open_btn.raise_()


class _RunRow(QtWidgets.QWidget):
    """One recent run: a dot, what ran and when, and an "Open" link."""

    def __init__(self, run: RecentRun, dot: str, on_open, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from gui.themes.glass import apply_font
        from gui.widgets.link_button import LinkButton

        self.setObjectName("homeRunRow")
        self.run = run
        self._dot = dot
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(32, 7, 8, 7)
        lay.setSpacing(10)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(1)
        self.title_label = ElidedLabel(run.title)
        self.title_label.setObjectName("homeRowTitle")
        apply_font(self.title_label, "body", weight=600)
        self.detail_label = ElidedLabel(
            f"{relative_when(run.mtime)} · {os.path.basename(run.path)}",
            QtCore.Qt.TextElideMode.ElideMiddle)
        self.detail_label.setObjectName("mutedLabel")
        apply_font(self.detail_label, "caption")
        self.detail_label.setToolTip(run.path)
        text.addWidget(self.title_label)
        text.addWidget(self.detail_label)
        lay.addLayout(text, 1)
        self.open_btn = LinkButton("Open", role="caption")
        self.open_btn.setProperty("openPath", run.path)
        self.open_btn.setToolTip(f"Open {run.path}")
        self.open_btn.setAccessibleName(f"Open the {run.title} report")
        self.open_btn.clicked.connect(on_open)
        lay.addWidget(self.open_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface, qcolor
        from gui.themes.tokens import Surface

        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            paint_surface(p, QtCore.QRectF(self.rect()), Surface(m.hover, radius=12))
            c = QtCore.QPointF(16.0, self.height() / 2.0)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(qcolor(m.hover))
            p.drawEllipse(c, 8.0, 8.0)
            p.setBrush(qcolor(getattr(_tokens(), self._dot, _tokens().accent)))
            p.drawEllipse(c, 4.0, 4.0)
        finally:
            p.end()


# Dot colour (a token name) per kind of run.
RUN_DOTS = {
    "Indexer run": "accent_pressed",
    "Duplicate cleanup": "warning",
    "Library Sync run": "accent",
    "Tag changes": "success",
    "File cleanup": "text_secondary",
}
MAX_RUN_ROWS = 6


class _AlbumTile(QtWidgets.QAbstractButton):
    """An album tile: the cover (radius 20, inset ring), title and artist.

    Placeholder (``album is None``): a faint well with two text bars, shown
    while the scan runs. Hovering fades in a play disc (painter only)."""

    RADIUS = 20
    TEXT_GAP = 10

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("homeAlbumTile")
        self.album: AlbumPick | None = None
        self._pixmap: QtGui.QPixmap | None = None
        self._scaled: tuple[tuple[int, float], QtGui.QPixmap] | None = None
        self._hover = 0.0
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.set_album(None)

    def set_album(self, album: AlbumPick | None) -> None:
        self.album = album
        self._pixmap = None
        self._scaled = None
        if album is None:
            self.setEnabled(False)
            self.setToolTip("")
            self.setAccessibleName("Loading an album")
            self.unsetCursor()
        else:
            self.setEnabled(True)
            n = len(album.tracks)
            by = f" by {album.artist}" if album.artist else ""
            self.setToolTip(f"Play {album.title}{by} ({n} track{'s' if n != 1 else ''})")
            self.setAccessibleName(f"Play {album.title}{by}")
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.update()

    @staticmethod
    def text_height() -> int:
        from gui.themes.glass import font
        return (QtGui.QFontMetrics(font("body", weight=600)).height() + 2
                + QtGui.QFontMetrics(font("caption")).height())

    def height_for(self, width: float) -> int:
        return int(math.ceil(width)) + self.TEXT_GAP + self.text_height()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(150, self.height_for(150))

    def cover_rect(self) -> QtCore.QRectF:
        side = float(min(self.width(), max(0, self.height() - self.TEXT_GAP - self.text_height())))
        return QtCore.QRectF(0.0, 0.0, side, side)

    def _cover(self, size: int, dpr: float) -> QtGui.QPixmap | None:
        album = self.album
        if album is None or album.image is None or album.image.isNull():
            return None
        if self._scaled is not None and self._scaled[0] == (size, dpr):
            return self._scaled[1]
        if self._pixmap is None:
            self._pixmap = QtGui.QPixmap.fromImage(album.image)
        px = max(1, int(round(size * dpr)))
        scaled = self._pixmap.scaled(px, px, QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                     QtCore.Qt.TransformationMode.SmoothTransformation)
        scaled.setDevicePixelRatio(dpr)
        self._scaled = ((size, dpr), scaled)
        return scaled

    def _set_hover(self, value: float) -> None:
        self._hover = float(value)
        self.update()

    def event(self, event: QtCore.QEvent) -> bool:  # noqa: D401
        kind = event.type()
        if kind in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            from gui.themes import motion
            target = 1.0 if kind == QtCore.QEvent.Type.HoverEnter and self.album is not None else 0.0
            motion.animate_value(self, "hover", self._hover, target, self._set_hover,
                                 duration="fast", easing="standard")
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, font, paint_surface, qcolor
        from gui.themes.tokens import Surface
        from gui.widgets.glyphs import paint_glyph

        t = _tokens()
        m = current_material()
        cover = self.cover_rect()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            path = QtGui.QPainterPath()
            path.addRoundedRect(cover, self.RADIUS, self.RADIUS)
            pm = self._cover(int(cover.width()), self.devicePixelRatioF())
            if pm is not None:
                p.save()
                p.setClipPath(path)
                w, h = pm.width() / pm.devicePixelRatio(), pm.height() / pm.devicePixelRatio()
                p.drawPixmap(QtCore.QPointF(cover.x() - (w - cover.width()) / 2,
                                            cover.y() - (h - cover.height()) / 2), pm)
                p.restore()
            else:
                paint_surface(p, cover, m.well, radius=self.RADIUS)
                if self.album is not None:
                    paint_glyph(p, cover.adjusted(cover.width() * 0.36, cover.height() * 0.36,
                                                  -cover.width() * 0.36, -cover.height() * 0.36),
                                "music", qcolor(t.text_secondary))
            # The inset ring (the mockup's ``inset 0 0 0 1px rgba(255,255,255,.14)``).
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.setPen(QtGui.QPen(qcolor(m.card.border or m.hairline), 1.0))
            p.drawRoundedRect(cover.adjusted(0.5, 0.5, -0.5, -0.5), self.RADIUS - 0.5, self.RADIUS - 0.5)

            if self._hover > 0.0 and self.album is not None:
                p.save()
                p.setOpacity(self._hover)
                p.fillPath(path, qcolor(m.pressed))
                d = min(46.0, cover.width() * 0.34)
                disc = QtCore.QRectF(cover.center().x() - d / 2, cover.center().y() - d / 2, d, d)
                paint_surface(p, disc, Surface(m.knob, radius=999), opacity=0.92)
                paint_glyph(p, disc.adjusted(d * 0.26, d * 0.26, -d * 0.24, -d * 0.26), "play",
                            qcolor(t.text_inverse))
                p.restore()
            if self.hasFocus():
                p.setPen(QtGui.QPen(qcolor(t.accent), 2.0))
                p.drawRoundedRect(cover.adjusted(1, 1, -1, -1), self.RADIUS - 1, self.RADIUS - 1)

            top = cover.bottom() + self.TEXT_GAP
            width = int(cover.width())
            tf, cf = font("body", weight=600), font("caption")
            tfm, cfm = QtGui.QFontMetrics(tf), QtGui.QFontMetrics(cf)
            if self.album is None:
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(qcolor(m.hover))
                p.drawRoundedRect(QtCore.QRectF(0, top + 3, width * 0.7, tfm.height() - 6), 4, 4)
                p.drawRoundedRect(QtCore.QRectF(0, top + tfm.height() + 2 + 2, width * 0.45,
                                                cfm.height() - 4), 4, 4)
            else:
                p.setFont(tf)
                p.setPen(qcolor(t.text_primary))
                p.drawText(QtCore.QRectF(0, top, width, tfm.height()),
                           int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                           tfm.elidedText(self.album.title, QtCore.Qt.TextElideMode.ElideRight, width))
                p.setFont(cf)
                p.setPen(qcolor(t.text_secondary))
                p.drawText(QtCore.QRectF(0, top + tfm.height() + 2, width, cfm.height()),
                           int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                           cfm.elidedText(self.album.artist, QtCore.Qt.TextElideMode.ElideRight, width))
        finally:
            p.end()


class _AlbumGrid(_ReflowGrid):
    """The six album tiles: one row of six, or two rows of three when narrow."""

    album_clicked = Signal(object)   # AlbumPick

    EQUAL_WIDTHS = True
    GAP = 16
    MIN_TILE = 80    # six across down to the minimum window (900 px)

    def __init__(self, count: int = SPIN_COUNT, parent: QtWidgets.QWidget | None = None) -> None:
        tiles = [_AlbumTile() for _ in range(count)]
        super().__init__(tiles, min_item=self.MIN_TILE, max_cols=count, gap=self.GAP, parent=parent)
        self._shadow_host: QtWidgets.QWidget | None = None
        for tile in tiles:
            tile.clicked.connect(self._on_tile_clicked)

    def tiles(self) -> list[_AlbumTile]:
        return list(self._items)  # type: ignore[return-value]

    def columns_for(self, width: int) -> int:
        n = len(self._items)
        if (width - (n - 1) * self.GAP) / n >= self.MIN_TILE:
            return n
        return max(1, n // 2)

    def _row_height(self, item_w: float) -> int:
        return self._items[0].height_for(item_w)   # type: ignore[attr-defined]

    def set_albums(self, albums: list[AlbumPick], placeholders: int = 0) -> None:
        """Show ``albums``, then up to ``placeholders`` loading tiles."""
        for i, tile in enumerate(self.tiles()):
            if i < len(albums):
                tile.set_album(albums[i])
                tile.show()
            elif i < max(len(albums), placeholders):
                tile.set_album(None)
                tile.show()
            else:
                tile.set_album(None)
                tile.hide()
        self.relayout()

    def set_shadow_host(self, host: QtWidgets.QWidget) -> None:
        self._shadow_host = host

    def _place(self) -> None:
        super()._place()
        if self._shadow_host is not None:
            self._shadow_host.update()

    @Slot()
    def _on_tile_clicked(self) -> None:
        tile = self.sender()
        album = getattr(tile, "album", None)
        if album is not None:
            self.album_clicked.emit(album)


class _SpinCard(GlassCard):
    """The hero card: a ``GlassCard`` that also paints its album tiles'
    shadows (``0 14 30 rgba(0,0,0,.40)``) on its glass, beneath them (a tile
    cannot paint outside itself, and its own grid would clip the shadow)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("headerCard")
        self.grid: _AlbumGrid | None = None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        grid = self.grid
        if grid is None or not grid.isVisible():
            return
        from gui.themes.glass import current_material, paint_shadow
        from gui.themes.tokens import Surface

        m = current_material()
        if m.is_glass:
            spec = Surface("", shadow="rgba(0,0,0,0.40)", blur=30, offset_y=14,
                           radius=_AlbumTile.RADIUS)
        else:
            spec = Surface("", shadow=m.card.shadow, blur=20, offset_y=8, radius=_AlbumTile.RADIUS)
        p = QtGui.QPainter(self)
        try:
            p.setClipRect(self.rect().adjusted(1, 1, -1, -1))
            for tile in grid.visible_items():
                r = tile.cover_rect()   # type: ignore[attr-defined]
                origin = tile.mapTo(self, QtCore.QPoint(0, 0))
                paint_shadow(p, r.translated(origin.x(), origin.y()), spec)
        finally:
            p.end()


# ── The page ──────────────────────────────────────────────────────────────────

class HomeWorkspace(WorkspaceBase):
    RESCAN_AFTER_S = 3.0   # a re-shown Home re-reads Docs/ if older than this
    # Two card columns from this content width (about a 1200 px window).
    TWO_COLUMNS_AT = 860
    _HYSTERESIS = 24

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._generation = 0
        self._worker: HomeScanWorker | None = None
        self._health: LibraryHealth | None = None
        self._scanned_at: float | None = None
        self._stats: tuple[int, float, int, int | None] | None = None
        # Albums: per library (normalised path) once a scan has finished.
        self._album_generation = 0
        self._album_worker: HomeAlbumScan | None = None
        self._album_pools: dict[str, list[AlbumPick]] = {}
        self._scanning_pool: list[AlbumPick] = []
        self._shown: list[AlbumPick] = []
        self._seen: set[tuple[str, str]] = set()
        self._columns = 0
        self._build_ui()
        self._inner.installEventFilter(self)
        self._on_library_changed(library_path)

    # ── UI ────────────────────────────────────────────────────────────────

    def _card(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout]:
        card = self._make_card()
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(14)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(self._make_card_title(title))
        head.addStretch(1)
        lay.addLayout(head)
        return card, lay, head

    @staticmethod
    def _muted(text: str = "", role: str = "label") -> QtWidgets.QLabel:
        from gui.themes.glass import apply_font

        label = QtWidgets.QLabel(text)
        label.setObjectName("mutedLabel")
        label.setWordWrap(True)
        apply_font(label, role)
        return label

    def _build_ui(self) -> None:
        from gui.themes.glass import apply_font
        from gui.themes.tokens import SPACE
        from gui.widgets.glass_parts import EmptyState, StatRow, StatTile
        from gui.widgets.glyphs import set_glyph

        cl = self.content_layout
        cl.addWidget(self._make_subtitle("Your library at a glance, and something to play."))

        # ── Spin something (the hero card) ────────────────────────────────
        spin = _SpinCard()
        self._cards.append(spin)
        self._inner.add_elevated(spin)
        self._spin_card = spin
        sl = QtWidgets.QVBoxLayout(spin)
        sl.setContentsMargins(16, 16, 16, 16)
        sl.setSpacing(16)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(12)
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(4)
        title = QtWidgets.QLabel("Spin something")
        title.setObjectName("homeHeroTitle")
        apply_font(title, "title", weight=650)
        titles.addWidget(title)
        self._spin_hint = self._muted("Six random albums from your library. Click one to play it.")
        titles.addWidget(self._spin_hint)
        head.addLayout(titles, 1)
        self._shuffle_btn = QtWidgets.QPushButton("Shuffle again")
        self._shuffle_btn.setObjectName("homeShuffleBtn")
        self._shuffle_btn.setToolTip("Show six other albums")
        set_glyph(self._shuffle_btn, "shuffle")
        self._shuffle_btn.clicked.connect(self.shuffle_albums)
        head.addWidget(self._shuffle_btn, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        sl.addLayout(head)
        self._album_grid = _AlbumGrid(SPIN_COUNT)
        self._album_grid.set_shadow_host(spin)
        spin.grid = self._album_grid
        self._album_grid.album_clicked.connect(self._play_album)
        sl.addWidget(self._album_grid)
        self._spin_empty = EmptyState("No library chosen")
        sl.addWidget(self._spin_empty)
        cl.addWidget(spin)

        # ── Library health ────────────────────────────────────────────────
        card, lay, head = self._card("Library health")
        self._health_card = card
        self._scan_state = self._muted("", "caption")
        self._scan_state.setWordWrap(False)
        head.addWidget(self._scan_state)
        self._refresh_btn = QtWidgets.QPushButton()
        self._refresh_btn.setObjectName("homeRefreshBtn")
        self._refresh_btn.setFlat(True)
        self._refresh_btn.setFixedSize(32, 32)
        self._refresh_btn.setToolTip("Check the library again")
        self._refresh_btn.setAccessibleName("Check the library again")
        set_glyph(self._refresh_btn, "refresh", role="muted")
        self._refresh_btn.clicked.connect(self.refresh)
        head.addWidget(self._refresh_btn)

        self._library_tiles: dict[str, StatTile] = {}
        tiles = []
        for key, label in (("tracks", "tracks"), ("size", "on disk"),
                           ("artists", "artists"), ("albums", "albums")):
            tile = StatTile("–", label)
            tile.setObjectName(f"stat_{key}")
            self._library_tiles[key] = tile
            tiles.append(tile)
        self._library_grid = _ReflowGrid(tiles, min_item=104, max_cols=4, even=True)
        lay.addWidget(self._library_grid)

        self._folder_tiles: dict[str, _FolderTile] = {}
        for name, why in ATTENTION_FOLDERS:
            ft = _FolderTile(name)
            ft.tile.setToolTip(f"{name}/: {why}")
            ft.open_btn.clicked.connect(self._on_open_clicked)
            self._folder_tiles[name] = ft
        self._folder_grid = _ReflowGrid(list(self._folder_tiles.values()), min_item=96, max_cols=3,
                                        min_cols=1)
        lay.addWidget(self._folder_grid)

        check = QtWidgets.QHBoxLayout()
        check.setSpacing(6)
        self._layout_caption = self._muted("Layout check:", "caption")
        self._layout_caption.setWordWrap(False)
        check.addWidget(self._layout_caption)
        self._layout_label = ElidedLabel("")
        self._layout_label.setObjectName("homeLayout")
        apply_font(self._layout_label, "caption", weight=650)
        check.addWidget(self._layout_label, 1)
        lay.addLayout(check)
        self._indexer_btn = QtWidgets.QPushButton("Organize with the Indexer")
        self._indexer_btn.setObjectName("homeIndexerBtn")
        self._indexer_btn.setToolTip("Indexer: preview, then sort the library into By Artist/ and By Year/")
        set_glyph(self._indexer_btn, "folder")
        self._indexer_btn.setProperty("navKey", "indexer")
        self._indexer_btn.clicked.connect(self._on_quick_action)
        self._indexer_btn.hide()
        lay.addWidget(self._indexer_btn, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        lay.addStretch(1)

        # ── Recent runs ───────────────────────────────────────────────────
        card, lay, _head = self._card("Recent runs")
        self._runs_card = card
        self._runs_box = QtWidgets.QVBoxLayout()
        self._runs_box.setSpacing(8)
        lay.addLayout(self._runs_box)
        self._runs_empty = EmptyState(
            "No runs yet",
            "Reports from the Indexer, Duplicates, Library Sync, the tag tools and "
            "File Cleanup appear here.")
        lay.addWidget(self._runs_empty)
        lay.addStretch(1)

        # ── Duplicates ────────────────────────────────────────────────────
        card, lay, _head = self._card("Duplicates")
        self._dupes_card = card
        self._dupes_text = self._muted("")
        lay.addWidget(self._dupes_text)
        self._dupes_stats = StatRow([("groups", "groups found"), ("review", "need review")])
        lay.addWidget(self._dupes_stats)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        self._find_dupes_btn = QtWidgets.QPushButton("Find duplicates")
        self._find_dupes_btn.setToolTip("Duplicates: fingerprint, review, quarantine")
        set_glyph(self._find_dupes_btn, "search")
        self._find_dupes_btn.setProperty("navKey", "duplicates/find")
        self._find_dupes_btn.clicked.connect(self._on_quick_action)
        row.addWidget(self._find_dupes_btn)
        self._dupes_open_btn = QtWidgets.QPushButton("Open preview")
        self._dupes_open_btn.setObjectName("homeDupesOpenBtn")
        set_glyph(self._dupes_open_btn, "external")
        self._dupes_open_btn.clicked.connect(self._on_open_clicked)
        self._dupes_open_btn.hide()
        row.addWidget(self._dupes_open_btn)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)

        # ── Quick actions ─────────────────────────────────────────────────
        card, lay, _head = self._card("Quick actions")
        self._actions_card = card
        self._action_buttons: dict[str, QtWidgets.QPushButton] = {
            "duplicates/find": self._find_dupes_btn,
        }
        buttons = []
        for text, key, tip, glyph in QUICK_ACTIONS:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName("homeActionBtn")
            btn.setToolTip(tip)
            btn.setProperty("navKey", key)
            set_glyph(btn, glyph)
            btn.clicked.connect(self._on_quick_action)
            self._action_buttons[key] = btn
            buttons.append(btn)
        self._actions_grid = _ReflowGrid(buttons, min_item=180, max_cols=3, gap=10, even=True)
        lay.addWidget(self._actions_grid)
        lay.addStretch(1)

        # ── The card grid: two columns when wide, one otherwise ───────────
        self._grid_host = QtWidgets.QWidget()
        self._grid_host.setObjectName("homeCardGrid")
        self._grid = QtWidgets.QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(SPACE.gutter)
        self._grid.setVerticalSpacing(SPACE.gutter)
        cl.addWidget(self._grid_host)
        cl.addStretch(1)
        self._set_columns(1)

    def _grid_cards(self) -> list[QtWidgets.QWidget]:
        return [self._health_card, self._runs_card, self._dupes_card, self._actions_card]

    def columns(self) -> int:
        return self._columns

    def _set_columns(self, columns: int) -> None:
        if columns == self._columns:
            return
        self._columns = columns
        for card in self._grid_cards():
            self._grid.removeWidget(card)
        for i, card in enumerate(self._grid_cards()):
            r, c = divmod(i, columns)
            self._grid.addWidget(card, r, c)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1 if columns == 2 else 0)

    def _reflow(self) -> None:
        from gui.themes.tokens import SPACE

        width = self._inner.width() - 2 * SPACE.gutter
        if self._columns == 2:
            want = 2 if width >= self.TWO_COLUMNS_AT - self._HYSTERESIS else 1
        else:
            want = 2 if width >= self.TWO_COLUMNS_AT else 1
        self._set_columns(want)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self._inner and event.type() == QtCore.QEvent.Type.Resize:
            self._reflow()
        return super().eventFilter(obj, event)

    @staticmethod
    def _clear(layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif item.layout() is not None:
                HomeWorkspace._clear(item.layout())
                item.layout().deleteLater()

    # ── Library ───────────────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        from gui.workers import job_registry

        job_registry().cancel_all(self.running_jobs())
        self._generation += 1
        self._worker = None
        self._health = None
        self._scanned_at = None
        self._stats = None
        self._album_generation += 1
        self._album_worker = None
        self._scanning_pool = []
        self._shown = []
        self._seen = set()
        self._render_stats()
        self._render()
        if path:
            self._start_scan()
            self._show_albums_or_scan()
        else:
            self._scan_state.setText("")
            self._render_albums()

    def set_library_stats(self, tracks: int, size_gb: float, artists: int) -> None:
        """The page header's numbers (the main window's stats scan), shown
        here too, with the album count the same walk found."""
        if not self._library_path:
            self._stats = None
        else:
            from gui.library_stats import last_counts
            counts = last_counts(self._library_path)
            albums = counts.albums if counts is not None and counts.tracks == tracks else None
            self._stats = (tracks, size_gb, artists, albums)
        self._render_stats()

    def library_stats_shown(self) -> dict[str, str]:
        """What the library tiles show now (tests, accessibility)."""
        return {k: t.value() for k, t in self._library_tiles.items() if not t.isHidden()}

    def _render_stats(self) -> None:
        from gui.library_stats import format_size

        stats = self._stats
        tiles = self._library_tiles
        if stats is None:
            for tile in tiles.values():
                tile.set_value("–")
            tiles["albums"].setVisible(True)
        else:
            tracks, size_gb, artists, albums = stats
            tiles["tracks"].set_value(f"{tracks:,}")
            tiles["tracks"].set_label("track" if tracks == 1 else "tracks")
            tiles["size"].set_value(format_size(size_gb))
            tiles["artists"].set_value(f"{artists:,}")
            tiles["artists"].set_label("artist" if artists == 1 else "artists")
            tiles["albums"].setVisible(albums is not None)
            if albums is not None:
                tiles["albums"].set_value(f"{albums:,}")
                tiles["albums"].set_label("album" if albums == 1 else "albums")
        self._library_grid.relayout()

    # ── Health scan ───────────────────────────────────────────────────────

    def is_scanning(self) -> bool:
        return self._worker is not None

    @Slot()
    def refresh(self) -> None:
        if self._library_path and not self.is_scanning():
            self._start_scan()

    def on_activated(self) -> None:
        """Shown again: other pages may have left new reports in Docs/."""
        import time
        if (
            self._library_path and not self.is_scanning()
            and (self._scanned_at is None or time.monotonic() - self._scanned_at > self.RESCAN_AFTER_S)
        ):
            self._start_scan()
        if self._library_path and self._album_worker is None and not self._shown \
                and self._pool() is None:
            self._show_albums_or_scan()

    def _start_scan(self) -> None:
        self._generation += 1
        worker = HomeScanWorker(self._library_path, self._generation)
        worker.health_ready.connect(self._on_health_ready)
        worker.finished.connect(self._on_scan_thread_done)
        self._worker = worker
        self._refresh_btn.setEnabled(False)
        self._scan_state.setText("Checking…")
        self._start_worker(worker, "Home: library health", ask_before_quit=False)

    @Slot()
    def _on_scan_thread_done(self) -> None:
        if self.sender() is self._worker:
            self._worker = None
            self._refresh_btn.setEnabled(True)

    @Slot(int, object)
    def _on_health_ready(self, generation: int, health: LibraryHealth) -> None:
        if generation != self._generation:
            return   # an older library's (or an older) scan
        import time
        self._health = health
        self._scanned_at = time.monotonic()
        self._scan_state.setText("Checked " + datetime.datetime.now().strftime("%H:%M"))
        self._render()

    def health(self) -> LibraryHealth | None:
        return self._health

    # ── Rendering: health, runs, duplicates ───────────────────────────────

    def layout_check(self) -> tuple[bool | None, str]:
        """(True: organised · False: not · None: unknown yet, the text shown)."""
        health = self._health
        if not self._library_path or health is None:
            return None, self._layout_label.text()
        return health.layout_ok, self._layout_label.text()

    def _set_layout_state(self, ok: bool | None, text: str, tip: str = "") -> None:
        t = _tokens()
        colour = {True: t.success, False: t.warning}.get(ok, t.text_secondary)  # type: ignore[arg-type]
        pal = self._layout_label.palette()
        pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(colour))
        self._layout_label.setPalette(pal)
        self._layout_label.setText(text)
        self._layout_label.setToolTip(tip or text)
        self._indexer_btn.setVisible(ok is False)
        self._layout_ok = ok

    def _on_theme_changed_base(self, tokens: object) -> None:
        super()._on_theme_changed_base(tokens)
        ok = getattr(self, "_layout_ok", None)
        if hasattr(self, "_layout_label"):
            self._set_layout_state(ok, self._layout_label.text(), self._layout_label.toolTip())

    def _render(self) -> None:
        health = self._health
        self._clear(self._runs_box)
        if not self._library_path:
            self._set_layout_state(None, "no library chosen")
        elif health is None:
            self._set_layout_state(None, "checking…")
        elif health.layout_ok:
            self._set_layout_state(True, "matches the Indexer's layout",
                                   "The folder has the Indexer's By Artist/ · By Year/ layout.")
        else:
            messages = health.layout_messages or []
            first = messages[0] if messages else ""
            if not first or first.startswith("This folder has not been organised"):
                text = "not organised by the Indexer yet"
            else:
                text = first
            self._set_layout_state(False, text, "\n".join(messages) or text)

        for name, why in ATTENTION_FOLDERS:
            ft = self._folder_tiles[name]
            count = None if health is None else health.folder_counts.get(name)
            folder = _top_level_dir(health.root, name) if health is not None and count is not None else None
            if health is None:
                ft.tile.set_value("–", tone="")
                ft.tile.setToolTip(f"{name}/: {why}")
            elif count is None:
                ft.tile.set_value("–", tone="")
                ft.tile.setToolTip(f"{name}/ has not been created yet ({why}).")
            else:
                tone = "warning" if count and name in (MANUAL_REVIEW, QUARANTINE) else ""
                ft.tile.set_value(f"{count:,}", tone=tone)
                noun = "track" if count == 1 else "tracks"
                ft.tile.setToolTip(f"{name}/: {count:,} {noun}, {why}.")
            ft.open_btn.setProperty("openPath", folder or "")
            ft.open_btn.setVisible(bool(folder))

        runs = health.recent_runs if health is not None else []
        for run in runs[:MAX_RUN_ROWS]:
            row = _RunRow(run, RUN_DOTS.get(run.kind, "accent"), self._on_open_clicked)
            self._runs_box.addWidget(row)
        if not self._library_path:
            self._runs_empty.set_text("No library chosen",
                                      "Choose your music folder with Change library (Ctrl+O).")
        elif health is None:
            self._runs_empty.set_text("Checking…", "")
        else:
            self._runs_empty.set_text(
                "No runs yet",
                "Reports from the Indexer, Duplicates, Library Sync, the tag tools and "
                "File Cleanup appear here.")
        self._runs_empty.setVisible(not runs)
        self._render_duplicates()

    def run_rows(self) -> list[_RunRow]:
        rows = []
        for i in range(self._runs_box.count()):
            w = self._runs_box.itemAt(i).widget()
            if isinstance(w, _RunRow):
                rows.append(w)
        return rows

    def _render_duplicates(self) -> None:
        health = self._health
        scan = health.duplicates if health is not None else None
        if scan is None:
            if self._library_path and health is not None:
                self._dupes_text.setText(
                    "No duplicate scan yet. Find duplicates fingerprints the library and "
                    "shows a preview before anything moves.")
            elif self._library_path:
                self._dupes_text.setText("Checking for the last duplicate scan…")
            else:
                self._dupes_text.setText("Find duplicates fingerprints a library and shows a "
                                         "preview before anything moves.")
            self._dupes_stats.hide()
            self._dupes_open_btn.hide()
            self._dupes_open_btn.setProperty("openPath", "")
            return
        when = relative_when(scan.scanned_at)
        if scan.cleaned_since:
            self._dupes_text.setText(f"From the last scan, {when}; a cleanup has run since.")
        else:
            self._dupes_text.setText(f"From the last scan, {when}. Review them before anything moves.")
        self._dupes_stats.tile("groups").set_value(f"{scan.groups:,}")
        self._dupes_stats.tile("groups").set_label("group found" if scan.groups == 1 else "groups found")
        self._dupes_stats.tile("review").set_value(
            f"{scan.review_required:,}", tone="warning" if scan.review_required else "")
        self._dupes_stats.show()
        path = scan.preview_html or scan.preview_json
        self._dupes_open_btn.setProperty("openPath", path)
        self._dupes_open_btn.setToolTip(f"Open {path}")
        self._dupes_open_btn.show()

    # ── Albums ────────────────────────────────────────────────────────────

    def _pool_key(self) -> str:
        return os.path.normcase(os.path.abspath(self._library_path)) if self._library_path else ""

    def _pool(self) -> list[AlbumPick] | None:
        return self._album_pools.get(self._pool_key())

    def album_pool(self) -> list[AlbumPick]:
        """Every album found for the current library so far."""
        pool = self._pool()
        return list(pool if pool is not None else self._scanning_pool)

    def shown_albums(self) -> list[AlbumPick]:
        return list(self._shown)

    def is_finding_albums(self) -> bool:
        return self._album_worker is not None

    def _show_albums_or_scan(self) -> None:
        pool = self._pool()
        if pool is not None:
            self._shown = []
            self._seen = set()
            self._pick_albums()
            return
        self._start_album_scan()

    def _start_album_scan(self) -> None:
        self._album_generation += 1
        self._scanning_pool = []
        self._shown = []
        worker = HomeAlbumScan(self._library_path, self._album_generation)
        worker.album_found.connect(self._on_album_found)
        worker.scan_done.connect(self._on_albums_done)
        worker.finished.connect(self._on_album_thread_done)
        self._album_worker = worker
        self._render_albums()
        self._start_worker(worker, "Home: albums to play", ask_before_quit=False)

    @Slot()
    def _on_album_thread_done(self) -> None:
        if self.sender() is self._album_worker:
            self._album_worker = None
            self._render_albums()

    @Slot(int, object)
    def _on_album_found(self, generation: int, album: AlbumPick) -> None:
        if generation != self._album_generation:
            return
        self._scanning_pool.append(album)
        if len(self._shown) < SPIN_COUNT:
            self._shown.append(album)
            self._seen.add(album.key)
            self._render_albums()

    @Slot(int, int)
    def _on_albums_done(self, generation: int, _found: int) -> None:
        if generation != self._album_generation:
            return
        pools = self._album_pools
        if len(pools) >= 4:
            pools.pop(next(iter(pools)))
        pools[self._pool_key()] = list(self._scanning_pool)
        self._scanning_pool = []
        self._album_worker = None
        self._render_albums()

    @Slot()
    def shuffle_albums(self) -> None:
        """Show six other albums (all of the pool is shown before any repeats)."""
        if self._pool() is None:
            return
        self._pick_albums()

    def _pick_albums(self) -> None:
        pool = self._pool() or []
        current = {a.key for a in self._shown}
        unseen = [a for a in pool if a.key not in self._seen]
        if len(unseen) < SPIN_COUNT:
            self._seen = set(current)
            unseen = [a for a in pool if a.key not in self._seen]
        random.shuffle(unseen)
        pick = unseen[:SPIN_COUNT]
        if len(pick) < SPIN_COUNT:   # fewer than twelve albums: fill from the rest
            rest = [a for a in pool if a not in pick]
            random.shuffle(rest)
            pick += rest[:SPIN_COUNT - len(pick)]
        self._shown = pick
        self._seen |= {a.key for a in pick}
        self._render_albums()

    def _render_albums(self) -> None:
        loading = self._album_worker is not None
        pool = self._pool()
        grid, empty = self._album_grid, self._spin_empty
        if not self._library_path:
            state = "none"
        elif self._shown or loading:
            state = "albums"
        else:
            state = "empty"
        if state == "albums":
            grid.set_albums(self._shown, SPIN_COUNT if loading else 0)
        grid.setVisible(state == "albums")
        empty.setVisible(state != "albums")
        if state == "none":
            empty.set_text("No library chosen",
                           "Choose your music folder with Change library (Ctrl+O); "
                           "albums from it appear here.")
        elif state == "empty":
            empty.set_text("No albums to play yet",
                           "Albums appear here once their folders hold tracks and cover art "
                           "(a cover.jpg, or art embedded in the tracks).")
        total = len(pool) if pool is not None else 0
        can_shuffle = pool is not None and total > len(self._shown)
        self._shuffle_btn.setEnabled(can_shuffle)
        if loading:
            self._shuffle_btn.setToolTip("Finding albums…")
        elif pool is not None and not can_shuffle and total:
            self._shuffle_btn.setToolTip(
                f"Every album found ({total}) is already showing.")
        else:
            self._shuffle_btn.setToolTip("Show six other albums")
        self._spin_card.update()

    @Slot(object)
    def _play_album(self, album: AlbumPick) -> None:
        if album is not None and album.tracks:
            self.play_tracks_requested.emit(list(album.tracks), album.label)

    # ── Actions ───────────────────────────────────────────────────────────

    @Slot()
    def _on_open_clicked(self) -> None:
        btn = self.sender()
        path = btn.property("openPath") if btn is not None else None
        if path and not open_path(str(path)):
            QtWidgets.QMessageBox.information(self, "Open", f"Could not open:\n{path}")

    @Slot()
    def _on_quick_action(self) -> None:
        btn = self.sender()
        key = btn.property("navKey") if btn is not None else None
        if key:
            self.navigate_requested.emit(str(key))

    def shutdown(self) -> None:
        super().shutdown()
