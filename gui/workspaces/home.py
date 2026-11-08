"""Home page: the library at a glance.

* the library path and its statistics (the main window's existing stats scan
  hands its numbers over — Home never walks the whole library itself);
* **Library health**: whether the folder has the Indexer's layout (the
  library Validator, formerly a Utilities tile) and how many tracks wait in
  ``Manual Review/``, ``Quarantine/`` and ``Not Sorted/``;
* **Latest runs**: the newest report or log each workflow left in ``Docs/``
  (Indexer execute log, duplicate execution report, Library Sync report,
  tag change logs, file cleanup log), each with "Open";
* **Quick actions** that navigate to the page that does the job.

Everything here is read-only and cheap: ``scan_library_health`` only lists
``Docs/`` and walks the three reserved folders above, in a read-only
JobRegistry job (``ask_before_quit=False``), so it never blocks the GUI and
quitting cancels it silently.
"""
from __future__ import annotations

import datetime
import glob
import os
from dataclasses import dataclass, field

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase
from utils.reserved_folders import DOCS, MANUAL_REVIEW, NOT_SORTED, QUARANTINE

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


@dataclass
class RecentRun:
    kind: str          # "Indexer run", "Tag changes", …
    path: str
    mtime: float
    detail: str = ""   # e.g. the tool that wrote a tag change log

    @property
    def when(self) -> str:
        return datetime.datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")


@dataclass
class LibraryHealth:
    root: str
    layout_ok: bool = False
    layout_messages: list[str] = field(default_factory=list)
    # folder name → audio files in it (None: the folder does not exist)
    folder_counts: dict[str, int | None] = field(default_factory=dict)
    recent_runs: list[RecentRun] = field(default_factory=list)


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
    """``tag_changes_<tool>_<stamp>.jsonl`` → ``<tool>``."""
    stem = os.path.basename(path)[len("tag_changes_"):]
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


# Quick actions: (button label, navigation key, tooltip)
QUICK_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("Organize folders", "indexer", "Indexer: preview, then normalize folders and names"),
    ("Merge new music", "library_sync", "Library Sync: bring an incoming folder in"),
    ("Clean up file names", "file_cleanup", "File Cleanup: remove ' (1)' / ' - Copy' suffixes"),
    ("Find duplicates", "duplicates/find", "Duplicates: fingerprint, review, quarantine"),
    ("Fix tags", "metadata/tag_fixer", "Metadata: look tracks up and fix their tags"),
    ("Explore the Sound Map", "sound_map", "Sound Map: analyse how tracks sound, then explore"),
    ("Make a playlist", "playlists", "Playlists: rule-based playlist creation"),
)


class HomeWorkspace(WorkspaceBase):
    RESCAN_AFTER_S = 3.0   # a re-shown Home re-reads Docs/ if older than this

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._generation = 0
        self._worker: HomeScanWorker | None = None
        self._health: LibraryHealth | None = None
        self._scanned_at: float | None = None
        self._build_ui()
        self._on_library_changed(library_path)

    # ── UI ────────────────────────────────────────────────────────────────

    def _card(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
        card = self._make_card()
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        lay.addWidget(self._make_card_title(title))
        return card, lay

    @staticmethod
    def _hint(text: str = "") -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("statusHint")
        label.setWordWrap(True)
        return label

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_section_title("Home"))
        cl.addWidget(self._make_subtitle(
            "Your library at a glance: what is in it, what needs attention, and what "
            "AlphaDEX did to it most recently."
        ))

        # Library
        card, lay = self._card("Library")
        self._path_label = QtWidgets.QLabel("No library selected")
        self._path_label.setObjectName("homeLibraryPath")
        self._path_label.setWordWrap(True)
        self._path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self._path_label)
        self._stats_label = self._hint("")
        lay.addWidget(self._stats_label)
        row = QtWidgets.QHBoxLayout()
        self._scan_state = self._hint("")
        row.addWidget(self._scan_state, 1)
        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip("Check the library again")
        self._refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self._refresh_btn)
        lay.addLayout(row)
        cl.addWidget(card)

        # Library health
        card, lay = self._card("Library health")
        self._layout_label = QtWidgets.QLabel("")
        self._layout_label.setObjectName("homeLayout")
        self._layout_label.setWordWrap(True)
        lay.addWidget(self._layout_label)
        self._folders_box = QtWidgets.QVBoxLayout()
        self._folders_box.setSpacing(6)
        lay.addLayout(self._folders_box)
        self._health_card = card
        cl.addWidget(card)

        # Latest runs
        card, lay = self._card("Latest runs")
        self._runs_box = QtWidgets.QVBoxLayout()
        self._runs_box.setSpacing(6)
        lay.addLayout(self._runs_box)
        cl.addWidget(card)

        # Quick actions
        card, lay = self._card("Quick actions")
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        self._action_buttons: dict[str, QtWidgets.QPushButton] = {}
        for i, (text, key, tip) in enumerate(QUICK_ACTIONS):
            btn = QtWidgets.QPushButton(text)
            btn.setToolTip(tip)
            btn.setProperty("navKey", key)
            btn.clicked.connect(self._on_quick_action)
            grid.addWidget(btn, i // 4, i % 4)
            self._action_buttons[key] = btn
        lay.addLayout(grid)
        cl.addWidget(card)
        cl.addStretch(1)

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

    def _row(self, layout: QtWidgets.QVBoxLayout, text: str, path: str | None,
             button: str = "Open", tip: str = "") -> QtWidgets.QPushButton | None:
        row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        row.addWidget(label, 1)
        btn = None
        if path:
            btn = QtWidgets.QPushButton(button)
            btn.setToolTip(tip or path)
            btn.setProperty("openPath", path)
            btn.clicked.connect(self._on_open_clicked)
            row.addWidget(btn)
        layout.addLayout(row)
        return btn

    # ── Library ───────────────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        from gui.workers import job_registry

        job_registry().cancel_all(self.running_jobs())
        self._generation += 1
        self._worker = None
        self._health = None
        self._scanned_at = None
        self._stats_label.setText("")
        self._render()
        if path:
            self._path_label.setText(path)
            self._start_scan()
        else:
            self._path_label.setText("No library selected")
            self._scan_state.setText("Choose a library with Change Library (Ctrl+O).")

    def set_library_stats(self, tracks: int, size_gb: float, artists: int) -> None:
        """The top bar's numbers (the main window's stats scan), shown here too."""
        if not self._library_path:
            self._stats_label.setText("")
            return
        if tracks == 0:
            self._stats_label.setText("No audio files found outside the reserved folders.")
            return
        unit = "artist" if artists == 1 else "artists"
        self._stats_label.setText(f"{tracks:,} tracks  ·  {size_gb:.1f} GB  ·  {artists:,} {unit}")

    # ── Scanning ──────────────────────────────────────────────────────────

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

    def _start_scan(self) -> None:
        self._generation += 1
        worker = HomeScanWorker(self._library_path, self._generation)
        worker.health_ready.connect(self._on_health_ready)
        worker.finished.connect(self._on_scan_thread_done)
        self._worker = worker
        self._refresh_btn.setEnabled(False)
        self._scan_state.setText("Checking the library…")
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
        self._scan_state.setText(
            "Checked at " + datetime.datetime.now().strftime("%H:%M") + "."
        )
        self._render()

    def health(self) -> LibraryHealth | None:
        return self._health

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render(self) -> None:
        self._clear(self._folders_box)
        self._clear(self._runs_box)
        health = self._health
        if not self._library_path:
            self._layout_label.setText("No library selected.")
            self._runs_box.addWidget(self._hint("No library selected."))
            return
        if health is None:
            self._layout_label.setText("Checking…")
            self._runs_box.addWidget(self._hint("Checking…"))
            return

        if health.layout_ok:
            self._layout_label.setText(
                "✓ Organized: the folder has the Indexer's By Artist/ · By Year/ layout."
            )
        else:
            text = "✗ " + "\n".join(health.layout_messages or ["Not organized by the Indexer yet."])
            self._layout_label.setText(text)

        for name, why in ATTENTION_FOLDERS:
            count = health.folder_counts.get(name)
            if count is None:
                self._row(self._folders_box, f"{name}/ — not created yet", None)
                continue
            noun = "track" if count == 1 else "tracks"
            folder = _top_level_dir(health.root, name)
            self._row(self._folders_box, f"{name}/ — {count:,} {noun} ({why})",
                      folder, "Open folder")

        if not health.recent_runs:
            self._runs_box.addWidget(self._hint(
                "No runs yet. Reports from the Indexer, Duplicates, Library Sync, the tag "
                "tools and File Cleanup appear here (they are written to Docs/)."
            ))
            return
        for run in health.recent_runs:
            kind = f"{run.kind} ({run.detail})" if run.detail else run.kind
            self._row(self._runs_box, f"{kind} — {run.when} — {os.path.basename(run.path)}",
                      run.path)

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
