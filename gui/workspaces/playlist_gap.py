"""Playlist Gap workspace — "which songs from this playlist do I not own?"

Phase 1: the walking skeleton. A tile strip that reports live state and doubles
as the step navigation, over three panes — Setup, Triage, Export list. The tiles
for Verify and Summary are present but disabled; they arrive in Phase 2.5 and 3.

UI only. Every decision of substance lives in the Qt-free backend
(``playlist_gap_*`` and ``controllers/playlist_gap_controller``), reached through
``QThread`` workers that mirror ``library_sync``'s pattern: nothing here touches a
widget from a worker, and results travel back in the ``finished`` payload.

See ``docs/playlist_gap_spec.md`` §11.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase

STEPS = (
    ("setup", "Setup", True),
    ("triage", "Triage", True),
    ("export", "Export list", True),
    ("verify", "Verify", False),
    ("summary", "Summary", False),
)

BUCKET_ORDER = ("missing", "unsure", "owned", "unavailable", "ignored")
BUCKET_LABELS = {
    "missing": "Missing — go download",
    "unsure": "Needs confirmation",
    "owned": "Already owned",
    "unavailable": "Unavailable upstream",
    "ignored": "Never want",
}


# ── Workers ───────────────────────────────────────────────────────────────────

class GapRunWorker(QtCore.QThread):
    """Read the source, snapshot the library, and match — off the main thread."""

    progress = Signal(int, str)              # percent, message
    log_line = Signal(str)
    finished = Signal(bool, str, object)     # success, message, RunResult | None

    def __init__(self, spec, library_root: str, cache_db: Optional[str] = None) -> None:
        super().__init__()
        self._spec = spec
        self._library_root = library_root
        self._cache_db = cache_db
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            from controllers.playlist_gap_controller import run_source

            def on_progress(done: int, total: int, message: str) -> None:
                percent = int(done / total * 100) if total else 0
                self.progress.emit(min(99, percent), message)

            result = run_source(
                self._spec,
                self._library_root,
                cache_db=self._cache_db,
                progress=on_progress,
                log=self.log_line.emit,
                should_cancel=self._cancel.is_set,
            )
            if self._cancel.is_set():
                self.finished.emit(False, "Cancelled.", None)
                return
            self.progress.emit(100, "Done")
            counts = result.counts
            self.finished.emit(
                True,
                f"{counts.get('missing', 0)} to download · "
                f"{counts.get('unsure', 0)} to confirm · "
                f"{counts.get('owned', 0)} already owned",
                result,
            )
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.finished.emit(False, f"{type(exc).__name__}: {exc}", None)


class GapExportWorker(QtCore.QThread):
    """Write the download list."""

    finished = Signal(bool, str, str)        # success, message, path

    def __init__(self, results, path: str, group_by: str, as_csv: bool) -> None:
        super().__init__()
        self._results = results
        self._path = path
        self._group_by = group_by
        self._as_csv = as_csv

    def run(self) -> None:
        try:
            from playlist_gap_report import write_csv, write_text

            if self._as_csv:
                path = write_csv(self._path, self._results)
            else:
                path = write_text(self._path, self._results, group_by=self._group_by)
            self.finished.emit(True, f"Saved {os.path.basename(path)}", path)
        except Exception as exc:
            self.finished.emit(False, f"{type(exc).__name__}: {exc}", "")


# ── Tile strip ────────────────────────────────────────────────────────────────

class StepTile(QtWidgets.QPushButton):
    """One step: a live status readout that is also the navigation."""

    def __init__(self, key: str, title: str, enabled: bool) -> None:
        super().__init__()
        self.key = key
        self.setCheckable(True)
        self.setEnabled(enabled)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(62)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(11, 7, 11, 7)
        layout.setSpacing(1)
        self._title = QtWidgets.QLabel(title)
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        self._line1 = QtWidgets.QLabel("—")
        self._line2 = QtWidgets.QLabel("")
        for label in (self._line1, self._line2):
            label.setObjectName("tileMetric")
            small = label.font()
            small.setPointSizeF(max(7.5, small.pointSizeF() - 1.5))
            label.setFont(small)
        layout.addWidget(self._title)
        layout.addWidget(self._line1)
        layout.addWidget(self._line2)
        if not enabled:
            self.setToolTip("Arrives in a later phase.")

    def set_metrics(self, line1: str, line2: str = "") -> None:
        self._line1.setText(line1)
        self._line2.setText(line2)


# ── Workspace ─────────────────────────────────────────────────────────────────

class PlaylistGapWorkspace(WorkspaceBase):
    """Compare a wanted-song list against the library."""

    def __init__(self, library_path: str = "", parent=None) -> None:
        super().__init__(library_path, parent)
        self._run_worker: Optional[GapRunWorker] = None
        self._export_worker: Optional[GapExportWorker] = None
        self._result = None
        self._tiles: Dict[str, StepTile] = {}
        self._build()
        self._refresh_tiles()

    # ── construction ─────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = self.content_layout
        layout.addWidget(self._make_section_title("Playlist Gap"))
        subtitle = self._make_subtitle(
            "Import a list of songs you want, compare it against your library, and get "
            "back a short list of what you still need to download."
        )
        layout.addWidget(subtitle)

        strip = QtWidgets.QHBoxLayout()
        strip.setSpacing(6)
        for key, title, enabled in STEPS:
            tile = StepTile(key, title, enabled)
            tile.clicked.connect(lambda _checked=False, k=key: self._show(k))
            self._tiles[key] = tile
            strip.addWidget(tile)
        layout.addLayout(strip)

        self._stack = QtWidgets.QStackedWidget()
        self._panes: Dict[str, int] = {}
        for key, builder in (
            ("setup", self._build_setup),
            ("triage", self._build_triage),
            ("export", self._build_export),
        ):
            self._panes[key] = self._stack.addWidget(builder())
        layout.addWidget(self._stack, stretch=1)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._show("setup")

    def _build_setup(self) -> QtWidgets.QWidget:
        card = self._make_card()
        box = QtWidgets.QVBoxLayout(card)
        box.setContentsMargins(16, 14, 16, 14)
        box.setSpacing(9)
        box.addWidget(self._make_card_title("Wanted list"))

        note = QtWidgets.QLabel(
            "Phase 1 reads CSV exports — a TuneMyMusic export of a YouTube Music "
            "playlist, for example. The importer is pluggable, so reading YouTube "
            "directly is a later drop-in, not a rewrite."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedText")
        box.addWidget(note)

        row = QtWidgets.QHBoxLayout()
        self._csv_entry, browse = self._make_browse_row("CSV", "Path to a CSV export…")
        browse.clicked.connect(self._pick_csv)
        row.addWidget(QtWidgets.QLabel("CSV file"))
        row.addWidget(self._csv_entry, stretch=1)
        row.addWidget(browse)
        box.addLayout(row)

        name_row = QtWidgets.QHBoxLayout()
        self._name_entry = QtWidgets.QLineEdit()
        self._name_entry.setPlaceholderText("A name for this list, e.g. Liked videos")
        name_row.addWidget(QtWidgets.QLabel("Name"))
        name_row.addWidget(self._name_entry, stretch=1)
        box.addLayout(name_row)

        box.addWidget(self._make_card_title("Library"))
        lib_note = QtWidgets.QLabel(
            "Not Sorted, Quarantine and Manual Review are counted as owned — they hold "
            "music you already have, so skipping them would report it as missing."
        )
        lib_note.setWordWrap(True)
        lib_note.setObjectName("mutedText")
        box.addWidget(lib_note)

        lib_row = QtWidgets.QHBoxLayout()
        self._lib_entry = QtWidgets.QLineEdit(self._library_path)
        self._lib_entry.setPlaceholderText("Library folder…")
        lib_browse = QtWidgets.QPushButton("Browse…")
        lib_browse.setFixedWidth(80)
        lib_browse.clicked.connect(self._pick_library)
        lib_row.addWidget(QtWidgets.QLabel("Folder"))
        lib_row.addWidget(self._lib_entry, stretch=1)
        lib_row.addWidget(lib_browse)
        box.addLayout(lib_row)

        actions = QtWidgets.QHBoxLayout()
        self._run_btn = self._make_primary_button("Compare against my library")
        self._run_btn.clicked.connect(self._start_run)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_run)
        actions.addWidget(self._run_btn)
        actions.addWidget(self._cancel_btn)
        actions.addStretch(1)
        box.addLayout(actions)

        self._fields_label = QtWidgets.QLabel("")
        self._fields_label.setWordWrap(True)
        self._fields_label.setObjectName("mutedText")
        box.addWidget(self._fields_label)
        box.addStretch(1)
        return card

    def _build_triage(self) -> QtWidgets.QWidget:
        card = self._make_card()
        box = QtWidgets.QVBoxLayout(card)
        box.setContentsMargins(16, 14, 16, 14)
        box.setSpacing(9)
        box.addWidget(self._make_card_title("Results"))

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Show"))
        self._bucket_combo = QtWidgets.QComboBox()
        self._bucket_combo.addItem("Everything", "")
        for key in BUCKET_ORDER:
            self._bucket_combo.addItem(BUCKET_LABELS[key], key)
        self._bucket_combo.setCurrentIndex(1)
        self._bucket_combo.currentIndexChanged.connect(lambda _i: self._fill_table())
        filter_row.addWidget(self._bucket_combo)
        filter_row.addStretch(1)
        box.addLayout(filter_row)

        self._table = QtWidgets.QTreeWidget()
        self._table.setHeaderLabels(["Wanted", "Verdict", "Why", "Best match"])
        self._table.setRootIsDecorated(False)
        self._table.setAlternatingRowColors(True)
        self._table.setColumnWidth(0, 300)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(2, 320)
        box.addWidget(self._table, stretch=1)

        self._triage_note = QtWidgets.QLabel(
            "Per-row overrides, the evidence panel and bulk actions arrive in Phase 2."
        )
        self._triage_note.setObjectName("mutedText")
        box.addWidget(self._triage_note)
        return card

    def _build_export(self) -> QtWidgets.QWidget:
        card = self._make_card()
        box = QtWidgets.QVBoxLayout(card)
        box.setContentsMargins(16, 14, 16, 14)
        box.setSpacing(9)
        box.addWidget(self._make_card_title("Download list"))

        options = QtWidgets.QHBoxLayout()
        options.addWidget(QtWidgets.QLabel("Group by"))
        self._group_combo = QtWidgets.QComboBox()
        self._group_combo.addItem("Flat list", "flat")
        self._group_combo.addItem("Album", "album")
        self._group_combo.addItem("Artist", "artist")
        self._group_combo.currentIndexChanged.connect(lambda _i: self._refresh_preview())
        options.addWidget(self._group_combo)
        options.addStretch(1)
        box.addLayout(options)

        self._preview = QtWidgets.QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setPlaceholderText("Run a comparison to see the list.")
        box.addWidget(self._preview, stretch=1)

        actions = QtWidgets.QHBoxLayout()
        self._copy_btn = QtWidgets.QPushButton("Copy to clipboard")
        self._copy_btn.clicked.connect(self._copy_list)
        self._save_txt_btn = QtWidgets.QPushButton("Save as .txt")
        self._save_txt_btn.clicked.connect(lambda: self._save(as_csv=False))
        self._save_csv_btn = QtWidgets.QPushButton("Save as .csv")
        self._save_csv_btn.clicked.connect(lambda: self._save(as_csv=True))
        for button in (self._copy_btn, self._save_txt_btn, self._save_csv_btn):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addStretch(1)
        box.addLayout(actions)
        return card

    # ── navigation ───────────────────────────────────────────────────────

    def _show(self, key: str) -> None:
        if key not in self._panes:
            return
        self._stack.setCurrentIndex(self._panes[key])
        for tile_key, tile in self._tiles.items():
            tile.setChecked(tile_key == key)

    def _refresh_tiles(self) -> None:
        counts = (self._result.counts if self._result else {}) or {}
        source_name = self._name_entry.text().strip() if hasattr(self, "_name_entry") else ""
        self._tiles["setup"].set_metrics(
            source_name or "no list chosen",
            f"{counts.get('total', 0)} rows" if counts else "",
        )
        self._tiles["triage"].set_metrics(
            f"{counts.get('owned', 0)} owned" if counts else "not run",
            f"{counts.get('unsure', 0)} unsure" if counts else "",
        )
        self._tiles["export"].set_metrics(
            f"{counts.get('missing', 0)} missing" if counts else "—",
            "ready" if counts.get("missing") else "",
        )
        self._tiles["verify"].set_metrics("Phase 2.5", "")
        self._tiles["summary"].set_metrics("Phase 3", "")

    # ── actions ──────────────────────────────────────────────────────────

    def _pick_csv(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a CSV export", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._csv_entry.setText(path)
            if not self._name_entry.text().strip():
                self._name_entry.setText(os.path.splitext(os.path.basename(path))[0])
            self._refresh_tiles()

    def _pick_library(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose your library folder")
        if path:
            self._lib_entry.setText(path)

    def _on_library_changed(self, path: str) -> None:
        if hasattr(self, "_lib_entry") and not self._lib_entry.text().strip():
            self._lib_entry.setText(path)

    def _start_run(self) -> None:
        csv_path = self._csv_entry.text().strip()
        library_root = self._lib_entry.text().strip()
        if not csv_path or not os.path.isfile(csv_path):
            self._warn("Choose a CSV export first.")
            return
        if not library_root or not os.path.isdir(library_root):
            self._warn("Choose your library folder first.")
            return

        from playlist_gap_sources import SourceSpec

        spec = SourceSpec(
            name=self._name_entry.text().strip() or os.path.basename(csv_path),
            kind="csv",
            location=csv_path,
        )
        cache_db = os.path.join(library_root, "Docs", "fingerprint_cache.db")

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self.status_changed.emit("Comparing…", "accent")

        self._run_worker = GapRunWorker(spec, library_root, cache_db)
        self._run_worker.progress.connect(self._on_progress)
        self._run_worker.log_line.connect(lambda line: self._log(line))
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.start()

    def _cancel_run(self) -> None:
        if self._run_worker is not None:
            self._run_worker.cancel()
            self._log("Cancelling…", "warn")

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        if message:
            self.status_changed.emit(message[:90], "accent")

    @Slot(bool, str, object)
    def _on_run_finished(self, success: bool, message: str, result) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._run_worker = None

        if not success or result is None:
            self._log(message, "error" if "Cancelled" not in message else "warn")
            self.status_changed.emit(message[:90], "danger")
            return

        self._result = result
        self._log(message, "success")
        self.status_changed.emit(message[:90], "success")
        caps = result.capabilities
        self._fields_label.setText(
            f"This file supplies: {caps.describe()}.  "
            + ("  ".join(result.warnings) if result.warnings else "")
        )
        for button in (self._copy_btn, self._save_txt_btn, self._save_csv_btn):
            button.setEnabled(True)
        self._refresh_tiles()
        self._fill_table()
        self._refresh_preview()
        self._show("triage")

    # ── rendering ────────────────────────────────────────────────────────

    def _fill_table(self) -> None:
        self._table.clear()
        if self._result is None:
            return
        wanted_bucket = self._bucket_combo.currentData()
        for row in self._result.results:
            if wanted_bucket and row.verdict.value != wanted_bucket:
                continue
            best = row.best
            item = QtWidgets.QTreeWidgetItem([
                row.wanted.label() or "(no title)",
                row.verdict.value,
                row.reason_text,
                os.path.basename(best.track.path) if best else "",
            ])
            if best:
                item.setToolTip(3, best.track.path)
            self._table.addTopLevelItem(item)

    def _refresh_preview(self) -> None:
        if self._result is None:
            return
        from playlist_gap_report import render_text

        self._preview.setPlainText(
            render_text(self._result.results, group_by=self._group_combo.currentData())
        )

    def _copy_list(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._preview.toPlainText())
            self._log("Download list copied to the clipboard.", "success")

    def _save(self, *, as_csv: bool) -> None:
        if self._result is None:
            return
        suffix = "csv" if as_csv else "txt"
        default = os.path.join(
            self._lib_entry.text().strip() or "", "Docs", f"playlist_gap_missing.{suffix}"
        )
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the download list", default, f"*.{suffix}"
        )
        if not path:
            return
        self._export_worker = GapExportWorker(
            self._result.results, path, self._group_combo.currentData(), as_csv
        )
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.start()

    @Slot(bool, str, str)
    def _on_export_finished(self, success: bool, message: str, path: str) -> None:
        self._export_worker = None
        self._log(message, "success" if success else "error")
        if success and self._result is not None:
            self._mark_exported()

    def _mark_exported(self) -> None:
        """Record what was exported, so a later verification pass can check it."""
        try:
            from playlist_gap_ledger import Ledger, ledger_path_for
            from playlist_gap_types import Verdict

            row_ids = [
                r.wanted.row_id for r in self._result.results if r.verdict is Verdict.MISSING
            ]
            if not row_ids:
                return
            ledger = Ledger(ledger_path_for(self._lib_entry.text().strip()))
            try:
                ledger.mark_exported(row_ids)
            finally:
                ledger.close()
            self._log(f"{len(row_ids)} row(s) marked as pending download.", "info")
        except Exception as exc:
            self._log(f"Could not record the export: {exc}", "warn")

    # ── helpers ──────────────────────────────────────────────────────────

    def _warn(self, message: str) -> None:
        self._log(message, "warn")
        self.status_changed.emit(message, "warning")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._run_worker is not None:
            self._run_worker.cancel()
            self._run_worker.wait(2000)
        super().closeEvent(event)
