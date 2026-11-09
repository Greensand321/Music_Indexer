"""Duplicate Scan Engine dialog — Qt port of DuplicateScanEngineTool."""
from __future__ import annotations

import importlib.util
import os
import threading

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.themes.tokens import SPACE
from gui.widgets.glass_parts import hero_title, section_card
from gui.widgets.glyphs import set_glyph
from gui.workers import start_worker


def _missing_engine_modules() -> list[str]:
    """Optional modules ``duplicate_scan_engine`` imports at load time."""
    return [name for name in ("numpy", "librosa") if importlib.util.find_spec(name) is None]


class _ScanWorker(QtCore.QThread):
    log_line = Signal(str)
    done = Signal(object)   # DuplicateScanSummary
    error = Signal(str)

    def __init__(self, library_path: str, db_path: str, config) -> None:
        super().__init__()
        self.library_path = library_path
        self.db_path = db_path
        self.config = config
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            from duplicate_scan_engine import run_duplicate_scan
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            summary = run_duplicate_scan(
                self.library_path,
                self.db_path,
                self.config,
                log_callback=self.log_line.emit,
                cancel_event=self._cancel_event,
            )
            self.done.emit(summary)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class ScanEngineDialog(QtWidgets.QDialog):
    """GUI wrapper for the staged duplicate scan engine."""

    _SETTINGS: list[tuple[str, str, str]] = [
        ("sample_rate",               "Sample rate (Hz)",           "11025"),
        ("max_analysis_sec",          "Max analysis seconds",       "120"),
        ("duration_tolerance_ms",     "Duration tolerance (ms)",    "2000"),
        ("duration_tolerance_ratio",  "Duration tolerance ratio",   "0.01"),
        ("fp_bands",                  "FP bands",                   "8"),
        ("min_band_collisions",       "Min band collisions",        "2"),
        ("fp_distance_threshold",     "FP distance threshold",      "0.2"),
        ("chroma_max_offset_frames",  "Chroma offset frames",       "12"),
        ("chroma_match_threshold",    "Chroma match threshold",     "0.82"),
        ("chroma_possible_threshold", "Chroma possible threshold",  "0.72"),
    ]

    def __init__(
        self,
        library_path: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Duplicate Scan Engine")
        # Its Duplicates-page launcher opens a fresh one each time; don't keep each one alive.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMinimumWidth(600)
        self.resize(680, 680)
        self._worker: _ScanWorker | None = None
        self._closing = False
        self._field_edits: dict[str, QtWidgets.QLineEdit] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(SPACE.lg, SPACE.lg, SPACE.lg, SPACE.lg)
        root.setSpacing(SPACE.md)

        root.addWidget(hero_title("A staged duplicate scan"))
        desc = QtWidgets.QLabel(
            "Runs a staged duplicate scan (audio headers → fingerprint LSH → "
            "verification) and writes results to a SQLite database. Read-only on audio. "
            "Nothing in AlphaDEX reads that database yet: the summary below is the only output."
        )
        desc.setWordWrap(True)
        desc.setObjectName("sectionSubtitle")
        root.addWidget(desc)

        # ── Paths ──────────────────────────────────────────────────────────
        paths_group = section_card("Paths")
        pl = paths_group.layout()

        self._lib_edit = QtWidgets.QLineEdit(library_path)
        lib_browse = QtWidgets.QPushButton("Browse…")
        set_glyph(lib_browse, "folder")
        lib_browse.setAccessibleName("Browse for the library root")
        lib_browse.clicked.connect(self._browse_library)
        self._db_edit = QtWidgets.QLineEdit(self._default_db_path(library_path))
        db_browse = QtWidgets.QPushButton("Browse…")
        set_glyph(db_browse, "folder")
        db_browse.setAccessibleName("Browse for the database file")
        db_browse.clicked.connect(self._browse_db)
        paths = QtWidgets.QGridLayout()
        paths.setHorizontalSpacing(SPACE.sm)
        paths.setVerticalSpacing(SPACE.xs)
        for row, (caption, edit, browse) in enumerate((("Library root", self._lib_edit, lib_browse),
                                                       ("Database", self._db_edit, db_browse))):
            lbl = QtWidgets.QLabel(caption)
            lbl.setObjectName("mutedLabel")
            lbl.setBuddy(edit)
            edit.setAccessibleName(caption)
            paths.addWidget(lbl, row, 0)
            paths.addWidget(edit, row, 1)
            paths.addWidget(browse, row, 2)
        paths.setColumnStretch(1, 1)
        pl.addLayout(paths)
        root.addWidget(paths_group)

        # ── Settings ───────────────────────────────────────────────────────
        settings_group = section_card("Scan settings")
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(SPACE.lg)
        grid.setVerticalSpacing(SPACE.xs)
        for i, (key, label, default) in enumerate(self._SETTINGS):
            edit = QtWidgets.QLineEdit(default)
            edit.setMaximumWidth(120)
            edit.setAccessibleName(label)
            self._field_edits[key] = edit
            caption = QtWidgets.QLabel(label)
            caption.setObjectName("mutedLabel")
            caption.setBuddy(edit)
            row, col = i // 2, (i % 2) * 2
            grid.addWidget(caption, row, col)
            grid.addWidget(edit, row, col + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        settings_group.layout().addLayout(grid)
        root.addWidget(settings_group)

        # ── Controls ───────────────────────────────────────────────────────
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(SPACE.sm)
        self._run_btn = QtWidgets.QPushButton("Run Scan")
        self._run_btn.setObjectName("primaryBtn")
        set_glyph(self._run_btn, "search", role="inverse")
        self._run_btn.clicked.connect(self._run_scan)
        self._status_lbl = QtWidgets.QLabel("Idle")
        self._status_lbl.setObjectName("statusHint")
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFlat(True)
        close_btn.clicked.connect(self.close)
        ctrl_row.addWidget(self._run_btn)
        ctrl_row.addWidget(self._status_lbl)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(close_btn)
        root.addLayout(ctrl_row)

        # ── Log ────────────────────────────────────────────────────────────
        log_group = section_card("Log")
        self._log_box = QtWidgets.QPlainTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(90)
        self._log_box.setObjectName("logText")
        self._log_box.setPlaceholderText("The engine's progress and its summary appear here.")
        log_group.layout().addWidget(self._log_box)
        root.addWidget(log_group, 1)

        missing = _missing_engine_modules()
        if missing:
            names = ", ".join(missing)
            self._run_btn.setEnabled(False)
            self._run_btn.setToolTip(f"Needs {names}: pip install {' '.join(missing)}")
            self._status_lbl.setText(f"Unavailable: needs {names}")

    # ── Path helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _default_db_path(library_path: str) -> str:
        # Only a suggestion: Docs/ is created when a scan actually runs, not
        # merely by opening the dialog.
        if not library_path:
            return ""
        return os.path.join(library_path, "Docs", "duplicate_scan.db")

    @Slot()
    def _browse_library(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Library Root", self._lib_edit.text() or ""
        )
        if path:
            self._lib_edit.setText(path)
            self._db_edit.setText(self._default_db_path(path))

    @Slot()
    def _browse_db(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Database Path",
            self._db_edit.text() or "",
            "SQLite DB (*.db);;All files (*.*)",
        )
        if path:
            self._db_edit.setText(path)

    # ── Run ────────────────────────────────────────────────────────────────

    @Slot()
    def _run_scan(self) -> None:
        if self._worker is not None:
            return
        library_path = self._lib_edit.text().strip()
        db_path      = self._db_edit.text().strip()
        if not library_path:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate Scan Engine", "Select a library first."
            )
            return
        if not db_path:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate Scan Engine", "Select a database path."
            )
            return

        try:
            from duplicate_scan_engine import DuplicateScanConfig
        except Exception as exc:  # noqa: BLE001 - ImportError and friends from librosa
            QtWidgets.QMessageBox.warning(
                self,
                "Duplicate Scan Engine",
                f"The scan engine could not be loaded:\n{exc}\n\n"
                "It needs numpy and librosa (pip install numpy librosa).",
            )
            self._log_box.appendPlainText(f"Error: scan engine unavailable: {exc}")
            return
        try:
            config = DuplicateScanConfig(
                sample_rate=int(self._field_edits["sample_rate"].text()),
                max_analysis_sec=float(self._field_edits["max_analysis_sec"].text()),
                duration_tolerance_ms=int(self._field_edits["duration_tolerance_ms"].text()),
                duration_tolerance_ratio=float(self._field_edits["duration_tolerance_ratio"].text()),
                fp_bands=int(self._field_edits["fp_bands"].text()),
                min_band_collisions=int(self._field_edits["min_band_collisions"].text()),
                fp_distance_threshold=float(self._field_edits["fp_distance_threshold"].text()),
                chroma_max_offset_frames=int(self._field_edits["chroma_max_offset_frames"].text()),
                chroma_match_threshold=float(self._field_edits["chroma_match_threshold"].text()),
                chroma_possible_threshold=float(self._field_edits["chroma_possible_threshold"].text()),
            )
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Duplicate Scan Engine",
                "Invalid settings — check that all numeric fields are valid.",
            )
            return

        self._log_box.clear()
        self._set_running(True)
        self._worker = _ScanWorker(library_path, db_path, config)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.done.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        start_worker(self._worker, owner=self, name="Duplicate scan engine")

    def reject(self) -> None:  # noqa: D401 - Esc, Close and the title bar all end here
        """Closing mid-run cancels the scan (it stops at the next track)."""
        if self._worker is not None:
            self._closing = True
            self._worker.cancel()
        super().reject()

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self._log_box.appendPlainText(line)

    @Slot(object)
    def _on_finished(self, summary) -> None:
        self._worker = None
        self._set_running(False)
        if getattr(summary, "cancelled", False):
            self._status_lbl.setText("Cancelled")
            return
        self._log_box.appendPlainText(
            f"Summary: {summary.tracks_total} tracks, "
            f"{summary.headers_updated} headers updated, "
            f"{summary.fingerprints_updated} fingerprints updated, "
            f"{summary.edges_written} edges, "
            f"{summary.groups_written} groups."
        )

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._worker = None
        self._set_running(False)
        self._status_lbl.setText("Failed")
        self._log_box.appendPlainText(f"Error: {message}")

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running and not _missing_engine_modules())
        self._status_lbl.setText("Running…" if running else "Idle")
