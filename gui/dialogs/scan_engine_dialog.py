"""Duplicate Scan Engine dialog — Qt port of DuplicateScanEngineTool."""
from __future__ import annotations

import os

from gui.compat import QtCore, QtWidgets, Signal, Slot


class _ScanWorker(QtCore.QThread):
    log_line = Signal(str)
    finished = Signal(object)   # DuplicateScanSummary
    error    = Signal(str)

    def __init__(self, library_path: str, db_path: str, config) -> None:
        super().__init__()
        self.library_path = library_path
        self.db_path      = db_path
        self.config       = config

    def run(self) -> None:
        try:
            from duplicate_scan_engine import run_duplicate_scan
            summary = run_duplicate_scan(
                self.library_path,
                self.db_path,
                self.config,
                log_callback=lambda msg: self.log_line.emit(msg),
            )
            self.finished.emit(summary)
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
        self.setMinimumWidth(560)
        self.resize(600, 580)
        self._worker: _ScanWorker | None = None
        self._field_edits: dict[str, QtWidgets.QLineEdit] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(10)

        desc = QtWidgets.QLabel(
            "Runs a staged duplicate scan (audio headers \u2192 fingerprint LSH \u2192 "
            "verification) and writes results to a SQLite database."
        )
        desc.setWordWrap(True)
        root.addWidget(desc)

        # ── Paths ──────────────────────────────────────────────────────────
        paths_group = QtWidgets.QGroupBox("Paths")
        paths_form  = QtWidgets.QFormLayout(paths_group)
        paths_form.setSpacing(6)

        lib_row = QtWidgets.QHBoxLayout()
        self._lib_edit = QtWidgets.QLineEdit(library_path)
        lib_browse = QtWidgets.QPushButton("Browse\u2026")
        lib_browse.clicked.connect(self._browse_library)
        lib_row.addWidget(self._lib_edit, 1)
        lib_row.addWidget(lib_browse)
        paths_form.addRow("Library Root:", lib_row)

        db_row = QtWidgets.QHBoxLayout()
        self._db_edit = QtWidgets.QLineEdit(self._default_db_path(library_path))
        db_browse = QtWidgets.QPushButton("Browse\u2026")
        db_browse.clicked.connect(self._browse_db)
        db_row.addWidget(self._db_edit, 1)
        db_row.addWidget(db_browse)
        paths_form.addRow("Database:", db_row)

        root.addWidget(paths_group)

        # ── Settings ───────────────────────────────────────────────────────
        settings_group = QtWidgets.QGroupBox("Scan Settings")
        settings_form  = QtWidgets.QFormLayout(settings_group)
        settings_form.setSpacing(4)
        for key, label, default in self._SETTINGS:
            edit = QtWidgets.QLineEdit(default)
            edit.setFixedWidth(100)
            self._field_edits[key] = edit
            settings_form.addRow(f"{label}:", edit)
        root.addWidget(settings_group)

        # ── Controls ───────────────────────────────────────────────────────
        ctrl_row = QtWidgets.QHBoxLayout()
        self._run_btn = QtWidgets.QPushButton("Run Scan")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.clicked.connect(self._run_scan)
        self._status_lbl = QtWidgets.QLabel("Idle")
        self._status_lbl.setObjectName("sectionSubtitle")
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        ctrl_row.addWidget(self._run_btn)
        ctrl_row.addWidget(self._status_lbl)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(close_btn)
        root.addLayout(ctrl_row)

        # ── Log ────────────────────────────────────────────────────────────
        log_group  = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self._log_box = QtWidgets.QPlainTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(120)
        self._log_box.setObjectName("logBox")
        log_layout.addWidget(self._log_box)
        root.addWidget(log_group)

    # ── Path helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _default_db_path(library_path: str) -> str:
        if not library_path:
            return ""
        docs_dir = os.path.join(library_path, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        return os.path.join(docs_dir, "duplicate_scan.db")

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
        if self._worker and self._worker.isRunning():
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
        self._worker.log_line.connect(self._log_box.appendPlainText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @Slot(object)
    def _on_finished(self, summary) -> None:
        self._set_running(False)
        self._log_box.appendPlainText(
            f"Summary: {summary.tracks_total} tracks, "
            f"{summary.headers_updated} headers updated, "
            f"{summary.fingerprints_updated} fingerprints updated, "
            f"{summary.edges_written} edges, "
            f"{summary.groups_written} groups."
        )
        self._worker = None

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self._log_box.appendPlainText(f"Error: {message}")
        self._worker = None

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._status_lbl.setText("Running\u2026" if running else "Idle")
