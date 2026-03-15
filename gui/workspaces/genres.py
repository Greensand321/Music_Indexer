"""Genre Normalizer workspace — batch-update genres via MusicBrainz / Last.fm."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class GenreWorker(QtCore.QThread):
    progress = Signal(int)
    log_line = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, library_path: str, dry_run: bool, overwrite: bool) -> None:
        super().__init__()
        self.library_path = library_path
        self.dry_run = dry_run
        self.overwrite = overwrite
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import update_genres
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.log_line.emit(msg)

            update_genres.run(
                self.library_path,
                dry_run=self.dry_run,
                overwrite=self.overwrite,
                log_callback=_log,
            )
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
            else:
                self.finished.emit(True, "Genre update complete.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class GenresWorkspace(WorkspaceBase):
    """Batch genre update via MusicBrainz / Last.fm."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: GenreWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Genre Normalizer"))
        cl.addWidget(self._make_subtitle(
            "Look up each track's genre via MusicBrainz or Last.fm and write "
            "standardised genre tags back to the files. Supports dry-run previews."
        ))

        # ── Options card ───────────────────────────────────────────────────
        opt_card = self._make_card()
        opt_layout = QtWidgets.QVBoxLayout(opt_card)
        opt_layout.setContentsMargins(16, 16, 16, 16)
        opt_layout.setSpacing(10)
        opt_layout.addWidget(QtWidgets.QLabel("Options"))

        row1 = QtWidgets.QHBoxLayout()
        self._dry_run_cb = QtWidgets.QCheckBox("Dry run (show changes, do not write)")
        self._dry_run_cb.setChecked(True)
        self._overwrite_cb = QtWidgets.QCheckBox("Overwrite existing genre tags")
        row1.addWidget(self._dry_run_cb)
        row1.addWidget(self._overwrite_cb)
        row1.addStretch(1)
        opt_layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Source:"))
        self._source_combo = QtWidgets.QComboBox()
        self._source_combo.addItems(["MusicBrainz", "Last.fm", "Both"])
        row2.addWidget(self._source_combo)
        row2.addStretch(1)
        opt_layout.addLayout(row2)
        cl.addWidget(opt_card)

        # ── Buttons ────────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = self._make_primary_button("🎸  Run Genre Update")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        # ── Progress ───────────────────────────────────────────────────────
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setValue(0)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setFixedHeight(6)
        self._prog_status = QtWidgets.QLabel("Idle")
        self._prog_status.setStyleSheet("color: #64748b; font-size: 12px;")
        cl.addWidget(self._prog_bar)
        cl.addWidget(self._prog_status)

        # ── Log ────────────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(QtWidgets.QLabel("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(260)
        self._log_area.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return

        self._log_area.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Running…")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._log("Starting genre update…", "info")
        self.status_changed.emit("Updating genres…", "#f59e0b")

        self._worker = GenreWorker(
            self._library_path,
            dry_run=self._dry_run_cb.isChecked(),
            overwrite=self._overwrite_cb.isChecked(),
        )
        self._worker.log_line.connect(self._log_area.appendPlainText)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._prog_status.setText(message)
        if success:
            self._prog_bar.setValue(100)
            self._log(message, "ok")
            self.status_changed.emit("Done", "#22c55e")
        else:
            self._log(message, "error" if "Cancelled" not in message else "warn")
            self.status_changed.emit("Error", "#ef4444")
        self._worker = None
