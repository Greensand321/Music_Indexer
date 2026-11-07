"""Duplicate Bucketing POC dialog — Qt port of DuplicateBucketingPocDialog."""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.workers import start_worker


class _BucketingWorker(QtCore.QThread):
    done = Signal(str)       # report_path
    error = Signal(str)
    cancelled = Signal()
    log_line = Signal(str)

    def __init__(self, folder: str) -> None:
        super().__init__()
        self.folder = folder
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            from duplicate_bucketing_poc import BucketingCancelled, run_duplicate_bucketing_poc
        except Exception as exc:  # noqa: BLE001 - e.g. a missing optional dependency
            self.error.emit(f"Could not load the bucketing engine: {exc}")
            return
        try:
            report_path = run_duplicate_bucketing_poc(
                self.folder,
                log_callback=self.log_line.emit,
                cancel_event=self._cancel_event,
            )
            self.done.emit(report_path)
        except BucketingCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class BucketingPocDialog(QtWidgets.QDialog):
    """Minimal UI for the Duplicate Bucketing proof-of-concept tool."""

    def __init__(
        self,
        library_path: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Duplicate Bucketing POC")
        # The Tools launcher creates one per click; don't keep each one alive.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMinimumWidth(480)
        self._worker: _BucketingWorker | None = None
        self._closing = False
        self._without_fingerprints = False

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(10)

        desc = QtWidgets.QLabel(
            "Select a folder to scan for duplicate bucketing. "
            "An HTML report will be written to the Docs/ subfolder. "
            "Read-only: no audio file is changed."
        )
        desc.setWordWrap(True)
        root.addWidget(desc)

        folder_row = QtWidgets.QHBoxLayout()
        self._folder_edit = QtWidgets.QLineEdit(library_path)
        self._folder_edit.setPlaceholderText("Library folder…")
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_edit, 1)
        folder_row.addWidget(browse_btn)
        root.addLayout(folder_row)

        self._status_lbl = QtWidgets.QLabel("Idle")
        self._status_lbl.setObjectName("sectionSubtitle")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.clicked.connect(self._run)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

    @Slot()
    def _browse(self) -> None:
        initial = self._folder_edit.text() or os.getcwd()
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder for Duplicate Bucketing", initial
        )
        if path:
            self._folder_edit.setText(path)

    @Slot()
    def _run(self) -> None:
        if self._worker is not None:
            return
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(
                self, "Folder Required", "Please select a valid folder."
            )
            return
        import chromaprint_utils

        missing = chromaprint_utils.missing_tools()
        self._without_fingerprints = bool(missing)
        if missing:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Duplicate Bucketing POC",
                f"{' and '.join(missing)} not found on PATH, so nothing can be fingerprinted: "
                "the report will show metadata/artwork buckets only, with every comparison "
                "marked 'missing'.\n\nRun anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._set_running(True, "Running…")
        self._worker = _BucketingWorker(folder)
        self._worker.done.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.log_line.connect(self._on_log_line)
        start_worker(self._worker, owner=self, name="Duplicate bucketing POC")

    def reject(self) -> None:  # noqa: D401 - Esc, Close and the title bar all end here
        """Closing mid-run cancels the run (it stops at the next track)."""
        if self._worker is not None:
            self._closing = True
            self._worker.cancel()
        super().reject()

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        if not self._closing:
            self._status_lbl.setText(line)

    @Slot(str)
    def _on_finished(self, report_path: str) -> None:
        self._worker = None
        if self._closing:
            return  # the dialog was closed: no pop-ups from a dialog that is gone
        status = "Completed"
        if self._without_fingerprints:
            status = "Completed without fingerprints (fpcalc not found)"
        self._set_running(False, status)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Duplicate Bucketing POC",
            f"Report saved to:\n{report_path}\n\nOpen it now?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                uri = Path(report_path).resolve().as_uri()
            except Exception:
                uri = report_path
            webbrowser.open(uri)

    @Slot()
    def _on_cancelled(self) -> None:
        self._worker = None
        if not self._closing:
            self._set_running(False, "Cancelled")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._worker = None
        if self._closing:
            return
        self._set_running(False, "Failed")
        QtWidgets.QMessageBox.critical(
            self, "Duplicate Bucketing POC", f"Run failed:\n{message}"
        )

    def _set_running(self, running: bool, status: str = "") -> None:
        self._run_btn.setEnabled(not running)
        if status:
            self._status_lbl.setText(status)
