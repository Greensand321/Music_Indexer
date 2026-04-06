"""Duplicate Bucketing POC dialog — Qt port of DuplicateBucketingPocDialog."""
from __future__ import annotations

import os
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot


class _BucketingWorker(QtCore.QThread):
    finished = Signal(str)   # report_path
    error    = Signal(str)

    def __init__(self, folder: str) -> None:
        super().__init__()
        self.folder = folder

    def run(self) -> None:
        try:
            from duplicate_bucketing_poc import run_duplicate_bucketing_poc
            report_path = run_duplicate_bucketing_poc(self.folder)
            self.finished.emit(report_path)
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
        self.setMinimumWidth(480)
        self._worker: _BucketingWorker | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(10)

        desc = QtWidgets.QLabel(
            "Select a folder to scan for duplicate bucketing. "
            "An HTML report will be written to the Docs/ subfolder."
        )
        desc.setWordWrap(True)
        root.addWidget(desc)

        folder_row = QtWidgets.QHBoxLayout()
        self._folder_edit = QtWidgets.QLineEdit(library_path)
        self._folder_edit.setPlaceholderText("Library folder…")
        browse_btn = QtWidgets.QPushButton("Browse\u2026")
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_edit, 1)
        folder_row.addWidget(browse_btn)
        root.addLayout(folder_row)

        self._status_lbl = QtWidgets.QLabel("Idle")
        self._status_lbl.setObjectName("sectionSubtitle")
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
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(
                self, "Folder Required", "Please select a valid folder."
            )
            return
        self._set_running(True, "Running\u2026")
        self._worker = _BucketingWorker(folder)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @Slot(str)
    def _on_finished(self, report_path: str) -> None:
        self._set_running(False, "Completed")
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
        self._worker = None

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._set_running(False, "Failed")
        QtWidgets.QMessageBox.critical(
            self, "Duplicate Bucketing POC", f"Run failed:\n{message}"
        )
        self._worker = None

    def _set_running(self, running: bool, status: str = "") -> None:
        self._run_btn.setEnabled(not running)
        if status:
            self._status_lbl.setText(status)
