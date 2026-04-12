"""Library Compression workspace — mirror library converting FLAC → Opus."""
from __future__ import annotations

import os
import shutil
import webbrowser
from datetime import datetime
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class MirrorWorker(QtCore.QThread):
    """Background thread that runs mirror_library()."""

    progress = Signal(int, int, int, int, int, int)  # total_tasks, completed, converted, copied, skipped, errors
    log_line = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        source: str,
        destination: str,
        overwrite: bool,
        bitrate_kbps: int,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.destination = destination
        self.overwrite = overwrite
        self.bitrate_kbps = bitrate_kbps

    def run(self) -> None:
        try:
            from utils.opus_library_mirror import mirror_library
            summary = mirror_library(
                self.source,
                self.destination,
                self.overwrite,
                progress_callback=self._on_progress,
                log_callback=self.log_line.emit,
                bitrate_kbps=self.bitrate_kbps,
            )
            self.finished.emit(summary)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def _on_progress(
        self,
        total_tasks: int,
        completed: int,
        converted: int,
        copied: int,
        skipped: int,
        errors: int,
    ) -> None:
        self.progress.emit(total_tasks, completed, converted, copied, skipped, errors)


class CompressionWorkspace(WorkspaceBase):
    """Mirror a library, converting FLAC → Opus and copying everything else."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: MirrorWorker | None = None
        self._report_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Library Compression"))
        cl.addWidget(self._make_subtitle(
            "Create a compressed mirror of your library. FLAC files are converted to Opus "
            "at the chosen bitrate; all other formats are copied unchanged. "
            "Originals in the source library are never modified."
        ))

        # ── Source / Destination ──────────────────────────────────────────
        paths_card = self._make_card()
        paths_layout = QtWidgets.QFormLayout(paths_card)
        paths_layout.setContentsMargins(16, 16, 16, 16)
        paths_layout.setSpacing(10)
        paths_layout.addRow(self._make_card_title("Folders"))

        src_row = QtWidgets.QHBoxLayout()
        self._source_edit = QtWidgets.QLineEdit()
        self._source_edit.setPlaceholderText("Source library folder…")
        if self._library_path:
            self._source_edit.setText(self._library_path)
        src_browse = QtWidgets.QPushButton("Browse…")
        src_browse.setFixedWidth(80)
        src_browse.clicked.connect(self._browse_source)
        src_row.addWidget(self._source_edit)
        src_row.addWidget(src_browse)
        paths_layout.addRow("Source:", src_row)

        dst_row = QtWidgets.QHBoxLayout()
        self._dest_edit = QtWidgets.QLineEdit()
        self._dest_edit.setPlaceholderText("Destination (mirror) folder…")
        dst_browse = QtWidgets.QPushButton("Browse…")
        dst_browse.setFixedWidth(80)
        dst_browse.clicked.connect(self._browse_destination)
        dst_row.addWidget(self._dest_edit)
        dst_row.addWidget(dst_browse)
        paths_layout.addRow("Destination:", dst_row)
        cl.addWidget(paths_card)

        # ── Transcode settings ────────────────────────────────────────────
        cfg_card = self._make_card()
        cfg_layout = QtWidgets.QVBoxLayout(cfg_card)
        cfg_layout.setContentsMargins(16, 16, 16, 16)
        cfg_layout.setSpacing(10)
        cfg_layout.addWidget(self._make_card_title("Transcode Settings"))

        info = QtWidgets.QLabel("FLAC → Opus  |  All other formats: copied unchanged")
        info.setStyleSheet("color: #64748b; font-size: 12px;")
        cfg_layout.addWidget(info)

        bitrate_row = QtWidgets.QHBoxLayout()
        bitrate_row.addWidget(QtWidgets.QLabel("Opus bitrate (kbps):"))
        self._bitrate_spin = QtWidgets.QSpinBox()
        self._bitrate_spin.setRange(48, 320)
        self._bitrate_spin.setValue(96)
        self._bitrate_spin.setSingleStep(16)
        bitrate_row.addWidget(self._bitrate_spin)
        bitrate_row.addStretch(1)
        cfg_layout.addLayout(bitrate_row)

        self._overwrite_cb = QtWidgets.QCheckBox("Overwrite existing files in destination")
        self._overwrite_cb.setChecked(False)
        cfg_layout.addWidget(self._overwrite_cb)
        cl.addWidget(cfg_card)

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._estimate_btn = self._make_primary_button("🔍  Estimate Space Savings")
        self._estimate_btn.clicked.connect(self._on_estimate)
        self._run_btn = QtWidgets.QPushButton("▶  Start Compression")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        self._report_btn = QtWidgets.QPushButton("📄  Open Report")
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._on_open_report)
        btn_row.addWidget(self._estimate_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._report_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        # ── Progress / results card ───────────────────────────────────────
        prog_card = self._make_card()
        prog_layout = QtWidgets.QVBoxLayout(prog_card)
        prog_layout.setContentsMargins(16, 16, 16, 16)
        prog_layout.setSpacing(8)
        prog_layout.addWidget(self._make_card_title("Progress"))
        self._result_lbl = QtWidgets.QLabel(
            "Run an estimate to preview space savings, then start compression."
        )
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setStyleSheet("color: #64748b;")
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setValue(0)
        self._prog_bar.setFixedHeight(8)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setVisible(False)
        self._prog_counter = QtWidgets.QLabel("")
        self._prog_counter.setStyleSheet("color: #64748b; font-size: 12px;")
        prog_layout.addWidget(self._result_lbl)
        prog_layout.addWidget(self._prog_bar)
        prog_layout.addWidget(self._prog_counter)
        cl.addWidget(prog_card)

        note = QtWidgets.QLabel(
            "Note: FFmpeg must be installed and on your PATH for FLAC → Opus conversion."
        )
        note.setStyleSheet("color: #94a3b8; font-size: 11px;")
        cl.addWidget(note)
        cl.addStretch(1)

    # ── Library path sync ─────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        self._source_edit.setText(path)

    # ── Browse ────────────────────────────────────────────────────────────

    @Slot()
    def _browse_source(self) -> None:
        initial = self._source_edit.text() or os.path.expanduser("~")
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Source Library", initial
        )
        if folder:
            self._source_edit.setText(folder)

    @Slot()
    def _browse_destination(self) -> None:
        initial = self._dest_edit.text() or os.path.expanduser("~")
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Destination Folder", initial
        )
        if folder:
            self._dest_edit.setText(folder)

    # ── Validation ────────────────────────────────────────────────────────

    def _validate_paths(self) -> tuple[str, str] | None:
        """Return (source, destination) or None after showing a warning."""
        source = self._source_edit.text().strip()
        destination = self._dest_edit.text().strip()

        if not source or not os.path.isdir(source):
            QtWidgets.QMessageBox.warning(
                self, "Source Required", "Select a valid source library folder."
            )
            return None
        if not destination:
            QtWidgets.QMessageBox.warning(
                self, "Destination Required", "Select a destination folder."
            )
            return None
        if not os.path.isdir(destination):
            try:
                os.makedirs(destination, exist_ok=True)
            except OSError as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Destination",
                    f"Cannot create destination folder:\n{exc}"
                )
                return None
        if os.path.abspath(source) == os.path.abspath(destination):
            QtWidgets.QMessageBox.critical(
                self, "Invalid Destination",
                "Destination must be different from source."
            )
            return None
        try:
            if os.path.commonpath([
                os.path.abspath(source), os.path.abspath(destination)
            ]) == os.path.abspath(source):
                QtWidgets.QMessageBox.critical(
                    self, "Invalid Destination",
                    "Destination cannot be inside the source library."
                )
                return None
        except ValueError:
            pass
        return source, destination

    # ── Estimate ──────────────────────────────────────────────────────────

    @Slot()
    def _on_estimate(self) -> None:
        paths = self._validate_paths()
        if paths is None:
            return
        source, _ = paths

        flac_count = 0
        flac_bytes = 0
        other_count = 0
        other_bytes = 0

        for root, _dirs, files in os.walk(source):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                try:
                    size = os.path.getsize(os.path.join(root, name))
                except OSError:
                    size = 0
                if ext == ".flac":
                    flac_count += 1
                    flac_bytes += size
                else:
                    other_count += 1
                    other_bytes += size

        # Estimate Opus output size: FLAC averages ~900 kbps for 16-bit/44.1 kHz stereo.
        # Opus output ratio ≈ target_bitrate / 900.
        bitrate = self._bitrate_spin.value()
        estimated_opus_bytes = int(flac_bytes * bitrate / 900)
        savings = flac_bytes - estimated_opus_bytes

        def _fmt(n: int) -> str:
            if n >= 1_073_741_824:
                return f"{n / 1_073_741_824:.1f} GB"
            if n >= 1_048_576:
                return f"{n / 1_048_576:.1f} MB"
            return f"{n / 1024:.0f} KB"

        msg = (
            f"FLAC files to convert:    {flac_count}  ({_fmt(flac_bytes)})\n"
            f"Estimated Opus output:    {_fmt(estimated_opus_bytes)}  ({bitrate} kbps)\n"
            f"Estimated space saving:   {_fmt(savings)}\n"
            f"Other files to copy:      {other_count}  ({_fmt(other_bytes)})"
        )
        self._result_lbl.setText(msg)
        self._result_lbl.setStyleSheet("color: #e2e8f0;")
        self._run_btn.setEnabled(flac_count > 0 or other_count > 0)
        self._log(
            f"Compression estimate: {flac_count} FLAC ({_fmt(flac_bytes)}) → "
            f"~{_fmt(estimated_opus_bytes)} Opus, saving ~{_fmt(savings)}",
            "info",
        )

    # ── Run ───────────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self) -> None:
        paths = self._validate_paths()
        if paths is None:
            return
        source, destination = paths

        if not shutil.which("ffmpeg"):
            QtWidgets.QMessageBox.critical(
                self, "FFmpeg Required",
                "FFmpeg was not found on your PATH.\n"
                "Install FFmpeg to convert FLAC files."
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Start Compression",
            f"Convert FLAC → Opus ({self._bitrate_spin.value()} kbps) and copy all other files.\n\n"
            f"Source:       {source}\n"
            f"Destination:  {destination}\n\n"
            "Originals in the source are never modified. Proceed?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self._set_running(True)
        self._report_path = None
        self._report_btn.setEnabled(False)
        self._log(f"Starting compression: {source} → {destination}", "info")

        self._worker = MirrorWorker(
            source,
            destination,
            overwrite=self._overwrite_cb.isChecked(),
            bitrate_kbps=self._bitrate_spin.value(),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(lambda msg: self._log(msg, "info"))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _set_running(self, running: bool) -> None:
        self._estimate_btn.setEnabled(not running)
        self._run_btn.setEnabled(not running)
        self._source_edit.setEnabled(not running)
        self._dest_edit.setEnabled(not running)
        self._overwrite_cb.setEnabled(not running)
        self._prog_bar.setVisible(running)
        if running:
            self._prog_bar.setMaximum(0)  # indeterminate until first tick
            self._prog_bar.setValue(0)
            self._prog_counter.setText("Starting…")
            self._result_lbl.setText("Running…")
            self._result_lbl.setStyleSheet("color: #64748b;")

    @Slot(int, int, int, int, int, int)
    def _on_progress(
        self,
        total_tasks: int,
        completed: int,
        converted: int,
        copied: int,
        skipped: int,
        errors: int,
    ) -> None:
        if total_tasks > 0:
            self._prog_bar.setMaximum(total_tasks)
            self._prog_bar.setValue(completed)
        self._prog_counter.setText(
            f"Progress: {completed} / {total_tasks}  "
            f"(converted {converted}, copied {copied}, "
            f"skipped {skipped}, errors {errors})"
        )

    @Slot(dict)
    def _on_finished(self, summary: dict) -> None:
        self._set_running(False)
        self._prog_bar.setMaximum(100)
        self._prog_bar.setValue(100)

        msg = (
            f"Done.  Converted {summary['converted']} FLAC → Opus, "
            f"copied {summary['copied']}, "
            f"skipped {summary['skipped']}, "
            f"errors {summary['errors']}."
        )
        self._result_lbl.setText(msg)
        self._result_lbl.setStyleSheet("color: #4ade80;")
        self._log(msg, "ok")

        destination = self._dest_edit.text().strip()
        if destination:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(
                destination, "Docs", "opus_library_mirror_reports"
            )
            report_path = os.path.join(
                report_dir, f"opus_library_mirror_report_{timestamp}.html"
            )
            try:
                from utils.opus_library_mirror import write_mirror_report
                write_mirror_report(report_path, summary)
                self._report_path = report_path
                self._report_btn.setEnabled(True)
                self._log(f"Report saved: {report_path}", "ok")
            except OSError as exc:
                self._log(f"Failed to write report: {exc}", "warn")

        self._worker = None

    @Slot(str)
    def _on_error(self, error: str) -> None:
        self._set_running(False)
        self._result_lbl.setText(f"Error: {error}")
        self._result_lbl.setStyleSheet("color: #f87171;")
        self._log(f"Compression failed: {error}", "error")
        self._worker = None

    # ── Open report ───────────────────────────────────────────────────────

    @Slot()
    def _on_open_report(self) -> None:
        if not self._report_path or not os.path.exists(self._report_path):
            QtWidgets.QMessageBox.warning(
                self, "Report Not Found",
                "The report file could not be found. Run compression again to generate one."
            )
            return
        try:
            uri = Path(self._report_path).resolve().as_uri()
        except Exception:  # noqa: BLE001
            uri = self._report_path
        webbrowser.open(uri)
