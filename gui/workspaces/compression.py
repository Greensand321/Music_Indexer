"""Library Compression workspace — mirror library converting FLAC → Opus."""
from __future__ import annotations

import os
import shutil
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase
from gui.workspaces.label_tone import refresh_tones, set_tone


class MirrorWorker(QtCore.QThread):
    """Background thread that runs mirror_library(), with a real cancel."""

    progress = Signal(int, int, int, int, int, int)  # total_tasks, completed, converted, copied, skipped, errors
    log_line = Signal(str)
    error_line = Signal(str)    # per-file failures (shown at "error" level)
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
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Stop: running ffmpeg processes are terminated and their partial
        output removed; files not started yet are left alone."""
        self._cancel.set()

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
                cancel_event=self._cancel,
                error_callback=self.error_line.emit,
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


class EstimateWorker(QtCore.QThread):
    """Count FLAC vs other files and bytes (was a GUI-thread walk — TOOL-19)."""

    finished = Signal(int, int, int, int)  # flac_count, flac_bytes, other_count, other_bytes
    cancelled = Signal()

    def __init__(self, source: str, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.source = source
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        from utils.opus_library_mirror import MIRROR_EXCLUDED_FOLDERS
        from utils.reserved_folders import walk_library

        flac_count = flac_bytes = other_count = other_bytes = 0
        # Same exclusions as the mirror itself, so the estimate matches the run.
        for root, _dirs, files in walk_library(self.source, MIRROR_EXCLUDED_FOLDERS):
            if self._cancel.is_set():
                self.cancelled.emit()
                return
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
        self.finished.emit(flac_count, flac_bytes, other_count, other_bytes)


def _fmt(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    return f"{n / 1024:.0f} KB"


class CompressionWorkspace(WorkspaceBase):
    """Mirror a library, converting FLAC → Opus and copying everything else."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: MirrorWorker | None = None
        self._estimate_worker: EstimateWorker | None = None
        self._estimate_key: tuple | None = None   # (source, destination, bitrate) the estimate is for
        self._estimate_running_key: tuple | None = None
        self._pending_source: str | None = None   # library change that arrived mid-run
        self._report_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Library Compression"))
        cl.addWidget(self._make_subtitle(
            "Create a compressed mirror of your library. FLAC files are converted to Opus "
            "at the chosen bitrate; all other formats are copied unchanged. "
            "Originals in the source library are never modified. "
            "Quarantine/, Trash/ and Docs/ are not mirrored."
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
        self._src_browse = QtWidgets.QPushButton("Browse…")
        self._src_browse.setFixedWidth(80)
        self._src_browse.clicked.connect(self._browse_source)
        src_row.addWidget(self._source_edit)
        src_row.addWidget(self._src_browse)
        paths_layout.addRow("Source:", src_row)

        dst_row = QtWidgets.QHBoxLayout()
        self._dest_edit = QtWidgets.QLineEdit()
        self._dest_edit.setPlaceholderText("Destination (mirror) folder…")
        self._dst_browse = QtWidgets.QPushButton("Browse…")
        self._dst_browse.setFixedWidth(80)
        self._dst_browse.clicked.connect(self._browse_destination)
        dst_row.addWidget(self._dest_edit)
        dst_row.addWidget(self._dst_browse)
        paths_layout.addRow("Destination:", dst_row)
        cl.addWidget(paths_card)

        # ── Transcode settings ────────────────────────────────────────────
        cfg_card = self._make_card()
        cfg_layout = QtWidgets.QVBoxLayout(cfg_card)
        cfg_layout.setContentsMargins(16, 16, 16, 16)
        cfg_layout.setSpacing(10)
        cfg_layout.addWidget(self._make_card_title("Transcode Settings"))

        info = QtWidgets.QLabel("FLAC → Opus  |  All other formats: copied unchanged")
        info.setObjectName("statusHint")
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
        self._cancel_btn = QtWidgets.QPushButton("■  Cancel")
        self._cancel_btn.setObjectName("compressionCancelBtn")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._report_btn = QtWidgets.QPushButton("📄  Open Report")
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._on_open_report)
        btn_row.addWidget(self._estimate_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
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
        self._result_lbl.setObjectName("compressionResult")
        self._result_lbl.setWordWrap(True)
        set_tone(self._result_lbl, "muted")
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setValue(0)
        self._prog_bar.setFixedHeight(8)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setVisible(False)
        self._prog_counter = QtWidgets.QLabel("")
        self._prog_counter.setObjectName("statusHint")
        prog_layout.addWidget(self._result_lbl)
        prog_layout.addWidget(self._prog_bar)
        prog_layout.addWidget(self._prog_counter)
        cl.addWidget(prog_card)

        note = QtWidgets.QLabel(
            "Note: FFmpeg must be installed and on your PATH for FLAC → Opus conversion."
        )
        note.setObjectName("notesHint")
        cl.addWidget(note)
        cl.addStretch(1)

        # Any change to what was estimated invalidates the estimate (TOOL-23).
        self._source_edit.textChanged.connect(self._invalidate_estimate)
        self._dest_edit.textChanged.connect(self._invalidate_estimate)
        self._bitrate_spin.valueChanged.connect(self._invalidate_estimate)

    # ── Theme ─────────────────────────────────────────────────────────────

    def _on_theme_changed_base(self, tokens: object) -> None:
        super()._on_theme_changed_base(tokens)
        refresh_tones(self, tokens)

    # ── Library path sync ─────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        if self._worker is not None:
            # Don't change the Source field under a running mirror; apply the
            # new library when the run ends.
            self._pending_source = path
            return
        if self._estimate_worker is not None:
            self._estimate_worker.cancel()
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
        """Return (source, destination) or None after showing a warning.

        Never creates anything: the destination is created by the mirror run
        itself, after confirmation. (It used to ``makedirs`` the destination
        *before* checking it was outside the source, leaving empty folders in
        the user's library — TOOL-13.)
        """
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
        src_real = os.path.normcase(os.path.realpath(source))
        dst_real = os.path.normcase(os.path.realpath(destination))
        if src_real == dst_real:
            QtWidgets.QMessageBox.critical(
                self, "Invalid Destination",
                "Destination must be different from source."
            )
            return None
        try:
            inside = os.path.commonpath([src_real, dst_real]) == src_real
        except ValueError:  # different drives
            inside = False
        if inside:
            QtWidgets.QMessageBox.critical(
                self, "Invalid Destination",
                "Destination cannot be inside the source library."
            )
            return None
        if os.path.exists(destination) and not os.path.isdir(destination):
            QtWidgets.QMessageBox.warning(
                self, "Invalid Destination", "The destination exists and is not a folder."
            )
            return None
        return source, destination

    def _current_key(self) -> tuple:
        return (
            self._source_edit.text().strip(),
            self._dest_edit.text().strip(),
            self._bitrate_spin.value(),
        )

    @Slot()
    def _invalidate_estimate(self, *_args) -> None:
        if self._worker is not None:
            return
        if self._estimate_worker is not None:
            self._estimate_worker.cancel()
        if self._estimate_key is not None:
            self._estimate_key = None
            self._result_lbl.setText(
                "Settings changed — run the estimate again before starting."
            )
            set_tone(self._result_lbl, "muted")
        self._run_btn.setEnabled(False)

    # ── Estimate ──────────────────────────────────────────────────────────

    @Slot()
    def _on_estimate(self) -> None:
        if self._estimate_worker is not None or self._worker is not None:
            return
        paths = self._validate_paths()
        if paths is None:
            return
        source, _ = paths
        self._run_btn.setEnabled(False)
        self._estimate_btn.setEnabled(False)
        self._result_lbl.setText("Estimating…")
        set_tone(self._result_lbl, "muted")
        self._estimate_running_key = self._current_key()
        worker = EstimateWorker(source, parent=self)
        worker.finished.connect(self._on_estimate_done)
        worker.cancelled.connect(self._on_estimate_cancelled)
        self._estimate_worker = worker
        self._start_worker(worker, "Compression estimate", ask_before_quit=False)

    @Slot()
    def _on_estimate_cancelled(self) -> None:
        self._estimate_worker = None
        self._estimate_btn.setEnabled(self._worker is None)

    @Slot(int, int, int, int)
    def _on_estimate_done(
        self, flac_count: int, flac_bytes: int, other_count: int, other_bytes: int
    ) -> None:
        key = self._estimate_running_key
        self._estimate_worker = None
        self._estimate_btn.setEnabled(self._worker is None)
        if key != self._current_key():
            return  # settings changed while estimating; this result is stale

        # Estimate Opus output size: FLAC averages ~900 kbps for 16-bit/44.1 kHz stereo.
        # Opus output ratio ≈ target_bitrate / 900.
        bitrate = key[2]
        estimated_opus_bytes = int(flac_bytes * bitrate / 900)
        savings = flac_bytes - estimated_opus_bytes

        msg = (
            f"FLAC files to convert:    {flac_count}  ({_fmt(flac_bytes)})\n"
            f"Estimated Opus output:    {_fmt(estimated_opus_bytes)}  ({bitrate} kbps)\n"
            f"Estimated space saving:   {_fmt(savings)}\n"
            f"Other files to copy:      {other_count}  ({_fmt(other_bytes)})"
        )
        self._result_lbl.setText(msg)
        set_tone(self._result_lbl, "primary")
        self._estimate_key = key
        self._run_btn.setEnabled(flac_count > 0 or other_count > 0)
        self._log(
            f"Compression estimate: {flac_count} FLAC ({_fmt(flac_bytes)}) → "
            f"~{_fmt(estimated_opus_bytes)} Opus, saving ~{_fmt(savings)}",
            "info",
        )

    # ── Run ───────────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self) -> None:
        if self._worker is not None:
            return
        if self._estimate_key is None or self._estimate_key != self._current_key():
            self._invalidate_estimate()
            return
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

        overwrite = self._overwrite_cb.isChecked()
        reply = QtWidgets.QMessageBox.question(
            self, "Start Compression",
            f"Convert FLAC → Opus ({self._bitrate_spin.value()} kbps) and copy all other files.\n\n"
            f"Source:       {source}\n"
            f"Destination:  {destination}\n"
            "Existing files in the destination: "
            f"{'OVERWRITTEN' if overwrite else 'kept (skipped)'}\n"
            "Not mirrored: Quarantine/, Trash/, Docs/\n\n"
            "Originals in the source are never modified. Proceed?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self._set_running(True)
        self._report_path = None
        self._report_btn.setEnabled(False)
        self._log(f"Starting compression: {source} → {destination}", "info")

        worker = MirrorWorker(
            source,
            destination,
            overwrite=overwrite,
            bitrate_kbps=self._bitrate_spin.value(),
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.log_line.connect(self._on_worker_log)
        worker.error_line.connect(self._on_worker_error_line)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._worker = worker
        self._start_worker(worker, "Library compression")

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._prog_counter.setText("Cancelling — stopping FFmpeg…")
            self._log("Compression: cancelling…", "warn")

    @Slot(str)
    def _on_worker_log(self, msg: str) -> None:
        self._log(msg, "info")

    @Slot(str)
    def _on_worker_error_line(self, msg: str) -> None:
        self._log(msg, "error")

    def _set_running(self, running: bool) -> None:
        self._estimate_btn.setEnabled(not running)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(running)
        self._source_edit.setEnabled(not running)
        self._dest_edit.setEnabled(not running)
        self._src_browse.setEnabled(not running)
        self._dst_browse.setEnabled(not running)
        self._bitrate_spin.setEnabled(not running)
        self._overwrite_cb.setEnabled(not running)
        self._prog_bar.setVisible(running)
        if running:
            self._prog_bar.setMaximum(0)  # indeterminate until first tick
            self._prog_bar.setValue(0)
            self._prog_counter.setText("Starting…")
            self._result_lbl.setText("Running…")
            set_tone(self._result_lbl, "muted")

    def _run_ended(self) -> None:
        self._worker = None
        self._set_running(False)
        # The estimate described the run that just finished; ask for a fresh one.
        self._estimate_key = None
        if self._pending_source is not None:
            path, self._pending_source = self._pending_source, None
            self._source_edit.setText(path)

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
        destination = self._worker.destination if self._worker is not None else self._dest_edit.text().strip()
        self._run_ended()
        self._prog_bar.setMaximum(100)
        self._prog_bar.setValue(100)

        counts = (
            f"Converted {summary['converted']} FLAC → Opus, "
            f"copied {summary['copied']}, "
            f"skipped {summary['skipped']}, "
            f"errors {summary['errors']}."
        )
        if summary.get("cancelled"):
            msg = (
                f"Cancelled.  {counts}  {summary.get('not_started', 0)} file(s) were not "
                "processed; no partial files were left behind."
            )
            tone, level = "warning", "warn"
        elif summary["errors"]:
            msg = f"Done with errors.  {counts}  See the report for the failed files."
            tone, level = "danger", "error"
        else:
            msg = f"Done.  {counts}"
            tone, level = "success", "ok"
        self._result_lbl.setText(msg)
        set_tone(self._result_lbl, tone)
        self._prog_counter.setText("")
        self._log(msg, level)
        for path, reason in summary.get("error_files", [])[:20]:
            self._log(f"Compression error: {path}: {reason}", "error")

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

    @Slot(str)
    def _on_error(self, error: str) -> None:
        self._run_ended()
        self._result_lbl.setText(f"Error: {error}")
        set_tone(self._result_lbl, "danger")
        self._log(f"Compression failed: {error}", "error")

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
