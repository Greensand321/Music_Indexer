"""Indexer workspace — scan → preview → execute workflow."""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


# ── Worker thread ─────────────────────────────────────────────────────────────

class IndexerWorker(QtCore.QThread):
    progress = Signal(int, str)   # (percent, phase_name)
    log_line = Signal(str)
    finished = Signal(bool, str)  # (success, message)

    def __init__(self, library_path: str, dry_run: bool, create_playlists: bool,
                 flush_cache: bool, max_workers: int) -> None:
        super().__init__()
        self.library_path = library_path
        self.dry_run = dry_run
        self.create_playlists = create_playlists
        self.flush_cache = flush_cache
        self.max_workers = max_workers
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import music_indexer_api as api
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return

        def _progress(pct: int, phase: str = "") -> None:
            if not self._cancelled:
                self.progress.emit(pct, phase)

        def _log(msg: str) -> None:
            if not self._cancelled:
                self.log_line.emit(msg)

        try:
            docs_dir = Path(self.library_path) / "Docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            output_html = str(docs_dir / "MusicIndex.html")

            api.run_full_indexer(
                self.library_path,
                output_html,
                dry_run_only=self.dry_run,
                create_playlists=self.create_playlists,
                flush_cache=self.flush_cache,
                max_workers=self.max_workers or None,
                progress_callback=_progress,
                log_callback=_log,
            )
            if self._cancelled:
                self.finished.emit(False, "Cancelled by user.")
            else:
                verb = "Preview" if self.dry_run else "Indexer"
                self.finished.emit(True, f"{verb} completed successfully.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


# ── Workspace ─────────────────────────────────────────────────────────────────

class IndexerWorkspace(WorkspaceBase):
    """Full Indexer workflow: options → run → view report."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: IndexerWorker | None = None
        self._report_path: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        # ── Header card — title + description + workflow steps ─────────────
        header_card = self._make_card()
        header_card.setObjectName("headerCard")
        hl = QtWidgets.QVBoxLayout(header_card)
        hl.setContentsMargins(20, 16, 16, 16)
        hl.setSpacing(8)
        hl.addWidget(self._make_section_title("Indexer"))
        hl.addWidget(self._make_subtitle(
            "Scan your library, preview the rename/move plan as HTML, then execute when ready. "
            "Always run a preview first — the HTML report shows exactly what will change."
        ))

        stepper = QtWidgets.QFrame()
        stepper.setObjectName("workflowStepper")
        sl = QtWidgets.QHBoxLayout(stepper)
        sl.setContentsMargins(12, 8, 12, 8)
        sl.setSpacing(4)
        for i, step in enumerate(["1. Preview (dry run)", "2. Review HTML report", "3. Execute changes"]):
            lbl = QtWidgets.QLabel(step)
            lbl.setObjectName("stepActive" if i == 0 else "stepInactive")
            sl.addWidget(lbl)
            if i < 2:
                arr = QtWidgets.QLabel("→")
                arr.setObjectName("stepArrow")
                sl.addWidget(arr)
        sl.addStretch(1)
        hl.addWidget(stepper)
        cl.addWidget(header_card)

        # ── Options card ──────────────────────────────────────────────────
        opt_card = self._make_card()
        opt_layout = QtWidgets.QVBoxLayout(opt_card)
        opt_layout.setContentsMargins(16, 16, 16, 16)
        opt_layout.setSpacing(12)
        opt_layout.addWidget(self._make_card_title("Options"))

        row1 = QtWidgets.QHBoxLayout()
        self._dry_run_cb = QtWidgets.QCheckBox("Dry run (preview only — no files moved)")
        self._dry_run_cb.setChecked(True)
        self._dry_run_cb.setToolTip(
            "When checked, generates Docs/MusicIndex.html and Docs/indexer_log.txt "
            "without moving or renaming any files."
        )
        row1.addWidget(self._dry_run_cb)
        row1.addStretch(1)
        opt_layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self._playlists_cb = QtWidgets.QCheckBox("Create playlists after execution")
        self._playlists_cb.setChecked(True)
        self._cross_album_cb = QtWidgets.QCheckBox("Cross-album scan (Phase C)")
        self._cross_album_cb.setToolTip(
            "Validates duplicates that span multiple albums. Slower but more thorough."
        )
        row2.addWidget(self._playlists_cb)
        row2.addWidget(self._cross_album_cb)
        row2.addStretch(1)
        opt_layout.addLayout(row2)

        # Advanced collapsible
        adv_group = QtWidgets.QGroupBox("Advanced")
        adv_group.setCheckable(True)
        adv_group.setChecked(False)
        adv_form = QtWidgets.QFormLayout(adv_group)
        adv_form.setContentsMargins(8, 4, 8, 8)

        self._flush_cache_cb = QtWidgets.QCheckBox("Flush fingerprint cache before scan")
        adv_form.addRow(self._flush_cache_cb)

        self._max_workers_spin = QtWidgets.QSpinBox()
        self._max_workers_spin.setRange(1, 32)
        self._max_workers_spin.setValue(4)
        self._max_workers_spin.setToolTip("Number of parallel worker threads for scanning.")
        adv_form.addRow("Max workers:", self._max_workers_spin)

        opt_layout.addWidget(adv_group)
        cl.addWidget(opt_card)

        # ── Action card ───────────────────────────────────────────────────
        action_card = self._make_card()
        action_card.setObjectName("actionCard")
        action_layout = QtWidgets.QVBoxLayout(action_card)
        action_layout.setContentsMargins(16, 14, 16, 14)
        action_layout.setSpacing(10)
        action_layout.addWidget(self._make_card_title("Actions"))

        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = self._make_primary_button("▶  Run Preview")
        self._run_btn.setMinimumWidth(140)
        self._run_btn.clicked.connect(self._on_run)

        self._report_btn = QtWidgets.QPushButton("📄  Open Report")
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._on_open_report)

        self._cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)

        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._report_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        action_layout.addLayout(btn_row)
        cl.addWidget(action_card)

        # ── Progress card ─────────────────────────────────────────────────
        prog_card = self._make_card()
        prog_layout = QtWidgets.QVBoxLayout(prog_card)
        prog_layout.setContentsMargins(16, 16, 16, 16)
        prog_layout.setSpacing(10)
        prog_layout.addWidget(self._make_card_title("Progress"))

        self._phase_bars: list[tuple[QtWidgets.QLabel, QtWidgets.QProgressBar]] = []
        phases = [
            ("Phase A — Scanning", "Index folders and gather metadata"),
            ("Phase B — Planning", "Build rename/move plan and HTML report"),
            ("Phase C — Cross-album", "Validate duplicates across albums"),
        ]
        for title, desc in phases:
            phase_lbl = QtWidgets.QLabel(title)
            phase_lbl.setObjectName("phaseLabel")
            desc_lbl = QtWidgets.QLabel(desc)
            desc_lbl.setObjectName("phaseDesc")
            bar = QtWidgets.QProgressBar()
            bar.setValue(0)
            bar.setFixedHeight(6)
            bar.setTextVisible(False)
            prog_layout.addWidget(phase_lbl)
            prog_layout.addWidget(desc_lbl)
            prog_layout.addWidget(bar)
            self._phase_bars.append((phase_lbl, bar))

        self._status_lbl = QtWidgets.QLabel("Ready to preview.")
        self._status_lbl.setObjectName("statusHint")
        prog_layout.addWidget(self._status_lbl)
        cl.addWidget(prog_card)

        # ── Log card ──────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.setSpacing(6)
        log_layout.addWidget(self._make_card_title("Run log"))

        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(160)
        self._log_area.setPlaceholderText("Output will appear here…")
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        # ── Output notes ──────────────────────────────────────────────────
        notes = QtWidgets.QLabel(
            "Outputs: Docs/MusicIndex.html  ·  Docs/indexer_log.txt  ·  "
            "Manual Review/ for missing metadata  ·  Playlists/ when enabled"
        )
        notes.setObjectName("notesHint")
        notes.setWordWrap(True)
        cl.addWidget(notes)

        cl.addStretch(1)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return

        dry_run = self._dry_run_cb.isChecked()
        verb = "preview" if dry_run else "execute"

        if not dry_run:
            reply = QtWidgets.QMessageBox.question(
                self, "Confirm Execute",
                "This will move and rename files in your library.\n\n"
                "Have you reviewed the preview report first?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self._log_area.clear()
        self._reset_progress()
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._report_btn.setEnabled(False)
        self._status_lbl.setText(f"Running {verb}…")
        self._log(f"Starting indexer {verb}…", "info")
        self.status_changed.emit("Running…", "#f59e0b")

        # Update run button label based on mode
        self._run_btn.setText("▶  Run Preview" if dry_run else "▶  Execute")

        self._worker = IndexerWorker(
            library_path=self._library_path,
            dry_run=dry_run,
            create_playlists=self._playlists_cb.isChecked(),
            flush_cache=self._flush_cache_cb.isChecked(),
            max_workers=self._max_workers_spin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._status_lbl.setText("Cancelling…")
            self._cancel_btn.setEnabled(False)

    @Slot()
    def _on_open_report(self) -> None:
        if self._report_path and Path(self._report_path).exists():
            webbrowser.open(f"file://{self._report_path}")
        else:
            # Try to find it
            candidate = Path(self._library_path) / "Docs" / "MusicIndex.html"
            if candidate.exists():
                webbrowser.open(f"file://{candidate}")
            else:
                QtWidgets.QMessageBox.information(
                    self, "No Report", "No report found. Run a preview first."
                )

    @Slot(int, str)
    def _on_progress(self, pct: int, phase: str) -> None:
        phase_map = {"A": 0, "B": 1, "C": 2}
        idx = phase_map.get(phase[:1], 0) if phase else 0
        if idx < len(self._phase_bars):
            self._phase_bars[idx][1].setValue(pct)
        self._status_lbl.setText(f"Phase {phase}: {pct}%" if phase else f"{pct}%")

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self._log_area.appendPlainText(line)
        self._log(line)

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._status_lbl.setText(message)

        if success:
            self._log(message, "ok")
            self.status_changed.emit("Done", "#22c55e")
            # Set all bars to 100
            for _, bar in self._phase_bars:
                bar.setValue(100)
            # Enable report button
            candidate = Path(self._library_path) / "Docs" / "MusicIndex.html"
            if candidate.exists():
                self._report_path = str(candidate)
                self._report_btn.setEnabled(True)
            QtWidgets.QMessageBox.information(self, "Done", message)
        else:
            self._log(message, "error")
            self.status_changed.emit("Error", "#ef4444")
            if message and message != "Cancelled by user.":
                QtWidgets.QMessageBox.critical(self, "Error", message)

        self._worker = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _reset_progress(self) -> None:
        for _, bar in self._phase_bars:
            bar.setValue(0)

    def _on_library_changed(self, path: str) -> None:
        self._status_lbl.setText("Ready to preview.")
        self._reset_progress()
        self._log_area.clear()
        self._report_btn.setEnabled(False)
        # Check for existing report
        candidate = Path(path) / "Docs" / "MusicIndex.html"
        if candidate.exists():
            self._report_path = str(candidate)
            self._report_btn.setEnabled(True)
