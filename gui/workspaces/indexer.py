"""Indexer workspace — scan → preview → execute workflow.

The preview computes one ``music_indexer_api.IndexerPlan``; Execute applies
*that* plan (never a fresh recomputation) and is only available while a
preview for the current library exists. The backend refuses a plan whose
library changed since the preview.
"""
from __future__ import annotations

import os
import threading
import time
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

_STEPS = ("1. Preview (dry run)", "2. Review HTML report", "3. Execute changes")
_RUN_PREVIEW_TEXT = "▶  Run Preview"
_RUN_EXECUTE_TEXT = "▶  Execute Previewed Plan"
_IGNORED_TIP = (
    "Not used: the current indexer engine ignores this option "
    "(kept visible until the redesign decides its fate)."
)


def _chip_colour(kind: str) -> str:
    """Status-chip colour from the active theme ("running" | "ok" | "error" | other)."""
    t = get_manager().current
    return {"running": t.warning, "ok": t.success, "error": t.danger}.get(kind, t.text_muted)


# ── Worker thread ─────────────────────────────────────────────────────────────

class IndexerWorker(QtCore.QThread):
    """Runs a preview (computes the plan) or an execute (applies a given plan).

    Cancellation uses the worker's own event, so a Cancel clicked before the
    thread gets going is never lost (the old code cleared a process-global
    flag at the top of ``run()``).
    """

    progress = Signal(int, int, str)   # (current, total, phase) — total 0 = unknown
    log_line = Signal(str)
    finished = Signal(bool, str)       # (success, message)

    PREVIEW = "preview"
    EXECUTE = "execute"

    def __init__(self, library_path: str, mode: str = PREVIEW, plan=None,
                 create_playlists: bool = True) -> None:
        super().__init__()
        self.library_path = library_path
        self.mode = mode
        self.plan = plan                 # execute: input; preview: result
        self.create_playlists = create_playlists
        self.summary: dict | None = None
        self.report_path = ""
        self.outcome = ""                # "ok" | "cancelled" | "stale" | "partial" | "error"
        self._cancel_event = threading.Event()
        self._last_emit = 0.0

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    # ── run ────────────────────────────────────────────────────────────────
    def run(self) -> None:
        try:
            import music_indexer_api as api
            from indexer_control import IndexCancelled
        except Exception as exc:  # noqa: BLE001
            self.outcome = "error"
            self.finished.emit(False, f"Import error: {exc}")
            return

        def _emit_progress(current: int, total: int, phase: str) -> None:
            now = time.monotonic()
            if (total and current >= total) or now - self._last_emit >= 0.05:
                self._last_emit = now
                self.progress.emit(int(current), int(total or 0), str(phase or ""))

        def _progress(current: int, total: int, _message: str = "", phase: str = "A") -> None:
            if self._cancel_event.is_set():
                raise IndexCancelled()
            _emit_progress(current, total, phase)

        def _log(msg: str) -> None:
            self.log_line.emit(str(msg))

        try:
            if self.mode == self.EXECUTE:
                self._execute(api, _log, _emit_progress)
            else:
                self._preview(api, _log, _progress)
        except IndexCancelled:
            self.outcome = "cancelled"
            self.finished.emit(False, "Cancelled by user.")
        except api.StalePlanError as exc:
            self.outcome = "stale"
            self.finished.emit(False, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.outcome = "error"
            self.finished.emit(False, str(exc) or type(exc).__name__)

    def _preview(self, api, log, progress) -> None:
        docs = os.path.join(self.library_path, "Docs")
        os.makedirs(docs, exist_ok=True)
        output_html = os.path.join(docs, "MusicIndex.html")
        result = api.run_full_indexer(
            self.library_path,
            output_html,
            dry_run_only=True,
            log_callback=log,
            progress_callback=progress,
            cancel_event=self._cancel_event,
        )
        if self._cancel_event.is_set():
            self.outcome = "cancelled"
            self.finished.emit(False, "Cancelled by user.")
            return
        self.plan = result["plan"]
        self.report_path = output_html
        c = self.plan.counts()
        if not c["audio_files"]:
            msg = "Preview complete — no audio files were found in this library, so there is nothing to change."
        else:
            msg = (
                f"Preview ready — {c['moves']} file(s) to move or rename, {c['unchanged']} already "
                f"in place, {c['manual_review']} to Manual Review, {c['leftovers']} non-audio "
                f"file(s) to Trash/Docs, {c['empty_dirs']} empty folder(s) to remove."
            )
        self.outcome = "ok"
        self.finished.emit(True, msg)

    def _execute(self, api, log, emit_progress) -> None:
        def _progress(current: int, total: int, _message: str = "", phase: str = "C") -> None:
            # Never raise mid-execute: the backend stops at the next file and
            # still writes its Docs/ log when cancelled.
            emit_progress(current, total, phase)

        summary = api.apply_indexer_plan(
            self.plan,
            log_callback=log,
            progress_callback=_progress,
            create_playlists=self.create_playlists,
            cancel_event=self._cancel_event,
        )
        self.summary = summary
        moved = summary.get("moved", 0)
        leftovers = summary.get("leftovers_moved", 0)
        removed = len(summary.get("removed_dirs", []))
        errors = summary.get("errors", [])
        log_path = summary.get("log_path") or "Docs/"
        if summary.get("cancelled"):
            self.outcome = "cancelled"
            self.finished.emit(
                False,
                f"Execution cancelled after {moved} file(s) were moved. "
                f"Everything done so far is listed in {log_path}.",
            )
        elif errors:
            self.outcome = "partial"
            self.finished.emit(
                False,
                f"Execution finished with {len(errors)} problem(s): {moved} file(s) moved, "
                f"{leftovers} leftover(s) moved. Files that could not be moved were left in "
                f"place (nothing was overwritten). First problem: {errors[0]}. "
                f"Full log: {log_path}",
            )
        else:
            self.outcome = "ok"
            self.finished.emit(
                True,
                f"Execution complete — {moved} file(s) moved or renamed, {leftovers} non-audio "
                f"file(s) moved to Trash/Docs, {removed} empty folder(s) removed. Log: {log_path}",
            )


# ── Workspace ─────────────────────────────────────────────────────────────────

class IndexerWorkspace(WorkspaceBase):
    """Full Indexer workflow: options → preview → review → execute that preview."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: IndexerWorker | None = None
        self._report_path: str = ""
        self._plan = None              # IndexerPlan from the last successful preview
        self._plan_library: str = ""   # library the plan belongs to
        self._step_labels: list[QtWidgets.QLabel] = []
        self._current_step = 0
        self._build_ui()
        self._refresh_controls()

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
            "Execute applies exactly the previewed plan — run a preview first."
        ))

        stepper = QtWidgets.QFrame()
        stepper.setObjectName("workflowStepper")
        sl = QtWidgets.QHBoxLayout(stepper)
        sl.setContentsMargins(12, 8, 12, 8)
        sl.setSpacing(4)
        for i, step in enumerate(_STEPS):
            lbl = QtWidgets.QLabel(step)
            lbl.setObjectName("stepActive" if i == 0 else "stepInactive")
            sl.addWidget(lbl)
            self._step_labels.append(lbl)
            if i < len(_STEPS) - 1:
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
            "without moving or renaming any files. Uncheck to execute the last preview."
        )
        self._dry_run_cb.toggled.connect(self._on_mode_toggled)
        row1.addWidget(self._dry_run_cb)
        row1.addStretch(1)
        opt_layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self._playlists_cb = QtWidgets.QCheckBox("Create playlists after execution")
        self._playlists_cb.setChecked(True)
        self._cross_album_cb = QtWidgets.QCheckBox("Cross-album scan (Phase C)")
        self._cross_album_cb.setEnabled(False)
        self._cross_album_cb.setToolTip(_IGNORED_TIP)
        row2.addWidget(self._playlists_cb)
        row2.addWidget(self._cross_album_cb)
        row2.addStretch(1)
        opt_layout.addLayout(row2)

        # Advanced options. The engine ignores both, so they are shown disabled
        # (a checkable group box would re-enable them when ticked).
        adv_group = QtWidgets.QGroupBox("Advanced (not used by the current engine)")
        adv_form = QtWidgets.QFormLayout(adv_group)
        adv_form.setContentsMargins(8, 4, 8, 8)

        self._flush_cache_cb = QtWidgets.QCheckBox("Flush fingerprint cache before scan")
        self._flush_cache_cb.setEnabled(False)
        self._flush_cache_cb.setToolTip(_IGNORED_TIP)
        adv_form.addRow(self._flush_cache_cb)

        self._max_workers_spin = QtWidgets.QSpinBox()
        self._max_workers_spin.setRange(1, 32)
        self._max_workers_spin.setValue(4)
        self._max_workers_spin.setEnabled(False)
        self._max_workers_spin.setToolTip(_IGNORED_TIP)
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
        self._run_btn = self._make_primary_button(_RUN_PREVIEW_TEXT)
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
            ("Phase A — Scanning", "Find audio files (reserved folders are skipped)"),
            ("Phase B — Planning", "Read metadata, build the rename/move plan and HTML report"),
            ("Phase C — Applying changes", "Move files into place (Execute only)"),
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
        self._status_lbl.setWordWrap(True)
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
            "Docs/indexer_executed_*.txt/.json (after Execute)  ·  "
            "Manual Review/ for missing metadata  ·  Playlists/ when enabled"
        )
        notes.setObjectName("notesHint")
        notes.setWordWrap(True)
        cl.addWidget(notes)

        cl.addStretch(1)

    # ── State helpers ─────────────────────────────────────────────────────

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _fresh_plan(self):
        """The previewed plan if it belongs to the current library, else None."""
        if self._plan is not None and self._plan_library == self._library_path:
            return self._plan
        return None

    def current_step(self) -> int:
        return self._current_step

    def _set_step(self, index: int) -> None:
        self._current_step = index
        for i, lbl in enumerate(self._step_labels):
            name = "stepActive" if i == index else "stepInactive"
            if lbl.objectName() != name:
                lbl.setObjectName(name)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)

    def _refresh_controls(self) -> None:
        busy = self._busy()
        dry = self._dry_run_cb.isChecked()
        plan = self._fresh_plan()
        self._dry_run_cb.setEnabled(not busy)
        if dry:
            self._run_btn.setText(_RUN_PREVIEW_TEXT)
            self._run_btn.setEnabled(not busy)
            self._run_btn.setToolTip("Compute the plan and write the HTML preview. No file is moved.")
        else:
            self._run_btn.setText(_RUN_EXECUTE_TEXT)
            self._run_btn.setEnabled(not busy and plan is not None)
            self._run_btn.setToolTip(
                "Apply exactly the last preview to this library."
                if plan is not None else
                "Run a preview first — Execute applies the previewed plan."
            )
        if not busy:
            if plan is None:
                self._set_step(0)
            else:
                self._set_step(1 if dry else 2)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot(bool)
    def _on_mode_toggled(self, _checked: bool) -> None:
        if not self._busy() and not self._dry_run_cb.isChecked() and self._fresh_plan() is None:
            self._status_lbl.setText("Run a preview first — Execute applies the previewed plan.")
        self._refresh_controls()

    @Slot()
    def _on_run(self) -> None:
        if self._busy():
            return
        path = self._library_path
        if not path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        if not os.path.isdir(path):
            QtWidgets.QMessageBox.warning(
                self, "Library Not Found",
                f"The library folder does not exist:\n{path}\n\nChoose an existing folder.",
            )
            return

        if self._dry_run_cb.isChecked():
            self._start(IndexerWorker(path, IndexerWorker.PREVIEW), "preview")
            return

        plan = self._fresh_plan()
        if plan is None:
            QtWidgets.QMessageBox.information(
                self, "Preview Required",
                "Run a preview first. Execute applies exactly the previewed plan.",
            )
            self._refresh_controls()
            return
        c = plan.counts()
        create_playlists = self._playlists_cb.isChecked()
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Execute",
            f"Apply the previewed plan to:\n{path}\n\n"
            f"• {c['moves']} audio file(s) will be moved or renamed "
            f"({c['manual_review']} into Manual Review/)\n"
            f"• {c['leftovers']} non-audio file(s) will be moved into Trash/ and Docs/ "
            "(folder structure kept)\n"
            f"• {c['empty_dirs']} emptied folder(s) will be removed\n"
            f"• Playlists: {'regenerated' if create_playlists else 'existing ones updated only'}\n\n"
            "Nothing else is touched, no file is overwritten, and every action is "
            "logged to Docs/. If the library changed since the preview, nothing is moved.\n\n"
            "Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._start(
            IndexerWorker(path, IndexerWorker.EXECUTE, plan=plan, create_playlists=create_playlists),
            "execute",
        )

    def _start(self, worker: IndexerWorker, verb: str) -> None:
        self._log_area.clear()
        self._reset_progress()
        self._cancel_btn.setEnabled(True)
        self._report_btn.setEnabled(False)
        self._status_lbl.setText(f"Running {verb}…")
        self._log(f"Starting indexer {verb}…", "info")
        self.status_changed.emit("Running…", _chip_colour("running"))
        if verb == "execute":
            self._set_step(2)
        else:
            self._set_step(0)

        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.log_line.connect(self._on_log_line)
        worker.finished.connect(self._on_finished)
        self._start_worker(worker, f"Indexer {verb}")
        self._refresh_controls()

    @Slot()
    def _on_cancel(self) -> None:
        if self._busy():
            self._worker.cancel()
            self._status_lbl.setText("Cancelling…")
            self._cancel_btn.setEnabled(False)

    @Slot()
    def _on_open_report(self) -> None:
        candidates = [self._report_path] if self._report_path else []
        if self._library_path:
            candidates.append(str(Path(self._library_path) / "Docs" / "MusicIndex.html"))
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                webbrowser.open(Path(candidate).resolve().as_uri())
                return
        QtWidgets.QMessageBox.information(
            self, "No Report", "No report found. Run a preview first."
        )

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, phase: str) -> None:
        if self.sender() is not None and self.sender() is not self._worker:
            return
        idx = {"A": 0, "B": 1, "C": 2}.get((phase or "A")[:1].upper(), 0)
        for earlier in range(idx):
            self._phase_bars[earlier][1].setValue(100)
        bar = self._phase_bars[idx][1]
        if total:
            pct = max(0, min(100, int(current * 100 / total)))
            bar.setValue(pct)
            self._status_lbl.setText(f"Phase {phase}: {pct}%")
        else:
            self._status_lbl.setText(f"Phase {phase}: {current} file(s) found…")

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        if self.sender() is not None and self.sender() is not self._worker:
            return
        self._log_area.appendPlainText(line)
        self._log(line)

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str) -> None:
        worker = self.sender() if isinstance(self.sender(), IndexerWorker) else self._worker
        if worker is None or worker is not self._worker:
            return  # a run abandoned after a library change
        self._worker = None
        self._cancel_btn.setEnabled(False)
        self._status_lbl.setText(message)
        outcome = worker.outcome or ("ok" if success else "error")
        same_library = worker.library_path == self._library_path

        if worker.mode == IndexerWorker.PREVIEW:
            if success and same_library:
                self._plan = worker.plan
                self._plan_library = worker.library_path
                self._report_path = worker.report_path
                self._report_btn.setEnabled(bool(self._report_path))
                for _, bar in self._phase_bars[:2]:
                    bar.setValue(100)
                self._log(message, "ok")
                self.status_changed.emit("Preview ready", _chip_colour("ok"))
                QtWidgets.QMessageBox.information(self, "Preview Ready", message)
            elif outcome == "cancelled":
                self._log("Preview cancelled.", "warn")
                self.status_changed.emit("Cancelled", _chip_colour("neutral"))
            else:
                self._log(message, "error")
                self.status_changed.emit("Error", _chip_colour("error"))
                QtWidgets.QMessageBox.critical(self, "Preview Failed", message)
        else:
            # A plan is applied at most once: whatever happened, the library
            # has (possibly) changed, so a new preview is needed.
            self._plan = None
            self._plan_library = ""
            if self._library_path and (Path(self._library_path) / "Docs" / "MusicIndex.html").exists():
                self._report_btn.setEnabled(True)
            if outcome == "ok":
                for _, bar in self._phase_bars:
                    bar.setValue(100)
                self._log(message, "ok")
                self.status_changed.emit("Done", _chip_colour("ok"))
                self._dry_run_cb.setChecked(True)   # next click previews again
                QtWidgets.QMessageBox.information(self, "Execution Complete", message)
                self._set_step(2)
            elif outcome == "cancelled":
                self._log(message, "warn")
                self.status_changed.emit("Cancelled", _chip_colour("neutral"))
                QtWidgets.QMessageBox.information(self, "Execution Cancelled", message)
            elif outcome in ("stale", "partial"):
                self._log(message, "warn")
                self.status_changed.emit("Needs attention", _chip_colour("running"))
                QtWidgets.QMessageBox.warning(
                    self, "Library Changed" if outcome == "stale" else "Execution Problems", message
                )
            else:
                self._log(message, "error")
                self.status_changed.emit("Error", _chip_colour("error"))
                QtWidgets.QMessageBox.critical(self, "Execution Failed", message)
        self._refresh_controls()
        if worker.mode == IndexerWorker.EXECUTE and outcome == "ok":
            self._set_step(2)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _reset_progress(self) -> None:
        for _, bar in self._phase_bars:
            bar.setValue(0)

    def _on_library_changed(self, path: str) -> None:
        running = self._worker if self._busy() else None
        if running is not None and running.mode == IndexerWorker.PREVIEW:
            # A preview of the old library is useless now: stop it and ignore
            # its result (the JobRegistry keeps the thread alive until it ends).
            running.cancel()
            self._worker = None
            self._cancel_btn.setEnabled(False)
            self._log("Library changed — the running preview was cancelled.", "warn")
            running = None
        elif running is not None:
            # Never interrupt file moves because the user navigated: the
            # execute finishes (or is cancelled explicitly) on its own library.
            self._log(
                f"Library changed — the running execute continues on {running.library_path}.",
                "warn",
            )

        self._plan = None
        self._plan_library = ""
        self._report_path = ""
        self._report_btn.setEnabled(False)
        if running is None:
            self._status_lbl.setText("Ready to preview.")
            self._reset_progress()
            self._log_area.clear()
            candidate = Path(path) / "Docs" / "MusicIndex.html" if path else None
            if candidate is not None and candidate.exists():
                self._report_path = str(candidate)
                self._report_btn.setEnabled(True)
            self._dry_run_cb.setChecked(True)
        self._refresh_controls()
