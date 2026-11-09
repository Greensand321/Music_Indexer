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
from gui.themes import motion
from gui.themes.glass import apply_font
from gui.themes.manager import get_manager
from gui.widgets.glass import ElidedLabel, GlassPanel, path_label
from gui.widgets.glass_parts import (
    EmptyState, StatRow, StepPills, caption, glyph_button, hero_title, path_well, thin_bar,
)
from gui.workspaces.base import WorkspaceBase
from gui.widgets.layouts import FlowLayout

# The step pills show real state: no preview → 0, a preview of this library
# → 1 (review its report), Execute selected or running → 2.
_STEPS = ("Preview", "Review the report", "Execute")
_RUN_PREVIEW_TEXT = "Run preview"
_RUN_EXECUTE_TEXT = "Execute previewed plan"
_PHASES = (
    ("Phase A — Scanning", "Find audio files (reserved folders are skipped)."),
    ("Phase B — Planning", "Read tags, build the rename/move plan and the HTML report."),
    ("Phase C — Applying changes", "Move files into place (Execute only)."),
)
# Results tiles: the previewed plan, then (relabelled) what the execute did.
# Labels stay short (five tiles share 560 px at the minimum window); each
# tile's tooltip says it in full.
_PLAN_TILES = (
    ("moves", "to move"),
    ("unchanged", "in place"),
    ("manual_review", "untagged"),
    ("leftovers", "non-audio"),
    ("empty_dirs", "empty folders"),
)
_PLAN_TIPS = {
    "moves": "Audio files the plan moves or renames.",
    "unchanged": "Audio files already where they belong.",
    "manual_review": "Files missing the tags the layout needs: they go to Manual Review/.",
    "leftovers": "Non-audio files swept into Trash/ or Docs/ (their folders are kept).",
    "empty_dirs": "Folders left empty by the moves, removed afterwards.",
}
_EXECUTE_LABELS = {
    "moves": "moved",
    "unchanged": "in place",
    "manual_review": "problems",
    "leftovers": "tidied",
    "empty_dirs": "removed",
}
_EXECUTE_TIPS = {
    "moves": "Audio files moved or renamed (Manual Review/ included).",
    "unchanged": "Audio files that were already where they belong.",
    "manual_review": "Files that could not be moved; they were left in place, nothing was "
                     "overwritten. The status line names the first one.",
    "leftovers": "Non-audio files moved into Trash/ or Docs/.",
    "empty_dirs": "Empty folders removed.",
}
_NO_LIBRARY = "No library selected. Choose one with Change library."
_NO_PREVIEW = (
    "No preview yet",
    "Run preview to see how many files would move, which go to Manual Review and "
    "what gets tidied. Nothing changes until you execute.",
)
_PHASE_C_TIP = (
    "Not used: the current indexer engine has no cross-album phase, so it ignores "
    "this option (kept visible until the redesign decides its fate)."
)
_FLUSH_TIP = (
    "Not used: the Indexer computes no fingerprints and keeps no cache, so there is "
    "nothing to flush. Each preview always reads every file's tags fresh."
)
_WORKERS_TIP = (
    "How many files have their tags read at once during a preview.\n"
    "1 is fastest on a local disk (tag reading is CPU-bound). Raise it for a "
    "network share or a slow external drive, where each read waits on the device."
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
                 create_playlists: bool = True, max_workers: int = 1) -> None:
        super().__init__()
        self.library_path = library_path
        self.mode = mode
        self.max_workers = max(1, int(max_workers or 1))
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
            max_workers=self.max_workers,
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
        self._current_step = 0
        self._build_ui()
        self._refresh_controls()

    def _build_ui(self) -> None:
        cl = self.content_layout

        # ── Hero: the library, the mode and the main action ───────────────
        hero = self._make_card()
        hero.setObjectName("headerCard")
        hl = QtWidgets.QVBoxLayout(hero)
        hl.setContentsMargins(16, 16, 16, 16)
        hl.setSpacing(12)
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(4)
        title_box.addWidget(hero_title("Organize this library"))
        title_box.addWidget(self._make_subtitle(
            "Rename and file every track into By Artist/ and By Year/. Preview first: "
            "nothing moves until you execute the plan you reviewed."
        ))
        hl.addLayout(title_box)

        self._steps = StepPills(_STEPS)
        hl.addWidget(self._steps)

        self._library_lbl = path_label("", _NO_LIBRARY)
        self._library_lbl.setObjectName("indexerLibraryPath")
        hl.addWidget(path_well(self._library_lbl))
        self._show_library(self._library_path)

        self._dry_run_cb = QtWidgets.QCheckBox("Dry run: preview only, nothing is moved")
        self._dry_run_cb.setChecked(True)
        self._dry_run_cb.setToolTip(
            "When checked, Run writes Docs/MusicIndex.html and Docs/indexer_log.txt "
            "without moving or renaming any file. Uncheck to execute the last preview."
        )
        self._dry_run_cb.toggled.connect(self._on_mode_toggled)
        hl.addWidget(self._dry_run_cb)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)
        self._run_btn = glyph_button(_RUN_PREVIEW_TEXT, "play", object_name="primaryBtn")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = glyph_button("Cancel", "close")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._report_btn = glyph_button(
            "Open report", "external",
            tooltip="Open the HTML preview (Docs/MusicIndex.html) in your browser.",
        )
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._on_open_report)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._report_btn)
        hl.addLayout(btn_row)
        cl.addWidget(hero)

        # ── Options ───────────────────────────────────────────────────────
        opt_card = self._make_card()
        ol = QtWidgets.QVBoxLayout(opt_card)
        ol.setContentsMargins(16, 16, 16, 16)
        ol.setSpacing(12)
        ol.addWidget(self._make_card_title("Options"))

        self._playlists_cb = QtWidgets.QCheckBox("Create playlists after execution")
        self._playlists_cb.setChecked(True)
        self._playlists_cb.setToolTip(
            "Checked: regenerate playlists after the moves. Unchecked: only update the "
            "paths in existing playlists."
        )
        ol.addWidget(self._playlists_cb)

        # Advanced. Phase C and Flush stay visible but disabled (the engine
        # ignores them), each saying why in its tooltip.
        adv = GlassPanel(surface="well", radius=16)
        adv.setObjectName("indexerAdvanced")
        al = QtWidgets.QVBoxLayout(adv)
        al.setContentsMargins(14, 12, 14, 12)
        al.setSpacing(8)
        al.addWidget(caption("Advanced"))
        flow = FlowLayout(spacing=24, line_spacing=8)
        self._cross_album_cb = QtWidgets.QCheckBox("Cross-album scan (Phase C)")
        self._cross_album_cb.setEnabled(False)
        self._cross_album_cb.setToolTip(_PHASE_C_TIP)
        flow.addWidget(self._cross_album_cb)
        self._flush_cache_cb = QtWidgets.QCheckBox("Flush fingerprint cache before scan")
        self._flush_cache_cb.setEnabled(False)
        self._flush_cache_cb.setToolTip(_FLUSH_TIP)
        flow.addWidget(self._flush_cache_cb)
        workers = QtWidgets.QWidget()
        workers_row = QtWidgets.QHBoxLayout(workers)
        workers_row.setContentsMargins(0, 0, 0, 0)
        workers_row.setSpacing(10)
        workers_lbl = QtWidgets.QLabel("Max workers")
        workers_lbl.setToolTip(_WORKERS_TIP)
        self._max_workers_spin = QtWidgets.QSpinBox()
        self._max_workers_spin.setRange(1, 32)
        self._max_workers_spin.setValue(1)
        self._max_workers_spin.setToolTip(_WORKERS_TIP)
        self._max_workers_spin.setFixedWidth(96)
        workers_lbl.setBuddy(self._max_workers_spin)
        workers_row.addWidget(workers_lbl)
        workers_row.addWidget(self._max_workers_spin)
        flow.addWidget(workers)
        al.addLayout(flow)
        ol.addWidget(adv)
        cl.addWidget(opt_card)
        # ── Progress and results ──────────────────────────────────────────
        res_card = self._make_card()
        rl = QtWidgets.QVBoxLayout(res_card)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(12)
        rl.addWidget(self._make_card_title("Progress and results"))

        # One thin pill per phase, side by side (Phase C only runs on Execute).
        phases = QtWidgets.QHBoxLayout()
        phases.setSpacing(12)
        self._phase_bars: list[tuple[QtWidgets.QLabel, QtWidgets.QProgressBar]] = []
        for title, desc in _PHASES:
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(6)
            bar = thin_bar()
            bar.setToolTip(desc)
            phase_lbl = ElidedLabel(title)
            phase_lbl.setObjectName("mutedLabel")
            phase_lbl.setToolTip(f"{title}: {desc}")
            apply_font(phase_lbl, "caption")
            col.addWidget(bar)
            col.addWidget(phase_lbl)
            phases.addLayout(col, 1)
            self._phase_bars.append((phase_lbl, bar))
        rl.addLayout(phases)

        self._status_lbl = QtWidgets.QLabel("Ready to preview.")
        self._status_lbl.setObjectName("statusHint")
        self._status_lbl.setWordWrap(True)
        rl.addWidget(self._status_lbl)

        self._results_empty = EmptyState(*_NO_PREVIEW)
        self._results_empty.setObjectName("indexerResultsEmpty")
        rl.addWidget(self._results_empty)

        self._stats_box = QtWidgets.QWidget()
        sb = QtWidgets.QVBoxLayout(self._stats_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(8)
        self._stats_caption = caption("Previewed plan")
        sb.addWidget(self._stats_caption)
        self._stats = StatRow(list(_PLAN_TILES), spacing=8)
        self._stats.setObjectName("indexerStats")
        sb.addWidget(self._stats)
        self._stats_box.setVisible(False)
        rl.addWidget(self._stats_box)
        cl.addWidget(res_card)

        # ── Run log ───────────────────────────────────────────────────────
        log_card = self._make_card()
        ll = QtWidgets.QVBoxLayout(log_card)
        ll.setContentsMargins(16, 16, 16, 16)
        ll.setSpacing(8)
        ll.addWidget(self._make_card_title("Run log"))
        notes = caption(
            "Written to Docs/: MusicIndex.html and indexer_log.txt on preview, "
            "indexer_executed_*.txt / .json after Execute. Files missing tags go to "
            "Manual Review/; playlists to Playlists/ when enabled."
        )
        notes.setWordWrap(True)
        ll.addWidget(notes)
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(160)
        self._log_area.setPlaceholderText("Each run's log lines appear here.")
        self._log_area.setObjectName("monoText")
        ll.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── Results ───────────────────────────────────────────────────────────

    def _show_library(self, path: str) -> None:
        self._library_lbl.setText(path or "")

    def _show_results_empty(self, title: str, message: str) -> None:
        self._results_empty.set_text(title, message)
        self._results_empty.setVisible(True)
        self._stats_box.setVisible(False)

    def _show_stats(self, caption_text: str, values: dict, labels: dict | None = None,
                    tones: dict | None = None, tips: dict | None = None) -> None:
        labels = labels or {}
        tones = tones or {}
        tips = tips or _PLAN_TIPS
        self._stats_caption.setText(caption_text)
        for key, default_label in _PLAN_TILES:
            tile = self._stats.tile(key)
            tile.set_label(labels.get(key, default_label))
            tile.set_value(values.get(key, "–"), tone=tones.get(key, ""))
            tile.setToolTip(tips.get(key, ""))
        was_hidden = self._stats_box.isHidden()
        self._results_empty.setVisible(False)
        self._stats_box.setVisible(True)
        if was_hidden and self._stats_box.isVisible():
            motion.reveal(self._stats_box)

    def stats(self) -> dict[str, tuple[str, str]]:
        """``{key: (value, label)}`` of the results tiles (tests, accessibility)."""
        return {k: (self._stats.tile(k).value(), self._stats.tile(k).label())
                for k, _ in _PLAN_TILES}

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
        self._steps.set_current(index)

    def _refresh_controls(self) -> None:
        busy = self._busy()
        dry = self._dry_run_cb.isChecked()
        plan = self._fresh_plan()
        self._dry_run_cb.setEnabled(not busy)
        self._max_workers_spin.setEnabled(not busy)
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
            self._start(
                IndexerWorker(path, IndexerWorker.PREVIEW,
                              max_workers=self._max_workers_spin.value()),
                "preview",
            )
            return

        plan = self._fresh_plan()
        if plan is None:
            QtWidgets.QMessageBox.information(
                self, "Preview Required",
                "Run a preview first. Execute applies exactly the previewed plan.",
            )
            self._refresh_controls()
            return
        if self._library_writes_blocked("Indexer execute"):
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
        self._start_worker(worker, f"Indexer {verb}",
                           ask_before_quit=verb == "execute",
                           library_write=verb == "execute")
        self._refresh_controls()

    @Slot()
    def _on_cancel(self) -> None:
        # Not just while the thread runs: a worker whose thread has finished
        # but whose result is still queued must honour the click too
        # (_on_finished drops a preview that was cancelled this late).
        if self._worker is not None:
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

        if worker.mode == IndexerWorker.PREVIEW and success and worker.cancel_requested:
            # Cancel was pressed after the worker's last cancel check, so the
            # preview finished anyway. A preview writes nothing to the library,
            # so honour the click: drop the result instead of showing a plan
            # (and a "Preview Ready" box) the user just cancelled.
            success, outcome, message = False, "cancelled", "Cancelled by user."
            self._status_lbl.setText(message)

        if worker.mode == IndexerWorker.PREVIEW:
            if success and same_library:
                self._plan = worker.plan
                self._plan_library = worker.library_path
                self._report_path = worker.report_path
                self._report_btn.setEnabled(bool(self._report_path))
                for _, bar in self._phase_bars[:2]:
                    bar.setValue(100)
                counts = self._plan.counts()
                if counts.get("audio_files"):
                    self._show_stats("Previewed plan", counts)
                    # The tiles carry the numbers; the line says what to do next.
                    self._status_lbl.setText(
                        "Preview ready. Open the report to check every move, then untick "
                        "Dry run to execute exactly this plan."
                    )
                else:
                    self._show_results_empty(
                        "Nothing to organise",
                        "No audio files were found outside the reserved folders, so the "
                        "preview has nothing to move.",
                    )
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
            self._show_execute_results(worker, outcome)
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

    def _show_execute_results(self, worker: IndexerWorker, outcome: str) -> None:
        """Relabel the results tiles with what the execute did (the plan is
        used up either way, so its counts would mislead)."""
        summary = worker.summary
        if not summary:
            self._show_results_empty(
                "Preview again",
                "The previewed plan was not applied (see the status above). Run a new "
                "preview before executing.",
            )
            return
        errors = list(summary.get("errors", []) or [])
        planned = worker.plan.counts() if worker.plan is not None else {}
        values = {
            "moves": summary.get("moved", 0),
            "unchanged": planned.get("unchanged", "–"),
            "manual_review": len(errors),
            "leftovers": summary.get("leftovers_moved", 0),
            "empty_dirs": len(summary.get("removed_dirs", []) or []),
        }
        title = {"ok": "Last execution", "cancelled": "Last execution (cancelled part-way)",
                 "partial": "Last execution (finished with problems)"}.get(outcome, "Last execution")
        self._show_stats(title, values, _EXECUTE_LABELS,
                         {"manual_review": "danger" if errors else ""}, _EXECUTE_TIPS)

    def _on_library_changed(self, path: str) -> None:
        self._show_library(path)
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
            self._show_results_empty(*_NO_PREVIEW)
            candidate = Path(path) / "Docs" / "MusicIndex.html" if path else None
            if candidate is not None and candidate.exists():
                self._report_path = str(candidate)
                self._report_btn.setEnabled(True)
            self._dry_run_cb.setChecked(True)
        self._refresh_controls()
