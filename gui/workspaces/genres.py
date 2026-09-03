"""Genre Normalizer workspace.

Two genuinely different tools live behind this one sidebar entry, so they are
two tabs rather than one blurred panel:

* **Fill from MusicBrainz** — for tracks with few or no genre tags, look the
  recording up and write MusicBrainz's most popular community tags.
* **Normalize with a mapping** — collapse the messy vocabulary a library has
  accumulated ("hip hop", "Hip-Hop", "hiphoprap") into a canonical one, using a
  raw -> canonical mapping you build with an LLM's help. This is the canonical
  mapping workflow that previously existed only in the legacy Tkinter app,
  rebuilt preview-first: you see every file that would change before anything
  is written.

All decisions live in ``controllers.normalize_controller``; this file only
collects input and displays results.
"""
from __future__ import annotations

import json
import os
import threading

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase

from controllers import normalize_controller as nc


# ── Workers ───────────────────────────────────────────────────────────────────


class GenreWorker(QtCore.QThread):
    """Fill-from-MusicBrainz worker (unchanged behaviour)."""

    progress = Signal(int)
    log_line = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, library_path: str, dry_run: bool) -> None:
        super().__init__()
        self.library_path = library_path
        self.dry_run = dry_run
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
            worker_ref = self

            class _LogProxy:
                def write(self, msg: str) -> None:  # noqa: PLR6301
                    line = msg.rstrip("\n")
                    if line and not worker_ref._cancelled:
                        worker_ref.log_line.emit(line)

                def flush(self) -> None:
                    pass

            log_proxy = _LogProxy()

            files: list[str] = []
            for dirpath, _, filenames in os.walk(self.library_path):
                for f in filenames:
                    if os.path.splitext(f)[1].lower() in update_genres.SUPPORTED_EXTS:
                        files.append(os.path.join(dirpath, f))

            total = len(files)
            for idx, filepath in enumerate(files, start=1):
                if self._cancelled:
                    break
                if not self.dry_run:
                    update_genres.process_file(filepath, log_proxy)
                else:
                    worker_ref.log_line.emit(f"[dry-run] would process: {os.path.basename(filepath)}")
                self.progress.emit(int(idx * 100 / max(total, 1)))

            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
            else:
                self.finished.emit(True, f"Genre update complete ({total} files processed).")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class _ScanWorker(QtCore.QThread):
    progress = Signal(int, int)
    finished = Signal(bool, object)  # ok, list[str] | error message

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self.library_path = library_path

    def run(self) -> None:
        try:
            genres = nc.scan_raw_genres(self.library_path, self.progress.emit)
            self.finished.emit(True, genres)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class _PlanWorker(QtCore.QThread):
    progress = Signal(int, int)
    finished = Signal(bool, object)  # ok, NormalizationPlan | error message

    def __init__(self, library_path: str, mapping: dict) -> None:
        super().__init__()
        self.library_path = library_path
        self.mapping = mapping

    def run(self) -> None:
        try:
            plan = nc.plan_genre_normalization(
                self.library_path, self.mapping, self.progress.emit
            )
            self.finished.emit(True, plan)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class _ApplyWorker(QtCore.QThread):
    progress = Signal(int, int)
    finished = Signal(bool, object)  # ok, ApplyResult | error message

    def __init__(self, changes: list) -> None:
        super().__init__()
        self.changes = changes
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            result = nc.apply_genre_changes(
                self.changes, self.progress.emit, self.cancel_event
            )
            self.finished.emit(True, result)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


# ── Workspace ─────────────────────────────────────────────────────────────────


class GenresWorkspace(WorkspaceBase):
    """Genre tools: MusicBrainz fill, and canonical mapping normalization."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: GenreWorker | None = None
        self._scan_worker: _ScanWorker | None = None
        self._plan_worker: _PlanWorker | None = None
        self._apply_worker: _ApplyWorker | None = None
        self._raw_genres: list[str] = []
        self._plan: nc.NormalizationPlan | None = None
        self._plan_mapping_text: str = ""
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_section_title("Genre Normalizer"))
        cl.addWidget(self._make_subtitle(
            "Two ways to tidy genre tags: fill in missing ones from MusicBrainz, or "
            "collapse the messy vocabulary your library has accumulated into a "
            "canonical one with a mapping you build once and reuse."
        ))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_fill_tab(), "Fill from MusicBrainz")
        tabs.addTab(self._build_normalize_tab(), "Normalize with a mapping")
        cl.addWidget(tabs, 1)
        self._refresh_library_state()

    # ── Tab 1: MusicBrainz fill ───────────────────────────────────────────

    def _build_fill_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        opt_card = self._make_card()
        opt_layout = QtWidgets.QVBoxLayout(opt_card)
        opt_layout.setContentsMargins(16, 16, 16, 16)
        opt_layout.setSpacing(10)
        opt_layout.addWidget(self._make_card_title("What this does"))
        info = QtWidgets.QLabel(
            "Looks each track up on MusicBrainz by its artist and title and writes "
            "the three most popular community tags into its genre field. Files that "
            "already carry two or more genre tags are left untouched, so this fills "
            "gaps rather than overwriting choices you've already made."
        )
        info.setWordWrap(True)
        info.setObjectName("statusHint")
        opt_layout.addWidget(info)

        self._dry_run_cb = QtWidgets.QCheckBox("Dry run (list files, do not write)")
        self._dry_run_cb.setChecked(True)
        opt_layout.addWidget(self._dry_run_cb)
        layout.addWidget(opt_card)

        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = self._make_primary_button("🎸  Run Genre Update")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setValue(0)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setFixedHeight(6)
        self._prog_status = QtWidgets.QLabel("Idle")
        self._prog_status.setObjectName("statusHint")
        layout.addWidget(self._prog_bar)
        layout.addWidget(self._prog_status)

        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(self._make_card_title("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(220)
        self._log_area.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        log_layout.addWidget(self._log_area)
        layout.addWidget(log_card, 1)
        return page

    # ── Tab 2: canonical mapping ──────────────────────────────────────────

    def _build_normalize_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        # 1 · Scan
        scan_card = self._make_card()
        sl = QtWidgets.QVBoxLayout(scan_card)
        sl.setContentsMargins(16, 16, 16, 16)
        sl.setSpacing(8)
        sl.addWidget(self._make_card_title("1 · Collect the raw genres in your library"))
        row = QtWidgets.QHBoxLayout()
        self._scan_btn = self._make_primary_button("🔍  Scan library")
        self._scan_btn.clicked.connect(self._on_scan)
        self._copy_raw_btn = QtWidgets.QPushButton("Copy list")
        self._copy_raw_btn.setEnabled(False)
        self._copy_raw_btn.clicked.connect(
            lambda: self._copy("\n".join(self._raw_genres), "Raw genre list copied.")
        )
        self._scan_status = QtWidgets.QLabel("Scan to see every distinct genre string.")
        self._scan_status.setObjectName("statusHint")
        row.addWidget(self._scan_btn)
        row.addWidget(self._copy_raw_btn)
        row.addWidget(self._scan_status, 1)
        sl.addLayout(row)
        self._raw_list = QtWidgets.QListWidget()
        self._raw_list.setMaximumHeight(140)
        sl.addWidget(self._raw_list)
        layout.addWidget(scan_card)

        # 2 · Mapping
        map_card = self._make_card()
        ml = QtWidgets.QVBoxLayout(map_card)
        ml.setContentsMargins(16, 16, 16, 16)
        ml.setSpacing(8)
        ml.addWidget(self._make_card_title("2 · Build the mapping"))
        how = QtWidgets.QLabel(
            "Copy the prompt, paste it and the raw list into an LLM, then paste the "
            "JSON it returns below. Keys are raw genres; values are the canonical "
            "name(s) to use. A value of [\"invalid\"] or null means \"drop this — "
            "it isn't a genre\". Unmapped genres pass through unchanged."
        )
        how.setWordWrap(True)
        how.setObjectName("statusHint")
        ml.addWidget(how)

        prow = QtWidgets.QHBoxLayout()
        self._copy_prompt_btn = QtWidgets.QPushButton("Copy prompt")
        self._copy_prompt_btn.clicked.connect(
            lambda: self._copy(nc.PROMPT_TEMPLATE.strip(), "Prompt copied.")
        )
        self._load_map_btn = QtWidgets.QPushButton("Load saved mapping")
        self._load_map_btn.clicked.connect(self._on_load_mapping)
        self._save_map_btn = QtWidgets.QPushButton("Save mapping")
        self._save_map_btn.clicked.connect(self._on_save_mapping)
        prow.addWidget(self._copy_prompt_btn)
        prow.addWidget(self._load_map_btn)
        prow.addWidget(self._save_map_btn)
        prow.addStretch(1)
        ml.addLayout(prow)

        self._map_edit = QtWidgets.QPlainTextEdit()
        self._map_edit.setPlaceholderText('{\n  "rock & roll": ["Rock"],\n  "90s": ["invalid"]\n}')
        self._map_edit.setMinimumHeight(160)
        self._map_edit.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        self._map_edit.textChanged.connect(self._on_mapping_text_changed)
        ml.addWidget(self._map_edit)

        self._map_status = QtWidgets.QLabel("No mapping yet.")
        self._map_status.setObjectName("statusHint")
        ml.addWidget(self._map_status)
        layout.addWidget(map_card)

        # 3 · Preview
        prev_card = self._make_card()
        pl = QtWidgets.QVBoxLayout(prev_card)
        pl.setContentsMargins(16, 16, 16, 16)
        pl.setSpacing(8)
        pl.addWidget(self._make_card_title("3 · Preview what would change"))
        prow2 = QtWidgets.QHBoxLayout()
        self._preview_btn = self._make_primary_button("👁  Preview changes")
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._on_preview)
        self._preview_status = QtWidgets.QLabel("Nothing previewed yet — nothing is written until step 4.")
        self._preview_status.setObjectName("statusHint")
        prow2.addWidget(self._preview_btn)
        prow2.addWidget(self._preview_status, 1)
        pl.addLayout(prow2)

        self._changes_table = QtWidgets.QTableWidget(0, 3)
        self._changes_table.setHorizontalHeaderLabels(["File", "Current genres", "After"])
        self._changes_table.horizontalHeader().setStretchLastSection(True)
        self._changes_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._changes_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._changes_table.setMinimumHeight(200)
        pl.addWidget(self._changes_table)
        layout.addWidget(prev_card, 1)

        # 4 · Apply
        apply_card = self._make_card()
        al = QtWidgets.QVBoxLayout(apply_card)
        al.setContentsMargins(16, 16, 16, 16)
        al.setSpacing(8)
        al.addWidget(self._make_card_title("4 · Apply"))
        arow = QtWidgets.QHBoxLayout()
        self._apply_btn = QtWidgets.QPushButton("✎  Apply changes")
        self._apply_btn.setObjectName("dangerBtn")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        self._apply_cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._apply_cancel_btn.setEnabled(False)
        self._apply_cancel_btn.clicked.connect(self._on_apply_cancel)
        self._apply_status = QtWidgets.QLabel("Preview first; Apply writes exactly what the preview shows.")
        self._apply_status.setObjectName("statusHint")
        arow.addWidget(self._apply_btn)
        arow.addWidget(self._apply_cancel_btn)
        arow.addWidget(self._apply_status, 1)
        al.addLayout(arow)
        self._norm_bar = QtWidgets.QProgressBar()
        self._norm_bar.setValue(0)
        self._norm_bar.setTextVisible(False)
        self._norm_bar.setFixedHeight(6)
        al.addWidget(self._norm_bar)
        layout.addWidget(apply_card)

        return page

    # ── Shared helpers ────────────────────────────────────────────────────

    def _copy(self, text: str, message: str) -> None:
        QtWidgets.QApplication.clipboard().setText(text)
        self._log(message, "ok")

    def _refresh_library_state(self) -> None:
        has_lib = bool(self._library_path)
        for b in (self._run_btn, self._scan_btn, self._load_map_btn, self._save_map_btn):
            b.setEnabled(has_lib)
        if has_lib:
            mapping, path = nc.load_mapping(self._library_path)
            if mapping and not self._map_edit.toPlainText().strip():
                self._map_edit.setPlainText(json.dumps(mapping, indent=2))
                self._map_status.setText(f"Loaded saved mapping from {path}")
        self._update_preview_gate()

    def _on_library_changed(self, path: str) -> None:
        self._raw_genres = []
        self._raw_list.clear()
        self._copy_raw_btn.setEnabled(False)
        self._map_edit.clear()
        self._invalidate_plan("Library changed — preview again before applying.")
        self._refresh_library_state()

    def _current_mapping(self) -> dict | None:
        """Parse the editor; returns None (and sets status) on bad JSON."""
        text = self._map_edit.toPlainText().strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self._map_status.setText(f"Invalid JSON: {exc.msg} (line {exc.lineno})")
            return None
        if not isinstance(data, dict):
            self._map_status.setText("Mapping must be a JSON object of raw → canonical.")
            return None
        cleaned = nc.clean_mapping(data)
        dropped = sum(1 for v in cleaned.values() if v is None)
        self._map_status.setText(
            f"Valid mapping: {len(cleaned)} raw genres"
            + (f", {dropped} marked to drop" if dropped else "")
        )
        return data

    def _update_preview_gate(self) -> None:
        ok = bool(self._library_path) and self._current_mapping() is not None
        self._preview_btn.setEnabled(ok and self._plan_worker is None)

    @Slot()
    def _on_mapping_text_changed(self) -> None:
        # The plan is only valid for the mapping it was computed from — the same
        # rule Library Sync applies to its plan when inputs change.
        if self._plan is not None and self._map_edit.toPlainText() != self._plan_mapping_text:
            self._invalidate_plan("Mapping changed — preview again before applying.")
        self._update_preview_gate()

    def _invalidate_plan(self, why: str) -> None:
        self._plan = None
        self._plan_mapping_text = ""
        self._changes_table.setRowCount(0)
        self._apply_btn.setEnabled(False)
        self._preview_status.setText(why)

    # ── Tab 1 slots ───────────────────────────────────────────────────────

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

        self._worker = GenreWorker(self._library_path, dry_run=self._dry_run_cb.isChecked())
        self._worker.log_line.connect(self._log_area.appendPlainText)
        self._worker.progress.connect(self._prog_bar.setValue)
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

    # ── Tab 2 slots ───────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if not self._library_path or self._scan_worker is not None:
            return
        self._scan_btn.setEnabled(False)
        self._scan_status.setText("Scanning…")
        self._scan_worker = _ScanWorker(self._library_path)
        self._scan_worker.progress.connect(
            lambda i, t: self._scan_status.setText(f"Scanning… {i}/{t}")
        )
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.start()

    @Slot(bool, object)
    def _on_scan_finished(self, ok: bool, payload) -> None:
        self._scan_worker = None
        self._scan_btn.setEnabled(bool(self._library_path))
        if not ok:
            self._scan_status.setText(f"Scan failed: {payload}")
            self._log(f"Genre scan failed: {payload}", "error")
            return
        self._raw_genres = list(payload)
        self._raw_list.clear()
        self._raw_list.addItems(self._raw_genres)
        self._copy_raw_btn.setEnabled(bool(self._raw_genres))
        self._scan_status.setText(f"Found {len(self._raw_genres)} distinct genre strings.")
        self._log(f"Genre scan: {len(self._raw_genres)} distinct raw genres.", "ok")

    @Slot()
    def _on_load_mapping(self) -> None:
        mapping, path = nc.load_mapping(self._library_path)
        if not mapping:
            self._map_status.setText(f"No saved mapping at {path}")
            return
        self._map_edit.setPlainText(json.dumps(mapping, indent=2))
        self._map_status.setText(f"Loaded saved mapping from {path}")

    @Slot()
    def _on_save_mapping(self) -> None:
        mapping = self._current_mapping()
        if mapping is None:
            return
        path = nc.save_mapping(self._library_path, nc.clean_mapping(mapping))
        self._map_status.setText(f"Saved mapping to {path}")
        self._log(f"Genre mapping saved to {path}", "ok")

    @Slot()
    def _on_preview(self) -> None:
        mapping = self._current_mapping()
        if mapping is None or self._plan_worker is not None:
            return
        self._invalidate_plan("Previewing…")
        self._preview_btn.setEnabled(False)
        self._plan_mapping_text = self._map_edit.toPlainText()
        self._plan_worker = _PlanWorker(self._library_path, mapping)
        self._plan_worker.progress.connect(
            lambda i, t: self._preview_status.setText(f"Reading tags… {i}/{t}")
        )
        self._plan_worker.finished.connect(self._on_plan_finished)
        self._plan_worker.start()

    @Slot(bool, object)
    def _on_plan_finished(self, ok: bool, payload) -> None:
        self._plan_worker = None
        self._update_preview_gate()
        if not ok:
            self._invalidate_plan(f"Preview failed: {payload}")
            self._log(f"Genre preview failed: {payload}", "error")
            return

        plan: nc.NormalizationPlan = payload
        self._plan = plan
        self._changes_table.setRowCount(len(plan.changes))
        for row, change in enumerate(plan.changes):
            cells = (
                os.path.relpath(change.path, self._library_path)
                if self._library_path else change.path,
                "; ".join(change.before),
                "; ".join(change.after) if change.after else "(none — all dropped)",
            )
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(change.path if col == 0 else text)
                self._changes_table.setItem(row, col, item)
        self._changes_table.resizeColumnToContents(0)

        n = len(plan.changes)
        self._preview_status.setText(
            f"{n} of {plan.with_genres} tagged files would change "
            f"({plan.unchanged} already canonical, {plan.scanned - plan.with_genres} with no genre tag)."
        )
        self._apply_btn.setEnabled(n > 0)
        self._apply_btn.setText(f"✎  Apply {n} change{'s' if n != 1 else ''}")
        self._log(f"Genre preview: {n} file(s) would change.", "ok")

    @Slot()
    def _on_apply(self) -> None:
        if self._plan is None or not self._plan.changes:
            return
        if self._map_edit.toPlainText() != self._plan_mapping_text:
            self._invalidate_plan("Mapping changed — preview again before applying.")
            return
        n = len(self._plan.changes)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Apply genre changes",
            f"Rewrite the genre tag on {n} file(s) exactly as previewed?\n\n"
            "This edits the files in place. The mapping will also be saved to "
            "Docs/.genre_mapping.json so you can reuse it.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        mapping = self._current_mapping()
        if mapping is not None:
            nc.save_mapping(self._library_path, nc.clean_mapping(mapping))

        self._apply_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
        self._apply_cancel_btn.setEnabled(True)
        self._norm_bar.setValue(0)
        self._apply_status.setText("Writing…")
        self.status_changed.emit("Normalizing genres…", "#f59e0b")

        self._apply_worker = _ApplyWorker(list(self._plan.changes))
        self._apply_worker.progress.connect(self._on_apply_progress)
        self._apply_worker.finished.connect(self._on_apply_finished)
        self._apply_worker.start()

    @Slot(int, int)
    def _on_apply_progress(self, done: int, total: int) -> None:
        self._norm_bar.setValue(int(done * 100 / max(total, 1)))
        self._apply_status.setText(f"Writing… {done}/{total}")

    @Slot()
    def _on_apply_cancel(self) -> None:
        if self._apply_worker and self._apply_worker.isRunning():
            self._apply_worker.cancel()
            self._apply_cancel_btn.setEnabled(False)

    @Slot(bool, object)
    def _on_apply_finished(self, ok: bool, payload) -> None:
        self._apply_worker = None
        self._apply_cancel_btn.setEnabled(False)
        self._update_preview_gate()
        if not ok:
            self._apply_status.setText(f"Apply failed: {payload}")
            self._log(f"Genre apply failed: {payload}", "error")
            self.status_changed.emit("Error", "#ef4444")
            return

        result: nc.ApplyResult = payload
        summary = f"Rewrote {len(result.applied)} file(s)"
        if result.failed:
            summary += f", {len(result.failed)} failed"
        if result.cancelled:
            summary += " (cancelled)"
        self._apply_status.setText(summary + ".")
        self._log(f"Genre normalization: {summary}.", "warn" if result.failed else "ok")
        for path in result.failed:
            self._log(f"  could not write: {path}", "error")
        self.status_changed.emit("Done", "#22c55e" if not result.failed else "#f59e0b")
        # The files on disk no longer match the plan; require a fresh preview.
        self._invalidate_plan("Applied. Preview again to see the library's new state.")
