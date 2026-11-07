"""Genre Normalizer workspace.

Two genuinely different tools live behind this one sidebar entry, so they are
two tabs rather than one blurred panel:

* **Fill from MusicBrainz** — for tracks with few or no genre tags, look the
  recording up and add MusicBrainz's most popular community tags. Preview
  first: the lookups only build a list of proposed additions; nothing is
  written until you confirm, and then exactly that list is written.
* **Normalize with a mapping** — collapse the messy vocabulary a library has
  accumulated ("hip hop", "Hip-Hop", "hiphoprap") into a canonical one, using a
  raw -> canonical mapping you build with an LLM's help. This is the canonical
  mapping workflow that previously existed only in the legacy Tkinter app,
  rebuilt preview-first: you see every file that would change before anything
  is written.

Both read *every* genre value of a file (not just the first), write genres as
real multi-value lists, skip files that changed since the preview, log every
write's before/after values to ``Docs/tag_changes_*.jsonl``, and share one
"tags are being written" guard with the Tag Fixer so two tools never write at
once.

All decisions live in ``controllers.normalize_controller``; this file only
collects input and displays results.
"""
from __future__ import annotations

import json
import os
import threading

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

from controllers import normalize_controller as nc

_FILL_OWNER = "Genre fill (MusicBrainz) apply"
_NORMALIZE_OWNER = "Genre normalize apply"


# ── Workers ───────────────────────────────────────────────────────────────────


class GenreWorker(QtCore.QThread):
    """Fill-from-MusicBrainz *lookup* worker: builds a FillPlan, never writes."""

    progress = Signal(int, int)
    log_line = Signal(str)
    finished = Signal(bool, object)  # ok, FillPlan | error message

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self.library_path = library_path
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            plan = nc.plan_genre_fill(
                self.library_path,
                self.progress.emit,
                should_cancel=self.is_cancelled,
                log_callback=self.log_line.emit,
            )
            self.finished.emit(True, plan)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class _ScanWorker(QtCore.QThread):
    progress = Signal(int, int)
    finished = Signal(bool, object)  # ok, list[str] | error message

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self.library_path = library_path
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            genres = nc.scan_raw_genres(
                self.library_path, self.progress.emit,
                should_cancel=lambda: self._cancel.is_set() or self.isInterruptionRequested(),
            )
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
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            plan = nc.plan_genre_normalization(
                self.library_path, self.mapping, self.progress.emit,
                should_cancel=lambda: self._cancel.is_set() or self.isInterruptionRequested(),
            )
            self.finished.emit(True, plan)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class _ApplyWorker(QtCore.QThread):
    """Writes exactly a plan's changes, logging before/after to Docs/."""

    progress = Signal(int, int)
    finished = Signal(bool, object)  # ok, ApplyResult | error message

    def __init__(
        self,
        changes: list,
        library_root: str = "",
        tool: str = "genre_normalize",
        owner: str = _NORMALIZE_OWNER,
    ) -> None:
        super().__init__()
        self.changes = changes
        self.library_root = library_root
        self.tool = tool
        self.owner = owner
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        from controllers import tag_write_guard

        try:
            change_log = None
            if self.library_root:
                from utils.tag_change_log import TagChangeLog

                change_log = TagChangeLog(self.library_root, self.tool)
            try:
                result = nc.apply_genre_changes(
                    self.changes, self.progress.emit, self.cancel_event,
                    change_log=change_log,
                )
            finally:
                if change_log is not None:
                    change_log.close()
            self.finished.emit(True, result)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))
        finally:
            tag_write_guard.release(self.owner)


# ── Workspace ─────────────────────────────────────────────────────────────────


class GenresWorkspace(WorkspaceBase):
    """Genre tools: MusicBrainz fill, and canonical mapping normalization."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: GenreWorker | None = None              # fill lookup
        self._fill_apply_worker: _ApplyWorker | None = None
        self._fill_plan: nc.FillPlan | None = None
        self._scan_worker: _ScanWorker | None = None
        self._plan_worker: _PlanWorker | None = None
        self._apply_worker: _ApplyWorker | None = None
        self._raw_genres: list[str] = []
        self._plan: nc.NormalizationPlan | None = None
        self._plan_mapping_text: str = ""
        # Unsaved mapping text per library, so switching library (or back)
        # never silently throws away a pasted mapping.
        self._drafts: dict[str, str] = {}
        self._draft_owner: str = library_path
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

    @staticmethod
    def _mono_font() -> QtGui.QFont:
        return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)

    def _emit_status(self, text: str, role: str) -> None:
        self.status_changed.emit(text, getattr(get_manager().current, role))

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
            "Looks each track up on MusicBrainz by its artist and title and proposes "
            "adding the three most popular community tags to its genre field. Genres "
            "a file already has are kept — the tags are added after them — and files "
            "that already carry two or more genres are skipped. Nothing is written "
            "until you review the proposals and confirm."
        )
        info.setWordWrap(True)
        info.setObjectName("statusHint")
        opt_layout.addWidget(info)
        layout.addWidget(opt_card)

        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = self._make_primary_button("🔍  Look up genres (preview)")
        self._run_btn.clicked.connect(self._on_run)
        self._fill_apply_btn = QtWidgets.QPushButton("✎  Apply genre additions")
        self._fill_apply_btn.setObjectName("dangerBtn")
        self._fill_apply_btn.setEnabled(False)
        self._fill_apply_btn.clicked.connect(self._on_fill_apply)
        self._cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._fill_apply_btn)
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

        prev_card = self._make_card()
        pl = QtWidgets.QVBoxLayout(prev_card)
        pl.setContentsMargins(16, 16, 16, 16)
        pl.setSpacing(8)
        pl.addWidget(self._make_card_title("Proposed genre additions"))
        self._fill_table = QtWidgets.QTableWidget(0, 4)
        self._fill_table.setHorizontalHeaderLabels(
            ["File", "Current genres", "After", "MusicBrainz match"]
        )
        self._fill_table.horizontalHeader().setStretchLastSection(True)
        self._fill_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._fill_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._fill_table.setMinimumHeight(160)
        pl.addWidget(self._fill_table)
        self._fill_summary = QtWidgets.QLabel("Nothing looked up yet — nothing is written until you apply.")
        self._fill_summary.setObjectName("statusHint")
        self._fill_summary.setWordWrap(True)
        pl.addWidget(self._fill_summary)
        layout.addWidget(prev_card)

        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(self._make_card_title("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(160)
        self._log_area.setFont(self._mono_font())
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
        self._map_edit.setFont(self._mono_font())
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
        self._preview_status.setWordWrap(True)
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
        self._apply_status.setWordWrap(True)
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

    def _any_apply_running(self) -> bool:
        return self._apply_worker is not None or self._fill_apply_worker is not None

    def _refresh_library_state(self) -> None:
        has_lib = bool(self._library_path)
        for b in (self._load_map_btn, self._save_map_btn):
            b.setEnabled(has_lib)
        self._scan_btn.setEnabled(has_lib and self._scan_worker is None)
        if has_lib:
            mapping, path = nc.load_mapping(self._library_path)
            if mapping and not self._map_edit.toPlainText().strip():
                self._map_edit.setPlainText(json.dumps(mapping, indent=2))
                self._map_status.setText(f"Loaded saved mapping from {path}")
        self._update_fill_gate()
        self._update_preview_gate()

    def _on_library_changed(self, path: str) -> None:
        # Results belong to the library they were computed for. Read-only
        # jobs in flight are cancelled and their results discarded (applying
        # a stale plan wrote the *old* library's files). A confirmed apply is
        # allowed to finish — it writes exactly what was confirmed — and its
        # outcome is logged.
        for attr in ("_worker", "_scan_worker", "_plan_worker"):
            worker = getattr(self, attr)
            if worker is not None:
                worker.cancel()
                setattr(self, attr, None)

        # Keep an unsaved mapping as a per-library draft instead of wiping it.
        old_owner = self._draft_owner
        text = self._map_edit.toPlainText()
        if old_owner:
            if text.strip() and self._mapping_is_dirty(old_owner, text):
                self._drafts[old_owner] = text
            else:
                self._drafts.pop(old_owner, None)
        self._draft_owner = path

        self._raw_genres = []
        self._raw_list.clear()
        self._copy_raw_btn.setEnabled(False)
        self._scan_status.setText("Scan to see every distinct genre string.")
        self._invalidate_fill_plan("Library changed — look up again before applying.")
        self._prog_bar.setValue(0)
        if self._fill_apply_worker is None:
            self._prog_status.setText("Idle")
        draft = self._drafts.get(path)
        self._map_edit.clear()
        self._invalidate_plan("Library changed — preview again before applying.")
        if draft is not None:
            self._map_edit.setPlainText(draft)
            self._map_status.setText("Restored your unsaved mapping for this library.")
        self._refresh_library_state()

    def _mapping_is_dirty(self, library: str, text: str) -> bool:
        """True when ``text`` differs from the mapping saved for ``library``."""
        saved, _path = nc.load_mapping(library) if library else ({}, "")
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return bool(text.strip())
        if not isinstance(data, dict):
            return True
        return nc.clean_mapping(data) != nc.clean_mapping(saved)

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
        # No preview while tags are being written: it would read half-written
        # state and could produce a plan mid-apply.
        self._preview_btn.setEnabled(ok and self._plan_worker is None and not self._any_apply_running())

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

    def _update_fill_gate(self) -> None:
        has_lib = bool(self._library_path)
        lookup_running = self._worker is not None
        applying = self._fill_apply_worker is not None
        self._run_btn.setEnabled(has_lib and not lookup_running and not applying)
        self._fill_apply_btn.setEnabled(
            bool(self._fill_plan and self._fill_plan.changes)
            and not lookup_running and not self._any_apply_running()
        )
        self._cancel_btn.setEnabled(lookup_running or applying)

    def _invalidate_fill_plan(self, why: str) -> None:
        self._fill_plan = None
        self._fill_table.setRowCount(0)
        self._fill_apply_btn.setText("✎  Apply genre additions")
        self._fill_summary.setText(why)
        self._update_fill_gate()

    @Slot()
    def _on_run(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        if self._worker is not None or self._fill_apply_worker is not None:
            return
        self._log_area.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Looking up genres… (nothing is written)")
        self._invalidate_fill_plan("Looking up…")
        self._log("Starting genre lookup (preview)…", "info")
        self._emit_status("Looking up genres…", "warning")

        worker = GenreWorker(self._library_path)
        worker.log_line.connect(self._on_fill_log_line)
        worker.progress.connect(self._on_fill_progress)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self._update_fill_gate()
        self._start_worker(worker, "Genre lookup (MusicBrainz)")

    @Slot(str)
    def _on_fill_log_line(self, line: str) -> None:
        if self.sender() is self._worker:
            self._log_area.appendPlainText(line)

    @Slot(int, int)
    def _on_fill_progress(self, done: int, total: int) -> None:
        sender = self.sender()
        if sender is not self._worker and sender is not self._fill_apply_worker:
            return
        self._prog_bar.setValue(int(done * 100 / max(total, 1)))
        verb = "Writing" if sender is self._fill_apply_worker else "Looking up"
        self._prog_status.setText(f"{verb}… {done}/{total}")

    @Slot()
    def _on_cancel(self) -> None:
        for worker in (self._worker, self._fill_apply_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._prog_status.setText("Cancelling…")

    @Slot(bool, object)
    def _on_finished(self, success: bool, payload) -> None:
        if self.sender() is not self._worker:
            return  # stale: cancelled by a library change
        self._worker = None
        if not success:
            self._prog_status.setText(f"Lookup failed: {payload}")
            self._log(f"Genre lookup failed: {payload}", "error")
            self._emit_status("Error", "danger")
            self._invalidate_fill_plan(f"Lookup failed: {payload}")
            return
        plan: nc.FillPlan = payload
        if plan.cancelled:
            self._prog_status.setText("Cancelled.")
            self._log("Genre lookup cancelled.", "warn")
            self._emit_status("Cancelled", "warning")
            self._invalidate_fill_plan("Cancelled — nothing was written.")
            return
        self._fill_plan = plan
        self._prog_bar.setValue(100)
        self._fill_table.setRowCount(len(plan.changes))
        for row, change in enumerate(plan.changes):
            cells = (
                os.path.relpath(change.path, self._library_path)
                if self._library_path else change.path,
                "; ".join(change.before) if change.before else "(none)",
                "; ".join(change.after),
                change.note,
            )
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(change.path if col == 0 else text)
                self._fill_table.setItem(row, col, item)
        self._fill_table.resizeColumnToContents(0)
        n = len(plan.changes)
        parts = [f"{n} file(s) would get genres added"]
        if plan.skipped_multi:
            parts.append(f"{len(plan.skipped_multi)} skipped (already 2+ genres)")
        if plan.no_match:
            parts.append(f"{len(plan.no_match)} with no usable match")
        if plan.errors:
            parts.append(f"{len(plan.errors)} lookup error(s)")
        if plan.blocked:
            parts.append(f"{len(plan.blocked)} can't be written (see log)")
            for path, reason in plan.blocked:
                self._log_area.appendPlainText(f"can't write {path}: {reason}")
        summary = " · ".join(parts) + "."
        self._fill_summary.setText(summary + " Nothing has been written yet.")
        self._prog_status.setText("Lookup complete — review, then apply.")
        self._fill_apply_btn.setText(f"✎  Apply {n} addition{'s' if n != 1 else ''}")
        self._log(f"Genre lookup: {summary}", "ok")
        self._emit_status("Lookup complete", "success")
        self._update_fill_gate()

    @Slot()
    def _on_fill_apply(self) -> None:
        plan = self._fill_plan
        if plan is None or not plan.changes or self._worker is not None or self._any_apply_running():
            return
        from controllers import tag_write_guard

        if self._library_writes_blocked("Adding genres"):
            return
        n = len(plan.changes)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Add genres from MusicBrainz",
            f"Add MusicBrainz genres to {n} file(s) exactly as listed?\n\n"
            "Existing genres are kept. The files are edited in place; every change "
            "is logged with its previous value in Docs/tag_changes_genre_fill_….jsonl.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if not tag_write_guard.acquire(_FILL_OWNER):
            QtWidgets.QMessageBox.information(
                self, "Tags Busy", tag_write_guard.busy_message("Adding genres")
            )
            return
        self._prog_bar.setValue(0)
        self._prog_status.setText("Writing…")
        self._emit_status("Adding genres…", "warning")
        worker = _ApplyWorker(list(plan.changes), self._library_path, "genre_fill", _FILL_OWNER)
        worker.progress.connect(self._on_fill_progress)
        worker.finished.connect(self._on_fill_apply_finished)
        self._fill_apply_worker = worker
        self._update_fill_gate()
        self._update_preview_gate()
        self._start_worker(worker, _FILL_OWNER, library_write=True)

    @Slot(bool, object)
    def _on_fill_apply_finished(self, ok: bool, payload) -> None:
        worker = self.sender()
        if worker is not self._fill_apply_worker:
            return
        self._fill_apply_worker = None
        self._report_apply("Genre fill", worker, ok, payload, self._prog_status, self._log_area)
        self._invalidate_fill_plan("Applied. Look up again to see the library's new state.")
        self._update_preview_gate()
        self._update_apply_gate()

    def _report_apply(self, label: str, worker, ok: bool, payload, status_lbl, log_area=None) -> None:
        if not ok:
            status_lbl.setText(f"Apply failed: {payload}")
            self._log(f"{label} apply failed: {payload}", "error")
            self._emit_status("Error", "danger")
            return
        result: nc.ApplyResult = payload
        summary = f"Rewrote {len(result.applied)} file(s)"
        if result.failed:
            summary += f", {len(result.failed)} failed"
        if result.skipped:
            summary += f", {len(result.skipped)} skipped (changed since the preview)"
        if result.cancelled:
            summary += " (cancelled)"
        if result.log_path and worker.library_root:
            summary += f". Change log: {os.path.relpath(result.log_path, worker.library_root)}"
        if worker.library_root and worker.library_root != self._library_path:
            summary += f" (library: {worker.library_root})"
        status_lbl.setText(summary + ".")
        problems = bool(result.failed or result.skipped)
        self._log(f"{label}: {summary}.", "warn" if problems else "ok")
        for path in result.failed:
            self._log(f"  could not write: {path}", "error")
            if log_area is not None:
                log_area.appendPlainText(f"could not write: {path}")
        for path in result.skipped:
            self._log(f"  skipped (changed since the preview): {path}", "warn")
        self._emit_status("Done", "warning" if problems else "success")

    # ── Tab 2 slots ───────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if not self._library_path or self._scan_worker is not None:
            return
        self._scan_btn.setEnabled(False)
        self._scan_status.setText("Scanning…")
        worker = _ScanWorker(self._library_path)
        worker.progress.connect(self._on_scan_progress)
        worker.finished.connect(self._on_scan_finished)
        self._scan_worker = worker
        self._start_worker(worker, "Genre scan")

    @Slot(int, int)
    def _on_scan_progress(self, i: int, t: int) -> None:
        if self.sender() is self._scan_worker:
            self._scan_status.setText(f"Scanning… {i}/{t}")

    @Slot(bool, object)
    def _on_scan_finished(self, ok: bool, payload) -> None:
        if self.sender() is not self._scan_worker:
            return  # stale: cancelled by a library change
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
        text = self._map_edit.toPlainText()
        if text.strip() and self._mapping_is_dirty(self._library_path, text):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Replace mapping?",
                "Replace the mapping in the editor with the saved one?\n\n"
                "Your unsaved edits in the editor will be lost.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._map_edit.setPlainText(json.dumps(mapping, indent=2))
        self._map_status.setText(f"Loaded saved mapping from {path}")

    @Slot()
    def _on_save_mapping(self) -> None:
        mapping = self._current_mapping()
        if mapping is None:
            return
        path = nc.save_mapping(self._library_path, nc.clean_mapping(mapping))
        self._drafts.pop(self._library_path, None)
        self._map_status.setText(f"Saved mapping to {path}")
        self._log(f"Genre mapping saved to {path}", "ok")

    @Slot()
    def _on_preview(self) -> None:
        mapping = self._current_mapping()
        if mapping is None or self._plan_worker is not None or self._any_apply_running():
            return
        self._invalidate_plan("Previewing…")
        self._preview_btn.setEnabled(False)
        self._plan_mapping_text = self._map_edit.toPlainText()
        worker = _PlanWorker(self._library_path, mapping)
        worker.progress.connect(self._on_plan_progress)
        worker.finished.connect(self._on_plan_finished)
        self._plan_worker = worker
        self._start_worker(worker, "Genre preview")

    @Slot(int, int)
    def _on_plan_progress(self, i: int, t: int) -> None:
        if self.sender() is self._plan_worker:
            self._preview_status.setText(f"Reading tags… {i}/{t}")

    @Slot(bool, object)
    def _on_plan_finished(self, ok: bool, payload) -> None:
        if self.sender() is not self._plan_worker:
            return  # stale: cancelled by a library change
        self._plan_worker = None
        self._update_preview_gate()
        if not ok:
            self._invalidate_plan(f"Preview failed: {payload}")
            self._log(f"Genre preview failed: {payload}", "error")
            return
        if self._map_edit.toPlainText() != self._plan_mapping_text:
            self._invalidate_plan("Mapping changed — preview again before applying.")
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
        text = (
            f"{n} of {plan.with_genres} tagged files would change "
            f"({plan.unchanged} already canonical, {plan.scanned - plan.with_genres} with no genre tag)."
        )
        if plan.blocked:
            text += (
                f" {len(plan.blocked)} more would change but can't be written "
                "(read-only or unsupported) and are left out — see the log."
            )
            for path, reason in plan.blocked:
                self._log(f"  can't write {path}: {reason}", "warn")
        self._preview_status.setText(text)
        self._update_apply_gate()
        self._apply_btn.setText(f"✎  Apply {n} change{'s' if n != 1 else ''}")
        self._log(f"Genre preview: {n} file(s) would change.", "ok")

    def _update_apply_gate(self) -> None:
        self._apply_btn.setEnabled(
            bool(self._plan and self._plan.changes) and not self._any_apply_running()
        )

    @Slot()
    def _on_apply(self) -> None:
        if self._plan is None or not self._plan.changes or self._any_apply_running():
            return
        if self._map_edit.toPlainText() != self._plan_mapping_text:
            self._invalidate_plan("Mapping changed — preview again before applying.")
            return
        from controllers import tag_write_guard

        if self._library_writes_blocked("Genre normalization"):
            return
        n = len(self._plan.changes)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Apply genre changes",
            f"Rewrite the genre tag on {n} file(s) exactly as previewed?\n\n"
            "This edits the files in place; every change is logged with its previous "
            "value in Docs/tag_changes_genre_normalize_….jsonl. The mapping will also "
            "be saved to Docs/.genre_mapping.json so you can reuse it.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if not tag_write_guard.acquire(_NORMALIZE_OWNER):
            QtWidgets.QMessageBox.information(
                self, "Tags Busy", tag_write_guard.busy_message("Genre normalization")
            )
            return

        mapping = self._current_mapping()
        if mapping is not None:
            nc.save_mapping(self._library_path, nc.clean_mapping(mapping))
            self._drafts.pop(self._library_path, None)

        self._apply_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
        self._apply_cancel_btn.setEnabled(True)
        self._norm_bar.setValue(0)
        self._apply_status.setText("Writing…")
        self._emit_status("Normalizing genres…", "warning")

        worker = _ApplyWorker(
            list(self._plan.changes), self._library_path, "genre_normalize", _NORMALIZE_OWNER
        )
        worker.progress.connect(self._on_apply_progress)
        worker.finished.connect(self._on_apply_finished)
        self._apply_worker = worker
        self._update_fill_gate()
        self._start_worker(worker, _NORMALIZE_OWNER, library_write=True)

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
        worker = self.sender()
        if worker is not self._apply_worker:
            return
        self._apply_worker = None
        self._apply_cancel_btn.setEnabled(False)
        self._report_apply("Genre normalization", worker, ok, payload, self._apply_status)
        # The files on disk no longer match the plan; require a fresh preview.
        self._invalidate_plan("Applied. Preview again to see the library's new state.")
        self._update_preview_gate()
        self._update_fill_gate()
