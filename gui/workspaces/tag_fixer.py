"""Tag Fixer workspace — repair metadata using AcoustID / MusicBrainz.

Flow: **Scan** (lookups only, never writes) → review grid with one row per
proposed *field* change → tick exactly the changes you want → **Apply
Selected** → confirm → exactly the ticked changes are written, each one
logged with its before/after value in ``Docs/tag_changes_tagfixer_*.jsonl``.

The grid is built from ``tag_fixer.planned_changes`` — the same function the
writer uses — so a row always shows precisely what applying it writes.
Files the lookups could not identify are listed separately ("No match"),
with the reason, instead of appearing as blank proposals.
"""
from __future__ import annotations

import dataclasses
import os
import threading
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

# Field names the backend understands, mapped from UI labels
_FIELD_MAP = {
    "Title": "title",
    "Artist": "artist",
    "Album": "album",
    "Genre": "genres",
}
_FIELD_LABELS = {v: k for k, v in _FIELD_MAP.items()}

_ROW_ROLE = QtCore.Qt.ItemDataRole.UserRole
_SCORE_COL = 5
_WRITE_OWNER = "Tag Fixer apply"

_SOURCES_HINT = "Lookup sources come from Settings → Metadata services."
_EMPTY_BEFORE = (
    "No fixes to review yet",
    "Find tag fixes looks each track up online and lists every proposed "
    "change here, one row per field.",
)
_EMPTY_NONE = (
    "Nothing to fix",
    "Every track the lookups matched already has these tags.",
)
_EMPTY_UNMATCHED = (
    "Nothing could be matched",
    "No track was identified online, so there is nothing to propose. "
    "The reason is shown above.",
)


def _fmt(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value) if value else "(none)"
    return "(empty)" if value in (None, "") else str(value)


# Note: no QTreeWidgetItem subclass with ``__lt__`` for numeric sorting —
# PySide6 6.12 segfaults when such an item is re-sorted after a check-state
# change. Scores are fixed two-decimal values in [0, 1], which sort correctly
# as text now that unmatched files (formerly "?") are not rows.


class TagFixWorker(QtCore.QThread):
    progress = Signal(int)          # percent, 0-100
    log_line = Signal(str)
    proposal_ready = Signal(list)   # every FileRecord of the scan
    finished = Signal(bool, str)

    def __init__(self, library_path: str, scan_root: str = "") -> None:
        super().__init__()
        self.library_path = library_path
        # The folder to look up: the library, or one folder inside it. The
        # database and change log always belong to the library.
        self.scan_root = scan_root or library_path
        self._cancel = threading.Event()
        # Populated during run(); read by workspace after finished
        self.db_path: str = ""
        self.all_records: list = []
        self.diagnostics = None

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            from controllers import tagfix_controller
            from tag_fixer import ScanDiagnostics
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            def _log(msg: str) -> None:
                if not self.is_cancelled():
                    self.log_line.emit(msg)

            self.db_path, _ = tagfix_controller.prepare_library(self.library_path)
            if self.is_cancelled():
                self.finished.emit(False, "Cancelled.")
                return

            self.diagnostics = ScanDiagnostics()
            records = tagfix_controller.gather_records(
                self.scan_root,
                self.db_path,
                show_all=True,
                progress_callback=None,
                log_callback=_log,
                # A real percentage (it used to send the file *index* to a
                # 0-100 bar, which sat at 100% from file 100 onwards).
                progress_detail=lambda i, t: self.progress.emit(int(i * 100 / max(t, 1))),
                should_cancel=self.is_cancelled,
                diagnostics=self.diagnostics,
            )
            if self.is_cancelled() or self.diagnostics.cancelled:
                self.finished.emit(False, "Cancelled.")
                return

            self.all_records = records
            self.proposal_ready.emit(records)
            self.finished.emit(True, "Scan complete.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class TagFixApplyWorker(QtCore.QThread):
    log_line = Signal(str)
    progress = Signal(int)
    result_ready = Signal(object)   # tag_fixer.TagApplyResult
    finished = Signal(bool, str)

    def __init__(
        self,
        selections: list,
        all_records: list,
        db_path: str,
        library_root: str,
    ) -> None:
        super().__init__()
        self.selections = selections      # [(FileRecord, [field, ...]), ...]
        self.all_records = all_records
        self.db_path = db_path
        self.library_root = library_root
        self._cancel = threading.Event()
        self.result = None

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        from controllers import tag_write_guard

        try:
            try:
                from controllers.tagfix_controller import apply_selected_fields
            except ImportError as exc:
                self.finished.emit(False, f"Import error: {exc}")
                return
            try:
                result = apply_selected_fields(
                    self.selections,
                    self.all_records,
                    self.db_path,
                    log_callback=self.log_line.emit,
                    should_cancel=self.is_cancelled,
                    progress_callback=lambda i, t: self.progress.emit(int(i * 100 / max(t, 1))),
                    library_root=self.library_root,
                )
            except Exception as exc:  # noqa: BLE001
                self.finished.emit(False, str(exc))
                return
            self.result = result
            self.result_ready.emit(result)
            parts = [f"Applied tags to {len(result.applied)} file(s)"]
            if result.failed:
                parts.append(f"{len(result.failed)} failed")
            if result.skipped:
                parts.append(f"{len(result.skipped)} skipped (changed since the scan)")
            message = ", ".join(parts)
            if result.cancelled:
                message += " — cancelled"
            self.finished.emit(True, message + ".")
        finally:
            # Released here, not in a GUI slot, so the claim can never be
            # left behind if the page is gone before the slot runs.
            tag_write_guard.release(_WRITE_OWNER)


class TagFixerWorkspace(WorkspaceBase):
    """Repair metadata tags using AcoustID fingerprint lookups."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._records: list = []          # every FileRecord of the last scan
        self._all_records: list = []      # (kept name) same list, for the DB update
        self._proposals: list = []        # records with at least one planned change
        self._rows: list = []             # row index -> (FileRecord, field)
        self._db_path: str = ""
        self._scan_library: str = ""      # library the shown proposals belong to
        self._scope_folder: str = ""      # "" = the whole library
        self._scanned_once = False        # the review card shows "Nothing to fix"
        self._worker: TagFixWorker | None = None
        self._apply_worker: TagFixApplyWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        from gui.widgets.glass import ElidedLabel
        from gui.widgets.glass_parts import (
            EmptyState, StatRow, card_heading, status_label, thin_bar,
        )
        from gui.widgets.glass_views import GlassList, GlassTree, stretch_columns
        from gui.widgets.glyphs import set_glyph
        from gui.widgets.layouts import FlowLayout
        from gui.widgets.pill import Pill

        cl = self.content_layout
        cl.addWidget(self._make_subtitle(
            "Fix artist, title, album and genre tags from online lookups, "
            "reviewed field by field and written only when you apply."
        ))

        # ── Hero: what to check, and the main action ──────────────────────
        hero = self._make_card()
        hero.setObjectName("headerCard")
        hl = QtWidgets.QVBoxLayout(hero)
        hl.setContentsMargins(16, 16, 16, 16)
        hl.setSpacing(12)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)
        heading, _title, self._reason_lbl = card_heading("Tag lookup", " ", hero=True)
        # The sources line: which lookup services ran, and why nothing
        # matched when nothing did (filled after each scan).
        self._reason_lbl.setObjectName("statusHint")
        self._reason_lbl.setText(_SOURCES_HINT)
        top.addWidget(heading, 1)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setFlat(True)
        self._cancel_btn.setToolTip("Stop the scan or the apply at the next file")
        set_glyph(self._cancel_btn, "close")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._scan_btn = self._make_primary_button("Find tag fixes")
        self._scan_btn.setToolTip("Look every track in the scope up online. Never writes.")
        set_glyph(self._scan_btn, "search", role="inverse")
        self._scan_btn.clicked.connect(self._on_scan)
        top.addWidget(self._cancel_btn, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        top.addWidget(self._scan_btn, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        hl.addLayout(top)

        # Scope: the whole library, or one folder inside it.
        scope = QtWidgets.QHBoxLayout()
        scope.setSpacing(8)
        self._scope_lib_pill = Pill("Whole library", "accent",
                                    "Every track in the library (reserved folders skipped)")
        self._scope_dir_pill = Pill("One folder", "success", "Only the tracks in this folder")
        self._scope_path = ElidedLabel("", QtCore.Qt.TextElideMode.ElideMiddle)
        self._scope_path.setObjectName("mutedLabel")
        self._scope_path.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Preferred)
        self._scope_btn = QtWidgets.QPushButton("Choose folder…")
        self._scope_btn.setToolTip("Check one folder inside the library instead of all of it")
        set_glyph(self._scope_btn, "folder")
        self._scope_btn.clicked.connect(self._on_choose_folder)
        self._scope_reset_btn = QtWidgets.QPushButton("Whole library")
        self._scope_reset_btn.setFlat(True)
        self._scope_reset_btn.setToolTip("Check the whole library again")
        set_glyph(self._scope_reset_btn, "refresh")
        self._scope_reset_btn.clicked.connect(self._on_whole_library)
        for w in (self._scope_lib_pill, self._scope_dir_pill):
            scope.addWidget(w, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        scope.addWidget(self._scope_path, 1)
        scope.addWidget(self._scope_reset_btn)
        scope.addWidget(self._scope_btn)
        hl.addLayout(scope)

        foot = QtWidgets.QHBoxLayout()
        foot.setSpacing(10)
        self._prog_status = status_label("Idle")
        foot.addWidget(self._prog_status, 1)
        self._sources_btn = QtWidgets.QPushButton("Lookup sources")
        self._sources_btn.setFlat(True)
        self._sources_btn.setToolTip("Choose the lookup services in Settings → Metadata services")
        set_glyph(self._sources_btn, "external", role="muted")
        self._sources_btn.clicked.connect(self._on_open_sources)
        foot.addWidget(self._sources_btn, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        hl.addLayout(foot)
        self._prog_bar = thin_bar()
        hl.addWidget(self._prog_bar)
        cl.addWidget(hero)

        # ── Review: one row per proposed field change ─────────────────────
        review = self._make_card()
        rl = QtWidgets.QVBoxLayout(review)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(12)
        heading, _t, _s = card_heading(
            "Review fixes", "Each row is one field of one file. Apply writes exactly the ticked rows.")
        rl.addWidget(heading)

        self._stats = StatRow([
            ("proposed", "Proposed changes"), ("files", "Files to fix"),
            ("ticked", "Ticked to write"), ("nomatch", "No match"),
        ])
        self._stats.setVisible(False)
        rl.addWidget(self._stats)

        # Field filter: shows/hides that field's rows. Apply writes exactly
        # the rows that are visible *and* ticked.
        self._filter_row = QtWidgets.QWidget()
        fields_row = FlowLayout(self._filter_row, spacing=14, line_spacing=8)
        show = QtWidgets.QLabel("Show")
        show.setObjectName("mutedLabel")
        fields_row.addWidget(show)
        self._field_cbs: dict[str, QtWidgets.QCheckBox] = {}
        for field in ("Title", "Artist", "Album", "Genre"):
            cb = QtWidgets.QCheckBox(field)
            cb.setChecked(True)
            cb.setToolTip(f"Show {field.lower()} proposals; hidden rows are never written.")
            cb.toggled.connect(self._apply_field_filter)
            fields_row.addWidget(cb)
            self._field_cbs[field] = cb
        self._filter_row.setVisible(False)
        rl.addWidget(self._filter_row)

        self._empty = EmptyState(*_EMPTY_BEFORE)
        rl.addWidget(self._empty)

        self._table = GlassTree()
        self._table.setObjectName("tagFixerProposals")
        self._table.setHeaderLabels(["", "File", "Field", "Current", "Proposed", "Score"])
        self._table.headerItem().setToolTip(0, "Ticked rows are written when you apply")
        stretch_columns(self._table.header(), {
            0: 40, 1: "stretch", 2: "contents", 3: "stretch", 4: "stretch", 5: 64,
        })
        self._table.setSortingEnabled(True)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.setVisible(False)
        rl.addWidget(self._table)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(8)
        self._select_all_btn = QtWidgets.QPushButton("Tick all")
        self._select_all_btn.setFlat(True)
        self._select_all_btn.setToolTip("Tick every shown row")
        set_glyph(self._select_all_btn, "check")
        self._select_all_btn.setEnabled(False)
        self._select_all_btn.clicked.connect(self._on_select_all)
        self._deselect_btn = QtWidgets.QPushButton("Untick all")
        self._deselect_btn.setFlat(True)
        self._deselect_btn.setToolTip("Untick every shown row")
        set_glyph(self._deselect_btn, "minus")
        self._deselect_btn.setEnabled(False)
        self._deselect_btn.clicked.connect(self._on_deselect_all)
        self._summary_lbl = status_label("")
        self._dry_run_cb = QtWidgets.QCheckBox("Dry run")
        self._dry_run_cb.setToolTip(
            "While ticked, Apply only reminds you that nothing is written. "
            "Untick it to write the ticked changes."
        )
        self._dry_run_cb.setChecked(True)
        self._apply_btn = QtWidgets.QPushButton("Apply changes")
        self._apply_btn.setObjectName("successBtn")
        self._apply_btn.setToolTip("Write exactly the ticked rows (asks first)")
        set_glyph(self._apply_btn, "check", role="inverse")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        actions.addWidget(self._select_all_btn)
        actions.addWidget(self._deselect_btn)
        actions.addStretch(1)
        actions.addWidget(self._dry_run_cb)
        actions.addWidget(self._apply_btn)
        rl.addWidget(self._summary_lbl)
        rl.addLayout(actions)
        cl.addWidget(review)

        # ── Files with no match: listed here, never as blank proposals ────
        self._nomatch_card = self._make_card()
        nl = QtWidgets.QVBoxLayout(self._nomatch_card)
        nl.setContentsMargins(16, 16, 16, 16)
        nl.setSpacing(10)
        heading, self._nomatch_title, _s = card_heading(
            "No match found", "Nothing is written to these files.")
        nl.addWidget(heading)
        self._nomatch_list = GlassList(max_rows=5)
        nl.addWidget(self._nomatch_list)
        self._nomatch_card.setVisible(False)
        cl.addWidget(self._nomatch_card)

        # ── Log ────────────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.setSpacing(10)
        log_layout.addWidget(card_heading("Log")[0])
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setObjectName("logText")
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(100)
        self._log_area.setMaximumHeight(180)
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)
        self._refresh_scope()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _emit_status(self, text: str, role: str) -> None:
        tokens = get_manager().current
        self.status_changed.emit(text, getattr(tokens, role))

    def _busy(self) -> bool:
        return self._worker is not None or self._apply_worker is not None

    def _update_buttons(self) -> None:
        busy = self._busy()
        has_rows = self._table.topLevelItemCount() > 0
        self._scan_btn.setEnabled(not busy)
        self._apply_btn.setEnabled(has_rows and not busy)
        self._select_all_btn.setEnabled(has_rows and not busy)
        self._deselect_btn.setEnabled(has_rows and not busy)
        self._cancel_btn.setEnabled(busy)
        self._scope_btn.setEnabled(not busy and bool(self._library_path))
        self._scope_reset_btn.setEnabled(not busy)

    def _clear_results(self) -> None:
        self._table.clear()
        self._rows = []
        self._records = []
        self._all_records = []
        self._proposals = []
        self._db_path = ""
        self._scan_library = ""
        self._scanned_once = False
        self._summary_lbl.setText("")
        self._reason_lbl.setText(_SOURCES_HINT)
        self._nomatch_list.clear()
        self._nomatch_card.setVisible(False)
        self._show_results()

    def _show_results(self) -> None:
        """Swap the empty state and the review grid."""
        has_rows = self._table.topLevelItemCount() > 0
        self._table.setVisible(has_rows)
        self._filter_row.setVisible(has_rows)
        self._stats.setVisible(bool(self._records))
        self._empty.setVisible(not has_rows)
        if not self._scanned_once or not self._records:
            self._empty.set_text(*_EMPTY_BEFORE)
        elif all(r.status == "unmatched" for r in self._records):
            self._empty.set_text(*_EMPTY_UNMATCHED)
        else:
            self._empty.set_text(*_EMPTY_NONE)

    # ── Scope: the whole library or one folder in it ──────────────────────

    def _scan_root(self) -> str:
        return self._scope_folder or self._library_path

    def _refresh_scope(self) -> None:
        folder = bool(self._scope_folder)
        self._scope_lib_pill.setVisible(not folder)
        self._scope_dir_pill.setVisible(folder)
        self._scope_reset_btn.setVisible(folder)
        path = self._scan_root()
        self._scope_path.setText(path or "No library selected")
        self._scope_path.setToolTip(path)
        self._update_buttons()

    def _scope_problem(self, folder: str) -> str:
        """Why ``folder`` can't be the scope ("" when it can)."""
        from tag_fixer import TAG_SCAN_SKIPPED_FOLDERS
        from utils.reserved_folders import is_reserved_path

        library = os.path.abspath(self._library_path)
        target = os.path.abspath(folder)
        try:
            inside = os.path.commonpath(
                [os.path.normcase(library), os.path.normcase(target)]
            ) == os.path.normcase(library)
        except ValueError:      # another drive on Windows
            inside = False
        if not inside:
            return "Choose a folder inside the library. The Tag Fixer only checks your library."
        if is_reserved_path(target, library, TAG_SCAN_SKIPPED_FOLDERS):
            return (
                "That folder is one the Tag Fixer never checks or changes "
                f"({', '.join(TAG_SCAN_SKIPPED_FOLDERS)}). Choose another one."
            )
        return ""

    def _set_scope(self, folder: str) -> None:
        if folder and os.path.normcase(os.path.abspath(folder)) == os.path.normcase(
                os.path.abspath(self._library_path or "")):
            folder = ""
        self._scope_folder = os.path.abspath(folder) if folder else ""
        self._refresh_scope()

    @Slot()
    def _on_whole_library(self) -> None:
        self._set_scope("")

    @Slot()
    def _on_choose_folder(self) -> None:
        if not self._library_path or self._busy():
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a folder to check", self._scan_root()
        )
        if not folder:
            return
        problem = self._scope_problem(folder)
        if problem:
            QtWidgets.QMessageBox.information(self, "Choose Another Folder", problem)
            return
        self._set_scope(folder)

    @Slot()
    def _on_open_sources(self) -> None:
        self.navigate_requested.emit("settings/services")

    def _selected_changes(self) -> list:
        """``[(FileRecord, [field, ...])]`` for every visible, ticked row.

        Keyed by the record's identity: ``FileRecord`` is an ``eq=True``
        dataclass and therefore unhashable (putting it in a set raised
        ``TypeError`` and Apply could never write anything).
        """
        by_record: dict[int, tuple] = {}
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            if item is None or item.isHidden():
                continue
            if item.checkState(0) != QtCore.Qt.CheckState.Checked:
                continue
            index = item.data(0, _ROW_ROLE)
            if index is None or not (0 <= int(index) < len(self._rows)):
                continue
            rec, field = self._rows[int(index)]
            by_record.setdefault(id(rec), (rec, []))[1].append(field)
        return list(by_record.values())

    def _render_rows(self, keep_unticked: set | None = None) -> None:
        """(Re)build the grid from ``self._records`` via planned_changes."""
        from tag_fixer import planned_changes
        from gui.widgets.glass_views import mark

        keep_unticked = keep_unticked or set()
        self._table.blockSignals(True)
        sorting = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        self._table.clear()
        self._rows = []
        self._proposals = []
        unmatched = []
        for rec in self._records:
            if rec.status == "unmatched":
                unmatched.append(rec)
                continue
            changes = planned_changes(rec)
            if not changes:
                continue
            self._proposals.append(rec)
            score_str = f"{rec.score:.2f}" if rec.score is not None else ""
            for field_name, (before, after) in changes.items():
                row = QtWidgets.QTreeWidgetItem([
                    "",
                    Path(rec.path).name,
                    _FIELD_LABELS.get(field_name, field_name),
                    _fmt(before),
                    _fmt(after),
                    score_str,
                ])
                index = len(self._rows)
                self._rows.append((rec, field_name))
                row.setData(0, _ROW_ROLE, index)
                row.setData(_SCORE_COL, _ROW_ROLE, float(rec.score or 0))
                ticked = (id(rec), field_name) not in keep_unticked
                row.setCheckState(
                    0, QtCore.Qt.CheckState.Checked if ticked else QtCore.Qt.CheckState.Unchecked
                )
                row.setToolTip(1, str(rec.path))
                row.setToolTip(3, _fmt(before))
                row.setToolTip(4, _fmt(after))
                # Current values muted; the value that would be written on
                # an accent-tinted chip (painted from the theme, no colours here).
                mark(row, "muted", 3)
                mark(row, "accent", 4)
                mark(row, "muted", _SCORE_COL)
                self._table.addTopLevelItem(row)
        self._table.setSortingEnabled(sorting)
        self._table.blockSignals(False)
        self._apply_field_filter()

        self._nomatch_list.clear()
        for rec in unmatched:
            item = QtWidgets.QListWidgetItem(
                f"{Path(rec.path).name} — {rec.note or 'no match found'}"
            )
            item.setToolTip(str(rec.path))
            self._nomatch_list.addItem(item)
        self._nomatch_title.setText(f"No match found ({len(unmatched)})")
        self._nomatch_card.setVisible(bool(unmatched))
        self._show_results()
        self._update_summary()

    def _update_summary(self) -> None:
        total_rows = self._table.topLevelItemCount()
        if not self._records:
            self._summary_lbl.setText("")
            self._apply_btn.setText("Apply changes")
            return
        selected = self._selected_changes()
        n_fields = sum(len(f) for _r, f in selected)
        unmatched = sum(1 for r in self._records if r.status == "unmatched")
        self._stats.tile("proposed").set_value(total_rows)
        self._stats.tile("files").set_value(len(self._proposals))
        self._stats.tile("ticked").set_value(n_fields, tone="accent" if n_fields else "")
        self._stats.tile("nomatch").set_value(unmatched, tone="warning" if unmatched else "")
        if not total_rows:
            self._summary_lbl.setText("")
        elif n_fields:
            self._summary_lbl.setText(f"Writes {n_fields} ticked in {len(selected)} file(s)")
        else:
            self._summary_lbl.setText("Nothing ticked")
        self._apply_btn.setText(
            f"Apply {n_fields} change{'s' if n_fields != 1 else ''}" if n_fields else "Apply changes"
        )

    # ── Library change ────────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        # Proposals belong to the library they were scanned in; applying
        # them after a switch would write the *old* library's files. A scan
        # in flight is cancelled and its result discarded. A confirmed apply
        # is allowed to finish (it writes exactly what was confirmed) and its
        # outcome is logged.
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
            self._prog_bar.setValue(0)
        had_results = bool(self._records)
        self._clear_results()
        self._scope_folder = ""            # a folder belongs to the old library
        self._refresh_scope()
        if self._apply_worker is None:
            self._prog_status.setText(
                "Library changed — scan again." if had_results else "Idle"
            )
        self._update_buttons()

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        if self._busy():
            return

        self._clear_results()
        self._log_area.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Scanning…")
        self._scan_library = self._library_path
        where = f" of {self._scope_folder}" if self._scope_folder else ""
        self._log(f"Starting tag scan{where}…", "info")
        self._emit_status("Scanning tags…", "warning")

        worker = TagFixWorker(self._library_path, self._scope_folder)
        worker.progress.connect(self._prog_bar.setValue)
        worker.log_line.connect(self._on_log_line)
        worker.proposal_ready.connect(self._on_proposals)
        worker.finished.connect(self._on_scan_finished)
        self._worker = worker
        self._update_buttons()
        self._start_worker(worker, "Tag Fixer scan")

    @Slot()
    def _on_apply(self) -> None:
        if self._dry_run_cb.isChecked():
            QtWidgets.QMessageBox.information(
                self, "Dry Run",
                "Dry run is enabled — uncheck it to write tags."
            )
            return
        if self._busy() or not self._db_path:
            return

        selections = self._selected_changes()
        if not selections:
            QtWidgets.QMessageBox.information(self, "Nothing Selected", "No proposals are checked.")
            return

        from controllers import tag_write_guard

        if self._library_writes_blocked("Tag Fixer apply"):
            return
        n_fields = sum(len(fields) for _rec, fields in selections)
        reply = QtWidgets.QMessageBox.question(
            self, "Apply Tags",
            f"Write {n_fields} ticked tag change(s) to {len(selections)} file(s), "
            "exactly as shown in the table?\n\n"
            "The files are edited in place. Every change is logged with its "
            "previous value in Docs/tag_changes_tagfixer_….jsonl.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if not tag_write_guard.acquire(_WRITE_OWNER):
            QtWidgets.QMessageBox.information(
                self, "Tags Busy", tag_write_guard.busy_message("Tag Fixer apply")
            )
            return

        self._prog_bar.setValue(0)
        self._prog_status.setText("Applying…")
        self._emit_status("Applying tags…", "warning")

        # The DB status pass rewrites ``status`` on the records it is given;
        # hand it copies so the worker thread never mutates the objects this
        # page displays (it turned "unmatched" into "skipped", emptying the
        # No-match list after an apply).
        status_copies = [dataclasses.replace(rec) for rec in self._records]
        worker = TagFixApplyWorker(
            selections, status_copies, self._db_path, self._scan_library
        )
        worker.log_line.connect(self._on_log_line)
        worker.progress.connect(self._prog_bar.setValue)
        worker.result_ready.connect(self._on_apply_result)
        worker.finished.connect(self._on_apply_finished)
        self._apply_worker = worker
        self._update_buttons()
        self._start_worker(worker, _WRITE_OWNER, library_write=True)

    @Slot()
    def _on_select_all(self) -> None:
        self._set_all_checked(QtCore.Qt.CheckState.Checked)

    @Slot()
    def _on_deselect_all(self) -> None:
        self._set_all_checked(QtCore.Qt.CheckState.Unchecked)

    def _set_all_checked(self, state: QtCore.Qt.CheckState) -> None:
        self._table.blockSignals(True)
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            if item and not item.isHidden():
                item.setCheckState(0, state)
        self._table.blockSignals(False)
        self._update_summary()

    def _apply_field_filter(self, *_args) -> None:
        shown = {_FIELD_MAP[label] for label, cb in self._field_cbs.items() if cb.isChecked()}
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            index = item.data(0, _ROW_ROLE) if item is not None else None
            if index is None or not (0 <= int(index) < len(self._rows)):
                continue
            _rec, field = self._rows[int(index)]
            item.setHidden(field not in shown)
        self._table.fit_rows(10, 3)
        self._update_summary()

    @Slot(QtWidgets.QTreeWidgetItem, int)
    def _on_item_changed(self, _item, column: int) -> None:
        if column == 0:
            self._update_summary()

    @Slot()
    def _on_cancel(self) -> None:
        for worker in (self._worker, self._apply_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._prog_status.setText("Cancelling…")

    @Slot(list)
    def _on_proposals(self, records: list) -> None:
        if self.sender() is not self._worker:
            return  # a scan cancelled by a library change; its result is stale
        self._records = list(records)
        self._all_records = self._records
        self._scanned_once = True
        self._render_rows()

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._worker and sender is not self._apply_worker:
            return
        self._log_area.appendPlainText(line)

    @Slot(bool, str)
    def _on_scan_finished(self, success: bool, message: str) -> None:
        worker = self.sender()
        if worker is not self._worker:
            return  # stale (cancelled by a library change)
        self._worker = None
        self._db_path = worker.db_path if success else ""
        diag = getattr(worker, "diagnostics", None)
        if success:
            self._prog_bar.setValue(100)
            n_rows = self._table.topLevelItemCount()
            message = (
                f"{n_rows} change(s) proposed for {len(self._proposals)} file(s)."
                if n_rows else "No tag changes to propose."
            )
            self._log(message, "ok")
            self._emit_status("Scan complete", "success")
        else:
            self._clear_results()
            self._log(message, "error" if message != "Cancelled." else "warn")
            self._emit_status("Cancelled" if message == "Cancelled." else "Error",
                              "warning" if message == "Cancelled." else "danger")
        self._prog_status.setText(message)
        if diag is not None and success:
            lines = [diag.sources_line()]
            if diag.unmatched:
                lines.append(f"{diag.unmatched} file(s) unmatched: {diag.unmatched_reason()}")
            self._reason_lbl.setText("\n".join(lines))
        self._update_buttons()

    @Slot(object)
    def _on_apply_result(self, result) -> None:
        if self.sender() is not self._apply_worker:
            return
        worker = self._apply_worker
        if worker is None or worker.library_root != self._scan_library:
            return  # the library changed during the apply; the grid is gone
        # Update the records in place for every field that was written, so
        # those rows drop out; failed / unticked rows stay for review.
        unticked = set()
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            index = item.data(0, _ROW_ROLE)
            if index is None or not (0 <= int(index) < len(self._rows)):
                continue
            rec, field = self._rows[int(index)]
            if item.checkState(0) != QtCore.Qt.CheckState.Checked:
                unticked.add((id(rec), field))
        from tag_fixer import planned_changes

        written = getattr(result, "written_fields", {}) or {}
        for rec in self._records:
            fields = written.get(str(rec.path))
            if not fields:
                continue
            for field, (_before, after) in planned_changes(rec, fields).items():
                if field == "genres":
                    rec.old_genres = list(after)
                else:
                    setattr(rec, f"old_{field}", after)
            if not planned_changes(rec):
                rec.status = "applied"
        self._render_rows(unticked)
        if result.failed:
            self._reason_lbl.setText(
                f"{len(result.failed)} file(s) could not be written — their rows are "
                "still listed; see the log for why."
            )

    @Slot(bool, str)
    def _on_apply_finished(self, success: bool, message: str) -> None:
        worker = self.sender()
        if worker is not self._apply_worker:
            return
        self._apply_worker = None
        result = getattr(worker, "result", None)
        log_path = getattr(result, "log_path", "") if result is not None else ""
        if log_path:
            message = f"{message} Change log: {os.path.relpath(log_path, worker.library_root)}"
        self._prog_status.setText(message)
        if worker.library_root != self._library_path:
            message = f"{message} (library: {worker.library_root})"
        if success:
            failed = bool(result and (result.failed or result.skipped))
            self._log(message, "warn" if failed else "ok")
            self._emit_status("Tags applied", "warning" if failed else "success")
        else:
            self._log(message, "error")
            self._emit_status("Error", "danger")
        # No automatic rescan: it cleared the log (hiding per-file failures)
        # and re-queried every file online. Written rows were updated above.
        self._update_buttons()
