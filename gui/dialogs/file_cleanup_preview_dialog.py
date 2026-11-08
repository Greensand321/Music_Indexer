"""Preview + confirm step for Organize → File Cleanup (and its undo).

Shows every planned rename (folder, current name → new name) and every
collision that will be skipped, then asks for explicit confirmation. Only the
``rename`` rows of the plan shown here are executed. With ``undo=True`` the
plan is a ``file_cleanup.UndoPlan`` and the rows restore original names.

The dialog window itself is an opaque top-level window (its chrome is the
shell's); its contents use the glass parts: a title, stat tiles for the
counts, the plan list on a card, a muted note and pill buttons.
"""
from __future__ import annotations

import os

from gui.compat import QtCore, QtWidgets
from gui.widgets.glass import GlassPanel
from gui.widgets.glass_parts import StatRow
from gui.widgets.glyphs import set_glyph
from gui.workspaces.organize_parts import caption, hero_title, style_item_view

_RESERVED_NOTE = (
    "Reserved folders (Quarantine, Trash, Docs, Playlists, Not Sorted, Manual Review) "
    "were not touched."
)


class FileCleanupPreviewDialog(QtWidgets.QDialog):
    COLUMNS = ("Folder", "Current name", "New name", "Action")
    UNDO_COLUMNS = ("Folder", "Current name", "Original name", "Action")

    def __init__(
        self, plan, parent: QtWidgets.QWidget | None = None, *, undo: bool = False
    ) -> None:
        super().__init__(parent)
        self.plan = plan
        self.undo = undo
        columns = self.UNDO_COLUMNS if undo else self.COLUMNS
        self.setWindowTitle("Undo File Cleanup — Preview" if undo else "File Cleanup — Preview")
        self.setMinimumSize(760, 460)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        n_ren, n_col = len(plan.renames), len(plan.collisions)
        root.addWidget(hero_title("Restore the original names" if undo else "Review the renames"))

        # The counts as tiles; the line under them only explains.
        tiles = [("renames", "to restore" if undo else "to rename"),
                 ("skipped", "skipped")]
        if not undo:
            tiles.append(("checked", "files checked"))
            if plan.kept_by_title:
                tiles.append(("kept", "kept (title)"))
        self.stats = StatRow(tiles)
        self.stats.setObjectName("cleanupPreviewStats")
        self.stats.tile("renames").set_value(n_ren, tone="accent" if n_ren else "")
        self.stats.tile("skipped").set_value(n_col, tone="warning" if n_col else "")
        self.stats.tile("skipped").setToolTip(
            "Files whose target name already exists (or, for an undo, can't be restored). "
            "They are skipped: nothing is ever overwritten."
        )
        if not undo:
            self.stats.tile("checked").set_value(plan.scanned)
            if plan.kept_by_title:
                self.stats.tile("kept").set_value(plan.kept_by_title)
                self.stats.tile("kept").setToolTip(
                    "Left alone because the suffix is part of the title tag "
                    "(e.g. “Love Song (2)”)."
                )
        root.addWidget(self.stats)

        if undo:
            log_name = os.path.basename(getattr(plan, "log_path", "") or "")
            summary = (
                f"Undoes the cleanup logged in Docs/{log_name}. Files that can't be restored "
                "are skipped; nothing is overwritten."
            )
        else:
            summary = (
                "Files are renamed in place. Collisions are skipped, nothing is overwritten. "
                + _RESERVED_NOTE
            )
            if plan.kept_by_title:
                summary += (
                    f" {plan.kept_by_title} file(s) were left alone because the suffix is "
                    "part of their title tag."
                )
        self._summary = QtWidgets.QLabel(summary)
        self._summary.setObjectName("cleanupPreviewSummary")
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        # The plan, on a card.
        card = GlassPanel(surface="card", radius=20)
        card.setObjectName("cleanupPreviewCard")
        cl = QtWidgets.QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(6)
        self.table = QtWidgets.QTableWidget(len(plan.items), len(columns))
        self.table.setObjectName("cleanupPreviewTable")
        self.table.setHorizontalHeaderLabels(list(columns))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        style_item_view(self.table, elide=QtCore.Qt.TextElideMode.ElideMiddle)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        # Renames first, then collisions.
        items = sorted(plan.items, key=lambda i: (not i.is_rename, i.src.lower()))
        for row, item in enumerate(items):
            try:
                folder = os.path.relpath(os.path.dirname(item.src), plan.library_root)
            except ValueError:  # another drive (an old log after the library moved)
                folder = os.path.dirname(item.src)
            if undo:
                action = "Restore" if item.is_rename else f"Skip — {item.note}"
            else:
                action = "Rename" if item.is_rename else f"Skip — collision: {item.note}"
            for col, text in enumerate((
                folder, os.path.basename(item.src), os.path.basename(item.dst), action,
            )):
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setToolTip(text)
                self.table.setItem(row, col, cell)
        self.table.resizeColumnsToContents()
        for col in range(len(columns) - 1):   # long folder paths elide instead of widening
            self.table.setColumnWidth(col, min(self.table.columnWidth(col), 260))
        cl.addWidget(self.table)
        root.addWidget(card, 1)

        if undo:
            note_text = (
                "Each restored name is recorded in the same log. Playlist entries that point "
                "at the cleaned names are pointed back (every changed playlist is backed up "
                "to Docs/playlist_backups/)."
            )
        else:
            note_text = (
                "Every rename is logged to Docs/file_cleanup_<time>.jsonl, so Undo last "
                "cleanup on the File Cleanup page can restore it. Playlists that point at a "
                "renamed file are updated (each changed playlist is backed up to "
                "Docs/playlist_backups/ first)."
            )
        note = caption(note_text)
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox()
        self.confirm_btn = buttons.addButton(
            f"Restore {n_ren} name(s)" if undo else f"Rename {n_ren} file(s)",
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole,
        )
        # The dialog's one accent button (it was "cleanupConfirmBtn", which
        # nothing styled or looked up). Enter still cancels: see below.
        self.confirm_btn.setObjectName("primaryBtn")
        self.confirm_btn.setAccessibleName(self.confirm_btn.text())
        set_glyph(self.confirm_btn, "undo" if undo else "check", role="inverse")
        self.confirm_btn.setEnabled(n_ren > 0)
        cancel = buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel.setDefault(True)  # Enter must not rename files
        self.confirm_btn.setAutoDefault(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def rows(self) -> list[tuple[str, str, str, str]]:
        """The table contents (for tests and accessibility)."""
        out = []
        for r in range(self.table.rowCount()):
            out.append(tuple(self.table.item(r, c).text() for c in range(self.table.columnCount())))
        return out
