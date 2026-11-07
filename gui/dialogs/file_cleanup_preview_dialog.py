"""Preview + confirm step for Utilities → File Cleanup (and its undo).

Shows every planned rename (folder, current name → new name) and every
collision that will be skipped, then asks for explicit confirmation. Only the
``rename`` rows of the plan shown here are executed. With ``undo=True`` the
plan is a ``file_cleanup.UndoPlan`` and the rows restore original names.
"""
from __future__ import annotations

import os

from gui.compat import QtWidgets


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
        self.setMinimumSize(760, 420)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        n_ren, n_col = len(plan.renames), len(plan.collisions)
        if undo:
            log_name = os.path.basename(getattr(plan, "log_path", "") or "")
            summary = (
                f"{n_ren} file(s) will get their original names back, undoing the "
                f"cleanup logged in Docs/{log_name}. "
                f"{n_col} file(s) can't be restored and will be skipped "
                "(nothing is overwritten)."
            )
        else:
            summary = (
                f"{n_ren} file(s) will be renamed in place. "
                f"{n_col} collision(s) will be skipped (nothing is overwritten). "
                f"{plan.scanned} audio files were checked; reserved folders "
                "(Quarantine, Trash, Docs, Playlists, Not Sorted, Manual Review) were skipped."
            )
        if plan.kept_by_title and not undo:
            summary += (
                f" {plan.kept_by_title} file(s) were left alone because the suffix is "
                "part of their title tag."
            )
        self._summary = QtWidgets.QLabel(summary)
        self._summary.setObjectName("cleanupPreviewSummary")
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        self.table = QtWidgets.QTableWidget(len(plan.items), len(columns))
        self.table.setObjectName("cleanupPreviewTable")
        self.table.setHorizontalHeaderLabels(list(columns))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
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
        root.addWidget(self.table, 1)

        if undo:
            note_text = (
                "Each restored name is recorded in the same log. Playlist entries "
                "that point at the cleaned names are pointed back at the original "
                "names (every changed playlist is backed up to Docs/playlist_backups/)."
            )
        else:
            note_text = (
                "Every rename is logged to Docs/file_cleanup_<time>.jsonl so it can be "
                "undone with \u2018Undo Last Cleanup\u2019 on the File Cleanup tile. "
                "Playlists that reference renamed files are updated (each changed "
                "playlist is backed up to Docs/playlist_backups/ first)."
            )
        note = QtWidgets.QLabel(note_text)
        note.setObjectName("statusHint")
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox()
        self.confirm_btn = buttons.addButton(
            f"Restore {n_ren} name(s)" if undo else f"Rename {n_ren} file(s)",
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.confirm_btn.setObjectName("cleanupConfirmBtn")
        self.confirm_btn.setEnabled(n_ren > 0)
        cancel = buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel.setDefault(True)  # Enter must not rename files
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def rows(self) -> list[tuple[str, str, str, str]]:
        """The table contents (for tests and accessibility)."""
        out = []
        for r in range(self.table.rowCount()):
            out.append(tuple(self.table.item(r, c).text() for c in range(self.table.columnCount())))
        return out
