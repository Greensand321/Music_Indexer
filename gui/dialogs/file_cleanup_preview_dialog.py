"""Preview + confirm step for Utilities → File Cleanup.

Shows every planned rename (folder, current name → new name) and every
collision that will be skipped, then asks for explicit confirmation. Only the
``rename`` rows of the plan shown here are executed.
"""
from __future__ import annotations

import os

from gui.compat import QtWidgets


class FileCleanupPreviewDialog(QtWidgets.QDialog):
    COLUMNS = ("Folder", "Current name", "New name", "Action")

    def __init__(self, plan, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plan = plan
        self.setWindowTitle("File Cleanup — Preview")
        self.setMinimumSize(760, 420)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        n_ren, n_col = len(plan.renames), len(plan.collisions)
        summary = (
            f"{n_ren} file(s) will be renamed in place. "
            f"{n_col} collision(s) will be skipped (nothing is overwritten). "
            f"{plan.scanned} audio files were checked; reserved folders "
            "(Quarantine, Trash, Docs, Playlists, Not Sorted, Manual Review) were skipped."
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

        self.table = QtWidgets.QTableWidget(len(plan.items), len(self.COLUMNS))
        self.table.setObjectName("cleanupPreviewTable")
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        # Renames first, then collisions.
        items = sorted(plan.items, key=lambda i: (not i.is_rename, i.src.lower()))
        for row, item in enumerate(items):
            folder = os.path.relpath(os.path.dirname(item.src), plan.library_root)
            action = "Rename" if item.is_rename else f"Skip — collision: {item.note}"
            for col, text in enumerate((
                folder, os.path.basename(item.src), os.path.basename(item.dst), action,
            )):
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setToolTip(text)
                self.table.setItem(row, col, cell)
        self.table.resizeColumnsToContents()
        root.addWidget(self.table, 1)

        note = QtWidgets.QLabel(
            "Every rename is logged to Docs/file_cleanup_<time>.jsonl so it can be "
            "undone. Playlists that reference renamed files are updated."
        )
        note.setObjectName("statusHint")
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox()
        self.confirm_btn = buttons.addButton(
            f"Rename {n_ren} file(s)", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
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
