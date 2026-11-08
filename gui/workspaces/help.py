"""Help workspace — documentation, shortcuts, and about info."""
from __future__ import annotations

import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


# (button label, path relative to the repository root). The maintained,
# plain-English docs suite lives in docs/ (index: docs/README.md). Every entry
# must exist — tests/ui/test_ui_tools_help.py checks it.
DOC_LINKS: tuple[tuple[str, str], ...] = (
    ("Documentation index", "docs/README.md"),
    ("Overview — what AlphaDEX is", "docs/overview.md"),
    ("Architecture", "docs/architecture.md"),
    ("Indexer", "docs/features/indexer.md"),
    ("Duplicate Finder", "docs/features/duplicate_finder.md"),
    ("Library Sync", "docs/features/library_sync.md"),
    ("Tag Fixer & Genres", "docs/features/tag_fixer.md"),
    ("Playlists, Clustering & Music Graph", "docs/features/playlists_and_clustering.md"),
    ("Screen-by-screen GUI inventory", "docs/gui_inventory.md"),
    ("Roadmap & known bugs", "docs/ROADMAP.md"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def doc_path(rel_path: str) -> Path:
    return REPO_ROOT / rel_path


class HelpWorkspace(WorkspaceBase):
    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Help & Documentation"))

        # ── Quick links ────────────────────────────────────────────────────
        links_card = self._make_card()
        links_layout = QtWidgets.QVBoxLayout(links_card)
        links_layout.setContentsMargins(16, 16, 16, 16)
        links_layout.setSpacing(8)
        links_layout.addWidget(QtWidgets.QLabel("Documentation"))

        self._doc_buttons: list[QtWidgets.QPushButton] = []
        for label, path in DOC_LINKS:
            btn = QtWidgets.QPushButton(f"📄  {label}")
            btn.setProperty("docPath", path)
            btn.setToolTip(path)
            btn.clicked.connect(lambda checked=False, p=path: self._open_doc(p))
            links_layout.addWidget(btn)
            self._doc_buttons.append(btn)
        cl.addWidget(links_card)

        # ── Keyboard shortcuts ─────────────────────────────────────────────
        shortcuts_card = self._make_card()
        shortcuts_layout = QtWidgets.QVBoxLayout(shortcuts_card)
        shortcuts_layout.setContentsMargins(16, 16, 16, 16)
        shortcuts_layout.addWidget(QtWidgets.QLabel("Keyboard Shortcuts"))

        shortcuts = [
            ("Ctrl+O", "Change Library"),
            ("Ctrl+,", "Settings page"),
            ("Ctrl+L", "Toggle Activity Log"),
            ("Ctrl+1–9", "Go to the first nine pages (sidebar order)"),
            ("Ctrl+W", "Clear activity log"),
        ]
        table = QtWidgets.QTableWidget(len(shortcuts), 2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for row, (key, action) in enumerate(shortcuts):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(action))
        table.setFixedHeight(len(shortcuts) * 30 + 28)
        shortcuts_layout.addWidget(table)
        cl.addWidget(shortcuts_card)

        # ── About card ─────────────────────────────────────────────────────
        about_card = self._make_card()
        about_layout = QtWidgets.QVBoxLayout(about_card)
        about_layout.setContentsMargins(16, 16, 16, 16)
        about_layout.addWidget(QtWidgets.QLabel("About AlphaDEX"))

        about_text = QtWidgets.QLabel(
            "AlphaDEX (formerly SoundVault) — Music Library Manager\n"
            "PySide6 Qt Widgets interface\n\n"
            "Config stored in: ~/.soundvault_config.json\n"
            "Supported formats: .flac  .m4a  .aac  .mp3  .wav  .ogg  .opus\n"
        )
        about_text.setObjectName("statusHint")  # themed muted text (was hard-coded slate)
        about_layout.addWidget(about_text)

        issues_btn = QtWidgets.QPushButton("Report an Issue (GitHub)")
        issues_btn.clicked.connect(
            lambda: webbrowser.open("https://github.com/Greensand321/Music_Indexer/issues")
        )
        about_layout.addWidget(issues_btn)
        cl.addWidget(about_card)

        cl.addStretch(1)

    @Slot()
    def _open_doc(self, rel_path: str) -> None:
        p = doc_path(rel_path)
        if p.exists():
            # as_uri() is correct on every platform ("file://" + a Windows
            # path with backslashes is not).
            webbrowser.open(p.as_uri())
        else:
            QtWidgets.QMessageBox.information(self, "Not Found", f"File not found:\n{p}")
