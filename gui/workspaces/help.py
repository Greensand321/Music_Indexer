"""Help page: the documentation, the keyboard shortcuts, and about AlphaDEX.

Three glass cards: **Documentation** (a link per maintained doc, in the
accent colour, opening in the browser), **Keyboard shortcuts** (the
app-wide ones, as key chips) and **About AlphaDEX** (config location,
supported formats, and a link to report an issue). Page names follow
``gui/navigation.py``.
"""
from __future__ import annotations

import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Slot
from gui.workspaces.base import WorkspaceBase


# (link label, path relative to the repository root). The maintained,
# plain-English docs suite lives in docs/ (index: docs/README.md). Every entry
# must exist — tests/ui/test_ui_tools_help_media.py checks it.
DOC_LINKS: tuple[tuple[str, str], ...] = (
    ("Documentation index", "docs/README.md"),
    ("Overview", "docs/overview.md"),
    ("Architecture", "docs/architecture.md"),
    ("Indexer", "docs/features/indexer.md"),
    ("Library Sync", "docs/features/library_sync.md"),
    ("Duplicates", "docs/features/duplicate_finder.md"),
    ("Metadata: Tag Fixer and genres", "docs/features/tag_fixer.md"),
    ("Sound Map and Playlists", "docs/features/playlists_and_clustering.md"),
    ("Every page and control", "docs/gui_inventory.md"),
    ("Roadmap and known bugs", "docs/ROADMAP.md"),
)

# One line under each link: what the document is for.
DOC_NOTES: dict[str, str] = {
    "docs/README.md": "Where to start, and the reading order",
    "docs/overview.md": "What AlphaDEX does and the ideas behind it",
    "docs/architecture.md": "How the parts fit together",
    "docs/features/indexer.md": "Sort folders and file names, preview first",
    "docs/features/library_sync.md": "Bring an incoming folder into the library",
    "docs/features/duplicate_finder.md": "Find, review and quarantine duplicates",
    "docs/features/tag_fixer.md": "Fix tags, fill and normalize genres, undo changes",
    "docs/features/playlists_and_clustering.md": "Analyse and map how tracks sound; build playlists",
    "docs/gui_inventory.md": "A map of every page, section and control",
    "docs/ROADMAP.md": "What is planned, and what is known to be broken",
}

# (keys, what they do): the app-wide shortcuts (gui/main_window.py). The
# Player page's own keys (gui/workspaces/player.py) are named in one line.
SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("Ctrl+O", "Change library"),
    ("Ctrl+,", "Settings"),
    ("Ctrl+1–9", "The first nine pages, in sidebar order"),
    ("Ctrl+L", "Show or hide the Activity Log"),
    ("Ctrl+W", "Clear the Activity Log"),
    ("Ctrl+Q", "Quit"),
)

ISSUES_URL = "https://github.com/Greensand321/Music_Indexer/issues"

REPO_ROOT = Path(__file__).resolve().parents[2]


def doc_path(rel_path: str) -> Path:
    return REPO_ROOT / rel_path


class _KeyChip(QtWidgets.QWidget):
    """A key combination as a small raised pill (mono caption text)."""

    def __init__(self, keys: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._keys = keys
        self.setAccessibleName(keys)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def keys(self) -> str:
        return self._keys

    def _font(self) -> QtGui.QFont:
        from gui.themes.glass import font
        return font("mono", px=12)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        fm = QtGui.QFontMetrics(self._font())
        return QtCore.QSize(fm.horizontalAdvance(self._keys) + 20, fm.height() + 10)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface, qcolor
        from gui.themes.manager import get_manager

        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().raised, radius=8)
            p.setFont(self._font())
            p.setPen(qcolor(get_manager().current.text_primary))
            p.drawText(self.rect(), int(QtCore.Qt.AlignmentFlag.AlignCenter), self._keys)
        finally:
            p.end()


class HelpWorkspace(WorkspaceBase):
    TWO_COLUMNS_AT = 860   # content width for shortcuts and about side by side

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._columns = 0
        self._build_ui()
        self._inner.installEventFilter(self)

    # ── UI ────────────────────────────────────────────────────────────────

    def _card(self, title: str, subtitle: str = "") -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
        card = self._make_card()
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        lay.addWidget(self._make_card_title(title))
        if subtitle:
            lay.addWidget(self._muted(subtitle))
        return card, lay

    @staticmethod
    def _muted(text: str, role: str = "label") -> QtWidgets.QLabel:
        from gui.themes.glass import apply_font

        label = QtWidgets.QLabel(text)
        label.setObjectName("mutedLabel")
        label.setWordWrap(True)
        apply_font(label, role)
        return label

    def _build_ui(self) -> None:
        from gui.themes.glass import apply_font
        from gui.themes.tokens import SPACE
        from gui.widgets.link_button import LinkButton

        cl = self.content_layout
        cl.addWidget(self._make_subtitle(
            "Guides to every page, the keyboard shortcuts, and where AlphaDEX keeps its settings."))

        # ── Documentation ──────────────────────────────────────────────────
        docs_card, lay = self._card("Documentation",
                                    "Plain-English guides; they open in your browser.")
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(SPACE.lg)
        grid.setVerticalSpacing(10)
        self._doc_buttons: list[LinkButton] = []
        for i, (label, path) in enumerate(DOC_LINKS):
            cell = QtWidgets.QVBoxLayout()
            cell.setSpacing(0)
            btn = LinkButton(label, "file", role="body")
            btn.setProperty("docPath", path)
            btn.setToolTip(path)
            btn.setAccessibleName(f"Open {label}")
            btn.clicked.connect(lambda checked=False, p=path: self._open_doc(p))
            cell.addWidget(btn, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            note = QtWidgets.QLabel(DOC_NOTES.get(path, ""))
            note.setObjectName("mutedLabel")
            note.setWordWrap(True)
            apply_font(note, "caption")
            note.setContentsMargins(24, 0, 0, 0)
            cell.addWidget(note)
            grid.addLayout(cell, i // 2, i % 2)
            self._doc_buttons.append(btn)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        cl.addWidget(docs_card)

        # ── Keyboard shortcuts ─────────────────────────────────────────────
        self._shortcuts_card, lay = self._card("Keyboard shortcuts")
        self._shortcut_rows: list[tuple[_KeyChip, QtWidgets.QLabel]] = []
        rows = QtWidgets.QGridLayout()
        rows.setHorizontalSpacing(14)
        rows.setVerticalSpacing(8)
        for i, (keys, action) in enumerate(SHORTCUTS):
            chip = _KeyChip(keys)
            text = QtWidgets.QLabel(action)
            text.setWordWrap(True)
            apply_font(text, "label")
            rows.addWidget(chip, i, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            rows.addWidget(text, i, 1)
            self._shortcut_rows.append((chip, text))
        rows.setColumnStretch(1, 1)
        lay.addLayout(rows)
        lay.addWidget(self._muted(
            "On the Player page: Space plays or pauses, N and P skip, Shift+Left and "
            "Shift+Right seek 30 s.", "caption"))
        lay.addStretch(1)

        # ── About ──────────────────────────────────────────────────────────
        self._about_card, lay = self._card("About AlphaDEX")
        lay.addWidget(self._muted(
            "AlphaDEX (formerly SoundVault) organizes large music libraries: it previews "
            "every change before it touches a file."))
        facts = QtWidgets.QGridLayout()
        facts.setHorizontalSpacing(14)
        facts.setVerticalSpacing(6)
        for i, (name, value) in enumerate((
            ("Settings", "~/.soundvault_config.json"),
            ("Formats", ".flac  .m4a  .aac  .mp3  .wav  .ogg  .opus"),
        )):
            key = self._muted(name, "caption")
            key.setWordWrap(False)
            val = QtWidgets.QLabel(value)
            val.setWordWrap(True)
            val.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            apply_font(val, "mono", px=12)
            facts.addWidget(key, i, 0, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            facts.addWidget(val, i, 1)
        facts.setColumnStretch(1, 1)
        lay.addLayout(facts)
        self._issues_btn = LinkButton("Report an issue on GitHub", "external")
        self._issues_btn.setToolTip(ISSUES_URL)
        self._issues_btn.clicked.connect(self._open_issues)
        lay.addWidget(self._issues_btn, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        lay.addStretch(1)

        self._grid_host = QtWidgets.QWidget()
        self._grid = QtWidgets.QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(SPACE.gutter)
        self._grid.setVerticalSpacing(SPACE.gutter)
        cl.addWidget(self._grid_host)
        cl.addStretch(1)
        self._set_columns(1)

    def columns(self) -> int:
        return self._columns

    def _set_columns(self, columns: int) -> None:
        if columns == self._columns:
            return
        self._columns = columns
        cards = (self._shortcuts_card, self._about_card)
        for card in cards:
            self._grid.removeWidget(card)
        for i, card in enumerate(cards):
            r, c = divmod(i, columns)
            self._grid.addWidget(card, r, c)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1 if columns == 2 else 0)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self._inner and event.type() == QtCore.QEvent.Type.Resize:
            from gui.themes.tokens import SPACE
            width = self._inner.width() - 2 * SPACE.gutter
            threshold = self.TWO_COLUMNS_AT - (24 if self._columns == 2 else 0)
            self._set_columns(2 if width >= threshold else 1)
        return super().eventFilter(obj, event)

    # ── Actions ───────────────────────────────────────────────────────────

    @Slot()
    def _open_doc(self, rel_path: str) -> None:
        p = doc_path(rel_path)
        if p.exists():
            # as_uri() is correct on every platform ("file://" + a Windows
            # path with backslashes is not).
            webbrowser.open(p.as_uri())
        else:
            QtWidgets.QMessageBox.information(self, "Not Found", f"File not found:\n{p}")

    @Slot()
    def _open_issues(self) -> None:
        webbrowser.open(ISSUES_URL)
