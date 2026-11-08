"""Mirror → "Verify the mirror": check that a converted file kept its tags and art.

The Opus and M4A testers (``MediaTesterPanel``) used to be dialogs launched
from the Utilities page's Diagnostics tile. Here they sit side by side, next
to the mirror they are meant to check (stacked when the window is narrow).
Read-only: they open a file and show what AlphaDEX's shared metadata reader
sees in it — a stat row, one glass row per problem, the cover and the tags.
"""
from __future__ import annotations

from gui.compat import QtWidgets
from gui.dialogs.media_tester_dialog import MediaTesterPanel
from gui.workspaces.base import WorkspaceBase

# Below this panel width the two tester cards stack instead of sitting side
# by side (each needs room for three stat tiles and the cover).
_SIDE_BY_SIDE_MIN = 960


class MirrorVerifyWorkspace(WorkspaceBase):
    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self.testers: dict[str, MediaTesterPanel] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        from gui.themes.tokens import SPACE

        cl = self.content_layout
        cl.addWidget(self._make_subtitle(
            "Open a file from your mirror and check that its tags and cover art came "
            "through the conversion. Nothing is changed."
        ))
        self._row = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.LeftToRight)
        self._row.setSpacing(SPACE.gutter)
        for codec, title, hint in (
            ("opus", "Opus file", "What the Indexer, Tag Fixer and Player read from it."),
            ("m4a", "M4A file", "Same check for AAC files in an .m4a container."),
        ):
            card = self._make_card()
            lay = QtWidgets.QVBoxLayout(card)
            lay.setContentsMargins(16, 16, 16, 16)
            lay.setSpacing(SPACE.sm)
            head = QtWidgets.QVBoxLayout()
            head.setSpacing(SPACE.xxs)
            head.addWidget(self._make_card_title(title))
            sub = QtWidgets.QLabel(hint)
            sub.setObjectName("mutedLabel")
            sub.setWordWrap(True)
            head.addWidget(sub)
            lay.addLayout(head)
            tester = MediaTesterPanel(card, codec=codec)
            lay.addWidget(tester)
            lay.addStretch(1)
            self.testers[codec] = tester
            self._row.addWidget(card, 1)
        cl.addLayout(self._row)
        cl.addStretch(1)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        direction = (QtWidgets.QBoxLayout.Direction.LeftToRight
                     if self.width() >= _SIDE_BY_SIDE_MIN
                     else QtWidgets.QBoxLayout.Direction.TopToBottom)
        if self._row.direction() != direction:
            self._row.setDirection(direction)
