"""Mirror → "Verify the mirror": check that a converted file kept its tags and art.

The Opus and M4A testers (``MediaTesterPanel``) used to be dialogs launched
from the Utilities page's Diagnostics tile. Here they sit side by side, next
to the mirror they are meant to check. Read-only: they open a file and show
what AlphaDEX's shared metadata reader sees in it.
"""
from __future__ import annotations

from gui.compat import QtWidgets
from gui.dialogs.media_tester_dialog import MediaTesterPanel
from gui.workspaces.base import WorkspaceBase


class MirrorVerifyWorkspace(WorkspaceBase):
    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self.testers: dict[str, MediaTesterPanel] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_section_title("Verify the mirror"))
        cl.addWidget(self._make_subtitle(
            "Open a file from your Opus mirror (or an M4A) and check that its tags and "
            "cover art came through the conversion. This reads the file the same way "
            "the Indexer, Tag Fixer and Player do; nothing is changed."
        ))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(16)
        for codec, title in (("opus", "Opus file"), ("m4a", "M4A file")):
            card = self._make_card()
            lay = QtWidgets.QVBoxLayout(card)
            lay.setContentsMargins(16, 16, 16, 16)
            lay.setSpacing(10)
            lay.addWidget(self._make_card_title(title))
            tester = MediaTesterPanel(card, codec=codec)
            lay.addWidget(tester)
            self.testers[codec] = tester
            row.addWidget(card, 1)
        cl.addLayout(row)
        cl.addStretch(1)
