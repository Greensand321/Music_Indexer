"""Details panel for the track under the cursor or currently selected."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from gui.compat import QtCore, QtGui, QtWidgets

_ART_SIZE = 132

#: Fields shown, in order, mapped to the labels used in the panel.
_FIELDS = (
    ("artist", "Artist"),
    ("title", "Title"),
    ("album", "Album"),
    ("genre", "Genre"),
    ("year", "Year"),
    ("cluster", "Cluster"),
)


class TrackDetailsPanel(QtWidgets.QWidget):
    """Shows one track's artwork and tags.

    Artwork is only read from disk when :meth:`set_track` is called with
    ``load_art=True``. Hovering a scatter plot fires continuously, and decoding
    an embedded cover on every hover would stutter the UI, so the graph hovers
    with art off and turns it on when a point is actually selected.
    """

    play_requested = QtCore.Signal(str)     # track path
    reveal_requested = QtCore.Signal(str)   # track path

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._path: Optional[str] = None
        self._metadata: Dict[str, object] = {}
        self._build_ui()
        self.clear()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Track Details")
        title.setStyleSheet("font-weight:600;")
        layout.addWidget(title)

        self._art = QtWidgets.QLabel()
        self._art.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._art.setFixedHeight(_ART_SIZE)
        self._art.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout.addWidget(self._art)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._values: Dict[str, QtWidgets.QLabel] = {}
        for key, caption in _FIELDS:
            value = QtWidgets.QLabel("—")
            value.setWordWrap(True)
            form.addRow(f"{caption}:", value)
            self._values[key] = value
        layout.addLayout(form)

        self._path_label = QtWidgets.QLabel("")
        self._path_label.setObjectName("statusHint")
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)

        buttons = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶  Play")
        self.play_btn.setToolTip("Play this track in the Player workspace")
        self.play_btn.clicked.connect(self._emit_play)
        self.reveal_btn = QtWidgets.QPushButton("Open folder")
        self.reveal_btn.setToolTip("Open the folder containing this track")
        self.reveal_btn.clicked.connect(self._emit_reveal)
        buttons.addWidget(self.play_btn)
        buttons.addWidget(self.reveal_btn)
        layout.addLayout(buttons)

        layout.addStretch(1)

    # ── Population ────────────────────────────────────────────────────────

    def set_track(
        self,
        path: str | None,
        metadata: Dict[str, object] | None = None,
        cluster: int | None = None,
        *,
        load_art: bool = False,
    ) -> None:
        """Display *path*, optionally decoding its embedded cover art."""
        if not path:
            self.clear()
            return

        self._path = path
        self._metadata = dict(metadata or {})

        if load_art:
            # One read serves both the artwork and the tag fields — the reader
            # returns them together, so pulling tags here costs nothing extra
            # and means a selected track shows real artist/album/year rather
            # than just its file name.
            tags, covers = self._read_from_file(path)
            for key, value in tags.items():
                if value not in (None, "", []):
                    self._metadata.setdefault(key, value)
            self._apply_art(covers)
        else:
            self._set_art_placeholder("Select a point to load artwork")

        if cluster is not None:
            self._metadata["cluster"] = (
                "Unclustered" if cluster < 0 else str(cluster)
            )

        for key, _caption in _FIELDS:
            self._values[key].setText(self._display_value(key))

        self._path_label.setText(path)
        self.play_btn.setEnabled(True)
        self.reveal_btn.setEnabled(True)

    def _display_value(self, key: str) -> str:
        value = self._metadata.get(key)
        if value in (None, "", []):
            return "—"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value if v) or "—"
        return str(value)

    @staticmethod
    def _read_from_file(path: str) -> tuple[Dict[str, object], List[bytes]]:
        """Return ``(tags, cover_payloads)``, degrading quietly on any failure."""
        try:
            from utils.audio_metadata_reader import read_metadata

            tags, covers, _error, _hint = read_metadata(path, include_cover=True)
            return dict(tags or {}), list(covers or [])
        except Exception:  # noqa: BLE001 - detail is nice-to-have, never fatal
            return {}, []

    def _apply_art(self, covers: List[bytes]) -> None:
        """Show the first decodable cover payload, or a placeholder."""
        pixmap = QtGui.QPixmap()
        for payload in covers or []:
            if payload and pixmap.loadFromData(payload):
                break

        if pixmap.isNull():
            self._set_art_placeholder("No artwork")
            return

        self._art.setText("")
        self._art.setPixmap(
            pixmap.scaled(
                _ART_SIZE,
                _ART_SIZE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _set_art_placeholder(self, message: str) -> None:
        self._art.setPixmap(QtGui.QPixmap())
        self._art.setText(message)

    def clear(self) -> None:
        """Reset the panel to its empty state."""
        self._path = None
        self._metadata = {}
        for value in self._values.values():
            value.setText("—")
        self._path_label.setText("")
        self._set_art_placeholder("No track selected")
        self.play_btn.setEnabled(False)
        self.reveal_btn.setEnabled(False)

    # ── Actions ───────────────────────────────────────────────────────────

    def _emit_play(self) -> None:
        if self._path:
            self.play_requested.emit(self._path)

    def _emit_reveal(self) -> None:
        if self._path:
            self.reveal_requested.emit(self._path)

    def current_path(self) -> Optional[str]:
        return self._path
