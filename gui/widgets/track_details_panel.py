"""Details panel for the track under the cursor or currently selected.

Glass side-card content: before anything is hovered or picked it shows an
empty state; then the cover (rounded, or a well with a music glyph), the
title, artist and album, a small genre / year / cluster grid, the file path
(middle-elided, full path in the tooltip) and Play / Open folder. The Sound
Map puts it inside a glass card.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import apply_font, current_material, paint_surface, qcolor
from gui.widgets.glass import ElidedLabel
from gui.widgets.glass_parts import EmptyState
from gui.widgets.glyphs import paint_glyph, set_glyph

_ART_SIZE = 72
_ART_RADIUS = 16

#: Fields shown, in order, mapped to the labels used in the panel.
_FIELDS = (
    ("artist", "Artist"),
    ("title", "Title"),
    ("album", "Album"),
    ("genre", "Genre"),
    ("year", "Year"),
    ("cluster", "Cluster"),
)
#: The ones shown as a caption / value grid under the header.
_GRID_FIELDS = ("genre", "year", "cluster")


class _ArtView(QtWidgets.QWidget):
    """A rounded cover, or a well with a music glyph while there is none."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QtGui.QPixmap()
        self._message = ""
        self.setFixedSize(_ART_SIZE, _ART_SIZE)

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:  # noqa: N802 - QLabel-like
        self._pixmap = QtGui.QPixmap(pixmap) if pixmap is not None else QtGui.QPixmap()
        self.update()

    def pixmap(self) -> QtGui.QPixmap:
        return self._pixmap

    def has_art(self) -> bool:
        return not self._pixmap.isNull()

    def setText(self, text: str) -> None:  # noqa: N802 - QLabel-like
        self._message = text
        self.setToolTip(text)

    def text(self) -> str:
        return self._message

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager

        rect = QtCore.QRectF(self.rect())
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            if self._pixmap.isNull():
                paint_surface(p, rect, current_material().well, radius=_ART_RADIUS)
                glyph = QtCore.QRectF(0, 0, 30, 30)
                glyph.moveCenter(rect.center())
                paint_glyph(p, glyph, "music", qcolor(get_manager().current.text_muted))
                return
            path = QtGui.QPainterPath()
            path.addRoundedRect(rect, _ART_RADIUS, _ART_RADIUS)
            p.setClipPath(path)
            pm = self._pixmap
            # Cover-fill the square (crop the longer side).
            src = QtCore.QRectF(pm.rect())
            side = min(src.width(), src.height())
            src = QtCore.QRectF(src.center().x() - side / 2, src.center().y() - side / 2, side, side)
            p.drawPixmap(rect, pm, src)
            p.setClipping(False)
            p.setPen(QtGui.QPen(qcolor(current_material().well.border or current_material().hairline), 1.0))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), _ART_RADIUS, _ART_RADIUS)
        finally:
            p.end()


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
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        self._stack = QtWidgets.QStackedWidget()
        outer.addWidget(self._stack, 1)

        self._empty = EmptyState(
            "No track selected",
            "Hover a dot to preview a track; click it to keep it here.",
        )
        self._stack.addWidget(self._empty)

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._values: Dict[str, QtWidgets.QLabel] = {}

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(12)
        self._art = _ArtView()
        head.addWidget(self._art, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        names = QtWidgets.QVBoxLayout()
        names.setContentsMargins(0, 2, 0, 0)
        names.setSpacing(2)
        for key, role, obj in (("title", "heading", ""), ("artist", "label", "mutedLabel"),
                               ("album", "caption", "mutedLabel")):
            value = ElidedLabel("—")
            if obj:
                value.setObjectName(obj)
            apply_font(value, role)
            names.addWidget(value)
            self._values[key] = value
        names.addStretch(1)
        head.addLayout(names, 1)
        layout.addLayout(head)

        # Genre · Year · Cluster: a caption over each value, side by side.
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(0)
        captions = dict(_FIELDS)
        for col, key in enumerate(_GRID_FIELDS):
            caption = QtWidgets.QLabel(captions[key])
            caption.setObjectName("mutedLabel")
            apply_font(caption, "caption")
            value = ElidedLabel("—")
            apply_font(value, "label", weight=600)
            grid.addWidget(caption, 0, col)
            grid.addWidget(value, 1, col)
            grid.setColumnStretch(col, 2 if key == "genre" else 1)
            self._values[key] = value
        layout.addLayout(grid)

        self._path_label = ElidedLabel("", QtCore.Qt.TextElideMode.ElideMiddle)
        self._path_label.setObjectName("mutedLabel")
        apply_font(self._path_label, "caption")
        layout.addWidget(self._path_label)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(8)
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setObjectName("primaryBtn")
        set_glyph(self.play_btn, "play", role="inverse")
        self.play_btn.setToolTip("Play this track in the Player")
        self.play_btn.setAccessibleName("Play this track")
        self.play_btn.clicked.connect(self._emit_play)
        self.reveal_btn = QtWidgets.QPushButton("Open folder")
        set_glyph(self.reveal_btn, "folder")
        self.reveal_btn.setToolTip("Open the folder containing this track")
        self.reveal_btn.clicked.connect(self._emit_reveal)
        for b in (self.play_btn, self.reveal_btn):
            b.setMinimumWidth(0)
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(1)

        self._body = body
        self._stack.addWidget(body)

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
            self._set_art_placeholder("Click the dot to load artwork")

        if cluster is not None:
            self._metadata["cluster"] = (
                "Unclustered" if cluster < 0 else str(cluster)
            )

        for key, _caption in _FIELDS:
            text = self._display_value(key)
            self._values[key].setText(text)
            self._values[key].setToolTip(text if text != "—" else "")

        self._path_label.setText(path)
        self._path_label.setToolTip(path)
        self.play_btn.setEnabled(True)
        self.reveal_btn.setEnabled(True)
        self._stack.setCurrentWidget(self._body)

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
        dpr = self.devicePixelRatioF() or 1.0
        side = int(_ART_SIZE * dpr)
        scaled = pixmap.scaled(
            side, side,
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        self._art.setPixmap(scaled)

    def _set_art_placeholder(self, message: str) -> None:
        self._art.setPixmap(QtGui.QPixmap())
        self._art.setText(message)

    def clear(self) -> None:
        """Reset the panel to its empty state."""
        self._path = None
        self._metadata = {}
        for value in self._values.values():
            value.setText("—")
            value.setToolTip("")
        self._path_label.setText("")
        self._path_label.setToolTip("")
        self._set_art_placeholder("No track selected")
        self.play_btn.setEnabled(False)
        self.reveal_btn.setEnabled(False)
        self._stack.setCurrentWidget(self._empty)

    def is_empty(self) -> bool:
        return self._path is None

    # ── Actions ───────────────────────────────────────────────────────────

    def _emit_play(self) -> None:
        if self._path:
            self.play_requested.emit(self._path)

    def _emit_reveal(self) -> None:
        if self._path:
            self.reveal_requested.emit(self._path)

    def current_path(self) -> Optional[str]:
        return self._path
