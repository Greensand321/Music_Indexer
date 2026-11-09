"""A folder or file path shown on a glass well, with a glyph button to choose it.

``PathField`` replaces a raw ``QLineEdit`` + "Browse…" pair wherever a page
shows a chosen path (Mirror's source and destination, the default library in
Settings, the file a Mirror → Verify tester opened):

* the path is the shared path label (``gui.widgets.glass.path_label``: a
  *middle* elide, so a long path never widens or wraps its card, the full
  path in the tooltip, the placeholder painted muted while it is empty);
* a round glyph button (``folder`` by default) asks to choose another one —
  the field only emits ``browse_requested``: the owner opens the dialog, so
  tests keep answering ``QFileDialog`` through the ``dialogs`` fixture;
* clicking anywhere on the well does the same;
* ``clearable=True`` adds a clear button while a path is set.

The text API mirrors ``QLineEdit`` (``text`` / ``setText`` / ``textChanged``
/ ``placeholderText``), so code and tests that drove the old line edits keep
working. Colours and the surface come from the active material at paint
time; a theme switch only needs a repaint.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.widgets.glass import path_label
from gui.widgets.glyphs import set_glyph


class PathField(QtWidgets.QWidget):
    """A path on a well, middle-elided, with a glyph button to choose it."""

    textChanged = Signal(str)        # noqa: N815 - mirrors QLineEdit
    browse_requested = Signal()

    HEIGHT = 40
    _BUTTON = 32

    def __init__(self, placeholder: str = "", *, glyph: str = "folder",
                 button_tip: str = "Choose a folder", clearable: bool = False,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pathField")
        self._text = ""
        self._placeholder = placeholder
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(16, 4, 4, 4)
        lay.setSpacing(2)
        self.label = path_label("", placeholder)
        self.label.setObjectName("pathFieldText")
        self.label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay.addWidget(self.label, 1)

        self.clear_button: QtWidgets.QPushButton | None = None
        if clearable:
            self.clear_button = self._glyph_button("close", "Clear")
            self.clear_button.clicked.connect(lambda: self.setText(""))
            lay.addWidget(self.clear_button)
        self.button = self._glyph_button(glyph, button_tip)
        self.button.clicked.connect(self.browse_requested.emit)
        lay.addWidget(self.button)
        self._sync()

    def _glyph_button(self, glyph: str, tip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton()
        btn.setFlat(True)
        btn.setAutoDefault(False)   # in a dialog, Enter never "clicks" a path button
        btn.setFixedSize(self._BUTTON, self._BUTTON)
        btn.setToolTip(tip)
        btn.setAccessibleName(tip)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        set_glyph(btn, glyph, role="muted" if glyph == "close" else "text")
        return btn

    # ── QLineEdit-like API ────────────────────────────────────────────────

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:  # noqa: N802
        text = str(text or "")
        if text == self._text:
            return
        self._text = text
        self._sync()
        self.textChanged.emit(text)

    def clear(self) -> None:
        self.setText("")

    def placeholderText(self) -> str:  # noqa: N802
        return self._placeholder

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802
        self._placeholder = text
        self._sync()

    def setAccessibleName(self, name: str) -> None:  # noqa: N802
        super().setAccessibleName(name)
        self.label.setAccessibleName(name)

    # ── Internals ─────────────────────────────────────────────────────────

    def _sync(self) -> None:
        self.label.set_placeholder(self._placeholder)
        self.label.setText(self._text)
        self.setToolTip(self._text or self._placeholder)
        self.label.setToolTip(self.toolTip())
        if self.clear_button is not None:
            self.clear_button.setVisible(bool(self._text))
        self.label.update()

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        buttons = self._BUTTON * (2 if self.clear_button is not None else 1)
        return QtCore.QSize(16 + 64 + buttons + 8, self.HEIGHT)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(320, self.HEIGHT)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if (event.button() == QtCore.Qt.MouseButton.LeftButton and self.isEnabled()
                and self.rect().contains(event.position().toPoint())):
            self.browse_requested.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        if event.type() == QtCore.QEvent.Type.EnabledChange:
            self.update()
        super().changeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface

        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().well,
                          radius=self.height() / 2, opacity=1.0 if self.isEnabled() else 0.55)
        finally:
            p.end()
