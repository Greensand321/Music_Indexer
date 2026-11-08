"""Small Liquid Glass building blocks shared by every page.

Use these instead of building a private copy in a workspace:

``StatTile``
    A number and a label on a "well" surface (radius 16, faint fill): the
    mockup's Library health tiles. ``set_value`` / ``set_label`` update it in
    place; ``set_tone`` tints the value ("" · "success" · "warning" ·
    "danger" · "accent").

``StatRow``
    A row of ``StatTile``s that share the width evenly, built from
    ``(key, label)`` pairs; ``tile(key)`` returns one to update.

``EmptyState``
    What a list or results area shows before there is anything in it: a
    heading, one muted sentence and optionally one action button (it is only
    shown when a slot is connected, so it never offers something that does
    nothing).

All of them paint from the active material and the type ramp at paint time,
so a theme, accent or transparency change only needs a repaint.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.glass import current_material, font, paint_surface, qcolor

_TONES = ("", "success", "warning", "danger", "accent")


def _tone_colour(tone: str) -> QtGui.QColor:
    from gui.themes.manager import get_manager

    t = get_manager().current
    value = {
        "success": t.success,
        "warning": t.warning,
        "danger": t.danger,
        "accent": t.accent,
    }.get(tone, t.text_primary)
    return qcolor(value)


class StatTile(QtWidgets.QWidget):
    """A value over a label on a well surface (24/700 tabular, 12 muted)."""

    def __init__(self, value: str = "–", label: str = "", *, tone: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = value
        self._label = label
        self._tone = tone if tone in _TONES else ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setAccessibleName(f"{label}: {value}")

    # ── Public API ────────────────────────────────────────────────────────
    def value(self) -> str:
        return self._value

    def label(self) -> str:
        return self._label

    def tone(self) -> str:
        return self._tone

    def set_value(self, value: object, tone: str | None = None) -> None:
        self._value = str(value)
        if tone is not None:
            self._tone = tone if tone in _TONES else ""
        self.setAccessibleName(f"{self._label}: {self._value}")
        self.update()

    def set_label(self, label: str) -> None:
        self._label = label
        self.setAccessibleName(f"{self._label}: {self._value}")
        self.updateGeometry()
        self.update()

    def set_tone(self, tone: str) -> None:
        self._tone = tone if tone in _TONES else ""
        self.update()

    # ── Qt ────────────────────────────────────────────────────────────────
    _PAD_X = 14
    _PAD_Y = 12
    _GAP = 2

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        vf = QtGui.QFontMetrics(font("stat"))
        lf = QtGui.QFontMetrics(font("caption"))
        w = max(vf.horizontalAdvance(self._value), lf.horizontalAdvance(self._label))
        h = vf.height() + self._GAP + lf.height()
        return QtCore.QSize(w + 2 * self._PAD_X, h + 2 * self._PAD_Y)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        hint = self.sizeHint()
        return QtCore.QSize(min(hint.width(), 96), hint.height())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager

        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().well)
            inner = self.rect().adjusted(self._PAD_X, self._PAD_Y, -self._PAD_X, -self._PAD_Y)
            vfont, lfont = font("stat"), font("caption")
            vh = QtGui.QFontMetrics(vfont).height()
            p.setFont(vfont)
            p.setPen(_tone_colour(self._tone))
            value_rect = QtCore.QRect(inner.left(), inner.top(), inner.width(), vh)
            p.drawText(value_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                       QtGui.QFontMetrics(vfont).elidedText(
                           self._value, QtCore.Qt.TextElideMode.ElideRight, inner.width()))
            p.setFont(lfont)
            p.setPen(qcolor(get_manager().current.text_secondary))
            label_rect = QtCore.QRect(inner.left(), inner.top() + vh + self._GAP,
                                      inner.width(), inner.bottom() - inner.top() - vh - self._GAP + 1)
            p.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
                       QtGui.QFontMetrics(lfont).elidedText(
                           self._label, QtCore.Qt.TextElideMode.ElideRight, inner.width()))
        finally:
            p.end()


class StatRow(QtWidgets.QWidget):
    """``StatTile``s side by side, sharing the width evenly."""

    def __init__(self, tiles: list[tuple[str, str]], *, spacing: int = 10,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(spacing)
        self._tiles: dict[str, StatTile] = {}
        for key, label in tiles:
            tile = StatTile("–", label, parent=self)
            tile.setObjectName(f"stat_{key}")
            self._tiles[key] = tile
            lay.addWidget(tile, 1)

    def tile(self, key: str) -> StatTile:
        return self._tiles[key]

    def keys(self) -> list[str]:
        return list(self._tiles)


class EmptyState(QtWidgets.QWidget):
    """A heading, one muted sentence, and an optional action button.

    The button stays hidden until ``action_requested`` has a receiver (or
    ``set_action`` is given a callable), so an empty state never offers an
    action that does nothing."""

    action_requested = Signal()

    def __init__(self, title: str, message: str = "", action_text: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 20, 8, 20)
        lay.setSpacing(6)
        lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self._title = QtWidgets.QLabel(title)
        self._title.setObjectName("cardTitle")
        self._title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._title.setFont(font("heading"))
        self._message = QtWidgets.QLabel(message)
        self._message.setObjectName("mutedLabel")
        self._message.setWordWrap(True)
        self._message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._message.setVisible(bool(message))
        self._button = QtWidgets.QPushButton(action_text)
        self._button.setVisible(False)
        self._button.clicked.connect(self.action_requested.emit)
        self._has_action = False

        lay.addWidget(self._title)
        lay.addWidget(self._message)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._button)
        row.addStretch(1)
        lay.addSpacing(6)
        lay.addLayout(row)

    def set_text(self, title: str, message: str = "") -> None:
        self._title.setText(title)
        self._message.setText(message)
        self._message.setVisible(bool(message))

    def set_action(self, text: str, callback=None) -> None:
        """Show the button with ``text``; ``callback`` (optional) is connected
        to ``action_requested``. Pass ``text=""`` to hide it again."""
        if callback is not None:
            self.action_requested.connect(callback)
            self._has_action = True
        self._button.setText(text)
        self._button.setVisible(bool(text) and self._has_action)

    def connectNotify(self, signal: QtCore.QMetaMethod) -> None:  # noqa: N802
        if signal.name() == b"action_requested":
            self._has_action = True
            self._button.setVisible(bool(self._button.text()))
        super().connectNotify(signal)

    def title(self) -> str:
        return self._title.text()

    def message(self) -> str:
        return self._message.text()

    def action_button(self) -> QtWidgets.QPushButton:
        return self._button
