"""Hover animation mixin and animated button widgets for AlphaDEX themes."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal


# ── HoverMixin ────────────────────────────────────────────────────────────────

class HoverMixin:
    """Mixin for QWidget subclasses — drives a _hover_p float (0.0 → 1.0).

    AlphaDEXStyle reads ``widget._hover_p`` in _hover_t() during paint.
    Install by calling ``_init_hover()`` from the subclass ``__init__``.
    """

    _DURATION = 150  # ms

    def _init_hover(self) -> None:
        self._hover_p: float = 0.0
        self._hover_anim = QtCore.QVariantAnimation(self)
        self._hover_anim.setDuration(self._DURATION)
        self._hover_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._hover_anim.valueChanged.connect(self._on_hover_value)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    def _on_hover_value(self, value: float) -> None:
        self._hover_p = value
        self.update()

    def _start_hover(self, entering: bool) -> None:
        anim = self._hover_anim
        anim.stop()
        anim.setStartValue(self._hover_p)
        anim.setEndValue(1.0 if entering else 0.0)
        anim.start()

    # QWidget event overrides — call super() so subclass chain works
    def enterEvent(self, event) -> None:
        self._start_hover(True)
        super().enterEvent(event)  # type: ignore[misc]

    def leaveEvent(self, event) -> None:
        self._start_hover(False)
        super().leaveEvent(event)  # type: ignore[misc]


# ── AnimatedButton ────────────────────────────────────────────────────────────

class AnimatedButton(HoverMixin, QtWidgets.QPushButton):
    """QPushButton with smooth hover animation driving AlphaDEXStyle."""

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._init_hover()


# ── AnimatedNavButton ─────────────────────────────────────────────────────────

class AnimatedNavButton(HoverMixin, QtWidgets.QPushButton):
    """Sidebar navigation button with hover animation + active state.

    Properties
    ----------
    active : bool
        Whether this item is the currently selected nav item.
    badge : int
        Badge count shown at trailing edge (0 = hidden).
    """

    clicked_key = Signal(str)  # emitted with self._nav_key

    def __init__(
        self,
        label: str,
        nav_key: str,
        icon_text: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._nav_key = nav_key
        self._label = label
        self._icon_text = icon_text
        self._active = False
        self._badge = 0

        self._init_hover()
        self.setObjectName("navButton")
        self.setCheckable(False)
        self.setFlat(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(40)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.clicked.connect(lambda: self.clicked_key.emit(self._nav_key))
        self._update_text()

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        if self._active == value:
            return
        self._active = value
        # Ensure hover_p starts at correct end when activated externally
        if value:
            self._hover_p = 1.0
            self._hover_anim.stop()
        self.update()

    @property
    def badge(self) -> int:
        return self._badge

    @badge.setter
    def badge(self, count: int) -> None:
        self._badge = max(0, count)
        self._update_text()
        self.update()

    def nav_key(self) -> str:
        return self._nav_key

    # ── Internal ──────────────────────────────────────────────────────────

    def _update_text(self) -> None:
        parts = []
        if self._icon_text:
            parts.append(self._icon_text)
        parts.append(self._label)
        if self._badge:
            parts.append(f"  {self._badge}")
        self.setText("  ".join(parts) if self._icon_text else self._label)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        # Delegate entirely to AlphaDEXStyle via QProxyStyle
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        p = QtWidgets.QStylePainter(self)
        p.drawControl(QtWidgets.QStyle.ControlElement.CE_PushButton, opt)

        # Draw badge overlay if needed
        if self._badge:
            self._draw_badge(p)

    def _draw_badge(self, p: QtWidgets.QStylePainter) -> None:
        text = str(min(self._badge, 99)) + ("+" if self._badge > 99 else "")
        fm = QtGui.QFontMetrics(self.font())
        tw = fm.horizontalAdvance(text)
        bw = max(tw + 10, 20)
        bh = 16
        margin = 8
        x = self.width() - bw - margin
        y = (self.height() - bh) // 2
        rect = QtCore.QRect(x, y, bw, bh)

        # Badge pill — accent colour
        from gui.themes.manager import get_manager
        t = get_manager().current
        p.setBrush(QtGui.QColor(t.accent))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect), bh / 2, bh / 2)
        p.drawPath(path)

        # Badge text
        p.setPen(QtGui.QColor(t.text_inverse))
        p.setFont(self.font())
        p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)


# ── AnimatedTabButton ─────────────────────────────────────────────────────────

class AnimatedTabButton(HoverMixin, QtWidgets.QPushButton):
    """Tab-bar style button with hover animation (used in workspace sub-tabs)."""

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._init_hover()
        self.setCheckable(True)
        self.setFlat(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setObjectName("tabButton")
