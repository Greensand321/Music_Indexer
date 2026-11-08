"""Hover animation mixin and animated button widgets for AlphaDEX themes."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal


# ── HoverMixin ────────────────────────────────────────────────────────────────

class HoverMixin:
    """Mixin for QWidget subclasses — drives a _hover_p float (0.0 → 1.0).

    AlphaDEXStyle reads ``widget._hover_p`` in _hover_t() during paint.
    Install by calling ``_init_hover()`` from the subclass ``__init__``.

    The animation comes from ``gui.themes.motion`` (``fast`` token,
    ``standard`` easing): one parented animation per widget, restarted from
    the current value, and instant under reduced motion.
    """

    _MOTION_KEY = "hover"

    def _init_hover(self) -> None:
        self._hover_p: float = 0.0
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    @property
    def _hover_anim(self):  # kept for callers/tests that inspect it
        from gui.themes import motion
        return motion.animation_for(self, self._MOTION_KEY)

    def _on_hover_value(self, value: float) -> None:
        self._hover_p = float(value)
        self.update()

    def _start_hover(self, entering: bool) -> None:
        from gui.themes import motion
        end = 1.0 if entering else 0.0
        if self._hover_p == end and self._hover_anim is None:
            return
        motion.animate_value(
            self, self._MOTION_KEY, float(self._hover_p), end, self._on_hover_value,
            duration="fast", easing="standard",
        )

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
        is_exit: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._nav_key = nav_key
        self._label = label
        self._icon_text = icon_text
        self._active = False
        self._badge = 0
        self._is_exit = is_exit

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
        # Transparent background so the NavContainer's pill shows through
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
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
        # Activation never touches the hover progress: the pill shows the
        # active state. On deactivation, ease the hover back to what the
        # pointer says, so an item activated from code (Ctrl+1..9,
        # navigate_requested, play_tracks) never keeps a hover tint (DS-05).
        if not value:
            self._start_hover(self.underMouse())
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
        # The accessible text. The item paints only the label (no icon) and
        # the badge as a "N running" pill.
        text = self._label
        if self._badge:
            text += f" ({self._badge} running)"
        self.setText(text)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Glass nav item: a soft wash on hover, white 600 when active (the
        sliding pill behind it is the sidebar's NavContainer), and the
        "N running" pill for a badge."""
        try:
            from gui.themes.manager import get_manager
            from gui.themes.effects import lerp_color
            from gui.themes.glass import qcolor
            t = get_manager().current
        except Exception:
            super().paintEvent(event)
            return

        m = t.material
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)

        r = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, 14.0, 14.0)

        # A running job tints the row faintly (the Activity entry).
        if self._badge and not self._active:
            p.fillPath(path, QtGui.QBrush(qcolor(m.hover, 0.75)))
        # Hover wash: composites over the panel (the active item has the pill).
        if self._hover_p > 0 and not self._active:
            if self._is_exit:
                col = QtGui.QColor(t.danger)
                col.setAlpha(int(self._hover_p * 45))
            else:
                col = qcolor(m.hover, self._hover_p * 0.75)
            p.fillPath(path, QtGui.QBrush(col))

        idle = QtGui.QColor(t.sidebar_text if m.is_glass else t.text_secondary)
        if self._is_exit:
            text_col = lerp_color(idle.name(), t.danger, self._hover_p) if self._hover_p > 0 else idle
        elif self._active:
            text_col = QtGui.QColor(t.text_primary)
        elif self._hover_p > 0:
            text_col = lerp_color(idle.name(), t.text_primary, self._hover_p)
        else:
            text_col = idle

        font = QtGui.QFont(self.font())
        if self._active:
            font.setWeight(QtGui.QFont.Weight.DemiBold)
        p.setFont(font)
        p.setPen(text_col)
        badge_w = int(self._badge_rect().width()) + 8 if self._badge else 0
        p.drawText(
            self.rect().adjusted(14, 0, -(14 + badge_w), 0),
            int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft),
            self._label,
        )

        if self._badge:
            self._draw_badge(p)

        p.end()

    def _badge_text(self) -> str:
        n = min(self._badge, 99)
        return f"{n}{'+' if self._badge > 99 else ''} running"

    @staticmethod
    def _badge_font() -> QtGui.QFont:
        from gui.themes.glass import font
        f = font("micro", weight=600)
        f.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, 0.0)
        return f

    def _badge_rect(self) -> QtCore.QRectF:
        fm = QtGui.QFontMetrics(self._badge_font())
        w = fm.horizontalAdvance(self._badge_text()) + 16
        h = fm.height() + 4
        return QtCore.QRectF(self.width() - 12 - w, (self.height() - h) / 2, w, h)

    def _draw_badge(self, p: QtGui.QPainter) -> None:
        """A "1 running" pill: the accent at .22 with the light accent text."""
        from gui.themes.manager import get_manager
        from gui.themes.glass import qcolor
        from gui.themes.tokens import mix, rgba

        t = get_manager().current
        rect = self._badge_rect()
        if t.material.is_glass:
            fill, text = qcolor(rgba(t.accent, 0.22)), QtGui.QColor(t.accent_hover)
        else:
            fill, text = QtGui.QColor(mix(t.sidebar_bg, t.accent, 0.22)), QtGui.QColor(t.text_primary)
        p.save()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        p.fillPath(path, fill)
        p.setFont(self._badge_font())
        p.setPen(text)
        p.drawText(rect, int(QtCore.Qt.AlignmentFlag.AlignCenter), self._badge_text())
        p.restore()


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
