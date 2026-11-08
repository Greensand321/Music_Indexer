"""The Activity Log strip at the bottom of the window.

A glass strip consistent with the shell. Collapsed, it is a pill-shaped bar
(the material's card surface; radius 18 is fully round at its 34 px height)
holding a chevron, the muted 12 px "Activity Log" caption, the status dot
with the active page's status in the same muted caption, and Clear.
Expanded, the same surface grows to hold the log: a rounded dark inset
(``logText``). It expands by itself when a warning or an error arrives.

Expanding and collapsing are instant: the body is shown or hidden once per
toggle (220 px when the window has room, at least 96 px in a short one). Nothing is animated but the chevron's turn, which is
painter-only (``motion.animate_value``; reduced motion makes it instant). So
the window is laid out once per toggle and nothing slides under the pointer.

Text and status colours come from the active theme (the status dot and the
log lines from its ``log_*`` tokens, the captions from ``text_secondary``)
and follow theme switches, including lines already shown.

The Now Playing capsule floats over the page area *above* this strip: the
main window places it inside the page stack's rectangle, which ends where
the strip's row begins and shrinks when the strip expands. So the capsule
never covers the strip's controls, collapsed or expanded
(``tests/ui/test_ui_home_glass.py`` checks it).
"""
from __future__ import annotations

import datetime

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager

_DRAWER_EXPANDED = 220   # the open log's height when the window has room
_BODY_MIN = 96           # … and the least it shrinks to in a short window
_HANDLE_HEIGHT = 34
_RADIUS = 18
_DOT = 8

_LEVELS = ("info", "ok", "warn", "error")

# Workspaces still emit Midnight hex values with status_changed; map them
# (and semantic names) onto the theme's log tokens so the dot follows the
# theme. Anything else is used verbatim.
_STATUS_TOKENS = {
    "#22c55e": "log_ok", "#4ade80": "log_ok", "ok": "log_ok", "success": "log_ok",
    "idle": "log_ok", "done": "log_ok",
    "#f59e0b": "log_warn", "#fbbf24": "log_warn", "warn": "log_warn",
    "warning": "log_warn", "running": "log_warn",
    "#ef4444": "log_err", "#f87171": "log_err", "error": "log_err", "danger": "log_err",
    "info": "log_text",
}


def _level_colour(tokens, level: str) -> str:
    return {
        "ok": tokens.log_ok,
        "warn": tokens.log_warn,
        "error": tokens.log_err,
    }.get(level, tokens.log_text)


class _Handle(QtWidgets.QWidget):
    """The always-visible strip (transparent: the drawer paints the glass)."""

    def __init__(self, drawer: "LogDrawer") -> None:
        super().__init__(drawer)
        self._drawer = drawer
        self.setObjectName("logDrawerHandle")
        self.setFixedHeight(_HANDLE_HEIGHT)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setToolTip("Show or hide the Activity Log (Ctrl+L)")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._drawer.toggle()


class _Chevron(QtWidgets.QWidget):
    """Points up while the log is closed (it opens upward), down while open."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.turn = 0.0          # 0 = closed (up) … 1 = open (down)
        self.setFixedSize(14, 14)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.widgets.glyphs import paint_glyph

        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            c = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
            p.translate(c)
            p.rotate(180.0 * (1.0 - self.turn))
            p.translate(-c)
            paint_glyph(p, QtCore.QRectF(self.rect()), "chevron_down",
                        get_manager().current.text_secondary, weight=2.2)
        finally:
            p.end()


class _StatusDot(QtWidgets.QWidget):
    """The status dot (8 px, the status colour) in a faint ring."""

    def __init__(self, drawer: "LogDrawer") -> None:
        super().__init__(drawer)
        self._drawer = drawer
        self.setFixedSize(_DOT + 8, _DOT + 8)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def colour(self) -> QtGui.QColor:
        return QtGui.QColor(self._drawer.status_colour())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, qcolor

        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            c = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
            p.setBrush(qcolor(current_material().hover))
            p.drawEllipse(c, self.width() / 2.0, self.height() / 2.0)
            p.setBrush(self.colour())
            p.drawEllipse(c, _DOT / 2.0, _DOT / 2.0)
        finally:
            p.end()


class _Body(QtWidgets.QWidget):
    """The open log: 220 px when the window has room, never under 96 px
    (a short window shrinks it instead of pushing it past the edge)."""

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(super().sizeHint().width(), _DRAWER_EXPANDED)


class LogDrawer(QtWidgets.QWidget):
    """A collapsible log strip at the bottom of the window.

    Usage::

        drawer.append("Scan complete", level="ok")
        drawer.append("Warning: missing tag", level="warn")
        drawer.append("Error: file not found", level="error")
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from gui.themes.glass import font
        from gui.widgets.glass import ElidedLabel

        self.setObjectName("logDrawer")
        self._expanded = False
        self._status_key: str = "log_ok"
        self._status_literal: str | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Handle: chevron, caption, status dot and text, Clear ──────────
        handle = _Handle(self)
        self._handle = handle
        handle_layout = QtWidgets.QHBoxLayout(handle)
        handle_layout.setContentsMargins(16, 0, 8, 0)
        handle_layout.setSpacing(8)

        self._chevron = _Chevron(handle)
        handle_layout.addWidget(self._chevron)
        self._toggle_icon = QtWidgets.QLabel("Activity Log")
        self._toggle_icon.setObjectName("logDrawerTitle")
        self._toggle_icon.setFont(font("caption", weight=600))
        handle_layout.addWidget(self._toggle_icon)
        handle_layout.addStretch(1)

        self._status_dot = _StatusDot(self)
        handle_layout.addWidget(self._status_dot)
        self._status_chip = ElidedLabel("Idle")
        self._status_chip.setObjectName("logStatusChip")
        self._status_chip.setFont(font("caption"))
        self._status_chip.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                        QtWidgets.QSizePolicy.Policy.Preferred)
        handle_layout.addWidget(self._status_chip)

        self._clear_btn = QtWidgets.QPushButton("Clear")
        self._clear_btn.setObjectName("logClearButton")
        self._clear_btn.setFlat(True)
        self._clear_btn.setFixedSize(56, 26)
        self._clear_btn.setFont(font("caption", weight=600))
        self._clear_btn.setToolTip("Clear the Activity Log (Ctrl+W); the Activity page keeps its copy")
        self._clear_btn.clicked.connect(self._on_clear)
        handle_layout.addWidget(self._clear_btn)

        root.addWidget(handle)

        # ── Log body: shown at its full height, or not at all ─────────────
        self._log_body = _Body(self)
        self._log_body.setObjectName("logDrawerBody")
        self._log_body.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                     QtWidgets.QSizePolicy.Policy.Maximum)
        self._log_body.setMinimumHeight(_BODY_MIN)
        self._log_body.setMaximumHeight(_DRAWER_EXPANDED)
        body_layout = QtWidgets.QVBoxLayout(self._log_body)
        body_layout.setContentsMargins(8, 0, 8, 8)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setObjectName("logText")
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(500)
        body_layout.addWidget(self._text)
        self._log_body.hide()

        root.addWidget(self._log_body)

        get_manager().theme_changed.connect(self._on_theme_changed)
        self._apply_theme(get_manager().current)

    # ── Public API ────────────────────────────────────────────────────────

    def append(self, message: str, level: str = "info") -> None:
        """Add a timestamped message. level: 'ok' | 'warn' | 'error' | 'info'"""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        level = level if level in _LEVELS else "info"
        doc = self._text.document()
        cursor = QtGui.QTextCursor(doc)
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        first = cursor.block()
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(_level_colour(get_manager().current, level)))
        cursor.setCharFormat(fmt)
        cursor.insertText(f"[{ts}] {message}\n")
        # Remember each line's level so a theme switch can recolour it.
        code = _LEVELS.index(level)
        block, last = first, cursor.block()
        while block.isValid() and block != last:
            block.setUserState(code)
            block = block.next()
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

        # Auto-expand on errors / warnings
        if level in ("error", "warn") and not self._expanded:
            self.toggle()

    def set_status(self, text: str, colour: str = "ok") -> None:
        """Update the status. ``colour`` is a semantic name ("ok", "warn",
        "error", "running", "idle") or a hex; the hex values the workspaces
        send are mapped to the theme's log tokens. The dot shows the colour;
        the text stays muted."""
        key = _STATUS_TOKENS.get(str(colour).strip().lower())
        self._status_key = key or ""
        self._status_literal = None if key else colour
        self._status_chip.setText(text)
        self._status_chip.setToolTip(text)
        self._status_dot.update()

    def status_text(self) -> str:
        return self._status_chip.text()

    def status_colour(self) -> str:
        """The dot's current colour (resolved against the active theme)."""
        if self._status_literal:
            return QtGui.QColor(self._status_literal).name()
        return getattr(get_manager().current, self._status_key or "log_ok")

    def is_expanded(self) -> bool:
        return self._expanded

    def toggle(self) -> None:
        """Open or close the log at once (one relayout); the chevron turns."""
        from gui.themes import motion

        self._expanded = not self._expanded
        self._log_body.setVisible(self._expanded)
        self.update()
        motion.animate_value(
            self, "chevron", self._chevron.turn, 1.0 if self._expanded else 0.0,
            self._on_chevron, duration="fast", easing="standard",
        )

    # ── Painting: the glass strip ─────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface

        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().card, radius=_RADIUS)
        finally:
            p.end()

    # ── Private ───────────────────────────────────────────────────────────

    def _on_chevron(self, value: float) -> None:
        self._chevron.turn = float(value)
        self._chevron.update()

    def _on_clear(self) -> None:
        self._text.clear()

    def _on_theme_changed(self, tokens: object) -> None:
        self._apply_theme(tokens)

    def _apply_theme(self, t) -> None:
        def _fg(widget: QtWidgets.QWidget, colour: str) -> None:
            pal = widget.palette()
            pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(colour))
            widget.setPalette(pal)

        _fg(self._toggle_icon, t.text_secondary)
        _fg(self._status_chip, t.text_secondary)
        self._status_dot.update()
        self._chevron.update()
        self._recolour_lines(t)
        self._handle.update()

    def _recolour_lines(self, t) -> None:
        doc = self._text.document()
        block = doc.begin()
        cursor = QtGui.QTextCursor(doc)
        fmt = QtGui.QTextCharFormat()
        cursor.beginEditBlock()
        while block.isValid():
            state = block.userState()
            if 0 <= state < len(_LEVELS) and block.length() > 1:
                fmt.setForeground(QtGui.QColor(_level_colour(t, _LEVELS[state])))
                cursor.setPosition(block.position())
                cursor.setPosition(block.position() + block.length() - 1,
                                   QtGui.QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(fmt)
            block = block.next()
        cursor.endEditBlock()
