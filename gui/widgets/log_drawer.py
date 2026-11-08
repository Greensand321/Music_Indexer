"""Slide-up activity log drawer anchored to the bottom of the window.

Under Liquid Glass the drawer is a glass strip below the pages (the
material's card surface, radius 18): the handle carries the title, the
status chip and Clear; the expanded log is a rounded dark inset. Text and
status colours come from the active theme's ``log_*`` tokens and follow
theme switches (they used to be fixed Midnight constants, DS-12 / SHELL-18).

Motion (DS-18): the drawer's layout footprint changes once per toggle (the
body gets its full height before opening, and gives it back after closing);
in between, only a "sheet" child slides inside the body, positioned by hand
in a widget that has no layout. So the window is laid out once per toggle,
not on every frame, and the slide always covers the full distance.
"""
from __future__ import annotations

import datetime

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager

_DRAWER_EXPANDED = 220
_DRAWER_COLLAPSED = 0
_HANDLE_HEIGHT = 34
_RADIUS = 18

_LEVELS = ("info", "ok", "warn", "error")

# Workspaces still emit Midnight hex values with status_changed; map them
# (and semantic names) onto the theme's log tokens so the chip follows the
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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._drawer.toggle()


class _Body(QtWidgets.QWidget):
    """Viewport for the sliding sheet. Has no layout on purpose."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.sheet = QtWidgets.QWidget(self)
        self.sheet.setObjectName("logDrawerSheet")
        # The rounded log text paints its own log_bg; the sheet stays clear
        # so the drawer's glass shows round it.
        self.sheet.setAutoFillBackground(False)
        self.reveal = 0.0       # 0 = sheet below the body, 1 = fully shown

    def place_sheet(self) -> None:
        h = self.height()
        self.sheet.setGeometry(0, int(round((1.0 - self.reveal) * h)), self.width(), h)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.place_sheet()


class LogDrawer(QtWidgets.QWidget):
    """A collapsible log panel that slides up from the bottom of the window.

    Usage::

        drawer.append("Scan complete", level="ok")
        drawer.append("Warning: missing tag", level="warn")
        drawer.append("Error: file not found", level="error")
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("logDrawer")
        self._expanded = False
        self._status_key: str = "log_ok"
        self._status_literal: str | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toggle handle ─────────────────────────────────────────────────
        handle = _Handle(self)
        self._handle = handle
        handle_layout = QtWidgets.QHBoxLayout(handle)
        handle_layout.setContentsMargins(16, 0, 10, 0)
        handle_layout.setSpacing(10)

        from gui.themes.glass import font

        self._toggle_icon = QtWidgets.QLabel("▲  Activity Log")
        self._toggle_icon.setObjectName("logDrawerTitle")
        self._toggle_icon.setFont(font("caption", weight=600))
        handle_layout.addWidget(self._toggle_icon)
        handle_layout.addStretch(1)

        self._status_chip = QtWidgets.QLabel("● Idle")
        self._status_chip.setObjectName("logStatusChip")
        self._status_chip.setFont(font("caption"))
        handle_layout.addWidget(self._status_chip)

        self._clear_btn = QtWidgets.QPushButton("Clear")
        self._clear_btn.setObjectName("logClearButton")
        self._clear_btn.setFlat(True)
        self._clear_btn.setFixedSize(56, 24)
        self._clear_btn.setFont(font("caption", weight=600))
        self._clear_btn.clicked.connect(self._on_clear)
        handle_layout.addWidget(self._clear_btn)

        root.addWidget(handle)

        # ── Log body: a fixed-height viewport with a sliding sheet ────────
        self._log_body = _Body(self)
        self._log_body.setFixedHeight(_DRAWER_COLLAPSED)
        sheet_layout = QtWidgets.QVBoxLayout(self._log_body.sheet)
        sheet_layout.setContentsMargins(8, 0, 8, 8)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setObjectName("logText")
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(500)
        sheet_layout.addWidget(self._text)

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
        """Update the status chip. ``colour`` is a semantic name ("ok",
        "warn", "error", "running", "idle") or a hex; the hex values the
        workspaces send are mapped to the theme's log tokens."""
        key = _STATUS_TOKENS.get(str(colour).strip().lower())
        self._status_key = key or ""
        self._status_literal = None if key else colour
        self._status_chip.setText(f"● {text}")
        self._paint_status()

    def status_colour(self) -> str:
        """The chip's current colour (resolved against the active theme)."""
        if self._status_literal:
            return QtGui.QColor(self._status_literal).name()
        return getattr(get_manager().current, self._status_key or "log_ok")

    def is_expanded(self) -> bool:
        return self._expanded

    def toggle(self) -> None:
        from gui.themes import motion

        self._expanded = not self._expanded
        icon = "▼" if self._expanded else "▲"
        self._toggle_icon.setText(f"{icon}  Activity Log")
        body = self._log_body
        if self._expanded:
            # One relayout: take the full height, then slide the sheet in.
            body.setFixedHeight(_DRAWER_EXPANDED)
            body.place_sheet()
            motion.animate_value(
                self, "drawer", body.reveal, 1.0, self._on_reveal,
                duration="base", easing="standard",
            )
        else:
            motion.animate_value(
                self, "drawer", body.reveal, 0.0, self._on_reveal,
                duration="base", easing="exit", on_finished=self._on_collapsed,
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

    def _on_reveal(self, value: float) -> None:
        self._log_body.reveal = float(value)
        self._log_body.place_sheet()

    def _on_collapsed(self) -> None:
        if not self._expanded:
            self._log_body.setFixedHeight(_DRAWER_COLLAPSED)

    def _on_clear(self) -> None:
        self._text.clear()

    def _on_theme_changed(self, tokens: object) -> None:
        self._apply_theme(tokens)

    def _apply_theme(self, t) -> None:
        def _fg(widget: QtWidgets.QWidget, colour: str) -> None:
            pal = widget.palette()
            pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(colour))
            widget.setPalette(pal)

        _fg(self._toggle_icon, t.log_text)
        sheet_pal = self._log_body.sheet.palette()
        sheet_pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(t.log_bg))
        self._log_body.sheet.setPalette(sheet_pal)
        self._paint_status()
        self._recolour_lines(t)
        self._handle.update()

    def _paint_status(self) -> None:
        _colour = self.status_colour()
        pal = self._status_chip.palette()
        pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(_colour))
        self._status_chip.setPalette(pal)

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
