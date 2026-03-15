"""Slide-up activity log drawer anchored to the bottom of the window."""
from __future__ import annotations

import datetime

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.theme import LOG_BG, LOG_TEXT, LOG_TEXT_OK, LOG_TEXT_WARN, LOG_TEXT_ERR

_DRAWER_EXPANDED = 220
_DRAWER_COLLAPSED = 0
_HANDLE_HEIGHT = 30


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

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toggle handle ─────────────────────────────────────────────────
        handle = QtWidgets.QWidget()
        handle.setFixedHeight(_HANDLE_HEIGHT)
        handle.setStyleSheet(f"background: #1e293b;")
        handle.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        handle_layout = QtWidgets.QHBoxLayout(handle)
        handle_layout.setContentsMargins(16, 0, 16, 0)

        self._toggle_icon = QtWidgets.QLabel("▲  Activity Log")
        self._toggle_icon.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600;")
        handle_layout.addWidget(self._toggle_icon)
        handle_layout.addStretch(1)

        self._status_chip = QtWidgets.QLabel("● Idle")
        self._status_chip.setStyleSheet("color: #4ade80; font-size: 12px;")
        handle_layout.addWidget(self._status_chip)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setFixedSize(50, 20)
        clear_btn.setStyleSheet(
            "background: transparent; border: 1px solid #334155; "
            "color: #64748b; font-size: 11px; border-radius: 4px;"
        )
        clear_btn.clicked.connect(self._on_clear)
        handle_layout.addWidget(clear_btn)

        handle.mousePressEvent = lambda _e: self.toggle()
        root.addWidget(handle)

        # ── Log text area ─────────────────────────────────────────────────
        self._log_body = QtWidgets.QWidget()
        self._log_body.setStyleSheet(f"background: {LOG_BG};")
        self._log_body.setFixedHeight(0)
        log_inner = QtWidgets.QVBoxLayout(self._log_body)
        log_inner.setContentsMargins(0, 0, 0, 0)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setObjectName("logText")
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(500)
        log_inner.addWidget(self._text)

        root.addWidget(self._log_body)

        # Height animation
        self._anim = QtCore.QPropertyAnimation(self._log_body, b"maximumHeight")
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)

    # ── Public API ────────────────────────────────────────────────────────

    def append(self, message: str, level: str = "info") -> None:
        """Add a timestamped message. level: 'ok' | 'warn' | 'error' | 'info'"""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        colour_map = {
            "ok":    LOG_TEXT_OK,
            "warn":  LOG_TEXT_WARN,
            "error": LOG_TEXT_ERR,
            "info":  LOG_TEXT,
        }
        colour = colour_map.get(level, LOG_TEXT)
        # Insert coloured HTML line
        cursor = self._text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(colour))
        cursor.setCharFormat(fmt)
        cursor.insertText(f"[{ts}] {message}\n")
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

        # Auto-expand on errors / warnings
        if level in ("error", "warn") and not self._expanded:
            self.toggle()

    def set_status(self, text: str, colour: str = "#4ade80") -> None:
        """Update the status chip in the handle bar."""
        self._status_chip.setText(f"● {text}")
        self._status_chip.setStyleSheet(f"color: {colour}; font-size: 12px;")

    def toggle(self) -> None:
        self._expanded = not self._expanded
        target = _DRAWER_EXPANDED if self._expanded else _DRAWER_COLLAPSED
        icon = "▼" if self._expanded else "▲"
        self._toggle_icon.setText(f"{icon}  Activity Log")
        self._anim.stop()
        self._anim.setStartValue(self._log_body.maximumHeight())
        self._anim.setEndValue(target)
        self._anim.start()

    # ── Private ───────────────────────────────────────────────────────────

    def _on_clear(self) -> None:
        self._text.clear()
