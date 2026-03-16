"""AlphaDEX splash / loading screen.

Shows a branded loading screen with an animated progress bar that fills over
1.5 seconds, then fades the window out before emitting ``finished`` so the
caller can show the main window.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal

# Hard-coded palette so the splash works before ThemeManager is initialised.
_BG          = "#0d1117"
_SURFACE     = "#161b22"
_ACCENT      = "#6366f1"
_ACCENT_DIM  = "#1e1f3a"
_TEXT        = "#f8fafc"
_SUBTEXT     = "#64748b"
_BAR_TRACK   = "#1e2130"

_FILL_MS  = 1500   # progress bar fill duration
_FADE_MS  =  500   # window fade-out duration
_W, _H    =  520, 300


class SplashScreen(QtWidgets.QWidget):
    """Frameless branded loading screen.

    Usage::

        splash = SplashScreen()
        splash.finished.connect(main_window.show)
        splash.show()
    """

    finished = Signal()

    def __init__(self) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(_W, _H)
        self._center_on_screen()

        self._progress: float = 0.0   # 0.0 → 1.0

        # ── Progress bar fill animation ───────────────────────────────────
        self._fill_anim = QtCore.QVariantAnimation(self)
        self._fill_anim.setStartValue(0.0)
        self._fill_anim.setEndValue(1.0)
        self._fill_anim.setDuration(_FILL_MS)
        self._fill_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._fill_anim.valueChanged.connect(self._on_progress)
        self._fill_anim.finished.connect(self._start_fade)
        self._fill_anim.start()

        # ── Fade-out animation (started after fill completes) ─────────────
        self._fade_anim = QtCore.QVariantAnimation(self)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setDuration(_FADE_MS)
        self._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        self._fade_anim.valueChanged.connect(self._on_fade)
        self._fade_anim.finished.connect(self._on_done)

    # ── Internal slots ────────────────────────────────────────────────────

    def _on_progress(self, value: object) -> None:
        self._progress = float(value)  # type: ignore[arg-type]
        self.update()

    def _start_fade(self) -> None:
        self._fade_anim.start()

    def _on_fade(self, value: object) -> None:
        self.setWindowOpacity(float(value))  # type: ignore[arg-type]

    def _on_done(self) -> None:
        self.finished.emit()
        self.close()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _center_on_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.move(
            geo.left() + (geo.width()  - _W) // 2,
            geo.top()  + (geo.height() - _H) // 2,
        )

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)

        rect = QtCore.QRectF(self.rect())

        # ── Card background (rounded, slight border) ──────────────────────
        card = rect.adjusted(1, 1, -1, -1)
        card_path = QtGui.QPainterPath()
        card_path.addRoundedRect(card, 18, 18)

        p.fillPath(card_path, QtGui.QBrush(QtGui.QColor(_BG)))

        pen = QtGui.QPen(QtGui.QColor("#ffffff18"), 1.0)
        p.setPen(pen)
        p.drawPath(card_path)
        p.setPen(QtCore.Qt.PenStyle.NoPen)

        # ── Brand name ────────────────────────────────────────────────────
        try:
            from gui.fonts.loader import UI_FAMILY
            family = UI_FAMILY
        except ImportError:
            family = "Arial"

        brand_font = QtGui.QFont(family, 52)
        brand_font.setWeight(QtGui.QFont.Weight.Bold)
        brand_font.setHintingPreference(
            QtGui.QFont.HintingPreference.PreferNoHinting
        )
        p.setFont(brand_font)
        p.setPen(QtGui.QColor(_TEXT))
        p.drawText(
            QtCore.QRect(0, 80, _W, 90),
            int(QtCore.Qt.AlignmentFlag.AlignCenter),
            "AlphaDEX",
        )

        # ── Subtitle ──────────────────────────────────────────────────────
        sub_font = QtGui.QFont(family, 11)
        sub_font.setWeight(QtGui.QFont.Weight.Normal)
        sub_font.setHintingPreference(
            QtGui.QFont.HintingPreference.PreferNoHinting
        )
        p.setFont(sub_font)
        p.setPen(QtGui.QColor(_SUBTEXT))
        p.drawText(
            QtCore.QRect(0, 170, _W, 28),
            int(QtCore.Qt.AlignmentFlag.AlignCenter),
            "Music Library Manager  ·  v2.0",
        )

        # ── Progress bar ──────────────────────────────────────────────────
        bar_margin = 56
        bar_x      = float(bar_margin)
        bar_y      = float(_H - 56)
        bar_w      = float(_W - bar_margin * 2)
        bar_h      = 6.0
        radius     = bar_h / 2.0

        # Track
        track = QtCore.QRectF(bar_x, bar_y, bar_w, bar_h)
        track_path = QtGui.QPainterPath()
        track_path.addRoundedRect(track, radius, radius)
        p.fillPath(track_path, QtGui.QBrush(QtGui.QColor(_BAR_TRACK)))

        # Fill
        fill_w = bar_w * self._progress
        if fill_w > 0:
            fill = QtCore.QRectF(bar_x, bar_y, fill_w, bar_h)
            fill_path = QtGui.QPainterPath()
            fill_path.addRoundedRect(fill, radius, radius)

            grad = QtGui.QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
            grad.setColorAt(0.0, QtGui.QColor(_ACCENT))
            grad.setColorAt(1.0, QtGui.QColor("#a78bfa"))
            p.fillPath(fill_path, QtGui.QBrush(grad))

            # Glow dot at leading edge
            dot_cx = bar_x + fill_w
            dot_cy = bar_y + bar_h / 2.0
            dot_r  = 5.0
            dot_path = QtGui.QPainterPath()
            dot_path.addEllipse(
                QtCore.QRectF(dot_cx - dot_r, dot_cy - dot_r, dot_r * 2, dot_r * 2)
            )
            p.fillPath(dot_path, QtGui.QBrush(QtGui.QColor("#c4b5fd")))

        p.end()
