"""AlphaDEX splash / loading screen.

Sequence
--------
1. Fill animation plays (1 500 ms, InOutCubic).
2. Fill completes → ``reveal_ready`` is emitted so the caller can show the
   main window *behind* the splash while it is still visible.
3. Fade-out animation plays (450 ms, InCubic).
4. Fade completes → splash closes.

Colors are taken from the currently loaded ThemeManager tokens so the splash
matches whichever theme the user last saved.  If the manager is not yet
initialised the hardcoded Midnight-dark palette is used as a fallback.
"""
from __future__ import annotations

import sys

from gui.compat import QtCore, QtGui, QtWidgets, Signal

_FILL_MS = 1500
_FADE_MS = 450
_W, _H   = 520, 300


def _theme_colors() -> dict[str, str]:
    """Return a palette dict drawn from the active theme, or a safe fallback.

    Fallback palette (Midnight dark) is used if theme manager is unavailable
    or if an error occurs during theme loading. Errors are logged to stderr.
    """
    try:
        from gui.themes.manager import get_manager
        t = get_manager().current
        return {
            "bg":        t.sidebar_bg,
            "surface":   t.card_bg,
            "border":    t.card_border,
            "text":      t.text_primary,
            "subtext":   t.text_secondary,
            "accent":    t.accent,
            "accent2":   t.accent_hover,
            "bar_track": t.card_bg,
        }
    except ImportError:
        # Theme manager module not available; use fallback (normal on first startup)
        pass
    except AttributeError as e:
        # Real error in theme structure; log it for debugging
        print(f"[Warning] Theme manager attribute error: {e}; using fallback", file=sys.stderr)
    except Exception as e:
        # Other unexpected errors; log for debugging
        print(f"[Warning] Theme loading failed: {e}; using fallback", file=sys.stderr)

    # Fallback palette (Midnight dark theme)
    return {
        "bg":        "#0d1117",
        "surface":   "#161b22",
        "border":    "#30363d",
        "text":      "#f8fafc",
        "subtext":   "#64748b",
        "accent":    "#6366f1",
        "accent2":   "#a78bfa",
        "bar_track": "#1e2130",
    }


class SplashScreen(QtWidgets.QWidget):
    """Frameless branded loading screen.

    Usage::

        splash = SplashScreen()
        splash.reveal_ready.connect(main_window.show)   # show window beneath fade
        splash.show()
    """

    reveal_ready = Signal()   # emitted when the fade-out begins
    finished     = Signal()   # emitted when the window closes

    def __init__(self) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(_W, _H)
        self._center_on_screen()

        self._progress: float = 0.0
        self._colors = _theme_colors()   # snapshot once at creation

        # ── Progress fill ─────────────────────────────────────────────────
        self._fill_anim = QtCore.QVariantAnimation(self)
        self._fill_anim.setStartValue(0.0)
        self._fill_anim.setEndValue(1.0)
        self._fill_anim.setDuration(_FILL_MS)
        self._fill_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._fill_anim.valueChanged.connect(self._on_progress)
        self._fill_anim.finished.connect(self._start_fade)
        self._fill_anim.start()

        # ── Window fade-out ────────────────────────────────────────────────
        self._fade_anim = QtCore.QVariantAnimation(self)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setDuration(_FADE_MS)
        self._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        self._fade_anim.valueChanged.connect(self._on_fade)
        self._fade_anim.finished.connect(self._on_done)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_progress(self, value: object) -> None:
        self._progress = float(value)  # type: ignore[arg-type]
        self.update()

    def _start_fade(self) -> None:
        # Reveal the main window while the splash is still opaque, then fade.
        self.reveal_ready.emit()
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
        c = self._colors
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)

        rect = QtCore.QRectF(self.rect())

        # ── Card background ───────────────────────────────────────────────
        card = rect.adjusted(1, 1, -1, -1)
        card_path = QtGui.QPainterPath()
        card_path.addRoundedRect(card, 16, 16)

        p.fillPath(card_path, QtGui.QBrush(QtGui.QColor(c["bg"])))

        # Subtle border — semi-transparent white for dark themes, card_border
        # for light themes (card_border is already theme-appropriate).
        border_col = QtGui.QColor(c["border"])
        border_col.setAlpha(max(border_col.alpha(), 40))
        p.setPen(QtGui.QPen(border_col, 1.0))
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
        p.setPen(QtGui.QColor(c["text"]))
        p.drawText(
            QtCore.QRect(0, 72, _W, 96),
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
        p.setPen(QtGui.QColor(c["subtext"]))
        p.drawText(
            QtCore.QRect(0, 168, _W, 28),
            int(QtCore.Qt.AlignmentFlag.AlignCenter),
            "Music Library Manager  ·  v2.0",
        )

        # ── Progress bar ──────────────────────────────────────────────────
        margin   = 56
        bar_x    = float(margin)
        bar_y    = float(_H - 52)
        bar_w    = float(_W - margin * 2)
        bar_h    = 5.0
        radius   = bar_h / 2.0

        # Track
        track_path = QtGui.QPainterPath()
        track_path.addRoundedRect(
            QtCore.QRectF(bar_x, bar_y, bar_w, bar_h), radius, radius
        )
        p.fillPath(track_path, QtGui.QBrush(QtGui.QColor(c["bar_track"])))

        # Fill with gradient
        fill_w = bar_w * self._progress
        if fill_w > 0:
            fill_path = QtGui.QPainterPath()
            fill_path.addRoundedRect(
                QtCore.QRectF(bar_x, bar_y, fill_w, bar_h), radius, radius
            )
            grad = QtGui.QLinearGradient(bar_x, 0.0, bar_x + bar_w, 0.0)
            grad.setColorAt(0.0, QtGui.QColor(c["accent"]))
            grad.setColorAt(1.0, QtGui.QColor(c["accent2"]))
            p.fillPath(fill_path, QtGui.QBrush(grad))

            # Glow dot at leading edge
            dot_cx = bar_x + fill_w
            dot_cy = bar_y + bar_h / 2.0
            dot_r  = 4.5
            dot_path = QtGui.QPainterPath()
            dot_path.addEllipse(
                QtCore.QRectF(
                    dot_cx - dot_r, dot_cy - dot_r, dot_r * 2, dot_r * 2
                )
            )
            glow = QtGui.QColor(c["accent2"])
            glow.setAlpha(220)
            p.fillPath(dot_path, QtGui.QBrush(glow))

        p.end()
