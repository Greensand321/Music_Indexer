"""AlphaDEX splash / loading screen.

Sequence
--------
1. Fast fill animation to 50% (750 ms, InOutCubic) — initial UI load.
2. Pause and monitor image loading — progress bar tracks actual image load count.
3. Once images are loaded (or timeout expires), complete fill from 50% to 100%.
4. Fade-out animation plays (450 ms, InCubic).
5. Fade completes → splash closes.

Colors are taken from the currently loaded ThemeManager tokens so the splash
matches whichever theme the user last saved.  If the manager is not yet
initialised the hardcoded Midnight-dark palette is used as a fallback.

The splash can receive image load progress updates via ``set_image_progress()``
so the progress bar reflects actual asset loading, not just elapsed time.
"""
from __future__ import annotations

import sys

from gui.compat import QtCore, QtGui, QtWidgets, Signal

_FILL_PHASE1_MS = 750   # Time to load to 50%
_FILL_PHASE2_MS = 1200  # Slower phase 2: time to load from 50% to 100% (smooth transition)
_IMAGE_LOAD_TIMEOUT_MS = 5000  # Max time to wait for images before continuing
_FADE_MS = 450
_SPINNER_ROTATION_MS = 2000  # Full rotation of loading spinner
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
    """Frameless branded loading screen with image-aware progress.

    Two-phase loading:
    1. Phase 1 (0 → 50%): Fast fill while UI loads (750ms).
    2. Phase 2 (50% → 100%): Monitored wait for images to load, or timeout.

    Usage::

        splash = SplashScreen()
        splash.set_image_progress.connect(scanner.art_found)  # optional
        splash.reveal_ready.connect(main_window.show)
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
        self._spinner_rotation: float = 0.0  # Rotation angle for loading spinner

        # ── Image loading progress tracking ────────────────────────────────
        self._images_loaded: int = 0
        self._images_target: int = 0  # Total images to expect
        self._phase2_started: bool = False
        self._image_load_timeout: QtCore.QTimer | None = None

        # ── Spinner animation ──────────────────────────────────────────────
        self._spinner_anim = QtCore.QVariantAnimation(self)
        self._spinner_anim.setStartValue(0.0)
        self._spinner_anim.setEndValue(360.0)
        self._spinner_anim.setDuration(_SPINNER_ROTATION_MS)
        self._spinner_anim.setLoopCount(-1)  # Infinite loop
        self._spinner_anim.valueChanged.connect(self._on_spinner_rotation)
        self._spinner_anim.start()

        # ── Phase 1: Fast fill to 50% ─────────────────────────────────────
        self._fill_phase1 = QtCore.QVariantAnimation(self)
        self._fill_phase1.setStartValue(0.0)
        self._fill_phase1.setEndValue(0.5)
        self._fill_phase1.setDuration(_FILL_PHASE1_MS)
        self._fill_phase1.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._fill_phase1.valueChanged.connect(self._on_progress)
        self._fill_phase1.finished.connect(self._on_phase1_done)
        self._fill_phase1.start()

        # ── Phase 2: Completion fill (50% → 100%) ────────────────────────
        self._fill_phase2 = QtCore.QVariantAnimation(self)
        self._fill_phase2.setStartValue(0.5)
        self._fill_phase2.setEndValue(1.0)
        self._fill_phase2.setDuration(_FILL_PHASE2_MS)
        self._fill_phase2.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._fill_phase2.valueChanged.connect(self._on_progress)
        self._fill_phase2.finished.connect(self._start_fade)

        # ── Window fade-out ────────────────────────────────────────────────
        self._fade_anim = QtCore.QVariantAnimation(self)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setDuration(_FADE_MS)
        self._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        self._fade_anim.valueChanged.connect(self._on_fade)
        self._fade_anim.finished.connect(self._on_done)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_image_target(self, count: int) -> None:
        """Set the expected number of images to load (for progress calculation)."""
        self._images_target = count

    def report_image_loaded(self, index: int, _image: object = None) -> None:
        """Report that an image has been loaded (called by art scanner).

        Args:
            index: Tile index being loaded (used to track progress).
            _image: Unused; present for signal compatibility with art_found.
        """
        self._images_loaded = index + 1  # Convert 0-based index to count
        # Update progress bar to reflect image loading during phase 2
        if self._phase2_started and self._images_target > 0:
            # Map image progress (0 to _images_target) to bar progress (0.5 to 1.0)
            image_fraction = min(1.0, self._images_loaded / self._images_target)
            phase2_progress = 0.5 + (image_fraction * 0.5)
            self._progress = phase2_progress
            self.update()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_progress(self, value: object) -> None:
        self._progress = float(value)  # type: ignore[arg-type]
        self.update()

    def _on_spinner_rotation(self, value: object) -> None:
        """Update spinner rotation angle for continuous animation."""
        self._spinner_rotation = float(value)  # type: ignore[arg-type]
        self.update()

    def _on_phase1_done(self) -> None:
        """Phase 1 (0-50%) complete. Wait for images, then continue to phase 2."""
        # Set up timeout to continue even if no images load
        self._image_load_timeout = QtCore.QTimer()
        self._image_load_timeout.setSingleShot(True)
        self._image_load_timeout.timeout.connect(self._continue_to_phase2)
        self._image_load_timeout.start(_IMAGE_LOAD_TIMEOUT_MS)

    def _continue_to_phase2(self) -> None:
        """Continue fill animation from 50% to 100% (phase 2)."""
        # Stop timeout if still running
        if self._image_load_timeout is not None:
            self._image_load_timeout.stop()
            self._image_load_timeout = None
        # Mark phase 2 as started so progress updates during phase 2 work
        self._phase2_started = True
        # Start phase 2 animation
        self._fill_phase2.start()

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

        # Draw text shadow for better readability (subtle dark shadow)
        shadow_color = QtGui.QColor(0, 0, 0, 60)
        p.setPen(shadow_color)
        p.drawText(
            QtCore.QRect(2, 74, _W, 96),  # Slightly offset for shadow
            int(QtCore.Qt.AlignmentFlag.AlignCenter),
            "AlphaDEX",
        )

        # Draw main text in accent color for visibility and theme complement
        p.setPen(QtGui.QColor(c["accent"]))
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

        # Subtitle shadow for readability
        p.setPen(QtGui.QColor(0, 0, 0, 40))
        p.drawText(
            QtCore.QRect(1, 169, _W, 28),  # Slightly offset
            int(QtCore.Qt.AlignmentFlag.AlignCenter),
            "Music Library Manager  ·  v2.0",
        )

        # Subtitle text - use accent2 for visual harmony with the accent theme color
        p.setPen(QtGui.QColor(c["accent2"]))
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

        # ── Loading spinner (rotating circle) ──────────────────────────────
        # Draw a rotating circle above the progress bar
        spinner_cx = _W / 2.0
        spinner_cy = bar_y - 20.0  # Above the progress bar
        spinner_r = 10.0

        # Save painter state before rotation
        p.save()
        p.translate(spinner_cx, spinner_cy)
        p.rotate(self._spinner_rotation)

        # Draw spinning arc (12 o'clock position)
        spinner_pen = QtGui.QPen(QtGui.QColor(c["accent"]), 2.5)
        spinner_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        p.setPen(spinner_pen)

        # Draw an arc from 45 to 315 degrees (270 degrees of arc)
        arc_path = QtGui.QPainterPath()
        arc_rect = QtCore.QRectF(-spinner_r, -spinner_r, spinner_r * 2, spinner_r * 2)
        arc_path.arcMoveTo(arc_rect, 45.0)
        arc_path.arcTo(arc_rect, 45.0, 270.0)
        p.drawPath(arc_path)

        p.restore()  # Restore painter state

        p.end()
