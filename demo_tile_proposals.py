#!/usr/bin/env python3
"""
demo_tile_proposals.py — Visual preview of the Liquid Glass tile design.

Run from the repository root:
    python demo_tile_proposals.py
"""
from __future__ import annotations
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gui.compat import QtCore, QtGui, QtWidgets

# ── Colour helpers ─────────────────────────────────────────────────────────────

def _tok():
    from gui.themes.manager import get_manager
    return get_manager().current

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"

def _animated_btn(text: str) -> QtWidgets.QPushButton:
    try:
        from gui.themes.animations import AnimatedButton
        return AnimatedButton(text)
    except Exception:
        return QtWidgets.QPushButton(text)

def _card_title(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("cardTitle")
    return lbl

class ElidedLabel(QtWidgets.QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setToolTip(text)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self):
        return QtCore.QSize(10, self.fontMetrics().height())

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        m = self.fontMetrics()
        elided = m.elidedText(self.text(), QtCore.Qt.TextElideMode.ElideRight, self.width())
        self.style().drawItemText(
            p, self.rect(), int(self.alignment()), self.palette(),
            self.isEnabled(), elided, self.foregroundRole()
        )

def _subtitle(text: str, wrap: bool = False) -> QtWidgets.QLabel:
    lbl = ElidedLabel(text) if not wrap else QtWidgets.QLabel(text)
    lbl.setObjectName("sectionSubtitle")
    if wrap:
        lbl.setWordWrap(True)
    lbl.setContentsMargins(0, 2, 0, 2)
    return lbl

def _muted(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("sectionSubtitle")
    return lbl

def _std_buttons() -> tuple[QtWidgets.QPushButton, QtWidgets.QPushButton]:
    run = _animated_btn("Export Artist / Title List")
    run.setObjectName("primaryBtn")
    run.setMinimumHeight(34)
    openf = _animated_btn("Open File")
    openf.setMinimumHeight(34)
    openf.setEnabled(False)
    return run, openf

_ICON  = "\u2197"   # ↗
_TITLE = "Artist / Title"
_DESC  = ("Scan the library and write Docs/artist_title_list.txt "
          "with every Artist \u2013 Title pair.")

# ══════════════════════════════════════════════════════════════════════════════
# LIQUID GLASS — Base Components
# ══════════════════════════════════════════════════════════════════════════════

class _SwitchTrack(QtWidgets.QWidget):
    """Internal painted track + knob for the heavy slider switch."""
    toggled = QtCore.Signal(bool)
    _W, _H, _KD, _PAD = 50, 28, 22, 3

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self._checked = checked
        self._pos = 1.0 if checked else 0.0
        self._anim: QtCore.QVariantAnimation | None = None
        self.setFixedSize(self._W, self._H)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        from gui.themes.manager import get_manager
        get_manager().theme_changed.connect(lambda _: self.update())

    def mousePressEvent(self, e):  # noqa: N802
        if not self.isEnabled():
            return
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._checked = not self._checked
            self._animate(1.0 if self._checked else 0.0)
            self.toggled.emit(self._checked)

    def _animate(self, target: float):
        if self._anim:
            self._anim.stop()
        a = QtCore.QVariantAnimation(self)
        a.setStartValue(self._pos)
        a.setEndValue(target)
        a.setDuration(440)
        a.setEasingCurve(QtCore.QEasingCurve.Type.OutBack)
        a.valueChanged.connect(lambda v: self._set(float(v)))
        a.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        self._anim = a

    def _set(self, v: float):
        self._pos = v
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = _tok()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.4)
        W, H, KD, PAD = self._W, self._H, self._KD, self._PAD

        frac = max(0.0, min(1.0, self._pos))
        on_c  = QtGui.QColor(t.accent)
        off_c = QtGui.QColor(t.card_border)
        track_c = QtGui.QColor(
            int(off_c.red()   + frac * (on_c.red()   - off_c.red())),
            int(off_c.green() + frac * (on_c.green() - off_c.green())),
            int(off_c.blue()  + frac * (on_c.blue()  - off_c.blue())),
        )
        track_c.setAlphaF(0.65 + 0.35 * frac)

        p.setBrush(track_c)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(QtCore.QRectF(0, 0, W, H), H / 2, H / 2)

        sheen = QtGui.QLinearGradient(0, 0, 0, H * 0.55)
        s1 = QtGui.QColor(255, 255, 255); s1.setAlphaF(0.20)
        s2 = QtGui.QColor(255, 255, 255); s2.setAlphaF(0.0)
        sheen.setColorAt(0.0, s1); sheen.setColorAt(1.0, s2)
        p.setBrush(sheen)
        p.drawRoundedRect(QtCore.QRectF(0, 0, W, H), H / 2, H / 2)

        travel = W - 2 * PAD - KD
        kx = max(-PAD * 0.5, min(W - KD + PAD * 0.5, PAD + self._pos * travel))
        ky = (H - KD) / 2.0

        p.setBrush(QtGui.QColor(0, 0, 0, 50))
        p.drawEllipse(QtCore.QRectF(kx + 1, ky + 2, KD, KD))

        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawEllipse(QtCore.QRectF(kx, ky, KD, KD))

        p.setBrush(QtGui.QColor(255, 255, 255, 140))
        dot = KD * 0.30
        p.drawEllipse(QtCore.QRectF(kx + KD * 0.22, ky + KD * 0.15, dot, dot))
        p.end()


class _HeavySliderSwitch(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)

    def __init__(self, label: str = "", checked: bool = False, parent=None):
        super().__init__(parent)
        lo = QtWidgets.QHBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(10)
        self._track = _SwitchTrack(checked=checked)
        lo.addWidget(self._track)
        if label:
            lbl = QtWidgets.QLabel(label)
            lbl.setObjectName("sectionSubtitle")
            lo.addWidget(lbl)
        lo.addStretch(1)
        self._track.toggled.connect(self.toggled)
        self.setFixedHeight(_SwitchTrack._H)

    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.Type.EnabledChange:
            self._track.setEnabled(self.isEnabled())
            for i in range(self.layout().count()):
                item = self.layout().itemAt(i)
                w = item.widget() if item else None
                if w: w.setEnabled(self.isEnabled())
        super().changeEvent(e)


def _glass_options(parent=None) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget(parent)
    lo = QtWidgets.QHBoxLayout(w)
    lo.setContentsMargins(0, 0, 0, 0)
    lo.setSpacing(28)
    lo.addWidget(_HeavySliderSwitch("Exclude FLAC"))
    
    # Disabled slider to demonstrate the disabled state
    hw = _HeavySliderSwitch("Per-album duplicates")
    hw.setEnabled(False)
    lo.addWidget(hw)
    
    lo.addStretch(1)
    return w


class _GlassBadge(QtWidgets.QWidget):
    def __init__(self, icon: str, size: int = 40, parent=None):
        super().__init__(parent)
        self._icon = icon
        self.setFixedSize(size, size)
        from gui.themes.manager import get_manager
        get_manager().theme_changed.connect(lambda _: self.update())

    def paintEvent(self, e):  # noqa: N802
        t = _tok()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        sz  = float(self.width())
        r   = QtCore.QRectF(0, 0, sz, sz)
        rad = sz * 0.28
        ac  = QtGui.QColor(t.accent)

        fill = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 55)
        p.setBrush(fill); p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, rad, rad)

        spec = QtGui.QLinearGradient(0, 0, 0, sz * 0.55)
        s1 = QtGui.QColor(255, 255, 255); s1.setAlphaF(0.38)
        s2 = QtGui.QColor(255, 255, 255); s2.setAlphaF(0.0)
        spec.setColorAt(0.0, s1); spec.setColorAt(1.0, s2)
        p.setBrush(spec)
        p.drawRoundedRect(r, rad, rad)

        tint = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 70)
        p.setBrush(tint)
        p.drawRoundedRect(r, rad, rad)

        rim = QtGui.QLinearGradient(0, 0, 0, sz)
        r1 = QtGui.QColor(255, 255, 255); r1.setAlphaF(0.65)
        r2 = QtGui.QColor(ac.red(), ac.green(), ac.blue()); r2.setAlphaF(0.35)
        rim.setColorAt(0.0, r1); rim.setColorAt(1.0, r2)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QBrush(rim), 1.1))
        p.drawRoundedRect(r.adjusted(0.55, 0.55, -0.55, -0.55), rad, rad)

        p.setPen(QtGui.QColor(t.text_inverse))
        font = QtGui.QFont(self.font().family(), int(sz * 0.42))
        font.setBold(True)
        p.setFont(font)
        p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._icon)
        p.end()


class _GlassResultChip(QtWidgets.QWidget):
    NEUTRAL = 0
    SUCCESS = 1
    ERROR   = 2

    _COLOR = {
        NEUTRAL: "#94a3b8",   
        SUCCESS: "#22c55e",   
        ERROR:   "#ef4444",   
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text  = "\u29d6  Idle"
        self._state = self.NEUTRAL
        self._opacity = 1.0
        self._anim: QtCore.QVariantAnimation | None = None
        self.setFixedHeight(26)
        self._resize()
        from gui.themes.manager import get_manager
        get_manager().theme_changed.connect(lambda _: self.update())

    def _resize(self):
        fm = QtGui.QFontMetrics(self.font())
        self.setFixedWidth(fm.horizontalAdvance(self._text) + 34)

    def show_neutral(self, text: str = "\u29d6  Idle"):
        self._transition(text, self.NEUTRAL)

    def show_success(self, text: str = "\u2713  Done"):
        self._transition(text, self.SUCCESS)

    def show_error(self, text: str = "\u2717  Error"):
        self._transition(text, self.ERROR)

    def _transition(self, text: str, state: int):
        if self._anim:
            self._anim.stop()
        a = QtCore.QVariantAnimation(self)
        a.setStartValue(self._opacity)
        a.setEndValue(0.0)
        a.setDuration(160)
        a.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        a.valueChanged.connect(lambda v: self._set_op(float(v)))

        def _swap():
            self._text  = text
            self._state = state
            self._resize()
            b = QtCore.QVariantAnimation(self)
            b.setStartValue(0.0)
            b.setEndValue(1.0)
            b.setDuration(260)
            b.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            b.valueChanged.connect(lambda v: self._set_op(float(v)))
            b.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
            self._anim = b

        a.finished.connect(_swap)
        a.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        self._anim = a

    def _set_op(self, v: float):
        self._opacity = v
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setOpacity(self._opacity)

        w, h = float(self.width()), float(self.height())
        r    = QtCore.QRectF(0, 0, w, h)
        rad  = h / 2.0
        ac   = QtGui.QColor(self._COLOR[self._state])

        fill = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 38)
        p.setBrush(fill); p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, rad, rad)

        spec = QtGui.QLinearGradient(0, 0, 0, h * 0.58)
        s1 = QtGui.QColor(255, 255, 255); s1.setAlphaF(0.28)
        s2 = QtGui.QColor(255, 255, 255); s2.setAlphaF(0.0)
        spec.setColorAt(0.0, s1); spec.setColorAt(1.0, s2)
        p.setBrush(spec)
        p.drawRoundedRect(r, rad, rad)

        bdr = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 170)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(bdr, 1.0))
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), rad, rad)

        p.setPen(ac)
        f = p.font(); f.setPointSize(10); f.setBold(True); p.setFont(f)
        p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()


class LiquidGlassTile(QtWidgets.QFrame):
    """
    Liquid Glass Tile — See-through translucent card.
    Features:
    - High-quality translucent frosted glass body that lets the background bleed through.
    - An animated inner glow / rim that gracefully flows towards the cursor.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("liquidGlassTile")
        self.setStyleSheet("QFrame#liquidGlassTile { background: transparent; border: none; }")
        self.setMouseTracking(True)
        self._hover_t = 0.0
        self._time = 0.0
        
        # We start the tracked glow target in the center
        self._target_pos = QtCore.QPointF(200, 100)
        self._glow_pos = QtCore.QPointF(200, 100)
        self._is_hovered = False

        self.setFixedWidth(440)

        # Make sure our inner widget components don't block mouse tracking
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

        lo = QtWidgets.QVBoxLayout(self)
        lo.setContentsMargins(18, 16, 18, 16)
        lo.setSpacing(12)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self._badge = _GlassBadge(_ICON, size=40)
        row.addWidget(self._badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(6)
        info.addWidget(_card_title(_TITLE))
        info.addWidget(_subtitle(_DESC))
        row.addLayout(info, 1)
        self._chip = _GlassResultChip()
        row.addWidget(self._chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        lo.addLayout(row)
        lo.addWidget(_glass_options())

        btn_row = QtWidgets.QHBoxLayout()
        r, o = _std_buttons()
        r.clicked.connect(lambda: self._chip.show_success("\u2713  847 entries"))
        btn_row.addWidget(r); btn_row.addWidget(o); btn_row.addStretch(1)
        lo.addLayout(btn_row)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def mouseMoveEvent(self, e):
        self._target_pos = e.position()
        super().mouseMoveEvent(e)

    def enterEvent(self, e):
        self._is_hovered = True
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._is_hovered = False
        super().leaveEvent(e)

    def _tick(self):
        self._time += 0.016
        
        # Smooth hover alpha
        target_h = 1.0 if self._is_hovered else 0.0
        self._hover_t += (target_h - self._hover_t) * 0.15

        # If not hovered, return to center gently
        if not self._is_hovered:
            self._target_pos = QtCore.QPointF(self.width() / 2, self.height() / 2)

        # Move the glow quickly but smoothly toward target_pos
        dx = self._target_pos.x() - self._glow_pos.x()
        dy = self._target_pos.y() - self._glow_pos.y()
        # 0.15 interpolation per frame gives it a "fast but not snappy" feel
        self._glow_pos.setX(self._glow_pos.x() + dx * 0.15)
        self._glow_pos.setY(self._glow_pos.y() + dy * 0.15)
        
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = _tok()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        rect = QtCore.QRectF(self.rect())
        inner = rect.adjusted(2, 2, -2, -2)
        radius = 16.0
        ht = self._hover_t
        ac = QtGui.QColor(t.accent)

        # Clip path for interior glass
        path = QtGui.QPainterPath()
        path.addRoundedRect(inner, radius, radius)
        p.save()
        p.setClipPath(path)

        # 1. Main Glass Body: Translucent 
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        # Deep translucent tint mixed with white for frost
        # Low alpha makes the background underneath highly visible
        bg_alpha = 0.05 + ht * 0.03
        p.setBrush(QtGui.QColor(255, 255, 255, int(bg_alpha * 255)))
        p.drawRect(inner)

        # Subtle dark tint for extra legibility while remaining translucent
        t_alpha = 0.1 + ht * 0.05
        # Adjust base tint depending on light/dark mode for consistent contrast
        is_dark = getattr(t, "is_dark", True)
        if is_dark:
            tint_color = QtGui.QColor(0, 0, 0, int(t_alpha * 255))
        else:
            tint_color = QtGui.QColor(255, 255, 255, int(t_alpha * 255))
        p.setBrush(tint_color)
        p.drawRect(inner)

        # 2. Dynamic Glow chasing the cursor
        # The glow is a radial gradient centered exactly on _glow_pos
        glow_rad = inner.width() * 0.8
        glow = QtGui.QRadialGradient(self._glow_pos, glow_rad)
        
        # Center of glow is strongly tinted with the accent color
        c1 = QtGui.QColor(ac); c1.setAlphaF(0.15 + ht * 0.15)
        # Fades to transparent
        c2 = QtGui.QColor(ac); c2.setAlphaF(0.0)
        
        glow.setColorAt(0.0, c1)
        glow.setColorAt(1.0, c2)
        p.setBrush(glow)
        p.drawRect(inner)

        # 3. Top specular reflection for glass depth
        spec = QtGui.QLinearGradient(0, inner.top(), 0, inner.bottom())
        spec.setColorAt(0.0, QtGui.QColor(255, 255, 255, int((0.15 + ht * 0.05) * 255)))
        spec.setColorAt(0.4, QtGui.QColor(255, 255, 255, 0))
        p.setBrush(spec)
        p.drawRect(inner)
        p.restore()
        
        p.end()

# ══════════════════════════════════════════════════════════════════════════════
# Demo window
# ══════════════════════════════════════════════════════════════════════════════

class ScalableTileWrapper(QtWidgets.QGraphicsView):
    """
    Wraps any widget in a QGraphicsView to allow fractional zoom scaling.
    This resizes everything including fonts, layouts, and custom drawing.
    """
    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Prevent the wrapper from drawing a background so the widget's transparency works
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.proxy = self.scene.addWidget(widget)
        self.widget = widget
        
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        
        self.setStyleSheet("background: transparent; border: none;")
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        
        self._target_scale = 1.0
        self._current_scale = 1.0
        
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        
        QtCore.QTimer.singleShot(0, self._set_fixed_max_size)

    def set_size_level(self, level: int):
        scales = {1: 0.77, 2: 0.88, 3: 1.0, 4: 1.18, 5: 1.36}
        if level in scales:
            self._target_scale = scales[level]

    def _tick(self):
        if abs(self._target_scale - self._current_scale) > 0.001:
            self._current_scale += (self._target_scale - self._current_scale) * 0.15
            self.resetTransform()
            self.scale(self._current_scale, self._current_scale)

    def _set_fixed_max_size(self):
        rect = self.scene.itemsBoundingRect()
        max_scale = 1.36
        w = rect.width() * max_scale
        h = rect.height() * max_scale
        self.setFixedSize(int(math.ceil(w)) + 2, int(math.ceil(h)) + 2)


class DemoWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaDEX — Liquid Glass Tile Demo")
        self.resize(1000, 700)
        self._build()

    def _build(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        try:
            from gui.widgets.gradient_bg import GradientWidget
            inner = GradientWidget()
        except Exception:
            inner = QtWidgets.QWidget()

        page = QtWidgets.QVBoxLayout(inner)
        page.setContentsMargins(28, 24, 28, 28)
        page.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        hdr = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(3)
        h = QtWidgets.QLabel("Liquid Glass Tile")
        h.setObjectName("sectionTitle")
        s = QtWidgets.QLabel(
            "Move your mouse over the card to see the dynamic glow track your cursor. "
            "The background is translucent."
        )
        s.setObjectName("sectionSubtitle")
        s.setWordWrap(True)
        title_col.addWidget(h)
        title_col.addWidget(s)
        hdr.addLayout(title_col, 1)

        # Size slider
        sc = QtWidgets.QVBoxLayout()
        sc.setSpacing(4)
        sl = QtWidgets.QLabel("Card Size (1-5)")
        sl.setObjectName("sectionSubtitle")
        self._size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._size_slider.setFixedWidth(166)
        self._size_slider.setMinimum(1)
        self._size_slider.setMaximum(5)
        self._size_slider.setValue(3)
        self._size_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._size_slider.setTickInterval(1)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        sc.addWidget(sl)
        sc.addWidget(self._size_slider)
        sc.addStretch(1)

        # Theme switcher
        tc = QtWidgets.QVBoxLayout()
        tc.setSpacing(4)
        tl = QtWidgets.QLabel("Theme")
        tl.setObjectName("sectionSubtitle")
        self._combo = QtWidgets.QComboBox()
        self._combo.setFixedWidth(166)
        from gui.themes.tokens import THEMES
        from gui.themes.manager import get_manager
        mgr = get_manager()
        for name in THEMES:
            self._combo.addItem(name.title(), name)
        idx = list(THEMES.keys()).index(mgr.current.key) if mgr.current.key in THEMES else 0
        self._combo.setCurrentIndex(idx)
        self._combo.currentIndexChanged.connect(self._switch_theme)
        tc.addWidget(tl)
        tc.addWidget(self._combo)

        btn_row = QtWidgets.QHBoxLayout()
        for label, key in [("Dark", "midnight"), ("Light", "pearl")]:
            b = QtWidgets.QPushButton(label)
            b.setFixedSize(79, 28)
            b.clicked.connect(lambda _, k=key: self._apply_named(k))
            btn_row.addWidget(b)
        tc.addLayout(btn_row)
        
        hdr.addLayout(sc)
        hdr.addSpacing(40)
        hdr.addLayout(tc)
        page.addLayout(hdr)
        page.addSpacing(32)

        # ── Featured: Liquid Glass ──────────────────────────────
        lg_row = QtWidgets.QHBoxLayout()
        self.lg_tile = LiquidGlassTile()
        self.lg_wrapper = ScalableTileWrapper(self.lg_tile)
        lg_row.addWidget(self.lg_wrapper, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        lg_row.addStretch(1)
        page.addLayout(lg_row)
        page.addSpacing(20)

        page.addStretch(1)
        scroll.setWidget(inner)
        self.setCentralWidget(scroll)

    def _on_size_changed(self, val: int):
        self.lg_wrapper.set_size_level(val)

    def _switch_theme(self, idx: int):
        key = self._combo.itemData(idx)
        from gui.themes.manager import get_manager
        get_manager().apply(key)

    def _apply_named(self, key: str):
        from gui.themes.tokens import THEMES
        from gui.themes.manager import get_manager
        get_manager().apply(key)
        keys = list(THEMES.keys())
        if key in keys:
            self._combo.setCurrentIndex(keys.index(key))

def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    try:
        from gui.fonts.loader import load_fonts
        load_fonts()
    except Exception:
        pass
    try:
        from gui.themes.manager import get_manager
        get_manager()   
    except Exception as exc:
        print(f"Theme init warning: {exc}")

    win = DemoWindow()
    win.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())