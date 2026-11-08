"""The floating Now Playing capsule — transport on every page.

A glass capsule (the material's ``capsule`` surface: fill .56, border .18,
top highlight .30) floating at the bottom centre of the page area, 22 px up,
at most 880 px wide. The main window places it (it is an overlay, not in a
layout) and pages keep ``SPACE.page_bottom`` (104 px) of room under their
content so it never covers anything.

Contents, left to right: a 52 px round cover; the title (14/650) and
"artist · album" (12, muted); previous / play-pause (a 46 px white disc) /
next; the position, a 5 px seek track with an accent fill and a white knob,
and the duration (11 px, tabular figures); volume; and the keyboard-mode
pill.

It stays hidden until something has played: an empty transport floating
over every page would cover content and offer controls that do nothing.
When a track loads it fades in (with its shadow); when the queue is cleared
it hides at once.

Signals emitted (routed back to PlayerWorkspace by AlphaDEXWindow):
    play_pause_requested
    next_requested
    prev_requested
    seek_requested(int)   — target position in ms (drag release, wheel,
                            keys, groove click)
    volume_changed(int)   — 0-100, from any user interaction
    keyboard_mode_toggled(bool)  — arrow-key navigation mode

``update_now_playing(title, artist, album, pm)``: ``pm`` None keeps the
current thumbnail (art still loading); a *null* QPixmap means "no art"
(♪ placeholder); an empty title with a null pixmap means nothing is
loaded any more — the capsule resets and hides.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.manager import get_manager


def _round_pixmap(pm: QtGui.QPixmap, size: int) -> QtGui.QPixmap:
    """``pm`` cropped to a circle of ``size`` (device-pixel aware)."""
    dpr = pm.devicePixelRatio() or 1.0
    px = int(round(size * dpr))
    scaled = pm.scaled(px, px, QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                       QtCore.Qt.TransformationMode.SmoothTransformation)
    out = QtGui.QPixmap(px, px)
    out.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(out)
    try:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QRectF(0, 0, px, px))
        p.setClipPath(path)
        p.drawPixmap(-(scaled.width() - px) // 2, -(scaled.height() - px) // 2, scaled)
    finally:
        p.end()
    out.setDevicePixelRatio(dpr)
    return out


class _ArtLabel(QtWidgets.QLabel):
    """The round cover; a ♪ disc when there is no art."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            super().paintEvent(event)
            return
        from gui.themes.glass import current_material, qcolor
        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(qcolor(current_material().raised.fill))
            p.drawEllipse(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))
            p.setPen(QtGui.QColor(t.text_muted))
            p.setFont(self.font())
            p.drawText(self.rect(), int(QtCore.Qt.AlignmentFlag.AlignCenter), self.text())
        finally:
            p.end()


class _TransportButton(QtWidgets.QPushButton):
    """A round transport button. ``text()`` carries the state ("▶", "⏸",
    "⏮", "⏭") and the accessible name; the icon is painted.

    ``disc=True``: the 46 px white play/pause disc with a dark icon;
    otherwise a 40 px transparent button with a light icon."""

    def __init__(self, glyph: str, tip: str, *, disc: bool = False,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(glyph, parent)
        self._disc = disc
        self.setFlat(True)
        size = 46 if disc else 40
        self.setFixedSize(size, size)
        self.setToolTip(tip)
        self.setAccessibleName(tip)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, qcolor

        t = get_manager().current
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            hover = self.underMouse() and self.isEnabled()
            if self._disc:
                fill = qcolor(t.text_primary, 0.92 if m.is_glass else 1.0)
                if self.isDown():
                    fill = fill.darker(115)
                elif hover:
                    fill = qcolor(t.text_primary, 1.0)
                p.setBrush(fill)
                p.setPen(QtGui.QPen(qcolor("rgba(255,255,255,0.3)") if m.is_glass
                                    else QtGui.QColor(t.card_border), 1.0))
                p.drawEllipse(r)
                ink = QtGui.QColor(t.text_inverse)
            else:
                if self.isDown() or hover:
                    p.setPen(QtCore.Qt.PenStyle.NoPen)
                    p.setBrush(qcolor(m.pressed if self.isDown() else m.hover))
                    p.drawEllipse(r)
                ink = QtGui.QColor(t.text_primary if self.isEnabled() else t.text_muted)
            self._paint_glyph(p, r.center(), ink)
        finally:
            p.end()

    def _paint_glyph(self, p: QtGui.QPainter, c: QtCore.QPointF, ink: QtGui.QColor) -> None:
        """18 px icons from the mockup's 24-unit paths."""
        k = 18.0 / 24.0
        ox, oy = c.x() - 12 * k, c.y() - 12 * k

        def pt(x: float, y: float) -> QtCore.QPointF:
            return QtCore.QPointF(ox + x * k, oy + y * k)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(ink)
        glyph = self.text()
        if glyph == "⏸":
            p.drawRect(QtCore.QRectF(pt(7, 5), pt(11, 19)))
            p.drawRect(QtCore.QRectF(pt(13, 5), pt(17, 19)))
        elif glyph == "⏮":
            p.drawRect(QtCore.QRectF(pt(6, 5), pt(8, 19)))
            p.drawPolygon(QtGui.QPolygonF([pt(20, 5), pt(20, 19), pt(9, 12)]))
        elif glyph == "⏭":
            p.drawRect(QtCore.QRectF(pt(16, 5), pt(18, 19)))
            p.drawPolygon(QtGui.QPolygonF([pt(4, 5), pt(4, 19), pt(15, 12)]))
        else:   # ▶
            p.drawPolygon(QtGui.QPolygonF([pt(8, 5), pt(8, 19), pt(19, 12)]))


class _Speaker(QtWidgets.QWidget):
    """A small painted speaker glyph next to the volume slider."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(18, 18)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            ink = QtGui.QColor(t.text_secondary)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(ink)
            k = 18.0 / 24.0
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x * k, y * k) for x, y in (
                (3, 9), (7, 9), (12, 5), (12, 19), (7, 15), (3, 15))]))
            pen = QtGui.QPen(ink, 1.6, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawArc(QtCore.QRectF(11 * k, 8 * k, 6 * k, 8 * k), -60 * 16, 120 * 16)
            p.drawArc(QtCore.QRectF(11 * k, 5 * k, 10 * k, 14 * k), -60 * 16, 120 * 16)
        finally:
            p.end()


class _KbToggle(QtWidgets.QPushButton):
    """Keyboard-mode pill: raised glass at rest, the accent when on."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("⌨", parent)
        self.setCheckable(True)
        self.setFlat(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface, qcolor
        from gui.themes.tokens import PILL, Surface

        t = get_manager().current
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            r = QtCore.QRectF(self.rect())
            if self.isChecked():
                fill = t.accent_hover if self.underMouse() else t.accent
                paint_surface(p, r, Surface(fill, radius=PILL))
                ink = QtGui.QColor(t.text_inverse)
            else:
                paint_surface(p, r, m.raised,
                              fill=qcolor(m.hover) if self.underMouse() and not m.is_glass else None)
                ink = QtGui.QColor(t.accent if self.underMouse() else t.text_secondary)
            p.setPen(ink)
            p.setFont(self.font())
            p.drawText(self.rect(), int(QtCore.Qt.AlignmentFlag.AlignCenter), self.text())
        finally:
            p.end()


class NowPlayingBar(QtWidgets.QWidget):
    """The floating Now Playing capsule (see the module docstring)."""

    play_pause_requested = Signal()
    next_requested       = Signal()
    prev_requested       = Signal()
    seek_requested       = Signal(int)   # ms
    volume_changed       = Signal(int)   # 0-100
    keyboard_mode_toggled = Signal(bool)

    _ART_SIZE = 52     # px, round cover
    HEIGHT = 72        # 52 + 2 × 10 padding
    MAX_WIDTH = 880
    _KB_PULSE_MS = 120

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setObjectName("nowPlayingBar")
        self.setAutoFillBackground(False)

        self._is_playing  = False
        self._duration_ms = 0
        self._user_seeking = False
        self._shadow = None        # ShadowLayer, once placed in a window
        self._appear_fx: QtWidgets.QGraphicsOpacityEffect | None = None

        self._build_ui()
        get_manager().theme_changed.connect(self._apply_theme)
        self._apply_theme(get_manager().current)
        self.setVisible(False)   # hidden until first track is loaded

    # ── Construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from gui.themes.glass import font

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 16, 10)
        root.setSpacing(14)

        # Round cover
        self._art_lbl = _ArtLabel()
        self._art_lbl.setObjectName("npbArt")
        self._art_lbl.setFixedSize(self._ART_SIZE, self._ART_SIZE)
        self._art_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._art_lbl.setFont(font("title"))
        self._art_lbl.setText("♪")
        root.addWidget(self._art_lbl)

        # Title + "artist · album"
        meta = QtWidgets.QWidget()
        meta.setMinimumWidth(140)
        meta.setMaximumWidth(220)
        meta_col = QtWidgets.QVBoxLayout(meta)
        meta_col.setContentsMargins(0, 0, 0, 0)
        meta_col.setSpacing(1)
        self._title_lbl = QtWidgets.QLabel("—")
        self._title_lbl.setObjectName("capsuleTitle")
        self._title_lbl.setFont(font("body", weight=650))
        self._title_lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
        self._artist_lbl = QtWidgets.QLabel("")
        self._artist_lbl.setObjectName("capsuleMeta")
        self._artist_lbl.setFont(font("caption"))
        meta_col.addStretch(1)
        meta_col.addWidget(self._title_lbl)
        meta_col.addWidget(self._artist_lbl)
        meta_col.addStretch(1)
        root.addWidget(meta, 1)

        # Transport
        self._prev_btn = _TransportButton("⏮", "Previous track")
        self._prev_btn.clicked.connect(self.prev_requested)
        self._play_btn = _TransportButton("▶", "Play / Pause  [Space]", disc=True)
        self._play_btn.clicked.connect(self.play_pause_requested)
        self._next_btn = _TransportButton("⏭", "Next track")
        self._next_btn.clicked.connect(self.next_requested)
        transport = QtWidgets.QHBoxLayout()
        transport.setSpacing(6)
        for b in (self._prev_btn, self._play_btn, self._next_btn):
            transport.addWidget(b)
        root.addLayout(transport)

        # Progress: time · track · duration
        time_font = font("caption", px=11, tabular=True)
        self._pos_lbl = QtWidgets.QLabel("0:00")
        self._pos_lbl.setObjectName("capsuleTime")
        self._pos_lbl.setFont(time_font)
        self._pos_lbl.setMinimumWidth(30)
        self._pos_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self._seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._seek_slider.setObjectName("capsuleSeek")
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setMinimumWidth(120)
        self._seek_slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                        QtWidgets.QSizePolicy.Policy.Fixed)
        self._seek_slider.setToolTip("Seek")
        self._seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self._seek_slider.sliderReleased.connect(self._on_slider_released)
        # Wheel / keys / groove clicks change the value without a press.
        self._seek_slider.valueChanged.connect(self._on_seek_value_changed)

        self._dur_lbl = QtWidgets.QLabel("0:00")
        self._dur_lbl.setObjectName("capsuleTime")
        self._dur_lbl.setFont(time_font)
        self._dur_lbl.setMinimumWidth(30)

        progress = QtWidgets.QHBoxLayout()
        progress.setSpacing(10)
        progress.addWidget(self._pos_lbl)
        progress.addWidget(self._seek_slider, 1)
        progress.addWidget(self._dur_lbl)
        root.addLayout(progress, 3)

        # Volume
        self._vol_icon = _Speaker()
        self._vol_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._vol_slider.setObjectName("capsuleVolume")
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setFixedWidth(72)
        self._vol_slider.setToolTip("Volume")
        # Any user change (drag, wheel, keys, groove click) is emitted;
        # set_volume() blocks signals, so syncing from the page can't echo.
        self._vol_slider.valueChanged.connect(self.volume_changed)
        volume = QtWidgets.QHBoxLayout()
        volume.setSpacing(6)
        volume.addWidget(self._vol_icon)
        volume.addWidget(self._vol_slider)
        root.addLayout(volume)

        # ── Keyboard mode toggle (pill button) ────────────────────────────
        self._kb_btn = _KbToggle()
        self._kb_btn.setFixedSize(40, 30)
        self._kb_btn.setFont(font("body"))
        self._kb_btn.setToolTip(
            "Keyboard Mode: ↓ preview next  → next  ← prev  ↑ restart"
        )
        self._kb_btn.setObjectName("kbModeBtn")
        self._kb_btn.toggled.connect(self._on_kb_toggled)
        root.addWidget(self._kb_btn)

        # Press pulse: animate an opacity effect, never the geometry of a
        # layout-managed widget (that drifted when toggled quickly). One
        # reusable, parented animation that is restarted, not stacked.
        self._kb_fx = QtWidgets.QGraphicsOpacityEffect(self._kb_btn)
        self._kb_fx.setOpacity(1.0)
        self._kb_fx.setEnabled(False)
        self._kb_btn.setGraphicsEffect(self._kb_fx)
        self._kb_anim = QtCore.QPropertyAnimation(self._kb_fx, b"opacity", self)
        self._kb_anim.setDuration(self._KB_PULSE_MS)
        self._kb_anim.setKeyValueAt(0.0, 1.0)
        self._kb_anim.setKeyValueAt(0.5, 0.55)
        self._kb_anim.setKeyValueAt(1.0, 1.0)
        self._kb_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._kb_anim.finished.connect(self._on_kb_pulse_finished)

    # ── Floating placement (the main window owns it) ──────────────────────

    def attach_shadow(self) -> None:
        """Give the capsule its shadow, painted over the page beneath it
        (call once it has its parent)."""
        if self._shadow is None and self.parentWidget() is not None:
            from gui.widgets.glass import ShadowLayer
            self._shadow = ShadowLayer(self, "capsule")

    def place(self, area: QtCore.QRect) -> None:
        """Float at the bottom centre of ``area`` (parent coordinates):
        ``SPACE.capsule_bottom`` up, width min(880, area − 40)."""
        from gui.themes.tokens import SPACE

        width = max(320, min(self.MAX_WIDTH, area.width() - 40))
        x = area.x() + (area.width() - width) // 2
        y = area.bottom() + 1 - SPACE.capsule_bottom - self.HEIGHT
        geo = QtCore.QRect(x, y, width, self.HEIGHT)
        if self.geometry() != geo:
            self.setGeometry(geo)
        self.raise_()

    # ── Theme ─────────────────────────────────────────────────────────────

    def _apply_theme(self, tokens=None) -> None:
        # Colours come from the static stylesheet's roles and the painters;
        # nothing to recompute here but a repaint.
        self.update()

    # ── Keyboard mode ─────────────────────────────────────────────────────

    def _on_kb_toggled(self, checked: bool) -> None:
        self._animate_kb_press(checked)
        self.keyboard_mode_toggled.emit(checked)

    def _animate_kb_press(self, active: bool) -> None:
        """Short opacity pulse when toggling keyboard mode."""
        self._kb_anim.stop()
        self._kb_fx.setEnabled(True)
        self._kb_fx.setOpacity(1.0)
        self._kb_anim.start()

    def _on_kb_pulse_finished(self) -> None:
        self._kb_fx.setOpacity(1.0)
        self._kb_fx.setEnabled(False)

    # ── Appear ────────────────────────────────────────────────────────────

    def _appear(self) -> None:
        """Fade in with the shadow (the first track of a session)."""
        from gui.themes import motion

        if self._appear_fx is None:
            self._appear_fx = QtWidgets.QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(self._appear_fx)
        self._appear_fx.setEnabled(True)
        self._appear_fx.setOpacity(0.0)
        if self._shadow is not None:
            self._shadow.set_strength(0.0)
        self.setVisible(True)
        motion.animate_value(self, "appear", 0.0, 1.0, self._on_appear_value,
                             duration="base", easing="standard",
                             on_finished=self._on_appeared)

    def _on_appear_value(self, value: float) -> None:
        if self._appear_fx is not None:
            self._appear_fx.setOpacity(float(value))
        if self._shadow is not None:
            self._shadow.set_strength(float(value))

    def _on_appeared(self) -> None:
        if self._appear_fx is not None:
            self._appear_fx.setOpacity(1.0)
            self._appear_fx.setEnabled(False)   # nothing renders off-screen at rest
        if self._shadow is not None:
            self._shadow.set_strength(1.0)

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface

        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().capsule)
        finally:
            p.end()

    # ── Public slots (called by AlphaDEXWindow) ───────────────────────────

    @QtCore.Slot(str, str, str, object)
    def update_now_playing(
        self, title: str, artist: str, album: str, pm: object
    ) -> None:
        """Update track info and the art thumbnail (see the class docstring)."""
        is_null_pm = isinstance(pm, QtGui.QPixmap) and pm.isNull()
        if not title and is_null_pm:
            self.clear()
            return

        self._title_lbl.setText(title or "—")
        sub = "  ·  ".join(p for p in [artist, album] if p)
        self._artist_lbl.setText(sub)

        if isinstance(pm, QtGui.QPixmap) and not pm.isNull():
            self._art_lbl.setText("")
            self._art_lbl.setPixmap(_round_pixmap(pm, self._ART_SIZE))
        elif is_null_pm:
            self._reset_art()
        # else (None): art still loading — keep what is shown

        if not self.isVisible():
            self._appear()

    @QtCore.Slot()
    def clear(self) -> None:
        """Nothing is loaded any more: reset to the initial state and hide."""
        from gui.themes import motion

        self._title_lbl.setText("—")
        self._artist_lbl.setText("")
        self._reset_art()
        self.set_playing(False)
        self._duration_ms = 0
        self._set_seek_value(0)
        self._pos_lbl.setText("0:00")
        self._dur_lbl.setText("0:00")
        motion.stop(self, "appear")
        self._on_appeared()
        self.setVisible(False)

    def _reset_art(self) -> None:
        self._art_lbl.setPixmap(QtGui.QPixmap())
        self._art_lbl.setText("♪")

    @QtCore.Slot(bool)
    def set_playing(self, is_playing: bool) -> None:
        self._is_playing = is_playing
        self._play_btn.setText("⏸" if is_playing else "▶")
        self._play_btn.setAccessibleName("Pause" if is_playing else "Play")

    @QtCore.Slot(int, int)
    def update_position(self, pos_ms: int, dur_ms: int) -> None:
        self._duration_ms = dur_ms
        if not self._user_seeking and not self._seek_slider.isSliderDown():
            self._set_seek_value(int((pos_ms / dur_ms) * 1000) if dur_ms > 0 else 0)
            self._pos_lbl.setText(self._fmt(pos_ms))
        self._dur_lbl.setText(self._fmt(dur_ms))

    @QtCore.Slot(int)
    def set_volume(self, value: int) -> None:
        """Keep the bar's volume knob in sync with the player (no echo)."""
        was = self._vol_slider.blockSignals(True)
        self._vol_slider.setValue(max(0, min(100, int(value))))
        self._vol_slider.blockSignals(was)

    # ── Seek interaction ──────────────────────────────────────────────────

    def _set_seek_value(self, value: int) -> None:
        was = self._seek_slider.blockSignals(True)
        self._seek_slider.setValue(value)
        self._seek_slider.blockSignals(was)

    def _on_slider_pressed(self) -> None:
        self._user_seeking = True

    def _on_slider_released(self) -> None:
        self._user_seeking = False
        self._emit_seek(self._seek_slider.value())

    def _on_seek_value_changed(self, value: int) -> None:
        if self._user_seeking or self._seek_slider.isSliderDown():
            return  # the release emits the final position
        self._emit_seek(value)

    def _emit_seek(self, value: int) -> None:
        if self._duration_ms > 0:
            self.seek_requested.emit(int((value / 1000.0) * self._duration_ms))

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(ms: int) -> str:
        ms = max(0, ms)
        m, s = divmod(ms // 1000, 60)
        return f"{m}:{s:02d}"
