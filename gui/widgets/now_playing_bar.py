"""The floating Now Playing capsule — the app's only transport.

A glass capsule (the material's ``capsule`` surface: fill .56, border .18,
top highlight .30) floating at the bottom centre of the page area, 22 px up,
at most 880 px wide. The main window places it (it is an overlay, not in a
layout) and pages keep ``SPACE.page_bottom`` (104 px) of room under their
content so it never covers anything.

Contents, left to right: a 52 px round cover; the title (14/650) and
"artist · album" (12, muted); shuffle / previous / play-pause (a 46 px white
disc) / next / repeat; the position, a 5 px seek track with an accent fill
and a white knob, and the duration (11 px, tabular figures); volume; and the
keyboard-mode pill.

**Width tiers.** The capsule is ``min(880, page area − 40)`` wide, so at the
900 px minimum window it is ~620 px. It never overflows: when the full row
does not fit, the volume slider moves into a popup behind the speaker button
("compact"); when that is still too wide, the two times are dropped (the
seek bar's tooltip carries them) and the gaps tighten ("narrow"). Every
control stays reachable in every tier.

**When it shows.** Once something has played it stays on every page. Before
that it is hidden, except on the Player page, which *pins* it
(``set_pinned``): the capsule is the Player's transport, so that page always
shows it, in an idle state ("Nothing playing", or "Playback unavailable"
with the transport disabled). When a track loads it fades in (with its
shadow); when the queue is cleared it resets, and hides unless pinned.

Signals emitted (routed to PlayerWorkspace by AlphaDEXWindow):
    play_pause_requested
    next_requested
    prev_requested
    seek_requested(int)      — target position in ms (drag release, wheel,
                               keys, groove click)
    volume_changed(int)      — 0-100, from any user interaction
    shuffle_requested(bool)  — the shuffle toggle was clicked
    repeat_requested(str)    — the repeat button was clicked: the next mode
                               ("off" → "track" → "all" → "off", as the old page button)
    keyboard_mode_toggled(bool)  — arrow-key navigation mode

Slots: ``update_now_playing(title, artist, album, pm)`` (``pm`` None keeps
the current thumbnail while art loads; a *null* QPixmap means "no art"; an
empty title with a null pixmap means nothing is loaded any more),
``set_playing``, ``update_position``, ``set_volume``, ``set_shuffle``,
``set_repeat``, ``set_transport_state(available, can_play, reason)``,
``set_pinned``, ``clear``.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.manager import get_manager

#: Repeat modes, in the order the button cycles through them.
REPEAT_MODES = ("off", "track", "all")
_REPEAT_TIPS = {"off": "Repeat: Off", "all": "Repeat: All", "track": "Repeat: Track"}

_IDLE_TITLE = "Nothing playing"
_IDLE_META = "Double-click a track to play it"
_UNAVAILABLE_TITLE = "Playback unavailable"
_UNAVAILABLE_META = "VLC is needed to play music here"


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
    """The round cover; a glass disc with a music glyph when there is no art
    (``text()`` is then "♪", the state marker; the glyph is painted)."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            super().paintEvent(event)
            return
        from gui.themes.glass import current_material, qcolor
        from gui.widgets.glyphs import paint_glyph
        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            m = current_material()
            r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            p.setPen(QtGui.QPen(qcolor(m.raised.border or m.card.border), 1.0))
            p.setBrush(qcolor(m.raised.fill))
            p.drawEllipse(r)
            side = r.width() * 0.42
            paint_glyph(p, QtCore.QRectF(r.center().x() - side / 2, r.center().y() - side / 2,
                                         side, side), "music", qcolor(t.text_secondary))
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
            enabled = self.isEnabled()
            hover = self.underMouse() and enabled
            if self._disc:
                if not enabled:
                    p.setOpacity(0.4)
                fill = qcolor(t.text_primary, 0.92 if m.is_glass else 1.0)
                if self.isDown():
                    fill = fill.darker(115)
                elif hover:
                    fill = qcolor(t.text_primary, 1.0)
                p.setBrush(fill)
                p.setPen(QtGui.QPen(qcolor(m.capsule.highlight) if m.is_glass
                                    else QtGui.QColor(t.card_border), 1.0))
                p.drawEllipse(r)
                ink = QtGui.QColor(t.text_inverse)
            else:
                if self.isDown() or hover:
                    p.setPen(QtCore.Qt.PenStyle.NoPen)
                    p.setBrush(qcolor(m.pressed if self.isDown() else m.hover))
                    p.drawEllipse(r)
                ink = QtGui.QColor(t.text_primary if enabled else t.text_muted)
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


class _ModeButton(QtWidgets.QPushButton):
    """Shuffle / repeat: a 34 px round glyph button. Off: a muted glyph.
    On: the glyph in the accent with a small accent dot under it (so the
    state does not rest on colour alone); ``badge`` ("1" for Repeat: Track)
    is drawn at the glyph's top right."""

    SIZE = 34

    def __init__(self, glyph: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._glyph = glyph
        self._active = False
        self._badge = ""
        self.setFlat(True)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    def is_active(self) -> bool:
        return self._active

    def badge(self) -> str:
        return self._badge

    def set_mode(self, active: bool, badge: str = "") -> None:
        if (active, badge) != (self._active, self._badge):
            self._active, self._badge = active, badge
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, font, qcolor
        from gui.widgets.glyphs import paint_glyph

        t = get_manager().current
        m = current_material()
        active = self._active or self.isChecked()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            enabled = self.isEnabled()
            if enabled and (self.isDown() or self.underMouse()):
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(qcolor(m.pressed if self.isDown() else m.hover))
                p.drawEllipse(r)
            if not enabled:
                ink = qcolor(t.text_muted)
            elif active:
                ink = qcolor(t.accent)
            else:
                ink = qcolor(t.text_secondary)
            g = 17.0
            c = r.center()
            paint_glyph(p, QtCore.QRectF(c.x() - g / 2, c.y() - g / 2 - 1, g, g), self._glyph,
                        ink, weight=1.9)
            if active and enabled:
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(ink)
                p.drawEllipse(QtCore.QPointF(c.x(), r.bottom() - 4.5), 1.8, 1.8)
            if self._badge:
                f = font("micro", px=9, tracking=0.0, weight=800)
                p.setFont(f)
                p.setPen(ink)
                p.drawText(QtCore.QRectF(c.x() - 4, c.y() - 4.5, 8, 9),
                           int(QtCore.Qt.AlignmentFlag.AlignCenter), self._badge)
        finally:
            p.end()


class _VolumeButton(QtWidgets.QPushButton):
    """The speaker glyph. In the full tier it only labels the slider (it
    ignores the mouse); in the compact tiers it opens the volume popup."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFlat(True)
        self.setFixedSize(26, 30)
        self.setAccessibleName("Volume")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.set_interactive(False)

    def set_interactive(self, on: bool) -> None:
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, not on)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus if on
                            else QtCore.Qt.FocusPolicy.NoFocus)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor if on
                                     else QtCore.Qt.CursorShape.ArrowCursor))
        self.update()

    def is_interactive(self) -> bool:
        return not self.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, qcolor

        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(self.rect())
            if self.is_interactive() and self.isEnabled() and (self.underMouse() or self.isDown()):
                m = current_material()
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(qcolor(m.pressed if self.isDown() else m.hover))
                p.drawRoundedRect(r.adjusted(0.5, 2.5, -0.5, -2.5), 12, 12)
            ink = QtGui.QColor(t.text_secondary if self.isEnabled() else t.text_muted)
            k = 18.0 / 24.0
            ox, oy = r.center().x() - 12 * k, r.center().y() - 12 * k
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(ink)
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(ox + x * k, oy + y * k) for x, y in (
                (3, 9), (7, 9), (12, 5), (12, 19), (7, 15), (3, 15))]))
            pen = QtGui.QPen(ink, 1.6, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawArc(QtCore.QRectF(ox + 11 * k, oy + 8 * k, 6 * k, 8 * k), -60 * 16, 120 * 16)
            p.drawArc(QtCore.QRectF(ox + 11 * k, oy + 5 * k, 10 * k, 14 * k), -60 * 16, 120 * 16)
        finally:
            p.end()


class _VolumePopup(QtWidgets.QFrame):
    """The volume slider when the capsule is too narrow to show it inline:
    a small opaque pill above the speaker (a popup window, so it is opaque:
    nothing behind it is the app's own glass)."""

    WIDTH, HEIGHT = 180, 44

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Popup
                         | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setObjectName("capsuleVolumePopup")
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(18, 0, 18, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setObjectName("capsuleVolume")
        self.slider.setRange(0, 100)
        self.slider.setToolTip("Volume")
        self.slider.setAccessibleName("Volume")
        lay.addWidget(self.slider)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_surface
        from gui.themes.tokens import PILL, Surface

        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()),
                          Surface(t.card_bg, t.card_border, radius=PILL))
        finally:
            p.end()


class _KbToggle(QtWidgets.QPushButton):
    """Keyboard-mode pill: raised glass at rest, the accent when on. The
    keyboard glyph is painted (``text()`` "⌨" stays as its name)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("⌨", parent)
        self.setCheckable(True)
        self.setFlat(True)
        self.setAccessibleName("Keyboard mode")
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface, qcolor
        from gui.themes.tokens import PILL, Surface
        from gui.widgets.glyphs import paint_glyph

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
            g = 18.0
            paint_glyph(p, QtCore.QRectF(r.center().x() - g / 2, r.center().y() - g / 2, g, g),
                        "keyboard", ink, weight=1.6)
        finally:
            p.end()


class NowPlayingBar(QtWidgets.QWidget):
    """The floating Now Playing capsule (see the module docstring)."""

    play_pause_requested = Signal()
    next_requested       = Signal()
    prev_requested       = Signal()
    seek_requested       = Signal(int)   # ms
    volume_changed       = Signal(int)   # 0-100
    shuffle_requested    = Signal(bool)
    repeat_requested     = Signal(str)   # the mode to switch to
    keyboard_mode_toggled = Signal(bool)

    _ART_SIZE = 52     # px, round cover
    HEIGHT = 72        # 52 + 2 × 10 padding
    MAX_WIDTH = 880
    _KB_PULSE_MS = 120
    #: Width tiers, widest first (see the module docstring).
    TIERS = ("full", "compact", "narrow")

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setObjectName("nowPlayingBar")
        self.setAutoFillBackground(False)

        self._is_playing  = False
        self._duration_ms = 0
        self._position_ms = 0
        self._user_seeking = False
        self._loaded = False         # a track is loaded (title shown)
        self._pinned = False         # shown while idle (the Player page)
        self._available = True
        self._can_play = True
        self._reason = ""
        self._repeat = "off"
        self._tier = ""
        self._shadow = None        # ShadowLayer, once placed in a window
        self._appear_fx: QtWidgets.QGraphicsOpacityEffect | None = None
        self._vol_popup: _VolumePopup | None = None

        self._build_ui()
        get_manager().theme_changed.connect(self._apply_theme)
        self._apply_theme(get_manager().current)
        self._show_idle_text()
        self.setVisible(False)   # hidden until a track loads or a page pins it

    # ── Construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from gui.themes.glass import font
        from gui.widgets.glass import ElidedLabel

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 16, 10)
        root.setSpacing(14)
        # The main window places the capsule (place()); its row must never
        # widen it: the width tiers make the row fit instead.
        root.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        self._root = root

        # Round cover
        self._art_lbl = _ArtLabel()
        self._art_lbl.setObjectName("npbArt")
        self._art_lbl.setFixedSize(self._ART_SIZE, self._ART_SIZE)
        self._art_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._art_lbl.setText("♪")
        root.addWidget(self._art_lbl)

        # Title + "artist · album"
        self._meta = QtWidgets.QWidget()
        self._meta.setMinimumWidth(140)
        self._meta.setMaximumWidth(220)
        meta_col = QtWidgets.QVBoxLayout(self._meta)
        meta_col.setContentsMargins(0, 0, 0, 0)
        meta_col.setSpacing(1)
        self._title_lbl = ElidedLabel("—")
        self._title_lbl.setObjectName("capsuleTitle")
        self._title_lbl.setFont(font("body", weight=650))
        self._title_lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
        self._artist_lbl = ElidedLabel("")
        self._artist_lbl.setObjectName("capsuleMeta")
        self._artist_lbl.setFont(font("caption"))
        meta_col.addStretch(1)
        meta_col.addWidget(self._title_lbl)
        meta_col.addWidget(self._artist_lbl)
        meta_col.addStretch(1)
        # Like the mockup (flex 0 1 200px, min 140): the text keeps its width
        # up to 220 px and the progress track takes what is left.
        root.addWidget(self._meta, 0)

        # Transport: shuffle · prev · play/pause · next · repeat
        self._shuffle_btn = _ModeButton("shuffle")
        self._shuffle_btn.setCheckable(True)
        self._shuffle_btn.setToolTip("Shuffle: Off")
        self._shuffle_btn.setAccessibleName("Shuffle")
        self._shuffle_btn.toggled.connect(self._on_shuffle_clicked)
        self._prev_btn = _TransportButton("⏮", "Previous track")
        self._prev_btn.clicked.connect(self.prev_requested)
        self._play_btn = _TransportButton("▶", "Play / Pause  [Space]", disc=True)
        self._play_btn.clicked.connect(self.play_pause_requested)
        self._next_btn = _TransportButton("⏭", "Next track")
        self._next_btn.clicked.connect(self.next_requested)
        self._repeat_btn = _ModeButton("repeat")
        self._repeat_btn.setToolTip(_REPEAT_TIPS["off"])
        self._repeat_btn.setAccessibleName(_REPEAT_TIPS["off"])
        self._repeat_btn.clicked.connect(self._on_repeat_clicked)
        transport = QtWidgets.QHBoxLayout()
        transport.setSpacing(4)
        for b in (self._shuffle_btn, self._prev_btn, self._play_btn, self._next_btn,
                  self._repeat_btn):
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
        self._seek_slider.setAccessibleName("Seek")
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
        self._progress = progress
        root.addLayout(progress, 3)

        # Volume
        self._vol_icon = _VolumeButton()
        self._vol_icon.clicked.connect(self._open_volume_popup)
        self._vol_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._vol_slider.setObjectName("capsuleVolume")
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setFixedWidth(72)
        self._vol_slider.setToolTip("Volume")
        self._vol_slider.setAccessibleName("Volume")
        # Any user change (drag, wheel, keys, groove click) is emitted;
        # set_volume() blocks signals, so syncing from the page can't echo.
        self._vol_slider.valueChanged.connect(self.volume_changed)
        self._vol_slider.valueChanged.connect(self._sync_popup_slider)
        volume = QtWidgets.QHBoxLayout()
        volume.setSpacing(4)
        volume.addWidget(self._vol_icon)
        volume.addWidget(self._vol_slider)
        root.addLayout(volume)

        # ── Keyboard mode toggle (pill button) ────────────────────────────
        self._kb_btn = _KbToggle()
        self._kb_btn.setFixedSize(40, 30)
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

    # ── Width tiers ───────────────────────────────────────────────────────

    def tier(self) -> str:
        """The width tier in use: "full", "compact" or "narrow"."""
        return self._tier

    def _set_tier(self, tier: str) -> None:
        full, narrow = tier == "full", tier == "narrow"
        self._vol_slider.setVisible(full)
        self._vol_icon.set_interactive(not full)
        self._vol_icon.setToolTip("Volume" if full else "Volume…")
        self._pos_lbl.setVisible(not narrow)
        self._dur_lbl.setVisible(not narrow)
        self._root.setSpacing(8 if narrow else 14)
        self._meta.setMinimumWidth({"full": 140, "compact": 120}.get(tier, 92))
        self._meta.setMaximumWidth({"full": 220, "compact": 180}.get(tier, 130))
        self._seek_slider.setMinimumWidth({"full": 120, "compact": 100}.get(tier, 56))
        self._tier = tier
        self._update_seek_tip()

    def _apply_density(self) -> None:
        """Pick the widest tier whose row fits the capsule's width."""
        lay = self.layout()
        for tier in self.TIERS:
            self._set_tier(tier)
            self._invalidate(lay)
            if lay.minimumSize().width() <= self.width():
                break
        if self._tier == "full" and self._vol_popup is not None:
            self._vol_popup.hide()

    @classmethod
    def _invalidate(cls, lay: QtWidgets.QLayout) -> None:
        """Drop the cached sizes of ``lay`` and its nested rows (hiding a
        widget only invalidates its parent widget's top layout)."""
        for i in range(lay.count()):
            sub = lay.itemAt(i).layout()
            if sub is not None:
                cls._invalidate(sub)
        lay.invalidate()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_density()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802
        """Hidden (by us, or with its window): a fade-in still running ends
        at its rest state at once, so no animation outlives the capsule."""
        from gui.themes import motion

        motion.stop(self, "appear")
        self._on_appeared()
        super().hideEvent(event)

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

    # ── Shuffle / repeat ──────────────────────────────────────────────────

    def _on_shuffle_clicked(self, checked: bool) -> None:
        self._show_shuffle(checked)
        self.shuffle_requested.emit(checked)

    def _on_repeat_clicked(self) -> None:
        nxt = REPEAT_MODES[(REPEAT_MODES.index(self._repeat) + 1) % len(REPEAT_MODES)]
        self.set_repeat(nxt)
        self.repeat_requested.emit(nxt)

    def _show_shuffle(self, on: bool) -> None:
        tip = "Shuffle: On" if on else "Shuffle: Off"
        self._shuffle_btn.setToolTip(tip)
        self._shuffle_btn.set_mode(on)

    @QtCore.Slot(bool)
    def set_shuffle(self, on: bool) -> None:
        """Show the player's shuffle state (no echo)."""
        was = self._shuffle_btn.blockSignals(True)
        self._shuffle_btn.setChecked(bool(on))
        self._shuffle_btn.blockSignals(was)
        self._show_shuffle(bool(on))

    def shuffle(self) -> bool:
        return self._shuffle_btn.isChecked()

    @QtCore.Slot(str)
    def set_repeat(self, mode: str) -> None:
        """Show the player's repeat mode: "off", "all" or "track"."""
        if mode not in REPEAT_MODES:
            mode = "off"
        self._repeat = mode
        tip = _REPEAT_TIPS[mode]
        self._repeat_btn.setToolTip(tip)
        self._repeat_btn.setAccessibleName(tip)
        self._repeat_btn.set_mode(mode != "off", "1" if mode == "track" else "")

    def repeat(self) -> str:
        return self._repeat

    # ── Volume popup (compact tiers) ──────────────────────────────────────

    def _open_volume_popup(self) -> None:
        if not self._vol_icon.is_interactive():
            return
        if self._vol_popup is None:
            self._vol_popup = _VolumePopup(self)
            self._vol_popup.slider.valueChanged.connect(self._vol_slider.setValue)
        pop = self._vol_popup
        was = pop.slider.blockSignals(True)
        pop.slider.setValue(self._vol_slider.value())
        pop.slider.blockSignals(was)
        anchor = self._vol_icon.mapToGlobal(QtCore.QPoint(self._vol_icon.width() // 2, 0))
        pop.move(anchor.x() - pop.width() // 2, anchor.y() - pop.height() - 12)
        pop.show()
        pop.slider.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def _sync_popup_slider(self, value: int) -> None:
        if self._vol_popup is not None:
            was = self._vol_popup.slider.blockSignals(True)
            self._vol_popup.slider.setValue(value)
            self._vol_popup.slider.blockSignals(was)

    # ── Appear ────────────────────────────────────────────────────────────

    def _wanted(self) -> bool:
        return self._loaded or self._pinned

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

    def _hide_now(self) -> None:
        from gui.themes import motion

        motion.stop(self, "appear")
        self._on_appeared()
        if self._vol_popup is not None:
            self._vol_popup.hide()
        self.setVisible(False)

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

        self._loaded = True
        self._title_lbl.setText(title or "—")
        sub = "  ·  ".join(p for p in [artist, album] if p)
        self._artist_lbl.setText(sub)
        self._title_lbl.setToolTip(" · ".join(p for p in [title, artist, album] if p))

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
        """Nothing is loaded any more: reset to the idle state, and hide
        unless a page pins the capsule."""
        self._loaded = False
        self._reset_art()
        self.set_playing(False)
        self._duration_ms = 0
        self._position_ms = 0
        self._set_seek_value(0)
        self._pos_lbl.setText("0:00")
        self._dur_lbl.setText("0:00")
        self._show_idle_text()
        self._update_seek_tip()
        if not self._wanted():
            self._hide_now()

    @QtCore.Slot(bool)
    def set_pinned(self, pinned: bool) -> None:
        """Show the capsule even with nothing loaded (the Player page is on
        screen: the capsule is its transport). Unpinned and idle, it hides."""
        pinned = bool(pinned)
        if pinned == self._pinned:
            return
        self._pinned = pinned
        if self._wanted():
            if not self.isVisible():
                # Pinned by a page switch: the page itself fades in, so the
                # capsule simply comes with it (a track arriving fades it in).
                self._on_appeared()
                self.setVisible(True)
        elif self.isVisible():
            self._hide_now()

    def is_pinned(self) -> bool:
        return self._pinned

    def is_loaded(self) -> bool:
        return self._loaded

    @QtCore.Slot(bool, bool, str)
    def set_transport_state(self, available: bool, can_play: bool, reason: str = "") -> None:
        """Enable the controls the player can act on. ``available`` False:
        no playback backend (``reason`` says why; it goes in the tooltips and
        the idle text). ``can_play`` False: nothing to play yet."""
        self._available, self._can_play, self._reason = bool(available), bool(can_play), reason
        for w in (self._prev_btn, self._next_btn, self._seek_slider, self._shuffle_btn,
                  self._repeat_btn, self._vol_icon, self._vol_slider):
            w.setEnabled(self._available)
        self._play_btn.setEnabled(self._available and self._can_play)
        if self._available:
            self._prev_btn.setToolTip("Previous track  [P]")
            self._next_btn.setToolTip("Next track  [N]")
            self._play_btn.setToolTip("Play / Pause  [Space]")
            self._update_seek_tip()
        else:
            tip = f"Playback unavailable — {reason or 'no playback backend'}"
            for w in (self._prev_btn, self._next_btn, self._play_btn, self._seek_slider):
                w.setToolTip(tip)
        if not self._loaded:
            self._show_idle_text()

    def _show_idle_text(self) -> None:
        if self._available:
            self._title_lbl.setText(_IDLE_TITLE)
            self._artist_lbl.setText(_IDLE_META)
            self._title_lbl.setToolTip("")
        else:
            self._title_lbl.setText(_UNAVAILABLE_TITLE)
            self._artist_lbl.setText(_UNAVAILABLE_META)
            self._title_lbl.setToolTip(self._reason)

    @QtCore.Slot(bool)
    def set_playing(self, is_playing: bool) -> None:
        self._is_playing = is_playing
        self._play_btn.setText("⏸" if is_playing else "▶")
        self._play_btn.setAccessibleName("Pause" if is_playing else "Play")
        self._play_btn.update()

    @QtCore.Slot(int, int)
    def update_position(self, pos_ms: int, dur_ms: int) -> None:
        self._duration_ms = dur_ms
        if not self._user_seeking and not self._seek_slider.isSliderDown():
            self._position_ms = pos_ms
            self._set_seek_value(int((pos_ms / dur_ms) * 1000) if dur_ms > 0 else 0)
            self._pos_lbl.setText(self._fmt(pos_ms))
        self._dur_lbl.setText(self._fmt(dur_ms))
        self._update_seek_tip()

    @QtCore.Slot(int)
    def set_volume(self, value: int) -> None:
        """Keep the bar's volume knob in sync with the player (no echo)."""
        value = max(0, min(100, int(value)))
        was = self._vol_slider.blockSignals(True)
        self._vol_slider.setValue(value)
        self._vol_slider.blockSignals(was)
        self._sync_popup_slider(value)

    def _reset_art(self) -> None:
        self._art_lbl.setPixmap(QtGui.QPixmap())
        self._art_lbl.setText("♪")

    # ── Seek interaction ──────────────────────────────────────────────────

    def _update_seek_tip(self) -> None:
        """In the narrow tier the times are hidden; the seek bar says them."""
        if not self._available:
            return
        if self._tier == "narrow" and self._duration_ms > 0:
            tip = f"{self._fmt(self._position_ms)} / {self._fmt(self._duration_ms)}"
        else:
            tip = "Seek"
        if self._seek_slider.toolTip() != tip:
            self._seek_slider.setToolTip(tip)

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
            if self._duration_ms > 0:   # live time readout while dragging
                self._pos_lbl.setText(self._fmt(int(value / 1000.0 * self._duration_ms)))
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
