"""AlphaDEX — Mosaic Reveal landing page (circular carousel edition).

Shown once after the splash screen fades.

Layout
------
18 album-art tiles are arranged on a rotating ellipse centred in the window.
A frosted-glass CTA card floats in the centre hole.

Depth / coverflow effect
------------------------
Each tile's size and opacity are driven by its sin(angle) value:
  • sin ≈ +1  (bottom of ellipse, "front") → full size & opacity
  • sin ≈ −1  (top of ellipse, "back")     → 62 % size, 38 % opacity
Tiles are re-stacked every tick so front tiles always paint on top.

Sequence
--------
1. ``show_animated()`` — fade in, then staggered tile fly-in from edges.
2. Fly-in completes → rotation timer starts (eases in over ~1.5 s).
3. User clicks "Open Library" → QFileDialog → ``_accept(path)``.
   (Or "Continue" if a saved library is already in config.)
4. Rotation timer stops; tiles scatter outward from current positions.
5. ``library_selected(path)`` emitted at start of fade-out so caller can
   cross-fade the main window in simultaneously.
6. ``finished`` emitted after the landing window is hidden.

Album art (proof-of-concept)
-----------------------------
If a saved library path exists, ``_quick_scan_art`` does a 2-level-deep
walk (artist/album) looking for preferred image names (folder.jpg,
cover.jpg, etc.).  Found images are pre-scaled and painted into tiles.
Tiles without art fall back to the colourful gradient.
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal

# ── Circle / ellipse layout ───────────────────────────────────────────────────
_N_TILES    = 18       # number of tiles on the ring
_TILE_SZ    = 108      # base (front/maximum) tile size in px
_RX_FRAC    = 0.400    # ellipse x-radius as fraction of window width
_RY_FRAC    = 0.355    # ellipse y-radius as fraction of window height
_MIN_SCALE  = 0.62     # tile scale at the back of the circle (depth = 0)
_MAX_SCALE  = 1.00     # tile scale at the front             (depth = 1)
_MIN_ALPHA  = 0.38     # tile opacity at the back

# Rotation
_ROT_SPEED   = 0.0038   # radians / timer-tick (≈ 60 fps → ~13°/s, lap ~28 s)
_ROT_RAMP    = 95       # ticks to ease rotation from 0 → full speed (~1.5 s)
_DEPTH_RAMP  = 160      # ticks to ease depth from 1.0 → true value   (~2.5 s)
                        # Longer than _ROT_RAMP so tiles finish morphing
                        # gently after the ring is already spinning.

# CTA card geometry (centred in the window; values are fractions of window dims)
_CARD_W     = 340
_CARD_H     = 352

# ── Timing ────────────────────────────────────────────────────────────────────
_STAGGER_MS  = 38
_FLY_IN_MS   = 500
_FLY_OUT_MS  = 380
_SCATTER_MAX = 160
_FADE_IN_MS  = 320

# Exported — alpha_dex_gui.py uses this to match the cross-fade duration.
FADE_OUT_MS  = 420

# ── Colour pool ───────────────────────────────────────────────────────────────
_GRADS: list[tuple[str, str]] = [
    ("#6366f1", "#a78bfa"),
    ("#0ea5e9", "#38bdf8"),
    ("#10b981", "#34d399"),
    ("#f59e0b", "#fbbf24"),
    ("#ef4444", "#f87171"),
    ("#8b5cf6", "#c4b5fd"),
    ("#06b6d4", "#67e8f9"),
    ("#ec4899", "#f9a8d4"),
    ("#84cc16", "#a3e635"),
    ("#f97316", "#fb923c"),
    ("#14b8a6", "#2dd4bf"),
    ("#a855f7", "#d8b4fe"),
    ("#3b82f6", "#93c5fd"),
    ("#d946ef", "#f0abfc"),
    ("#22c55e", "#86efac"),
    ("#eab308", "#fde047"),
    ("#64748b", "#94a3b8"),
    ("#1e40af", "#60a5fa"),
]

# Preferred album-art filenames (lower-case), checked first during scan.
_ART_NAMES = frozenset({
    "folder.jpg", "folder.png", "cover.jpg", "cover.png",
    "album.jpg",  "album.png",  "front.jpg", "front.png",
    "artwork.jpg", "artwork.png",
})
_ART_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _darken(hex_color: str, pct: int) -> str:
    c = QtGui.QColor(hex_color)
    h, s, v, a = c.getHsvF()
    c.setHsvF(h, s, max(0.0, v - pct / 100.0), a)
    return c.name()


def _quick_scan_art(library_path: str, n: int) -> list[QtGui.QPixmap]:
    """Walk up to 2 levels deep (artist/album) collecting cover images.

    Returns a list of QPixmaps pre-scaled to *_TILE_SZ × _TILE_SZ*, stopping
    once *n* images have been found.  Fast and non-blocking for typical
    library structures.
    """
    found: list[QtGui.QPixmap] = []
    if not library_path:
        return found
    try:
        for artist in os.scandir(library_path):
            if not artist.is_dir() or artist.name.startswith("."):
                continue
            try:
                for album in os.scandir(artist.path):
                    if not album.is_dir() or album.name.startswith("."):
                        continue
                    try:
                        # Prefer exact names; fall back to any image in dir.
                        candidates: list[str] = []
                        others: list[str]     = []
                        for f in os.scandir(album.path):
                            lname = f.name.lower()
                            ext   = Path(f.name).suffix.lower()
                            if lname in _ART_NAMES:
                                candidates.insert(0, f.path)
                            elif ext in _ART_EXTS:
                                others.append(f.path)
                        pick = (candidates or others)
                        if pick:
                            pm = QtGui.QPixmap(pick[0])
                            if not pm.isNull():
                                scaled = pm.scaled(
                                    _TILE_SZ, _TILE_SZ,
                                    QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                    QtCore.Qt.TransformationMode.SmoothTransformation,
                                )
                                found.append(scaled)
                                if len(found) >= n:
                                    return found
                    except OSError:
                        pass
            except OSError:
                pass
    except OSError:
        pass
    return found


# ─────────────────────────────────────────────────────────────────────────────
# _Tile
# ─────────────────────────────────────────────────────────────────────────────

class _Tile(QtWidgets.QWidget):
    """Album-art tile: gradient fallback or real cover image.

    ``_target`` is the initial (rotation = 0) ellipse position used for the
    fly-in animation.  Once the rotation timer starts the timer owns the
    widget's position.

    ``set_depth(depth)`` is called every timer tick to update size and opacity.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        grad: tuple[str, str],
        target: QtCore.QPoint,
        pixmap: QtGui.QPixmap | None = None,
    ) -> None:
        super().__init__(parent)
        self._grad    = grad
        self._target  = target        # fly-in destination on the ellipse
        self._pixmap  = pixmap
        self._opacity = 1.0
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.resize(_TILE_SZ, _TILE_SZ)

    # ── Called by rotation timer ───────────────────────────────────────────

    def set_depth(self, depth: float) -> None:
        """Update size and opacity to reflect how far the tile is from the front."""
        self._opacity = _MIN_ALPHA + (1.0 - _MIN_ALPHA) * depth
        new_sz = max(
            4,
            int(_TILE_SZ * (_MIN_SCALE + (_MAX_SCALE - _MIN_SCALE) * depth)),
        )
        if self.width() != new_sz:
            self.resize(new_sz, new_sz)
        self.update()

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        p.setOpacity(self._opacity)

        r = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, 10, 10)

        if self._pixmap and not self._pixmap.isNull():
            # ── Album art ─────────────────────────────────────────────────
            p.setClipPath(path)
            # drawPixmap scales from the pre-sized source to the current rect.
            p.drawPixmap(self.rect(), self._pixmap)
            p.setClipping(False)

            # Subtle vignette over the art
            vig = QtGui.QRadialGradient(r.center(), max(r.width(), r.height()) * 0.7)
            vig.setColorAt(0.55, QtGui.QColor(0, 0, 0, 0))
            vig.setColorAt(1.00, QtGui.QColor(0, 0, 0, 70))
            p.fillPath(path, QtGui.QBrush(vig))
        else:
            # ── Gradient placeholder ───────────────────────────────────────
            grad = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
            grad.setColorAt(0.0, QtGui.QColor(self._grad[0]))
            grad.setColorAt(1.0, QtGui.QColor(self._grad[1]))
            p.fillPath(path, QtGui.QBrush(grad))

            # Inner shadow at bottom-right
            shadow = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
            shadow.setColorAt(0.55, QtGui.QColor(0, 0, 0, 0))
            shadow.setColorAt(1.00, QtGui.QColor(0, 0, 0, 55))
            p.fillPath(path, QtGui.QBrush(shadow))

        # Top-edge sheen (both art and gradient tiles)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 42), 1.0))
        p.drawLine(
            QtCore.QPointF(r.left() + 12, r.top() + 1.0),
            QtCore.QPointF(r.right() - 12, r.top() + 1.0),
        )
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# _CTACard
# ─────────────────────────────────────────────────────────────────────────────

class _CTACard(QtWidgets.QFrame):
    """Frosted-glass centre card: app name, tagline, and library buttons."""

    open_clicked  = Signal()
    reuse_clicked = Signal()

    def __init__(self, parent: QtWidgets.QWidget, saved_path: str = "") -> None:
        super().__init__(parent)
        self._saved = saved_path
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self._build()

    def _build(self) -> None:
        try:
            from gui.themes.manager import get_manager
            t            = get_manager().current
            accent       = t.accent
            text_primary = t.text_primary
            text_muted   = t.text_secondary
        except Exception:
            accent, text_primary, text_muted = "#6366f1", "#f8fafc", "#94a3b8"

        try:
            from gui.fonts.loader import UI_FAMILY
        except ImportError:
            UI_FAMILY = "Arial"

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(44, 48, 44, 48)
        lay.setSpacing(0)
        lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Decorative accent bar
        bar = QtWidgets.QFrame()
        bar.setFixedSize(36, 4)
        bar.setStyleSheet(f"background: {accent}; border-radius: 2px; border: none;")
        bar_wrap = QtWidgets.QHBoxLayout()
        bar_wrap.setContentsMargins(0, 0, 0, 0)
        bar_wrap.addStretch()
        bar_wrap.addWidget(bar)
        bar_wrap.addStretch()
        lay.addLayout(bar_wrap)
        lay.addSpacing(18)

        # App name
        name_lbl = QtWidgets.QLabel("AlphaDEX")
        nf = QtGui.QFont(UI_FAMILY, 34)
        nf.setWeight(QtGui.QFont.Weight.Bold)
        nf.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
        name_lbl.setFont(nf)
        name_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet(
            f"color: {text_primary}; background: transparent; letter-spacing: -1px;"
        )

        # Tagline
        tag_lbl = QtWidgets.QLabel("Your library, organized.")
        tf = QtGui.QFont(UI_FAMILY, 12)
        tag_lbl.setFont(tf)
        tag_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        tag_lbl.setStyleSheet(
            f"color: {text_muted}; background: transparent; letter-spacing: 0.3px;"
        )

        lay.addWidget(name_lbl)
        lay.addSpacing(6)
        lay.addWidget(tag_lbl)
        lay.addSpacing(32)

        # "Continue" button (returning user)
        if self._saved:
            short = Path(self._saved).name or self._saved
            reuse_btn = QtWidgets.QPushButton(f"Continue  ·  {short}")
            reuse_btn.setFixedHeight(38)
            reuse_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            reuse_btn.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(255,255,255,0.07);
                    color: {text_muted};
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 8px;
                    font-family: '{UI_FAMILY}';
                    font-size: 12px;
                    padding: 0 18px;
                }}
                QPushButton:hover {{
                    background: rgba(255,255,255,0.13);
                    color: {text_primary};
                    border-color: rgba(255,255,255,0.18);
                }}
            """)
            reuse_btn.clicked.connect(self.reuse_clicked.emit)
            lay.addWidget(reuse_btn)
            lay.addSpacing(8)

        # "Open Library" primary CTA
        open_btn = QtWidgets.QPushButton("Open Library  →")
        open_btn.setFixedHeight(46)
        open_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        open_btn.setStyleSheet(f"""
            QPushButton {{
                background: {accent};
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-family: '{UI_FAMILY}';
                font-size: 14px;
                font-weight: 600;
                padding: 0 28px;
                letter-spacing: 0.3px;
            }}
            QPushButton:hover   {{ background: {_darken(accent, 12)}; }}
            QPushButton:pressed {{ background: {_darken(accent, 26)}; }}
        """)
        open_btn.clicked.connect(self.open_clicked.emit)
        lay.addWidget(open_btn)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = QtCore.QRectF(self.rect())

        path = QtGui.QPainterPath()
        path.addRoundedRect(r, 20, 20)

        # Frosted dark fill
        p.fillPath(path, QtGui.QBrush(QtGui.QColor(10, 13, 20, 218)))

        # Hairline border
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 22), 1.0))
        p.drawPath(path)

        # Top-edge inner glow
        glow = QtGui.QLinearGradient(
            QtCore.QPointF(r.left(), r.top()),
            QtCore.QPointF(r.left(), r.top() + 60),
        )
        glow.setColorAt(0.0, QtGui.QColor(255, 255, 255, 14))
        glow.setColorAt(1.0, QtGui.QColor(255, 255, 255,  0))
        p.fillPath(path, QtGui.QBrush(glow))
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# MosaicLanding
# ─────────────────────────────────────────────────────────────────────────────

class MosaicLanding(QtWidgets.QWidget):
    """Full-window Mosaic Reveal landing with a coverflow-style rotating ellipse.

    Usage::

        landing = MosaicLanding(geometry, saved_path)
        landing.library_selected.connect(_on_library_selected)
        landing.finished.connect(_on_done)
        landing.show_animated()
    """

    library_selected = Signal(str)   # emitted at start of fade-out
    finished         = Signal()      # emitted after window is hidden

    def __init__(
        self,
        geometry: QtCore.QRect,
        saved_path: str = "",
    ) -> None:
        super().__init__(None, QtCore.Qt.WindowType.FramelessWindowHint)
        self.setGeometry(geometry)
        self._w     = geometry.width()
        self._h     = geometry.height()
        self._saved = saved_path

        self._pending   = ""
        self._tiles: list[_Tile] = []
        self._base_angles: list[float] = []   # initial ellipse angles per tile
        self._rotation  = 0.0                 # current rotation offset (rad)
        self._tick_count = 0                  # for ease-in ramp

        self._fly_in_grp:  QtCore.QParallelAnimationGroup | None = None
        self._fly_out_grp: QtCore.QParallelAnimationGroup | None = None
        self._rot_timer:   QtCore.QTimer | None = None
        # Keep animation objects alive
        self._fade_in_anim:  object = None
        self._fade_out_anim: object = None

        self._compute_circle()
        self._build_tiles()
        self._build_cta()

    # ── Ellipse / card geometry ────────────────────────────────────────────

    def _compute_circle(self) -> None:
        self._cx  = self._w / 2.0
        self._cy  = self._h / 2.0
        self._rx  = self._w * _RX_FRAC
        self._ry  = self._h * _RY_FRAC

        # CTA card centred in the window
        cw, ch = _CARD_W, _CARD_H
        self._center_rect = QtCore.QRect(
            int(self._cx - cw / 2),
            int(self._cy - ch / 2),
            cw, ch,
        )

        # Evenly distribute base angles; start at top (−π/2) so the first tile
        # comes in from directly above the window.
        step = 2 * math.pi / _N_TILES
        self._base_angles = [
            -math.pi / 2 + step * i for i in range(_N_TILES)
        ]

    def _ellipse_pos(self, angle: float, sz: int) -> QtCore.QPoint:
        """Top-left corner for a tile of *sz* px at *angle* on the ellipse."""
        return QtCore.QPoint(
            int(self._cx + self._rx * math.cos(angle) - sz / 2),
            int(self._cy + self._ry * math.sin(angle) - sz / 2),
        )

    def _off_screen(self, target: QtCore.QPoint) -> QtCore.QPoint:
        """Compute an off-screen start position for a tile (radially outward)."""
        tcx = target.x() + _TILE_SZ * 0.5
        tcy = target.y() + _TILE_SZ * 0.5
        dx, dy = tcx - self._cx, tcy - self._cy
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            dx, dy = 1.0, 0.0
        sx = (self._cx + _TILE_SZ) / (abs(dx) + 0.001)
        sy = (self._cy + _TILE_SZ) / (abs(dy) + 0.001)
        scale = min(sx, sy) * 1.3
        return QtCore.QPoint(
            int(self._cx + dx * scale - _TILE_SZ * 0.5),
            int(self._cy + dy * scale - _TILE_SZ * 0.5),
        )

    def _scatter_target(self, current: QtCore.QPoint, sz: int) -> QtCore.QPoint:
        """Random off-screen scatter destination from the tile's current position."""
        tcx = current.x() + sz * 0.5
        tcy = current.y() + sz * 0.5
        dx = (tcx - self._cx) + random.uniform(-90, 90)
        dy = (tcy - self._cy) + random.uniform(-90, 90)
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            dx, dy = 60.0, 60.0
        sx = (self._cx + _TILE_SZ * 2.5) / (abs(dx) + 0.001)
        sy = (self._cy + _TILE_SZ * 2.5) / (abs(dy) + 0.001)
        scale = min(sx, sy) * 1.45
        return QtCore.QPoint(
            int(self._cx + dx * scale - sz * 0.5),
            int(self._cy + dy * scale - sz * 0.5),
        )

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_tiles(self) -> None:
        # Scan for real album art (quick, synchronous, 2 levels deep)
        art = _quick_scan_art(self._saved, _N_TILES)

        pool = (_GRADS * 4)[: _N_TILES]
        random.shuffle(pool)

        for i in range(_N_TILES):
            angle  = self._base_angles[i]
            target = self._ellipse_pos(angle, _TILE_SZ)
            pixmap = art[i] if i < len(art) else None
            tile   = _Tile(self, pool[i % len(pool)], target, pixmap)
            tile.move(self._off_screen(target))
            tile.show()
            self._tiles.append(tile)

    def _build_cta(self) -> None:
        self._cta = _CTACard(self, self._saved)
        self._cta.setGeometry(self._center_rect)
        self._cta.open_clicked.connect(self._on_open_clicked)
        self._cta.reuse_clicked.connect(lambda: self._accept(self._saved))
        self._cta.raise_()
        self._cta.show()

    # ── Public API ────────────────────────────────────────────────────────

    def show_animated(self) -> None:
        """Fade the window in, then start the staggered tile fly-in."""
        self.setWindowOpacity(0.0)
        self.show()
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(_FADE_IN_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        anim.finished.connect(self._fly_in)
        self._fade_in_anim = anim
        anim.start()

    # ── Fly-in ────────────────────────────────────────────────────────────

    def _fly_in(self) -> None:
        """Staggered fly-in from edges → initial ellipse positions.
        When complete, rotation begins.
        """
        order = list(range(_N_TILES))
        random.shuffle(order)

        grp = QtCore.QParallelAnimationGroup(self)
        for seq_i, tile_i in enumerate(order):
            tile = self._tiles[tile_i]
            seq  = QtCore.QSequentialAnimationGroup(grp)
            seq.addAnimation(QtCore.QPauseAnimation(seq_i * _STAGGER_MS))

            anim = QtCore.QPropertyAnimation(tile, b"pos")
            anim.setStartValue(self._off_screen(tile._target))
            anim.setEndValue(tile._target)
            anim.setDuration(_FLY_IN_MS)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutBack)
            seq.addAnimation(anim)

            grp.addAnimation(seq)

        self._fly_in_grp = grp
        grp.finished.connect(self._start_rotation)
        grp.start()

    # ── Rotation ──────────────────────────────────────────────────────────

    def _start_rotation(self) -> None:
        """Begin the continuous ellipse rotation after fly-in completes."""
        self._tick_count = 0
        timer = QtCore.QTimer(self)
        timer.setInterval(16)          # ~60 fps
        timer.timeout.connect(self._on_tick)
        self._rot_timer = timer
        timer.start()

    def _on_tick(self) -> None:
        """Advance the rotation angle and reposition every tile with depth.

        Two independent smooth-step ramps run from this tick counter:

        • *rot_ramp*   (0 → 1 over _ROT_RAMP ticks)  — scales rotation speed.
        • *depth_ramp* (0 → 1 over _DEPTH_RAMP ticks) — blends each tile's
          depth value from 1.0 (full-size / full-opacity, exactly where the
          fly-in left every tile) toward its true geometric depth.  Using a
          longer ramp than rotation means tiles continue to settle visually
          even after the ring is already spinning at full speed, giving one
          seamless transition instead of two jarring snaps.
        """
        t = self._tick_count

        # ── Rotation ease-in ─────────────────────────────────────────────
        rot_t = min(1.0, t / _ROT_RAMP)
        rot_ramp = rot_t * rot_t * (3.0 - 2.0 * rot_t)   # smoothstep
        self._rotation   += _ROT_SPEED * rot_ramp
        self._tick_count += 1

        # ── Depth blend ease-in ──────────────────────────────────────────
        dep_t = min(1.0, t / _DEPTH_RAMP)
        depth_ramp = dep_t * dep_t * (3.0 - 2.0 * dep_t)  # smoothstep

        # Compute depth for each tile and collect for z-sorting
        tile_data: list[tuple[float, _Tile, QtCore.QPoint, int]] = []
        for i, tile in enumerate(self._tiles):
            angle      = self._base_angles[i] + self._rotation
            true_depth = (1.0 + math.sin(angle)) / 2.0    # 0 = back, 1 = front

            # Blend: depth starts at 1.0 (fly-in state) and transitions to
            # true_depth as depth_ramp approaches 1.  This removes the abrupt
            # size/opacity jump that used to occur the instant the timer started.
            depth = 1.0 - depth_ramp * (1.0 - true_depth)

            sz  = max(
                4,
                int(_TILE_SZ * (_MIN_SCALE + (_MAX_SCALE - _MIN_SCALE) * depth)),
            )
            pos = self._ellipse_pos(angle, sz)
            tile_data.append((depth, tile, pos, sz))

        # Sort ascending so we raise() in depth order (front tile raised last)
        tile_data.sort(key=lambda d: d[0])
        for depth, tile, pos, sz in tile_data:
            tile.set_depth(depth)
            tile.move(pos)
            tile.raise_()

        # CTA card always on top
        self._cta.raise_()

    # ── User action handlers ───────────────────────────────────────────────

    def _on_open_clicked(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Music Library Folder",
            str(Path.home()),
            QtWidgets.QFileDialog.Option.ShowDirsOnly,
        )
        if path:
            self._accept(path)

    def _accept(self, path: str) -> None:
        if not path:
            return
        self._pending = path

        # Stop fly-in if still running
        if (
            self._fly_in_grp is not None
            and self._fly_in_grp.state()
            == QtCore.QAbstractAnimation.State.Running
        ):
            self._fly_in_grp.stop()

        # Stop rotation timer
        if self._rot_timer is not None:
            self._rot_timer.stop()
            self._rot_timer = None

        self._do_scatter()

    # ── Scatter / exit ────────────────────────────────────────────────────

    def _do_scatter(self) -> None:
        """Scatter tiles outward from their current positions, then fade out."""
        order = list(range(len(self._tiles)))
        random.shuffle(order)

        grp = QtCore.QParallelAnimationGroup(self)
        for tile_i in order:
            tile = self._tiles[tile_i]
            seq  = QtCore.QSequentialAnimationGroup(grp)
            seq.addAnimation(QtCore.QPauseAnimation(random.randint(0, _SCATTER_MAX)))

            anim = QtCore.QPropertyAnimation(tile, b"pos")
            anim.setStartValue(tile.pos())
            anim.setEndValue(self._scatter_target(tile.pos(), tile.width()))
            anim.setDuration(_FLY_OUT_MS)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.InBack)
            seq.addAnimation(anim)
            grp.addAnimation(seq)

        self._fly_out_grp = grp
        grp.finished.connect(self._fade_out)
        grp.start()

    def _fade_out(self) -> None:
        # Emit path now so the caller can start fading the main window in
        # simultaneously — creating a smooth cross-dissolve.
        self.library_selected.emit(self._pending)

        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setDuration(FADE_OUT_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        anim.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        anim.finished.connect(self._done)
        self._fade_out_anim = anim
        anim.start()

    def _done(self) -> None:
        self.hide()
        self.finished.emit()

    # ── Background painting ────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            from gui.themes.manager import get_manager
            from gui.widgets.gradient_bg import (
                _gradient_enabled,
                paint_window_gradient,
            )
            t = get_manager().current
            p.fillRect(self.rect(), QtGui.QColor(t.sidebar_bg))
            if _gradient_enabled():
                paint_window_gradient(
                    p, self, t,
                    radius_scale=0.85,
                    alpha_tr=55,
                    alpha_bl=70,
                )
        except Exception:
            p.fillRect(self.rect(), QtGui.QColor("#0d1117"))
        finally:
            p.end()
