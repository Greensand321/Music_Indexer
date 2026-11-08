"""AlphaDEX — the album-art landing (Liquid Glass).

The first window the app shows. A 7 × 5 grid of album tiles fills the
window (padding 28, gap 16, radius 22): placeholders fly in from off-screen,
and the library's own covers replace them as the scanner finds them — random
albums on every launch. A radial dark scrim calms the centre, where a glass
card holds the brand, the library and the choices:

* **Initialize** — open the saved library (when one exists);
* **Choose Library Folder** — the main button when there is no saved
  library, or the saved one is gone ("Saved library not found");
* **Choose another folder** — a link under Initialize;
* **click any album** — open the library and start playing that album;
* **quit** — the glass ✕ button at the top right (``quit_requested``).

Hovering a tile lifts it (scale 1.06), rings it (2 px white) and shows a
glass caption chip with a play disc, the album and the artist. The lift is
painted by a mouse-transparent overlay (``_HoverLift``) — tiles never change
geometry for it.

Sequence
--------
1. ``show_animated()`` — fade the window in (``faded_in`` is emitted when it
   is fully visible), pause on the card, then start the tile fly-in.
2. The user clicks Initialize / chooses a folder / clicks a tile → ``_accept``.
3. The pending fly-in (timer and animation) is cancelled; tiles scatter to
   random off-screen targets from wherever they are.  ``_accept`` runs once:
   later clicks are ignored.
4. ``library_selected(path)`` is emitted at the *start* of the fade-out so
   the caller can cross-fade the main window in simultaneously.  A safety
   timer guarantees the fade-out happens even if the scatter animation is
   interrupted.
5. ``finished`` is emitted after the landing window is hidden.

With reduced motion the fly-in and the scatter are skipped (same signals,
same order).

Cover discovery (``_CoverScan``) is shared with the main window's backdrop
(``gui/widgets/backdrop.py``): one scanner, two consumers. The scanner and
the folder-row track count run under ``gui.workers.start_worker`` and are
cancelled (never terminated) when the landing exits or is closed.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.tokens import PLACEHOLDER_GRADIENTS, PILL, Surface, rgba

# ── Grid constants ─────────────────────────────────────────────────────────────
_COLS      = 7
_ROWS      = 5
_PAD       = 28    # window edge → grid
_GAP       = 16    # between tiles
_RADIUS    = 22    # tile corners
_TILE_SZ   = 110   # default cover bake size (px, square) when no size is given

# The glass card in the middle.
_CARD_W      = 440
_CARD_RADIUS = 34

# Hover lift.
_LIFT_SCALE  = 1.06
_LIFT_MARGIN = 48   # room around a lifted tile for its ring and shadow

# ── Timing ────────────────────────────────────────────────────────────────────
_STAGGER_MS  = 38    # delay increment per tile during fly-in
_FLY_IN_MS   = 500   # each tile's fly-in duration
_FLY_OUT_MS  = 360   # each tile's scatter duration
_SCATTER_MAX = 160   # max random pre-scatter delay (ms)
_FADE_IN_MS  = 320   # landing window fade-in

# Exported so alpha_dex_gui.py can match the cross-fade duration exactly.
FADE_OUT_MS  = 420   # landing window fade-out / main-window fade-in

_SCAN_DEPTH = 7   # max folder depth searched for art (root = 0)

# ── Art-scanner diagnostics ────────────────────────────────────────────────────
# A one-line scan summary is logged at DEBUG level (never printed).
_log = logging.getLogger(__name__)

# Pause between the window finishing its fade-in and the tiles starting to fly.
# During this window the card is the only visible element (the logo moment).
_LOGO_PAUSE_MS: int = 600

# ── Colour pool — diagonal gradient pairs for placeholder tiles ───────────────
_GRADS: list[tuple[str, str]] = list(PLACEHOLDER_GRADIENTS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_existing_dir(path: str) -> str:
    """The deepest existing ancestor of *path* (or "" if none / no path)."""
    if not path:
        return ""
    p = Path(path)
    for candidate in (p, *p.parents):
        try:
            if candidate.is_dir():
                return str(candidate)
        except OSError:
            continue
    return ""


def _tokens():
    from gui.themes.manager import get_manager
    return get_manager().current


def _reduced_motion() -> bool:
    try:
        from gui.themes import motion
        return motion.reduced_motion()
    except Exception:  # pragma: no cover
        return False


def _font(role: str, **kw) -> QtGui.QFont:
    from gui.themes.glass import font
    return font(role, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# _Tile
# ─────────────────────────────────────────────────────────────────────────────

class _Tile(QtWidgets.QWidget):
    """One album tile: a cover (or a colourful placeholder) with round corners."""

    clicked = Signal(str)        # emitted on click; payload is the directory path
    hovered = Signal(object, bool)  # (tile, entering)

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        grad: tuple[str, str],
        target: QtCore.QPoint,
        size: QtCore.QSize | None = None,
    ) -> None:
        super().__init__(parent)
        self._grad        = grad          # kept for lazy placeholder bake
        self._target      = target
        self._dirpath:    str             = ""   # set when art scanner assigns this tile
        self._title:      str             = ""
        self._artist:     str             = ""
        self._ready_pm:   QtGui.QPixmap | None = None
        self._placeholder: QtGui.QPixmap | None = None  # baked on first paintEvent
        size = size or QtCore.QSize(_TILE_SZ, _TILE_SZ)
        self.setFixedSize(size)
        # Prevent Qt from pre-filling the background so rounded corners show
        # the backdrop through them.
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

    def set_pixmap(self, pm: QtGui.QPixmap) -> None:
        """Store the pre-baked pixmap (baking done off-thread); tile repaints."""
        self._ready_pm = pm
        self.update()

    def set_dirpath(self, path: str) -> None:
        """Record which library directory this tile's cover came from."""
        self._dirpath = path

    def set_info(self, title: str, artist: str) -> None:
        self._title, self._artist = title, artist

    def info(self) -> tuple[str, str]:
        return self._title, self._artist

    def cover(self) -> QtGui.QPixmap:
        """What the tile shows now (the cover, or its placeholder)."""
        if self._ready_pm is not None:
            return self._ready_pm
        if self._placeholder is None:
            self._placeholder = self._bake_placeholder(self._grad, self.width(), self.height())
        return self._placeholder

    def has_art(self) -> bool:
        return self._ready_pm is not None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._dirpath:
            self.clicked.emit(self._dirpath)
        super().mousePressEvent(event)

    def enterEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(self, True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(self, False)
        super().leaveEvent(event)

    # ── Bake helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _bake_placeholder(grad: tuple[str, str], w: int, h: int | None = None,
                          radius: int = _RADIUS) -> QtGui.QPixmap:
        """Render gradient + sheen + shade into a QPixmap once (GUI thread)."""
        h = h or w
        out = QtGui.QPixmap(w, h)
        out.fill(QtGui.QColor(0, 0, 0, 0))
        r = QtCore.QRectF(0, 0, w, h)
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, radius, radius)
        p = QtGui.QPainter(out)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            lg = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
            lg.setColorAt(0.0, QtGui.QColor(grad[0]))
            lg.setColorAt(1.0, QtGui.QColor(grad[1]))
            p.fillPath(path, QtGui.QBrush(lg))
            shadow = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
            shadow.setColorAt(0.55, QtGui.QColor(0, 0, 0, 0))
            shadow.setColorAt(1.0,  QtGui.QColor(0, 0, 0, 70))
            p.fillPath(path, QtGui.QBrush(shadow))
        finally:
            p.end()
        return out

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.drawPixmap(0, 0, self.cover())
        finally:
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# _HoverLift — the lifted, ringed, captioned copy of the hovered tile
# ─────────────────────────────────────────────────────────────────────────────

class _HoverLift(QtWidgets.QWidget):
    """Paints the hovered tile lifted (scale 1.06), with a 2 px white ring, a
    deep shadow and a glass caption chip. Transparent for the mouse, so the
    tile underneath keeps its hover and click; tiles never move for it."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setObjectName("landingHoverLift")
        self._tile: _Tile | None = None
        self._progress = 0.0
        self.hide()

    def tile(self) -> _Tile | None:
        return self._tile

    def progress(self) -> float:
        return self._progress

    def show_for(self, tile: _Tile) -> None:
        from gui.themes import motion
        if self._tile is not tile:
            self._tile = tile
            self._progress = 0.0
        geo = tile.geometry().adjusted(-_LIFT_MARGIN, -_LIFT_MARGIN, _LIFT_MARGIN, _LIFT_MARGIN)
        self.setGeometry(geo)
        self.show()
        self.raise_()
        motion.animate_value(self, "lift", self._progress, 1.0, self._on_value,
                             duration="fast", easing="standard")

    def hide_for(self, tile: _Tile) -> None:
        from gui.themes import motion
        if tile is not self._tile:
            return
        motion.animate_value(self, "lift", self._progress, 0.0, self._on_value,
                             duration="fast", easing="exit", on_finished=self._on_gone)

    def drop(self) -> None:
        from gui.themes import motion
        motion.stop(self, "lift")
        self._tile = None
        self._progress = 0.0
        self.hide()

    def _on_gone(self) -> None:
        if self._progress <= 0.001:
            self._tile = None
            self.hide()

    def _on_value(self, value: float) -> None:
        self._progress = float(value)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tile = self._tile
        if tile is None or self._progress <= 0.0:
            return
        from gui.themes.glass import paint_shadow, paint_surface, qcolor

        t = self._progress
        rect = QtCore.QRectF(_LIFT_MARGIN, _LIFT_MARGIN, tile.width(), tile.height())
        scale = 1.0 + (_LIFT_SCALE - 1.0) * t
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            c = rect.center()
            p.translate(c)
            p.scale(scale, scale)
            p.translate(-c)
            # 0 24px 60px rgba(0,0,0,.6), scaled into the margin we have.
            paint_shadow(p, rect, Surface("", shadow="rgba(0,0,0,0.6)", blur=28, offset_y=14,
                                          radius=_RADIUS), strength=t)
            p.drawPixmap(rect.topLeft(), tile.cover())
            ring = QtGui.QPen(qcolor("rgba(255,255,255,0.55)", t), 2.0)
            p.setPen(ring)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), _RADIUS + 1, _RADIUS + 1)
            title, artist = tile.info()
            if tile.has_art() and (title or artist):
                self._paint_caption(p, rect, title, artist, t, paint_surface, qcolor)
        finally:
            p.end()

    @staticmethod
    def _paint_caption(p, rect, title, artist, t, paint_surface, qcolor) -> None:
        chip = QtCore.QRectF(rect.left() + 10, rect.bottom() - 10 - 46, rect.width() - 20, 46)
        paint_surface(p, chip, Surface("rgba(20,24,36,0.62)", "rgba(255,255,255,0.24)",
                                       "rgba(255,255,255,0.30)", radius=16), opacity=t)
        disc = QtCore.QRectF(chip.left() + 8, chip.center().y() - 15, 30, 30)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(qcolor("rgba(255,255,255,1)", t))
        p.drawEllipse(disc)
        tri = QtGui.QPolygonF([
            QtCore.QPointF(disc.center().x() - 3.5, disc.center().y() - 6),
            QtCore.QPointF(disc.center().x() + 6.5, disc.center().y()),
            QtCore.QPointF(disc.center().x() - 3.5, disc.center().y() + 6),
        ])
        p.setBrush(qcolor(_tokens().text_inverse, t))
        p.drawPolygon(tri)
        text_left = disc.right() + 10
        width = chip.right() - 10 - text_left
        f1 = _font("label", weight=650)
        f2 = _font("micro", weight=400)
        f2.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, 0.0)
        fm1, fm2 = QtGui.QFontMetrics(f1), QtGui.QFontMetrics(f2)
        top = chip.center().y() - (fm1.height() + fm2.height()) / 2
        p.setFont(f1)
        p.setPen(qcolor("rgba(255,255,255,1)", t))
        p.drawText(QtCore.QRectF(text_left, top, width, fm1.height()),
                   int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                   fm1.elidedText(title or artist, QtCore.Qt.TextElideMode.ElideRight, int(width)))
        if title and artist:
            p.setFont(f2)
            p.setPen(qcolor("rgba(236,241,255,0.8)", t))
            p.drawText(QtCore.QRectF(text_left, top + fm1.height(), width, fm2.height()),
                       int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                       fm2.elidedText(artist, QtCore.Qt.TextElideMode.ElideRight, int(width)))


# ─────────────────────────────────────────────────────────────────────────────
# _Scrim — the radial dark centre (cached pixmap, mouse-transparent)
# ─────────────────────────────────────────────────────────────────────────────

class _Scrim(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._pm: QtGui.QPixmap | None = None

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._pm = None
        super().resizeEvent(event)

    def _pixmap(self) -> QtGui.QPixmap:
        if self._pm is None or self._pm.size() != self.size():
            base = QtGui.QColor(_tokens().material.backdrop_base).darker(140)
            pm = QtGui.QPixmap(self.size())
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            p = QtGui.QPainter(pm)
            try:
                # radial-gradient(520px 360px at 50% 50%, .72 → .16 at 70%)
                c = QtCore.QPointF(self.width() / 2, self.height() / 2)
                p.translate(c)
                p.scale(1.0, 360.0 / 520.0)
                g = QtGui.QRadialGradient(QtCore.QPointF(0, 0), 520.0)
                inner, outer = QtGui.QColor(base), QtGui.QColor(base)
                inner.setAlphaF(0.72)
                outer.setAlphaF(0.16)
                g.setColorAt(0.0, inner)
                g.setColorAt(0.7, outer)
                g.setColorAt(1.0, outer)
                p.fillRect(QtCore.QRectF(-self.width(), -self.height() * 2,
                                         self.width() * 2, self.height() * 4), QtGui.QBrush(g))
            finally:
                p.end()
            self._pm = pm
        return self._pm

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            r = event.rect()
            p.drawPixmap(r, self._pixmap(), r)
        finally:
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Small painted controls
# ─────────────────────────────────────────────────────────────────────────────

class _BrandTile(QtWidgets.QWidget):
    """The AlphaDEX mark: a glass tile with accent level bars."""

    def __init__(self, size: int = 64, bars: tuple[int, ...] = (16, 30, 22, 12),
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._bars = bars
        self.setFixedSize(size, size)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_brand_tile
        p = QtGui.QPainter(self)
        try:
            paint_brand_tile(p, QtCore.QRectF(self.rect()), self._bars)
        finally:
            p.end()


class _QuitButton(QtWidgets.QAbstractButton):
    """A 44 px glass circle with a ✕."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setToolTip("Quit AlphaDEX")
        self.setAccessibleName("Quit AlphaDEX")
        self.setObjectName("landingQuit")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_surface, qcolor
        t = _tokens()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            hover = self.underMouse()
            paint_surface(p, QtCore.QRectF(self.rect()),
                          Surface("rgba(20,24,36,0.44)" if not hover else "rgba(40,44,60,0.6)",
                                  "rgba(255,255,255,0.26)", radius=PILL))
            pen = QtGui.QPen(qcolor(t.text_primary), 2.2, QtCore.Qt.PenStyle.SolidLine,
                             QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            c = QtCore.QRectF(self.rect()).center()
            d = 6.0
            p.drawLine(QtCore.QPointF(c.x() - d, c.y() - d), QtCore.QPointF(c.x() + d, c.y() + d))
            p.drawLine(QtCore.QPointF(c.x() + d, c.y() - d), QtCore.QPointF(c.x() - d, c.y() + d))
        finally:
            p.end()


class _InitButton(QtWidgets.QPushButton):
    """Full-width white button (52 high, radius 18) with dark text."""

    def __init__(self, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setFixedHeight(52)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFont(_font("heading", weight=700))
        self.setObjectName("landingInitialize")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_shadow, qcolor
        t = _tokens()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(self.rect()).adjusted(0, 0, 0, -4)
            fill = qcolor(t.text_primary)
            if self.isDown():
                fill = fill.darker(112)
            elif self.underMouse():
                fill = fill.lighter(104)
            path = QtGui.QPainterPath()
            path.addRoundedRect(r, 18, 18)
            paint_shadow(p, r.adjusted(6, 2, -6, 0),
                         Surface("", shadow="rgba(0,0,0,0.35)", blur=6, offset_y=2, radius=18))
            p.fillPath(path, fill)
            p.setPen(qcolor(t.text_inverse))
            p.setFont(self.font())
            p.drawText(r, int(QtCore.Qt.AlignmentFlag.AlignCenter), self.text())
        finally:
            p.end()


class _LinkButton(QtWidgets.QPushButton):
    """A text link ("Choose another folder")."""

    def __init__(self, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setFlat(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFont(_font("label", weight=600))
        self.setObjectName("landingLink")

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        fm = QtGui.QFontMetrics(self.font())
        return QtCore.QSize(fm.horizontalAdvance(self.text()) + 8, fm.height() + 8)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import qcolor
        t = _tokens()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            p.setPen(qcolor(t.text_primary if self.underMouse() else t.accent_hover))
            p.setFont(self.font())
            p.drawText(self.rect(), int(QtCore.Qt.AlignmentFlag.AlignCenter), self.text())
        finally:
            p.end()


class _FolderRow(QtWidgets.QWidget):
    """The library path and its counts, on a well."""

    def __init__(self, path: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("landingFolderRow")
        self._path = path
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(14 + 18 + 10, 12, 14, 12)
        lay.setSpacing(0)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(1)
        self.path_label = QtWidgets.QLabel()
        self.path_label.setObjectName("landingFolderPath")
        self.path_label.setFont(_font("body", weight=600))
        self.path_label.setToolTip(path)
        self.stats_label = QtWidgets.QLabel("Counting tracks…")
        self.stats_label.setObjectName("landingFolderStats")
        self.stats_label.setFont(_font("caption"))
        col.addWidget(self.path_label)
        col.addWidget(self.stats_label)
        lay.addLayout(col, 1)
        self._elide()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        fm = QtGui.QFontMetrics(self.path_label.font())
        width = max(80, self.width() - 14 - 18 - 10 - 14) if self.width() > 0 else 320
        self.path_label.setText(fm.elidedText(self._path, QtCore.Qt.TextElideMode.ElideMiddle, width))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_surface, qcolor
        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()),
                          Surface("rgba(255,255,255,0.08)", "rgba(255,255,255,0.14)", radius=16))
            # Folder glyph: M3 7h6l2 2h10v10H3z, 18 px.
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            x0, y0 = 14.0, self.height() / 2 - 9
            k = 18.0 / 24.0
            path = QtGui.QPainterPath()
            pts = [(3, 7), (9, 7), (11, 9), (21, 9), (21, 19), (3, 19)]
            path.moveTo(x0 + pts[0][0] * k, y0 + pts[0][1] * k)
            for px, py in pts[1:]:
                path.lineTo(x0 + px * k, y0 + py * k)
            path.closeSubpath()
            p.setPen(QtGui.QPen(qcolor("rgba(236,241,255,0.85)"), 1.6, QtCore.Qt.PenStyle.SolidLine,
                                QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        finally:
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# _CTACard
# ─────────────────────────────────────────────────────────────────────────────

class _CTACard(QtWidgets.QFrame):
    """The glass centre card: brand, library, Initialize and the choices."""

    open_clicked  = Signal()
    reuse_clicked = Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        saved_path: str = "",
        missing_path: str = "",
    ) -> None:
        super().__init__(parent)
        self._saved = saved_path
        self._missing = missing_path
        self.main_button: QtWidgets.QPushButton | None = None
        self.other_button: QtWidgets.QPushButton | None = None
        self.tagline: QtWidgets.QLabel | None = None
        self.folder_row: _FolderRow | None = None
        self.hint: QtWidgets.QLabel | None = None
        self.setObjectName("landingCard")
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFixedWidth(_CARD_W)
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(34, 34, 34, 28)
        lay.setSpacing(18)
        center = QtCore.Qt.AlignmentFlag.AlignHCenter

        lay.addWidget(_BrandTile(64), 0, center)

        brand_col = QtWidgets.QVBoxLayout()
        brand_col.setSpacing(6)
        name_lbl = QtWidgets.QLabel("AlphaDEX")
        name_lbl.setObjectName("landingTitle")
        nf = _font("display", px=40, weight=800)
        nf.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, -0.04 * 40)
        name_lbl.setFont(nf)
        name_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        brand_col.addWidget(name_lbl)

        if self._missing:
            # The saved library is gone (folder deleted, drive unplugged):
            # say so instead of offering an "Initialize" that cannot work.
            tag_lbl = QtWidgets.QLabel("Saved library not found.\nChoose a folder to continue.")
            tag_lbl.setToolTip(f"Not found: {self._missing}")
        elif self._saved:
            tag_lbl = QtWidgets.QLabel("Your library is ready.")
        else:
            tag_lbl = QtWidgets.QLabel("Choose your music library to begin.")
        tag_lbl.setObjectName("landingHint")
        tag_lbl.setFont(_font("body"))
        tag_lbl.setWordWrap(True)
        tag_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.tagline = tag_lbl
        brand_col.addWidget(tag_lbl)
        lay.addLayout(brand_col)

        if self._saved:
            self.folder_row = _FolderRow(self._saved)
            lay.addWidget(self.folder_row)

        # Button text and behaviour depend on whether a library is saved:
        # saved → "Initialize" opens it; none / missing → choose a folder.
        if self._saved:
            main_btn = _InitButton("Initialize")
            main_btn.setToolTip(f"Open saved library: {self._saved}")
            main_btn.clicked.connect(self.reuse_clicked.emit)
        else:
            main_btn = _InitButton("Choose Library Folder")
            main_btn.setToolTip("Select a music library folder")
            main_btn.clicked.connect(self.open_clicked.emit)
        lay.addWidget(main_btn)
        self.main_button = main_btn

        if self._saved:
            other = _LinkButton("Choose another folder")
            other.setToolTip("Open a different music library folder")
            other.clicked.connect(self.open_clicked.emit)
            lay.addWidget(other, 0, center)
            self.other_button = other
            hint = QtWidgets.QLabel("Or click any album to start playing it.")
            hint.setObjectName("landingHint")
            hint.setFont(_font("caption"))
            hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(hint)
            self.hint = hint

    def set_stats(self, text: str) -> None:
        if self.folder_row is not None:
            self.folder_row.stats_label.setText(text)

    def set_frost(self, image: QtGui.QImage | None) -> None:
        """The blurred tiles behind the card (the backdrop-filter of the
        mockup, pre-blurred off the GUI thread)."""
        self._frost = QtGui.QPixmap.fromImage(image) if image is not None and not image.isNull() else None
        self.update()

    def frost(self) -> QtGui.QPixmap | None:
        return getattr(self, "_frost", None)

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_surface, qcolor
        p = QtGui.QPainter(self)
        try:
            m = _tokens().material
            frost = self.frost()
            if frost is not None and m.is_glass:
                p.save()
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                clip = QtGui.QPainterPath()
                clip.addRoundedRect(QtCore.QRectF(self.rect()), _CARD_RADIUS, _CARD_RADIUS)
                p.setClipPath(clip)
                p.drawPixmap(self.rect(), frost)
                p.restore()
            spec = m.capsule
            if m.is_glass:
                # rgba(22,26,40,.5) · border .20 · highlight .34
                spec = Surface("rgba(22,26,40,0.5)", "rgba(255,255,255,0.2)",
                               "rgba(255,255,255,0.34)", radius=_CARD_RADIUS)
            paint_surface(p, QtCore.QRectF(self.rect()), spec, radius=_CARD_RADIUS)
            if m.is_glass:
                # inset 0 -1px 0 rgba(255,255,255,.06): a faint bottom edge.
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                p.setPen(QtGui.QPen(qcolor("rgba(255,255,255,0.06)"), 1.0))
                y = self.height() - 1.5
                p.drawLine(QtCore.QPointF(_CARD_RADIUS, y), QtCore.QPointF(self.width() - _CARD_RADIUS, y))
        finally:
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# _ArtHistory — persistent "do-not-repeat" log for the mosaic scanner
# ─────────────────────────────────────────────────────────────────────────────

# Audio extensions understood by read_metadata (including the opus reader).
_AUDIO_EXTS: frozenset[str] = frozenset(
    {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
)
# Lowercase variants for fast extension checking in hot path
_AUDIO_EXTS_LOWER: frozenset[str] = frozenset(
    ext.lower() for ext in _AUDIO_EXTS
)


def _is_audio_file(filename: str) -> bool:
    """Fast check if filename has an audio extension (case-insensitive)."""
    dot_pos = filename.rfind('.')
    if dot_pos <= 0:
        return False
    ext = filename[dot_pos:].lower()
    return ext in _AUDIO_EXTS_LOWER


class _ArtHistory:
    """FIFO log of directory paths recently used for the landing mosaic.

    Persisted as a JSON array at ``~/.soundvault_art_history.json``.  Capped
    at ``MAX_ENTRIES`` paths; when full the oldest entries are evicted so the
    scanner keeps cycling through fresh parts of the library indefinitely.

    The scanner passes this object to ``_ordered_dirs()`` which puts unseen
    directories first (shuffled) and recently-seen directories last (also
    shuffled).  After emitting, the directories that contributed to the on-
    screen tiles are appended here and marked dirty. Saves are deferred to
    avoid blocking I/O on the scanner thread.
    """

    MAX_ENTRIES = 10_000
    _PATH = os.path.expanduser("~/.soundvault_art_history.json")

    def __init__(self) -> None:
        self._log: list[str] = []   # ordered oldest → newest
        self._set: set[str] = set() # fast membership test
        self._dirty: bool = False   # whether changes need to be persisted
        self._load()

    def _load(self) -> None:
        try:
            with open(self._PATH, encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                self._log = [p for p in raw if isinstance(p, str)]
                self._set = set(self._log)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[Warning] Failed to load art history: {e}", file=sys.stderr)

    def add_and_save(self, paths: list[str]) -> None:
        """Append *paths* (deduped), trim to MAX_ENTRIES, and mark dirty."""
        for p in paths:
            if p not in self._set:
                self._log.append(p)
                self._set.add(p)
        if len(self._log) > self.MAX_ENTRIES:
            evicted    = self._log[: len(self._log) - self.MAX_ENTRIES]
            self._log  = self._log[-self.MAX_ENTRIES :]
            self._set -= set(evicted)
        self._dirty = True

    def save_if_dirty(self) -> None:
        """Persist changes to disk if any have been made since last save."""
        if not self._dirty:
            return
        try:
            temp_path = self._PATH + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as fh:
                json.dump(self._log, fh, separators=(",", ":"))
            os.replace(temp_path, self._PATH)
            self._dirty = False
        except Exception as e:
            print(f"[Warning] Failed to save art history: {e}", file=sys.stderr)

    def __contains__(self, path: str) -> bool:
        return path in self._set

    def __len__(self) -> int:
        return len(self._log)


# ─────────────────────────────────────────────────────────────────────────────
# _CoverScan — album-cover discovery shared by the landing and the backdrop
# ─────────────────────────────────────────────────────────────────────────────

Deliver = Callable[[int, QtGui.QImage, str, str, str], None]


class _CoverScan:
    """Find album covers in a library and bake them into QImages.

    Mixed into QThread subclasses (``_ArtScanner`` for the landing mosaic,
    ``gui.widgets.backdrop.BackdropWorker`` for the window backdrop); call
    ``_init_cover_scan`` from ``__init__`` and ``scan(deliver)`` from
    ``run()``. Only QImage is touched (QPixmap is GUI-thread only).

    Cover extraction
    ----------------
    ``_covers_in_dir()`` checks each directory in two stages:

    1. **Sidecar images** (fast path) — ``cover.jpg``, ``folder.jpg``, …
    2. **Embedded art** — ``utils.audio_metadata_reader.read_metadata`` (the
       canonical extractor for the whole app).

    Each cover comes with its album and artist (tags, else folder names) for
    the landing's caption chip.

    Directory ordering — fresh-first with history
    ---------------------------------------------
    ``_ordered_dirs()`` collects all directories up to ``_SCAN_DEPTH`` levels
    (including the library root itself for flat libraries) and splits them into
    two independently-shuffled buckets: fresh (not in ``_ArtHistory``) first,
    then used. With ``use_history=False`` (the backdrop) the history is neither
    read nor written.
    """

    _MAX_SCAN_WORKERS   = 6  # parallel cover-extraction threads
    _IN_FLIGHT          = _MAX_SCAN_WORKERS * 3  # futures kept active at once

    def _init_cover_scan(self, library_path: str, count: int, *, tile_w: int = _TILE_SZ,
                         tile_h: int | None = None, radius: int = 10, vignette: bool = True,
                         use_history: bool = True) -> None:
        self._library = library_path
        self._n       = count
        self._tile_w  = int(tile_w)
        self._tile_h  = int(tile_h or tile_w)
        self._radius  = radius
        self._vignette = vignette
        self._use_history = use_history
        # Set by cancel(); also polled by the pool workers (thread-safe).
        self._cancel  = threading.Event()

    def cancel(self) -> None:
        """Stop at the next safe point (directory walk, file, or future)."""
        self._cancel.set()
        self.requestInterruption()  # type: ignore[attr-defined]

    def _stopping(self) -> bool:
        return self._cancel.is_set() or self.isInterruptionRequested()  # type: ignore[attr-defined]

    # ── Low-level helpers ─────────────────────────────────────────────────

    @staticmethod
    def _bake_image(raw: bytes, size: int = _TILE_SZ, radius: int = 10, *,
                    height: int | None = None, vignette: bool = True) -> QtGui.QImage | None:
        """Decode, scale, round-clip, and vignette *raw* cover bytes into a QImage.

        Called from worker threads (QPainter on QImage is reentrant).
        ``size`` is the width; ``height`` defaults to it (square)."""
        w, h = int(size), int(height or size)
        src = QtGui.QImage()
        if not src.loadFromData(raw) or src.isNull():
            return None
        src = src.scaled(
            QtCore.QSize(w, h),
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        out = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(QtGui.QColor(0, 0, 0, 0))
        r    = QtCore.QRectF(0, 0, w, h)
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, radius, radius)
        p = QtGui.QPainter(out)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setClipPath(path)
            ox = (src.width()  - w) // 2
            oy = (src.height() - h) // 2
            p.drawImage(QtCore.QPoint(-ox, -oy), src)
            p.setClipping(False)
            if vignette:
                vg = QtGui.QRadialGradient(r.center(), max(r.width(), r.height()) * 0.75)
                vg.setColorAt(0.5, QtGui.QColor(0, 0, 0, 0))
                vg.setColorAt(1.0, QtGui.QColor(0, 0, 0, 70))
                p.fillPath(path, QtGui.QBrush(vg))
        finally:
            p.end()
        return out

    @staticmethod
    def _scandir_files(dirpath: str) -> list[str]:
        """Shuffled list of filenames in *dirpath*; returns [] on OSError."""
        try:
            names = [e.name for e in os.scandir(dirpath) if e.is_file(follow_symlinks=False)]
        except OSError:
            return []
        random.shuffle(names)
        return names

    @staticmethod
    def _cover_from_file(path: str) -> tuple[str, bytes] | None:
        """``(method, image_bytes)`` for the first embedded cover, or None."""
        found = _CoverScan._read_file(path, include_cover=True)
        if found and found[1]:
            return ("read_metadata", found[1])
        return None

    @staticmethod
    def _read_file(path: str, include_cover: bool) -> tuple[dict, bytes | None] | None:
        try:
            from utils.audio_metadata_reader import read_metadata
            tags, covers, _err, _hint = read_metadata(path, include_cover=include_cover)
            return (tags or {}), (covers[0] if covers else None)
        except Exception:
            return None

    @staticmethod
    def _info(tags: dict | None, dirpath: str) -> tuple[str, str]:
        """(album, artist) from tags, else from the folder names."""
        tags = tags or {}
        album = str(tags.get("album") or "").strip()
        artist = str(tags.get("albumartist") or tags.get("artist") or "").strip()
        parts = Path(dirpath).parts
        if not album and parts:
            album = parts[-1]
        if not artist and len(parts) >= 2:
            artist = parts[-2]
        return album, artist

    # ── Directory ordering ────────────────────────────────────────────────

    def _ordered_dirs(self, history: _ArtHistory | None) -> list[str]:
        """All library dirs up to _SCAN_DEPTH, fresh dirs first."""
        fresh: list[str] = []
        used:  list[str] = []

        def _walk(dirpath: str, depth: int) -> None:
            if depth >= _SCAN_DEPTH or self._stopping():
                return
            try:
                entries = os.scandir(dirpath)
                subdirs = [e.name for e in entries if e.is_dir(follow_symlinks=False)]
            except OSError:
                return
            random.shuffle(subdirs)
            for name in subdirs:
                subdirpath = os.path.join(dirpath, name)
                bucket = used if history is not None and subdirpath in history else fresh
                bucket.append(subdirpath)
                _walk(subdirpath, depth + 1)

        # Include the library root itself so flat libraries (all files in one
        # folder) are not silently skipped.
        root_bucket = used if history is not None and self._library in history else fresh
        root_bucket.append(self._library)

        _walk(self._library, 0)
        random.shuffle(fresh)
        random.shuffle(used)
        return fresh + used

    # ── Per-directory cover helper ────────────────────────────────────────

    def _covers_in_dir(self, dirpath: str) -> list[tuple[str, str, bytes, str, str]]:
        """Every unique cover in *dirpath* as ``(method, dirpath, raw, album,
        artist)``; ``[]`` if none. Flat folders yield one entry per uniquely
        covered track; album folders one. Workers do I/O only — no Qt calls."""
        results:     list[tuple[str, str, bytes, str, str]] = []
        local_sigs:  set[bytes] = set()

        _COVER_NAMES = (
            "cover.jpg", "folder.jpg", "album.jpg", "artwork.jpg",
            "front.jpg", "cover.png", "folder.png", "album.png",
        )

        names = self._scandir_files(dirpath)
        audio = [n for n in names if _is_audio_file(n)]

        # ── 1. Sidecar image ───────────────────────────────────────────────
        for img_name in _COVER_NAMES:
            img_path = os.path.join(dirpath, img_name)
            try:
                if os.path.isfile(img_path):
                    with open(img_path, "rb") as fh:
                        raw = fh.read()
                    if raw:
                        sig = hashlib.md5(raw).digest()
                        local_sigs.add(sig)
                        tags = None
                        if audio:
                            found = self._read_file(os.path.join(dirpath, audio[0]), False)
                            tags = found[0] if found else None
                        album, artist = self._info(tags, dirpath)
                        results.append(("folder_image", dirpath, raw, album, artist))
                    break  # at most one sidecar per directory
            except OSError:
                pass

        # ── 2. Embedded art — every audio file in the directory ────────────
        for name in audio:
            if self._cancel.is_set():
                break
            found = self._read_file(os.path.join(dirpath, name), True)
            if found and found[1]:
                tags, raw = found
                sig = hashlib.md5(raw).digest()
                if sig not in local_sigs:
                    local_sigs.add(sig)
                    album, artist = self._info(tags, dirpath)
                    results.append(("read_metadata", dirpath, raw, album, artist))

        return results

    # ── Orchestration ─────────────────────────────────────────────────────

    def scan(self, deliver: Deliver) -> None:
        t0 = time.monotonic()

        history  = _ArtHistory() if self._use_history else None
        dirpaths = self._ordered_dirs(history)

        if self._stopping():
            return

        selected:       list[str] = []
        seen_sigs:      set[tuple[int, ...]] = set()   # visual fingerprints
        found:          list[tuple[QtGui.QImage, str, str, str]] = []
        tile_index:     int = 0
        method_counts:  dict[str, int] = {}
        no_cover_dirs:  int = 0
        w, h = self._tile_w, self._tile_h
        samples = [(int(w * fx), int(h * fy)) for fx in (0.3, 0.5, 0.7) for fy in (0.3, 0.5, 0.7)]

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._MAX_SCAN_WORKERS,
            thread_name_prefix="ArtScan",
        )
        pending:  dict[concurrent.futures.Future, None] = {}
        dir_iter  = iter(dirpaths)

        def _fill() -> None:
            while len(pending) < self._IN_FLIGHT:
                try:
                    d = next(dir_iter)
                    pending[executor.submit(self._covers_in_dir, d)] = None
                except StopIteration:
                    break

        try:
            _fill()
            while pending and tile_index < self._n and not self._stopping():
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                    timeout=1.0,
                )
                for f in done:
                    del pending[f]
                    try:
                        hits = f.result()
                    except Exception:
                        hits = []
                    if not hits:
                        no_cover_dirs += 1
                    else:
                        for method, dirpath, raw, album, artist in hits:
                            if tile_index >= self._n:
                                break
                            img = self._bake_image(raw, w, self._radius, height=h,
                                                   vignette=self._vignette)
                            if img is None:
                                continue
                            # 9-point visual fingerprint at 30/50/70 % (inset
                            # from the edges), 6 low bits masked: the same
                            # cover re-encoded or at another resolution dedups.
                            sig = tuple(img.pixel(x, y) & 0xC0C0C0C0 for x, y in samples)
                            if sig not in seen_sigs:
                                seen_sigs.add(sig)
                                deliver(tile_index, img, dirpath, album, artist)
                                found.append((img, dirpath, album, artist))
                                selected.append(dirpath)
                                method_counts[method] = method_counts.get(method, 0) + 1
                                tile_index += 1
                _fill()
        finally:
            # Tell in-flight directory reads to stop at the next file, then
            # wait for them: no pool thread outlives this scan.
            self._cancel.set()
            executor.shutdown(wait=True, cancel_futures=True)

        # ── Fill-cycle: fewer unique covers than tiles ─────────────────────
        fill_count = 0
        if tile_index < self._n and found and not self.isInterruptionRequested():  # type: ignore[attr-defined]
            pool = list(found)
            random.shuffle(pool)
            while tile_index < self._n and not self.isInterruptionRequested():  # type: ignore[attr-defined]
                img, d, album, artist = pool[fill_count % len(pool)]
                deliver(tile_index, img, d, album, artist)
                tile_index += 1
                fill_count += 1

        if _log.isEnabledFor(logging.DEBUG):
            unique = tile_index - fill_count
            parts = [f"tiles={tile_index}/{self._n}"]
            if fill_count:
                parts.append(f"unique={unique}, fill={fill_count}")
            parts += [f"{m}={c}" for m, c in sorted(method_counts.items())]
            if no_cover_dirs:
                parts.append(f"no-cover={no_cover_dirs}")
            parts.append(f"dirs={len(dirpaths)}")
            parts.append(f"elapsed={time.monotonic() - t0:.2f}s")
            _log.debug("[ArtScan] %s", ", ".join(parts))

        if history is not None:
            history.add_and_save(selected)
            history.save_if_dirty()


class _ArtScanner(_CoverScan, QtCore.QThread):
    """The landing's scanner: one baked tile image per tile, in a worker."""

    # (tile_index, image, dirpath, album, artist) — QImage is thread-safe
    art_found = Signal(int, QtGui.QImage, str, str, str)

    def __init__(self, library_path: str, tile_count: int,
                 tile_w: int = _TILE_SZ, tile_h: int | None = None) -> None:
        QtCore.QThread.__init__(self)
        self._init_cover_scan(library_path, tile_count, tile_w=tile_w, tile_h=tile_h,
                              radius=_RADIUS if tile_w != _TILE_SZ else 10)
        self.setObjectName("ArtScanner")

    def run(self) -> None:  # noqa: N802
        self.scan(self.art_found.emit)


class _FrostWorker(QtCore.QThread):
    """Blurs what is behind the landing card (read-only, off the GUI thread)."""

    frosted = Signal(int, QtGui.QImage)

    SIGMA = 38   # the mockup's backdrop-filter: blur(38px) saturate(190%)
    SATURATION = 1.9

    def __init__(self, image: QtGui.QImage, generation: int) -> None:
        super().__init__()
        self._image = QtGui.QImage(image)
        self._generation = generation

    def run(self) -> None:  # noqa: N802
        from gui.imaging import blur_image
        out = blur_image(self._image, self.SIGMA, scale=4, saturation=self.SATURATION)
        if not self.isInterruptionRequested():
            self.frosted.emit(self._generation, out)


class _CountWorker(QtCore.QThread):
    """Counts the saved library for the card's folder row (read-only)."""

    counted = Signal(str)

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self._library = library_path

    def run(self) -> None:  # noqa: N802
        from gui.library_stats import format_stats, library_stats
        try:
            result = library_stats(self._library, self.isInterruptionRequested)
        except Exception:  # noqa: BLE001 - best effort
            result = None
        if result is not None and not self.isInterruptionRequested():
            tracks, _size, artists = result
            self.counted.emit(format_stats(tracks, None, artists))


# ─────────────────────────────────────────────────────────────────────────────
# MosaicLanding
# ─────────────────────────────────────────────────────────────────────────────

class MosaicLanding(QtWidgets.QWidget):
    """Full-window album-art landing shown before the main window.

    Usage::

        landing = MosaicLanding(geometry, saved_path)
        landing.library_selected.connect(_on_library_selected)
        landing.finished.connect(_on_done)
        landing.show_animated()
    """

    library_selected = Signal(str)   # emitted at start of fade-out; path is ready
    finished         = Signal()      # emitted after window is hidden
    tile_clicked     = Signal(str)   # emitted when user clicks a tile; payload is dirpath
    faded_in         = Signal()      # emitted once the window is fully visible
    quit_requested   = Signal()      # the ✕ button

    def __init__(
        self,
        geometry: QtCore.QRect,
        saved_path: str = "",
        missing_path: str = "",
    ) -> None:
        """``saved_path``: a saved library that exists ("Initialize").
        ``missing_path``: a saved library that no longer exists; the card
        says so and offers "Choose Library Folder" instead."""
        super().__init__(None, QtCore.Qt.WindowType.FramelessWindowHint)
        self.setObjectName("landing")
        self.setGeometry(geometry)
        # Prevent the OS from maximizing or resizing a frameless window,
        # which would cause concurrent paintEvents at unexpected sizes.
        self.setFixedSize(geometry.width(), geometry.height())
        self._w       = geometry.width()
        self._h       = geometry.height()
        self._saved   = saved_path
        self._missing = "" if saved_path else missing_path
        self._pending = ""
        # Set by the first _accept(); makes the exit sequence run once.
        self._exiting = False
        self._fading_out = False

        self._tiles: list[_Tile] = []
        self._covers: list[QtGui.QImage] = []   # art found (handed to the backdrop)
        self._fly_in_grp:  QtCore.QParallelAnimationGroup | None = None
        self._fly_out_grp: QtCore.QParallelAnimationGroup | None = None
        # Keep animation references alive
        self._fade_in_anim:  object = None
        self._fade_out_anim: object = None
        self._scanner: object = None   # _ArtScanner | None
        self._counter: object = None   # _CountWorker | None
        self._art_history: _ArtHistory | None = None

        # The logo-moment pause before the fly-in. A member timer (not a bare
        # singleShot) so _accept() can cancel it: a fly-in that starts during
        # the scatter stops the scatter's animations and the landing never
        # exits (DS-01).
        self._fly_in_timer = QtCore.QTimer(self)
        self._fly_in_timer.setSingleShot(True)
        self._fly_in_timer.setInterval(_LOGO_PAUSE_MS)
        self._fly_in_timer.timeout.connect(self._fly_in)
        # Fallback in case the scatter group never reports ``finished``.
        self._exit_safety_timer = QtCore.QTimer(self)
        self._exit_safety_timer.setSingleShot(True)
        self._exit_safety_timer.timeout.connect(self._fade_out)
        # The card's frost (blurred tiles behind it), rebuilt — debounced —
        # when the tiles settle or new art lands.
        self._frost_timer = QtCore.QTimer(self)
        self._frost_timer.setSingleShot(True)
        self._frost_timer.setInterval(160)
        self._frost_timer.timeout.connect(self._refresh_frost)
        self._frost_generation = 0
        self._frost_worker: object = None

        # The landing lives for a few seconds; read the background setting
        # once instead of re-reading the config file on every paint.
        try:
            from gui.widgets.gradient_bg import _gradient_enabled
            self._gradient_on = bool(_gradient_enabled())
        except Exception:
            self._gradient_on = True

        self._compute_grid()
        self._build_cta()
        self._build_tiles()
        self._build_overlays()

    # ── Grid geometry ──────────────────────────────────────────────────────

    def _compute_grid(self) -> None:
        cw = max(60, (self._w - 2 * _PAD - (_COLS - 1) * _GAP) // _COLS)
        ch = max(60, (self._h - 2 * _PAD - (_ROWS - 1) * _GAP) // _ROWS)
        self._cell = QtCore.QSize(cw, ch)
        gw = _COLS * cw + (_COLS - 1) * _GAP
        gh = _ROWS * ch + (_ROWS - 1) * _GAP
        self._origin = QtCore.QPoint((self._w - gw) // 2, (self._h - gh) // 2)

    def _target_for(self, col: int, row: int) -> QtCore.QPoint:
        return QtCore.QPoint(
            self._origin.x() + col * (self._cell.width() + _GAP),
            self._origin.y() + row * (self._cell.height() + _GAP),
        )

    def _off_screen(self, target: QtCore.QPoint) -> QtCore.QPoint:
        """Return the off-screen start position for a tile (radially outward)."""
        tw, th = self._cell.width(), self._cell.height()
        tcx = target.x() + tw * 0.5
        tcy = target.y() + th * 0.5
        wcx, wcy = self._w * 0.5, self._h * 0.5
        dx, dy = tcx - wcx, tcy - wcy
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            dx, dy = 1.0, 0.0
        sx = (wcx + tw) / (abs(dx) + 0.001)
        sy = (wcy + th) / (abs(dy) + 0.001)
        scale = min(sx, sy) * 1.3
        return QtCore.QPoint(int(wcx + dx * scale - tw * 0.5), int(wcy + dy * scale - th * 0.5))

    def _scatter_target(self, target: QtCore.QPoint) -> QtCore.QPoint:
        """Return a randomised scatter destination beyond the window edges."""
        tw, th = self._cell.width(), self._cell.height()
        tcx = target.x() + tw * 0.5
        tcy = target.y() + th * 0.5
        wcx, wcy = self._w * 0.5, self._h * 0.5
        dx = (tcx - wcx) + random.uniform(-90, 90)
        dy = (tcy - wcy) + random.uniform(-90, 90)
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            dx, dy = 60.0, 60.0
        sx = (wcx + tw * 2.5) / (abs(dx) + 0.001)
        sy = (wcy + th * 2.5) / (abs(dy) + 0.001)
        scale = min(sx, sy) * 1.45
        return QtCore.QPoint(int(wcx + dx * scale - tw * 0.5), int(wcy + dy * scale - th * 0.5))

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_cta(self) -> None:
        self._cta = _CTACard(self, self._saved, self._missing)
        self._cta.adjustSize()
        size = self._cta.sizeHint().expandedTo(self._cta.minimumSizeHint())
        size.setWidth(_CARD_W)
        rect = QtCore.QRect(QtCore.QPoint(0, 0), size)
        rect.moveCenter(QtCore.QPoint(self._w // 2, self._h // 2))
        self._cta.setGeometry(rect)
        self._center_rect = rect
        self._cta.open_clicked.connect(self._on_open_clicked)
        self._cta.reuse_clicked.connect(self._on_reuse_clicked)

    def _build_tiles(self) -> None:
        pool = (_GRADS * 4)[: _COLS * _ROWS]
        random.shuffle(pool)
        card = self._center_rect
        i = 0
        for row in range(_ROWS):
            for col in range(_COLS):
                target = self._target_for(col, row)
                if card.contains(QtCore.QRect(target, self._cell)):
                    continue   # entirely behind the card: nobody would see it
                tile = _Tile(self, pool[i % len(pool)], target, self._cell)
                tile.move(self._off_screen(target))
                tile.show()
                self._tiles.append(tile)
                i += 1

        for tile in self._tiles:
            tile.clicked.connect(self._on_tile_clicked)
            tile.hovered.connect(self._on_tile_hovered)

        if self._saved:
            self._start_art_scanner()

    def _build_overlays(self) -> None:
        """Z-order: tiles < hover lift < scrim < card shadow < card < quit."""
        from gui.widgets.glass import ShadowLayer

        self._lift = _HoverLift(self)
        self._scrim = _Scrim(self)
        self._scrim.setGeometry(self.rect())
        self._scrim.show()
        self._scrim.raise_()
        self._cta.raise_()
        self._cta.show()
        # 0 30px 90px rgba(0,0,0,.55) under the card, over the tiles.
        self._cta_shadow = ShadowLayer(
            self._cta, Surface("", shadow="rgba(0,0,0,0.55)", blur=90, offset_y=30,
                               radius=_CARD_RADIUS))
        self._quit_btn = _QuitButton(self)
        self._quit_btn.move(self._w - 40 - 44, 40)
        self._quit_btn.clicked.connect(self._on_quit_clicked)
        self._quit_btn.show()
        self._quit_btn.raise_()

    # ── Public API ────────────────────────────────────────────────────────

    def cover_images(self) -> list[QtGui.QImage]:
        """The album art found so far (the main window's backdrop reuses it)."""
        return list(self._covers)

    def show_animated(self) -> None:
        """Fade the window in, pause on the card, then start the tile fly-in."""
        self.setWindowOpacity(0.0)
        self.show()
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(0 if _reduced_motion() else _FADE_IN_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(self._set_opacity)
        anim.finished.connect(self._on_faded_in)
        self._fade_in_anim = anim
        anim.start()
        self._start_counter()

    def _set_opacity(self, value) -> None:
        self.setWindowOpacity(float(value))

    def _on_faded_in(self) -> None:
        self.setWindowOpacity(1.0)
        if not self._exiting:
            self._fly_in_timer.start()
        self.faded_in.emit()

    # ── Fly-in ────────────────────────────────────────────────────────────

    def _fly_in(self) -> None:
        """Staggered fly-in: tiles arrive in random order from off-screen edges."""
        if self._exiting:
            return
        if _reduced_motion():
            for tile in self._tiles:
                tile.move(tile._target)
            self._schedule_frost()
            return
        order = list(range(len(self._tiles)))
        random.shuffle(order)

        grp = QtCore.QParallelAnimationGroup(self)
        for seq_i, tile_i in enumerate(order):
            tile = self._tiles[tile_i]
            seq  = QtCore.QSequentialAnimationGroup(grp)
            seq.addAnimation(QtCore.QPauseAnimation(seq_i * _STAGGER_MS))

            anim = QtCore.QPropertyAnimation(tile, b"pos", seq)
            anim.setStartValue(self._off_screen(tile._target))
            anim.setEndValue(tile._target)
            anim.setDuration(_FLY_IN_MS)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutBack)
            seq.addAnimation(anim)

            grp.addAnimation(seq)

        self._fly_in_grp = grp
        grp.finished.connect(self._schedule_frost)
        grp.start()

    # ── Frost (the card's pre-blurred backdrop) ───────────────────────────

    def _schedule_frost(self) -> None:
        if not self._exiting:
            self._frost_timer.start()

    def _tiles_moving(self) -> bool:
        grp = self._fly_in_grp
        return grp is not None and grp.state() == QtCore.QAbstractAnimation.State.Running

    def _refresh_frost(self) -> None:
        """Compose what is behind the card (tiles + scrim: a few blits) and
        blur it in a quiet worker; the card paints the result."""
        if self._exiting or not _tokens().material.is_glass or self._tiles_moving():
            return
        from gui.workers import cancel_thread, start_worker

        card = self._cta.geometry()
        img = QtGui.QImage(card.size(), QtGui.QImage.Format.Format_RGB32)
        img.fill(QtGui.QColor(_tokens().material.backdrop_base).darker(140))
        p = QtGui.QPainter(img)
        try:
            for tile in self._tiles:
                if tile.geometry().intersects(card):
                    p.drawPixmap(tile.pos() - card.topLeft(), tile.cover())
            p.drawPixmap(QtCore.QPoint(0, 0), self._scrim._pixmap(), card)
        finally:
            p.end()
        old = self._frost_worker
        if old is not None:
            cancel_thread(old)
        self._frost_generation += 1
        worker = _FrostWorker(img, self._frost_generation)
        worker.frosted.connect(self._on_frosted)
        self._frost_worker = worker
        start_worker(worker, owner=self, name="Landing frost", ask_before_quit=False, quiet=True)

    def _on_frosted(self, generation: int, image: QtGui.QImage) -> None:
        if generation == self._frost_generation and not self._exiting:
            self._frost_worker = None
            self._cta.set_frost(image)

    # ── User action handlers ───────────────────────────────────────────────

    def _on_reuse_clicked(self) -> None:
        self._accept(self._saved)

    def _on_quit_clicked(self) -> None:
        if self._exiting:
            return
        self.quit_requested.emit()
        self.close()

    def _on_open_clicked(self) -> None:
        if self._exiting:
            return
        start = self._saved or _nearest_existing_dir(self._missing) or str(Path.home())

        # Standalone top-level dialog (no parent) to avoid compositor
        # blank-outs on Linux; manually centred over this landing window.
        dlg = QtWidgets.QFileDialog()
        dlg.setWindowTitle("Select Music Library Folder")
        dlg.setDirectory(start)
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        dlg.resize(820, 560)
        if self.isVisible():
            fg = self.frameGeometry()
            dlg.move(
                fg.left() + (fg.width()  - dlg.width())  // 2,
                fg.top()  + (fg.height() - dlg.height()) // 2,
            )

        if dlg.exec():
            selected = dlg.selectedFiles()
            if selected:
                self._accept(selected[0])

    def _accept(self, path: str) -> None:
        """Start the exit sequence for *path*. Runs once; later calls (a second
        click, a tile click mid-scatter) are ignored."""
        if not path or self._exiting:
            return
        self._exiting = True
        self._pending = path

        # Cancel the pending fly-in (timer and animation). Tiles scatter from
        # wherever they are, so a click mid-fly-in never snaps them.
        self._fly_in_timer.stop()
        if (
            self._fly_in_grp is not None
            and self._fly_in_grp.state()
            == QtCore.QAbstractAnimation.State.Running
        ):
            self._fly_in_grp.stop()

        # The scanner's remaining covers would never be seen.
        self._stop_scanner()
        self._lift.drop()

        self._do_scatter()

    # ── Scatter / exit ────────────────────────────────────────────────────

    def _do_scatter(self) -> None:
        """Scatter tiles to random off-screen positions, then fade out."""
        if _reduced_motion():
            self._fade_out()
            return
        order = list(range(len(self._tiles)))
        random.shuffle(order)

        grp = QtCore.QParallelAnimationGroup(self)
        for tile_i in order:
            tile = self._tiles[tile_i]
            seq  = QtCore.QSequentialAnimationGroup(grp)
            seq.addAnimation(
                QtCore.QPauseAnimation(random.randint(0, _SCATTER_MAX))
            )
            anim = QtCore.QPropertyAnimation(tile, b"pos", seq)
            anim.setStartValue(tile.pos())
            anim.setEndValue(self._scatter_target(tile._target))
            anim.setDuration(_FLY_OUT_MS)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.InBack)
            seq.addAnimation(anim)
            grp.addAnimation(seq)

        self._fly_out_grp = grp
        grp.finished.connect(self._fade_out)
        grp.start()
        # Safety net: whatever happens to the scatter, the landing exits.
        self._exit_safety_timer.start(_SCATTER_MAX + _FLY_OUT_MS + 500)

    def _fade_out(self) -> None:
        if self._fading_out:
            return
        self._fading_out = True
        self._exit_safety_timer.stop()
        if self._fade_in_anim is not None:
            try:
                self._fade_in_anim.stop()
            except RuntimeError:
                pass
        # Emit path now so the caller can start fading the main window in,
        # creating a simultaneous cross-dissolve while we fade out.
        self.library_selected.emit(self._pending)

        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(float(self.windowOpacity()))
        anim.setEndValue(0.0)
        anim.setDuration(0 if _reduced_motion() else FADE_OUT_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        anim.valueChanged.connect(self._set_opacity)
        anim.finished.connect(self._done)
        self._fade_out_anim = anim
        anim.start()

    def _done(self) -> None:
        self._stop_scanner()
        self.hide()
        self.finished.emit()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Alt+F4 / ✕ on the landing: stop the workers so quitting never
        destroys a running thread. (The app quits; ``alpha_dex_gui.main``
        waits for the JobRegistry before returning.)"""
        self._fly_in_timer.stop()
        self._exit_safety_timer.stop()
        self._stop_scanner()
        super().closeEvent(event)

    # ── Tile click / hover ────────────────────────────────────────────────

    def _on_tile_clicked(self, dirpath: str) -> None:
        """User clicked an album tile — begin the exit sequence and signal
        the caller to navigate to the player and start playing *dirpath*."""
        if not dirpath or self._exiting:
            return
        self.tile_clicked.emit(dirpath)
        # Reuse the same exit flow as clicking Initialize.
        # If no saved library is set we have no library to open, so guard.
        if self._saved:
            self._accept(self._saved)

    def _on_tile_hovered(self, tile: _Tile, entering: bool) -> None:
        if self._exiting:
            return
        if entering:
            self._lift.show_for(tile)
            self._scrim.raise_()
            if self._cta_shadow.isVisible():
                self._cta_shadow.raise_()
            self._cta.raise_()
            self._quit_btn.raise_()
        else:
            self._lift.hide_for(tile)

    # ── Workers ───────────────────────────────────────────────────────────

    def _start_art_scanner(self) -> None:
        from gui.workers import start_worker

        scanner = _ArtScanner(self._saved, len(self._tiles),
                              self._cell.width(), self._cell.height())
        scanner.art_found.connect(self._on_art_found)
        self._scanner = scanner
        # The JobRegistry keeps the thread alive until it really finishes, so
        # dropping our reference (or closing the landing) mid-scan is safe.
        start_worker(scanner, owner=self, name="Album-art scan", ask_before_quit=False)

    def _start_counter(self) -> None:
        if not self._saved or self._counter is not None or self._cta.folder_row is None:
            return
        from gui.workers import start_worker

        counter = _CountWorker(self._saved)
        counter.counted.connect(self._cta.set_stats)
        self._counter = counter
        start_worker(counter, owner=self, name="Library count", ask_before_quit=False)

    def _stop_scanner(self) -> None:
        """Ask the workers to stop and forget them: never block, never terminate."""
        scanner = self._scanner
        self._scanner = None
        if scanner is not None:
            try:
                scanner.art_found.disconnect(self._on_art_found)
            except (RuntimeError, TypeError):
                pass
            try:
                scanner.cancel()
            except RuntimeError:
                pass
        from gui.workers import cancel_thread
        for attr in ("_counter", "_frost_worker"):
            worker = getattr(self, attr, None)
            setattr(self, attr, None)
            if worker is not None:
                cancel_thread(worker)
        self._frost_timer.stop()

    def _on_art_found(self, tile_index: int, image: QtGui.QImage, dirpath: str,
                      album: str = "", artist: str = "") -> None:
        # We are back on the main thread here; QPixmap conversion is safe.
        if 0 <= tile_index < len(self._tiles):
            pm = QtGui.QPixmap.fromImage(image)
            if not pm.isNull():
                tile = self._tiles[tile_index]
                tile.set_pixmap(pm)
                tile.set_dirpath(dirpath)
                tile.set_info(album, artist)
                self._covers.append(image)
                if self._lift.tile() is tile:
                    self._lift.update()
                if tile.geometry().intersects(self._cta.geometry()):
                    self._schedule_frost()

    # ── Background painting ────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            from gui.widgets.gradient_bg import paint_window_gradient
            t = _tokens()
            base = QtGui.QColor(t.material.backdrop_base)
            if t.material.is_glass:
                base = base.darker(140)
            p.fillRect(event.rect(), base)
            if self._gradient_on and not t.material.is_glass:
                paint_window_gradient(p, self, t, radius_scale=0.85, alpha_tr=55, alpha_bl=70)
        finally:
            p.end()
