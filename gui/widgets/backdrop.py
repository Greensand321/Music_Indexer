"""The window backdrop: the library's own album covers as a blurred wall.

Liquid Glass sits on this. Everything translucent in the window (sidebar,
cards, header pills, the capsule) reads as glass because what is behind it is
already blurred and static — nothing is blurred at paint time.

Modes (Settings → Appearance → Backdrop, ``gui.themes.appearance``):

* **Album art** — a wall of covers from the active library (8 columns, gap
  18, radius 24, bleeding 80 px past the edges), blurred ~46 px, saturated
  150 %, at 95 % over the base, under an ``rgba(8,10,18,.58)`` scrim.
  Falls back to the gradient when the library has no covers (or no library
  is open yet), and to plain while the first build runs.
* **Gradient** — soft accent glows over the base (classic themes: their
  original two corner glows).
* **Plain** — the base colour.

Classic themes have no album-art wall: "Album art" shows their gradient.

Performance contract (it must never stall a frame):

* the image is composed **off the GUI thread** in ``BackdropWorker`` (a
  read-only, quiet JobRegistry job) with ``QImage`` only: the wall is drawn
  at quarter scale, blurred there with a 3-pass box blur (numpy; a close
  Gaussian), saturated, scrimmed, smoothly upscaled and dithered;
* results are cached per (library, size bucket, mode, theme colours):
  covers per library, finished pixmaps in a small LRU;
* ``paintEvent`` only blits the cached pixmap (cropped to the dirty rect)
  plus the 9-slice shadows of the floating shell surfaces (the sidebar);
* rebuilds are debounced on resize (and only happen when the size *bucket*
  changes), and follow library, theme, accent and Settings changes.

Covers come from the landing's scanner (``gui.widgets.landing._CoverScan``):
the landing hands what it found to the window (``seed_covers``), so the wall
behind the main window is the albums you just saw.
"""
from __future__ import annotations

import collections
import math
import random
import zlib
from dataclasses import dataclass

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.manager import get_manager
from gui.widgets.landing import _CoverScan

# Window sizes are rounded up to this, so small resizes reuse the image.
BUCKET = 128
# Covers kept per library for the wall (square, px).
COVER_PX = 96
COVER_COUNT = 24
# The wall is composed at 1/SCALE of the window size, then upscaled.
SCALE = 4
_PIXMAP_CACHE_SIZE = 3


@dataclass(frozen=True)
class BackdropSpec:
    """Everything a backdrop image depends on (and its cache key)."""
    mode: str            # "art" | "gradient" | "plain" (effective, see appearance)
    library: str
    width: int           # size bucket, device-independent px
    height: int
    dpr: float
    glass: bool
    variant: str         # "dark" | "light"
    base: str
    scrim: str
    accent: str
    accent_hover: str
    accent_pressed: str
    columns: int = 8
    gap: int = 18
    radius: int = 24
    blur_px: int = 46
    saturation: float = 1.5
    art_opacity: float = 0.95
    bleed: int = 80


def bucket(size: QtCore.QSize) -> tuple[int, int]:
    w = max(BUCKET, int(math.ceil(max(1, size.width()) / BUCKET)) * BUCKET)
    h = max(BUCKET, int(math.ceil(max(1, size.height()) / BUCKET)) * BUCKET)
    return w, h


def spec_for(tokens, mode: str, library: str, size: QtCore.QSize, dpr: float) -> BackdropSpec:
    from gui.themes.appearance import effective_mode

    m = tokens.material
    w, h = bucket(size)
    eff = effective_mode(tokens, mode)
    return BackdropSpec(
        mode=eff, library=library if eff == "art" else "", width=w, height=h,
        dpr=round(float(dpr), 2), glass=m.is_glass, variant=tokens.variant,
        base=m.backdrop_base, scrim=m.backdrop_scrim, accent=tokens.accent,
        accent_hover=tokens.accent_hover, accent_pressed=tokens.accent_pressed,
        columns=m.wall_columns, gap=m.wall_gap, radius=m.wall_radius, blur_px=m.blur_px,
        saturation=m.saturation, art_opacity=m.art_opacity, bleed=m.bleed,
    )


# ── Caches (GUI thread only) ──────────────────────────────────────────────────

_COVERS: dict[str, list[QtGui.QImage]] = {}
_PIXMAPS: "collections.OrderedDict[BackdropSpec, QtGui.QPixmap]" = collections.OrderedDict()


def cached_covers(library: str) -> list[QtGui.QImage] | None:
    return _COVERS.get(library)


def remember_covers(library: str, covers: list[QtGui.QImage]) -> None:
    if library and covers:
        _COVERS[library] = list(covers)


def _cache_pixmap(spec: BackdropSpec, pm: QtGui.QPixmap) -> None:
    _PIXMAPS[spec] = pm
    _PIXMAPS.move_to_end(spec)
    while len(_PIXMAPS) > _PIXMAP_CACHE_SIZE:
        _PIXMAPS.popitem(last=False)


def clear_caches() -> None:
    _COVERS.clear()
    _PIXMAPS.clear()


# ── Composition (worker thread: QImage + numpy only) ──────────────────────────

from gui.imaging import (  # noqa: E402  (the shared, worker-safe image helpers)
    _np, box_radii as _box_radii, from_array as _from_array, gaussian_blur,
    saturate as _saturate, to_array as _to_array,
)


def _rgb(colour: str) -> tuple[float, float, float, float]:
    from gui.themes.tokens import parse_colour
    r, g, b, a = parse_colour(colour)
    return float(r), float(g), float(b), float(a)


def _finish(small: QtGui.QImage, spec: BackdropSpec) -> QtGui.QImage:
    """Upscale a small composed image to the device size and dither it."""
    np = _np()
    dw = max(1, int(round(spec.width * spec.dpr)))
    dh = max(1, int(round(spec.height * spec.dpr)))
    big = small.scaled(dw, dh, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                       QtCore.Qt.TransformationMode.SmoothTransformation)
    if np is not None:
        arr = _to_array(big)
        rng = np.random.default_rng(7)
        noise = (rng.random((dh, dw, 1), dtype=np.float32)
                 - rng.random((dh, dw, 1), dtype=np.float32))
        big = _from_array(arr + noise)   # ±1 LSB triangular dither: no banding
    big.setDevicePixelRatio(spec.dpr)
    return big


def _compose_art(spec: BackdropSpec, covers: list[QtGui.QImage]) -> QtGui.QImage:
    np = _np()
    k = float(SCALE)
    full_w = spec.width + 2 * spec.bleed
    full_h = spec.height + 2 * spec.bleed
    sw, sh = int(math.ceil(full_w / k)), int(math.ceil(full_h / k))
    canvas = QtGui.QImage(sw, sh, QtGui.QImage.Format.Format_RGB32)
    canvas.fill(QtGui.QColor(spec.base))
    tile = (full_w - (spec.columns - 1) * spec.gap) / spec.columns
    rows = int(math.ceil((full_h + spec.gap) / (tile + spec.gap)))
    # The same library always lays its wall out the same way (a resize or a
    # theme switch must not reshuffle it).
    rng = random.Random(zlib.crc32(spec.library.encode("utf-8", "replace")))
    order = list(range(len(covers)))
    rng.shuffle(order)
    p = QtGui.QPainter(canvas)
    try:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        ts, gap, rad = tile / k, spec.gap / k, spec.radius / k
        i = 0
        for row in range(rows):
            for col in range(spec.columns):
                img = covers[order[i % len(order)]]
                i += 1
                rect = QtCore.QRectF(col * (ts + gap), row * (ts + gap), ts, ts)
                path = QtGui.QPainterPath()
                path.addRoundedRect(rect, rad, rad)
                p.save()
                p.setClipPath(path)
                side = min(img.width(), img.height())
                src = QtCore.QRectF((img.width() - side) / 2, (img.height() - side) / 2, side, side)
                p.drawImage(rect, img, src)
                p.restore()
    finally:
        p.end()

    if np is None:  # pragma: no cover - fallback: blur by down/up-scaling
        tiny = canvas.scaled(max(1, sw // 8), max(1, sh // 8),
                             QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
        small = tiny.scaled(sw, sh, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation)
    else:
        arr = _to_array(canvas)
        arr = gaussian_blur(arr, spec.blur_px / k)
        arr = _saturate(arr, spec.saturation)
        br, bg, bb, _ = _rgb(spec.base)
        base = np.array([bb, bg, br], dtype=np.float32)
        arr = arr * spec.art_opacity + base * (1.0 - spec.art_opacity)
        if spec.scrim:
            r, g, b, a = _rgb(spec.scrim)
            arr = arr * (1.0 - a) + np.array([b, g, r], dtype=np.float32) * a
        bx, by = int(round(spec.bleed / k)), int(round(spec.bleed / k))
        cw, ch = int(round(spec.width / k)), int(round(spec.height / k))
        arr = arr[by:by + ch, bx:bx + cw]
        small = _from_array(np.ascontiguousarray(arr))
    return _finish(small, spec)


def _glow(p: QtGui.QPainter, x: float, y: float, radius: float, colour: str, alpha: float,
          vivid: float = 1.0) -> None:
    c = QtGui.QColor(colour)
    if vivid != 1.0:
        # Pastel accents make grey glows; push their saturation for the wash.
        h, s, v, _a = c.getHsvF()
        c.setHsvF(max(0.0, h), min(1.0, s * vivid), v * 0.92)
    on, off = QtGui.QColor(c), QtGui.QColor(c)
    on.setAlphaF(alpha)
    off.setAlphaF(0.0)
    g = QtGui.QRadialGradient(QtCore.QPointF(x, y), radius)
    g.setColorAt(0.0, on)
    g.setColorAt(1.0, off)
    p.fillRect(QtCore.QRectF(0, 0, p.device().width(), p.device().height()), QtGui.QBrush(g))


def _compose_gradient(spec: BackdropSpec) -> QtGui.QImage:
    if spec.glass:
        # Liquid Glass: soft accent glows over the deep base, scrimmed like art.
        sw = int(math.ceil(spec.width / SCALE))
        sh = int(math.ceil(spec.height / SCALE))
        img = QtGui.QImage(sw, sh, QtGui.QImage.Format.Format_RGB32)
        img.fill(QtGui.QColor(spec.base))
        big = max(sw, sh)
        p = QtGui.QPainter(img)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            _glow(p, sw * 0.86, sh * 0.06, big * 0.80, spec.accent, 0.42, vivid=2.4)
            _glow(p, sw * 0.08, sh * 0.98, big * 0.72, spec.accent_pressed, 0.30, vivid=2.0)
            _glow(p, sw * 0.46, sh * 0.50, big * 0.50, spec.accent_hover, 0.12, vivid=1.6)
            if spec.scrim:
                r, g, b, a = _rgb(spec.scrim)
                p.fillRect(img.rect(), QtGui.QColor(int(r), int(g), int(b), int(a * 255 * 0.45)))
        finally:
            p.end()
        return _finish(img, spec)
    # Classic: the original two corner glows (accent top-right, accent_hover
    # bottom-left) over the page colour, drawn at full size.
    dw = max(1, int(round(spec.width * spec.dpr)))
    dh = max(1, int(round(spec.height * spec.dpr)))
    img = QtGui.QImage(dw, dh, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(spec.base))
    radius = max(dw, dh) * 0.85
    dark = spec.variant == "dark"
    p = QtGui.QPainter(img)
    try:
        _glow(p, dw, 0.0, radius * 0.90, spec.accent, (22 if dark else 14) / 255)
        _glow(p, 0.0, dh, radius * 0.72, spec.accent_hover, (15 if dark else 9) / 255)
    finally:
        p.end()
    img.setDevicePixelRatio(spec.dpr)
    return img


def compose(spec: BackdropSpec, covers: list[QtGui.QImage] | None) -> QtGui.QImage | None:
    """The backdrop image for ``spec`` (None for "plain"). Worker thread only:
    QImage and numpy, never QPixmap."""
    if spec.mode == "plain":
        return None
    if spec.mode == "art" and covers:
        return _compose_art(spec, covers)
    return _compose_gradient(spec)


# ── Worker ────────────────────────────────────────────────────────────────────

class BackdropWorker(_CoverScan, QtCore.QThread):
    """Finds covers (when needed) and composes the backdrop, off the GUI thread."""

    # generation, spec, image (null for plain), covers found for spec.library
    built = Signal(int, object, QtGui.QImage, list)

    def __init__(self, spec: BackdropSpec, generation: int,
                 covers: list[QtGui.QImage] | None) -> None:
        QtCore.QThread.__init__(self)
        self._init_cover_scan(spec.library, COVER_COUNT, tile_w=COVER_PX, radius=0,
                              vignette=False, use_history=False)
        self._spec = spec
        self._generation = generation
        self._covers = list(covers) if covers is not None else None
        self.setObjectName("BackdropWorker")

    def run(self) -> None:  # noqa: N802
        spec = self._spec
        covers = self._covers
        if spec.mode == "art" and covers is None and spec.library:
            found: dict[int, QtGui.QImage] = {}

            def collect(_i, img, _d, _a, _ar) -> None:
                found.setdefault(img.cacheKey(), img)

            self.scan(collect)
            covers = list(found.values())
        if self.isInterruptionRequested():
            return
        image = compose(spec, covers)
        if self.isInterruptionRequested():
            return
        self.built.emit(self._generation, spec, image or QtGui.QImage(), list(covers or []))


# ── Widget ────────────────────────────────────────────────────────────────────

class Backdrop(QtWidgets.QWidget):
    """The main window's central widget: paints the backdrop, and the
    shadows of the floating shell surfaces registered with ``add_floating``
    (the sidebar), beneath everything else."""

    rebuilt = Signal(object)   # the BackdropSpec now shown

    DEBOUNCE_MS = 140

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("backdrop")
        self.setAutoFillBackground(False)
        # Every pixel is painted here: Qt need not paint anything beneath.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._library = ""
        self._spec: BackdropSpec | None = None     # what is shown
        self._pixmap: QtGui.QPixmap | None = None
        self._old: QtGui.QPixmap | None = None       # fading out (None: the base colour)
        self._fading = False
        self._fade = 1.0
        self._generation = 0
        self._worker: BackdropWorker | None = None
        self._floating: list[tuple[QtWidgets.QWidget, str]] = []
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.rebuild)
        get_manager().theme_changed.connect(self._on_theme_changed)

    # ── Public API ────────────────────────────────────────────────────────

    def library(self) -> str:
        return self._library

    def set_library(self, path: str) -> None:
        if path == self._library:
            return
        self._library = path
        self.schedule()

    def seed_covers(self, library: str, covers: list[QtGui.QImage]) -> None:
        """Covers the landing already found for ``library`` (no second scan)."""
        if covers and library and cached_covers(library) is None:
            remember_covers(library, covers)

    def on_appearance_changed(self) -> None:
        """Settings saved (backdrop mode may have changed)."""
        self.schedule(0)

    def add_floating(self, widget: QtWidgets.QWidget, surface: str) -> None:
        """Paint ``widget``'s shadow (``surface`` of the material) here."""
        self._floating.append((widget, surface))
        widget.installEventFilter(self)

    def spec(self) -> BackdropSpec | None:
        return self._spec

    def pixmap(self) -> QtGui.QPixmap | None:
        return self._pixmap

    def is_building(self) -> bool:
        w = self._worker
        try:
            return w is not None and w.isRunning()
        except RuntimeError:
            return False

    def schedule(self, delay: int | None = None) -> None:
        self._timer.start(self.DEBOUNCE_MS if delay is None else delay)

    def desired_spec(self) -> BackdropSpec:
        from gui.themes import appearance
        return spec_for(get_manager().current, appearance.backdrop_mode(), self._library,
                        self.size(), self.devicePixelRatioF())

    def rebuild(self) -> None:
        """Show the backdrop for the current state: from the cache at once,
        or by starting a worker (the base colour shows meanwhile)."""
        self._timer.stop()
        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            return
        spec = self.desired_spec()
        if spec == self._spec and (self._pixmap is not None or spec.mode == "plain"):
            return
        if spec.mode == "plain":
            self._cancel_worker()
            self._show(spec, None, fade=False)
            return
        cached = _PIXMAPS.get(spec)
        if cached is not None:
            self._cancel_worker()
            _PIXMAPS.move_to_end(spec)
            self._show(spec, cached, fade=False)
            return
        if self._worker is not None and getattr(self._worker, "_spec", None) == spec:
            return   # already building exactly this
        self._start_worker(spec)

    # ── Worker plumbing ───────────────────────────────────────────────────

    def _start_worker(self, spec: BackdropSpec) -> None:
        from gui.workers import start_worker

        self._cancel_worker()
        self._generation += 1
        covers = cached_covers(spec.library) if spec.mode == "art" else []
        worker = BackdropWorker(spec, self._generation, covers)
        worker.built.connect(self._on_built)
        self._worker = worker
        start_worker(worker, owner=self, name="Backdrop", ask_before_quit=False, quiet=True)

    def _cancel_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            try:
                worker.built.disconnect(self._on_built)
            except (RuntimeError, TypeError):
                pass
            from gui.workers import cancel_thread
            cancel_thread(worker)

    def _on_built(self, generation: int, spec: BackdropSpec, image: QtGui.QImage,
                  covers: list) -> None:
        if generation != self._generation:
            return   # superseded
        self._worker = None
        if spec.mode == "art" and covers:
            remember_covers(spec.library, covers)
        if image.isNull():
            self._show(spec, None, fade=False)
            return
        pm = QtGui.QPixmap.fromImage(image)
        _cache_pixmap(spec, pm)
        # Fade only when art (or a first image) replaces something; theme and
        # size changes swap at once like the rest of the window.
        fade = self._spec is None or self._pixmap is None or (
            self._spec.library != spec.library and spec.mode == "art")
        self._show(spec, pm, fade=fade)

    def _show(self, spec: BackdropSpec, pm: QtGui.QPixmap | None, *, fade: bool) -> None:
        from gui.themes import motion

        old = self._pixmap
        self._spec = spec
        self._pixmap = pm
        if fade and pm is not None and motion.duration("slow") > 0:
            self._old = old
            self._fading = True
            self._fade = 0.0
            motion.animate_value(self, "swap", 0.0, 1.0, self._on_fade,
                                 duration="slow", easing="standard",
                                 on_finished=self._on_fade_done)
        else:
            motion.stop(self, "swap")
            self._old = None
            self._fading = False
            self._fade = 1.0
            self.update()
        self.rebuilt.emit(spec)

    def _on_fade(self, value: float) -> None:
        self._fade = float(value)
        self.update()

    def _on_fade_done(self) -> None:
        self._old = None
        self._fading = False
        self._fade = 1.0
        self.update()

    # ── Events ────────────────────────────────────────────────────────────

    def _on_theme_changed(self, _tokens=None) -> None:
        if self.isVisible():
            self.rebuild()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self.schedule(0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        spec = self._spec
        if spec is None or bucket(self.size()) != (spec.width, spec.height):
            self.schedule()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if event.type() in (QtCore.QEvent.Type.Move, QtCore.QEvent.Type.Resize,
                            QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Hide):
            self.update()
        return False

    # ── Painting: blits only ──────────────────────────────────────────────

    def _blit(self, p: QtGui.QPainter, pm: QtGui.QPixmap, r: QtCore.QRect) -> None:
        dpr = pm.devicePixelRatio() or 1.0
        pw, ph = pm.width() / dpr, pm.height() / dpr
        w, h = self.width(), self.height()
        if pw >= w and ph >= h:
            ox, oy = (pw - w) / 2.0, (ph - h) / 2.0
            src = QtCore.QRectF((r.x() + ox) * dpr, (r.y() + oy) * dpr,
                                r.width() * dpr, r.height() * dpr)
            p.drawPixmap(QtCore.QRectF(r), pm, src)
        else:
            # Grown past the cached bucket: stretch until the rebuild lands.
            p.save()
            p.setClipRect(r)
            p.drawPixmap(QtCore.QRectF(self.rect()), pm, QtCore.QRectF(pm.rect()))
            p.restore()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_shadow

        t = get_manager().current
        r = event.rect()
        p = QtGui.QPainter(self)
        try:
            if self._pixmap is None:
                p.fillRect(r, QtGui.QColor(t.material.backdrop_base))
            else:
                self._blit(p, self._pixmap, r)
            if self._fading and self._fade < 1.0:
                # Cross-fade: what was shown before, fading out on top.
                p.setOpacity(1.0 - self._fade)
                if self._old is not None:
                    self._blit(p, self._old, r)
                else:
                    p.fillRect(r, QtGui.QColor(t.material.backdrop_base))
                p.setOpacity(1.0)
            for widget, surface in self._floating:
                if not widget.isVisible():
                    continue
                geo = QtCore.QRect(widget.mapTo(self, QtCore.QPoint(0, 0)), widget.size())
                paint_shadow(p, geo, t.material.surface(surface))
        finally:
            p.end()
