"""Player workspace — in-app audio player: library browser, queue, M3U.

Playback goes through a :class:`gui.playback_backend.PlaybackBackend`
(libVLC today). When no backend can be loaded the page says so and the
transport is disabled; it never shows a fake "playing" state.

Model (unchanged from the original design, see docs/redesign/pages/07_player.md):

* **Direct play** (double-click, "Play now", Browse…, Recently Played,
  keyboard-mode library navigation) plays one track outside the queue
  (``_direct_row``). When it ends, playback hands over to the queue.
* **Queue play**: ``_queue_index`` is the queue *cursor* and
  ``_cursor_played`` says whether the cursor's track has been started yet.
  A freshly built queue has its cursor on item 0, not yet played, so Play
  (or the end of a direct track) starts item 0 rather than skipping it.

Every background job (library scan, tag reads, cover art) runs through
``WorkspaceBase._start_worker`` and is cancellable; stale results are
dropped by generation/token checks, never by waiting on the GUI thread.
"""
from __future__ import annotations

import os
import random
import threading
from pathlib import Path

from gui import playback_backend
from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.playback_backend import ACTIVE_STATES, PlaybackBackend, PlaybackState
from gui.themes.manager import get_manager
from gui.widgets.glass import ElidedLabel, GlassCard, GlassPanel
from gui.widgets.glass_parts import EmptyState
from gui.widgets.glyphs import glyph_icon, set_glyph
from gui.widgets.gradient_bg import GradientWidget
from gui.workspaces.base import WorkspaceBase
from utils.reserved_folders import NON_LIBRARY_FOLDERS, walk_library

_AUDIO_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
_ART_SIZE = 200

# Repeat modes (cycled Off → Track → All) and their names on the capsule
_REPEAT_OFF   = 0
_REPEAT_TRACK = 1
_REPEAT_ALL   = 2
_REPEAT_NAMES = {_REPEAT_OFF: "off", _REPEAT_TRACK: "track", _REPEAT_ALL: "all"}

# 30-second highlight: seek to this position, auto-advance after duration
_HIGHLIGHT_START_MS    = 30_000
_HIGHLIGHT_DURATION_MS = 15_000
# VLC ignores seeks/volume until the stream is open; wait this long first.
_PREVIEW_SEEK_DELAY_MS = 300
_VOLUME_REAPPLY_MS     = 250

# A landing-page tile may point at an artist folder; look this deep for audio.
_DIR_PLAY_MAX_DEPTH = 3
_DIR_PLAY_MAX_FILES = 2000

_PENDING = "_pending"   # row-dict key: tags not read yet (placeholder row)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_row(path: str) -> dict:
    """Read tags for one audio file and return a row dict.  Never raises.

    Opens the file once with Mutagen, passes the pre-loaded object to
    read_metadata_from_mutagen for tags, and reads .info.length from the
    same object for duration — avoiding the previous double-open pattern.
    """
    title = artist = album = dur_str = track_num = ""
    track_int = 0
    dur_s = -1
    try:
        from mutagen import File as _MF  # type: ignore
        audio = _MF(path)
    except Exception:
        audio = None

    try:
        from utils.audio_metadata_reader import read_metadata_from_mutagen
        tags, _, _, _ = read_metadata_from_mutagen(audio, path, include_cover=False)
        title  = str(tags.get("title")  or "")
        artist = str(tags.get("artist") or "")
        album  = str(tags.get("album")  or "")
        tn = tags.get("track") or tags.get("tracknumber")
        if tn is not None:
            try:
                track_int = int(str(tn).split("/")[0])
                track_num = str(track_int)
            except Exception:
                pass
    except Exception:
        pass

    if not title:
        title = Path(path).stem

    length = getattr(getattr(audio, "info", None), "length", None) if audio else None
    if length:
        try:
            dur_s = int(length)
            m, s = divmod(dur_s, 60)
            dur_str = f"{m}:{s:02d}"
        except Exception:
            dur_s = -1

    return {
        "track_num": track_num,
        "title":     title,
        "artist":    artist,
        "album":     album,
        "dur":       dur_str,
        "path":      path,
        "track_int": track_int,
        "dur_s":     dur_s,
    }


def _placeholder_row(path: str) -> dict:
    """A row shown until the real tags arrive from a worker."""
    return {
        "track_num": "", "title": Path(path).stem, "artist": "", "album": "",
        "dur": "", "path": path, "track_int": 0, "dur_s": -1, _PENDING: True,
    }


def _text_key(s: str) -> tuple[bool, str]:
    """Sort key that puts empty values last."""
    s = (s or "").casefold()
    return (s == "", s)


def _sort_keys(row: dict) -> list[tuple]:
    """Per-column sort keys for the library table (# Title Artist Album Time)."""
    t = int(row.get("track_int") or 0)
    title, artist, album = (_text_key(row.get(k, "")) for k in ("title", "artist", "album"))
    dur = int(row.get("dur_s", -1))
    return [
        (t, artist, album, title),
        (title, artist, album),
        (artist, album, t, title),
        (album, t, title),
        (dur, title),
    ]


def _decode_m3u(data: bytes, path: str) -> str:
    """Decode playlist bytes: ``.m3u8`` is UTF-8; ``.m3u`` is often cp1252/Latin-1."""
    encodings = ("utf-8-sig",) if path.lower().endswith(".m3u8") else ("utf-8-sig", "cp1252", "latin-1")
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


#: Sidecar images next to a track, used when it has no embedded art (the
#: same names, in the same order, as the landing's album tiles).
_FOLDER_ART = (
    "cover.jpg", "folder.jpg", "album.jpg", "artwork.jpg",
    "front.jpg", "cover.png", "folder.png", "album.png",
)


def _folder_art_bytes(path: str) -> bytes | None:
    """The first sidecar cover in ``path``'s folder (worker threads only)."""
    folder = os.path.dirname(path)
    for name in _FOLDER_ART:
        candidate = os.path.join(folder, name)
        try:
            if os.path.isfile(candidate):
                with open(candidate, "rb") as fh:
                    data = fh.read()
                if data:
                    return data
        except OSError:
            continue
    return None


def _tokens():
    return get_manager().current


def _rounded(pm: QtGui.QPixmap, size: int, radius: float) -> QtGui.QPixmap:
    """``pm`` centre-cropped to a ``size`` square with rounded corners."""
    dpr = pm.devicePixelRatio() or 1.0
    px = max(1, int(round(size * dpr)))
    scaled = pm.scaled(px, px, QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                       QtCore.Qt.TransformationMode.SmoothTransformation)
    out = QtGui.QPixmap(px, px)
    out.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(out)
    try:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(0, 0, px, px), radius * dpr, radius * dpr)
        p.setClipPath(path)
        p.drawPixmap(-(scaled.width() - px) // 2, -(scaled.height() - px) // 2, scaled)
    finally:
        p.end()
    out.setDevicePixelRatio(dpr)
    return out


def _tile_surface(radius: int):
    """An album tile: a faint well (only seen while there is no art), the
    mockup's ``0 14 30 rgba(0,0,0,.4)`` shadow (the capsule's shadow colour
    under every theme) and an inset 1 px ring in the card border colour."""
    from gui.themes.glass import current_material
    from gui.themes.tokens import Surface

    m = current_material()
    return Surface(m.well.fill, m.card.border, "", m.capsule.shadow or m.card.shadow,
                   blur=30, offset_y=14, radius=radius)


def _paint_tile(p: QtGui.QPainter, rect: QtCore.QRectF, pm: QtGui.QPixmap | None,
                radius: float, *, shadow: bool = True) -> None:
    """A glass album tile: painted shadow, the art (or a music glyph on the
    well), and the inset ring on top."""
    from gui.themes.glass import paint_shadow, qcolor
    from gui.widgets.glyphs import paint_glyph

    spec = _tile_surface(int(radius))
    if shadow:
        paint_shadow(p, rect, spec, radius=radius)
    path = QtGui.QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    p.save()
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
    if pm is not None and not pm.isNull():
        p.setClipPath(path)
        side = int(round(rect.width()))
        scaled = pm.scaled(side, side, QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                           QtCore.Qt.TransformationMode.SmoothTransformation)
        p.drawPixmap(QtCore.QPointF(rect.x() - (scaled.width() - side) / 2,
                                    rect.y() - (scaled.height() - side) / 2), scaled)
        p.setClipping(False)
    else:
        p.fillPath(path, qcolor(spec.fill))
        g = rect.width() * 0.34
        c = rect.center()
        paint_glyph(p, QtCore.QRectF(c.x() - g / 2, c.y() - g / 2, g, g), "music",
                    qcolor(_tokens().text_muted))
    p.setPen(QtGui.QPen(qcolor(spec.border), 1.0))
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius - 0.5, radius - 0.5)
    p.restore()


class _SortItem(QtWidgets.QTableWidgetItem):
    """Table item that sorts by a typed key (numbers numerically, ties broken
    by artist/album/track) instead of by its display text."""

    def __init__(self, text: str, key: tuple) -> None:
        super().__init__(text)
        self._key = key

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:  # noqa: D105
        other_key = getattr(other, "_key", None)
        if other_key is None:
            return super().__lt__(other)
        return self._key < other_key


# ─────────────────────────────────────────────────────────────────────────────
# Background workers (all started through WorkspaceBase._start_worker)
# ─────────────────────────────────────────────────────────────────────────────

class _CancellableThread(QtCore.QThread):
    def __init__(self) -> None:
        super().__init__()
        self._cancel_evt = threading.Event()

    def cancel(self) -> None:
        self._cancel_evt.set()

    def _stopped(self) -> bool:
        return self._cancel_evt.is_set() or self.isInterruptionRequested()


class _LibraryScanner(_CancellableThread):
    """Walk a library folder and emit batches of metadata rows with auto-throttling.

    Every signal carries the scan's ``generation`` so the workspace can drop
    batches from a scan it has since replaced. A cancelled scan emits nothing.
    """

    tracks_ready  = Signal(int, list)   # generation, list[dict]
    scan_complete = Signal(int, int)    # generation, total count

    def __init__(self, library_path: str, generation: int = 0) -> None:
        super().__init__()
        self._path = library_path
        self.generation = generation
        self._ui_latency_ms = 0
        self._current_batch_size = 50
        self._sleep_ms = 0

    def report_ui_latency(self, elapsed_ms: int) -> None:
        self._ui_latency_ms = elapsed_ms

    def run(self) -> None:
        batch: list[dict] = []
        total = 0
        try:
            # Playlists/Docs/Trash/Quarantine are not music; Manual Review and
            # Not Sorted hold real tracks, so they stay visible.
            for root, dirs, files in walk_library(self._path, NON_LIBRARY_FOLDERS):
                dirs.sort()
                if self._stopped():
                    return
                for fname in sorted(files):
                    if self._stopped():
                        return
                    if os.path.splitext(fname)[1].lower() not in _AUDIO_EXTS:
                        continue
                    batch.append(_load_row(os.path.join(root, fname)))
                    total += 1

                    if len(batch) >= self._current_batch_size:
                        self.tracks_ready.emit(self.generation, list(batch))
                        batch.clear()

                        latency = self._ui_latency_ms
                        if latency > 15:
                            self._current_batch_size = max(10, self._current_batch_size - 10)
                            self._sleep_ms = min(50, self._sleep_ms + 5)
                        elif latency < 5:
                            self._current_batch_size = min(500, self._current_batch_size + 20)
                            self._sleep_ms = max(0, self._sleep_ms - 2)

                        if self._sleep_ms > 0:
                            self.msleep(self._sleep_ms)
        finally:
            if not self._stopped():
                if batch:
                    self.tracks_ready.emit(self.generation, batch)
                self.scan_complete.emit(self.generation, total)


class _TagLoader(_CancellableThread):
    """Read tags for paths the scan cache doesn't know (queue adds, M3U, graph)."""

    rows_ready = Signal(list)

    _BATCH = 25

    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self._paths = list(paths)

    def run(self) -> None:
        batch: list[dict] = []
        for p in self._paths:
            if self._stopped():
                return
            batch.append(_load_row(p))
            if len(batch) >= self._BATCH:
                self.rows_ready.emit(list(batch))
                batch.clear()
        if batch and not self._stopped():
            self.rows_ready.emit(batch)


class _ArtLoader(_CancellableThread):
    """Load and scale album art for one track off the GUI thread.

    Emits a ``QImage`` (NOT a QPixmap) so the caller can convert to
    QPixmap on the main/GUI thread.  Creating QPixmap off the GUI thread
    is undefined behaviour in Qt and silently produces null or broken
    pixmaps on Windows. ``token`` lets the receiver drop stale results.
    """

    art_ready = Signal(object, str, int)   # QImage | None, path, token

    def __init__(self, path: str, size: int = _ART_SIZE, token: int = 0) -> None:
        super().__init__()
        self._path = path
        self._size = size
        self.token = token

    def run(self) -> None:
        result = None
        try:
            if self._stopped():
                return
            from utils.audio_metadata_reader import read_metadata
            _, covers, _, _ = read_metadata(self._path, include_cover=True)
            if self._stopped():
                return
            if not covers:
                folder_art = _folder_art_bytes(self._path)
                covers = [folder_art] if folder_art else []
            if covers:
                img = QtGui.QImage()
                if img.loadFromData(covers[0]):
                    s = self._size
                    img = img.scaled(
                        s, s,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                    if img.width() > s:
                        img = img.copy((img.width() - s) // 2, 0, s, s)
                    elif img.height() > s:
                        img = img.copy(0, (img.height() - s) // 2, s, s)
                    if not img.isNull():
                        result = img
        except Exception:
            result = None
        if not self._stopped():
            self.art_ready.emit(result, self._path, self.token)


# ─────────────────────────────────────────────────────────────────────────────
# _LibraryCard — the library's glass card, lit by the playing track's cover
# ─────────────────────────────────────────────────────────────────────────────

class _LibraryCard(GlassCard):
    """The glass card that holds the library table.

    While a track plays, its cover, blurred, glows faintly through the card
    and cross-fades to the next track's (3.5 s). The table on top is a
    transparent item view, so the rows sit on the glass and the art: there
    is no opaque block. ``has_art`` / ``set_art`` keep the old background
    widget's API."""

    _BLUR_DIV    = 10    # scale-down factor for a cheap blur
    _ART_OPACITY = 0.22  # glass: how strongly the art shows through
    _CLASSIC_ART_OPACITY = 0.12   # opaque classic cards: a tint, no more
    _FADE_MS     = 3500   # cross-fade duration in ms

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pm_cur: QtGui.QPixmap | None = None
        self._pm_nxt: QtGui.QPixmap | None = None
        self._fading = False
        self._fade_val: float = 1.0
        self._scaled: dict[tuple[int, int, int], QtGui.QPixmap] = {}

        self._anim = QtCore.QPropertyAnimation(self, b"crossfade", self)
        self._anim.setDuration(self._FADE_MS)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutSine)

    # Qt property for the animation target
    def _get_cf(self) -> float:
        return self._fade_val

    def _set_cf(self, v: float) -> None:
        self._fade_val = v
        if v >= 1.0 and self._fading:
            self._pm_cur = self._pm_nxt
            self._pm_nxt = None
            self._fading = False
            self._prune_scaled()
        self.update()

    crossfade = QtCore.Property(float, _get_cf, _set_cf)

    @property
    def has_art(self) -> bool:
        return (self._pm_nxt if self._fading else self._pm_cur) is not None

    def set_art(self, pm: QtGui.QPixmap | None) -> None:
        """Cross-fade to ``pm``; ``None`` or a null pixmap fades the art out."""
        new = self._blur(pm) if pm is not None and not pm.isNull() else None
        if self._fading:
            # A fade is in flight: keep whichever image dominates right now.
            self._anim.stop()
            if self._fade_val >= 0.5:
                self._pm_cur = self._pm_nxt
            self._pm_nxt = None
            self._fading = False
        if new is None and self._pm_cur is None:
            self.update()
            return
        if self._pm_cur is None:
            self._pm_cur = new
            self._prune_scaled()
            self.update()
            return
        self._pm_nxt = new
        self._fading = True
        self._fade_val = 0.0
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _blur(self, pm: QtGui.QPixmap) -> QtGui.QPixmap:
        img = pm.toImage()
        d   = self._BLUR_DIV
        small = img.scaled(
            max(1, img.width() // d), max(1, img.height() // d),
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        big = small.scaled(
            img.width(), img.height(),
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        return QtGui.QPixmap.fromImage(big)

    def _prune_scaled(self) -> None:
        live = {id(pm) for pm in (self._pm_cur, self._pm_nxt) if pm is not None}
        self._scaled = {k: v for k, v in self._scaled.items() if k[0] in live}

    def _fitted(self, pm: QtGui.QPixmap, w: int, h: int) -> QtGui.QPixmap:
        """``pm`` scaled to cover w×h, cached per size (the table scrolling
        over the card repaints it often)."""
        key = (id(pm), w, h)
        out = self._scaled.get(key)
        if out is None:
            out = pm.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            QtCore.Qt.TransformationMode.FastTransformation)
            self._scaled = {k: v for k, v in self._scaled.items() if k[1:] == (w, h)}
            self._scaled[key] = out
        return out

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface

        m = current_material()
        rect = QtCore.QRectF(self.rect())
        radius = self.radius()
        p = QtGui.QPainter(self)
        try:
            paint_surface(p, rect, m.card, radius=radius)
            if self._pm_cur is None and self._pm_nxt is None:
                return
            clip = QtGui.QPainterPath()
            clip.addRoundedRect(rect.adjusted(1, 1, -1, -1), radius - 1, radius - 1)
            p.setClipPath(clip)
            w, h = self.width(), self.height()
            strength = self._ART_OPACITY if m.is_glass else self._CLASSIC_ART_OPACITY

            def _blit(pm: QtGui.QPixmap | None, op: float) -> None:
                if pm is None or pm.isNull() or op <= 0.0:
                    return
                s = self._fitted(pm, w, h)
                p.setOpacity(op)
                p.drawPixmap((w - s.width()) // 2, (h - s.height()) // 2, s)

            if self._fading:
                _blit(self._pm_cur, strength * (1.0 - self._fade_val))
                _blit(self._pm_nxt, strength * self._fade_val)
            else:
                _blit(self._pm_cur, strength)
        finally:
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# NowPlayingCovers — prev / current / next album tiles
# ─────────────────────────────────────────────────────────────────────────────

class NowPlayingCovers(QtWidgets.QWidget):
    """The previous, current and next covers as glass album tiles.

    * *Current* is a 130 px tile (radius 20, the mockup's painted shadow and
      inset ring), centred.
    * *Previous* / *next* are smaller tiles tucked behind it, faded; each is
      drawn only when the queue has such a neighbour.
    * Without art a tile is a faint well with a music glyph, never an opaque
      square.
    The tiles shrink with the widget's width (narrow side column).
    """

    _SIDE = 80       # side tile size at full width
    _MAIN = 130      # centre tile size at full width
    _RAD  = 20       # centre tile corner radius (side tiles scale with it)
    _TOP = 8
    _BOTTOM = 26     # room for the centre tile's shadow
    _SIDE_OPACITY = 0.55

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._prev_pm:    QtGui.QPixmap | None = None
        self._current_pm: QtGui.QPixmap | None = None
        self._next_pm:    QtGui.QPixmap | None = None
        self._has_prev = False
        self._has_next = False
        self.setMinimumSize(150, self._MAIN + self._TOP + self._BOTTOM)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._MAIN + self._SIDE * 2, self._MAIN + self._TOP + self._BOTTOM)

    def set_current(self, pm: QtGui.QPixmap | None) -> None:
        self._current_pm = pm
        self.update()

    def set_prev(self, pm: QtGui.QPixmap | None, present: bool = True) -> None:
        """``pm`` None with ``present`` True: a neighbour without art."""
        self._prev_pm = pm
        self._has_prev = present or pm is not None
        self.update()

    def set_next(self, pm: QtGui.QPixmap | None, present: bool = True) -> None:
        self._next_pm = pm
        self._has_next = present or pm is not None
        self.update()

    def clear_neighbors(self) -> None:
        self._prev_pm = None
        self._next_pm = None
        self._has_prev = self._has_next = False
        self.update()

    def clear_all(self) -> None:
        self._current_pm = None
        self.clear_neighbors()

    # ── painting ────────────────────────────────────────────────────────

    def _geometry(self) -> tuple[QtCore.QRectF, QtCore.QRectF, QtCore.QRectF, float, float]:
        w = self.width()
        main = float(min(self._MAIN, max(64, int(w * 0.5))))
        side = round(main * self._SIDE / self._MAIN)
        mx = (w - main) / 2
        my = float(self._TOP)
        sy = my + (main - side) / 2
        tuck = side * 0.62          # how far a side tile shows beside the centre
        cur = QtCore.QRectF(mx, my, main, main)
        prev = QtCore.QRectF(mx - tuck, sy, side, side)
        nxt = QtCore.QRectF(mx + main + tuck - side, sy, side, side)
        r_main = self._RAD * main / self._MAIN
        r_side = max(10.0, self._RAD * side / self._MAIN)
        return prev, cur, nxt, r_main, r_side

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        prev, cur, nxt, r_main, r_side = self._geometry()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHints(
                QtGui.QPainter.RenderHint.Antialiasing
                | QtGui.QPainter.RenderHint.SmoothPixmapTransform,
            )
            p.setOpacity(self._SIDE_OPACITY)
            if self._has_prev:
                _paint_tile(p, prev, self._prev_pm, r_side, shadow=False)
            if self._has_next:
                _paint_tile(p, nxt, self._next_pm, r_side, shadow=False)
            p.setOpacity(1.0)
            _paint_tile(p, cur, self._current_pm, r_main)
        finally:
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Row delegates: transparent glass rows for the table and the lists
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_TITLE   = int(QtCore.Qt.ItemDataRole.UserRole) + 1
_ROLE_ARTIST  = int(QtCore.Qt.ItemDataRole.UserRole) + 2
_ROLE_LEAD    = int(QtCore.Qt.ItemDataRole.UserRole) + 3   # "1", "2"… (queue position)
_ROLE_PLAYING = int(QtCore.Qt.ItemDataRole.UserRole) + 4

_COL_NUM, _COL_TITLE, _COL_ARTIST, _COL_ALBUM, _COL_TIME = range(5)


def _row_fill(selected: bool, playing: bool, hovered: bool, *, base: bool = False):
    """The fill of one row (``None``: nothing): selection, the playing
    track's accent tint, hover, or (lists) the resting glass row."""
    from gui.themes.glass import qcolor

    t = _tokens()
    m = t.material
    if selected:
        return qcolor(m.selection)
    if playing:
        return qcolor(t.accent, 0.16)
    if hovered:
        return qcolor(m.pressed if base else m.hover)
    if base:
        return qcolor(m.hover, 0.75)
    return None


def _paint_state_glyph(p: QtGui.QPainter, rect: QtCore.QRectF, playing_now: bool,
                       colour: QtGui.QColor) -> None:
    """The playing track's marker: a play glyph while it plays, pause while
    it is paused."""
    from gui.widgets.glyphs import paint_glyph

    g = 13.0
    paint_glyph(p, QtCore.QRectF(rect.x(), rect.center().y() - g / 2, g, g),
                "play" if playing_now else "pause", colour)


class _TrackTableDelegate(QtWidgets.QStyledItemDelegate):
    """Library rows: 34 px, no grid, a rounded glass row for hover and
    selection, the playing track tinted with the accent and marked with a
    play glyph in the # column. Title 14, artist/album 13 muted, # and time
    12 with tabular figures."""

    ROW_H = 34
    _PAD = 10

    def __init__(self, table: QtWidgets.QTableView, playing) -> None:
        super().__init__(table)
        self._table = table
        self._playing = playing          # () -> (path or "", is playing now)
        self.hover_row = -1

    @staticmethod
    def font_for(column: int) -> QtGui.QFont:
        from gui.themes.glass import font
        if column == _COL_TITLE:
            return font("body")
        if column in (_COL_NUM, _COL_TIME):
            return font("caption", tabular=True)
        return font("label")

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem,  # noqa: N802
                 index: QtCore.QModelIndex) -> QtCore.QSize:
        text = str(index.data(QtCore.Qt.ItemDataRole.DisplayRole) or "")
        fm = QtGui.QFontMetrics(self.font_for(index.column()))
        return QtCore.QSize(fm.horizontalAdvance(text) + 2 * self._PAD + 2, self.ROW_H)

    def paint(self, p: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem,
              index: QtCore.QModelIndex) -> None:
        from gui.themes.glass import qcolor

        t = _tokens()
        m = t.material
        col = index.column()
        playing_path, playing_now = self._playing()
        playing = bool(playing_path) and index.data(QtCore.Qt.ItemDataRole.UserRole) == playing_path
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        hovered = index.row() == self.hover_row

        p.save()
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            cell = QtCore.QRectF(option.rect)
            fill = _row_fill(selected, playing, hovered)
            if fill is not None:
                vp_w = self._table.viewport().width()
                row = QtCore.QRectF(2, cell.top() + 1, max(4.0, vp_w - 4.0), cell.height() - 2)
                path = QtGui.QPainterPath()
                path.addRoundedRect(row, 10, 10)
                p.setClipRect(cell)
                p.fillPath(path, fill)
                p.setClipping(False)

            if selected:
                ink = qcolor(m.selection_text)
                quiet = ink
            elif playing:
                ink = qcolor(t.accent)
                quiet = qcolor(t.text_secondary)
            else:
                ink = qcolor(t.text_primary)
                quiet = qcolor(t.text_secondary)

            text_rect = cell.adjusted(self._PAD, 0, -self._PAD, 0)
            if col == _COL_NUM and playing:
                g_rect = QtCore.QRectF(text_rect.right() - 13, text_rect.top(), 13, text_rect.height())
                _paint_state_glyph(p, g_rect, playing_now, ink)
                return
            text = str(index.data(QtCore.Qt.ItemDataRole.DisplayRole) or "")
            if not text:
                return
            f = self.font_for(col)
            if col == _COL_TITLE and playing:
                f.setWeight(QtGui.QFont.Weight(650))
            p.setFont(f)
            p.setPen(ink if col == _COL_TITLE else quiet)
            align = QtCore.Qt.AlignmentFlag.AlignVCenter | (
                QtCore.Qt.AlignmentFlag.AlignRight if col in (_COL_NUM, _COL_TIME)
                else QtCore.Qt.AlignmentFlag.AlignLeft)
            fm = QtGui.QFontMetrics(f)
            p.drawText(text_rect, int(align),
                       fm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, int(text_rect.width())))
        finally:
            p.restore()


class _QueueRowDelegate(QtWidgets.QStyledItemDelegate):
    """Queue and Recently played rows: glass rows (34 px, radius 12, a faint
    fill, 4 px apart) with the position (or the playing glyph), the title
    and a muted "· artist"."""

    ROW_H = 38   # 34 px row + 4 px gap

    def __init__(self, view: QtWidgets.QListWidget, playing=None) -> None:
        super().__init__(view)
        self._playing = playing or (lambda: ("", False))

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem,  # noqa: N802
                 index: QtCore.QModelIndex) -> QtCore.QSize:
        return QtCore.QSize(max(80, option.rect.width()), self.ROW_H)

    def paint(self, p: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem,
              index: QtCore.QModelIndex) -> None:
        from gui.themes.glass import font, qcolor

        t = _tokens()
        m = t.material
        playing = bool(index.data(_ROLE_PLAYING))
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QtWidgets.QStyle.StateFlag.State_MouseOver)
        row = QtCore.QRectF(option.rect).adjusted(0, 2, -1, -2)
        p.save()
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            path = QtGui.QPainterPath()
            path.addRoundedRect(row, 12, 12)
            p.fillPath(path, _row_fill(selected, playing, hovered, base=True))

            if selected:
                ink = quiet = qcolor(m.selection_text)
            else:
                ink = qcolor(t.accent if playing else t.text_primary)
                quiet = qcolor(t.text_secondary)

            x = row.left() + 12
            lead = str(index.data(_ROLE_LEAD) or "")
            if playing:
                _, playing_now = self._playing()
                _paint_state_glyph(p, QtCore.QRectF(x + 4, row.top(), 13, row.height()),
                                   playing_now, ink)
                x += 26
            elif lead:
                f = font("caption", tabular=True)
                p.setFont(f)
                p.setPen(quiet)
                p.drawText(QtCore.QRectF(x, row.top(), 20, row.height()),
                           int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter),
                           lead)
                x += 26

            title = str(index.data(_ROLE_TITLE) or index.data(QtCore.Qt.ItemDataRole.DisplayRole) or "")
            artist = str(index.data(_ROLE_ARTIST) or "")
            avail = row.right() - 12 - x
            tf = font("label", weight=650 if playing else 500)
            tm = QtGui.QFontMetrics(tf)
            title_txt = tm.elidedText(title, QtCore.Qt.TextElideMode.ElideRight, int(avail))
            p.setFont(tf)
            p.setPen(ink)
            p.drawText(QtCore.QRectF(x, row.top(), avail, row.height()),
                       int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                       title_txt)
            used = tm.horizontalAdvance(title_txt)
            rest = avail - used
            if artist and title_txt == title and rest > 40:
                af = font("caption")
                am = QtGui.QFontMetrics(af)
                p.setFont(af)
                p.setPen(quiet)
                p.drawText(QtCore.QRectF(x + used, row.top(), rest, row.height()),
                           int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                           am.elidedText(f"  ·  {artist}", QtCore.Qt.TextElideMode.ElideRight, int(rest)))
        finally:
            p.restore()


# ─────────────────────────────────────────────────────────────────────────────
# _TrackHoverPopup — album-art + metadata popup near cursor
# ─────────────────────────────────────────────────────────────────────────────

class _CoverLabel(QtWidgets.QLabel):
    """A rounded cover; a faint well with a music glyph while there is no
    art (``text()`` is then "♪", the state marker; the glyph is painted)."""

    def __init__(self, size: int, radius: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = size
        self._radius = radius
        self.setFixedSize(size, size)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setText("♪")

    def set_cover(self, pm: QtGui.QPixmap) -> None:
        self.setText("")
        self.setPixmap(_rounded(pm, self._size, self._radius))

    def clear_cover(self) -> None:
        self.setPixmap(QtGui.QPixmap())
        self.setText("♪")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            super().paintEvent(event)
            return
        p = QtGui.QPainter(self)
        try:
            _paint_tile(p, QtCore.QRectF(self.rect()), None, self._radius, shadow=False)
        finally:
            p.end()


class _TrackHoverPopup(QtWidgets.QFrame):
    """Floating card shown after hovering a library row for 1.5 s.

    A ``ToolTip``-type window: it never grabs the mouse or keyboard. Text
    comes from the row cache; the cover arrives later from an ``_ArtLoader``
    (``set_art``), so showing it never blocks the GUI thread. It is a
    window of its own, so it paints an opaque card (nothing of the app's
    glass is behind it)."""

    _ART_SZ = 120
    _DELAY_MS = 1500

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(
            parent,
            QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint,
        )
        from gui.themes.glass import font

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(self._ART_SZ + 200)
        self.setObjectName("trackHoverPopup")
        self.path: str = ""

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 12, 14, 12)
        lay.setSpacing(12)

        self._art_lbl = _CoverLabel(self._ART_SZ, 14)
        self._art_lbl.setObjectName("trackHoverArt")
        lay.addWidget(self._art_lbl)

        info = QtWidgets.QVBoxLayout()
        info.setSpacing(3)
        self._title_lbl = QtWidgets.QLabel("")
        self._title_lbl.setObjectName("trackHoverTitle")
        self._title_lbl.setFont(font("body", weight=650))
        self._title_lbl.setWordWrap(True)
        self._artist_lbl = QtWidgets.QLabel("")
        self._artist_lbl.setObjectName("mutedLabel")
        self._artist_lbl.setFont(font("label"))
        self._album_lbl = QtWidgets.QLabel("")
        self._album_lbl.setObjectName("mutedLabel")
        self._album_lbl.setFont(font("caption"))
        self._album_lbl.setWordWrap(True)
        info.addWidget(self._title_lbl)
        info.addWidget(self._artist_lbl)
        info.addWidget(self._album_lbl)
        info.addStretch()
        lay.addLayout(info, 1)

    def apply_theme(self, t=None) -> None:
        """Colours are read at paint time (and from the label roles)."""
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_surface
        from gui.themes.tokens import RADIUS, Surface

        t = _tokens()
        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()),
                          Surface(t.card_bg, t.card_border, radius=RADIUS.md))
        finally:
            p.end()

    def cancel(self) -> None:
        """Hide and forget the current track."""
        self.hide()
        self.path = ""

    def show_for(self, row: dict, cursor_pos: QtCore.QPoint) -> None:
        """Populate from ``row`` (no file I/O) and show near ``cursor_pos``."""
        path = row.get("path", "")
        self.path = path
        self._title_lbl.setText(row.get("title") or Path(path).stem)
        self._artist_lbl.setText(row.get("artist", ""))
        self._album_lbl.setText(row.get("album", ""))
        self._art_lbl.clear_cover()

        self.adjustSize()
        # Position near cursor but keep on-screen
        screen = QtGui.QGuiApplication.screenAt(cursor_pos)
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        geo = screen.availableGeometry() if screen else QtCore.QRect()
        x = cursor_pos.x() + 16
        y = cursor_pos.y() + 16
        if x + self.width() > geo.right():
            x = cursor_pos.x() - self.width() - 8
        if y + self.height() > geo.bottom():
            y = cursor_pos.y() - self.height() - 8
        self.move(x, y)
        self.show()

    def set_art(self, path: str, img: QtGui.QImage | None) -> None:
        if path != self.path or img is None or img.isNull():
            return
        self._art_lbl.set_cover(QtGui.QPixmap.fromImage(img))


# ─────────────────────────────────────────────────────────────────────────────
# PlayerWorkspace
# ─────────────────────────────────────────────────────────────────────────────

class PlayerWorkspace(WorkspaceBase):
    """Library browser, queue and playlists; the Now Playing capsule is its
    transport.

    The page has no transport of its own ("the bar is the only transport",
    docs/redesign/README.md): play/pause, previous/next, seek, volume,
    shuffle and repeat are on the capsule, which the main window wires to
    the public slots and signals below. Showing this page pins the capsule
    (``transport_pinned``), so the Player is never without a transport.
    """

    # ── Signals for NowPlayingBar ──────────────────────────────────────────
    # pm: QPixmap with art; None = art still loading (keep what is shown);
    # a *null* QPixmap = this track has no art. Title "" + null pixmap =
    # nothing is loaded any more (the bar resets and hides).
    now_playing_changed    = Signal(str, str, str, object)  # title, artist, album, QPixmap|None
    playback_state_changed = Signal(bool)                   # is_playing
    position_changed       = Signal(int, int)               # pos_ms, dur_ms
    volume_changed         = Signal(int)                    # 0-100, any source
    shuffle_changed        = Signal(bool)
    repeat_changed         = Signal(str)                    # "off" | "track" | "all"
    # available (a backend), can_play (something to play), reason (why not)
    transport_changed      = Signal(bool, bool, str)
    transport_pinned       = Signal(bool)                   # this page is on screen

    #: Library table widths below which a column is hidden (still searchable).
    _HIDE_ALBUM_BELOW = 470
    _HIDE_NUMBER_BELOW = 360

    # ── Setup ──────────────────────────────────────────────────────────────

    def _setup_scroll(self) -> None:
        """Override: no scroll area — the page fills the viewport (the table
        and the queue scroll themselves). It keeps the shell's room at the
        bottom, so the floating Now Playing capsule never covers anything."""
        from gui.themes.tokens import SPACE

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, SPACE.page_bottom)
        outer.setSpacing(0)
        self._inner = GradientWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._inner)
        self._content_layout.setContentsMargins(SPACE.gutter, SPACE.xs, SPACE.gutter, 0)
        self._content_layout.setSpacing(SPACE.sm)
        outer.addWidget(self._inner)

    def __init__(
        self,
        library_path: str = "",
        parent: QtWidgets.QWidget | None = None,
        backend: PlaybackBackend | None = None,
    ) -> None:
        super().__init__(library_path, parent)

        self._backend: PlaybackBackend = (
            backend if backend is not None else playback_backend.create_backend()
        )

        # Queue / playback state
        self._queue:        list[dict] = []
        self._queue_index:  int        = -1     # queue cursor
        self._cursor_played: bool      = False  # has the cursor's track been started?
        self._direct_row:   dict | None = None  # track played directly (not from queue)
        self._stopped_row:  dict | None = None  # direct track that was stopped / ended
        self._shuffle:      bool       = False
        self._repeat:       int        = _REPEAT_OFF
        self._volume:       int        = 80     # 0-100 (the capsule's knob)
        self._playing_now:  bool       = False  # last state sent to the capsule
        self._keyboard_mode: bool      = False  # arrow-key navigation mode
        self._kb_target: str           = "library"  # "library" or "queue"
        self._lib_kb_row: int          = -1  # last-played row in library table for kb nav
        self._recent_plays: list[dict] = []   # most-recent first, capped at 30
        self._error_streak: int        = 0
        self._dur_ms: int              = 0

        # 30 s preview ("checkbox", or a keyboard one-shot: "queue"/"library")
        self._preview_source: str | None = None
        self._highlight_remaining: int  = -1
        self._preview_seek_pending: bool = False

        # Shuffle history for prev/next cover awareness
        self._shuffle_history: list[int] = []  # indices played in shuffle order
        self._shuffle_next_idx: int | None = None  # pre-picked next shuffle index

        # Library cache
        self._all_rows: list[dict] = []          # full scan results
        self._rows_by_path: dict[str, dict] = {}  # path → row (scan + tag reads)
        self._pending_tag_paths: set[str] = set()

        # Progressive render state
        self._scan_finished: bool = False
        self._scan_gen: int = 0
        self._scanned_path: str = ""

        # Workers. The JobRegistry keeps each thread alive until it really
        # finishes, so these references can be replaced at any time.
        self._lib_scanner:      _LibraryScanner | None = None
        self._art_loader:       _ArtLoader | None      = None
        self._prev_art_loader:  _ArtLoader | None      = None
        self._next_art_loader:  _ArtLoader | None      = None
        self._hover_art_loader: _ArtLoader | None      = None
        self._art_gen = 0
        self._nb_gen = 0
        self._hover_gen = 0

        # Timers (all parented, all cancellable)
        self._pos_timer = QtCore.QTimer(self)
        self._pos_timer.setInterval(400)
        self._pos_timer.timeout.connect(self._tick_position)

        self._highlight_timer = QtCore.QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self._highlight_advance)

        self._preview_seek_timer = QtCore.QTimer(self)
        self._preview_seek_timer.setSingleShot(True)
        self._preview_seek_timer.timeout.connect(self._on_preview_seek)

        self._vol_apply_timer = QtCore.QTimer(self)
        self._vol_apply_timer.setSingleShot(True)
        self._vol_apply_timer.setInterval(_VOLUME_REAPPLY_MS)
        self._vol_apply_timer.timeout.connect(self._reapply_volume)

        self._skip_timer = QtCore.QTimer(self)
        self._skip_timer.setSingleShot(True)
        self._skip_timer.setInterval(50)
        self._skip_timer.timeout.connect(self._advance)

        self._build_ui()
        self._init_backend()
        self._install_shortcuts()
        get_manager().theme_changed.connect(self._apply_theme)
        self._apply_theme(get_manager().current)
        self._update_library_empty()

    # ── UI construction ────────────────────────────────────────────────────

    def _icon_button(self, glyph: str, tip: str, slot) -> QtWidgets.QPushButton:
        """A 34 px round ghost button with a painted glyph."""
        btn = QtWidgets.QPushButton()
        btn.setFlat(True)
        btn.setFixedSize(34, 34)
        btn.setToolTip(tip)
        btn.setAccessibleName(tip.rstrip("…"))
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        set_glyph(btn, glyph, role="muted", size=17)
        btn.clicked.connect(slot)
        return btn

    def _elevate(self, card: QtWidgets.QWidget) -> None:
        """Register a card made here for its painted shadow."""
        self._cards.append(card)
        self._inner.add_elevated(card)

    @staticmethod
    def _overlay(base: QtWidgets.QWidget, cover: QtWidgets.QWidget) -> QtWidgets.QGridLayout:
        """``cover`` (an empty state) over ``base`` in one grid cell; the
        base stays in place (and keeps keyboard focus) underneath."""
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(base, 0, 0)
        grid.addWidget(cover, 0, 0)
        return grid

    def _build_ui(self) -> None:
        from gui.themes.glass import font
        from gui.themes.tokens import SPACE

        cl = self._content_layout

        # ── Purpose (one line; the header already says "Player") ──────────
        self._subtitle = ElidedLabel(
            "Double-click a track to play it, or build a queue. "
            "The bar at the bottom is the transport.")
        self._subtitle.setObjectName("sectionSubtitle")
        self._subtitle.setFont(font("label"))
        cl.addWidget(self._subtitle)

        # ── "Playback unavailable": a quiet notice, shown without a backend ─
        self._notice = GlassPanel(surface="well")
        self._notice.setObjectName("playerNotice")
        nl = QtWidgets.QHBoxLayout(self._notice)
        nl.setContentsMargins(12, 7, 14, 7)
        nl.setSpacing(8)
        self._notice_icon = QtWidgets.QLabel()
        self._notice_icon.setFixedSize(16, 16)
        nl.addWidget(self._notice_icon)
        self._vlc_status_lbl = QtWidgets.QLabel("")
        self._vlc_status_lbl.setObjectName("playerBackendStatus")
        self._vlc_status_lbl.setFont(font("label", weight=600))
        nl.addWidget(self._vlc_status_lbl)
        self._vlc_reason_lbl = ElidedLabel("")
        self._vlc_reason_lbl.setObjectName("mutedLabel")
        self._vlc_reason_lbl.setFont(font("label"))
        nl.addWidget(self._vlc_reason_lbl, 1)
        self._notice.setVisible(False)
        cl.addWidget(self._notice)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(SPACE.gutter)
        cl.addLayout(row, 1)

        # ── Library card (the table on glass; the playing cover glows) ────
        self._bg_art = _LibraryCard()
        self._bg_art.setObjectName("playerLibraryCard")
        self._bg_art.setMinimumWidth(330)
        self._elevate(self._bg_art)
        lib = QtWidgets.QVBoxLayout(self._bg_art)
        lib.setContentsMargins(14, 14, 14, 10)
        lib.setSpacing(8)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(4)
        self._search = QtWidgets.QLineEdit()
        self._search.setObjectName("playerSearch")
        self._search.setPlaceholderText("Search titles, artists, albums")
        self._search.setAccessibleName("Search the library")
        self._search.setClearButtonEnabled(True)
        self._search.setMinimumWidth(120)
        self._search_action = self._search.addAction(
            QtGui.QIcon(), QtWidgets.QLineEdit.ActionPosition.LeadingPosition)
        self._search.textChanged.connect(self._apply_filter)
        head.addWidget(self._search, 1)
        head.addSpacing(4)
        self._reload_btn = self._icon_button("refresh", "Re-scan the library folder",
                                             self._reload_library)
        self._browse_btn = self._icon_button("file", "Open and play an audio file…",
                                             self._on_browse_file)
        head.addWidget(self._reload_btn)
        head.addWidget(self._browse_btn)
        lib.addLayout(head)

        self._lib_table = QtWidgets.QTableWidget(0, 5)
        self._lib_table.setObjectName("playerLibraryTable")
        self._lib_table.setHorizontalHeaderLabels(["#", "Title", "Artist", "Album", "Time"])
        self._lib_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._lib_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._lib_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._lib_table.setAlternatingRowColors(False)
        self._lib_table.setShowGrid(False)
        self._lib_table.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._lib_table.setWordWrap(False)
        self._lib_table.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._lib_table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._lib_table.verticalHeader().setVisible(False)
        self._lib_table.verticalHeader().setDefaultSectionSize(_TrackTableDelegate.ROW_H)
        self._lib_table.verticalHeader().setMinimumSectionSize(_TrackTableDelegate.ROW_H)
        self._lib_delegate = _TrackTableDelegate(self._lib_table, self._playing_marker)
        self._lib_table.setItemDelegate(self._lib_delegate)

        hh = self._lib_table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setHighlightSections(False)
        hh.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Fixed)
        for col in (_COL_NUM, _COL_TIME):
            item = self._lib_table.horizontalHeaderItem(col)
            if item is not None:
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                      | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._lib_table.setColumnWidth(0, 40)
        self._lib_table.setColumnWidth(4, 60)
        # Default order: Artist → Album → Track, ascending.
        hh.setSortIndicator(2, QtCore.Qt.SortOrder.AscendingOrder)
        self._lib_table.setSortingEnabled(True)

        self._lib_table.doubleClicked.connect(self._on_table_double_click)
        self._lib_table.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._lib_table.customContextMenuRequested.connect(self._on_table_context_menu)
        self._lib_table.installEventFilter(self)   # width → which columns fit

        # Hover popup for album art preview (1.5 s dwell). Mouse tracking is
        # needed for button-less MouseMove events to reach the filter (it
        # also drives the hovered-row highlight).
        self._hover_popup = _TrackHoverPopup(self)
        self._hover_timer = QtCore.QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(_TrackHoverPopup._DELAY_MS)
        self._hover_timer.timeout.connect(self._on_hover_timeout)
        self._hover_path: str = ""
        self._lib_table.setMouseTracking(True)
        self._lib_table.viewport().setMouseTracking(True)
        self._lib_table.viewport().installEventFilter(self)
        # Track which panel was last interacted with for keyboard mode targeting
        self._lib_table.viewport().installEventFilter(self._make_focus_tracker("library"))

        self._lib_empty = EmptyState("No library selected")
        lib.addLayout(self._overlay(self._lib_table, self._lib_empty), 1)

        foot = QtWidgets.QHBoxLayout()
        foot.setSpacing(10)
        self._status_lbl = ElidedLabel("Load a library to begin.")
        self._status_lbl.setObjectName("mutedLabel")
        self._status_lbl.setFont(font("caption"))
        foot.addWidget(self._status_lbl, 1)
        self._scan_progress = QtWidgets.QProgressBar()
        self._scan_progress.setRange(0, 0)
        self._scan_progress.setFixedSize(72, 4)
        self._scan_progress.setTextVisible(False)
        self._scan_progress.setVisible(False)
        foot.addWidget(self._scan_progress)
        self._highlight_chk = QtWidgets.QCheckBox("30 s preview")
        self._highlight_chk.setToolTip(
            "Audition mode: each track starts at 0:30 and moves on after 15 s"
        )
        self._highlight_chk.toggled.connect(self._on_highlight_toggled)
        foot.addWidget(self._highlight_chk)
        lib.addLayout(foot)
        row.addWidget(self._bg_art, 1)

        # ── Side column: now playing (covers) + queue / recently played ───
        self._side = QtWidgets.QWidget()
        self._side.setObjectName("playerSide")
        self._side.setMinimumWidth(250)
        self._side.setMaximumWidth(330)
        self._side.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                 QtWidgets.QSizePolicy.Policy.Preferred)
        self._side.installEventFilter(self)          # height → covers or not
        side = QtWidgets.QVBoxLayout(self._side)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(SPACE.gutter)

        self._np_card = GlassCard()
        self._np_card.setObjectName("playerNowPlayingCard")
        self._elevate(self._np_card)
        npl = QtWidgets.QVBoxLayout(self._np_card)
        npl.setContentsMargins(12, 10, 12, 14)
        npl.setSpacing(2)
        self._covers = NowPlayingCovers()
        npl.addWidget(self._covers)
        self._np_title = ElidedLabel("Nothing playing")
        self._np_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._np_title.setFont(font("body", weight=650))
        self._np_artist = ElidedLabel("")
        self._np_artist.setObjectName("mutedLabel")
        self._np_artist.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._np_artist.setFont(font("caption"))
        npl.addWidget(self._np_title)
        npl.addWidget(self._np_artist)
        self._np_card.setFixedHeight(self._np_card_height())
        side.addWidget(self._np_card)

        self._queue_card = GlassCard()
        self._queue_card.setObjectName("playerQueueCard")
        self._elevate(self._queue_card)
        ql = QtWidgets.QVBoxLayout(self._queue_card)
        ql.setContentsMargins(14, 12, 12, 10)
        ql.setSpacing(6)

        q_head = QtWidgets.QHBoxLayout()
        q_head.setSpacing(0)
        self._q_title = QtWidgets.QLabel("Queue")
        self._q_title.setObjectName("cardTitle")
        q_head.addWidget(self._q_title)
        q_head.addSpacing(8)
        self._q_count_lbl = QtWidgets.QLabel("")
        self._q_count_lbl.setObjectName("mutedLabel")
        self._q_count_lbl.setFont(font("caption", tabular=True))
        q_head.addWidget(self._q_count_lbl, 0, QtCore.Qt.AlignmentFlag.AlignBaseline)
        q_head.addStretch(1)
        self._add_queue_btn = self._icon_button(
            "plus", "Add the selected library tracks to the queue", self._add_selected_to_queue)
        self._load_m3u_btn = self._icon_button(
            "folder", "Load an M3U playlist into the queue…", self._load_m3u)
        self._save_m3u_btn = self._icon_button(
            "save", "Save the queue as an M3U playlist…", self._save_m3u)
        self._clear_queue_btn = self._icon_button("trash", "Clear the queue", self._clear_queue)
        for b in (self._add_queue_btn, self._load_m3u_btn, self._save_m3u_btn,
                  self._clear_queue_btn):
            q_head.addWidget(b)
        ql.addLayout(q_head)

        self._queue_list = QtWidgets.QListWidget()
        self._queue_list.setObjectName("playerQueueList")
        self._style_row_list(self._queue_list)
        self._queue_list.doubleClicked.connect(self._on_queue_double_click)
        self._queue_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._queue_list.customContextMenuRequested.connect(self._on_queue_context_menu)
        # Track keyboard mode focus on queue panel
        self._queue_list.viewport().installEventFilter(self._make_focus_tracker("queue"))
        self._queue_empty = EmptyState("Queue is empty", "Select tracks and press +.")
        ql.addLayout(self._overlay(self._queue_list, self._queue_empty), 1)

        # Recently played (collapsed by default — expands in place)
        rp_row = QtWidgets.QHBoxLayout()
        rp_row.setSpacing(4)
        self._rp_hdr = QtWidgets.QLabel("Recently played")
        self._rp_hdr.setObjectName("mutedLabel")
        self._rp_hdr.setFont(font("caption", weight=600))
        rp_row.addWidget(self._rp_hdr)
        rp_row.addStretch(1)
        self._rp_toggle_btn = QtWidgets.QPushButton()
        self._rp_toggle_btn.setFlat(True)
        self._rp_toggle_btn.setFixedSize(28, 26)
        self._rp_toggle_btn.setToolTip("Show / hide recently played")
        self._rp_toggle_btn.setAccessibleName("Show recently played")
        self._rp_toggle_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        set_glyph(self._rp_toggle_btn, "chevron_right", role="muted", size=14)
        self._rp_toggle_btn.clicked.connect(self._toggle_recent)
        rp_row.addWidget(self._rp_toggle_btn)
        ql.addLayout(rp_row)

        self._recent_list = QtWidgets.QListWidget()
        self._recent_list.setObjectName("playerRecentList")
        self._style_row_list(self._recent_list, playing=False)
        self._recent_list.setFixedHeight(0)   # collapsed initially
        self._recent_list.doubleClicked.connect(self._on_recent_double_click)
        ql.addWidget(self._recent_list)
        side.addWidget(self._queue_card, 1)
        # An explicit minimum: what the column holds never props the page
        # open; _fit_side_column drops the carousel when it runs short.
        self._side.setMinimumHeight(self._queue_card.minimumSizeHint().height())
        row.addWidget(self._side)

    def _style_row_list(self, view: QtWidgets.QListWidget, *, playing: bool = True) -> None:
        """A transparent list of glass rows (see ``_QueueRowDelegate``)."""
        view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        view.setAlternatingRowColors(False)
        view.setUniformItemSizes(True)
        view.setMouseTracking(True)
        view.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setItemDelegate(_QueueRowDelegate(view, self._playing_marker if playing else None))

    # ── Responsive pieces ─────────────────────────────────────────────────

    def _fit_side_column(self) -> None:
        """Show the cover carousel only when the column has room for it at
        full size *and* a usable queue under it (the capsule shows the
        current cover anyway). The column's own minimum is explicit, so the
        card never props the page open: it is hidden first."""
        from gui.themes.tokens import SPACE

        card_h = self._np_card_height()
        queue_min = self._queue_card.minimumSizeHint().height()
        show = self._side.height() >= card_h + SPACE.gutter + queue_min
        if self._np_card.isVisibleTo(self._side) != show:
            self._np_card.setVisible(show)
        # A narrow column drops the queue count from the header (the title
        # keeps its room; the tooltip still says it).
        self._q_count_lbl.setVisible(self._side.width() >= 300)

    def _np_card_height(self) -> int:
        lay = self._np_card.layout()
        m = lay.contentsMargins()
        cm = self._np_card.contentsMargins()
        inner = (self._covers.sizeHint().height() + self._np_title.sizeHint().height()
                 + self._np_artist.sizeHint().height() + 2 * lay.spacing())
        return inner + m.top() + m.bottom() + cm.top() + cm.bottom()

    def _fit_table_columns(self) -> None:
        """Narrow table: hide Album, then #, so Title keeps room (the search
        still matches the hidden Album text)."""
        w = self._lib_table.width()
        self._lib_table.setColumnHidden(_COL_ALBUM, w < self._HIDE_ALBUM_BELOW)
        self._lib_table.setColumnHidden(_COL_NUM, w < self._HIDE_NUMBER_BELOW)

    def _apply_theme(self, tokens=None) -> None:
        """Repaint what paints itself; re-tint the icons that aren't glyph
        buttons (those follow the theme on their own)."""
        t = tokens if tokens is not None else get_manager().current
        self._search_action.setIcon(glyph_icon("search", t.text_secondary, 16))
        self._notice_icon.setPixmap(
            glyph_icon("warning", t.warning, 16).pixmap(16, 16))
        self._hover_popup.apply_theme(t)
        self._refresh_queue_list()
        self._lib_table.viewport().update()
        self._recent_list.viewport().update()
        self._bg_art.update()
        self._covers.update()

    def _init_backend(self) -> None:
        b = self._backend
        if b.available:
            self._vlc_status_lbl.setText(f"{b.name}: ready")
            self._vlc_reason_lbl.setText("")
            self._notice.setToolTip("")
            self._notice.setVisible(False)
            self._apply_volume(self._volume)
        else:
            reason = b.unavailable_reason or "No playback backend is available."
            self._vlc_status_lbl.setText("Playback unavailable")
            self._vlc_reason_lbl.setText(reason)
            self._notice.setToolTip(reason)
            self._notice.setVisible(True)
            # "info", not "warn": a missing optional dependency is reported on
            # the page itself, and a warning would pop the Activity Log open
            # on every launch of a machine without VLC.
            self._log(f"Playback unavailable: {reason}", "info")
        self._refresh_transport()

    @property
    def backend(self) -> PlaybackBackend:
        return self._backend

    def _install_shortcuts(self) -> None:
        def _sc(key: str, fn) -> None:
            QtGui.QShortcut(QtGui.QKeySequence(key), self).activated.connect(fn)

        _sc("Space",       self._on_play_pause)
        _sc("N",           self.play_next)
        _sc("P",           self.play_prev)
        _sc("Shift+Right", lambda: self._seek_relative(+30000))
        _sc("Shift+Left",  lambda: self._seek_relative(-30000))

    # ── Library scanning ───────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        """Rescan when the library changes (but not for the same path again:
        the launch sequence sets the persisted library twice)."""
        if path and os.path.normcase(os.path.abspath(path)) == self._scanned_path:
            return
        self._cancel_scan()
        self._all_rows.clear()
        self._lib_table.setRowCount(0)
        self._lib_kb_row = -1
        self._status_lbl.setText("Library changed — click Reload to scan.")
        if path:
            self._reload_library()
        else:
            self._scanned_path = ""
        self._update_library_empty()

    def _cancel_scan(self) -> None:
        scanner = self._lib_scanner
        if scanner is not None:
            try:
                if scanner.isRunning():
                    scanner.cancel()
                    scanner.requestInterruption()
            except RuntimeError:
                pass
        self._lib_scanner = None

    def _reload_library(self) -> None:
        if not self._library_path:
            self._status_lbl.setText("No library selected.")
            return
        # Stop any running scanner (non-blocking: the registry keeps it alive
        # until it notices; its generation makes its late signals harmless).
        self._cancel_scan()
        self._scan_gen += 1

        self._all_rows.clear()
        self._scan_finished = False
        self._lib_kb_row = -1

        self._lib_table.setSortingEnabled(False)
        self._lib_table.setRowCount(0)
        self._reload_btn.setEnabled(False)
        self._status_lbl.setText("Scanning library…")

        # Indeterminate pulsing progress bar during the directory walk
        self._scan_progress.setRange(0, 0)
        self._scan_progress.setVisible(True)

        scanner = _LibraryScanner(self._library_path, self._scan_gen)
        scanner.tracks_ready.connect(self._on_tracks_ready)
        scanner.scan_complete.connect(self._on_scan_complete)
        self._lib_scanner = scanner
        self._scanned_path = os.path.normcase(os.path.abspath(self._library_path))
        self._start_worker(scanner, "Player library scan", ask_before_quit=False)
        self._update_library_empty()
        try:
            scanner.setPriority(QtCore.QThread.Priority.LowPriority)
        except RuntimeError:
            pass

    @Slot(int, list)
    def _on_tracks_ready(self, generation: int, batch: list) -> None:
        """Add incoming rows to the table immediately and report latency for auto-throttling."""
        if generation != self._scan_gen:
            return  # batch from a scan that has since been replaced
        t0 = QtCore.QElapsedTimer()
        t0.start()

        self._all_rows.extend(batch)
        for row in batch:
            self._rows_by_path[row["path"]] = row

        # Turn off sorting during bulk insert to prevent lag
        self._lib_table.setSortingEnabled(False)

        start_row = self._lib_table.rowCount()
        self._lib_table.setRowCount(start_row + len(batch))

        for i, row in enumerate(batch):
            self._set_table_row(start_row + i, row)

        loaded = self._lib_table.rowCount()
        self._status_lbl.setText(f"Loading… {loaded} tracks found")
        if loaded == len(batch):
            self._update_library_empty()     # the first rows: show the table

        # Report latency back to scanner for dynamic backpressure
        if self._lib_scanner is not None:
            self._lib_scanner.report_ui_latency(t0.elapsed())

    def _set_table_row(self, r: int, row: dict) -> None:
        keys = _sort_keys(row)
        texts = (row["track_num"], row["title"], row["artist"], row["album"], row["dur"])
        for c, (text, key) in enumerate(zip(texts, keys)):
            item = _SortItem(text, key)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row["path"])
            self._lib_table.setItem(r, c, item)

    @Slot(int, int)
    def _on_scan_complete(self, generation: int, total: int) -> None:
        """Scanner has finished walking the directory."""
        if generation != self._scan_gen:
            return
        self._scan_finished = True
        self._lib_scanner = None
        self._reload_btn.setEnabled(True)

        # Final cleanup and sort (by the header's indicator: Artist ↑ by default)
        self._lib_table.setSortingEnabled(True)
        self._apply_filter(self._search.text())

        n = len(self._all_rows)
        suffix = "track" if n == 1 else "tracks"
        self._status_lbl.setText(f"Loaded {n} {suffix}.")

        self._scan_progress.setVisible(False)
        self._update_library_empty()

    def _apply_filter(self, text: str) -> None:
        query = text.strip().lower()
        shown = 0
        for r in range(self._lib_table.rowCount()):
            if not query:
                self._lib_table.setRowHidden(r, False)
                shown += 1
                continue
            row_text = " ".join(
                (self._lib_table.item(r, c) or QtWidgets.QTableWidgetItem()).text().lower()
                for c in range(1, 4)  # Title, Artist, Album
            )
            hidden = query not in row_text
            self._lib_table.setRowHidden(r, hidden)
            shown += not hidden
        self._update_library_empty(no_matches=bool(query) and shown == 0)

    def _update_library_empty(self, *, no_matches: bool = False) -> None:
        """What the library card shows when the table has nothing to show."""
        empty = self._lib_empty
        if self._lib_table.rowCount() and not no_matches:
            empty.setVisible(False)
            return
        if no_matches:
            empty.set_text("No matches",
                           f"Nothing in the library matches \u201c{self._search.text().strip()}\u201d.")
        elif not self._library_path:
            empty.set_text("No library selected",
                           "Choose your music folder with Change library at the top; "
                           "its tracks will be listed here.")
        elif self._lib_scanner is not None:
            empty.set_text("Reading your library…", "Tracks appear here as they are found.")
        elif not self._scan_finished:
            empty.set_text("Not scanned yet",
                           "Press the reload button to list this folder's tracks.")
        else:
            empty.set_text("No tracks found",
                           "This folder has no audio files (Docs, Playlists, Trash and "
                           "Quarantine are not searched).")
        empty.setVisible(True)

    # ── Rows for paths (scan cache first, tags read off the GUI thread) ───

    def _rows_for(self, paths: list[str]) -> list[dict]:
        """Row dicts for ``paths``: cached scan rows where known, placeholders
        (filename as title) elsewhere — the tags are read in a worker and the
        rows are patched in place when they arrive."""
        rows: list[dict] = []
        missing: list[str] = []
        for p in paths:
            cached = self._rows_by_path.get(p)
            if cached is not None:
                rows.append(dict(cached))
            else:
                rows.append(_placeholder_row(p))
                if p not in self._pending_tag_paths:
                    self._pending_tag_paths.add(p)
                    missing.append(p)
        if missing:
            loader = _TagLoader(missing)
            loader.rows_ready.connect(self._on_tags_loaded)
            self._start_worker(loader, "Player tag read", ask_before_quit=False)
        return rows

    def _row_for(self, path: str) -> dict:
        return self._rows_for([path])[0]

    @Slot(list)
    def _on_tags_loaded(self, rows: list) -> None:
        by_path: dict[str, dict] = {}
        for r in rows:
            self._rows_by_path[r["path"]] = r
            self._pending_tag_paths.discard(r["path"])
            by_path[r["path"]] = r

        def _patch(row: dict | None) -> bool:
            if row is None or not row.get(_PENDING):
                return False
            fresh = by_path.get(row.get("path"))
            if fresh is None:
                return False
            row.pop(_PENDING, None)
            row.update(fresh)
            return True

        queue_changed = any([_patch(r) for r in self._queue])
        recent_changed = any([_patch(r) for r in self._recent_plays])
        _patch(self._direct_row)
        _patch(self._stopped_row)
        if queue_changed:
            self._refresh_queue_list()
        if recent_changed:
            self._refresh_recent_list()
        cur = self._current_playing_row()
        if cur is not None and cur.get("path") in by_path:
            self._show_now_playing_text(cur)
            self.now_playing_changed.emit(
                cur.get("title") or Path(cur["path"]).stem,
                cur.get("artist", ""), cur.get("album", ""), None,
            )

    # ── Table interaction ──────────────────────────────────────────────────

    @Slot(QtCore.QModelIndex)
    def _on_table_double_click(self, index: QtCore.QModelIndex) -> None:
        path = self._get_lib_path_at(index.row())
        if path:
            self._play_directly(path)

    def _on_table_context_menu(self, pos: QtCore.QPoint) -> None:
        rows = self._get_selected_paths()
        if not rows:
            return
        ink = _tokens().text_secondary
        menu = QtWidgets.QMenu(self)
        menu.addAction(glyph_icon("play", ink), "Play now",
                       lambda: self._play_directly(rows[0]))
        menu.addAction(glyph_icon("plus", ink), "Add to queue",
                       lambda: self._add_paths_to_queue(rows))
        menu.addSeparator()
        menu.addAction(glyph_icon("folder", ink), "Open folder",
                       lambda: self._open_in_explorer(rows[0]))
        menu.addAction(glyph_icon("save", ink), "Save selection as M3U…",
                       lambda: self._save_paths_as_m3u(rows))
        menu.exec(self._lib_table.viewport().mapToGlobal(pos))

    def _get_selected_paths(self) -> list[str]:
        rows = sorted({idx.row() for idx in self._lib_table.selectedIndexes()})
        paths: list[str] = []
        for r in rows:
            p = self._get_lib_path_at(r)
            if p:
                paths.append(p)
        return paths

    # ── Playback core ──────────────────────────────────────────────────────

    def _backend_state(self) -> PlaybackState:
        try:
            return self._backend.state()
        except Exception:  # noqa: BLE001
            return PlaybackState.ERROR

    def _report_unavailable(self) -> None:
        reason = self._backend.unavailable_reason or "No playback backend is available."
        self._status_lbl.setText(f"Playback unavailable — {reason}")

    def _refresh_transport(self) -> None:
        """Tell the capsule what it can do: a backend at all, and something
        to play (Play is disabled before anything is queued or loaded)."""
        self.transport_changed.emit(*self.transport_state())
        self._lib_table.viewport().update()     # the playing-row marker

    def transport_state(self) -> tuple[bool, bool, str]:
        """(available, can_play, reason): what ``transport_changed`` says."""
        avail = self._backend.available
        has_something = bool(self._queue) or self._current_playing_row() is not None \
            or self._stopped_row is not None
        reason = "" if avail else (self._backend.unavailable_reason
                                   or "No playback backend is available.")
        return avail, avail and has_something, reason

    def _set_playing_state(self, playing: bool) -> None:
        """Playing / paused / stopped: tell the capsule, re-mark the rows."""
        self._playing_now = playing
        self.playback_state_changed.emit(playing)
        self._lib_table.viewport().update()
        self._queue_list.viewport().update()

    def is_playing(self) -> bool:
        return self._playing_now

    def _playing_marker(self) -> tuple[str, bool]:
        """(path of the loaded track or "", whether it is playing) for the
        row delegates."""
        row = self._current_playing_row()
        return (row["path"] if row else ""), self._playing_now

    def _show_now_playing_text(self, row: dict) -> None:
        title = row.get("title") or Path(row["path"]).stem
        self._np_title.setText(title)
        self._np_title.setToolTip(title)
        sub = "  ·  ".join(p for p in [row.get("artist", ""), row.get("album", "")] if p)
        self._np_artist.setText(sub)
        self._np_artist.setToolTip(sub)

    def _start_playback(self, row: dict) -> bool:
        """Load and play ``row``; update the page and notify the bar. The
        caller has already set ``_direct_row`` / the queue cursor."""
        path = row["path"]
        self._clear_preview_timers()
        if not self._backend.available:
            self._report_unavailable()
            return False
        error = ""
        try:
            self._backend.load(path)
            ok = self._backend.play()
        except Exception as exc:  # noqa: BLE001
            ok, error = False, str(exc)
        if not ok:
            self._on_track_error(error or "the backend refused to play it")
            return False

        self._apply_volume(self._volume)
        self._vol_apply_timer.start()   # VLC drops volume set before the stream opens

        self._dur_ms = 0
        self._pos_timer.start()

        self._show_now_playing_text(row)
        self._push_recent(row)
        self._load_art(path)
        self._refresh_transport()

        title = row.get("title") or Path(path).stem
        self.now_playing_changed.emit(title, row.get("artist", ""), row.get("album", ""), None)
        self.position_changed.emit(0, 0)
        self._set_playing_state(True)
        self._log(f"Playing: {title}", "ok")

        if self._highlight_chk.isChecked():
            self._begin_preview("checkbox")
        return True

    def _play_directly(self, path: str) -> bool:
        """Play a track immediately without adding it to the queue.

        The queue (waiting room) is untouched — only the ``+ Add Selected``
        button modifies it.  This is the handler for double-click, context
        menu "Play now", Browse, and Recently Played double-click.
        """
        if not path:
            return False
        row = self._row_for(path)
        self._direct_row = row
        self._stopped_row = None
        if not self._start_playback(row):
            if self._direct_row is row:
                self._direct_row = None
            self._refresh_transport()
            return False
        self._refresh_queue_list()
        return True

    # ── Helper: current playing state ─────────────────────────────────────

    def _current_playing_row(self) -> dict | None:
        """The loaded track's row: the direct-play track, else the queue
        cursor's track once it has been started."""
        if self._direct_row is not None:
            return self._direct_row
        if self._cursor_played and 0 <= self._queue_index < len(self._queue):
            return self._queue[self._queue_index]
        return None

    def _is_currently_playing(self, path: str) -> bool:
        """Check if *path* matches the currently playing track."""
        row = self._current_playing_row()
        return row is not None and row["path"] == path

    def _play_current(self, record_history: bool = True) -> bool:
        """Play the track at ``_queue_index`` (queue-based playback)."""
        if not (0 <= self._queue_index < len(self._queue)):
            return False
        if not self._backend.available:
            self._report_unavailable()
            return False
        row = self._queue[self._queue_index]
        self._direct_row = None  # playing from queue — clear direct-play state
        self._stopped_row = None
        self._cursor_played = True

        # Track shuffle history
        if self._shuffle and record_history:
            self._shuffle_history.append(self._queue_index)
            if len(self._shuffle_history) > 200:
                self._shuffle_history = self._shuffle_history[-100:]

        ok = self._start_playback(row)
        self._refresh_queue_list()
        return ok

    # ── 30 s preview (checkbox or keyboard one-shot) ──────────────────────

    def _begin_preview(self, source: str) -> None:
        """After the stream opens, jump to 0:30 and advance 15 s later.
        ``source`` decides what "advance" means (see ``_highlight_advance``)."""
        self._highlight_timer.stop()
        self._highlight_remaining = -1
        self._preview_seek_pending = False
        self._preview_source = source
        self._preview_seek_timer.start(_PREVIEW_SEEK_DELAY_MS)

    @Slot()
    def _on_preview_seek(self) -> None:
        st = self._backend_state()
        if st is PlaybackState.PAUSED:
            self._preview_seek_pending = True   # finish it when playback resumes
            return
        if st not in ACTIVE_STATES:
            return
        try:
            self._backend.set_time_ms(_HIGHLIGHT_START_MS)
        except Exception:  # noqa: BLE001
            pass
        self._highlight_timer.start(_HIGHLIGHT_DURATION_MS)

    def _pause_preview(self) -> None:
        if self._highlight_timer.isActive():
            self._highlight_remaining = max(1, self._highlight_timer.remainingTime())
            self._highlight_timer.stop()
        if self._preview_seek_timer.isActive():
            self._preview_seek_timer.stop()
            self._preview_seek_pending = True

    def _resume_preview(self) -> None:
        if self._preview_seek_pending:
            self._preview_seek_pending = False
            self._preview_seek_timer.start(_PREVIEW_SEEK_DELAY_MS)
        elif self._highlight_remaining > 0:
            self._highlight_timer.start(self._highlight_remaining)
            self._highlight_remaining = -1

    def _clear_preview_timers(self) -> None:
        self._highlight_timer.stop()
        self._preview_seek_timer.stop()
        self._highlight_remaining = -1
        self._preview_seek_pending = False
        self._preview_source = None

    @Slot()
    def _highlight_advance(self) -> None:
        source = self._preview_source
        self._clear_preview_timers()
        if source == "library":
            self._kb_lib_step(+1, preview=False)
        elif source == "queue":
            self._advance()
        else:
            self._on_track_finished()

    # ── Transport ─────────────────────────────────────────────────────────

    @Slot()
    def _on_play_pause(self) -> None:
        if not self._backend.available:
            self._report_unavailable()
            return
        st = self._backend_state()
        if st in (PlaybackState.PLAYING, PlaybackState.OPENING):
            try:
                self._backend.pause()
            except Exception as exc:  # noqa: BLE001
                self._log(f"Could not pause: {exc}", "error")
                return
            self._pause_preview()
            self._pos_timer.stop()
            self._set_playing_state(False)
        elif st is PlaybackState.PAUSED and self._current_playing_row() is not None:
            try:
                ok = self._backend.play()
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                self._on_track_error("could not resume playback")
                return
            self._resume_preview()
            self._pos_timer.start()
            self._set_playing_state(True)
        else:
            self._start_from_idle()
        self._refresh_transport()

    def _start_from_idle(self) -> None:
        """Play pressed with nothing loaded (fresh queue, after Stop, after the end)."""
        n = len(self._queue)
        if 0 <= self._queue_index < n:
            self._play_current()
        elif n:
            self._queue_index = 0
            self._cursor_played = False
            self._play_current()
        elif self._stopped_row is not None:
            self._play_directly(self._stopped_row["path"])

    @Slot()
    def _on_stop(self) -> None:
        if self._backend.available:
            try:
                self._backend.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._direct_row is not None:
            self._stopped_row = self._direct_row
        self._direct_row = None
        self._pos_timer.stop()
        self._vol_apply_timer.stop()
        self._skip_timer.stop()
        self._clear_preview_timers()
        self._refresh_transport()
        self._refresh_queue_list()
        self._set_playing_state(False)
        self.position_changed.emit(0, self._dur_ms)

    def _seek_relative(self, delta_ms: int) -> None:
        if not self._backend.available or self._current_playing_row() is None:
            return
        t = self._backend.time_ms()
        if t >= 0:
            try:
                self._backend.set_time_ms(max(0, t + delta_ms))
            except Exception:  # noqa: BLE001
                pass

    # ── Volume ────────────────────────────────────────────────────────────

    def _apply_volume(self, value: int) -> None:
        """Send *value* (0-100) to the backend, guarded against failures."""
        if self._backend.available:
            try:
                self._backend.set_volume(value)
            except Exception:  # noqa: BLE001
                pass

    @Slot()
    def _reapply_volume(self) -> None:
        self._apply_volume(self._volume)

    def set_volume(self, value: int) -> None:
        """Public API — set the volume (0-100) and apply it to the backend.

        A change is announced on ``volume_changed`` (the capsule's knob
        follows); setting the current value again only re-applies it, so a
        capsule ↔ player sync cannot loop.
        """
        value = max(0, min(100, int(value)))
        self._apply_volume(value)
        if value != self._volume:
            self._volume = value
            self.volume_changed.emit(value)

    def volume(self) -> int:
        return self._volume

    def seek_to_ms(self, ms: int) -> None:
        """Public API — seek to an absolute position in milliseconds."""
        if not self._backend.available or self._current_playing_row() is None:
            return
        try:
            self._backend.set_time_ms(max(0, int(ms)))
        except Exception:  # noqa: BLE001
            pass

    @Slot()
    def _tick_position(self) -> None:
        b = self._backend
        if not b.available:
            self._pos_timer.stop()
            return
        st = self._backend_state()
        dur = b.length_ms()
        pos = b.time_ms()

        if st is PlaybackState.PLAYING:
            self._error_streak = 0
        if dur > 0:
            self._dur_ms = dur
            pos = max(0, pos)
            self.position_changed.emit(pos, dur)

        if st is PlaybackState.ENDED:
            self._pos_timer.stop()
            self._on_track_finished()
        elif st is PlaybackState.ERROR:
            self._pos_timer.stop()
            self._on_track_error("the file could not be opened or decoded")

    def _on_track_finished(self) -> None:
        """The loaded track ended (or its 30 s preview did)."""
        self._clear_preview_timers()
        if self._direct_row is not None:
            if self._queue:
                # Direct play ended — hand over to the queue.
                self._direct_row = None
                self._advance()
            else:
                self._on_stop()
        elif self._repeat == _REPEAT_TRACK and 0 <= self._queue_index < len(self._queue):
            self._play_current(record_history=False)
        else:
            self._advance()

    def _on_track_error(self, message: str) -> None:
        """The backend couldn't play the loaded track: say so and skip it."""
        self._pos_timer.stop()
        self._clear_preview_timers()
        row = self._current_playing_row()
        name = (row.get("title") or Path(row["path"]).name) if row else "track"
        self._error_streak += 1
        self._log(f"Could not play {name}: {message}", "error")
        self._status_lbl.setText(f"Could not play {name} — skipped.")
        limit = max(1, len(self._queue))
        if (
            self._direct_row is not None
            or not self._queue
            or self._repeat == _REPEAT_TRACK
            or self._error_streak >= limit
        ):
            if self._error_streak >= limit and self._queue:
                self._log(
                    f"Stopped: {self._error_streak} track(s) in a row could not be played.",
                    "warn",
                )
            self._error_streak = 0
            self._on_stop()
            return
        # Deferred, so a queue of unplayable files can't recurse.
        self._skip_timer.start()

    # ── Next / Previous ────────────────────────────────────────────────────

    def _advance(self) -> None:
        """Move the queue cursor forward and play. An unplayed cursor (fresh
        queue, or after a direct play) plays itself first."""
        n = len(self._queue)
        if n == 0:
            return
        idx = self._queue_index
        if not self._cursor_played and 0 <= idx < n:
            target = idx
        elif self._shuffle:
            nxt = self._shuffle_next_idx
            target = nxt if nxt is not None and 0 <= nxt < n else random.randrange(n)
            self._shuffle_next_idx = random.randrange(n) if n > 1 else None
        else:
            target = idx + 1
            if target >= n:
                if self._repeat == _REPEAT_ALL:
                    target = 0
                else:
                    # End of the queue: stop and rewind so Play starts over.
                    self._on_stop()
                    self._queue_index = 0
                    self._cursor_played = False
                    self._refresh_queue_list()
                    self._status_lbl.setText("End of queue.")
                    return
        self._queue_index = target
        self._play_current()

    def play_next(self) -> None:
        """Advance to the next queue item (Repeat: Track only affects the end
        of a track, never an explicit Next)."""
        if not self._backend.available:
            self._report_unavailable()
            return
        if not self._queue:
            return
        self._clear_preview_timers()
        self._advance()

    def play_prev(self) -> None:
        """Go back to the previous queue item (or restart current if >3s in)."""
        if not self._backend.available:
            self._report_unavailable()
            return
        self._clear_preview_timers()
        if self._current_playing_row() is not None and self._backend.time_ms() > 3000:
            self._restart_current()
            return
        if not self._queue:
            self._restart_current()
            return
        n = len(self._queue)
        if self._shuffle and len(self._shuffle_history) >= 2:
            self._shuffle_history.pop()
            self._queue_index = self._shuffle_history[-1]
            self._play_current(record_history=False)
            return
        if not self._cursor_played and self._queue_index <= 0:
            self._restart_current()
            return
        if not (0 <= self._queue_index < n):
            self._queue_index = 0
        else:
            self._queue_index = (self._queue_index - 1) % n
        self._play_current()

    # ── Shuffle / Repeat ───────────────────────────────────────────────────

    @Slot(bool)
    def set_shuffle(self, on: bool) -> None:
        """Public API (the capsule's shuffle toggle). Announced on
        ``shuffle_changed``."""
        on = bool(on)
        if on != self._shuffle:
            self._on_shuffle_toggled(on)
        self.shuffle_changed.emit(on)   # also keeps a toggle that disagreed in step

    def shuffle_enabled(self) -> bool:
        return self._shuffle

    def _on_shuffle_toggled(self, checked: bool) -> None:
        self._shuffle = checked
        if checked:
            self._shuffle_history = (
                [self._queue_index] if self._queue_index >= 0 and self._cursor_played else []
            )
            if self._queue and len(self._queue) > 1:
                self._shuffle_next_idx = random.randrange(len(self._queue))
            else:
                self._shuffle_next_idx = None
        else:
            self._shuffle_history.clear()
            self._shuffle_next_idx = None
        self._load_neighbor_art()

    def _cycle_repeat(self) -> None:
        """Off → Track → All → Off (the capsule's repeat button)."""
        self.set_repeat(_REPEAT_NAMES[(self._repeat + 1) % 3])

    def cycle_repeat(self) -> None:
        self._cycle_repeat()

    @Slot(str)
    def set_repeat(self, mode: str) -> None:
        """Public API: "off", "track" or "all". Announced on ``repeat_changed``."""
        by_name = {name: value for value, name in _REPEAT_NAMES.items()}
        self._repeat = by_name.get(mode, _REPEAT_OFF)
        self.repeat_changed.emit(_REPEAT_NAMES[self._repeat])

    def repeat_mode(self) -> str:
        return _REPEAT_NAMES[self._repeat]

    def _on_highlight_toggled(self, checked: bool) -> None:
        if not checked and self._preview_source == "checkbox":
            self._clear_preview_timers()

    # ── Queue management ───────────────────────────────────────────────────

    def _add_paths_to_queue(self, paths: list[str]) -> None:
        if not paths:
            return
        self._queue.extend(self._rows_for(paths))
        if not (0 <= self._queue_index < len(self._queue)):
            self._queue_index = 0
            self._cursor_played = False
        self._refresh_queue_list()
        self._refresh_transport()

    def _add_selected_to_queue(self) -> None:
        paths = self._get_selected_paths()
        if not paths:
            self._status_lbl.setText("Select tracks in the library to add to queue.")
            return
        self._add_paths_to_queue(paths)
        self._status_lbl.setText(f"Added {len(paths)} track(s) to queue.")

    def _clear_queue(self) -> None:
        self._on_stop()
        self._queue.clear()
        self._queue_index = -1
        self._cursor_played = False
        self._stopped_row = None
        self._shuffle_history.clear()
        self._shuffle_next_idx = None
        self._refresh_queue_list()
        self._np_title.setText("Nothing playing")
        self._np_artist.setText("")
        self._covers.clear_all()
        self._bg_art.set_art(None)
        self._refresh_transport()
        # Title "" + null pixmap: nothing is loaded — the bar resets.
        self.now_playing_changed.emit("", "", "", QtGui.QPixmap())

    def _refresh_queue_list(self) -> None:
        self._queue_list.clear()
        playing_idx = (
            self._queue_index
            if self._direct_row is None and self._cursor_played else -1
        )
        accent = QtGui.QColor(get_manager().current.accent)
        for i, row in enumerate(self._queue):
            title = row.get("title") or Path(row["path"]).stem
            artist = row.get("artist", "")
            label = f"{i + 1}. {title}"
            if artist:
                label += f"  –  {artist}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            item.setData(_ROLE_TITLE, title)
            item.setData(_ROLE_ARTIST, artist)
            item.setData(_ROLE_LEAD, str(i + 1))
            item.setToolTip(label)
            if i == playing_idx:
                # The delegate paints the accent and the play glyph; the
                # item's own font/colour say the same to accessibility.
                item.setData(_ROLE_PLAYING, True)
                item.setForeground(accent)
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self._queue_list.addItem(item)
        # Scroll to current
        if 0 <= playing_idx < self._queue_list.count():
            self._queue_list.scrollToItem(self._queue_list.item(playing_idx))
        n = len(self._queue)
        self._q_count_lbl.setText(f"{n} track{'s' if n != 1 else ''}" if n else "")
        self._q_title.setToolTip(self._q_count_lbl.text())
        self._queue_empty.setVisible(n == 0)
        for b in (self._save_m3u_btn, self._clear_queue_btn):
            b.setEnabled(n > 0)

    @Slot(QtCore.QModelIndex)
    def _on_queue_double_click(self, index: QtCore.QModelIndex) -> None:
        item = self._queue_list.item(index.row())
        if item:
            qi = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(qi, int):
                self._jump_queue(qi)

    def _on_queue_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._queue_list.itemAt(pos)
        if not item:
            return
        qi = item.data(QtCore.Qt.ItemDataRole.UserRole)
        ink = _tokens().text_secondary
        menu = QtWidgets.QMenu(self)
        menu.addAction(glyph_icon("play", ink), "Play now", lambda: self._jump_queue(qi))
        menu.addAction(glyph_icon("close", ink), "Remove from queue",
                       lambda: self._remove_from_queue(qi))
        menu.exec(self._queue_list.mapToGlobal(pos))

    def _jump_queue(self, qi: int) -> None:
        if 0 <= qi < len(self._queue):
            self._queue_index = qi
            self._play_current()

    def _remove_from_queue(self, qi: int) -> None:
        if not (0 <= qi < len(self._queue)):
            return
        idx = self._queue_index
        playing_this = (
            qi == idx and self._direct_row is None and self._cursor_played
            and self._backend_state() in ACTIVE_STATES
        )
        removed = self._queue.pop(qi)
        n = len(self._queue)
        if qi < idx:
            self._queue_index = idx - 1
        elif qi == idx:
            if playing_this:
                # Keep playing it, now outside the queue; when it ends the
                # track that moved into its slot plays next.
                self._direct_row = removed
            if n == 0:
                self._queue_index, self._cursor_played = -1, False
            elif qi < n:
                self._cursor_played = False          # the next track slid into qi
            else:
                self._queue_index, self._cursor_played = n - 1, True  # removed the last one

        def _remap(i: int) -> int:
            return i - 1 if i > qi else i

        self._shuffle_history = [_remap(i) for i in self._shuffle_history if i != qi]
        nxt = self._shuffle_next_idx
        if nxt is not None:
            if nxt == qi:
                self._shuffle_next_idx = random.randrange(n) if n > 1 else None
            else:
                self._shuffle_next_idx = _remap(nxt)
        self._refresh_queue_list()
        self._refresh_transport()
        self._load_neighbor_art()

    # ── Recently played ────────────────────────────────────────────────────

    _MAX_RECENT = 30

    def _push_recent(self, row: dict) -> None:
        """Add *row* to the front of the recently-played list (deduped)."""
        path = row.get("path", "")
        self._recent_plays = [r for r in self._recent_plays if r.get("path") != path]
        self._recent_plays.insert(0, row)
        self._recent_plays = self._recent_plays[: self._MAX_RECENT]
        self._refresh_recent_list()

    def _refresh_recent_list(self) -> None:
        self._recent_list.clear()
        for row in self._recent_plays:
            title  = row.get("title") or Path(row["path"]).stem
            artist = row.get("artist", "")
            label  = f"{title}  –  {artist}" if artist else title
            item   = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row["path"])
            item.setData(_ROLE_TITLE, title)
            item.setData(_ROLE_ARTIST, artist)
            item.setToolTip(label)
            self._recent_list.addItem(item)
        n = len(self._recent_plays)
        self._rp_hdr.setText(f"Recently played  ·  {n}" if n else "Recently played")

    def _toggle_recent(self) -> None:
        collapsed = self._recent_list.height() == 0
        self._recent_list.setFixedHeight(_QueueRowDelegate.ROW_H * 3 + 4 if collapsed else 0)
        set_glyph(self._rp_toggle_btn, "chevron_down" if collapsed else "chevron_right",
                  role="muted", size=14)
        self._rp_toggle_btn.setAccessibleName(
            "Hide recently played" if collapsed else "Show recently played")

    @Slot(QtCore.QModelIndex)
    def _on_recent_double_click(self, index: QtCore.QModelIndex) -> None:
        item = self._recent_list.item(index.row())
        if item:
            path = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if path:
                self._play_directly(path)

    # ── File browser ───────────────────────────────────────────────────────

    @Slot()
    def _on_browse_file(self) -> None:
        start = self._library_path or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Audio File", start,
            "Audio (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All Files (*)"
        )
        if path:
            self._play_directly(path)

    # ── Open in Explorer ───────────────────────────────────────────────────

    def _open_in_explorer(self, path: str) -> None:
        import sys, subprocess
        folder = str(Path(path).parent)
        try:
            if sys.platform == "win32":
                # Select the file in Explorer, not just open the folder
                subprocess.Popen(["explorer", "/select,", path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            self._log(f"Could not open folder: {exc}", "warn")

    # ── Album art ──────────────────────────────────────────────────────────

    @staticmethod
    def _cancel_loader(loader: _CancellableThread | None) -> None:
        """Ask an art loader to stop (never blocks; the registry owns it)."""
        if loader is None:
            return
        try:
            if loader.isRunning():
                loader.cancel()
                loader.requestInterruption()
        except RuntimeError:
            pass

    def _start_art_loader(self, path: str, size: int, token: int, slot) -> _ArtLoader:
        loader = _ArtLoader(path, size, token)
        loader.art_ready.connect(slot)
        self._start_worker(loader, "Player cover art", ask_before_quit=False)
        return loader

    def _load_art(self, path: str) -> None:
        for loader in (self._art_loader, self._prev_art_loader, self._next_art_loader):
            self._cancel_loader(loader)
        self._prev_art_loader = self._next_art_loader = None
        self._art_gen += 1
        self._nb_gen += 1
        self._art_loader = self._start_art_loader(path, _ART_SIZE, self._art_gen, self._on_art_ready)

    @Slot(object, str, int)
    def _on_art_ready(self, img_or_none, path: str, token: int = 0) -> None:
        if token != self._art_gen:
            return  # a newer track's art has been requested since
        # Only apply if this is still the current track (queue or direct play)
        current_row = self._current_playing_row()
        if current_row is None or current_row["path"] != path:
            return
        title = current_row.get("title") or Path(path).stem
        if img_or_none is not None and not img_or_none.isNull():
            # _ArtLoader emits QImage — convert to QPixmap on the GUI thread
            pm = QtGui.QPixmap.fromImage(img_or_none)
            self._covers.set_current(pm)
            self._bg_art.set_art(pm)
        else:
            pm = QtGui.QPixmap()   # null = "this track has no art"
            self._covers.set_current(None)
            self._bg_art.set_art(None)
        self.now_playing_changed.emit(
            title, current_row.get("artist", ""), current_row.get("album", ""), pm,
        )
        # Load prev/next neighbor art
        self._load_neighbor_art()

    def _get_prev_next_indices(self) -> tuple[int | None, int | None]:
        """Return (prev_index, next_index) accounting for shuffle mode.

        * Linear mode: prev = queue_index - 1, next = queue_index + 1.
        * Shuffle mode: prev = last item in shuffle_history (before current),
          next = pre-picked _shuffle_next_idx.
        """
        if not self._queue or self._queue_index < 0:
            return None, None
        n = len(self._queue)
        if not self._cursor_played:
            return (self._queue_index - 1 if self._queue_index > 0 else None), self._queue_index

        if self._shuffle:
            # Prev: look back in shuffle history
            prev_idx = None
            if len(self._shuffle_history) >= 2:
                prev_idx = self._shuffle_history[-2]
            # Next: use pre-picked index, or pick one now
            next_idx = self._shuffle_next_idx
            if next_idx is None and n > 1:
                next_idx = random.randrange(n)
                self._shuffle_next_idx = next_idx
            return prev_idx, next_idx
        prev_idx = (self._queue_index - 1) if self._queue_index > 0 else None
        next_idx = (self._queue_index + 1) if self._queue_index + 1 < n else None
        return prev_idx, next_idx

    def _load_neighbor_art(self) -> None:
        """Load art for prev and next tracks in the queue (shuffle-aware)."""
        prev_idx, next_idx = self._get_prev_next_indices()
        self._cancel_loader(self._prev_art_loader)
        self._cancel_loader(self._next_art_loader)
        self._prev_art_loader = self._next_art_loader = None
        self._nb_gen += 1
        token = self._nb_gen

        if prev_idx is not None and 0 <= prev_idx < len(self._queue):
            self._covers.set_prev(None)       # a neighbour; its art follows
            self._prev_art_loader = self._start_art_loader(
                self._queue[prev_idx]["path"], 100, token, self._on_prev_art_ready)
        else:
            self._covers.set_prev(None, present=False)

        if next_idx is not None and 0 <= next_idx < len(self._queue):
            self._covers.set_next(None)
            self._next_art_loader = self._start_art_loader(
                self._queue[next_idx]["path"], 100, token, self._on_next_art_ready)
        else:
            self._covers.set_next(None, present=False)

    @Slot(object, str, int)
    def _on_prev_art_ready(self, img_or_none, path: str, token: int = 0) -> None:
        if token != self._nb_gen:
            return
        ok = img_or_none is not None and not img_or_none.isNull()
        self._covers.set_prev(QtGui.QPixmap.fromImage(img_or_none) if ok else None)

    @Slot(object, str, int)
    def _on_next_art_ready(self, img_or_none, path: str, token: int = 0) -> None:
        if token != self._nb_gen:
            return
        ok = img_or_none is not None and not img_or_none.isNull()
        self._covers.set_next(QtGui.QPixmap.fromImage(img_or_none) if ok else None)

    # ── M3U playlist ───────────────────────────────────────────────────────

    def _save_m3u(self) -> None:
        self._save_paths_as_m3u([r["path"] for r in self._queue])

    def _save_paths_as_m3u(self, paths: list[str]) -> None:
        if not paths:
            QtWidgets.QMessageBox.information(self, "Save M3U", "Queue is empty.")
            return
        playlists_dir = (
            os.path.join(self._library_path, "Playlists")
            if self._library_path else str(Path.home())
        )
        # Nothing is created until the user has picked a destination.
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Playlist", os.path.join(playlists_dir, "Queue.m3u"),
            "M3U Playlist (*.m3u)"
        )
        if not dest:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write("#EXTM3U\n")
                for p in paths:
                    fh.write(f"{p}\n")
            self._status_lbl.setText(f"Saved {len(paths)} tracks to {Path(dest).name}")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save M3U", f"Could not save: {exc}")

    def _load_m3u(self) -> None:
        start = (
            os.path.join(self._library_path, "Playlists")
            if self._library_path and os.path.isdir(
                os.path.join(self._library_path, "Playlists")
            ) else (self._library_path or str(Path.home()))
        )
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Playlist", start,
            "M3U Playlist (*.m3u *.m3u8);;All Files (*)"
        )
        if not chosen:
            return
        try:
            with open(chosen, "rb") as fh:
                text = _decode_m3u(fh.read(), chosen)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Load M3U", f"Could not read file: {exc}")
            return
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
        base = os.path.dirname(chosen)
        paths = []
        for line in lines:
            p = line if os.path.isabs(line) else os.path.normpath(os.path.join(base, line))
            if os.path.isfile(p):
                paths.append(p)
        if not paths:
            QtWidgets.QMessageBox.information(self, "Load M3U", "No valid tracks found.")
            return
        self.load_tracks_and_play(paths, Path(chosen).stem)

    # ── Public API ─────────────────────────────────────────────────────────

    def load_directory_and_play(self, dirpath: str) -> None:
        """Load the audio files in *dirpath* (and, for an artist folder, its
        sub-folders, a few levels deep) as the queue and start playing."""
        skip = {n.casefold() for n in NON_LIBRARY_FOLDERS}
        paths: list[str] = []
        try:
            for root, dirs, files in os.walk(dirpath):
                rel = os.path.relpath(root, dirpath)
                depth = 0 if rel == os.curdir else len(Path(rel).parts)
                dirs[:] = sorted(d for d in dirs if d.casefold() not in skip) \
                    if depth < _DIR_PLAY_MAX_DEPTH else []
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in _AUDIO_EXTS:
                        paths.append(os.path.join(root, f))
                if len(paths) >= _DIR_PLAY_MAX_FILES:
                    break
        except OSError:
            paths = []
        name = Path(dirpath).name or dirpath
        if not paths:
            msg = f"No audio files found in {name}."
            self._status_lbl.setText(msg)
            self._log(msg, "warn")
            return
        self.load_tracks_and_play(paths[:_DIR_PLAY_MAX_FILES], name)

    def load_tracks_and_play(self, paths: list[str], label: str = "") -> None:
        """Replace the queue with *paths* and start playing from the first track.

        Called by other pages (the Sound Map's Map today) and the landing
        page. Replacing a non-empty queue is logged, never silent.
        """
        paths = [p for p in paths if p]
        if not paths:
            return
        if self._queue:
            self._log(
                f"Replaced the queue ({len(self._queue)} track(s)) with "
                f"{label or 'a new selection'} ({len(paths)} track(s)).",
                "info",
            )
        self._on_stop()
        self._queue = self._rows_for(paths)
        self._queue_index = 0
        self._cursor_played = False
        self._stopped_row = None
        self._shuffle_history.clear()
        self._shuffle_next_idx = None
        self._refresh_queue_list()
        if label:
            self._status_lbl.setText(f"Now playing: {label}  ({len(paths)} tracks)")
        self._play_current()
        self._refresh_transport()

    def toggle_shuffle(self) -> None:
        self.set_shuffle(not self._shuffle)

    # ── Event filter for hover popup ────────────────────────────────────

    def eventFilter(self, obj, event):  # noqa: N802
        try:
            viewport = self._lib_table.viewport()
        except RuntimeError:
            # The table is being destroyed with the window; its viewport
            # still receives a few events on the way out.
            return False
        et = event.type()
        if obj is viewport:
            if et == QtCore.QEvent.Type.MouseMove:
                row = self._lib_table.rowAt(int(event.position().y()))
                self._set_hover_row(row)
                path = self._get_lib_path_at(row)
                if path:
                    if path != self._hover_path:
                        self._hover_popup.cancel()
                        self._hover_path = path
                        self._hover_timer.start()
                else:
                    self._cancel_hover()
            elif et in (QtCore.QEvent.Type.Leave, QtCore.QEvent.Type.MouseButtonPress,
                        QtCore.QEvent.Type.Wheel):
                if et == QtCore.QEvent.Type.Leave:
                    self._set_hover_row(-1)
                self._cancel_hover()
        elif et == QtCore.QEvent.Type.Resize:
            if obj is self._lib_table:
                self._fit_table_columns()
            elif obj is self._side:
                self._fit_side_column()
        return super().eventFilter(obj, event)

    def _set_hover_row(self, row: int) -> None:
        if row != self._lib_delegate.hover_row:
            self._lib_delegate.hover_row = row
            self._lib_table.viewport().update()

    def _cancel_hover(self) -> None:
        self._hover_timer.stop()
        self._hover_popup.cancel()
        self._hover_path = ""

    def _on_hover_timeout(self) -> None:
        path = self._hover_path
        if not path or not self.isVisible():
            return
        row = self._rows_by_path.get(path) or _placeholder_row(path)
        self._hover_popup.show_for(row, QtGui.QCursor.pos())
        self._cancel_loader(self._hover_art_loader)
        self._hover_gen += 1
        self._hover_art_loader = self._start_art_loader(
            path, _TrackHoverPopup._ART_SZ, self._hover_gen, self._on_hover_art_ready)

    @Slot(object, str, int)
    def _on_hover_art_ready(self, img_or_none, path: str, token: int = 0) -> None:
        if token == self._hover_gen and self._hover_popup.isVisible():
            self._hover_popup.set_art(path, img_or_none)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        """On screen: pin the capsule (this page's transport) and bring it up
        to date."""
        super().showEvent(event)
        self._refresh_transport()
        self.shuffle_changed.emit(self._shuffle)
        self.repeat_changed.emit(_REPEAT_NAMES[self._repeat])
        self.transport_pinned.emit(True)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802
        self._cancel_hover()
        self._set_hover_row(-1)
        self.transport_pinned.emit(False)
        super().hideEvent(event)

    def _make_focus_tracker(self, target: str):
        """Return an event filter that sets ``_kb_target`` on mouse press."""
        ws = self
        class _Tracker(QtCore.QObject):
            def eventFilter(self, obj, event):  # noqa: N802
                if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                    ws._kb_target = target
                return False
        return _Tracker(self)

    def set_keyboard_mode(self, active: bool) -> None:
        """Enable or disable arrow-key navigation mode (toggled from NowPlayingBar)."""
        self._keyboard_mode = active
        if active:
            self._log("Keyboard mode ON: ↓ preview  → next  ← prev  ↑ restart", "ok")
        else:
            self._log("Keyboard mode OFF", "info")

    # ── Keyboard-mode: library navigation ──────────────────────────────────

    def kb_target(self) -> str:
        """Return the current keyboard navigation target ('library' or 'queue')."""
        return self._kb_target

    def _get_lib_path_at(self, row: int) -> str | None:
        """Return the file path for a visible library table row, or None."""
        if row < 0 or row >= self._lib_table.rowCount():
            return None
        item = self._lib_table.item(row, 1) or self._lib_table.item(row, 0)
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _play_lib_row(self, row: int, preview: bool = False) -> None:
        """Play a library table row by index, optionally as a one-shot preview."""
        path = self._get_lib_path_at(row)
        if not path:
            return
        self._lib_kb_row = row
        if self._play_directly(path) and preview:
            # The preview's "next" is the next *library* row, not the queue.
            self._begin_preview("library")

    def _kb_lib_step(self, step: int, preview: bool) -> None:
        total = self._lib_table.rowCount()
        if total == 0:
            return
        if self._lib_kb_row >= 0:
            start = self._lib_kb_row
        else:
            start = -1 if step > 0 else 0
        for offset in range(1, total + 1):
            r = (start + step * offset) % total
            if not self._lib_table.isRowHidden(r):
                self._play_lib_row(r, preview=preview)
                return

    def kb_lib_down(self) -> None:
        """↓ Preview next library track from the last-played position."""
        self._kb_lib_step(+1, preview=True)

    def kb_lib_right(self) -> None:
        """→ Play next library track in full."""
        self._kb_lib_step(+1, preview=False)

    def kb_lib_left(self) -> None:
        """← Play previous library track."""
        self._kb_lib_step(-1, preview=False)

    def kb_lib_up(self) -> None:
        """↑ Restart current track (same for library and queue)."""
        self._restart_current()

    # ── Keyboard mode arrow-key handling ───────────────────────────────────

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handle arrow keys when keyboard mode is active.

        • Down  — play next track as a one-shot 30 s preview
        • Right — skip to next track (full play)
        • Left  — go to previous track
        • Up    — restart current track from the beginning
        """
        if self._keyboard_mode:
            key = event.key()
            if key == QtCore.Qt.Key.Key_Down:
                self._preview_next()
                return
            elif key == QtCore.Qt.Key.Key_Right:
                self.play_next()
                return
            elif key == QtCore.Qt.Key.Key_Left:
                self.play_prev()
                return
            elif key == QtCore.Qt.Key.Key_Up:
                self._restart_current()
                return
        super().keyPressEvent(event)

    def _preview_next(self) -> None:
        """Advance to the next queue track with a one-shot 30 s preview."""
        if not self._queue:
            self._log("Queue is empty — nothing to preview.", "warn")
            return
        if not self._backend.available:
            self._report_unavailable()
            return
        self._clear_preview_timers()
        self._advance()
        if self._current_playing_row() is not None and \
                self._backend_state() in ACTIVE_STATES:
            self._begin_preview("queue")

    def _restart_current(self) -> None:
        """Restart the current track from 0:00."""
        if not self._backend.available or self._current_playing_row() is None:
            return
        try:
            self._backend.set_time_ms(0)
        except Exception:  # noqa: BLE001
            pass

    def toggle_repeat(self) -> None:
        self.cycle_repeat()

    # ── Shutdown ───────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """App is closing: stop timers, cancel workers, release the backend."""
        for timer in (self._pos_timer, self._highlight_timer, self._preview_seek_timer,
                      self._vol_apply_timer, self._skip_timer, self._hover_timer):
            timer.stop()
        self._hover_popup.cancel()
        self.cancel_jobs()
        try:
            self._backend.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._backend.release()
        except Exception:  # noqa: BLE001
            pass
        super().shutdown()
