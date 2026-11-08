"""Exports and File Cleanup panels, and the tool tiles they are built from.

These were one "Utilities" page; the information architecture split it up:

* ``ExportsWorkspace``  — Exports page (Take away): Artist/Title export and
  Codec list export.
* ``FileCleanupWorkspace`` — File Cleanup page (Organize): plan → preview →
  confirm → rename → log, with "Undo Last Cleanup".

The rest of Utilities moved too: the library Validator is Home's "Library
health" card, the duplicate prototypes are sections of the Duplicates page
(``gui/workspaces/launchers.py``), the M4A/Opus testers are Mirror → Verify
the mirror, and the crash log is on the Activity page.
"""
from __future__ import annotations

import os
import threading
from collections import Counter
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase
from gui.workspaces.label_tone import flash_tone, refresh_tones, set_tone
from utils.reserved_folders import NON_LIBRARY_FOLDERS, walk_library

_SUPPORTED_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour + alpha float to a CSS rgba() string.

    Qt stylesheets do not reliably support 8-digit hex (#rrggbbaa), so
    this helper is used everywhere an alpha-tinted colour is needed.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


# ── Background workers ─────────────────────────────────────────────────────────

def _read_title(path: str) -> object:
    """Title tag of ``path`` (or None) \u2014 lets File Cleanup keep real titles
    such as "Love Song (2)" whose suffix is part of the tag."""
    from utils.audio_metadata_reader import read_tags
    return read_tags(path).get("title")


def _update_playlists(rename_map: dict[str, str], library_root: str) -> int:
    """Rewrite playlists for renamed files; returns how many playlists changed.

    The root is passed explicitly (TOOL-03: deriving it from the renamed paths
    found ``<album>/Playlists`` and silently did nothing).
    """
    from playlist_generator import update_playlists
    return update_playlists(rename_map, library_root=library_root)


class FileCleanupPlanWorker(QtCore.QThread):
    """Dry run: compute the File Cleanup rename plan. Touches nothing."""

    plan_ready = Signal(object)   # file_cleanup.CleanupPlan
    error = Signal(str)

    def __init__(self, library_path: str, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.library_path = library_path
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            from file_cleanup import build_cleanup_plan
            plan = build_cleanup_plan(
                self.library_path, read_title=_read_title, cancel_event=self._cancel
            )
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return
        self.plan_ready.emit(plan)


class FileCleanupWorker(QtCore.QThread):
    """Execute exactly a previewed File Cleanup plan, then patch playlists.

    With ``undo=True`` the plan is a ``file_cleanup.UndoPlan``: the files get
    their original names back and the playlists that cleanup re-pointed are
    pointed back at them.
    """

    log_line = Signal(str)
    finished = Signal(bool, str)   # (success, summary message)
    result_ready = Signal(object)  # file_cleanup.CleanupResult

    def __init__(self, plan, parent: QtCore.QObject | None = None, *, undo: bool = False) -> None:
        super().__init__(parent)
        self.plan = plan
        self.undo = undo
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _on_item(self, item, outcome: str) -> None:
        old, new = os.path.basename(item.src), os.path.basename(item.dst)
        if outcome == "renamed":
            self.log_line.emit(f"\u2192 {old} \u2192 {new}")
        else:
            self.log_line.emit(f"! {old}: {outcome}")

    def run(self) -> None:
        from file_cleanup import execute_cleanup_plan, execute_undo
        execute = execute_undo if self.undo else execute_cleanup_plan
        try:
            result = execute(self.plan, cancel_event=self._cancel, on_item=self._on_item)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, f"{'Undo' if self.undo else 'Cleanup'} failed: {exc}")
            return

        playlist_note = ""
        if result.renamed:
            try:
                changed = _update_playlists(result.renamed, self.plan.library_root)
            except Exception as exc:  # noqa: BLE001
                playlist_note = f" Playlist update failed: {exc}."
                self.log_line.emit(f"! Playlist update failed: {exc}")
            else:
                if isinstance(changed, int) and not isinstance(changed, bool):
                    playlist_note = f" {changed} playlist(s) updated."
                else:
                    playlist_note = " Playlists checked."
                self.log_line.emit(playlist_note.strip())

        if self.undo:
            parts = [f"{len(result.renamed)} original name(s) restored"]
            if result.failed:
                parts.append(f"{len(result.failed)} not restored")
            if result.skipped_collisions:
                parts.append(f"{result.skipped_collisions} skipped")
            head = "Undo cancelled" if result.cancelled else "Undone"
        else:
            parts = [f"{len(result.renamed)} renamed"]
            if result.failed:
                parts.append(f"{len(result.failed)} not renamed")
            if result.skipped_collisions:
                parts.append(f"{result.skipped_collisions} collisions skipped")
            head = "Cancelled" if result.cancelled else "Done"
        message = f"{head}: {', '.join(parts)}.{playlist_note}"
        if result.log_path:
            message += f" {'Log' if self.undo else 'Undo log'}: {result.log_path}"
        self.result_ready.emit(result)
        self.finished.emit(not result.failed and not result.cancelled, message)


class ArtistTitleWorker(QtCore.QThread):
    """Background thread for the Artist/Title export."""

    progress = Signal(int, int)       # (done, total)
    log_line = Signal(str)
    finished = Signal(str, int, int)  # (output_path, entry_count, error_count)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        library_path: str,
        exclude_flac: bool,
        add_album_duplicates: bool,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.library_path = library_path
        self.exclude_flac = exclude_flac
        self.add_album_duplicates = add_album_duplicates
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @staticmethod
    def _clean(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    return value.decode(enc).strip() or None
                except UnicodeDecodeError:
                    continue
            return value.decode("utf-8", errors="replace").strip() or None
        if isinstance(value, (list, tuple)):
            for item in value:
                cleaned = ArtistTitleWorker._clean(item)
                if cleaned:
                    return cleaned
            return None
        cleaned = str(value).strip()
        return cleaned or None

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def _run(self) -> None:
        from utils.audio_metadata_reader import read_tags
        from utils.opus_metadata_reader import read_opus_metadata

        audio_files: list[str] = []
        # Quarantined duplicate losers, Trash and Docs are not library tracks.
        for dirpath, _, files in walk_library(self.library_path, NON_LIBRARY_FOLDERS):
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if self.exclude_flac and ext == ".flac":
                    continue
                if ext in _SUPPORTED_EXTS:
                    audio_files.append(os.path.join(dirpath, filename))

        total = len(audio_files)
        self.log_line.emit(f"Found {total} audio files to process.")

        entry_data: list[tuple[str, str, str | None, str | None]] = []
        error_count = 0

        for idx, full_path in enumerate(audio_files, start=1):
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            ext = os.path.splitext(full_path)[1].lower()
            filename = os.path.basename(full_path)

            if ext == ".opus":
                tags, _covers, read_error = read_opus_metadata(full_path)
                if read_error:
                    error_count += 1
                    self.log_line.emit(f"Skipped unreadable OPUS file: {filename}")
            else:
                tags = read_tags(full_path)

            artist = self._clean(tags.get("artist") or tags.get("albumartist"))
            title  = self._clean(tags.get("title"))
            album  = self._clean(tags.get("album"))
            track_raw = self._clean(tags.get("tracknumber") or tags.get("track"))
            if track_raw and "/" in track_raw:
                track_raw = track_raw.split("/", 1)[0].strip() or None

            if not title:
                title = os.path.splitext(filename)[0]
            if not artist:
                artist = "Unknown Artist"

            entry_data.append((artist, title, album, track_raw))

            if idx == 1 or idx % 50 == 0 or idx == total:
                self.progress.emit(idx, total)

        entries: list[str] = []
        duplicate_counts = Counter(
            (artist, title) for artist, title, _album, _track in entry_data
        )
        album_dup_counts = Counter(
            (artist, title, album) for artist, title, album, _track in entry_data
        )

        for artist, title, album, track in entry_data:
            if self.add_album_duplicates and duplicate_counts[(artist, title)] > 1:
                album_label = album or "Unknown Album"
                if album_dup_counts[(artist, title, album)] > 1:
                    track_label = track or "Unknown Track"
                    entries.append(f"{artist} - {title} - {album_label} - {track_label}")
                else:
                    entries.append(f"{artist} - {title} - {album_label}")
            else:
                entries.append(f"{artist} - {title}")

        entries = sorted(set(entries), key=str.lower)

        out = Path(self.library_path) / "Docs" / "artist_title_list.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(entries), encoding="utf-8")

        self.finished.emit(str(out), len(entries), error_count)


class CodecExportWorker(QtCore.QThread):
    """Background thread for the Codec List export (was a GUI-thread walk)."""

    finished = Signal(str, int)   # (output_path, file_count)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        library_path: str,
        extensions: set[str],
        omit_paths: bool,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.library_path = library_path
        self.extensions = set(extensions)
        self.omit_paths = omit_paths
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            by_ext: dict[str, list[str]] = {e: [] for e in sorted(self.extensions)}
            total = 0
            for dirpath, _, files in walk_library(self.library_path, NON_LIBRARY_FOLDERS):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in by_ext:
                        by_ext[ext].append(f if self.omit_paths else os.path.join(dirpath, f))
                        total += 1

            out = Path(self.library_path) / "Docs" / "codec_file_list.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            for ext in sorted(by_ext):
                lines.append(f"=== {ext} ({len(by_ext[ext])} files) ===")
                lines.extend(sorted(by_ext[ext]))
                lines.append("")
            out.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return
        self.finished.emit(str(out), total)


class _GlassBadge(QtWidgets.QWidget):
    def __init__(self, icon: str, size: int = 40, parent=None):
        super().__init__(parent)
        self._icon = icon
        self.setFixedSize(size, size)
        from gui.themes.manager import get_manager
        # Bound method (not a lambda) so the singleton drops the connection
        # when this widget is destroyed.
        get_manager().theme_changed.connect(self._repaint_on_theme)


    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()
    def set_size(self, size: int) -> None:
        self.setFixedSize(size, size)
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = get_manager().current
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

    # Theme token per state (read at paint time, so theme switches apply).
    _TOKEN = {
        NEUTRAL: "text_muted",
        SUCCESS: "success",
        ERROR:   "danger",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text  = "\u29d6  Idle"
        self._state = self.NEUTRAL
        self._opacity = 1.0
        self._pending: tuple[str, int] | None = None
        # Two long-lived, parented animations reused for every transition.
        # (A stored reference to a DeleteWhenStopped animation dangled after
        # the first transition, so the second one raised RuntimeError \u2014 DS-06.)
        self._fade_out = QtCore.QVariantAnimation(self)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setDuration(160)
        self._fade_out.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._fade_out.valueChanged.connect(self._on_opacity)
        self._fade_out.finished.connect(self._swap)
        self._fade_in = QtCore.QVariantAnimation(self)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setDuration(260)
        self._fade_in.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._fade_in.valueChanged.connect(self._on_opacity)
        self.setFixedHeight(26)
        self._resize()
        from gui.themes.manager import get_manager
        # Bound method (not a lambda) so the singleton drops the connection
        # when this widget is destroyed.
        get_manager().theme_changed.connect(self._repaint_on_theme)


    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()
    def _resize(self):
        fm = QtGui.QFontMetrics(self.font())
        self.setFixedWidth(fm.horizontalAdvance(self._text) + 34)

    def set_text_fast(self, text: str, state: int | None = None):
        self._text = text
        if state is not None:
            self._state = state
        self._resize()
        self.update()

    def show_neutral(self, text: str = "\u29d6  Idle"):
        self._transition(text, self.NEUTRAL)

    def show_success(self, text: str = "\u2713  Done"):
        self._transition(text, self.SUCCESS)

    def show_error(self, text: str = "\u2717  Error"):
        self._transition(text, self.ERROR)

    @property
    def text(self) -> str:
        """The text the chip shows (or is transitioning to)."""
        return self._pending[0] if self._pending else self._text

    @property
    def state(self) -> int:
        return self._pending[1] if self._pending else self._state

    def _transition(self, text: str, state: int):
        self._pending = (text, state)
        self._fade_in.stop()
        if self._fade_out.state() == QtCore.QAbstractAnimation.State.Running:
            return  # the running fade-out will swap in the newest text
        if not self.isVisible():
            self._swap()        # nothing to animate on a hidden chip
            self._set_op(1.0)
            return
        self._fade_out.setStartValue(self._opacity)
        self._fade_out.start()

    def _swap(self) -> None:
        if self._pending is None:
            return
        self._text, self._state = self._pending
        self._pending = None
        self._resize()
        if self.isVisible():
            self._fade_in.start()
        else:
            self._set_op(1.0)

    def _on_opacity(self, v) -> None:
        self._set_op(float(v))

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
        ac   = QtGui.QColor(getattr(get_manager().current, self._TOKEN[self._state]))

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


# ── ToolTile ───────────────────────────────────────────────────────────────────

class ToolTile(QtWidgets.QFrame):
    """Self-contained tool card: header ▸ options ▸ footer ▸ animated drawer.

    Usage
    -----
    tile = ToolTile("↗", "My Tool", "Short description.")
    tile.add_option(some_checkbox)
    tile.set_run_button("Run", self._my_slot)
    secondary = tile.add_secondary_button("Open File")
    tile.finish_footer()

    When an operation starts call tile.open_drawer() to reveal the progress
    area and tile.set_running(True/False) to lock/unlock footer buttons.
    """

    def __init__(
        self,
        icon: str,
        title: str,
        description: str,
        *,
        log_height: int = 110,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("liquidGlassTile")
        self.setStyleSheet("QFrame#liquidGlassTile { background: transparent; border: none; }")
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self._hover_t = 0.0
        self._time = 0.0
        self._target_pos = QtCore.QPointF(200, 100)
        self._glow_pos = QtCore.QPointF(200, 100)
        self._is_hovered = False

        self._footer_buttons: list[QtWidgets.QPushButton] = []
        self._drawer_open = False

        # Hover-glow animation timer. It only runs while the glow is actually
        # moving (hover, mouse move, settling back after leave) and never while
        # the tile is hidden — it used to tick at 60 Hz forever (DS-09/TOOL-18).
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        header = QtWidgets.QWidget()
        header_l = QtWidgets.QVBoxLayout(header)
        header_l.setContentsMargins(18, 16, 18, 12)
        header_l.setSpacing(4)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(12)
        
        self._badge = _GlassBadge(icon, size=40)
        title_row.addWidget(self._badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(6)
        
        self._title_lbl = QtWidgets.QLabel(title)
        self._title_lbl.setObjectName("cardTitle")
        info.addWidget(self._title_lbl)

        self._desc_lbl: QtWidgets.QLabel | None = None
        if description:
            self._desc_lbl = QtWidgets.QLabel(description)
            self._desc_lbl.setObjectName("sectionSubtitle")
            self._desc_lbl.setWordWrap(True)
            self._desc_lbl.setContentsMargins(0, 2, 0, 2)
            info.addWidget(self._desc_lbl)
            
        title_row.addLayout(info, 1)

        self.status_chip = _GlassResultChip()
        self.status_chip.show_neutral("Ready.")
        title_row.addWidget(self.status_chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        header_l.addLayout(title_row)
        outer.addWidget(header)

        # ── Options area ──────────────────────────────────────────────────
        self._options = QtWidgets.QWidget()
        self._opts_l = QtWidgets.QVBoxLayout(self._options)
        self._opts_l.setContentsMargins(18, 0, 18, 12)
        self._opts_l.setSpacing(8)
        outer.addWidget(self._options)

        # ── Footer ────────────────────────────────────────────────────────
        self._footer = QtWidgets.QWidget()
        self._footer_l = QtWidgets.QHBoxLayout(self._footer)
        self._footer_l.setContentsMargins(18, 8, 18, 16)
        self._footer_l.setSpacing(8)
        outer.addWidget(self._footer)

        # ── Separator (hidden until drawer opens) ─────────────────────────
        self._sep = QtWidgets.QFrame()
        self._sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self._sep.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self._sep.setVisible(False)
        outer.addWidget(self._sep)

        # ── Drawer ────────────────────────────────────────────────────────
        self._drawer = QtWidgets.QWidget()
        drawer_l = QtWidgets.QVBoxLayout(self._drawer)
        drawer_l.setContentsMargins(18, 10, 18, 16)
        drawer_l.setSpacing(6)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)

        # Hidden but retained so existing tool logic doesn't crash on .setText()
        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setVisible(False)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(log_height)
        self.log_box.setObjectName("logBox")
        self.log_box.setStyleSheet("QPlainTextEdit { background: rgba(0, 0, 0, 0.1); border-radius: 6px; }")

        drawer_l.addWidget(self.progress_bar)
        drawer_l.addWidget(self.status_label)
        drawer_l.addWidget(self.log_box)

        self._drawer.setMinimumHeight(0)
        self._drawer.setMaximumHeight(0)
        outer.addWidget(self._drawer)

        # ── Theme ─────────────────────────────────────────────────────────
        self._apply_sep_color(get_manager().current)
        get_manager().theme_changed.connect(self._on_theme_changed)

    def mouseMoveEvent(self, e):
        self._target_pos = e.position()
        self._ensure_ticking()
        super().mouseMoveEvent(e)

    def enterEvent(self, e):
        self._is_hovered = True
        self._ensure_ticking()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._is_hovered = False
        self._ensure_ticking()
        super().leaveEvent(e)

    def hideEvent(self, e):  # noqa: N802
        self._timer.stop()
        self._is_hovered = False
        super().hideEvent(e)

    def showEvent(self, e):  # noqa: N802
        super().showEvent(e)
        # Resume settling if the glow was left mid-way when the page was hidden.
        if self._hover_t > 0.001:
            self._ensure_ticking()

    def _ensure_ticking(self) -> None:
        if self.isVisible() and not self._timer.isActive():
            self._timer.start()

    @property
    def glow_animating(self) -> bool:
        return self._timer.isActive()

    def _tick(self):
        target_h = 1.0 if self._is_hovered else 0.0
        hover_delta = (target_h - self._hover_t) * 0.15
        self._hover_t += hover_delta

        if not self._is_hovered:
            self._target_pos = QtCore.QPointF(self.width() / 2, self.height() / 2)

        dx = self._target_pos.x() - self._glow_pos.x()
        dy = self._target_pos.y() - self._glow_pos.y()
        self._glow_pos.setX(self._glow_pos.x() + dx * 0.15)
        self._glow_pos.setY(self._glow_pos.y() + dy * 0.15)

        # Converged: snap to the target and stop the timer until the next
        # enter / move / leave event.
        if abs(hover_delta) < 0.001 and abs(dx) < 0.5 and abs(dy) < 0.5:
            self._hover_t = target_h
            self._glow_pos = QtCore.QPointF(self._target_pos)
            self._timer.stop()
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        rect = QtCore.QRectF(self.rect())
        inner = rect.adjusted(2, 2, -2, -2)
        radius = 16.0
        ht = self._hover_t
        ac = QtGui.QColor(t.accent)

        path = QtGui.QPainterPath()
        path.addRoundedRect(inner, radius, radius)
        p.save()
        p.setClipPath(path)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        bg_alpha = 0.05 + ht * 0.03
        p.setBrush(QtGui.QColor(255, 255, 255, int(bg_alpha * 255)))
        p.drawRect(inner)

        t_alpha = 0.1 + ht * 0.05
        # ThemeTokens has ``variant`` ("dark"/"light"), not ``is_dark``: the old
        # getattr fallback gave every light theme the dark tint (DS-08/TOOL-15).
        if t.variant == "dark":
            tint_color = QtGui.QColor(0, 0, 0, int(t_alpha * 255))
        else:
            tint_color = QtGui.QColor(255, 255, 255, int(t_alpha * 255))
        p.setBrush(tint_color)
        p.drawRect(inner)

        glow_rad = inner.width() * 0.8
        glow = QtGui.QRadialGradient(self._glow_pos, glow_rad)
        c1 = QtGui.QColor(ac); c1.setAlphaF(0.15 + ht * 0.15)
        c2 = QtGui.QColor(ac); c2.setAlphaF(0.0)
        glow.setColorAt(0.0, c1)
        glow.setColorAt(1.0, c2)
        p.setBrush(glow)
        p.drawRect(inner)

        spec = QtGui.QLinearGradient(0, inner.top(), 0, inner.bottom())
        spec.setColorAt(0.0, QtGui.QColor(255, 255, 255, int((0.15 + ht * 0.05) * 255)))
        spec.setColorAt(0.4, QtGui.QColor(255, 255, 255, 0))
        p.setBrush(spec)
        p.drawRect(inner)
        p.restore()
        
        p.end()

    # ── Public API ────────────────────────────────────────────────────────

    def add_option(self, widget: QtWidgets.QWidget) -> None:
        """Add a widget to the options section."""
        self._opts_l.addWidget(widget)

    def add_option_layout(self, layout: QtWidgets.QLayout) -> None:
        """Add a sub-layout to the options section."""
        self._opts_l.addLayout(layout)

    def set_run_button(self, label: str, slot) -> QtWidgets.QPushButton:
        """Add the primary (accent-coloured) action button and return it."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(label)
        btn.setObjectName("primaryBtn")
        btn.setMinimumHeight(34)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.clicked.connect(slot)
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def add_secondary_button(self, label: str) -> QtWidgets.QPushButton:
        """Add a secondary (default-style) button and return it."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(label)
        btn.setMinimumHeight(34)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def add_icon_button(self, icon: str, tooltip: str = "") -> QtWidgets.QPushButton:
        """Add a small square icon-only button and return it."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(icon)
        btn.setFixedSize(34, 34)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        if tooltip:
            btn.setToolTip(tooltip)
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def finish_footer(self) -> None:
        """Add trailing stretch after all buttons. Call last."""
        self._footer_l.addStretch(1)

    def hide_footer(self) -> None:
        """Hide the footer row entirely (e.g. for tiles with no run action)."""
        self._footer.setVisible(False)

    def open_drawer(self) -> None:
        """Animate the log drawer from 0 → its natural height (once)."""
        if self._drawer_open:
            return
        self._drawer_open = True
        self._sep.setVisible(True)
        target = max(self._drawer.sizeHint().height(), 100)
        anim = QtCore.QPropertyAnimation(self._drawer, b"maximumHeight", self)
        anim.setDuration(250)
        anim.setStartValue(0)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self._drawer.setMaximumHeight(16777215))
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def set_running(self, running: bool) -> None:
        """Enable or disable all footer buttons (call during long operations)."""
        for btn in self._footer_buttons:
            btn.setEnabled(not running)

    def flash_status(self, from_tone: str, duration: int = 1600) -> None:
        """Fade status_label from a theme tone ("success", "danger", …) back to
        the secondary text colour (one reused animation; palette only)."""
        flash_tone(self.status_label, from_tone, "secondary", duration)

    def report(self, state: str, text: str, tone: str | None = None) -> None:
        """Show a run's outcome on the status chip and the status label.

        ``state`` is "running", "ok", "error", "cancelled" or "idle".
        (The chip used to say "Ready." forever — TOOL-17.)
        """
        chip = self.status_chip
        if state == "ok":
            chip.show_success("✓  Done")
        elif state == "error":
            chip.show_error("✗  Error")
        elif state == "running":
            chip.show_neutral("⧖  Running…")
        elif state == "cancelled":
            chip.show_neutral("Cancelled")
        else:
            chip.show_neutral("Ready.")
        self.status_label.setText(text)
        if tone:
            self.flash_status(tone)
        else:
            set_tone(self.status_label, "secondary")

    # ── Text scale ────────────────────────────────────────────────────────

    _TITLE_PT = {1: 10, 2: 11, 3: 12, 4: 14, 5: 16}
    _DESC_PT  = {1:  8, 2:  9, 3: 10, 4: 11, 5: 13}
    _BADGE_SZ = {1: 30, 2: 34, 3: 40, 4: 46, 5: 52}

    def set_text_scale(self, level: int) -> None:
        """Adjust font sizes and badge icon to match the given 1–5 level."""
        if level not in self._TITLE_PT:
            return
        f = self._title_lbl.font()
        f.setPointSize(self._TITLE_PT[level])
        self._title_lbl.setFont(f)
        if self._desc_lbl is not None:
            f2 = self._desc_lbl.font()
            f2.setPointSize(self._DESC_PT[level])
            self._desc_lbl.setFont(f2)
        self._badge.set_size(self._BADGE_SZ[level])

    # ── Internal ──────────────────────────────────────────────────────────

    def _apply_sep_color(self, tokens) -> None:
        self._sep.setStyleSheet(f"QFrame {{ color: {tokens.card_border}; }}")

    def _on_theme_changed(self, tokens) -> None:
        self._apply_sep_color(tokens)



# ── TileGrid ───────────────────────────────────────────────────────────────────

class TileGrid(QtWidgets.QWidget):
    """Responsive grid container for ToolTile instances.

    Column count is computed from available width divided by _TILE_MIN_WIDTH.
    Full-width tiles span all columns and always start a fresh row.

    Adding a new tool tile:
        self._tile_grid.add_tile(self._build_new_tile(), full_width=False)
    """

    _TILE_MIN_WIDTH = 500   # px — 2 columns from ~1050 px wide, 1 column on narrow windows

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tiles: list[tuple[ToolTile, bool]] = []   # (tile, is_full_width)
        self._current_cols: int = 2                      # default; updated on resize

        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setSpacing(16)
        self._grid.setContentsMargins(0, 0, 0, 0)

    def add_tile(self, tile: ToolTile, full_width: bool = False) -> None:
        """Register a tile and add it to the current grid layout."""
        self._tiles.append((tile, full_width))
        self._reflow()

    def set_size_level(self, level: int) -> None:
        for tile, _ in self._tiles:
            tile.set_text_scale(level)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        new_cols = max(1, event.size().width() // self._TILE_MIN_WIDTH)
        if new_cols != self._current_cols:
            self._current_cols = new_cols
            self._reflow()
        super().resizeEvent(event)

    def _reflow(self) -> None:
        """Re-place all tiles in the grid with the current column count."""
        cols = self._current_cols

        # Detach every tile from the layout without destroying it
        for tile, _ in self._tiles:
            self._grid.removeWidget(tile)

        col = row = 0
        for tile, full_width in self._tiles:
            if full_width:
                if col > 0:          # push to a new row first
                    row += 1
                    col = 0
                self._grid.addWidget(tile, row, 0, 1, cols)
                row += 1
            else:
                self._grid.addWidget(tile, row, col)
                col += 1
                if col >= cols:
                    col = 0
                    row += 1

        # Columns beyond the current count keep no stretch: going from two
        # columns to one used to leave an empty, stretching second column,
        # so every tile (full-width ones too) took half the page.
        for c in range(max(cols, self._grid.columnCount())):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)


# ── _PillToggle ────────────────────────────────────────────────────────────────

class _PillToggle(QtWidgets.QAbstractButton):
    """Animated iOS-style pill toggle switch.

    Drop-in replacement for QCheckBox for single on/off options.
    isChecked() / toggled signal / setChecked() all work identically.
    """

    _TRACK_W = 38
    _TRACK_H = 22
    _KNOB_D  = 16
    _GAP     = 10   # px between track right edge and label text

    def __init__(self, label: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(label)
        self._knob_x: float = 0.0          # 0.0 = off, 1.0 = on
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(lambda v: self._set_knob(float(v)))
        self.toggled.connect(self._start_anim)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        # Bound method (not a lambda) so the singleton drops the connection
        # when this widget is destroyed.
        get_manager().theme_changed.connect(self._repaint_on_theme)

    # ── Animation ──────────────────────────────────────────────────────────


    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()
    def _start_anim(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob_x)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _set_knob(self, v: float) -> None:
        self._knob_x = v
        self.update()

    # ── Sizing ─────────────────────────────────────────────────────────────

    def sizeHint(self) -> QtCore.QSize:
        label = self.text()
        w = self._TRACK_W
        h = self._TRACK_H
        if label:
            fm = self.fontMetrics()
            w += self._GAP + fm.horizontalAdvance(label)
            h = max(h, fm.height() + 4)
        return QtCore.QSize(w, h)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, e) -> None:  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        tw = float(self._TRACK_W)
        th = float(self._TRACK_H)
        kd = float(self._KNOB_D)
        cy = self.height() / 2.0

        # Track
        tr = QtCore.QRectF(0, cy - th / 2, tw, th)
        if self.isChecked():
            ac = QtGui.QColor(t.accent)
            track_col = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 220)
        else:
            track_col = QtGui.QColor(t.card_border)
        p.setBrush(track_col)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(tr, th / 2, th / 2)

        # Knob (white circle)
        margin = (th - kd) / 2.0
        travel = tw - kd - 2 * margin
        kx = margin + self._knob_x * travel
        ky = cy - kd / 2.0
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawEllipse(QtCore.QRectF(kx, ky, kd, kd))

        # Label text
        label = self.text()
        if label:
            p.setPen(QtGui.QColor(t.text_primary))
            f = p.font()
            f.setPointSize(10)
            p.setFont(f)
            lx = int(tw) + self._GAP
            p.drawText(
                lx, 0, self.width() - lx, self.height(),
                int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft),
                label,
            )
        p.end()


# ── Shared base for the tool pages ────────────────────────────────────────────

class _ToolPanel(WorkspaceBase):
    """Helpers shared by the Exports and File Cleanup panels."""

    def _flash_label(self, label: QtWidgets.QLabel, from_tone: str, duration: int = 1600) -> None:
        """Fade ``label`` from a theme tone back to the secondary text colour.

        One reused animation per label, palette only — it used to restyle the
        label every frame, and overlapping flashes fought each other (DS-20).
        """
        flash_tone(label, from_tone, "secondary", duration)

    # ── Theme refresh ──────────────────────────────────────────────────────

    def _on_theme_changed_base(self, tokens: object) -> None:
        super()._on_theme_changed_base(tokens)
        refresh_tones(self, tokens)

    def _require_library(self) -> bool:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return False
        return True

    # ── Utility ───────────────────────────────────────────────────────────

    def _open_file(self, path: str | None) -> None:
        import subprocess
        import sys
        if not path:
            return
        if sys.platform == "win32":
            import os as _os
            _os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _open_folder(self, path: str | None) -> None:
        """Open the folder containing *path* in the system file manager."""
        import subprocess
        import sys
        if not path:
            return
        folder = str(Path(path).parent)
        if sys.platform == "win32":
            # /select highlights the file inside the folder
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", folder])


# ── Exports ────────────────────────────────────────────────────────────────────

class ExportsWorkspace(_ToolPanel):
    """Artist/Title export and Codec list export (Exports page)."""

    def __init__(
        self, library_path: str = "", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(library_path, parent)
        self._at_worker: ArtistTitleWorker | None = None
        self._codec_worker: CodecExportWorker | None = None
        # Last output files (the Open buttons read these; no re-connecting lambdas)
        self._at_out_path: str | None = None
        self._codec_out_path: str | None = None
        # Tile references (assigned by _build_*_tile helpers)
        self._at_tile: ToolTile | None = None
        self._codec_tile: ToolTile | None = None
        # Codec chip checkboxes for theme-refresh
        self._codec_chips: dict[str, QtWidgets.QCheckBox] = {}
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_section_title("Exports"))
        cl.addWidget(self._make_subtitle(
            "Lists of what is in your library, for use elsewhere. Reserved folders "
            "(Quarantine/, Trash/, Docs/, …) are skipped, and every export file lands "
            "in Docs/ inside your library folder. Nothing in the library is changed."
        ))

        grid = TileGrid()
        grid.add_tile(self._build_at_tile())
        grid.add_tile(self._build_codec_tile())
        cl.addWidget(grid, 1)
        cl.addStretch(1)

    def _build_at_tile(self) -> ToolTile:
        tile = ToolTile(
            "\u2197", "Artist / Title Export",
            "Scan the library and write Docs/artist_title_list.txt "
            "with every Artist \u2013 Title pair.",
        )
        self._at_tile = tile

        self._exclude_flac_cb = _PillToggle("Exclude FLAC files")
        self._dupe_tracks_cb  = _PillToggle("Include per-album duplicate titles")
        tile.add_option(self._exclude_flac_cb)
        tile.add_option(self._dupe_tracks_cb)

        # Wire drawer widgets to the same attribute names used by the slots
        tile.status_label.setVisible(True)
        self._at_status = tile.status_label
        self._at_log    = tile.log_box
        self._at_prog   = tile.progress_bar

        self._at_run_btn = tile.set_run_button("Export", self._on_export_at)
        self._at_cancel_btn = tile.add_secondary_button("Cancel")
        self._at_cancel_btn.setObjectName("atCancelBtn")
        self._at_cancel_btn.setEnabled(False)
        self._at_cancel_btn.clicked.connect(self._on_cancel_at)
        self._at_open = tile.add_secondary_button("Open File")
        self._at_open.setEnabled(False)
        self._at_open.clicked.connect(lambda: self._open_file(self._at_out_path))
        self._at_folder = tile.add_icon_button("\U0001f4c1", "Open folder")
        self._at_folder.setEnabled(False)
        self._at_folder.clicked.connect(lambda: self._open_folder(self._at_out_path))
        tile.finish_footer()
        return tile

    def _build_codec_tile(self) -> ToolTile:
        tile = ToolTile(
            "\u2261", "Codec List Export",
            "Export a grouped list of all tracks by codec format.",
        )
        self._codec_tile = tile

        chips_row = QtWidgets.QHBoxLayout()
        chips_row.setSpacing(6)
        self._codec_ext_cbs: dict[str, QtWidgets.QCheckBox] = {}
        for ext in (".flac", ".mp3", ".m4a", ".aac", ".wav", ".opus", ".ogg"):
            cb = self._make_chip(ext)
            self._codec_ext_cbs[ext] = cb
            chips_row.addWidget(cb)
        chips_row.addStretch(1)
        tile.add_option_layout(chips_row)

        self._omit_paths_cb = _PillToggle("Filenames only (no full paths)")
        tile.add_option(self._omit_paths_cb)

        # Status label always visible in the options area
        self._codec_status = QtWidgets.QLabel("Ready.")
        self._codec_status.setObjectName("statusLabel")
        tile.add_option(self._codec_status)
        self._codec_prog = tile.progress_bar   # available but drawer stays closed

        self._codec_run_btn = tile.set_run_button("Export", self._on_export_codec)
        self._codec_cancel_btn = tile.add_secondary_button("Cancel")
        self._codec_cancel_btn.setObjectName("codecCancelBtn")
        self._codec_cancel_btn.setEnabled(False)
        self._codec_cancel_btn.clicked.connect(self._on_cancel_codec)
        self._codec_open = tile.add_secondary_button("Open File")
        self._codec_open.setEnabled(False)
        self._codec_open.clicked.connect(lambda: self._open_file(self._codec_out_path))
        self._codec_folder = tile.add_icon_button("\U0001f4c1", "Open folder")
        self._codec_folder.setEnabled(False)
        self._codec_folder.clicked.connect(lambda: self._open_folder(self._codec_out_path))
        tile.finish_footer()
        return tile

    # ── Codec chip helpers ─────────────────────────────────────────────────

    def _make_chip(self, label: str) -> QtWidgets.QCheckBox:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(True)
        self._codec_chips[label] = cb
        self._refresh_chip(cb)
        cb.stateChanged.connect(lambda _s, c=cb: self._refresh_chip(c))
        return cb

    def _refresh_chip(self, cb: QtWidgets.QCheckBox) -> None:
        t = get_manager().current
        if cb.isChecked():
            style = (
                f"QCheckBox {{"
                f" background: {_rgba(t.accent, 0.16)};"
                f" border: 1px solid {t.accent};"
                f" border-radius: 11px;"
                f" padding: 4px 12px;"
                f" color: {t.accent};"
                f" font-size: 13px;"
                f"}}"
                f"QCheckBox::indicator {{ width: 0; height: 0; }}"
            )
        else:
            style = (
                f"QCheckBox {{"
                f" background: transparent;"
                f" border: 1px solid {t.card_border};"
                f" border-radius: 11px;"
                f" padding: 4px 12px;"
                f" color: {t.text_secondary};"
                f" font-size: 13px;"
                f"}}"
                f"QCheckBox::indicator {{ width: 0; height: 0; }}"
            )
        cb.setStyleSheet(style)

    def _on_theme_changed_base(self, tokens: object) -> None:
        super()._on_theme_changed_base(tokens)
        for cb in self._codec_chips.values():
            self._refresh_chip(cb)

    # ── Library change ─────────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        """A different library: stop the exports that belong to the old one
        (they are read-only) and drop results that point into it (Open File
        buttons, logs, statuses)."""
        from gui.workers import job_registry
        job_registry().cancel_all(self.running_jobs())
        self._at_out_path = None
        self._codec_out_path = None
        for btn in (self._at_open, self._at_folder, self._codec_open, self._codec_folder):
            btn.setEnabled(False)
        self._at_log.clear()
        for tile in (self._at_tile, self._codec_tile):
            if tile is not None:
                tile.report("idle", "Ready.")
        self._codec_status.setText("Ready.")
        set_tone(self._codec_status, "secondary")

    # ── Slots: Artist / Title ──────────────────────────────────────────────

    def _is_stale_export(self, worker: object) -> bool:
        """True when ``worker``'s result belongs to a library that is no longer
        selected (it finished after a library change)."""
        lib = getattr(worker, "library_path", None)
        if lib is None:
            return False
        return os.path.abspath(lib) != os.path.abspath(self._library_path or "")

    def _export_sender(self, current: object) -> tuple[object, bool]:
        """``(worker, ignore)`` for an export slot: ``ignore`` when the signal
        comes from a job we no longer track."""
        worker = self.sender()
        if worker is None:  # called directly
            return current, False
        return worker, worker is not current

    @Slot()
    def _on_export_at(self) -> None:
        if not self._require_library():
            return
        if self._at_worker is not None:
            return  # already running (the button is disabled too)

        self._at_log.clear()
        self._at_prog.setValue(0)
        self._at_open.setEnabled(False)
        self._at_folder.setEnabled(False)
        self._at_run_btn.setEnabled(False)
        self._at_cancel_btn.setEnabled(True)
        if self._at_tile is not None:
            self._at_tile.report("running", "Scanning…")
            self._at_tile.open_drawer()

        worker = ArtistTitleWorker(
            self._library_path,
            self._exclude_flac_cb.isChecked(),
            self._dupe_tracks_cb.isChecked(),
            parent=self,
        )
        worker.progress.connect(self._on_at_progress)
        worker.log_line.connect(self._at_log.appendPlainText)
        worker.finished.connect(self._on_at_finished)
        worker.error.connect(self._on_at_error)
        worker.cancelled.connect(self._on_at_cancelled)
        self._at_worker = worker
        self._start_worker(worker, "Artist/Title export")
        self._log("Starting artist/title export…", "info")

    @Slot()
    def _on_cancel_at(self) -> None:
        if self._at_worker is not None:
            self._at_worker.cancel()
            self._at_cancel_btn.setEnabled(False)

    def _at_done(self) -> None:
        self._at_worker = None
        self._at_run_btn.setEnabled(True)
        self._at_cancel_btn.setEnabled(False)

    @Slot(int, int)
    def _on_at_progress(self, done: int, total: int) -> None:
        worker, ignore = self._export_sender(self._at_worker)
        if ignore or self._is_stale_export(worker):
            return
        self._at_prog.setMaximum(max(total, 1))
        self._at_prog.setValue(done)
        self._at_status.setText(f"{done} / {total} processed")

    @Slot(str, int, int)
    def _on_at_finished(self, out_path: str, entry_count: int, error_count: int) -> None:
        worker, ignore = self._export_sender(self._at_worker)
        if ignore:
            return
        self._at_done()
        if self._is_stale_export(worker):
            # Finished after a library change: its file is in the old library,
            # so the Open buttons must not point at it.
            self._log(f"Artist/title export for the previous library finished: {out_path}", "info")
            return
        self._at_prog.setValue(self._at_prog.maximum())
        note = f"  ({error_count} files skipped)" if error_count else ""
        self._at_log.appendPlainText(f"Export complete: {out_path}")
        if error_count:
            self._at_log.appendPlainText(f"Skipped {error_count} files due to read errors.")
        if self._at_tile is not None:
            self._at_tile.report("ok", f"Done — {entry_count} entries written{note}", "success")
        self._at_out_path = out_path
        self._at_open.setEnabled(True)
        self._at_folder.setEnabled(True)
        self._log(f"Artist/title export complete: {entry_count} entries → {out_path}", "ok")

    @Slot(str)
    def _on_at_error(self, message: str) -> None:
        worker, ignore = self._export_sender(self._at_worker)
        if ignore:
            return
        self._at_done()
        if self._is_stale_export(worker):
            self._log(f"Artist/title export for the previous library failed: {message}", "warn")
            return
        if self._at_tile is not None:
            self._at_tile.report("error", f"Error: {message}", "danger")
        self._at_log.appendPlainText(f"Export failed: {message}")
        self._log(f"Artist/title export failed: {message}", "error")

    @Slot()
    def _on_at_cancelled(self) -> None:
        worker, ignore = self._export_sender(self._at_worker)
        if ignore:
            return
        self._at_done()
        if self._is_stale_export(worker):
            return
        if self._at_tile is not None:
            self._at_tile.report("cancelled", "Cancelled — nothing written.", "warning")
        self._log("Artist/title export cancelled.", "warn")

    # ── Slots: Codec Export ────────────────────────────────────────────────

    @Slot()
    def _on_export_codec(self) -> None:
        if not self._require_library():
            return
        if self._codec_worker is not None:
            return
        selected_exts = {ext for ext, cb in self._codec_ext_cbs.items() if cb.isChecked()}
        if not selected_exts:
            QtWidgets.QMessageBox.warning(self, "No Codecs", "Select at least one codec.")
            return

        self._codec_prog.setValue(0)
        self._codec_status.setText("Scanning…")
        set_tone(self._codec_status, "secondary")
        self._codec_open.setEnabled(False)
        self._codec_folder.setEnabled(False)
        self._codec_run_btn.setEnabled(False)
        self._codec_cancel_btn.setEnabled(True)
        if self._codec_tile is not None:
            self._codec_tile.status_chip.show_neutral("⧖  Running…")
        self._log("Starting codec list export…", "info")

        worker = CodecExportWorker(
            self._library_path, selected_exts, self._omit_paths_cb.isChecked(), parent=self
        )
        worker.finished.connect(self._on_codec_finished)
        worker.error.connect(self._on_codec_error)
        worker.cancelled.connect(self._on_codec_cancelled)
        self._codec_worker = worker
        self._start_worker(worker, "Codec list export")

    @Slot()
    def _on_cancel_codec(self) -> None:
        if self._codec_worker is not None:
            self._codec_worker.cancel()
            self._codec_cancel_btn.setEnabled(False)

    def _codec_done(self) -> None:
        self._codec_worker = None
        self._codec_run_btn.setEnabled(True)
        self._codec_cancel_btn.setEnabled(False)

    @Slot(str, int)
    def _on_codec_finished(self, out_path: str, total: int) -> None:
        worker, ignore = self._export_sender(self._codec_worker)
        if ignore:
            return
        self._codec_done()
        if self._is_stale_export(worker):
            self._log(f"Codec list export for the previous library finished: {out_path}", "info")
            return
        self._codec_prog.setValue(100)
        self._codec_status.setText(f"Written: {out_path}  ({total} files)")
        self._flash_label(self._codec_status, "success")
        if self._codec_tile is not None:
            self._codec_tile.status_chip.show_success("✓  Done")
        self._codec_out_path = out_path
        self._codec_open.setEnabled(True)
        self._codec_folder.setEnabled(True)
        self._log(f"Codec list exported: {total} files → {out_path}", "ok")

    @Slot(str)
    def _on_codec_error(self, message: str) -> None:
        worker, ignore = self._export_sender(self._codec_worker)
        if ignore:
            return
        self._codec_done()
        if self._is_stale_export(worker):
            self._log(f"Codec list export for the previous library failed: {message}", "warn")
            return
        self._codec_status.setText(message)
        self._flash_label(self._codec_status, "danger")
        if self._codec_tile is not None:
            self._codec_tile.status_chip.show_error("✗  Error")
        self._log(message, "error")

    @Slot()
    def _on_codec_cancelled(self) -> None:
        worker, ignore = self._export_sender(self._codec_worker)
        if ignore:
            return
        self._codec_done()
        if self._is_stale_export(worker):
            return
        self._codec_status.setText("Cancelled — nothing written.")
        set_tone(self._codec_status, "secondary")
        if self._codec_tile is not None:
            self._codec_tile.status_chip.show_neutral("Cancelled")
        self._log("Codec list export cancelled.", "warn")


# ── File Cleanup ───────────────────────────────────────────────────────────────

class FileCleanupWorkspace(_ToolPanel):
    """File Cleanup page: remove copy suffixes, previewed, logged, undoable."""

    def __init__(
        self, library_path: str = "", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(library_path, parent)
        self._cleanup_worker: FileCleanupWorker | None = None
        self._cleanup_plan_worker: FileCleanupPlanWorker | None = None
        self._cleanup_tile: ToolTile | None = None
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_section_title("File Cleanup"))
        cl.addWidget(self._make_subtitle(
            "Tidy file names that browsers, Windows Explorer and macOS Finder left "
            "behind. Nothing is renamed until you have reviewed the plan, every rename "
            "is logged to Docs/, and the last cleanup can be undone."
        ))

        grid = TileGrid()
        grid.add_tile(self._build_cleanup_tile(), full_width=True)
        cl.addWidget(grid, 1)
        cl.addStretch(1)

    def _build_cleanup_tile(self) -> ToolTile:
        tile = ToolTile(
            "\u2736", "File Cleanup",
            "Remove trailing \u2018 (1)\u2019, \u2018 - Copy\u2019, \u2018 copy\u2019 suffixes "
            "left by browsers, Windows Explorer and macOS Finder. You review every "
            "rename before anything changes; reserved folders are skipped, a log "
            "is written to Docs/, and playlists are updated. \u2018Undo Last "
            "Cleanup\u2019 previews, then restores, the original names (and "
            "playlist entries).",
        )
        self._cleanup_tile = tile

        tile.status_label.setVisible(True)
        self._cleanup_status = tile.status_label
        self._cleanup_log    = tile.log_box

        self._cleanup_run_btn = tile.set_run_button("Preview Cleanup\u2026", self._on_file_cleanup)
        self._cleanup_cancel_btn = tile.add_secondary_button("Cancel")
        self._cleanup_cancel_btn.setObjectName("cleanupCancelBtn")
        self._cleanup_cancel_btn.setEnabled(False)
        self._cleanup_cancel_btn.clicked.connect(self._on_cancel_cleanup)
        self._cleanup_undo_btn = tile.add_secondary_button("Undo Last Cleanup\u2026")
        self._cleanup_undo_btn.setObjectName("cleanupUndoBtn")
        self._cleanup_undo_btn.setToolTip(
            "Preview and restore the original names from the newest cleanup log in Docs/"
        )
        self._cleanup_undo_btn.clicked.connect(self._on_undo_cleanup)
        tile.finish_footer()
        return tile

    # ── Library change ─────────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        """A different library: stop a preview that belongs to the old one.

        A confirmed library write (File Cleanup renames, or its undo) is left
        to finish, as the Indexer and Library Sync do: stopping it part-way
        would leave the old library half renamed with its playlists not yet
        updated. The tile keeps reporting on it until it ends.
        """
        from gui.workers import LIBRARY_WRITES, job_registry
        job_registry().cancel_all(
            [j for j in self.running_jobs() if j.group != LIBRARY_WRITES]
        )
        if self._cleanup_worker is None:
            self._cleanup_log.clear()
            if self._cleanup_tile is not None:
                self._cleanup_tile.report("idle", "Ready.")
        else:
            self._cleanup_log.appendPlainText(
                "Library changed \u2014 the confirmed renames in the previous library "
                "keep running until they finish."
            )

    # ── Slots: File Cleanup (plan → preview → confirm → execute → log) ────

    def _cleanup_set_running(self, running: bool) -> None:
        self._cleanup_run_btn.setEnabled(not running)
        self._cleanup_undo_btn.setEnabled(not running)
        self._cleanup_cancel_btn.setEnabled(running)

    @Slot()
    def _on_file_cleanup(self) -> None:
        if not self._require_library():
            return
        if self._cleanup_plan_worker is not None or self._cleanup_worker is not None:
            return

        self._cleanup_log.clear()
        self._cleanup_set_running(True)
        if self._cleanup_tile is not None:
            self._cleanup_tile.report("running", "Looking for copy suffixes… (nothing is renamed yet)")
            self._cleanup_tile.open_drawer()
        self._log("File cleanup: building a rename preview…", "info")

        worker = FileCleanupPlanWorker(self._library_path, parent=self)
        worker.plan_ready.connect(self._on_cleanup_plan_ready)
        worker.error.connect(self._on_cleanup_plan_error)
        self._cleanup_plan_worker = worker
        self._start_worker(worker, "File cleanup preview", ask_before_quit=False)

    @Slot()
    def _on_cancel_cleanup(self) -> None:
        for worker in (self._cleanup_plan_worker, self._cleanup_worker):
            if worker is not None:
                worker.cancel()
        self._cleanup_cancel_btn.setEnabled(False)

    @Slot(str)
    def _on_cleanup_plan_error(self, message: str) -> None:
        self._cleanup_plan_worker = None
        self._cleanup_set_running(False)
        if self._cleanup_tile is not None:
            self._cleanup_tile.report("error", f"Preview failed: {message}", "danger")
        self._log(f"File cleanup preview failed: {message}", "error")

    @Slot(object)
    def _on_cleanup_plan_ready(self, plan) -> None:
        self._cleanup_plan_worker = None
        tile = self._cleanup_tile
        if plan.cancelled:
            self._cleanup_set_running(False)
            if tile is not None:
                tile.report("cancelled", "Cancelled — nothing was renamed.", "warning")
            self._log("File cleanup cancelled before preview.", "warn")
            return
        if plan.library_root != os.path.abspath(self._library_path or ""):
            # The library changed while the preview was being built.
            self._cleanup_set_running(False)
            if tile is not None:
                tile.report("idle", "Ready.")
            return
        if not plan.items:
            self._cleanup_set_running(False)
            msg = f"Nothing to clean up — {plan.scanned} audio files checked."
            if tile is not None:
                tile.report("ok", msg, "success")
            self._log(msg, "ok")
            QtWidgets.QMessageBox.information(self, "File Cleanup", msg)
            return

        if self._library_writes_blocked("File cleanup"):
            self._cleanup_set_running(False)
            if tile is not None:
                tile.report("idle", "Library busy — nothing was renamed.")
            return
        from gui.dialogs.file_cleanup_preview_dialog import FileCleanupPreviewDialog
        dlg = FileCleanupPreviewDialog(plan, self)
        try:
            accepted = dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted.value
        finally:
            dlg.deleteLater()
        if not accepted or not plan.renames:
            self._cleanup_set_running(False)
            if tile is not None:
                tile.report("idle", "Preview closed — nothing was renamed.")
            self._log("File cleanup: preview closed, nothing renamed.", "info")
            return

        if tile is not None:
            tile.report("running", f"Renaming {len(plan.renames)} file(s)…")
        self._log(f"File cleanup: renaming {len(plan.renames)} file(s)…", "info")
        worker = FileCleanupWorker(plan, parent=self)
        worker.log_line.connect(self._cleanup_log.appendPlainText)
        worker.finished.connect(self._on_cleanup_finished)
        self._cleanup_worker = worker
        self._start_worker(worker, "File cleanup", library_write=True)

    @Slot()
    def _on_undo_cleanup(self) -> None:
        """Preview, confirm, then restore the names from the newest cleanup log
        that still has renames to undo (and point playlists back)."""
        if not self._require_library():
            return
        if self._cleanup_plan_worker is not None or self._cleanup_worker is not None:
            return
        from file_cleanup import latest_undoable_log, plan_undo

        tile = self._cleanup_tile
        log_path = latest_undoable_log(self._library_path)
        if not log_path:
            QtWidgets.QMessageBox.information(
                self, "Undo File Cleanup",
                "There is no file cleanup to undo in this library "
                "(no Docs/file_cleanup_*.jsonl log with renames left to restore).",
            )
            return
        try:
            plan = plan_undo(log_path)
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Undo File Cleanup", f"The cleanup log could not be read:\n{exc}"
            )
            return
        if self._library_writes_blocked("Undo file cleanup"):
            return
        from gui.dialogs.file_cleanup_preview_dialog import FileCleanupPreviewDialog
        dlg = FileCleanupPreviewDialog(plan, self, undo=True)
        try:
            accepted = dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted.value
        finally:
            dlg.deleteLater()
        if not accepted or not plan.renames:
            if tile is not None:
                tile.report("idle", "Undo preview closed \u2014 nothing was renamed.")
            self._log("File cleanup undo: preview closed, nothing renamed.", "info")
            return

        self._cleanup_log.clear()
        self._cleanup_set_running(True)
        if tile is not None:
            tile.report("running", f"Restoring {len(plan.renames)} original name(s)\u2026")
            tile.open_drawer()
        self._log(f"File cleanup undo: restoring {len(plan.renames)} name(s)\u2026", "info")
        worker = FileCleanupWorker(plan, parent=self, undo=True)
        worker.log_line.connect(self._cleanup_log.appendPlainText)
        worker.finished.connect(self._on_cleanup_finished)
        self._cleanup_worker = worker
        self._start_worker(worker, "Undo file cleanup", library_write=True)

    @Slot(bool, str)
    def _on_cleanup_finished(self, success: bool, message: str) -> None:
        self._cleanup_worker = None
        self._cleanup_set_running(False)
        if self._cleanup_tile is not None:
            self._cleanup_tile.report(
                "ok" if success else "error", message, "success" if success else "danger"
            )
        self._log(message, "ok" if success else "error")
