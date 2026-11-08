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
from gui.themes import motion
from gui.themes.glass import apply_font as glass_apply_font
from gui.themes.glass import current_material, qcolor
from gui.themes.glass import font as glass_font
from gui.themes.manager import get_manager
from gui.widgets.glass import ElidedLabel, GlassCard
from gui.widgets.glass_parts import EmptyState
from gui.widgets.glyphs import paint_glyph, set_glyph
from gui.workspaces.base import WorkspaceBase
from gui.workspaces.label_tone import flash_tone, refresh_tones, set_tone
from gui.workspaces.organize_parts import FlowLayout, hero_title, path_well, thin_bar
from utils.reserved_folders import NON_LIBRARY_FOLDERS, walk_library

_SUPPORTED_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"}


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


class _GlassResultChip(QtWidgets.QWidget):
    """The tile's status pill: "Ready", "Running…", "Done", "Error",
    "Cancelled". Neutral states sit on a well; done and error are tinted with
    their theme tone and carry a glyph. Text changes cross-fade (painter
    opacity only, two parented animations reused for every change)."""

    NEUTRAL = 0
    SUCCESS = 1
    ERROR   = 2

    # Theme token per state (read at paint time, so theme switches apply).
    _TOKEN = {
        NEUTRAL: "text_secondary",
        SUCCESS: "success",
        ERROR:   "danger",
    }
    _GLYPH = {SUCCESS: "check", ERROR: "warning"}
    _H = 26
    _PAD = 11
    _GLYPH_W = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolStatusPill")
        self._text  = "Ready"
        self._state = self.NEUTRAL
        self._opacity = 1.0
        self._pending: tuple[str, int] | None = None
        # Two long-lived, parented animations reused for every transition.
        # (A stored reference to a DeleteWhenStopped animation dangled after
        # the first transition, so the second one raised RuntimeError — DS-06.)
        self._fade_out = QtCore.QVariantAnimation(self)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setDuration(motion.duration("micro"))
        self._fade_out.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._fade_out.valueChanged.connect(self._on_opacity)
        self._fade_out.finished.connect(self._swap)
        self._fade_in = QtCore.QVariantAnimation(self)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setDuration(motion.duration("base"))
        self._fade_in.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._fade_in.valueChanged.connect(self._on_opacity)
        self.setFixedHeight(self._H)
        self._resize()
        # Bound method (not a lambda) so the singleton drops the connection
        # when this widget is destroyed.
        get_manager().theme_changed.connect(self._repaint_on_theme)

    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()

    @staticmethod
    def _font() -> QtGui.QFont:
        return glass_font("caption", weight=600)

    def _resize(self):
        fm = QtGui.QFontMetrics(self._font())
        glyph = self._GLYPH_W + 5 if self._state in self._GLYPH else 0
        self.setFixedWidth(fm.horizontalAdvance(self._text) + glyph + 2 * self._PAD)
        self.setAccessibleName(f"Status: {self._text}")

    def set_text_fast(self, text: str, state: int | None = None):
        self._text = text
        if state is not None:
            self._state = state
        self._resize()
        self.update()

    def show_neutral(self, text: str = "Ready"):
        self._transition(text, self.NEUTRAL)

    def show_success(self, text: str = "Done"):
        self._transition(text, self.SUCCESS)

    def show_error(self, text: str = "Error"):
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
        if not self.isVisible() or motion.reduced_motion():
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
        if self.isVisible() and not motion.reduced_motion():
            self._fade_in.start()
        else:
            self._set_op(1.0)

    def _on_opacity(self, v) -> None:
        self._set_op(float(v))

    def _set_op(self, v: float):
        self._opacity = v
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = get_manager().current
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setOpacity(self._opacity)
            rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            rad = rect.height() / 2.0
            tone = getattr(t, self._TOKEN[self._state])
            if self._state == self.NEUTRAL:
                fill, border, ink = qcolor(m.well.fill), qcolor(m.well.border), qcolor(tone)
            else:
                fill, border, ink = qcolor(tone, 0.16), qcolor(tone, 0.45), qcolor(tone)
            p.setPen(QtGui.QPen(border, 1.0))
            p.setBrush(fill)
            p.drawRoundedRect(rect, rad, rad)
            x = float(self._PAD)
            glyph = self._GLYPH.get(self._state)
            if glyph:
                g = QtCore.QRectF(x, (self.height() - self._GLYPH_W) / 2, self._GLYPH_W, self._GLYPH_W)
                paint_glyph(p, g, glyph, ink, weight=2.2)
                x += self._GLYPH_W + 5
            p.setPen(ink)
            p.setFont(self._font())
            p.drawText(QtCore.QRectF(x, 0, self.width() - x, self.height()),
                       int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                       self._text)
        finally:
            p.end()


# ── ToolTile ───────────────────────────────────────────────────────────────────

class ToolTile(GlassCard):
    """A tool's glass card: header ▸ options ▸ empty state ▸ footer ▸ result.

    Usage
    -----
    tile = ToolTile("My tool", "One line on what it does.",
                    empty_title="Nothing yet", empty_message="What appears here.")
    tile.add_option(some_toggle)
    tile.set_run_button("Run", self._my_slot, glyph="play")
    secondary = tile.add_secondary_button("Open file", glyph="external")
    tile.finish_footer()

    When an operation starts call ``tile.open_drawer()`` to swap the empty
    state for the result area (progress pill, status line, output path, log)
    and ``tile.set_running(True/False)`` to lock/unlock the footer buttons.
    ``hero=True`` makes it the page's hero card (radius 30, title role).
    """

    def __init__(
        self,
        title: str,
        description: str,
        *,
        hero: bool = False,
        empty_title: str = "",
        empty_message: str = "",
        log_height: int = 110,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if hero:
            self.setObjectName("headerCard")
        self._footer_buttons: list[QtWidgets.QPushButton] = []
        self._drawer_open = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(12)
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(4)
        if hero:
            self._title_lbl = hero_title(title)
        else:
            self._title_lbl = QtWidgets.QLabel(title)
            self._title_lbl.setObjectName("cardTitle")
        info.addWidget(self._title_lbl)
        self._desc_lbl: QtWidgets.QLabel | None = None
        if description:
            self._desc_lbl = QtWidgets.QLabel(description)
            self._desc_lbl.setObjectName("sectionSubtitle")
            self._desc_lbl.setWordWrap(True)
            info.addWidget(self._desc_lbl)
        head.addLayout(info, 1)
        self.status_chip = _GlassResultChip()
        self.status_chip.show_neutral("Ready")
        head.addWidget(self.status_chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        outer.addLayout(head)

        # ── Options (shown once something is added) ───────────────────────
        self._options = QtWidgets.QWidget()
        self._opts_l = QtWidgets.QVBoxLayout(self._options)
        self._opts_l.setContentsMargins(0, 0, 0, 0)
        self._opts_l.setSpacing(10)
        self._options.setVisible(False)
        outer.addWidget(self._options)

        # ── Before the first run ──────────────────────────────────────────
        self.empty_state: EmptyState | None = None
        if empty_title:
            self.empty_state = EmptyState(empty_title, empty_message)
            outer.addWidget(self.empty_state)

        # ── Footer ────────────────────────────────────────────────────────
        self._footer = QtWidgets.QWidget()
        self._footer_l = QtWidgets.QHBoxLayout(self._footer)
        self._footer_l.setContentsMargins(0, 0, 0, 0)
        self._footer_l.setSpacing(10)
        outer.addWidget(self._footer)

        # ── Result area ("drawer"; hidden until a run starts) ─────────────
        self._drawer = QtWidgets.QWidget()
        drawer_l = QtWidgets.QVBoxLayout(self._drawer)
        drawer_l.setContentsMargins(0, 4, 0, 0)
        drawer_l.setSpacing(8)

        self.progress_bar = thin_bar()

        # Prose about the run (tests and slots read it). No role name: its
        # colour is a palette tone (label_tone), which a QSS role would mask.
        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setWordWrap(True)
        glass_apply_font(self.status_label, "label")

        self._output_lbl = ElidedLabel("", QtCore.Qt.TextElideMode.ElideMiddle)
        self._output_lbl.setObjectName("toolOutputPath")
        self._output_row = path_well(self._output_lbl, "file")
        self._output_row.setVisible(False)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(log_height)
        self.log_box.setObjectName("organizeLog")

        drawer_l.addWidget(self.progress_bar)
        drawer_l.addWidget(self.status_label)
        drawer_l.addWidget(self._output_row)
        drawer_l.addWidget(self.log_box)
        self._drawer.setVisible(False)
        outer.addWidget(self._drawer)

    # ── Public API ────────────────────────────────────────────────────────

    def add_option(self, widget: QtWidgets.QWidget) -> None:
        """Add a widget to the options section."""
        self._opts_l.addWidget(widget)
        self._options.setVisible(True)

    def add_option_layout(self, layout: QtWidgets.QLayout) -> None:
        """Add a sub-layout to the options section."""
        self._opts_l.addLayout(layout)
        self._options.setVisible(True)

    def _add_footer_button(self, btn: QtWidgets.QPushButton) -> QtWidgets.QPushButton:
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def set_run_button(self, label: str, slot, glyph: str = "") -> QtWidgets.QPushButton:
        """Add the primary (accent-coloured) action button and return it."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(label)
        btn.setObjectName("primaryBtn")
        if glyph:
            set_glyph(btn, glyph, role="inverse")
        btn.clicked.connect(slot)
        return self._add_footer_button(btn)

    def add_secondary_button(self, label: str, glyph: str = "") -> QtWidgets.QPushButton:
        """Add a secondary (glass pill) button and return it."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(label)
        if glyph:
            set_glyph(btn, glyph)
        btn.setAccessibleName(label)
        return self._add_footer_button(btn)

    def add_icon_button(self, glyph: str, tooltip: str) -> QtWidgets.QPushButton:
        """Add a round, icon-only button (a painted glyph) and return it; the
        tooltip is also its accessible name."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton("")
        set_glyph(btn, glyph)
        btn.setFixedSize(36, 36)
        btn.setToolTip(tooltip)
        btn.setAccessibleName(tooltip)
        return self._add_footer_button(btn)

    def finish_footer(self) -> None:
        """Add trailing stretch after all buttons. Call last."""
        self._footer_l.addStretch(1)

    def hide_footer(self) -> None:
        """Hide the footer row entirely (e.g. for tiles with no run action)."""
        self._footer.setVisible(False)

    def drawer_open(self) -> bool:
        return self._drawer_open

    def open_drawer(self) -> None:
        """Swap the empty state for the result area, with a short fade
        (opacity only: nothing animates layout-owned geometry)."""
        if self._drawer_open:
            return
        self._drawer_open = True
        if self.empty_state is not None:
            self.empty_state.setVisible(False)
        self._drawer.setVisible(True)
        if self._drawer.isVisible():
            motion.reveal(self._drawer)

    def reset_view(self) -> None:
        """Back to the state before the first run (a library change)."""
        self._drawer_open = False
        self._drawer.setVisible(False)
        self.set_output(None)
        self.log_box.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        if self.empty_state is not None:
            self.empty_state.setVisible(True)

    def set_output(self, path: str | None) -> None:
        """Show the file a run wrote (middle-elided; full path in the tooltip)."""
        self._output_lbl.setText(path or "")
        self._output_lbl.setToolTip(path or "")
        self._output_row.setVisible(bool(path))

    def output_path(self) -> str:
        return self._output_lbl.text() if not self._output_row.isHidden() else ""

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
            chip.show_success("Done")
        elif state == "error":
            chip.show_error("Error")
        elif state == "running":
            chip.show_neutral("Running…")
        elif state == "cancelled":
            chip.show_neutral("Cancelled")
        else:
            chip.show_neutral("Ready")
        self.status_label.setText(text)
        if tone:
            self.flash_status(tone)
        else:
            set_tone(self.status_label, "secondary")


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
        self._grid.setSpacing(18)
        self._grid.setContentsMargins(0, 0, 0, 0)

    def add_tile(self, tile: ToolTile, full_width: bool = False) -> None:
        """Register a tile and add it to the current grid layout."""
        self._tiles.append((tile, full_width))
        self._reflow()

    def tiles(self) -> list[ToolTile]:
        return [tile for tile, _ in self._tiles]

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
                self._grid.addWidget(tile, row, col, QtCore.Qt.AlignmentFlag.AlignTop)
                col += 1
                if col >= cols:
                    col = 0
                    row += 1

        # Columns beyond the current count keep no stretch: going from two
        # columns to one used to leave an empty, stretching second column,
        # so every tile (full-width ones too) took half the page.
        for c in range(max(cols, self._grid.columnCount())):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)


# ── Toggles ────────────────────────────────────────────────────────────────────

class _PillToggle(QtWidgets.QAbstractButton):
    """Animated pill switch with a label.

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
        self.setAccessibleName(label)
        self._knob_x: float = 0.0          # 0.0 = off, 1.0 = on
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(motion.duration("fast"))
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_knob)
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
        end = 1.0 if checked else 0.0
        if motion.reduced_motion() or not self.isVisible():
            self._set_knob(end)
            return
        self._anim.setStartValue(self._knob_x)
        self._anim.setEndValue(end)
        self._anim.start()

    def _on_knob(self, v) -> None:
        self._set_knob(float(v))

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
            w += self._GAP + fm.horizontalAdvance(label) + 2
            h = max(h, fm.height() + 4)
        return QtCore.QSize(w, h)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, e) -> None:  # noqa: N802
        t = get_manager().current
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            tw, th, kd = float(self._TRACK_W), float(self._TRACK_H), float(self._KNOB_D)
            cy = self.height() / 2.0
            tr = QtCore.QRectF(0.5, cy - th / 2, tw - 1, th)
            on = qcolor(t.accent)
            off = qcolor(m.well.fill)
            fill = QtGui.QColor(
                round(off.red() + (on.red() - off.red()) * self._knob_x),
                round(off.green() + (on.green() - off.green()) * self._knob_x),
                round(off.blue() + (on.blue() - off.blue()) * self._knob_x),
                round(off.alpha() + (255 - off.alpha()) * self._knob_x),
            )
            p.setPen(QtGui.QPen(qcolor(m.well.border), 1.0) if self._knob_x < 0.5
                     else QtCore.Qt.PenStyle.NoPen)
            p.setBrush(fill)
            p.drawRoundedRect(tr, th / 2, th / 2)

            margin = (th - kd) / 2.0
            travel = tw - kd - 2 * margin
            kx = margin + self._knob_x * travel
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(255, 255, 255))
            p.drawEllipse(QtCore.QRectF(kx, cy - kd / 2.0, kd, kd))
            if self.hasFocus():
                p.setPen(QtGui.QPen(qcolor(t.accent, 0.8), 1.5))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(tr.adjusted(-2, -2, 2, 2), th / 2 + 2, th / 2 + 2)

            label = self.text()
            if label:
                p.setPen(qcolor(t.text_primary if self.isEnabled() else t.text_muted))
                p.setFont(self.font())
                lx = int(tw) + self._GAP
                p.drawText(
                    lx, 0, self.width() - lx, self.height(),
                    int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft),
                    label,
                )
        finally:
            p.end()


class _ChipToggle(QtWidgets.QAbstractButton):
    """A checkable pill for one choice in a set (the codec chips): tinted
    with the accent and a check when on, a hairline outline when off.
    Painted from the theme (no per-chip stylesheet)."""

    _H = 30
    _PAD = 12
    _GLYPH_W = 12

    def __init__(self, label: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(label)
        self.setAccessibleName(label)
        self.setFont(glass_font("label", weight=600))
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        get_manager().theme_changed.connect(self._repaint_on_theme)

    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        fm = QtGui.QFontMetrics(self.font())
        return QtCore.QSize(2 * self._PAD + self._GLYPH_W + 6 + fm.horizontalAdvance(self.text()),
                            self._H)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.sizeHint()

    def event(self, e: QtCore.QEvent) -> bool:
        if e.type() in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()
        return super().event(e)

    def paintEvent(self, e) -> None:  # noqa: N802
        t = get_manager().current
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            rad = rect.height() / 2
            if self.isChecked():
                fill, border, ink = qcolor(t.accent, 0.18), qcolor(t.accent, 0.60), qcolor(t.text_primary)
            else:
                fill, border, ink = QtGui.QColor(0, 0, 0, 0), qcolor(m.well.border), qcolor(t.text_secondary)
            if self.underMouse():
                fill = qcolor(t.accent, 0.26) if self.isChecked() else qcolor(m.hover)
            if self.hasFocus():
                border = qcolor(t.accent)
            p.setPen(QtGui.QPen(border, 1.0))
            p.setBrush(fill)
            p.drawRoundedRect(rect, rad, rad)
            g = QtCore.QRectF(self._PAD, (self.height() - self._GLYPH_W) / 2, self._GLYPH_W, self._GLYPH_W)
            paint_glyph(p, g, "check" if self.isChecked() else "plus",
                        qcolor(t.accent) if self.isChecked() else qcolor(t.text_muted), weight=2.2)
            p.setPen(ink)
            p.setFont(self.font())
            x = self._PAD + self._GLYPH_W + 6
            p.drawText(QtCore.QRectF(x, 0, self.width() - x, self.height()),
                       int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                       self.text())
        finally:
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

    def _adopt_tile(self, tile: "ToolTile") -> "ToolTile":
        """Register a tile like a ``_make_card`` card: the scroll surface
        paints its shadow (it is a GlassCard, so it paints its own glass)."""
        self._cards.append(tile)
        surface = getattr(self, "_inner", None)
        if hasattr(surface, "add_elevated"):
            surface.add_elevated(tile)
        return tile

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
        self._codec_chips: dict[str, _ChipToggle] = {}
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_subtitle(
            "Lists of what is in your library, written to Docs/ inside it. Reserved "
            "folders are skipped and nothing in the library changes."
        ))

        grid = TileGrid()
        grid.add_tile(self._adopt_tile(self._build_at_tile()))
        grid.add_tile(self._adopt_tile(self._build_codec_tile()))
        cl.addWidget(grid, 1)
        cl.addStretch(1)

    def _export_footer(self, tile: ToolTile, run_slot, cancel_slot, cancel_name: str,
                       out_attr: str) -> tuple:
        """Export · Cancel · Open file · show-in-folder, the same on both tiles."""
        run = tile.set_run_button("Export", run_slot, glyph="download")
        cancel = tile.add_secondary_button("Cancel", glyph="close")
        cancel.setObjectName(cancel_name)
        cancel.setEnabled(False)
        cancel.clicked.connect(cancel_slot)
        open_btn = tile.add_secondary_button("Open file", glyph="external")
        open_btn.setToolTip("Open the exported list in your text editor.")
        open_btn.setEnabled(False)
        open_btn.clicked.connect(lambda: self._open_file(getattr(self, out_attr)))
        folder = tile.add_icon_button("folder", "Show the exported file in its folder")
        folder.setEnabled(False)
        folder.clicked.connect(lambda: self._open_folder(getattr(self, out_attr)))
        tile.finish_footer()
        return run, cancel, open_btn, folder

    def _build_at_tile(self) -> ToolTile:
        tile = ToolTile(
            "Artist / Title list",
            "Every Artist – Title pair, written to Docs/artist_title_list.txt.",
            empty_title="No list yet",
            empty_message="Export reads every track's tags. Files that cannot be read "
                          "are skipped and counted.",
        )
        self._at_tile = tile

        self._exclude_flac_cb = _PillToggle("Exclude FLAC files")
        self._dupe_tracks_cb  = _PillToggle("Include per-album duplicate titles")
        toggles = FlowLayout(hspacing=24, vspacing=10)
        toggles.addWidget(self._exclude_flac_cb)
        toggles.addWidget(self._dupe_tracks_cb)
        tile.add_option_layout(toggles)

        # Wire the result area to the attribute names the slots use
        self._at_status = tile.status_label
        self._at_log    = tile.log_box
        self._at_log.setPlaceholderText("The export's progress and any unreadable files appear here.")
        self._at_prog   = tile.progress_bar

        (self._at_run_btn, self._at_cancel_btn, self._at_open,
         self._at_folder) = self._export_footer(tile, self._on_export_at, self._on_cancel_at,
                                                "atCancelBtn", "_at_out_path")
        return tile

    def _build_codec_tile(self) -> ToolTile:
        tile = ToolTile(
            "Codec list",
            "Every track grouped by format, written to Docs/codec_file_list.txt.",
            empty_title="No list yet",
            empty_message="Pick the formats to include, then export.",
            log_height=0,
        )
        self._codec_tile = tile
        tile.log_box.setVisible(False)      # one file, one line: no log to show

        chips = FlowLayout(hspacing=8, vspacing=8)
        self._codec_ext_cbs: dict[str, _ChipToggle] = {}
        for ext in (".flac", ".mp3", ".m4a", ".aac", ".wav", ".opus", ".ogg"):
            cb = self._make_chip(ext)
            self._codec_ext_cbs[ext] = cb
            chips.addWidget(cb)
        tile.add_option_layout(chips)

        self._omit_paths_cb = _PillToggle("Filenames only (no full paths)")
        tile.add_option(self._omit_paths_cb)

        self._codec_status = tile.status_label
        self._codec_prog = tile.progress_bar

        (self._codec_run_btn, self._codec_cancel_btn, self._codec_open,
         self._codec_folder) = self._export_footer(tile, self._on_export_codec,
                                                   self._on_cancel_codec, "codecCancelBtn",
                                                   "_codec_out_path")
        return tile

    # ── Codec chips ────────────────────────────────────────────────────────

    def _make_chip(self, label: str) -> "_ChipToggle":
        chip = _ChipToggle(label)
        chip.setChecked(True)
        chip.setToolTip(f"Include {label} files in the list")
        self._codec_chips[label] = chip
        return chip
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
                tile.reset_view()
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
            self._at_tile.set_output(None)
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
            self._at_tile.set_output(out_path)
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

        self._codec_prog.setRange(0, 0)      # busy: the walk reports no progress
        self._codec_status.setText("Scanning…")
        set_tone(self._codec_status, "secondary")
        self._codec_open.setEnabled(False)
        self._codec_folder.setEnabled(False)
        self._codec_run_btn.setEnabled(False)
        self._codec_cancel_btn.setEnabled(True)
        if self._codec_tile is not None:
            self._codec_tile.set_output(None)
            self._codec_tile.status_chip.show_neutral("Running…")
            self._codec_tile.open_drawer()
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
        self._codec_prog.setRange(0, 100)
        self._codec_prog.setValue(0)

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
        # The path is shown (elided) in the output row, not in this line.
        self._codec_status.setText(f"Done — {total} files listed.")
        self._flash_label(self._codec_status, "success")
        if self._codec_tile is not None:
            self._codec_tile.status_chip.show_success("Done")
            self._codec_tile.set_output(out_path)
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
            self._codec_tile.status_chip.show_error("Error")
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
        grid = TileGrid()
        grid.add_tile(self._adopt_tile(self._build_cleanup_tile()), full_width=True)
        cl.addWidget(grid, 1)
        cl.addStretch(1)

    def _build_cleanup_tile(self) -> ToolTile:
        tile = ToolTile(
            "Clean up copy suffixes",
            "Remove the ‘ (1)’, ‘ - Copy’ and ‘ copy’ endings that "
            "browsers, Windows Explorer and macOS Finder add. You review every rename first, "
            "each one is logged to Docs/, and the last cleanup can be undone.",
            hero=True,
            empty_title="Nothing previewed yet",
            empty_message="Preview cleanup checks every audio file outside the reserved "
                          "folders and lists the renames. Nothing is renamed until you confirm; "
                          "playlists that point at a renamed file are updated.",
        )
        self._cleanup_tile = tile

        self._cleanup_status = tile.status_label
        self._cleanup_log    = tile.log_box
        self._cleanup_log.setPlaceholderText("Each rename is listed here as it happens.")
        # Planning and renaming report no progress: no bar that never moves.
        tile.progress_bar.setVisible(False)

        self._cleanup_run_btn = tile.set_run_button(
            "Preview cleanup…", self._on_file_cleanup, glyph="search")
        self._cleanup_run_btn.setToolTip(
            "Find the copy suffixes and show every planned rename. Nothing is renamed yet."
        )
        self._cleanup_cancel_btn = tile.add_secondary_button("Cancel", glyph="close")
        self._cleanup_cancel_btn.setObjectName("cleanupCancelBtn")
        self._cleanup_cancel_btn.setEnabled(False)
        self._cleanup_cancel_btn.clicked.connect(self._on_cancel_cleanup)
        self._cleanup_undo_btn = tile.add_secondary_button("Undo last cleanup…", glyph="undo")
        self._cleanup_undo_btn.setObjectName("cleanupUndoBtn")
        self._cleanup_undo_btn.setToolTip(
            "Preview and restore the original names from the newest cleanup log in Docs/ "
            "(playlist entries are pointed back too)."
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
                self._cleanup_tile.reset_view()
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
