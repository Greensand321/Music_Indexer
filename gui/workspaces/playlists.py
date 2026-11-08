"""Playlists page — folder, tempo/energy and Auto-DJ playlists, and repair.

Every mode is preview-first: a background job works out exactly which
``.m3u`` files would be written (and which existing ones overwritten), the
user confirms that list, and only then is exactly that plan written, with a
timestamped ``.bak`` of every file it overwrites (see
``playlist_generator.write_playlist_plan``).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

_AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"}

#: How many planned files the confirmation lists before summarising the rest.
_CONFIRM_LIST_LIMIT = 12


def _find_audio_files(root: str) -> list[str]:
    """Library tracks, skipping the reserved folders (Docs, Quarantine, Trash…)."""
    from utils.reserved_folders import iter_library_files

    return sorted(str(p) for p in iter_library_files(root, _AUDIO_EXTS))


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class AudioFeatureCache(dict):
    """Auto-DJ's per-track feature vectors, kept between runs for one library.

    ``playlist_engine`` caches by path; this remembers each file's size and
    mtime too, so a file replaced or re-encoded since it was analysed is
    analysed again instead of reusing the old vector.
    """

    @staticmethod
    def _stamp(path):
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def __contains__(self, path) -> bool:
        entry = dict.get(self, path)
        return entry is not None and entry[0] is not None and entry[0] == self._stamp(path)

    def __getitem__(self, path):
        return dict.__getitem__(self, path)[1]

    def __setitem__(self, path, value) -> None:
        dict.__setitem__(self, path, (self._stamp(path), value))


class PlaylistWorker(QtCore.QThread):
    """Plans (``mode`` = folder/tempo/autodj/repair) or writes (``mode`` = write)."""

    log_line = Signal(str)
    # (success, message, payload). payload is the list of PlannedPlaylist for
    # the planning modes and the write_playlist_plan() summary for "write".
    # Deliberately not called ``finished``: that would shadow QThread.finished.
    done = Signal(bool, str, object)

    def __init__(
        self,
        library_path: str,
        mode: str,
        output_dir: str = "",
        tempo_range: tuple | None = None,
        energy_range: tuple | None = None,
        autodj_count: int = 20,
        seed_track: str = "",
        repair_paths: list | None = None,
        prefer_opus: bool = False,
        plan: list | None = None,
        feature_cache: dict | None = None,
    ) -> None:
        super().__init__()
        self.feature_cache = feature_cache
        self.library_path = library_path
        self.mode = mode
        self.output_dir = output_dir
        self.tempo_range = tempo_range
        self.energy_range = energy_range
        self.autodj_count = autodj_count
        self.seed_track = seed_track
        self.repair_paths = repair_paths or []
        self.prefer_opus = prefer_opus
        self.plan = plan or []
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set() or self.isInterruptionRequested()

    def _log(self, msg: str) -> None:
        self.log_line.emit(msg)

    def run(self) -> None:
        try:
            import playlist_generator as pg
        except ImportError as exc:
            self.done.emit(False, f"Import error: {exc}", [])
            return
        try:
            if self.mode == "write":
                self._run_write(pg)
                return
            plan = self._run_plan(pg)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "IndexCancelled" or self.is_cancelled():
                self.done.emit(False, "Cancelled.", [])
            else:
                self.done.emit(False, str(exc) or type(exc).__name__, [])
            return
        if plan is None:
            return  # done already emitted
        if self.is_cancelled():
            self.done.emit(False, "Cancelled.", [])
        else:
            self.done.emit(True, "Preview ready.", plan)

    # ── planning (no file is written) ────────────────────────────────────

    def _run_plan(self, pg):
        if self.mode == "folder":
            plan = pg.plan_folder_playlists(self.library_path, output_dir=self.output_dir or None)
            self._log(f"Found {len(plan)} album folder(s) with audio files.")
            return plan

        if self.mode == "tempo":
            from playlist_engine import bucket_by_tempo_energy

            tracks = _find_audio_files(self.library_path)
            self._log(f"Found {len(tracks)} tracks — analyzing tempo/energy…")
            result = bucket_by_tempo_energy(
                tracks,
                self.library_path,
                log_callback=self._log,
                cancel_event=self._cancel_event,
                tempo_range=self.tempo_range,
                energy_range=self.energy_range,
            )
            if result.get("excluded"):
                self._log(f"{result['excluded']} track(s) fell outside the tempo/energy range.")
            paths = result.get("playlist_paths", {})
            return [
                pg.plan_playlist(items, paths[key])
                for key, items in sorted(result.get("buckets", {}).items())
                if items and key in paths
            ]

        if self.mode == "autodj":
            from playlist_engine import autodj_playlist

            tracks = _find_audio_files(self.library_path)
            if not tracks:
                self.done.emit(False, "No audio files found in the library.", [])
                return None
            seed = next((t for t in tracks if _same_path(t, self.seed_track)), None)
            if seed is None:
                self.done.emit(
                    False,
                    f"Seed track is not in the library (or is in a reserved folder): {self.seed_track}",
                    [],
                )
                return None
            self._log(f"Building Auto-DJ playlist from {Path(seed).name}…")
            ordered = autodj_playlist(
                seed,
                tracks,
                n=self.autodj_count,
                log_callback=self._log,
                cancel_event=self._cancel_event,
                feature_cache=self.feature_cache,
            )
            out_path = os.path.join(self.library_path, "Playlists", "autodj.m3u")
            return [pg.plan_playlist(ordered, out_path)]

        if self.mode == "repair":
            paths = list(self.repair_paths)
            if not paths:
                pl_dir = Path(self.library_path) / "Playlists"
                paths = sorted(
                    str(p) for p in pl_dir.rglob("*")
                    if p.suffix.lower() in (".m3u", ".m3u8") and p.is_file()
                ) if pl_dir.is_dir() else []
                self._log(f"Checking the {len(paths)} playlist(s) in {pl_dir}…")
            plan = pg.plan_playlist_repair(
                paths, self.library_path, prefer_opus=self.prefer_opus, log_callback=self._log
            )
            for item in plan:
                name = Path(item.path).name
                if item.error:
                    self._log(f"! {name}: could not be read ({item.error})")
                    continue
                self._log(
                    f"{name}: {item.entries} entries, {len(item.changes)} fixable, "
                    f"{len(item.unresolved)} still missing"
                )
                for old, new in item.changes:
                    self._log(f"   {old}  →  {new}")
                for entry in item.unresolved:
                    self._log(f"   ! not found: {entry}")
            return plan

        raise ValueError(f"Unknown playlist mode: {self.mode}")

    # ── writing exactly the confirmed plan ───────────────────────────────

    def _run_write(self, pg) -> None:
        result = pg.write_playlist_plan(
            self.plan,
            backup=True,
            log_callback=self._log,
            log_dir=os.path.join(self.library_path, "Docs"),
        )
        n, errors = len(result["written"]), result["errors"]
        msg = f"Wrote {n} playlist(s)"
        if result["backups"]:
            msg += f", backed up {len(result['backups'])} overwritten file(s) as .bak"
        if errors:
            msg += f"; {len(errors)} failed"
        self.done.emit(not errors, msg + ".", result)


#: The four kinds of playlist: key, segment label, primary-button text, and
#: the one-line description shown under the segments.
_MODES = (
    ("folder", "Folder", "Preview folder playlists",
     "One playlist per album folder, named after its path (Artist_Album.m3u). "
     "The library's own folders (Docs, Playlists, Trash, …) are skipped."),
    ("tempo", "Tempo & energy", "Preview tempo playlists",
     "Groups tracks by BPM and energy into Playlists/slow_low.m3u … fast_high.m3u. "
     "Tracks outside the ranges are left out; analysis can take a while."),
    ("autodj", "Auto-DJ", "Preview Auto-DJ playlist",
     "Chains sonically close tracks from a seed track, so each flows into the next. "
     "Saved as Playlists/autodj.m3u."),
    ("repair", "Repair", "Find fixes",
     "Re-points broken entries after files moved or were renamed, by name or the same "
     "track in the other codec. Leave the list empty to check every playlist in Playlists/."),
)

_ROLE_DETAIL = int(QtCore.Qt.ItemDataRole.UserRole) + 1
_ROLE_BADGE = int(QtCore.Qt.ItemDataRole.UserRole) + 2
_ROLE_TONE = int(QtCore.Qt.ItemDataRole.UserRole) + 3


class _PlanRowDelegate(QtWidgets.QStyledItemDelegate):
    """A planned playlist as a glass row: the name, then a small status
    pill ("new", "replace", "written", …) and a muted detail on the right.
    Also used, without pills, for the repair list."""

    ROW_H = 38   # 34 px row + 4 px gap

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem,  # noqa: N802
                 index: QtCore.QModelIndex) -> QtCore.QSize:
        return QtCore.QSize(max(80, option.rect.width()), self.ROW_H)

    def paint(self, p: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem,
              index: QtCore.QModelIndex) -> None:
        from gui.themes.glass import font, paint_pill, qcolor
        from gui.themes.tokens import rgba

        t = get_manager().current
        m = t.material
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QtWidgets.QStyle.StateFlag.State_MouseOver)
        row = QtCore.QRectF(option.rect).adjusted(0, 2, -1, -2)
        p.save()
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            path = QtGui.QPainterPath()
            path.addRoundedRect(row, 12, 12)
            fill = (qcolor(m.selection) if selected
                    else qcolor(m.pressed) if hovered else qcolor(m.hover, 0.75))
            p.fillPath(path, fill)
            ink = qcolor(m.selection_text if selected else t.text_primary)
            quiet = qcolor(m.selection_text if selected else t.text_secondary)

            right = row.right() - 12
            badge = str(index.data(_ROLE_BADGE) or "")
            if badge:
                tone = {"accent": t.accent, "warning": t.warning, "success": t.success,
                        "danger": t.danger}.get(str(index.data(_ROLE_TONE) or ""), t.text_secondary)
                bf = font("micro", px=10, weight=700, tracking=0.06)
                bm = QtGui.QFontMetrics(bf)
                bw = bm.horizontalAdvance(badge.upper()) + 16
                pill = QtCore.QRectF(right - bw, row.center().y() - 10, bw, 20)
                paint_pill(p, pill, rgba(tone, 0.18), rgba(tone, 0.40))
                p.setFont(bf)
                p.setPen(qcolor(tone))
                p.drawText(pill, int(QtCore.Qt.AlignmentFlag.AlignCenter), badge.upper())
                right = pill.left() - 10
            detail = str(index.data(_ROLE_DETAIL) or "")
            if detail:
                df = font("caption", tabular=True)
                dm = QtGui.QFontMetrics(df)
                dw = min(dm.horizontalAdvance(detail), int((right - row.left()) * 0.45))
                p.setFont(df)
                p.setPen(quiet)
                p.drawText(QtCore.QRectF(right - dw, row.top(), dw, row.height()),
                           int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter),
                           dm.elidedText(detail, QtCore.Qt.TextElideMode.ElideLeft, dw))
                right -= dw + 12
            name = str(index.data(QtCore.Qt.ItemDataRole.DisplayRole) or "")
            nf = font("label", weight=500)
            nm = QtGui.QFontMetrics(nf)
            left = row.left() + 14
            p.setFont(nf)
            p.setPen(ink)
            p.drawText(QtCore.QRectF(left, row.top(), max(0.0, right - left), row.height()),
                       int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                       nm.elidedText(name, QtCore.Qt.TextElideMode.ElideMiddle,
                                     int(max(0.0, right - left))))
        finally:
            p.restore()


def _glass_list(name: str) -> QtWidgets.QListWidget:
    """A transparent list of glass rows (``_PlanRowDelegate``)."""
    view = QtWidgets.QListWidget()
    view.setObjectName(name)
    view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    view.setUniformItemSizes(True)
    view.setMouseTracking(True)
    view.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
    view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
    view.setItemDelegate(_PlanRowDelegate(view))
    return view


class PlaylistsWorkspace(WorkspaceBase):
    """Generate m3u playlists by folder, tempo/energy, or Auto-DJ, and repair
    existing ones — every one previewed before anything is written.

    Layout: a hero card (choose the kind, set its options, Preview), then a
    Preview card (the planned files as glass rows with a summary, or an
    empty state before the first run) whose Details hold the job's log."""

    #: Rows the preview list shows before it scrolls.
    _MAX_VISIBLE_ROWS = 8

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: PlaylistWorker | None = None
        self._run_buttons: list[QtWidgets.QPushButton] = []
        self._mode = _MODES[0][0]
        self._row_by_path: dict[str, QtWidgets.QListWidgetItem] = {}
        # Auto-DJ analysis is the slow part; keep it for the next run in this
        # library (one worker at a time uses it, so no locking is needed).
        self._feature_caches: dict[str, AudioFeatureCache] = {}
        self._build_ui()
        self._set_busy(False)

    # ── UI construction ───────────────────────────────────────────────────

    def _glyph_button(self, glyph: str, tip: str, slot) -> QtWidgets.QPushButton:
        from gui.widgets.glyphs import set_glyph

        btn = QtWidgets.QPushButton()
        btn.setFlat(True)
        btn.setFixedSize(34, 34)
        btn.setToolTip(tip)
        btn.setAccessibleName(tip.rstrip("…"))
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        set_glyph(btn, glyph, role="muted", size=17)
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _field_label(text: str) -> QtWidgets.QLabel:
        from gui.themes.glass import font

        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("mutedLabel")
        lbl.setFont(font("label"))
        return lbl

    def _build_ui(self) -> None:
        from gui.pages import SegmentedTabBar
        from gui.themes.glass import font
        from gui.widgets.glass import GlassPanel
        from gui.widgets.glass_parts import EmptyState, StatRow
        from gui.widgets.glyphs import set_glyph

        cl = self.content_layout

        subtitle = self._make_subtitle(
            "Build .m3u playlists from your library. You see every file before anything "
            "is written, and replaced playlists are kept as .bak copies.")
        cl.addWidget(subtitle)

        # ── Hero: the kind of playlist, its options, Preview ──────────────
        hero = self._make_card()
        hero.setObjectName("headerCard")
        hl = QtWidgets.QVBoxLayout(hero)
        hl.setContentsMargins(16, 16, 16, 16)
        hl.setSpacing(12)

        title = QtWidgets.QLabel("Make playlists")
        title.setObjectName("cardTitle")
        title.setFont(font("title"))
        hl.addWidget(title)

        self._mode_tabs = SegmentedTabBar()
        self._mode_tabs.setObjectName("playlistModeTabs")
        self._mode_tabs.setAccessibleName("Kind of playlist")
        for key, label, _button, _desc in _MODES:
            idx = self._mode_tabs.addTab(label)
            self._mode_tabs.setTabData(idx, key)
        tabs_row = QtWidgets.QHBoxLayout()
        tabs_row.addWidget(self._mode_tabs)
        tabs_row.addStretch(1)
        hl.addLayout(tabs_row)

        self._mode_desc = QtWidgets.QLabel(_MODES[0][3])
        self._mode_desc.setObjectName("sectionSubtitle")
        self._mode_desc.setWordWrap(True)
        hl.addWidget(self._mode_desc)

        # Options: one tidy well per kind
        self._options = QtWidgets.QStackedWidget()
        self._options.addWidget(self._folder_options(GlassPanel))
        self._options.addWidget(self._tempo_options(GlassPanel))
        self._options.addWidget(self._autodj_options(GlassPanel))
        self._options.addWidget(self._repair_options(GlassPanel, EmptyState))
        hl.addWidget(self._options)

        act = QtWidgets.QHBoxLayout()
        act.setSpacing(10)
        self._run_btn = self._make_primary_button(_MODES[0][2])
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.clicked.connect(lambda: self._on_run(self._mode))
        self._run_buttons.append(self._run_btn)
        act.addWidget(self._run_btn)
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setRange(0, 0)  # indeterminate
        self._prog_bar.setFixedSize(140, 6)
        self._prog_bar.setTextVisible(False)
        act.addWidget(self._prog_bar)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        set_glyph(self._cancel_btn, "close", size=14)
        self._cancel_btn.clicked.connect(self._on_cancel)
        act.addWidget(self._cancel_btn)
        act.addStretch(1)
        hl.addLayout(act)
        cl.addWidget(hero)

        self._mode_tabs.currentChanged.connect(self._on_mode_changed)
        self._on_mode_changed(0)

        # ── Preview: the planned files, then what was written ─────────────
        res = self._make_card()
        rl = QtWidgets.QVBoxLayout(res)
        rl.setContentsMargins(16, 16, 16, 14)
        rl.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        self._results_title = self._make_card_title("Preview")
        head.addWidget(self._results_title)
        head.addStretch(1)
        self._details_btn = QtWidgets.QPushButton("Details")
        self._details_btn.setFlat(True)
        self._details_btn.setCheckable(True)
        self._details_btn.setToolTip("Show the job's log")
        set_glyph(self._details_btn, "chevron_right", role="muted", size=14)
        self._details_btn.toggled.connect(self._toggle_details)
        head.addWidget(self._details_btn)
        rl.addLayout(head)

        self._results_stack = QtWidgets.QStackedWidget()
        self._results_empty = EmptyState(
            "No playlists yet",
            "Choose a kind above and press Preview. Nothing is saved until you "
            "confirm the list.")
        self._results_stack.addWidget(self._results_empty)

        body = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)
        self._stats = StatRow([("playlists", "playlists"), ("tracks", "tracks"),
                               ("replaced", "replaced (kept as .bak)")])
        bl.addWidget(self._stats)
        self._results_list = _glass_list("playlistResults")
        bl.addWidget(self._results_list)
        self._results_note = QtWidgets.QLabel("")
        self._results_note.setObjectName("mutedLabel")
        self._results_note.setFont(font("label"))
        self._results_note.setWordWrap(True)
        bl.addWidget(self._results_note)
        self._results_stack.addWidget(body)
        rl.addWidget(self._results_stack)

        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setObjectName("logText")
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(130)
        self._log_area.setPlaceholderText("The job's log appears here (and in Activity).")
        self._log_area.setVisible(False)
        rl.addWidget(self._log_area)
        cl.addWidget(res)

        cl.addStretch(1)

    def _options_well(self, panel_cls) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        well = panel_cls(surface="well")
        form = QtWidgets.QFormLayout(well)
        form.setContentsMargins(16, 12, 16, 12)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        return well, form

    def _folder_options(self, panel_cls) -> QtWidgets.QWidget:
        well, form = self._options_well(panel_cls)
        self._folder_out_entry = QtWidgets.QLineEdit()
        self._folder_out_entry.setPlaceholderText("Playlists/ inside the library")
        self._folder_out_entry.setMinimumWidth(120)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self._folder_out_entry, 1)
        row.addWidget(self._glyph_button(
            "folder", "Choose the output folder…",
            lambda: self._browse_dir(self._folder_out_entry)))
        form.addRow(self._field_label("Save to"), row)
        return well

    def _tempo_options(self, panel_cls) -> QtWidgets.QWidget:
        well, form = self._options_well(panel_cls)
        self._tempo_min = QtWidgets.QSpinBox()
        self._tempo_min.setRange(0, 300)
        self._tempo_min.setValue(0)
        self._tempo_max = QtWidgets.QSpinBox()
        self._tempo_max.setRange(0, 300)
        self._tempo_max.setValue(300)
        tempo = QtWidgets.QHBoxLayout()
        tempo.setSpacing(8)
        tempo.addWidget(self._tempo_min)
        tempo.addWidget(self._field_label("to"))
        tempo.addWidget(self._tempo_max)
        tempo.addWidget(self._field_label("BPM"))
        tempo.addStretch(1)
        form.addRow(self._field_label("Tempo"), tempo)

        self._energy_min = QtWidgets.QDoubleSpinBox()
        self._energy_min.setRange(0.0, 1.0)
        self._energy_min.setSingleStep(0.05)
        self._energy_min.setValue(0.0)
        self._energy_max = QtWidgets.QDoubleSpinBox()
        self._energy_max.setRange(0.0, 1.0)
        self._energy_max.setSingleStep(0.05)
        self._energy_max.setValue(1.0)
        for spin in (self._tempo_min, self._tempo_max, self._energy_min, self._energy_max):
            spin.setMinimumWidth(92)
        energy = QtWidgets.QHBoxLayout()
        energy.setSpacing(8)
        energy.addWidget(self._energy_min)
        energy.addWidget(self._field_label("to"))
        energy.addWidget(self._energy_max)
        energy.addStretch(1)
        form.addRow(self._field_label("Energy"), energy)
        return well

    def _autodj_options(self, panel_cls) -> QtWidgets.QWidget:
        well, form = self._options_well(panel_cls)
        self._autodj_count = QtWidgets.QSpinBox()
        self._autodj_count.setRange(5, 500)
        self._autodj_count.setValue(20)
        self._autodj_count.setMinimumWidth(92)
        count = QtWidgets.QHBoxLayout()
        count.addWidget(self._autodj_count)
        count.addWidget(self._field_label("tracks"))
        count.addStretch(1)
        form.addRow(self._field_label("Length"), count)

        self._autodj_seed = QtWidgets.QLineEdit()
        self._autodj_seed.setPlaceholderText("The track to start from (in your library)")
        self._autodj_seed.setMinimumWidth(120)
        seed = QtWidgets.QHBoxLayout()
        seed.setSpacing(6)
        seed.addWidget(self._autodj_seed, 1)
        seed.addWidget(self._glyph_button(
            "file", "Choose the seed track…", lambda: self._browse_file(self._autodj_seed)))
        form.addRow(self._field_label("Start from"), seed)
        return well

    def _repair_options(self, panel_cls, empty_cls) -> QtWidgets.QWidget:
        well = panel_cls(surface="well")
        lay = QtWidgets.QVBoxLayout(well)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(2)
        head.addWidget(self._field_label("Playlists to check"))
        head.addStretch(1)
        head.addWidget(self._glyph_button("plus", "Add playlists…", self._on_add_playlists))
        head.addWidget(self._glyph_button(
            "minus", "Remove the selected playlist",
            lambda: self._repair_list.takeItem(self._repair_list.currentRow())))
        head.addWidget(self._glyph_button("trash", "Clear the list",
                                          lambda: self._repair_list.clear()))
        lay.addLayout(head)

        self._repair_list = _glass_list("playlistRepairList")
        self._repair_list.setMinimumHeight(_PlanRowDelegate.ROW_H * 2 + 4)
        self._repair_list.setMaximumHeight(_PlanRowDelegate.ROW_H * 4 + 4)
        self._repair_empty = empty_cls(
            "Every playlist in Playlists/", "Add playlists to check only those.")
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self._repair_list, 0, 0)
        grid.addWidget(self._repair_empty, 0, 0)
        lay.addLayout(grid)
        model = self._repair_list.model()
        model.rowsInserted.connect(self._update_repair_empty)
        model.rowsRemoved.connect(self._update_repair_empty)
        model.modelReset.connect(self._update_repair_empty)

        self._prefer_opus_cb = QtWidgets.QCheckBox("Prefer Opus when a FLAC entry is missing")
        self._prefer_opus_cb.setToolTip(
            "Checked: a missing .flac entry is pointed at the same track's .opus.\n"
            "Unchecked: a missing .opus entry is pointed at the same track's .flac."
        )
        lay.addWidget(self._prefer_opus_cb)
        return well

    # ── View state ────────────────────────────────────────────────────────

    def _on_mode_changed(self, index: int) -> None:
        if not 0 <= index < len(_MODES):
            return
        key, _label, button, desc = _MODES[index]
        self._mode = key
        self._mode_desc.setText(desc)
        self._run_btn.setText(button)
        self._options.setCurrentIndex(index)
        # The stack sizes to its largest page; keep only the current one's.
        for i in range(self._options.count()):
            page = self._options.widget(i)
            page.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                               QtWidgets.QSizePolicy.Policy.Preferred if i == index
                               else QtWidgets.QSizePolicy.Policy.Ignored)
        self._options.adjustSize()

    def set_mode(self, key: str) -> None:
        """Show the options of playlist kind ``key`` (folder/tempo/autodj/repair)."""
        for i, (k, *_rest) in enumerate(_MODES):
            if k == key:
                self._mode_tabs.setCurrentIndex(i)
                self._on_mode_changed(i)
                return

    def mode(self) -> str:
        return self._mode

    def _update_repair_empty(self, *args) -> None:
        self._repair_empty.setVisible(self._repair_list.count() == 0)

    def _toggle_details(self, on: bool) -> None:
        from gui.widgets.glyphs import set_glyph

        self._log_area.setVisible(on)
        set_glyph(self._details_btn, "chevron_down" if on else "chevron_right",
                  role="muted", size=14)

    def _show_empty(self, title: str, message: str) -> None:
        from gui.themes import motion

        self._results_empty.set_text(title, message)
        if self._results_stack.currentIndex() != 0:
            self._results_stack.setCurrentIndex(0)
            motion.reveal(self._results_empty, duration="fast")

    def _show_plan(self, mode: str, plan: list) -> None:
        """The planned files as glass rows, with the summary above them."""
        from gui.themes import motion

        self._results_list.clear()
        self._row_by_path.clear()
        stats = self._stats
        if mode == "repair":
            fixes = sum(len(i.changes) for i in plan)
            missing = sum(len(i.unresolved) for i in plan)
            stats.tile("playlists").set_value(len(plan))
            stats.tile("tracks").set_label("entries re-pointed")
            stats.tile("tracks").set_value(fixes)
            stats.tile("replaced").set_label("still missing")
            stats.tile("replaced").set_value(missing, tone="warning" if missing else "")
        else:
            replaced = sum(1 for i in plan if i.exists)
            stats.tile("playlists").set_value(len(plan))
            stats.tile("tracks").set_label("tracks")
            stats.tile("tracks").set_value(f"{sum(i.entries for i in plan):,}")
            stats.tile("replaced").set_label("replaced (kept as .bak)")
            stats.tile("replaced").set_value(replaced, tone="warning" if replaced else "")
        for item in plan:
            if mode == "repair":
                name = Path(item.path).name
                detail = f"{len(item.changes)} fixed" + (
                    f" · {len(item.unresolved)} missing" if item.unresolved else "")
                badge, tone = "repair", "accent"
            else:
                name = self._display_path(item.path)
                detail = f"{item.entries} track{'s' if item.entries != 1 else ''}"
                badge, tone = ("replace", "warning") if item.exists else ("new", "accent")
            row = QtWidgets.QListWidgetItem(name)
            row.setToolTip(item.path)
            row.setData(_ROLE_DETAIL, detail)
            row.setData(_ROLE_BADGE, badge)
            row.setData(_ROLE_TONE, tone)
            self._results_list.addItem(row)
            self._row_by_path[os.path.normcase(os.path.abspath(item.path))] = row
        rows = min(len(plan), self._MAX_VISIBLE_ROWS)
        self._results_list.setFixedHeight(rows * _PlanRowDelegate.ROW_H + 4)
        self._results_note.setText("")
        if self._results_stack.currentIndex() != 1:
            self._results_stack.setCurrentIndex(1)
            motion.reveal(self._results_stack.currentWidget(), duration="fast")

    def _mark_written(self, result: dict) -> None:
        """After the write: each row says what happened to its file."""
        def _row(path: str):
            return self._row_by_path.get(os.path.normcase(os.path.abspath(path)))

        for path in result.get("written", []):
            row = _row(path)
            if row is not None:
                row.setData(_ROLE_BADGE, "written")
                row.setData(_ROLE_TONE, "success")
        for path, message in result.get("errors", []):
            row = _row(path)
            if row is not None:
                row.setData(_ROLE_BADGE, "failed")
                row.setData(_ROLE_TONE, "danger")
                row.setToolTip(f"{path}\n{message}")

    # ── State ─────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool) -> None:
        for btn in self._run_buttons:
            btn.setEnabled(not busy)
        self._prog_bar.setVisible(busy)
        self._cancel_btn.setVisible(busy)
        self._cancel_btn.setEnabled(busy)

    def is_busy(self) -> bool:
        return self._worker is not None

    def _status(self, text: str, role: str) -> None:
        tokens = get_manager().current
        colour = {
            "busy": tokens.warning, "ok": tokens.success,
            "error": tokens.danger, "neutral": tokens.text_muted,
        }[role]
        self.status_changed.emit(text, colour)

    def _start(self, worker: PlaylistWorker, name: str, *, library_write: bool = False) -> None:
        self._worker = worker
        worker.log_line.connect(self._on_worker_log)
        worker.done.connect(self._on_worker_done)
        self._set_busy(True)
        # Planning only reads; writing playlists changes files in the library
        # and takes part in the one-library-writer-at-a-time rule.
        self._start_worker(worker, name, library_write=library_write)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_run(self, mode: str) -> None:
        if self.is_busy():
            return  # buttons are disabled; guards a queued double-click
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        seed = self._autodj_seed.text().strip()
        if mode == "autodj" and not seed:
            QtWidgets.QMessageBox.warning(
                self, "Seed Track Needed",
                "Choose the track the Auto-DJ playlist should start from.",
            )
            return

        self._log_area.clear()
        self._log(f"Previewing {mode} playlists…", "info")
        self._status("Preparing playlists…", "busy")
        self._results_title.setText("Preview")
        self._show_empty("Working out the playlists…",
                         "Nothing is written yet. Open Details to follow the job.")

        repair_paths = [
            self._repair_list.item(i).text()
            for i in range(self._repair_list.count())
        ] if mode == "repair" else []
        tempo_range = (self._tempo_min.value(), self._tempo_max.value())
        energy_range = (self._energy_min.value(), self._energy_max.value())
        worker = PlaylistWorker(
            library_path=self._library_path,
            mode=mode,
            output_dir=self._folder_out_entry.text().strip(),
            tempo_range=(min(tempo_range), max(tempo_range)),
            energy_range=(min(energy_range), max(energy_range)),
            autodj_count=self._autodj_count.value(),
            seed_track=seed,
            repair_paths=repair_paths,
            prefer_opus=self._prefer_opus_cb.isChecked(),
            feature_cache=self._feature_caches.setdefault(
                os.path.normcase(os.path.abspath(self._library_path)), AudioFeatureCache()
            ) if mode == "autodj" else None,
        )
        self._start(worker, "Playlist preview")

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._log("Cancelling…", "warn")

    @Slot(str)
    def _on_worker_log(self, line: str) -> None:
        if self.sender() is not self._worker:
            return
        self._log_area.appendPlainText(line)
        self._log(line)

    @Slot(bool, str, object)
    def _on_worker_done(self, success: bool, message: str, payload: object) -> None:
        worker = self.sender()
        if worker is not self._worker:
            return  # a job we already abandoned (e.g. library changed)
        self._worker = None
        self._set_busy(False)

        if not success:
            if worker.mode == "write":
                # Some files failed (or the write was cut short): say which.
                if isinstance(payload, dict):
                    self._mark_written(payload)
                self._results_note.setText(message)
                self._log(message, "error")
                self._status("Error", "error")
            elif message.startswith("Cancelled"):
                self._log("Cancelled — nothing was written.", "warn")
                self._status("Cancelled", "neutral")
                self._show_empty("Cancelled", "Nothing was written.")
            else:
                self._log(message, "error")
                self._status("Error", "error")
                self._show_empty("The playlists could not be prepared", message)
            return

        if worker.mode == "write":
            self._log(message, "ok")
            self._log_area.appendPlainText(message)
            self._status("Done", "ok")
            if isinstance(payload, dict):
                self._mark_written(payload)
            self._results_title.setText("Written")
            self._results_note.setText(message)
            return

        plan = list(payload or [])
        if worker.mode == "repair":
            plan = [item for item in plan if item.changes]
        if not plan:
            text = (
                "Nothing to repair — no missing entry could be matched."
                if worker.mode == "repair" else "Nothing to write."
            )
            self._log(text, "info")
            self._log_area.appendPlainText(text)
            self._status("Nothing to write", "neutral")
            self._show_empty("Nothing to write", text if worker.mode == "repair"
                             else "No playlist came out of these settings.")
            return

        self._show_plan(worker.mode, plan)
        if self._library_writes_blocked("Playlist write"):
            self._log("Another job is changing the library — nothing was written.", "warn")
            self._log_area.appendPlainText("Nothing was written (library busy).")
            self._status("Not saved", "neutral")
            self._results_note.setText(
                "Nothing was written: another job is changing the library. Preview again "
                "when it has finished.")
            return
        if not self._confirm_plan(worker.mode, plan):
            self._log("Nothing was written.", "info")
            self._log_area.appendPlainText("Nothing was written.")
            self._status("Not saved", "neutral")
            self._results_note.setText("Nothing was written.")
            return

        self._status("Writing playlists…", "busy")
        self._results_note.setText("Writing…")
        self._start(
            PlaylistWorker(library_path=self._library_path, mode="write", plan=plan),
            "Write playlists",
            library_write=True,
        )

    def _confirm_plan(self, mode: str, plan: list) -> bool:
        overwrites = sum(1 for item in plan if item.exists)
        if mode == "repair":
            fixes = sum(len(item.changes) for item in plan)
            missing = sum(len(item.unresolved) for item in plan)
            head = (
                f"Repair {len(plan)} playlist(s): {fixes} entr{'y' if fixes == 1 else 'ies'} "
                f"will be re-pointed"
                + (f", {missing} still missing (left unchanged)." if missing else ".")
            )
            rows = [
                f"• {Path(i.path).name}: {len(i.changes)} fixed"
                + (f", {len(i.unresolved)} missing" if i.unresolved else "")
                for i in plan
            ]
        else:
            head = (
                f"Write {len(plan)} playlist(s)"
                + (f" — {overwrites} existing file(s) will be replaced." if overwrites else ".")
            )
            rows = [
                f"• {'replace' if i.exists else 'new'}: {self._display_path(i.path)} "
                f"({i.entries} track{'s' if i.entries != 1 else ''})"
                for i in plan
            ]
        if len(rows) > _CONFIRM_LIST_LIMIT:
            rows = rows[:_CONFIRM_LIST_LIMIT] + [f"… and {len(rows) - _CONFIRM_LIST_LIMIT} more"]
        tail = (
            "\n\nReplaced files are backed up first (name.m3u.<time>.bak)."
            if overwrites or mode == "repair" else ""
        )
        for row in rows:
            self._log_area.appendPlainText(row)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Save playlists?",
            head + "\n\n" + "\n".join(rows) + tail,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _display_path(self, path: str) -> str:
        if self._library_path:
            try:
                return os.path.relpath(path, self._library_path)
            except ValueError:
                pass
        return path

    def _on_library_changed(self, path: str) -> None:
        # A plan is only valid for the library it was computed against. A
        # confirmed write is left to finish (it writes exactly the confirmed
        # plan, in the old library) and keeps the page busy until it does.
        if self._worker is not None and self._worker.mode == "write":
            self._log(
                "Library changed — the confirmed playlist write for the previous "
                "library will finish first.", "warn",
            )
            return
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
            self._set_busy(False)
            self._log("Library changed — the running playlist job was cancelled.", "warn")
            self._status("Cancelled", "neutral")
        # A preview belongs to the library it was made for.
        self._results_title.setText("Preview")
        self._show_empty(
            "No playlists yet",
            "Choose a kind above and press Preview. Nothing is saved until you "
            "confirm the list.")

    @Slot()
    def _on_add_playlists(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Playlists", self._library_path or str(Path.home()),
            "Playlists (*.m3u *.m3u8);;All files (*)"
        )
        for p in paths:
            self._repair_list.addItem(p)

    def _browse_dir(self, entry: QtWidgets.QLineEdit) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            entry.setText(folder)

    def _browse_file(self, entry: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Audio File", self._library_path or "",
            "Audio (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All (*)"
        )
        if path:
            entry.setText(path)
