"""The page header: page title, library chip with stats, search, Change library.

Liquid Glass puts no card behind the header: the title (34/700, −0.03em)
sits on the backdrop, with a library-path chip (a pill: fill .10, border
.16) and the muted stats under it. On the right: a search pill and the
"Change library" pill (44 px).

The search pill is real: Enter shows the Player and filters its track list
(``search_requested``; the main window routes it to the Player's existing
filter). Theme and Settings buttons are gone from the header — Settings is
in the sidebar (Ctrl+, still works) and the theme lives in Settings →
Appearance; the ``settings_requested`` / ``theme_requested`` signals remain
for callers.

Narrow windows (``fit_mode``): the library chip keeps a useful width
(``CHIP_MIN``). First the stats text goes (the chip's tooltip carries it),
then the search pill folds into a round search button; clicking it opens
the field in its place, and the field folds back when it loses focus.
Nothing animates: the header is laid out once per change.
"""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.animations import AnimatedButton
from gui.themes.manager import get_manager


def _abbreviate_path(path: str) -> str:
    """Return the last two path segments prefixed with '…  ' when longer."""
    if not path:
        return "No library selected"
    parts = Path(path).parts
    if len(parts) <= 2:
        return path
    tail = "  /  ".join(str(p) for p in parts[-2:])
    return f"…  {tail}"


def _search_icon(colour: QtGui.QColor, size: int = 16) -> QtGui.QIcon:
    """The magnifier (circle r7 + handle), painted for the search pill."""
    dpr = 2.0
    pm = QtGui.QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    try:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        k = size / 24.0
        pen = QtGui.QPen(colour, 2.0 * k * 1.2, QtCore.Qt.PenStyle.SolidLine,
                         QtCore.Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawEllipse(QtCore.QPointF(11 * k, 11 * k), 7 * k, 7 * k)
        p.drawLine(QtCore.QPointF(20 * k, 20 * k), QtCore.QPointF(16.5 * k, 16.5 * k))
    finally:
        p.end()
    return QtGui.QIcon(pm)


class _LibraryChip(QtWidgets.QWidget):
    """The library path in a pill (fill .10, border .16)."""

    def __init__(self, label: QtWidgets.QLabel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("libraryChip")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 4, 12, 5)
        lay.addWidget(label)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface
        from gui.themes.tokens import PILL, Surface, is_opaque

        m = current_material()
        # The well, a touch brighter on glass (fill .10, border .16).
        spec = (Surface("rgba(255,255,255,0.10)", "rgba(255,255,255,0.16)", radius=PILL)
                if m.is_glass and not is_opaque(m.well.fill) else
                Surface(m.well.fill, m.well.border, radius=PILL))
        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), spec)
        finally:
            p.end()


class TopBar(QtWidgets.QWidget):
    """The page header above the page stack.

    Emits:
        library_changed(str)  — user selected a new library folder
        search_requested(str) — Enter in the search pill
        settings_requested()  — kept for callers (no button any more)
        theme_requested()     — kept for callers (no button any more)
    """

    library_changed = Signal(str)
    search_requested = Signal(str)
    settings_requested = Signal()
    theme_requested = Signal()

    PATH_MAX_WIDTH = 380
    CHIP_MIN = 280        # the chip's least useful width before the header gives way
    SEARCH_MIN = 200
    PILL_H = 44

    # fit_mode() values, widest first.
    FULL, NO_STATS, COMPACT = "full", "no_stats", "compact"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from gui.themes.glass import font

        self.setObjectName("pageHeader")
        self._library_path: str = ""
        self._stats_text: str = ""
        self._mode = self.FULL

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(24, 10, 24, 16)
        layout.setSpacing(16)

        # ── Title and library line ────────────────────────────────────────
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(6)
        col.setContentsMargins(0, 0, 0, 0)
        self._title = QtWidgets.QLabel("Home")
        self._title.setObjectName("pageTitle")
        self._title.setFont(font("display"))
        col.addWidget(self._title)

        line = QtWidgets.QHBoxLayout()
        line.setSpacing(10)
        line.setContentsMargins(0, 0, 0, 0)
        from gui.widgets.glass import ElidedLabel
        self._path_label = ElidedLabel("No library selected", QtCore.Qt.TextElideMode.ElideMiddle)
        self._path_label.setObjectName("libraryPath")
        self._path_label.setMaximumWidth(self.PATH_MAX_WIDTH)
        self._path_label.setFont(font("label"))
        self._chip = _LibraryChip(self._path_label)
        line.addWidget(self._chip)

        # Stats: three (number, unit) pairs: "4,812 tracks · 162.4 GB · 391 artists"
        # (one box, so a narrow header hides them together).
        self._stats_box = QtWidgets.QWidget()
        self._stats_box.setObjectName("headerStatsBox")
        stats = QtWidgets.QHBoxLayout(self._stats_box)
        stats.setContentsMargins(0, 0, 0, 0)
        stats.setSpacing(10)
        self._stat_widgets: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = []
        for i, (num, unit) in enumerate((("–", "tracks"), ("–", "GB"), ("–", "artists"))):
            if i:
                dot = QtWidgets.QLabel("·")
                dot.setObjectName("headerStats")
                dot.setFont(font("label"))
                stats.addWidget(dot)
            num_lbl = QtWidgets.QLabel(num)
            num_lbl.setObjectName("headerStats")
            num_lbl.setFont(font("label"))
            unit_lbl = QtWidgets.QLabel(unit)
            unit_lbl.setObjectName("headerStats")
            unit_lbl.setFont(font("label"))
            pair = QtWidgets.QHBoxLayout()
            pair.setSpacing(4)
            pair.addWidget(num_lbl)
            pair.addWidget(unit_lbl)
            stats.addLayout(pair)
            self._stat_widgets.append((num_lbl, unit_lbl))
        line.addWidget(self._stats_box)
        line.addStretch(1)
        col.addLayout(line)
        layout.addLayout(col, 1)

        # ── Search (routes to the Player's filter) ────────────────────────
        self._search = QtWidgets.QLineEdit()
        self._search.setObjectName("headerSearch")
        self._search.setPlaceholderText("Search artists, albums, tracks")
        self._search.setAccessibleName("Search the library")
        self._search.setToolTip("Press Enter to show matching tracks in the Player")
        self._search.setClearButtonEnabled(True)
        self._search.setFixedHeight(self.PILL_H)
        self._search.setMinimumWidth(self.SEARCH_MIN)
        self._search.setMaximumWidth(300)
        self._search.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                   QtWidgets.QSizePolicy.Policy.Fixed)
        self._search_action = self._search.addAction(
            QtGui.QIcon(), QtWidgets.QLineEdit.ActionPosition.LeadingPosition)
        self._search.returnPressed.connect(self._on_search)
        self._search.installEventFilter(self)
        layout.addWidget(self._search, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        # The same search as a round button, when the header is narrow.
        from gui.widgets.glyphs import set_glyph

        self._search_btn = AnimatedButton()
        self._search_btn.setObjectName("headerSearchButton")
        self._search_btn.setFixedSize(self.PILL_H, self.PILL_H)
        self._search_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._search_btn.setToolTip("Search artists, albums, tracks")
        self._search_btn.setAccessibleName("Search the library")
        set_glyph(self._search_btn, "search")
        self._search_btn.clicked.connect(self.open_search)
        self._search_btn.hide()
        layout.addWidget(self._search_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        # ── Change library ────────────────────────────────────────────────
        self._change_btn = AnimatedButton("Change library")
        self._change_btn.setObjectName("headerPill")
        self._change_btn.setFixedHeight(self.PILL_H)
        self._change_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._change_btn.setToolTip("Open another music library folder (Ctrl+O)")
        self._change_btn.clicked.connect(self._on_change_library)
        layout.addWidget(self._change_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        # ── Theme wiring ──────────────────────────────────────────────────
        mgr = get_manager()
        mgr.theme_changed.connect(self._on_theme_changed)
        self._apply_theme(mgr.current)

    # ── Public API ────────────────────────────────────────────────────────

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def title(self) -> str:
        return self._title.text()

    def set_library(self, path: str) -> None:
        self._library_path = path
        if path:
            self._path_label.setText(path)      # elided to fit when painted
        else:
            self._path_label.setText("No library selected")
        # The path colour depends on whether a library is set.
        self._apply_path_colour(get_manager().current)
        self._clear_stats()
        self._refit()

    def set_stats(self, tracks: int, size_gb: float, artists: int) -> None:
        if tracks == 0:
            self._clear_stats()
            return
        units = ["track" if tracks == 1 else "tracks", "GB",
                 "artist" if artists == 1 else "artists"]
        values = [f"{tracks:,}", f"{size_gb:.1f}", f"{artists:,}"]
        for (num_lbl, unit_lbl), val, unit in zip(self._stat_widgets, values, units):
            num_lbl.setText(val)
            unit_lbl.setText(unit)
        self._stats_text = " · ".join(f"{v} {u}" for v, u in zip(values, units))
        self._refit()

    @property
    def library_path(self) -> str:
        return self._library_path

    def search_field(self) -> QtWidgets.QLineEdit:
        return self._search

    def search_button(self) -> QtWidgets.QPushButton:
        return self._search_btn

    def fit_mode(self) -> str:
        """``FULL`` (chip, stats, search pill), ``NO_STATS`` (the stats text
        hidden) or ``COMPACT`` (the search pill folded into a round button)."""
        return self._mode

    def chip_width(self) -> int:
        return self._chip.width()

    def open_search(self) -> None:
        """Show the search field (in a compact header it takes the button's
        place until it loses focus) and put the cursor in it."""
        self._search_btn.hide()
        self._search.show()
        self._search.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self._search.selectAll()

    # ── Internal ──────────────────────────────────────────────────────────

    # ── Narrow widths ─────────────────────────────────────────────────────

    def _mode_for(self, width: int) -> str:
        lay = self.layout()
        m = lay.contentsMargins()
        sp = lay.spacing()
        room = width - m.left() - m.right() - self._change_btn.sizeHint().width() - sp
        with_search = room - self.SEARCH_MIN - sp
        chip = min(self._chip.sizeHint().width(), self.CHIP_MIN)
        stats = self._stats_box.sizeHint().width() + 10
        if with_search >= chip + stats:
            return self.FULL
        if with_search >= chip:
            return self.NO_STATS
        return self.COMPACT

    def _apply_mode(self, mode: str) -> None:
        self._mode = mode
        self._stats_box.setVisible(mode == self.FULL)
        compact = mode == self.COMPACT
        searching = self._search.hasFocus()
        self._search.setVisible(not compact or searching)
        self._search_btn.setVisible(compact and not searching)
        self._sync_chip_tooltip()

    def _refit(self) -> None:
        if self.width() > 0:
            self._apply_mode(self._mode_for(self.width()))

    def _sync_chip_tooltip(self) -> None:
        tip = self._library_path
        if tip and self._mode != self.FULL and self._stats_text:
            tip = f"{tip}\n{self._stats_text}"
        self._path_label.setToolTip(tip)
        self._chip.setToolTip(tip)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        mode = self._mode_for(event.size().width())
        if mode != self._mode:
            self._apply_mode(mode)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self._search and self._mode == self.COMPACT:
            kind = event.type()
            if kind == QtCore.QEvent.Type.FocusOut:
                # Fold back into the button once focus has really moved on.
                QtCore.QTimer.singleShot(0, self._fold_search)
            elif kind == QtCore.QEvent.Type.KeyPress and event.key() == QtCore.Qt.Key.Key_Escape:
                self._search.clear()
                self._search.clearFocus()
                self._fold_search()
                return True
        return super().eventFilter(obj, event)

    def _fold_search(self) -> None:
        if self._mode == self.COMPACT and not self._search.hasFocus():
            self._search.hide()
            self._search_btn.show()

    def _clear_stats(self) -> None:
        for (num_lbl, unit_lbl), unit in zip(self._stat_widgets, ("tracks", "GB", "artists")):
            num_lbl.setText("–")
            unit_lbl.setText(unit)
        self._stats_text = ""
        self._sync_chip_tooltip()

    def path_colour(self) -> str:
        t = get_manager().current
        return t.text_primary if self._library_path else t.text_secondary

    def _apply_path_colour(self, tokens) -> None:
        pal = self._path_label.palette()
        pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(self.path_colour()))
        self._path_label.setPalette(pal)

    def _apply_theme(self, tokens) -> None:
        self._apply_path_colour(tokens)
        self._search_action.setIcon(_search_icon(QtGui.QColor(tokens.text_secondary)))
        self._chip.update()

    def _on_theme_changed(self, tokens) -> None:
        self._apply_theme(tokens)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_search(self) -> None:
        text = self._search.text().strip()
        if text:
            self.search_requested.emit(text)

    def _on_change_library(self) -> None:
        start = self._library_path or str(Path.home())

        dlg = QtWidgets.QFileDialog()
        dlg.setWindowTitle("Select Music Library Folder")
        dlg.setDirectory(start)
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        win = self.window()
        if win and win.isVisible():
            dlg.resize(820, 560)
            fg = win.frameGeometry()
            dlg.move(
                fg.left() + (fg.width()  - dlg.width())  // 2,
                fg.top()  + (fg.height() - dlg.height()) // 2,
            )

        if dlg.exec():
            selected = dlg.selectedFiles()
            if selected:
                self.library_changed.emit(selected[0])
