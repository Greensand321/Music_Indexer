"""Item views on the glass: rounded rows, hover and selection, cell kinds.

One look for every list, table and tree that sits on a card (Metadata,
Organize, Library Sync, the File Cleanup preview, Playlists' plan rows, …).
Everything paints from the active theme at paint time, so a theme, accent or
transparency change only needs a repaint.

``GlassRowDelegate``
    Paints each row as one rounded glass band (radius 12, a 34 px band in a
    38 px row): an optional resting fill (the mockup's list rows), the
    material's hover and selection across the whole row, a continuous
    hairline between rows when there is no resting fill, and per-cell
    *kinds* set with ``mark()``:

    * ``"muted"``: secondary text (current values, rows left alone);
    * ``"accent"``: a changed value, drawn on an accent-tinted chip;
    * ``"pill"``: the cell drawn as a glass pill (``TONE_ROLE`` picks the
      tone, ``PILL_ROLE`` the text when it differs from the cell's);
    * ``"pill_reason"``: a pill followed by a muted reason (``DETAIL_ROLE``).

    The model text never changes: tests, sorting, tooltips and screen
    readers see the same strings. Hover is tracked per row by the delegate
    itself (``hover_row()``), so any view gets whole-row hover.
    Subclasses paint their own row contents with ``paint_band`` (the row's
    fill, hover and selection) and ``band_rect``.
``style_item_view``
    Puts an existing view on the glass: the delegate, transparent, no grid,
    no frame, no alternating colours, middle- or right-elided text.
``GlassTree`` / ``GlassTable`` / ``GlassList``
    ``QTreeWidget`` / ``QTableWidget`` / ``QListWidget`` set up that way, with
    ``fit_rows`` (exactly as tall as the shown rows, then scroll) and, on the
    tree, a muted placeholder while it is empty.
``ChipList``
    A ``QListWidget`` whose items wrap as chips (the raw genre list).
``stretch_columns``
    Column resize modes so a view fits its card instead of scrolling sideways.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import current_material, font, paint_surface, qcolor
from gui.themes.manager import get_manager
from gui.themes.tokens import RADIUS
from gui.widgets.pill import paint_tone_pill, pill_font

Qt = QtCore.Qt

KIND_ROLE = Qt.ItemDataRole.UserRole + 40
TONE_ROLE = Qt.ItemDataRole.UserRole + 41
PILL_ROLE = Qt.ItemDataRole.UserRole + 42
DETAIL_ROLE = Qt.ItemDataRole.UserRole + 43

ROW_HEIGHT = 38        # a 34 px band with a 2 px gap above and below
ROW_GAP = 2
CELL_PAD = 6           # text inset inside a cell (lines up with the header's)
_CHIP_PAD = 8
# Item views with this property are transparent on their card (the shared
# item-view rule in gui/themes/manager.py).
GLASS_PROPERTY = "glassView"


def mark(item, kind: str, column: int | None = None, *, tone: str = "",
         pill: str = "", detail: str = "") -> None:
    """Give a cell a painting kind (see the module doc).

    ``item`` is a ``QTreeWidgetItem`` (pass ``column``; ``None`` marks every
    column), a ``QTableWidgetItem`` or a ``QListWidgetItem``."""
    values = ((KIND_ROLE, kind), (TONE_ROLE, tone), (PILL_ROLE, pill), (DETAIL_ROLE, detail))
    if isinstance(item, QtWidgets.QTreeWidgetItem):
        columns = range(item.columnCount()) if column is None else (column,)
        for col in columns:
            for role, value in values:
                item.setData(col, role, value or None)
    else:
        for role, value in values:
            item.setData(role, value or None)


def kind_of(item, column: int = 0) -> str:
    """The painting kind of a cell ("" when plain)."""
    if isinstance(item, QtWidgets.QTreeWidgetItem):
        return str(item.data(column, KIND_ROLE) or "")
    return str(item.data(KIND_ROLE) or "")


# ── Hover ─────────────────────────────────────────────────────────────────────

class _HoverTracker(QtCore.QObject):
    """Records the row under the pointer (a whole row, whatever the column)
    and repaints the viewport when it changes."""

    def __init__(self, view: QtWidgets.QAbstractItemView) -> None:
        super().__init__(view)
        self._view = view
        self.row = -1
        self.parent_index = QtCore.QPersistentModelIndex()
        view.setMouseTracking(True)
        view.viewport().setMouseTracking(True)
        view.viewport().installEventFilter(self)

    def _set(self, index: QtCore.QModelIndex) -> None:
        row = index.row() if index.isValid() else -1
        parent = QtCore.QPersistentModelIndex(index.parent()) if index.isValid() \
            else QtCore.QPersistentModelIndex()
        if row != self.row or parent != self.parent_index:
            self.row, self.parent_index = row, parent
            self._view.viewport().update()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QtCore.QEvent.Type.MouseMove:
            self._set(self._view.indexAt(event.position().toPoint()))
        elif kind in (QtCore.QEvent.Type.Leave, QtCore.QEvent.Type.HoverLeave):
            self._set(QtCore.QModelIndex())
        return False

    def matches(self, index: QtCore.QModelIndex) -> bool:
        return (self.row >= 0 and index.row() == self.row
                and QtCore.QPersistentModelIndex(index.parent()) == self.parent_index)


def hover_tracker(view: QtWidgets.QAbstractItemView) -> _HoverTracker:
    """The view's one hover tracker (shared by all its delegates)."""
    tracker = view.findChild(_HoverTracker, "glassHoverTracker",
                             Qt.FindChildOption.FindDirectChildrenOnly)
    if tracker is None:
        tracker = _HoverTracker(view)
        tracker.setObjectName("glassHoverTracker")
    return tracker


# ── The delegate ──────────────────────────────────────────────────────────────

class GlassRowDelegate(QtWidgets.QStyledItemDelegate):
    """Rounded glass rows with hover/selection across the row and cell kinds."""

    def __init__(self, view: QtWidgets.QAbstractItemView, *, row_fill: bool = False,
                 row_height: int = ROW_HEIGHT, hairline: bool = True) -> None:
        super().__init__(view)
        self._view = view
        self._row_fill = row_fill
        self._row_height = row_height
        self._hairline = hairline
        self._hover = hover_tracker(view)

    def row_fill(self) -> bool:
        return self._row_fill

    def row_height(self) -> int:
        return self._row_height

    def hover_row(self) -> int:
        return self._hover.row

    def is_hovered(self, index: QtCore.QModelIndex) -> bool:
        return self._hover.matches(index)

    # ── Geometry ──────────────────────────────────────────────────────────
    def sizeHint(self, option, index) -> QtCore.QSize:  # noqa: N802
        hint = super().sizeHint(option, index)
        return QtCore.QSize(hint.width() + 2 * CELL_PAD, max(self._row_height, hint.height()))

    @staticmethod
    def _inset(rect: QtCore.QRect) -> QtCore.QRect:
        return rect.adjusted(CELL_PAD, 0, -CELL_PAD // 2, 0)

    def editorEvent(self, event, model, option, index) -> bool:  # noqa: N802
        # Hit-test the check box where it is painted (inset like the text).
        opt = QtWidgets.QStyleOptionViewItem(option)
        opt.rect = self._inset(option.rect)
        return super().editorEvent(event, model, opt, index)

    def band_rect(self, cell: QtCore.QRect) -> QtCore.QRectF:
        """The whole row's band, in viewport coordinates (cells clip it)."""
        view = self._view
        header = None
        if isinstance(view, QtWidgets.QTreeView):
            header = view.header()
        elif isinstance(view, QtWidgets.QTableView):
            header = view.horizontalHeader()
        width = view.viewport().width()
        left = 0
        if header is not None and header.length() > 0:
            left = -view.horizontalOffset()
            width = max(header.length(), width + view.horizontalOffset())
        return QtCore.QRectF(left, cell.top() + ROW_GAP, width,
                             max(0, cell.height() - 2 * ROW_GAP))

    # ── Painting ──────────────────────────────────────────────────────────
    def paint_band(self, painter: QtGui.QPainter, option, index, *,
                   tint: QtGui.QColor | None = None) -> tuple[bool, bool]:
        """Paint the row's band behind ``option.rect`` (the resting fill or
        the hairline, an optional ``tint``, hover and selection). Returns
        ``(selected, hovered)``."""
        m = current_material()
        state = option.state
        selected = bool(state & QtWidgets.QStyle.StateFlag.State_Selected)
        enabled = bool(state & QtWidgets.QStyle.StateFlag.State_Enabled)
        hovered = enabled and self.is_hovered(index)
        painter.save()
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setClipRect(option.rect)
            row = self.band_rect(option.rect)
            path = QtGui.QPainterPath()
            path.addRoundedRect(row, RADIUS.sm, RADIUS.sm)
            if self._row_fill:
                painter.fillPath(path, qcolor(m.hover, 0.75))
            elif self._hairline:
                # One continuous hairline between rows (the style's own is
                # drawn per cell, with gaps at the column edges).
                y = option.rect.bottom()
                painter.fillRect(QtCore.QRectF(row.left() + 8, y, row.width() - 16, 1),
                                 qcolor(m.hairline))
            if tint is not None and not selected:
                painter.fillPath(path, tint)
            if hovered:
                painter.fillPath(path, qcolor(m.hover))
            if selected:
                painter.fillPath(path, qcolor(m.selection))
        finally:
            painter.restore()
        return selected, hovered

    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        t = get_manager().current
        m = t.material
        selected, _hovered = self.paint_band(painter, option, index)
        enabled = bool(option.state & QtWidgets.QStyle.StateFlag.State_Enabled)

        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.rect = self._inset(option.rect)
        strip = (QtWidgets.QStyle.StateFlag.State_Selected
                 | QtWidgets.QStyle.StateFlag.State_MouseOver
                 | QtWidgets.QStyle.StateFlag.State_HasFocus)
        opt.state &= ~strip
        opt.backgroundBrush = QtGui.QBrush()
        opt.features &= ~QtWidgets.QStyleOptionViewItem.ViewItemFeature.Alternate

        kind = str(index.data(KIND_ROLE) or "")
        if selected:
            base = qcolor(m.selection_text) if m.kind == "classic" else qcolor(t.text_primary)
        else:
            base = None
        muted = qcolor(t.text_secondary)
        if selected and m.kind == "classic":
            muted = qcolor(m.selection_text, 0.8)
        colour = muted if kind in ("muted", "pill_reason") else base
        if colour is not None:
            for role in (QtGui.QPalette.ColorRole.Text, QtGui.QPalette.ColorRole.HighlightedText,
                         QtGui.QPalette.ColorRole.WindowText):
                opt.palette.setColor(role, colour)
        if not enabled:
            opt.palette.setColor(QtGui.QPalette.ColorRole.Text, qcolor(t.text_muted))

        widget = opt.widget or self._view
        style = widget.style()
        # Drawn without the widget: the band above replaces the style's (and
        # the stylesheet's) per-cell hover, selection and hairline.
        if kind in ("accent", "pill", "pill_reason") and opt.text:
            text = opt.text
            text_rect = style.subElementRect(
                QtWidgets.QStyle.SubElement.SE_ItemViewItemText, opt, widget)
            opt.text = ""
            style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, None)
            painter.save()
            try:
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                painter.setClipRect(option.rect)
                self._paint_kind(painter, QtCore.QRectF(text_rect), text, kind, index, opt,
                                 t, selected, enabled)
            finally:
                painter.restore()
            return
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, None)

    def _paint_kind(self, painter, rect: QtCore.QRectF, text: str, kind: str, index, opt,
                    t, selected: bool, enabled: bool) -> None:
        m = t.material
        rect = rect.adjusted(2, 0, -2, 0)
        if rect.width() <= 4:
            return
        if kind == "accent":
            fm = QtGui.QFontMetrics(opt.font)
            h = min(rect.height() - 4, fm.height() + 6)
            w = min(fm.horizontalAdvance(text) + 2 * _CHIP_PAD + 2, rect.width())
            chip = QtCore.QRectF(rect.left(), rect.center().y() - h / 2, w, h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(t.accent, 0.16 if enabled else 0.08))
            painter.drawRoundedRect(chip, h / 2, h / 2)
            if not enabled:
                pen = qcolor(t.text_muted)
            elif selected and m.kind == "classic":
                pen = qcolor(m.selection_text)
            else:
                # accent_ink: the accent darkened/lightened until it reads on
                # an accent tint (plain accent was 3.8:1 on Pearl).
                pen = qcolor(t.accent_ink)
            painter.setPen(pen)
            painter.setFont(opt.font)
            inner = chip.adjusted(_CHIP_PAD, 0, -_CHIP_PAD + 2, 0)
            painter.drawText(inner, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             fm.elidedText(text, Qt.TextElideMode.ElideRight, int(inner.width()) + 1))
            return

        label = str(index.data(PILL_ROLE) or text)
        tone = str(index.data(TONE_ROLE) or "") or "accent"
        pfont = pill_font()
        pfm = QtGui.QFontMetrics(pfont)
        h = min(rect.height() - 6, pfm.height() + 6)
        w = min(pfm.horizontalAdvance(label) + 18, rect.width())
        pill = QtCore.QRectF(rect.left(), rect.center().y() - h / 2, w, h)
        paint_tone_pill(painter, pill, label, tone, pfont, opacity=1.0 if enabled else 0.6)
        if kind == "pill_reason":
            detail = str(index.data(DETAIL_ROLE) or "")
            rest = QtCore.QRectF(pill.right() + 8, rect.top(), rect.right() - pill.right() - 8,
                                 rect.height())
            if detail and rest.width() > 8:
                fm = QtGui.QFontMetrics(opt.font)
                painter.setFont(opt.font)
                painter.setPen(opt.palette.color(QtGui.QPalette.ColorRole.Text))
                painter.drawText(rest, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                 fm.elidedText(detail, Qt.TextElideMode.ElideRight, int(rest.width())))


def style_item_view(view: QtWidgets.QAbstractItemView, *,
                    elide: QtCore.Qt.TextElideMode = Qt.TextElideMode.ElideRight,
                    row_fill: bool = False, row_height: int = ROW_HEIGHT,
                    delegate: QtWidgets.QAbstractItemDelegate | None = None,
                    ) -> QtWidgets.QAbstractItemView:
    """Put ``view`` on the glass: ``GlassRowDelegate`` (or ``delegate``),
    transparent on its card, no grid, no frame, no alternating colours,
    elided text, rows ``row_height`` high."""
    view.setProperty(GLASS_PROPERTY, "true")
    view.setItemDelegate(delegate or GlassRowDelegate(view, row_fill=row_fill,
                                                      row_height=row_height))
    view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    view.setAlternatingRowColors(False)
    view.setTextElideMode(elide)
    view.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
    view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
    if isinstance(view, QtWidgets.QTreeView):
        view.setUniformRowHeights(True)
        view.setAllColumnsShowFocus(True)
        view.header().setHighlightSections(False)
        view.header().setMinimumSectionSize(32)
    if isinstance(view, QtWidgets.QTableView):
        view.setShowGrid(False)
        view.setWordWrap(False)
        view.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        view.verticalHeader().setDefaultSectionSize(row_height)
        view.horizontalHeader().setHighlightSections(False)
        view.horizontalHeader().setMinimumSectionSize(32)
    if isinstance(view, QtWidgets.QListView):
        view.setUniformItemSizes(True)
    return view


def stretch_columns(header: QtWidgets.QHeaderView, modes: dict[int, object]) -> None:
    """Section resize modes so a view fits its card instead of scrolling
    sideways: ``{column: "stretch" | "contents" | <fixed px>}``."""
    RM = QtWidgets.QHeaderView.ResizeMode
    header.setStretchLastSection(False)
    for col, mode in modes.items():
        if mode == "stretch":
            header.setSectionResizeMode(col, RM.Stretch)
        elif mode == "contents":
            header.setSectionResizeMode(col, RM.ResizeToContents)
        else:
            header.setSectionResizeMode(col, RM.Fixed)
            header.resizeSection(col, int(mode))


# ── Views ─────────────────────────────────────────────────────────────────────

class _GlassViewMixin:
    """The shared view set-up and ``fit_rows``."""

    def _glass_setup(self, row_fill: bool, row_height: int, elide) -> None:
        self._row_h = row_height
        style_item_view(self, elide=elide, row_fill=row_fill, row_height=row_height)

    def hover_row(self) -> int:
        return hover_tracker(self).row

    def _row_count(self) -> int:
        if isinstance(self, QtWidgets.QTreeWidget):
            return sum(1 for i in range(self.topLevelItemCount())
                       if not self.topLevelItem(i).isHidden())
        if isinstance(self, QtWidgets.QTableWidget):
            return self.rowCount()
        return self.count()

    def fit_rows(self, max_rows: int = 10, min_rows: int = 1) -> None:
        """Be exactly as tall as the shown rows (``min_rows``..``max_rows``),
        then scroll: a short result never leaves an empty well in its card."""
        rows = max(min_rows, min(self._row_count(), max_rows))
        header = 0
        if isinstance(self, QtWidgets.QTreeView) and not self.isHeaderHidden():
            header = self.header().sizeHint().height()
        elif isinstance(self, QtWidgets.QTableView) and self.horizontalHeader().isVisibleTo(self):
            header = self.horizontalHeader().sizeHint().height()
        self.setFixedHeight(header + rows * self._row_h + 2 * self.frameWidth() + 2)


class GlassTree(_GlassViewMixin, QtWidgets.QTreeWidget):
    """A flat ``QTreeWidget`` drawn as glass rows."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, row_fill: bool = False,
                 row_height: int = ROW_HEIGHT,
                 elide: QtCore.Qt.TextElideMode = Qt.TextElideMode.ElideMiddle) -> None:
        super().__init__(parent)
        self._glass_setup(row_fill, row_height, elide)
        self.setRootIsDecorated(False)
        self.setIndentation(0)
        self._placeholder = ""

    def set_placeholder(self, text: str) -> None:
        """Muted text shown in the view while it has no rows."""
        self._placeholder = text
        self.viewport().update()

    def placeholder(self) -> str:
        return self._placeholder

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._placeholder and self.topLevelItemCount() == 0:
            p = QtGui.QPainter(self.viewport())
            try:
                p.setFont(font("label"))
                p.setPen(qcolor(get_manager().current.text_muted))
                rect = self.viewport().rect().adjusted(16, 12, -16, -12)
                p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                           self._placeholder)
            finally:
                p.end()


class GlassTable(_GlassViewMixin, QtWidgets.QTableWidget):
    """A ``QTableWidget`` drawn as glass rows (no grid, no row numbers)."""

    def __init__(self, rows: int = 0, columns: int = 0, parent: QtWidgets.QWidget | None = None,
                 *, row_fill: bool = False, row_height: int = ROW_HEIGHT,
                 elide: QtCore.Qt.TextElideMode = Qt.TextElideMode.ElideMiddle) -> None:
        super().__init__(rows, columns, parent)
        self._glass_setup(row_fill, row_height, elide)
        self.setCornerButtonEnabled(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)


class GlassList(_GlassViewMixin, QtWidgets.QListWidget):
    """A ``QListWidget`` drawn as glass rows.

    ``max_rows`` > 0: the list is exactly as tall as its rows, up to that
    many (then it scrolls), so a short list never leaves an empty well."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, row_fill: bool = True,
                 row_height: int = ROW_HEIGHT, max_rows: int = 0,
                 elide: QtCore.Qt.TextElideMode = Qt.TextElideMode.ElideMiddle) -> None:
        super().__init__(parent)
        self._glass_setup(row_fill, row_height, elide)
        self._max_rows = max_rows
        if max_rows:
            self.model().rowsInserted.connect(self._fit)
            self.model().rowsRemoved.connect(self._fit)
            self.model().modelReset.connect(self._fit)
            self._fit()

    def _fit(self, *_args) -> None:
        self.fit_rows(self._max_rows)


# ── Chips ─────────────────────────────────────────────────────────────────────

class ChipDelegate(QtWidgets.QStyledItemDelegate):
    """Each item as a small rounded chip on the well surface."""

    _PAD_X = 12

    def sizeHint(self, option, index) -> QtCore.QSize:  # noqa: N802
        fm = QtGui.QFontMetrics(font("label"))
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        return QtCore.QSize(fm.horizontalAdvance(text) + 2 * self._PAD_X + 6, fm.height() + 16)

    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        t = get_manager().current
        m = current_material()
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        rect = QtCore.QRectF(option.rect).adjusted(3, 4, -3, -4)
        painter.save()
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            paint_surface(painter, rect, m.well, radius=rect.height() / 2)
            if option.state & QtWidgets.QStyle.StateFlag.State_Selected:
                path = QtGui.QPainterPath()
                path.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
                painter.fillPath(path, qcolor(m.selection))
            f = font("label")
            painter.setFont(f)
            painter.setPen(qcolor(t.text_primary))
            fm = QtGui.QFontMetrics(f)
            inner = rect.adjusted(self._PAD_X - 3, 0, -(self._PAD_X - 3), 0)
            painter.drawText(inner, int(Qt.AlignmentFlag.AlignCenter),
                             fm.elidedText(text, Qt.TextElideMode.ElideRight, int(inner.width())))
        finally:
            painter.restore()


class ChipList(QtWidgets.QListWidget):
    """Items laid out as wrapping chips (left to right, then the next line).

    As tall as its lines of chips, up to ``max_height`` (then it scrolls)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, max_height: int = 140) -> None:
        super().__init__(parent)
        self._max_height = max_height
        self.setProperty(GLASS_PROPERTY, "true")
        self.setItemDelegate(ChipDelegate(self))
        self.setViewMode(QtWidgets.QListView.ViewMode.ListMode)
        self.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setUniformItemSizes(False)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.model().rowsInserted.connect(self._fit)
        self.model().rowsRemoved.connect(self._fit)
        self.model().modelReset.connect(self._fit)
        self._fit()

    def _fit(self, *_args) -> None:
        """Lay the chips out the way the view will and take that height."""
        width = max(1, self.viewport().width() - 4)
        option = QtWidgets.QStyleOptionViewItem()
        option.initFrom(self)
        delegate = self.itemDelegate()
        x, lines, line_h = 0, 1 if self.count() else 0, 0
        for i in range(self.count()):
            hint = delegate.sizeHint(option, self.indexFromItem(self.item(i)))
            line_h = max(line_h, hint.height())
            if x and x + hint.width() > width:
                lines += 1
                x = 0
            x += hint.width()
        height = lines * line_h + 2 * self.frameWidth() + 4
        self.setFixedHeight(max(line_h or 36, min(height, self._max_height)))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self._fit()
