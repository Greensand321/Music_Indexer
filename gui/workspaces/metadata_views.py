"""Glass item views and small layout helpers for the Metadata page.

Shared by the Metadata panels only (``tag_fixer.py``, ``genres.py``,
``tag_history.py``); everything here paints from the active theme at paint
time, so a theme, accent or transparency change only needs a repaint.

``GlassRowDelegate``
    Paints each row of an item view as one rounded glass row (radius 12,
    about 34 px high): an optional resting fill (the mockup's list rows),
    the material's hover and selection overlays across the whole row, and
    per-cell *kinds* set with ``mark()``:

    * ``"muted"``: secondary text (current values, rows left alone);
    * ``"accent"``: a changed value, drawn on an accent-tinted chip;
    * ``"pill"``: the cell drawn as a pill (``TONE_ROLE`` picks the tone,
      ``PILL_ROLE`` the text when it differs from the cell's);
    * ``"pill_reason"``: a pill followed by a muted reason (``DETAIL_ROLE``).

    The model text never changes: tests, sorting, tooltips and screen
    readers see the same strings as before.
``GlassTree`` / ``GlassTable`` / ``GlassList``
    ``QTreeWidget`` / ``QTableWidget`` / ``QListWidget`` with the delegate,
    no grid, no alternating colours, middle-elided text and whole-row hover.
``ChipList``
    A ``QListWidget`` whose items wrap as chips (the raw genre list).
``FlowLayout``
    A layout that wraps its widgets onto further lines when narrow, so
    button rows never force a horizontal scroll at the minimum window width.
``card_heading``
    A card's title (the hero card uses the title role) over one muted line.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import apply_font, current_material, font, paint_surface, qcolor
from gui.themes.manager import get_manager
from gui.themes.tokens import RADIUS
from gui.widgets.pill import pill_colours

Qt = QtCore.Qt

KIND_ROLE = Qt.ItemDataRole.UserRole + 40
TONE_ROLE = Qt.ItemDataRole.UserRole + 41
PILL_ROLE = Qt.ItemDataRole.UserRole + 42
DETAIL_ROLE = Qt.ItemDataRole.UserRole + 43

ROW_HEIGHT = 38        # 34 px painted row + a 2 px gap above and below
_ROW_GAP = 2
_CHIP_PAD = 8
_CELL_PAD = 6          # text inset inside a cell (lines up with the header's)
# Item views marked with this property are transparent on their card (QSS in
# gui/themes/manager.py, metadata block).
GLASS_PROPERTY = "metaGlass"


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


def _tone_ok(tokens, tone: str) -> str:
    return tone if tone and isinstance(getattr(tokens, tone, None), str) else "accent"


class GlassRowDelegate(QtWidgets.QStyledItemDelegate):
    """Rounded glass rows with hover/selection overlays and per-cell kinds."""

    def __init__(self, view: QtWidgets.QAbstractItemView, *, row_fill: bool = False,
                 row_height: int = ROW_HEIGHT) -> None:
        super().__init__(view)
        self._view = view
        self._row_fill = row_fill
        self._row_height = row_height

    def row_fill(self) -> bool:
        return self._row_fill

    # ── Geometry ──────────────────────────────────────────────────────────
    def sizeHint(self, option, index) -> QtCore.QSize:  # noqa: N802
        hint = super().sizeHint(option, index)
        return QtCore.QSize(hint.width() + 2 * _CELL_PAD, max(self._row_height, hint.height()))

    @staticmethod
    def _inset(rect: QtCore.QRect) -> QtCore.QRect:
        return rect.adjusted(_CELL_PAD, 0, -_CELL_PAD // 2, 0)

    def editorEvent(self, event, model, option, index) -> bool:  # noqa: N802
        # Hit-test the check box where it is painted (inset like the text).
        opt = QtWidgets.QStyleOptionViewItem(option)
        opt.rect = self._inset(option.rect)
        return super().editorEvent(event, model, opt, index)

    def _row_rect(self, cell: QtCore.QRect) -> QtCore.QRectF:
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
        return QtCore.QRectF(left, cell.top() + _ROW_GAP, width,
                             max(0, cell.height() - 2 * _ROW_GAP))

    # ── Painting ──────────────────────────────────────────────────────────
    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        t = get_manager().current
        m = t.material
        state = option.state
        selected = bool(state & QtWidgets.QStyle.StateFlag.State_Selected)
        enabled = bool(state & QtWidgets.QStyle.StateFlag.State_Enabled)
        hovered = enabled and getattr(self._view, "hover_row", lambda: -1)() == index.row()

        painter.save()
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setClipRect(option.rect)
            row = self._row_rect(option.rect)
            path = QtGui.QPainterPath()
            path.addRoundedRect(row, RADIUS.sm, RADIUS.sm)
            if self._row_fill:
                painter.fillPath(path, qcolor(m.hover, 0.75))
            else:
                # One continuous hairline between rows (the style's own is
                # drawn per cell, with gaps at the column edges).
                y = option.rect.bottom()
                painter.fillRect(QtCore.QRectF(row.left() + 8, y, row.width() - 16, 1),
                                 qcolor(m.hairline))
            if hovered:
                painter.fillPath(path, qcolor(m.hover))
            if selected:
                painter.fillPath(path, qcolor(m.selection))
        finally:
            painter.restore()

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
            opt.palette.setColor(QtGui.QPalette.ColorRole.Text, colour)
            opt.palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, colour)
        if not enabled:
            opt.palette.setColor(QtGui.QPalette.ColorRole.Text, qcolor(t.text_muted))

        widget = opt.widget or self._view
        style = widget.style()
        # Drawn without the widget: the row look above replaces the style's
        # per-cell hover, selection and hairline (nothing else changes).
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
                pen = qcolor(t.accent)
            painter.setPen(pen)
            painter.setFont(opt.font)
            inner = chip.adjusted(_CHIP_PAD, 0, -_CHIP_PAD + 2, 0)
            painter.drawText(inner, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             fm.elidedText(text, Qt.TextElideMode.ElideRight, int(inner.width()) + 1))
            return

        label = str(index.data(PILL_ROLE) or text)
        tone = _tone_ok(t, str(index.data(TONE_ROLE) or ""))
        fill, border, ink = pill_colours(t, tone)
        pfont = font("caption", weight=600)
        pfm = QtGui.QFontMetrics(pfont)
        h = min(rect.height() - 6, pfm.height() + 6)
        w = min(pfm.horizontalAdvance(label) + 18, rect.width())
        pill = QtCore.QRectF(rect.left(), rect.center().y() - h / 2, w, h)
        painter.setPen(QtGui.QPen(QtGui.QColor(border), 1.0))
        painter.setBrush(QtGui.QColor(fill))
        if not enabled:
            painter.setOpacity(0.6)
        painter.drawRoundedRect(pill.adjusted(0.5, 0.5, -0.5, -0.5), h / 2, h / 2)
        painter.setPen(QtGui.QColor(ink))
        painter.setFont(pfont)
        painter.drawText(pill, int(Qt.AlignmentFlag.AlignCenter),
                         pfm.elidedText(label, Qt.TextElideMode.ElideRight, int(pill.width() - 10)))
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


class _GlassViewMixin:
    """Whole-row hover tracking and the shared view settings."""

    _hover = -1

    def _glass_setup(self, row_fill: bool, row_height: int) -> None:
        self._hover = -1
        self._row_h = row_height
        self.setProperty(GLASS_PROPERTY, "true")
        self.setItemDelegate(GlassRowDelegate(self, row_fill=row_fill, row_height=row_height))
        self.setAlternatingRowColors(False)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

    def hover_row(self) -> int:
        return self._hover

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

    def _set_hover(self, row: int) -> None:
        if row != self._hover:
            self._hover = row
            self.viewport().update()

    def viewportEvent(self, event: QtCore.QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QtCore.QEvent.Type.MouseMove:
            self._set_hover(self.indexAt(event.position().toPoint()).row())
        elif kind in (QtCore.QEvent.Type.Leave, QtCore.QEvent.Type.HoverLeave):
            self._set_hover(-1)
        return super().viewportEvent(event)


class GlassTree(_GlassViewMixin, QtWidgets.QTreeWidget):
    """A flat ``QTreeWidget`` drawn as glass rows."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, row_fill: bool = False,
                 row_height: int = ROW_HEIGHT) -> None:
        super().__init__(parent)
        self._glass_setup(row_fill, row_height)
        self.setRootIsDecorated(False)
        self.setIndentation(0)
        self.setUniformRowHeights(True)
        self.setAllColumnsShowFocus(True)
        self.header().setHighlightSections(False)
        self.header().setMinimumSectionSize(32)
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
                 *, row_fill: bool = False, row_height: int = ROW_HEIGHT) -> None:
        super().__init__(rows, columns, parent)
        self._glass_setup(row_fill, row_height)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setCornerButtonEnabled(False)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.verticalHeader().setDefaultSectionSize(row_height)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setMinimumSectionSize(32)
        self.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)


class GlassList(_GlassViewMixin, QtWidgets.QListWidget):
    """A ``QListWidget`` drawn as glass rows.

    ``max_rows`` > 0: the list is exactly as tall as its rows, up to that
    many (then it scrolls), so a short list never leaves an empty well."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, row_fill: bool = True,
                 row_height: int = ROW_HEIGHT, max_rows: int = 0) -> None:
        super().__init__(parent)
        self._glass_setup(row_fill, row_height)
        self.setUniformItemSizes(True)
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


# ── Views of one panel ───────────────────────────────────────────────────────

class ViewTabs(QtWidgets.QTabWidget):
    """A ``QTabWidget`` as tall as the view on show.

    ``QTabWidget.sizeHint`` and ``QStackedLayout.heightForWidth`` take the
    tallest page, so a short view inside a scroll area got the long view's
    height as empty space. This one asks only the current page."""

    def _chrome_height(self) -> int:
        bar = self.tabBar()
        if not bar.isVisibleTo(self):
            return 0
        frame = self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_DefaultFrameWidth, None, self)
        return bar.sizeHint().height() + 2 * frame

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        base = super().sizeHint()
        page = self.currentWidget()
        if page is None:
            return base
        return QtCore.QSize(base.width(), page.sizeHint().height() + self._chrome_height())

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        page = self.currentWidget()
        if page is None:
            return super().heightForWidth(width)
        h = page.heightForWidth(width) if page.hasHeightForWidth() else -1
        if h < 0:
            h = page.sizeHint().height()
        return h + self._chrome_height()


# ── Layout helpers ────────────────────────────────────────────────────────────

class FlowLayout(QtWidgets.QLayout):
    """Lays its items out left to right and wraps onto further lines.

    ``sizeHint`` is the one-line width, ``minimumSize`` the widest single
    item, so a card asks for one line but never forces a horizontal scroll."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, spacing: int = 8,
                 line_spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._h = spacing
        self._v = line_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def addStretch(self, _stretch: int = 0) -> None:  # noqa: N802
        """API parity with QBoxLayout; a flow has no stretch (no-op)."""

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:  # noqa: N802
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._layout(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def _visible(self) -> list[QtWidgets.QLayoutItem]:
        return [i for i in self._items if not i.isEmpty()]

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        items = self._visible()
        m = self.contentsMargins()
        if not items:
            return QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        w = sum(i.sizeHint().width() for i in items) + self._h * (len(items) - 1)
        h = max(i.sizeHint().height() for i in items)
        return QtCore.QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        size = QtCore.QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect: QtCore.QRect, *, apply: bool) -> int:
        """Place the items line by line, each vertically centred on its line;
        returns the height used."""
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        lines: list[list[tuple[QtWidgets.QLayoutItem, int, int]]] = [[]]
        x = area.x()
        for item in self._visible():
            hint = item.sizeHint()
            w = min(hint.width(), max(area.width(), item.minimumSize().width()))
            if lines[-1] and x + w > area.right() + 1:
                lines.append([])
                x = area.x()
            lines[-1].append((item, x, w))
            x += w + self._h
        y = area.y()
        for line in lines:
            if not line:
                continue
            line_h = max(item.sizeHint().height() for item, _x, _w in line)
            if apply:
                for item, ix, w in line:
                    h = item.sizeHint().height()
                    item.setGeometry(QtCore.QRect(ix, y + (line_h - h) // 2, w, h))
            y += line_h + self._v
        used = y - self._v if any(lines) else y
        return used - rect.y() + m.bottom()


def card_heading(title: str, subtitle: str = "", *, hero: bool = False) -> tuple[
        QtWidgets.QWidget, QtWidgets.QLabel, QtWidgets.QLabel]:
    """A card title over one muted line: ``(container, title, subtitle)``.

    The hero card's title uses the title role (20/650); other cards keep the
    ``cardTitle`` heading role (16/650)."""
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    title_lbl = QtWidgets.QLabel(title)
    if hero:
        title_lbl.setObjectName("metaHeroTitle")
        apply_font(title_lbl, "title")
    else:
        title_lbl.setObjectName("cardTitle")
    sub = QtWidgets.QLabel(subtitle)
    sub.setObjectName("mutedLabel")
    sub.setWordWrap(True)
    sub.setVisible(bool(subtitle))
    lay.addWidget(title_lbl)
    lay.addWidget(sub)
    box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
    return box, title_lbl, sub


def status_label(text: str = "") -> QtWidgets.QLabel:
    """A muted status line that wraps and never widens its card."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("statusHint")
    lbl.setWordWrap(True)
    lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    return lbl


def thin_progress() -> QtWidgets.QProgressBar:
    bar = QtWidgets.QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedHeight(6)
    return bar


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
