"""Small Liquid Glass parts used by the Organize pages and Exports.

The Indexer, Library Sync, File Cleanup and Exports pages share these; they
are kept here (not in ``gui/widgets``) because nothing else uses them yet.

``StepPills``
    A compact row of numbered pills for a workflow step that *really*
    advances (the page calls ``set_current``): the active step is tinted
    with the accent, finished steps show a check.
``ReflowBox`` / ``ReflowSplitter``
    Children side by side when there is room and stacked when there is not,
    so a page fits the 900 px minimum window without scrolling sideways.
    Neither ever asks for its side-by-side width as a minimum.
``FlowLayout``
    Buttons that wrap onto the next line instead of widening the card.
``GlassRowDelegate`` / ``style_item_view``
    Item views on the glass: 34 px rows, rounded hover and selection from
    the material, no grid, no opaque background.
``hero_title``, ``caption``, ``thin_bar``, ``glyph_button``, ``icon_button``
    One-line builders for the pages' recurring pieces.

Everything paints from the active theme at paint time (tokens, material,
type ramp), so a theme, accent or transparency change only needs a repaint.
"""
from __future__ import annotations

from typing import Sequence

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import apply_font, current_material, font, qcolor
from gui.themes.manager import get_manager
from gui.widgets.glyphs import paint_glyph, set_glyph

Qt = QtCore.Qt

ROW_HEIGHT = 34          # item-view rows (the mockup's list rows)
BAR_HEIGHT = 6           # thin progress pills
SPLITTER_HANDLE = 14     # room between two item views in a ReflowSplitter


# ── Builders ──────────────────────────────────────────────────────────────────

def hero_title(text: str) -> QtWidgets.QLabel:
    """The hero card's title: the type ramp's title role (20 / 650)."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("organizeHeroTitle")
    apply_font(lbl, "title")
    return lbl


def caption(text: str, *, role: str = "caption", tooltip: str = "") -> QtWidgets.QLabel:
    """A muted caption (12 px) over a list, a bar or a group of controls."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("mutedLabel")
    apply_font(lbl, role)
    if tooltip:
        lbl.setToolTip(tooltip)
    return lbl


def thin_bar() -> QtWidgets.QProgressBar:
    """A thin progress pill (the style paints groove and fill as pills)."""
    bar = QtWidgets.QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedHeight(BAR_HEIGHT)
    return bar


_ACCENT_NAMES = ("primaryBtn", "dangerBtn", "successBtn")


def glyph_button(text: str, glyph: str, *, tooltip: str = "", object_name: str = "",
                 flat: bool = False) -> QtWidgets.QPushButton:
    """A pill button with a painted line glyph (never an emoji) and a label."""
    btn = QtWidgets.QPushButton(text)
    if object_name:
        btn.setObjectName(object_name)
    set_glyph(btn, glyph, role="inverse" if object_name in _ACCENT_NAMES else "text")
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setAccessibleName(text or tooltip)
    btn.setFlat(flat)
    return btn


def icon_button(glyph: str, tooltip: str, *, size: int = 36) -> QtWidgets.QPushButton:
    """A round, icon-only glass button; the tooltip doubles as its accessible name."""
    btn = QtWidgets.QPushButton()
    set_glyph(btn, glyph)
    btn.setToolTip(tooltip)
    btn.setAccessibleName(tooltip)
    btn.setFixedSize(size, size)
    return btn


class GlyphLabel(QtWidgets.QWidget):
    """A decorative glyph in the muted text colour (e.g. before a path)."""

    def __init__(self, name: str, size: int = 16, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def name(self) -> str:
        return self._name

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            paint_glyph(p, QtCore.QRectF(self.rect()), self._name,
                        qcolor(get_manager().current.text_secondary))
        finally:
            p.end()


def path_well(path_label: QtWidgets.QLabel, glyph: str = "folder") -> QtWidgets.QWidget:
    """A "well" row holding a glyph and an (elided) path label."""
    from gui.widgets.glass import GlassPanel

    well = GlassPanel(surface="well", radius=14)
    well.setObjectName("organizePathWell")
    row = QtWidgets.QHBoxLayout(well)
    row.setContentsMargins(12, 8, 12, 8)
    row.setSpacing(10)
    row.addWidget(GlyphLabel(glyph), 0, Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(path_label, 1)
    return well


# ── Step pills ────────────────────────────────────────────────────────────────

class StepPills(QtWidgets.QWidget):
    """Numbered workflow steps as a compact pill row.

    It only shows state: the page decides the current step (from its plan,
    scan and worker state) and calls ``set_current``. No click handling, so
    it never offers navigation it cannot perform."""

    _H = 28          # pill height
    _DISC = 16       # number disc
    _PAD = 10        # inner horizontal padding
    _GAP = 4         # disc → label
    _SEP = 22        # space between pills (holds a chevron)

    def __init__(self, steps: Sequence[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("organizeStepPills")
        self._steps = list(steps)
        self._current = 0
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setToolTip("  →  ".join(f"{i}. {s}" for i, s in enumerate(self._steps, 1)))
        self._update_name()
        get_manager().theme_changed.connect(self._repaint_on_theme)

    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()

    def steps(self) -> list[str]:
        return list(self._steps)

    def current(self) -> int:
        return self._current

    def set_current(self, index: int) -> None:
        index = max(0, min(len(self._steps) - 1, int(index)))
        if index != self._current:
            self._current = index
            self._update_name()
            self.update()

    def _update_name(self) -> None:
        if self._steps:
            self.setAccessibleName(
                f"Step {self._current + 1} of {len(self._steps)}: {self._steps[self._current]}")

    # Geometry ---------------------------------------------------------------
    def _label_font(self) -> QtGui.QFont:
        return font("caption", weight=600)

    def _pill_widths(self) -> list[int]:
        fm = QtGui.QFontMetrics(self._label_font())
        return [self._PAD * 2 + self._DISC + self._GAP + fm.horizontalAdvance(s) for s in self._steps]

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        widths = self._pill_widths()
        return QtCore.QSize(sum(widths) + self._SEP * max(0, len(widths) - 1) + 2, self._H + 2)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.sizeHint()

    # Painting ---------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = get_manager().current
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            lf = self._label_font()
            nf = font("micro", weight=700, tracking=0.0)
            x = 1.0
            y = (self.height() - self._H) / 2.0
            for i, (label, w) in enumerate(zip(self._steps, self._pill_widths())):
                rect = QtCore.QRectF(x, y, w, self._H).adjusted(0.5, 0.5, -0.5, -0.5)
                r = rect.height() / 2
                active, done = i == self._current, i < self._current
                if active:
                    fill, border, text = qcolor(t.accent, 0.20), qcolor(t.accent, 0.60), qcolor(t.text_primary)
                elif done:
                    fill, border, text = qcolor(m.well.fill), qcolor(m.well.border), qcolor(t.text_secondary)
                else:
                    fill, border, text = QtGui.QColor(0, 0, 0, 0), qcolor(m.hairline), qcolor(t.text_muted)
                p.setPen(QtGui.QPen(border, 1.0))
                p.setBrush(fill)
                p.drawRoundedRect(rect, r, r)

                disc = QtCore.QRectF(rect.left() + self._PAD, rect.center().y() - self._DISC / 2,
                                     self._DISC, self._DISC)
                if active:
                    p.setPen(QtCore.Qt.PenStyle.NoPen)
                    p.setBrush(qcolor(t.accent))
                    p.drawEllipse(disc)
                    p.setPen(qcolor(t.text_inverse))
                    p.setFont(nf)
                    p.drawText(disc, int(Qt.AlignmentFlag.AlignCenter), str(i + 1))
                elif done:
                    paint_glyph(p, disc.adjusted(1, 1, -1, -1), "check", qcolor(t.success), weight=2.4)
                else:
                    p.setPen(QtGui.QPen(qcolor(t.text_muted), 1.0))
                    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    p.drawEllipse(disc.adjusted(0.5, 0.5, -0.5, -0.5))
                    p.setFont(nf)
                    p.drawText(disc, int(Qt.AlignmentFlag.AlignCenter), str(i + 1))

                p.setFont(lf)
                p.setPen(text)
                text_rect = QtCore.QRectF(disc.right() + self._GAP, rect.top(),
                                          rect.right() - disc.right() - self._GAP - self._PAD + 1,
                                          rect.height())
                p.drawText(text_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                           label)
                x += w
                if i < len(self._steps) - 1:
                    chev = QtCore.QRectF(x + (self._SEP - 10) / 2, rect.center().y() - 5, 10, 10)
                    paint_glyph(p, chev, "chevron_right", qcolor(t.text_muted), weight=2.0)
                    x += self._SEP
        finally:
            p.end()


# ── Reflowing containers ──────────────────────────────────────────────────────

class ReflowBox(QtWidgets.QWidget):
    """Children side by side when the box is wide enough, stacked when not.

    Its minimum width is the widest child's, never the side-by-side sum, so a
    narrow window stacks the children instead of widening the page. The
    switch happens at ``breakpoint`` or at the side-by-side minimum,
    whichever is larger."""

    def __init__(self, breakpoint: int = 0, spacing: int = 18,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._breakpoint = breakpoint
        self._lay = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.LeftToRight, self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(spacing)
        # The layout must not impose its (side-by-side) minimum on the widget.
        self._lay.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        self._items: list[QtWidgets.QWidget] = []

    def add_widget(self, widget: QtWidgets.QWidget, stretch: int = 1) -> None:
        self._items.append(widget)
        self._lay.addWidget(widget, stretch)
        self.updateGeometry()

    def is_horizontal(self) -> bool:
        return self._lay.direction() == QtWidgets.QBoxLayout.Direction.LeftToRight

    def _child_min(self, w: QtWidgets.QWidget) -> int:
        return max(w.minimumWidth(), w.minimumSizeHint().width())

    def side_by_side_width(self) -> int:
        shown = [w for w in self._items if not w.isHidden()]
        return sum(self._child_min(w) for w in shown) + self._lay.spacing() * max(0, len(shown) - 1)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        shown = [w for w in self._items if not w.isHidden()] or [QtWidgets.QWidget()]
        width = max(self._child_min(w) for w in shown)
        return QtCore.QSize(width, self._lay.minimumSize().height())

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        hint = self._lay.sizeHint()
        return QtCore.QSize(min(hint.width(), self.minimumSizeHint().width()), hint.height())

    def _apply(self, width: int) -> None:
        horizontal = width >= max(self._breakpoint, self.side_by_side_width())
        direction = (QtWidgets.QBoxLayout.Direction.LeftToRight if horizontal
                     else QtWidgets.QBoxLayout.Direction.TopToBottom)
        if self._lay.direction() != direction:
            self._lay.setDirection(direction)
            self.updateGeometry()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        self._apply(event.size().width())
        super().resizeEvent(event)


class _GripHandle(QtWidgets.QSplitterHandle):
    """A gap with a short rounded grip in its middle (instead of a bar the
    full height of the lists)."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            colour = qcolor(get_manager().current.text_muted, 0.35)
            if self.underMouse():
                colour = qcolor(get_manager().current.accent, 0.7)
            c = QtCore.QRectF(self.rect()).center()
            if self.orientation() == Qt.Orientation.Horizontal:
                grip = QtCore.QRectF(c.x() - 2, c.y() - 16, 4, 32)
            else:
                grip = QtCore.QRectF(c.x() - 16, c.y() - 2, 32, 4)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(colour)
            p.drawRoundedRect(grip, 2, 2)
        finally:
            p.end()

    def enterEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().leaveEvent(event)


class ReflowSplitter(QtWidgets.QSplitter):
    """A splitter that is horizontal when wide and vertical when narrow (the
    user can still drag the handle). Like ``ReflowBox`` it never asks for its
    side-by-side width as a minimum."""

    def __init__(self, breakpoint: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._breakpoint = breakpoint
        self.setChildrenCollapsible(False)
        self.setHandleWidth(SPLITTER_HANDLE)

    def createHandle(self) -> QtWidgets.QSplitterHandle:  # noqa: N802
        return _GripHandle(self.orientation(), self)

    def _child_mins(self) -> list[QtCore.QSize]:
        out = []
        for i in range(self.count()):
            w = self.widget(i)
            hint = w.minimumSizeHint()
            out.append(QtCore.QSize(max(w.minimumWidth(), hint.width()),
                                    max(w.minimumHeight(), hint.height())))
        return out or [QtCore.QSize(0, 0)]

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        mins = self._child_mins()
        handles = self.handleWidth() * max(0, len(mins) - 1)
        if self.orientation() == Qt.Orientation.Horizontal:
            return QtCore.QSize(max(m.width() for m in mins), max(m.height() for m in mins))
        return QtCore.QSize(max(m.width() for m in mins), sum(m.height() for m in mins) + handles)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        hint = super().sizeHint()
        return QtCore.QSize(self.minimumSizeHint().width(), max(hint.height(), self.minimumSizeHint().height()))

    def _apply(self, width: int) -> None:
        mins = self._child_mins()
        side_by_side = sum(m.width() for m in mins) + self.handleWidth() * max(0, len(mins) - 1)
        wanted = (Qt.Orientation.Horizontal if width >= max(self._breakpoint, side_by_side)
                  else Qt.Orientation.Vertical)
        if self.orientation() != wanted:
            self.setOrientation(wanted)
            self.updateGeometry()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        self._apply(event.size().width())
        super().resizeEvent(event)


# ── Flow layout ───────────────────────────────────────────────────────────────

class FlowLayout(QtWidgets.QLayout):
    """Lays items out left to right and wraps to a new line when the row is
    full (the Qt "flow layout" example, trimmed). Its minimum width is the
    widest item, so a row of buttons never widens a card."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, hspacing: int = 10,
                 vspacing: int = 10) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._h = hspacing
        self._v = vspacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:  # noqa: N802
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        w = h = 0
        visible = [i for i in self._items if not i.isEmpty()]
        for item in visible:
            hint = item.sizeHint()
            w += hint.width()
            h = max(h, hint.height())
        w += self._h * max(0, len(visible) - 1)
        m = self.contentsMargins()
        return QtCore.QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QtCore.QRect, *, test_only: bool) -> int:
        """Break the items into lines; each item is centred vertically in
        its line (a checkbox next to taller buttons lines up with them)."""
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        lines: list[list[tuple[QtWidgets.QLayoutItem, QtCore.QSize]]] = [[]]
        x = area.x()
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and lines[-1]:
                lines.append([])
                x = area.x()
            lines[-1].append((item, hint))
            x += hint.width() + self._h
        y = area.y()
        for line in lines:
            if not line:
                continue
            line_h = max(h.height() for _, h in line)
            if not test_only:
                x = area.x()
                for item, hint in line:
                    top = y + (line_h - hint.height()) // 2
                    item.setGeometry(QtCore.QRect(QtCore.QPoint(x, top), hint))
                    x += hint.width() + self._h
            y += line_h + self._v
        used = y - self._v if any(lines) else y
        return used - rect.y() + m.bottom()


# ── Item views on the glass ───────────────────────────────────────────────────

class GlassRowDelegate(QtWidgets.QStyledItemDelegate):
    """34 px rows; hover and selection are one rounded band per row, painted
    from the material (the first and last column round the ends)."""

    RADIUS = 10

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem,  # noqa: N802
                 index: QtCore.QModelIndex) -> QtCore.QSize:
        hint = super().sizeHint(option, index)
        return QtCore.QSize(hint.width(), max(hint.height(), ROW_HEIGHT))

    def _ends(self, index: QtCore.QModelIndex) -> tuple[bool, bool]:
        view = self.parent()
        header = None
        if isinstance(view, QtWidgets.QTreeView):
            header = view.header()
        elif isinstance(view, QtWidgets.QTableView):
            header = view.horizontalHeader()
        if header is None or header.count() == 0:
            return True, True
        visual = [header.logicalIndex(v) for v in range(header.count())
                  if not header.isSectionHidden(header.logicalIndex(v))]
        if not visual:
            return True, True
        return index.column() == visual[0], index.column() == visual[-1]

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem,
              index: QtCore.QModelIndex) -> None:
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        state = QtWidgets.QStyle.StateFlag
        selected = bool(opt.state & state.State_Selected)
        hovered = bool(opt.state & state.State_MouseOver)
        m = current_material()
        if selected or hovered:
            first, last = self._ends(index)
            r = self.RADIUS
            band = QtCore.QRectF(opt.rect).adjusted(0, 2, 0, -2)
            band = band.adjusted(0 if first else -r - 1, 0, 0 if last else r + 1, 0)
            painter.save()
            painter.setClipRect(opt.rect)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(m.selection if selected else m.hover))
            painter.drawRoundedRect(band, r, r)
            painter.restore()
        if selected:
            colour = qcolor(m.selection_text)
            for role in (QtGui.QPalette.ColorRole.Text, QtGui.QPalette.ColorRole.HighlightedText,
                         QtGui.QPalette.ColorRole.WindowText):
                opt.palette.setColor(role, colour)
        opt.state &= ~(state.State_Selected | state.State_MouseOver | state.State_HasFocus)
        view = self.parent() if isinstance(self.parent(), QtWidgets.QWidget) else None
        style = view.style() if view is not None else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, view)


def style_item_view(view: QtWidgets.QAbstractItemView, *,
                    elide: QtCore.Qt.TextElideMode = Qt.TextElideMode.ElideRight
                    ) -> QtWidgets.QAbstractItemView:
    """Put ``view`` on the glass: transparent (the static stylesheet paints
    item views from palette roles), 34 px rows with a rounded hover and
    selection band, no grid, no frame, muted 12 px headers."""
    view.setItemDelegate(GlassRowDelegate(view))
    view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    view.setAlternatingRowColors(False)
    view.setMouseTracking(True)
    view.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
    view.setTextElideMode(elide)
    if isinstance(view, QtWidgets.QTreeView):
        view.setUniformRowHeights(True)
        view.setAllColumnsShowFocus(True)
    if isinstance(view, QtWidgets.QTableView):
        view.setShowGrid(False)
        view.setWordWrap(False)
        view.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
    return view
