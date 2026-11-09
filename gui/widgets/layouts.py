"""Layouts and containers that reflow instead of scrolling sideways.

Every page must fit the minimum window (900 × 600) without a horizontal
scroll bar. These never ask for their one-line or side-by-side width as a
minimum; they wrap or stack instead.

``FlowLayout``
    Items left to right, wrapping onto further lines (button rows, toggle
    rows, chips). ``sizeHint`` is the one-line width, ``minimumSize`` the
    widest single item; an item wider than the row is shrunk to it (an
    elided label), never past its own minimum.
``ReflowBox`` / ``ReflowSplitter``
    Children side by side when there is room and stacked when there is not
    (the splitter keeps its draggable handle, drawn as a short grip).
``ViewTabs``
    A ``QTabWidget`` as tall as the view on show, not its tallest view.
``fit_stack_to_current``
    The same for a ``QStackedWidget``.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import qcolor
from gui.themes.manager import get_manager

Qt = QtCore.Qt

SPLITTER_HANDLE = 14     # room between two item views in a ReflowSplitter


# ── Flow layout ───────────────────────────────────────────────────────────────

class FlowLayout(QtWidgets.QLayout):
    """Lays its items out left to right and wraps onto further lines.

    ``spacing`` is the gap between items on a line, ``line_spacing`` the gap
    between lines. Each item is centred vertically on its line (a check box
    next to taller buttons lines up with them)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, spacing: int = 10,
                 line_spacing: int = 10) -> None:
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

    def horizontal_spacing(self) -> int:
        return self._h

    def line_spacing(self) -> int:
        return self._v

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
        """Place the items line by line; returns the height used."""
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


def fit_stack_to_current(stack: QtWidgets.QStackedWidget) -> QtWidgets.QStackedWidget:
    """Make ``stack`` as tall as its *current* page, not its tallest one (an
    empty state would otherwise sit in the results page's height).
    ``QStackedLayout`` leaves pages with an Ignored size policy out of its
    size hints, so every other page is set to Ignored on each switch."""
    def _fit(index: int) -> None:
        for i in range(stack.count()):
            page = stack.widget(i)
            policy = (QtWidgets.QSizePolicy.Policy.Preferred if i == index
                      else QtWidgets.QSizePolicy.Policy.Ignored)
            page.setSizePolicy(policy, policy)
        stack.updateGeometry()

    stack.currentChanged.connect(_fit)
    _fit(stack.currentIndex())
    return stack
