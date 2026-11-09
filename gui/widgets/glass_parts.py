"""Small Liquid Glass building blocks shared by every page.

Use these instead of building a private copy in a workspace:

``StatTile``
    A number and a label on a "well" surface (radius 16, faint fill): the
    mockup's Library health tiles. ``set_value`` / ``set_label`` update it in
    place; ``set_tone`` tints the value ("" · "success" · "warning" ·
    "danger" · "accent").

``StatRow``
    A row of ``StatTile``s that share the width evenly, built from
    ``(key, label)`` pairs; ``tile(key)`` returns one to update.

``TileGrid``
    ``StatTile``s that reflow into balanced rows when the card is too narrow
    for one row (5 tiles become 3 + 2, never 4 + 1).

``EmptyState``
    What a list or results area shows before there is anything in it: a
    heading, one muted sentence and optionally one action button (it is only
    shown when a slot is connected, so it never offers something that does
    nothing).

``StepPills``
    A workflow's steps as a compact pill row that shows *real* state (the
    owner calls ``set_current``): the current step is tinted with the accent,
    finished steps show a check. When the row does not fit, the other steps
    shrink to their number (the tooltip names them all).

Builders for the recurring pieces of a card:

* ``hero_title`` (the ``heroTitle`` role, 20 / 650) and ``card_heading`` (a
  title over one muted line);
* ``caption`` (12 px muted) and ``status_label`` (a wrapping muted status line);
* ``thin_bar`` (a 6 px progress pill);
* ``glyph_button`` / ``icon_button`` (painted line glyphs, never emoji);
* ``path_well`` (a glyph and an elided path on a well);
* ``section_card`` and ``fields_row`` (dialog sections and compact fields).

All of them paint from the active material and the type ramp at paint time,
so a theme, accent or transparency change only needs a repaint.
"""
from __future__ import annotations

import math
from typing import Sequence

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.glass import apply_font, current_material, font, paint_surface, qcolor

_TONES = ("", "success", "warning", "danger", "accent")


def _tone_colour(tone: str) -> QtGui.QColor:
    from gui.themes.manager import get_manager

    t = get_manager().current
    value = {
        "success": t.success,
        "warning": t.warning,
        "danger": t.danger,
        "accent": t.accent,
    }.get(tone, t.text_primary)
    return qcolor(value)


class StatTile(QtWidgets.QWidget):
    """A value over a label on a well surface (24/700 tabular, 12 muted)."""

    def __init__(self, value: str = "–", label: str = "", *, tone: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = value
        self._label = label
        self._tone = tone if tone in _TONES else ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setAccessibleName(f"{label}: {value}")

    # ── Public API ────────────────────────────────────────────────────────
    def value(self) -> str:
        return self._value

    def label(self) -> str:
        return self._label

    def tone(self) -> str:
        return self._tone

    def set_value(self, value: object, tone: str | None = None) -> None:
        self._value = str(value)
        if tone is not None:
            self._tone = tone if tone in _TONES else ""
        self.setAccessibleName(f"{self._label}: {self._value}")
        self.update()

    def set_label(self, label: str) -> None:
        self._label = label
        self.setAccessibleName(f"{self._label}: {self._value}")
        self.updateGeometry()
        self.update()

    def set_tone(self, tone: str) -> None:
        self._tone = tone if tone in _TONES else ""
        self.update()

    # ── Qt ────────────────────────────────────────────────────────────────
    _PAD_X = 14
    _PAD_Y = 12
    _GAP = 2

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        vf = QtGui.QFontMetrics(font("stat"))
        lf = QtGui.QFontMetrics(font("caption"))
        w = max(vf.horizontalAdvance(self._value), lf.horizontalAdvance(self._label))
        h = vf.height() + self._GAP + lf.height()
        return QtCore.QSize(w + 2 * self._PAD_X, h + 2 * self._PAD_Y)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        hint = self.sizeHint()
        return QtCore.QSize(min(hint.width(), 96), hint.height())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager

        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().well)
            inner = self.rect().adjusted(self._PAD_X, self._PAD_Y, -self._PAD_X, -self._PAD_Y)
            vfont, lfont = font("stat"), font("caption")
            vh = QtGui.QFontMetrics(vfont).height()
            p.setFont(vfont)
            p.setPen(_tone_colour(self._tone))
            value_rect = QtCore.QRect(inner.left(), inner.top(), inner.width(), vh)
            p.drawText(value_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                       QtGui.QFontMetrics(vfont).elidedText(
                           self._value, QtCore.Qt.TextElideMode.ElideRight, inner.width()))
            p.setFont(lfont)
            p.setPen(qcolor(get_manager().current.text_secondary))
            label_rect = QtCore.QRect(inner.left(), inner.top() + vh + self._GAP,
                                      inner.width(), inner.bottom() - inner.top() - vh - self._GAP + 1)
            p.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
                       QtGui.QFontMetrics(lfont).elidedText(
                           self._label, QtCore.Qt.TextElideMode.ElideRight, inner.width()))
        finally:
            p.end()


class StatRow(QtWidgets.QWidget):
    """``StatTile``s side by side, sharing the width evenly."""

    def __init__(self, tiles: list[tuple[str, str]], *, spacing: int = 10,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(spacing)
        self._tiles: dict[str, StatTile] = {}
        for key, label in tiles:
            tile = StatTile("–", label, parent=self)
            tile.setObjectName(f"stat_{key}")
            self._tiles[key] = tile
            lay.addWidget(tile, 1)

    def tile(self, key: str) -> StatTile:
        return self._tiles[key]

    def keys(self) -> list[str]:
        return list(self._tiles)


class EmptyState(QtWidgets.QWidget):
    """A heading, one muted sentence, and an optional action button.

    The button stays hidden until ``action_requested`` has a receiver (or
    ``set_action`` is given a callable), so an empty state never offers an
    action that does nothing."""

    action_requested = Signal()

    def __init__(self, title: str, message: str = "", action_text: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 20, 8, 20)
        lay.setSpacing(6)
        lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self._title = QtWidgets.QLabel(title)
        self._title.setObjectName("cardTitle")
        self._title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._title.setFont(font("heading"))
        self._message = QtWidgets.QLabel(message)
        self._message.setObjectName("mutedLabel")
        self._message.setWordWrap(True)
        self._message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._message.setVisible(bool(message))
        self._button = QtWidgets.QPushButton(action_text)
        self._button.setVisible(False)
        self._button.clicked.connect(self.action_requested.emit)
        self._has_action = False

        lay.addWidget(self._title)
        lay.addWidget(self._message)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._button)
        row.addStretch(1)
        lay.addSpacing(6)
        lay.addLayout(row)

    def set_text(self, title: str, message: str = "") -> None:
        self._title.setText(title)
        self._message.setText(message)
        self._message.setVisible(bool(message))

    def set_action(self, text: str, callback=None) -> None:
        """Show the button with ``text``; ``callback`` (optional) is connected
        to ``action_requested``. Pass ``text=""`` to hide it again."""
        if callback is not None:
            self.action_requested.connect(callback)
            self._has_action = True
        self._button.setText(text)
        self._button.setVisible(bool(text) and self._has_action)

    def connectNotify(self, signal: QtCore.QMetaMethod) -> None:  # noqa: N802
        if signal.name() == b"action_requested":
            self._has_action = True
            self._button.setVisible(bool(self._button.text()))
        super().connectNotify(signal)

    def title(self) -> str:
        return self._title.text()

    def message(self) -> str:
        return self._message.text()

    def action_button(self) -> QtWidgets.QPushButton:
        return self._button


class TileGrid(QtWidgets.QWidget):
    """``StatTile``s in balanced rows: as many columns as fit at
    ``min_tile_width``, then the same number of rows with the tiles spread
    evenly (5 → 3 + 2). ``tile(key)`` returns one to update."""

    def __init__(self, tiles: list[tuple[str, str]], *, min_tile_width: int = 128,
                 spacing: int = 10, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._min_tile = min_tile_width
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(spacing)
        self._tiles: dict[str, StatTile] = {}
        for key, label in tiles:
            tile = StatTile("–", label, parent=self)
            tile.setObjectName(f"stat_{key}")
            tile.setToolTip(label)
            self._tiles[key] = tile
        self._columns = 0
        self._place(len(self._tiles))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def tile(self, key: str) -> StatTile:
        return self._tiles[key]

    def keys(self) -> list[str]:
        return list(self._tiles)

    def columns(self) -> int:
        return self._columns

    def columns_for(self, width: int) -> int:
        n = len(self._tiles)
        if n == 0:
            return 1
        spacing = self._grid.spacing()
        fit = max(1, min(n, (width + spacing) // (self._min_tile + spacing)))
        rows = math.ceil(n / fit)
        return math.ceil(n / rows)

    def _place(self, columns: int) -> None:
        if columns == self._columns:
            return
        for tile in self._tiles.values():
            self._grid.removeWidget(tile)
        for c in range(max(columns, self._columns)):
            self._grid.setColumnStretch(c, 1 if c < columns else 0)
        for i, tile in enumerate(self._tiles.values()):
            self._grid.addWidget(tile, i // columns, i % columns)
        self._columns = columns
        self.updateGeometry()

    def _height(self) -> int:
        """Height of the current arrangement. (Not height-for-width: a widget
        with a layout reports its layout's height-for-width, which a grid of
        fixed tiles does not have, so the arrangement is re-made on resize
        and ``updateGeometry`` asks the parent for the new height.)"""
        if not self._tiles:
            return 0
        rows = math.ceil(len(self._tiles) / max(1, self._columns))
        tile_h = next(iter(self._tiles.values())).sizeHint().height()
        return rows * tile_h + (rows - 1) * self._grid.spacing()

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(min(self._min_tile, 96), self._height())

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        n = max(1, len(self._tiles))
        return QtCore.QSize(n * self._min_tile + (n - 1) * self._grid.spacing(), self._height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place(self.columns_for(event.size().width()))


# ── Step pills ────────────────────────────────────────────────────────────────

class StepPills(QtWidgets.QWidget):
    """Numbered workflow steps as a compact pill row.

    It only shows state: the owner decides the current step (from its plan,
    scan and worker state, or the page a wizard shows) and calls
    ``set_current``. No click handling, so it never offers navigation it
    cannot perform. When the full row does not fit, the steps other than
    the current one shrink to their number disc; the tooltip lists them all."""

    _H = 28          # pill height
    _DISC = 16       # number disc
    _PAD = 10        # inner horizontal padding
    _GAP = 4         # disc → label
    _SEP = 22        # space between pills (holds a chevron)

    def __init__(self, steps: Sequence[str], parent: QtWidgets.QWidget | None = None) -> None:
        from gui.themes.manager import get_manager

        super().__init__(parent)
        self.setObjectName("stepPills")
        self._steps = list(steps)
        self._current = 0
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
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
            self.updateGeometry()
            self.update()

    def _update_name(self) -> None:
        if self._steps:
            self.setAccessibleName(
                f"Step {self._current + 1} of {len(self._steps)}: {self._steps[self._current]}")

    # Geometry ---------------------------------------------------------------
    @staticmethod
    def _label_font() -> QtGui.QFont:
        return font("caption", weight=600)

    def _full_widths(self) -> list[int]:
        fm = QtGui.QFontMetrics(self._label_font())
        return [self._PAD * 2 + self._DISC + self._GAP + fm.horizontalAdvance(s) for s in self._steps]

    def _total(self, widths: list[int]) -> int:
        return sum(widths) + self._SEP * max(0, len(widths) - 1) + 2

    def _compact_widths(self) -> list[int]:
        full = self._full_widths()
        return [full[i] if i == self._current else self._H for i in range(len(full))]

    def is_compact(self) -> bool:
        """True while the steps other than the current one show only their number."""
        return 0 < self.width() < self._total(self._full_widths())

    def _widths(self) -> list[int]:
        if not self.is_compact():
            return self._full_widths()
        widths = self._compact_widths()
        spare = self.width() - self._total(widths)
        if spare < 0 and self._steps:
            # Still too wide: the current label elides.
            widths[self._current] = max(self._H, widths[self._current] + spare)
        return widths

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(self._total(self._full_widths()), self._H + 2)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(min(self._total(self._compact_widths()), self.sizeHint().width()),
                            self._H + 2)

    # Painting ---------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager
        from gui.widgets.glyphs import paint_glyph

        t = get_manager().current
        m = current_material()
        align_left = int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        centre = int(QtCore.Qt.AlignmentFlag.AlignCenter)
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            lf = self._label_font()
            fm = QtGui.QFontMetrics(lf)
            nf = font("micro", weight=700, tracking=0.0)
            x = 1.0
            y = (self.height() - self._H) / 2.0
            for i, (label, w) in enumerate(zip(self._steps, self._widths())):
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

                labelled = w > self._H
                disc_x = rect.left() + self._PAD if labelled else rect.center().x() - self._DISC / 2
                disc = QtCore.QRectF(disc_x, rect.center().y() - self._DISC / 2, self._DISC, self._DISC)
                if active:
                    p.setPen(QtCore.Qt.PenStyle.NoPen)
                    p.setBrush(qcolor(t.accent))
                    p.drawEllipse(disc)
                    p.setPen(qcolor(t.text_inverse))
                    p.setFont(nf)
                    p.drawText(disc, centre, str(i + 1))
                elif done:
                    paint_glyph(p, disc.adjusted(1, 1, -1, -1), "check", qcolor(t.success), weight=2.4)
                else:
                    p.setPen(QtGui.QPen(qcolor(t.text_muted), 1.0))
                    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    p.drawEllipse(disc.adjusted(0.5, 0.5, -0.5, -0.5))
                    p.setFont(nf)
                    p.drawText(disc, centre, str(i + 1))

                if labelled:
                    p.setFont(lf)
                    p.setPen(text)
                    text_rect = QtCore.QRectF(disc.right() + self._GAP, rect.top(),
                                              rect.right() - disc.right() - self._GAP - self._PAD + 1,
                                              rect.height())
                    p.drawText(text_rect, align_left,
                               fm.elidedText(label, QtCore.Qt.TextElideMode.ElideRight,
                                             int(text_rect.width()) + 1))
                x += w
                if i < len(self._steps) - 1:
                    chev = QtCore.QRectF(x + (self._SEP - 10) / 2, rect.center().y() - 5, 10, 10)
                    paint_glyph(p, chev, "chevron_right", qcolor(t.text_muted), weight=2.0)
                    x += self._SEP
        finally:
            p.end()


# ── Builders ──────────────────────────────────────────────────────────────────

def hero_title(text: str) -> QtWidgets.QLabel:
    """A hero card's title: the ``heroTitle`` role (the title role, 20 / 650).

    The role's stylesheet rule carries the 20 px size, so the ``cardTitle``
    rule's 16 px can never override it (the reason for the old per-area
    roles)."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("heroTitle")
    apply_font(lbl, "title")
    return lbl


def card_heading(title: str, subtitle: str = "", *, hero: bool = False) -> tuple[
        QtWidgets.QWidget, QtWidgets.QLabel, QtWidgets.QLabel]:
    """A card title over one muted line: ``(container, title, subtitle)``.

    The hero card's title is ``hero_title``; other cards keep the
    ``cardTitle`` heading role (16 / 650)."""
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    if hero:
        title_lbl = hero_title(title)
    else:
        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("cardTitle")
    sub = QtWidgets.QLabel(subtitle)
    sub.setObjectName("mutedLabel")
    sub.setWordWrap(True)
    sub.setVisible(bool(subtitle))
    lay.addWidget(title_lbl)
    lay.addWidget(sub)
    box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
    return box, title_lbl, sub


def caption(text: str, *, role: str = "caption", tooltip: str = "") -> QtWidgets.QLabel:
    """A muted caption (12 px) over a list, a bar or a group of controls."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("mutedLabel")
    apply_font(lbl, role)
    if tooltip:
        lbl.setToolTip(tooltip)
    return lbl


def status_label(text: str = "") -> QtWidgets.QLabel:
    """A muted status line that wraps and never widens its card."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("statusHint")
    lbl.setWordWrap(True)
    lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    return lbl


BAR_HEIGHT = 6


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
    from gui.widgets.glyphs import set_glyph

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
    from gui.widgets.glyphs import set_glyph

    btn = QtWidgets.QPushButton()
    set_glyph(btn, glyph)
    btn.setToolTip(tooltip)
    btn.setAccessibleName(tooltip)
    btn.setFixedSize(size, size)
    return btn


def path_well(path_label: QtWidgets.QLabel, glyph: str = "folder", *,
              object_name: str = "pathWell") -> QtWidgets.QWidget:
    """A "well" row holding a glyph and a path label (``path_label`` in
    ``gui.widgets.glass``: middle-elided, the full path in its tooltip)."""
    from gui.themes.tokens import RADIUS
    from gui.widgets.glass import GlassPanel
    from gui.widgets.glyphs import GlyphLabel

    well = GlassPanel(surface="well", radius=RADIUS.sm)
    well.setObjectName(object_name)
    row = QtWidgets.QHBoxLayout(well)
    row.setContentsMargins(12, 8, 12, 8)
    row.setSpacing(10)
    row.addWidget(GlyphLabel(glyph), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(path_label, 1)
    return well


def section_card(title: str, *, badge: tuple[str, str] | None = None) -> QtWidgets.QFrame:
    """A glass card for a dialog section: a heading row (``layout().itemAt(0)``,
    an ``QHBoxLayout`` callers may append to) and 16 px margins.

    Dialogs have no scroll surface to paint card shadows, so these cards are
    fill, border and highlight only."""
    from gui.themes.tokens import SPACE
    from gui.widgets.glass import GlassCard
    from gui.widgets.pill import ToneBadge

    card = GlassCard()
    lay = QtWidgets.QVBoxLayout(card)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(SPACE.sm)
    head = QtWidgets.QHBoxLayout()
    head.setSpacing(SPACE.sm)
    lbl = QtWidgets.QLabel(title)
    lbl.setObjectName("cardTitle")
    head.addWidget(lbl)
    if badge:
        head.addWidget(ToneBadge(badge[0], badge[1]), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
    head.addStretch(1)
    lay.addLayout(head)
    return card


def fields_row(fields: list[tuple[str, QtWidgets.QWidget]], *,
               spacing: int = 14) -> QtWidgets.QHBoxLayout:
    """Compact fields side by side, each a muted caption above its editor."""
    from gui.themes.tokens import SPACE

    row = QtWidgets.QHBoxLayout()
    row.setSpacing(spacing)
    for text, editor in fields:
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(SPACE.xxs)
        lbl = caption(text)
        lbl.setBuddy(editor)
        if editor.toolTip():
            lbl.setToolTip(editor.toolTip())
        if not editor.accessibleName():
            editor.setAccessibleName(text)
        col.addWidget(lbl)
        col.addWidget(editor)
        row.addLayout(col, 1)
    return row
