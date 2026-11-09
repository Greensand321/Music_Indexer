"""Glass widgets: the card every page uses, and a generic panel.

``GlassCard`` is what ``WorkspaceBase._make_card()`` returns. It paints its
own fill, border and top highlight with ``gui.themes.glass.paint_surface``
(its shadow is painted by the scroll surface underneath, which can draw
outside the card). A card renamed ``headerCard`` is a *hero* card (radius
30); everything else uses the material's card radius (26).

``GlassPanel`` is the same thing for any elevation ("sidebar", "raised",
"capsule", "well", …), with no shadow of its own.

``ShadowLayer`` paints a floating sibling's shadow above the content under
it. ``ElidedLabel`` is the one eliding label (titles, statuses and, with
``path_label()``, file paths); ``gui.widgets.path_field.PathField`` is the
one path *field* (a path you choose).

All of them read the active theme at paint time, so a theme switch (or an accent
or transparency change) only needs a repaint, which the theme manager's
re-polish pass already does.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import current_material, paint_surface
from gui.themes.tokens import RADIUS, Surface

# Extra padding a card adds around its layout. Pages lay cards out with
# 16 px layout margins; with this, content sits 24 / 22 px from the edge
# (the mockup's card padding) without touching every page.
CARD_PADDING = (8, 6, 8, 6)


def card_radius(widget: QtWidgets.QWidget) -> int:
    return RADIUS.hero if widget.objectName() == "headerCard" else current_material().card.radius


class GlassCard(QtWidgets.QFrame):
    """A translucent card (opaque under classic themes / reduced transparency)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceCard")
        # The static stylesheet leaves frames marked glass="true" alone.
        self.setProperty("glass", "true")
        self.setContentsMargins(*CARD_PADDING)

    def radius(self) -> int:
        return card_radius(self)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().card,
                          radius=self.radius())
        finally:
            p.end()


class GlassPanel(QtWidgets.QWidget):
    """A plain container painting one surface of the material."""

    def __init__(self, surface: str = "card", radius: int | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._surface = surface
        self._radius = radius

    def surface_name(self) -> str:
        return self._surface

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().surface(self._surface),
                          radius=self._radius)
        finally:
            p.end()


class ShadowLayer(QtWidgets.QWidget):
    """Paints the soft shadow of a floating sibling (the Now Playing capsule,
    the landing card) *above* the content beneath it, as CSS would.

    A widget cannot paint outside its own rect, and a large transparent
    margin around the target would swallow clicks meant for the page below.
    So the shadow is a separate sibling that is transparent for the mouse,
    sits directly under its target in z-order, follows its geometry and
    visibility, and fades with it (``set_strength``)."""

    def __init__(self, target: QtWidgets.QWidget, surface: "str | Surface" = "capsule",
                 radius: float | None = None) -> None:
        super().__init__(target.parentWidget())
        self._target = target
        self._surface = surface
        self._radius = radius
        self._strength = 1.0
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        target.installEventFilter(self)
        self.sync()

    def strength(self) -> float:
        return self._strength

    def set_strength(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        if value != self._strength:
            self._strength = value
            self.update()

    def _spec(self):
        if isinstance(self._surface, Surface):
            return self._surface
        return current_material().surface(self._surface)

    def sync(self) -> None:
        """Match the target's geometry, visibility and z-order."""
        from gui.themes.glass import shadow_margins

        target = self._target
        geo = target.geometry().adjusted(*shadow_margins(self._spec()))
        if self.geometry() != geo:
            self.setGeometry(geo)
        visible = target.isVisibleTo(target.parentWidget()) if target.parentWidget() else False
        self.setVisible(visible)
        if visible:
            self.stackUnder(target)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self._target and event.type() in (
            QtCore.QEvent.Type.Move, QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Hide,
        ):
            self.sync()
        return False

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_shadow

        target = self._target.geometry().translated(-self.x(), -self.y())
        p = QtGui.QPainter(self)
        try:
            paint_shadow(p, QtCore.QRectF(target), self._spec(),
                         radius=self._radius, strength=self._strength)
        finally:
            p.end()


class ElidedLabel(QtWidgets.QLabel):
    """A label that shrinks with "…" instead of forcing its width.

    The one label for titles, statuses and paths:

    * ``text()`` stays the full text (tooltips, tests, accessibility); only
      the painting elides;
    * ``placeholder``: muted text (the palette's placeholder colour) painted
      while the label is empty;
    * ``path=True`` (or ``path_label()``): a file path — middle elide, the
      full path in the tooltip and accessible name, and it takes the spare
      width of its row.

    Colours come from the palette (the static stylesheet's object-name
    roles), like any label."""

    def __init__(self, text: str = "", mode: QtCore.Qt.TextElideMode = QtCore.Qt.TextElideMode.ElideRight,
                 parent: QtWidgets.QWidget | None = None, *, placeholder: str = "",
                 path: bool = False) -> None:
        super().__init__(text, parent)
        self._mode = QtCore.Qt.TextElideMode.ElideMiddle if path else mode
        self._placeholder = placeholder
        self._path = path
        self._explicit_name = False
        horizontal = (QtWidgets.QSizePolicy.Policy.Expanding if path
                      else QtWidgets.QSizePolicy.Policy.Preferred)
        self.setSizePolicy(horizontal, QtWidgets.QSizePolicy.Policy.Preferred)
        if path:
            self._sync_path()

    def placeholder(self) -> str:
        return self._placeholder

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        if self._path:
            self._sync_path()
        self.update()

    def elide_mode(self) -> QtCore.Qt.TextElideMode:
        return self._mode

    def _sync_path(self) -> None:
        self.setToolTip(self.text())
        if not self._explicit_name:
            super().setAccessibleName(self.text() or self._placeholder)

    def setAccessibleName(self, name: str) -> None:  # noqa: N802
        self._explicit_name = bool(name)
        super().setAccessibleName(name)

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        if self._path:
            self._sync_path()

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        return QtCore.QSize(min(hint.width(), 48), hint.height())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        text = self.text()
        if not text and not self._placeholder:
            return
        p = QtGui.QPainter(self)
        try:
            r = self.contentsRect()
            fm = QtGui.QFontMetrics(self.font())
            p.setFont(self.font())
            if text:
                p.setPen(self.palette().color(self.foregroundRole()))
                p.drawText(r, int(self.alignment()), fm.elidedText(text, self._mode, r.width()))
            else:
                p.setPen(self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText))
                p.drawText(r, int(self.alignment()),
                           fm.elidedText(self._placeholder, QtCore.Qt.TextElideMode.ElideRight, r.width()))
        finally:
            p.end()


def path_label(text: str = "", placeholder: str = "",
               parent: QtWidgets.QWidget | None = None) -> ElidedLabel:
    """A file path: middle-elided, the full path in the tooltip, a muted
    placeholder while it is empty."""
    return ElidedLabel(text, parent=parent, placeholder=placeholder, path=True)
