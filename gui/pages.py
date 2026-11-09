"""Pages: what one sidebar entry shows.

A page composes one or more *panels* (the workspace classes in
``gui.main_window._WORKSPACE_MAP``) as *sections*, switched with a row of
section tabs; a page with a single section just shows its panel. The page
structure itself is declared in :mod:`gui.navigation`.

A page never owns its panels' logic: panels keep their signals (the main
window connects them, as before) and their own jobs. The page only decides
which panel is visible, tells a panel that serves several sections which
view to show (``panel.set_section(view)``), and forwards the
``on_activated()`` hook to the panel that becomes visible.

Page-level actions (buttons that belong to the whole page, not one section)
go to the right of the section tabs: ``add_action(widget)``. The Sound Map
uses that for "Save cluster playlists…" and its quality report.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.navigation import Page, Section
from gui.themes.manager import get_manager
from gui.widgets.pill import ExperimentalPill


class _SectionBar(QtWidgets.QWidget):
    """The strip above a page's sections: the segmented control on the left,
    page actions on the right. Transparent: it sits on the backdrop.

    When the segmented control's natural width does not fit next to the
    actions, it is switched to its compact form (``SegmentedTabBar.set_compact``)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pageSectionBar")
        self.setAutoFillBackground(False)
        self.tabs: "SegmentedTabBar | None" = None
        self.actions: QtWidgets.QHBoxLayout | None = None

    def tab_room(self, width: int) -> int:
        """The width the segmented control may use in a bar ``width`` wide."""
        lay = self.layout()
        m = lay.contentsMargins()
        room = width - m.left() - m.right()
        if self.actions is not None and self.actions.count():
            room -= self.actions.sizeHint().width() + lay.spacing()
        return room

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.tabs is not None:
            room = self.tab_room(event.size().width())
            self.tabs.set_compact(self.tabs.natural_width() > room, room)


class SegmentedTabBar(QtWidgets.QTabBar):
    """The page section tabs as a glass segmented control.

    A pill track (the material's ``well``) holds every section; a raised
    thumb sits behind the current one and *slides* to the next — its rect is
    a painter value driven by ``motion.animate_value`` (``base`` duration,
    ``standard`` easing), never widget geometry. Tab buttons (the
    "Experimental" pills) keep their place at the right of their segment.
    Keyboard focus draws an accent ring round the thumb.

    It never scrolls sideways. When its natural width does not fit
    (``set_compact``, decided by the section bar), the segments take less
    padding and an "Experimental" pill becomes a small dot (its tab's
    tooltip says what it means); if even that is too wide, the labels elide.
    """

    PAD = 3          # track → thumb inset
    HEIGHT = 40
    COMPACT_PAD = 10     # text inset at each end of a compact segment
    MIN_CHARS = 4        # an elided label keeps at least this many characters

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._compact = False
        self._room: int | None = None
        self.setObjectName("pageSectionTabs")
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self._thumb: QtCore.QRectF | None = None
        self._from: QtCore.QRectF | None = None
        self._to: QtCore.QRectF | None = None
        self._hover = -1
        self._last_index = -1     # the segment the thumb rests on
        self.setUsesScrollButtons(False)
        self.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        # Compact widths depend on the current section: re-lay the tabs out
        # first, then slide the thumb to the new segment.
        self.currentChanged.connect(self._relayout_compact)
        self.currentChanged.connect(self._slide)

    # ── Geometry ──────────────────────────────────────────────────────────

    def is_compact(self) -> bool:
        return self._compact

    def set_compact(self, on: bool, room: int | None = None) -> None:
        """Compact form (see the class doc). ``room``: the width the bar may
        use; when even the compact segments are wider, the longest labels
        give way first (the short ones keep their full text)."""
        on = bool(on)
        room = room if on else None
        if on == self._compact and room == self._room:
            return
        if on != self._compact:
            for i in range(self.count()):
                button = self.tabButton(i, QtWidgets.QTabBar.ButtonPosition.RightSide)
                if hasattr(button, "set_dot"):
                    button.set_dot(on)
                    button.resize(button.sizeHint())   # the tab bar places it at this size
        self._compact, self._room = on, room
        # Re-run the tab layout with the new hints (setElideMode always does).
        self.setElideMode(self.elideMode())
        self.updateGeometry()
        self.update()

    def _relayout_compact(self, _index: int = -1) -> None:
        if self._compact and self._room is not None:
            self.setElideMode(self.elideMode())

    def label_font(self, active: bool = False) -> QtGui.QFont:
        """The segment font: label (13 px), caption size when compact."""
        f = QtGui.QFont(self.font())
        if self._compact:
            from gui.themes.tokens import TYPE
            f.setPixelSize(TYPE["caption"].px)
        f.setWeight(QtGui.QFont.Weight.DemiBold if active else QtGui.QFont.Weight.Medium)
        return f

    def _label_width(self, index: int, chars: int | None = None) -> int:
        fm = QtGui.QFontMetrics(self.label_font(active=True))
        text = self.tabText(index)
        if chars is not None and len(text) > chars:
            return fm.horizontalAdvance(text[:chars] + "…")
        return fm.horizontalAdvance(text)

    _BUTTON_GAP = 6      # label → tab button (the "Experimental" pill or dot)

    def _button_width(self, index: int) -> int:
        button = self.tabButton(index, QtWidgets.QTabBar.ButtonPosition.RightSide)
        if button is None:
            return 0
        return button.sizeHint().width() + self._BUTTON_GAP

    def natural_width(self) -> int:
        """The full (not compact) width of every segment together."""
        return sum(super(SegmentedTabBar, self).tabSizeHint(i).width() + 8
                   for i in range(self.count()))

    def _compact_width(self, index: int, chars: int | None = None) -> int:
        return self._label_width(index, chars) + 2 * self.COMPACT_PAD + self._button_width(index)

    def _compact_widths(self) -> list[int]:
        """Each compact segment's width within ``room``: the current section
        keeps its whole label when it can, then the widest others are capped
        first (never below their elided minimum)."""
        widths = [self._compact_width(i) for i in range(self.count())]
        room = self._room
        if room is None or sum(widths) <= room:
            return widths
        mins = [min(w, self._compact_width(i, self.MIN_CHARS)) for i, w in enumerate(widths)]
        current = self.currentIndex()
        if 0 <= current < len(widths) and \
                widths[current] + sum(m for i, m in enumerate(mins) if i != current) <= room:
            mins[current] = widths[current]

        def total(cap: int) -> int:
            return sum(max(lo, min(w, cap)) for w, lo in zip(widths, mins))

        lo, hi = 0, max(widths)
        while lo < hi:                      # the largest cap that fits
            mid = (lo + hi + 1) // 2
            if total(mid) <= room:
                lo = mid
            else:
                hi = mid - 1
        return [max(m, min(w, lo)) for w, m in zip(widths, mins)]

    def tabSizeHint(self, index: int) -> QtCore.QSize:  # noqa: N802
        if self._compact:
            widths = self._compact_widths()
            if 0 <= index < len(widths):
                return QtCore.QSize(widths[index], self.HEIGHT)
        s = super().tabSizeHint(index)
        return QtCore.QSize(s.width() + 8, self.HEIGHT)

    def minimumTabSizeHint(self, index: int) -> QtCore.QSize:  # noqa: N802
        """Elided: a few characters and "…", so the bar can always shrink
        instead of scrolling (or widening the page)."""
        width = self._compact_width(index, self.MIN_CHARS)
        return QtCore.QSize(min(width, self.tabSizeHint(index).width()), self.HEIGHT)

    def tabLayoutChange(self) -> None:  # noqa: N802
        """Compact: each tab button sits ``COMPACT_PAD`` from its segment's
        end (the style's own place, half the full tab padding in, would
        leave a gap the compact widths do not budget for)."""
        super().tabLayoutChange()
        if not getattr(self, "_compact", False):
            return
        for i in range(self.count()):
            button = self.tabButton(i, QtWidgets.QTabBar.ButtonPosition.RightSide)
            if button is None:
                continue
            rect = self.tabRect(i)
            size = button.size()
            button.move(rect.right() + 1 - self.COMPACT_PAD - size.width(),
                        rect.center().y() - size.height() // 2 + 1)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._compact:
            self.tabLayoutChange()
        if self._thumb_anim is None or self._thumb_anim.state() != QtCore.QAbstractAnimation.State.Running:
            self._thumb = None

    def _text_rect(self, index: int) -> QtCore.QRectF:
        """Where a segment's label is drawn (left of its tab button)."""
        rect = QtCore.QRectF(self.tabRect(index))
        button = self.tabButton(index, QtWidgets.QTabBar.ButtonPosition.RightSide)
        if self._compact:
            right = (button.geometry().left() - self._BUTTON_GAP + 1
                     if button is not None and button.isVisible() else rect.right() - self.COMPACT_PAD)
            return QtCore.QRectF(rect.left() + self.COMPACT_PAD, rect.top(),
                                 max(0.0, right - rect.left() - self.COMPACT_PAD), rect.height())
        if button is not None and button.isVisible():
            rect.setRight(button.geometry().left() - 4)
        return rect

    def _segment(self, index: int) -> QtCore.QRectF:
        return QtCore.QRectF(self.tabRect(index)).adjusted(
            self.PAD, self.PAD, -self.PAD, -self.PAD)

    def thumb_rect(self) -> QtCore.QRectF | None:
        """Where the thumb is painted now (mid-slide: in between)."""
        if self._thumb is None and self.currentIndex() >= 0:
            return self._segment(self.currentIndex())
        return self._thumb

    @property
    def _thumb_anim(self):
        from gui.themes import motion
        return motion.animation_for(self, "thumb")

    def _slide(self, index: int) -> None:
        from gui.themes import motion

        if index < 0:
            return
        previous, self._last_index = self._last_index, index
        target = self._segment(index)
        # currentChanged arrives after the switch: start from where the thumb
        # is painted now (mid-slide) or from the segment it rested on.
        start = self._thumb
        if start is None and 0 <= previous < self.count():
            start = self._segment(previous)
        if start is None or not self.isVisible() or start == target:
            motion.stop(self, "thumb")
            self._thumb = None
            self.update()
            return
        self._from, self._to = QtCore.QRectF(start), target
        motion.animate_value(self, "thumb", 0.0, 1.0, self._on_thumb, duration="base",
                             easing="standard", on_finished=self._on_thumb_done)

    def _on_thumb(self, value: float) -> None:
        a, b = self._from, self._to
        if a is None or b is None:
            return
        v = float(value)
        self._thumb = QtCore.QRectF(
            a.x() + (b.x() - a.x()) * v, a.y() + (b.y() - a.y()) * v,
            a.width() + (b.width() - a.width()) * v, a.height() + (b.height() - a.height()) * v,
        )
        self.update()

    def _on_thumb_done(self) -> None:
        # Settle onto the tab itself (it may have moved meanwhile).
        self._thumb = None
        self._from = self._to = None
        self.update()

    # ── Hover ─────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        index = self.tabAt(event.position().toPoint())
        if index != self._hover:
            self._hover = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = -1
        self.update()
        super().leaveEvent(event)

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface, qcolor

        n = self.count()
        if n == 0:
            return
        t = get_manager().current
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            track = QtCore.QRectF(self.tabRect(0)).united(QtCore.QRectF(self.tabRect(n - 1)))
            paint_surface(p, track, m.well, radius=track.height() / 2)
            if 0 <= self._hover < n and self._hover != self.currentIndex():
                seg = self._segment(self._hover)
                path = QtGui.QPainterPath()
                path.addRoundedRect(seg, seg.height() / 2, seg.height() / 2)
                p.fillPath(path, qcolor(m.hover))
            thumb = self.thumb_rect()
            if thumb is not None:
                paint_surface(p, thumb, m.raised, radius=thumb.height() / 2)
                if self.hasFocus():
                    p.setPen(QtGui.QPen(QtGui.QColor(t.input_focus), 1.5))
                    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    ring = thumb.adjusted(-1.5, -1.5, 1.5, 1.5)
                    p.drawRoundedRect(ring, ring.height() / 2, ring.height() / 2)
            for i in range(n):
                rect = self._text_rect(i)
                active = i == self.currentIndex()
                f = self.label_font(active)
                p.setFont(f)
                if not self.isTabEnabled(i):
                    colour = t.text_muted
                elif active or i == self._hover:
                    colour = t.text_primary
                else:
                    colour = t.text_secondary
                p.setPen(QtGui.QColor(colour))
                text = self.tabText(i)
                room = int(rect.width()) + (1 if self._compact else -8)
                fm = QtGui.QFontMetrics(f)
                if fm.horizontalAdvance(text) > room:
                    text = fm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, max(room, 0))
                p.drawText(rect, int(QtCore.Qt.AlignmentFlag.AlignCenter), text)
        finally:
            p.end()


class SectionedPage(QtWidgets.QWidget):
    """One sidebar page: section tabs (when there are several) over a stack
    of panels.

    Signals:
        section_changed(page_key, section_key) — the visible section changed
            (by a tab click or ``show_section``).
    """

    section_changed = Signal(str, str)

    def __init__(
        self, spec: Page, panels: dict[str, QtWidgets.QWidget],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(f"page_{spec.key}")
        self._spec = spec
        self._panels = dict(panels)
        self._current = spec.sections[0].key
        self._syncing = False
        self._pills: dict[str, ExperimentalPill] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs: QtWidgets.QTabBar | None = None
        self._actions_layout: QtWidgets.QHBoxLayout | None = None
        if spec.has_sections:
            from gui.themes.glass import font
            from gui.themes.tokens import SPACE

            bar = _SectionBar()
            row = QtWidgets.QHBoxLayout(bar)
            # Aligned with the cards below (the pages' gutter).
            row.setContentsMargins(SPACE.gutter, 0, SPACE.gutter, SPACE.md)
            row.setSpacing(8)
            tabs = SegmentedTabBar()
            tabs.setFont(font("label", weight=500))
            tabs.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
            for index, section in enumerate(spec.sections):
                tabs.addTab(section.label)
                tabs.setTabData(index, section.key)
                if section.experimental:
                    pill = ExperimentalPill(tabs, clickthrough=True, compact=True)
                    tabs.setTabButton(index, QtWidgets.QTabBar.ButtonPosition.RightSide, pill)
                    tabs.setTabToolTip(index, f"{section.label}: experimental — not ready to test")
                    self._pills[section.key] = pill
            tabs.currentChanged.connect(self._on_tab_changed)
            row.addWidget(tabs, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            row.addStretch(1)
            self._actions_layout = QtWidgets.QHBoxLayout()
            self._actions_layout.setContentsMargins(0, 0, 0, 0)
            self._actions_layout.setSpacing(8)
            row.addLayout(self._actions_layout)
            bar.tabs = tabs
            bar.actions = self._actions_layout
            root.addWidget(bar)
            self._bar = bar
            self._tabs = tabs
        else:
            self._bar = None

        self._stack = QtWidgets.QStackedWidget()
        root.addWidget(self._stack, 1)
        for section in spec.sections:
            panel = self._panels[section.panel]
            if self._stack.indexOf(panel) < 0:
                self._stack.addWidget(panel)
            if section.view is not None:
                # The page's tabs replace the panel's own view switcher.
                hide_tabs = getattr(panel, "set_section_tabs_visible", None)
                if callable(hide_tabs):
                    hide_tabs(False)
        self._show(self._current, activate=False, emit=False)

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def key(self) -> str:
        return self._spec.key

    @property
    def spec(self) -> Page:
        return self._spec

    def section_keys(self) -> list[str]:
        return [s.key for s in self._spec.sections]

    def current_section(self) -> str:
        return self._current

    def current_panel(self) -> QtWidgets.QWidget:
        return self.panel_for(self._current)

    def panel_for(self, section_key: str) -> QtWidgets.QWidget:
        section = self._spec.section(section_key)
        if section is None:
            raise KeyError(section_key)
        return self._panels[section.panel]

    def tab_bar(self) -> QtWidgets.QTabBar | None:
        return self._tabs

    def experimental_pills(self) -> dict[str, ExperimentalPill]:
        """Section key → the pill on its tab (prototypes only)."""
        return dict(self._pills)

    def section_for_panel(self, panel_key: str) -> str | None:
        """The section that shows ``panel_key``: for a panel serving several
        sections, the one matching the view it is showing now."""
        matches = [s for s in self._spec.sections if s.panel == panel_key]
        if not matches:
            return None
        if len(matches) > 1:
            panel = self._panels[panel_key]
            current = getattr(panel, "current_section", None)
            view = current() if callable(current) else None
            for section in matches:
                if section.view == view:
                    return section.key
        return matches[0].key

    # ── Page actions ──────────────────────────────────────────────────────

    def add_action(self, widget: QtWidgets.QWidget) -> None:
        """Put a page-level control to the right of the section tabs."""
        if self._actions_layout is None:
            raise RuntimeError(f"page {self.key!r} has no section bar for actions")
        self._actions_layout.addWidget(widget)

    # ── Switching ─────────────────────────────────────────────────────────

    def show_section(self, section_key: str, *, activate: bool = True) -> bool:
        """Show ``section_key``. ``activate`` runs the panel's
        ``on_activated()`` hook when the page is on screen. False if the page
        has no such section."""
        if self._spec.section(section_key) is None:
            return False
        self._show(section_key, activate=activate, emit=True)
        return True

    def activated(self) -> None:
        """The main window just made this page current."""
        self._activate_panel(self.current_panel())

    def _show(self, section_key: str, *, activate: bool, emit: bool) -> None:
        section: Section = self._spec.section(section_key)
        changed = section_key != self._current
        self._current = section_key
        panel = self._panels[section.panel]
        if self._tabs is not None:
            index = self.section_keys().index(section_key)
            if self._tabs.currentIndex() != index:
                self._syncing = True
                try:
                    self._tabs.setCurrentIndex(index)
                finally:
                    self._syncing = False
        if section.view is not None:
            set_section = getattr(panel, "set_section", None)
            if callable(set_section):
                set_section(section.view)
        switching = self._stack.currentWidget() is not panel
        self._stack.setCurrentWidget(panel)
        if switching and emit and self.isVisible():
            # Section transition: the new panel fades in (reduced motion: at once).
            from gui.themes import motion
            motion.reveal(panel, duration="fast")
        if activate and self.isVisible():
            self._activate_panel(panel)
        if emit and changed:
            self.section_changed.emit(self.key, section_key)

    def _on_tab_changed(self, index: int) -> None:
        if self._syncing or index < 0 or self._tabs is None:
            return
        key = self._tabs.tabData(index)
        if key and key != self._current:
            self._show(str(key), activate=True, emit=True)

    @staticmethod
    def _activate_panel(panel: QtWidgets.QWidget) -> None:
        hook = getattr(panel, "on_activated", None)
        if callable(hook):
            hook()


class SoundMapPage(SectionedPage):
    """Sound Map: analyse → map → select → play / save.

    *Analyse* is the clustering panel (Quick Start / Advanced / Results);
    *Map* is the interactive graph (select a cluster or lasso a region, then
    play it or export it as a playlist). A finished analysis offers "Open the
    map", which switches sections (``navigate_requested("graph")``).

    Page actions (right of the tabs, visible in both sections):

    * **Save cluster playlists…** — the previewed one-playlist-per-cluster
      export (``gui/cluster_playlist_jobs.py``,
      docs/specs/cluster_playlists.md). The Map panel owns the flow (it knows
      the hidden clusters, the run on the map and the colours, and refreshes
      the flow on every load and library change); this page places its
      button, makes the flow watch the *page* for visibility (so a held
      offer opens on either section, never over another page), forwards a
      successful analysis to ``offer_after_run`` and pauses the Analyse Run
      buttons while a preview or write runs.
    * **Quality report** — the latest analysis' cluster quality report,
      enabled once an analysis with scores has finished in this session.

    Both carry glyphs. The Analyse results card repeats *Save cluster
    playlists…* next to *Open the map*: it starts the same flow
    (``save_playlists_requested``) and mirrors the page button's enabled
    state and tooltip.
    """

    def __init__(
        self, spec: Page, panels: dict[str, QtWidgets.QWidget],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(spec, panels, parent)
        self._build_actions()

    @property
    def analysis_panel(self):
        return self.panel_for("analyse")

    @property
    def map_panel(self):
        return self.panel_for("map")

    @property
    def cluster_playlists_flow(self):
        """The Map panel's Save cluster playlists flow (None if it has none)."""
        return getattr(self.map_panel, "cluster_playlists_flow", None)

    def _build_actions(self) -> None:
        from gui.widgets.glyphs import set_glyph

        flow = self.cluster_playlists_flow
        analysis = self.analysis_panel
        if flow is not None:
            flow.set_show_target(self)
            set_glyph(flow.button, "save")
            self.add_action(flow.button)
            succeeded = getattr(analysis, "analysis_succeeded", None)
            if succeeded is not None:
                succeeded.connect(flow.offer_after_run)
            pause = getattr(analysis, "set_external_busy", None)
            if callable(pause):
                flow.busy_changed.connect(pause)
            # The results card's own "Save cluster playlists…" runs the same
            # flow and mirrors its button (enabled state and tooltip).
            requested = getattr(analysis, "save_playlists_requested", None)
            if requested is not None:
                requested.connect(self._on_save_playlists_requested)
            flow.button.installEventFilter(self)
            self._mirror_save_state()

        self._report_btn = QtWidgets.QPushButton("Quality report")
        self._report_btn.setObjectName("soundMapReportBtn")
        set_glyph(self._report_btn, "info")
        self._report_btn.setToolTip("How well the latest analysis separated your tracks")
        self._report_btn.clicked.connect(self._on_quality_report)
        self.add_action(self._report_btn)
        signal = getattr(analysis, "result_changed", None)
        if signal is not None:
            signal.connect(self._sync_actions)
        self._sync_actions()

    def _on_save_playlists_requested(self) -> None:
        flow = self.cluster_playlists_flow
        if flow is not None and flow.button.isEnabled() and not flow.is_busy():
            flow.start()

    def _mirror_save_state(self) -> None:
        flow = self.cluster_playlists_flow
        mirror = getattr(self.analysis_panel, "set_save_playlists_state", None)
        if flow is None or not callable(mirror):
            return
        # While a preview runs the flow button reads "Cancel preview"; the
        # results card only ever starts one.
        mirror(flow.button.isEnabled() and not flow.is_busy(), flow.button.toolTip())

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        flow = self.cluster_playlists_flow
        if flow is not None and obj is flow.button and event.type() in (
            QtCore.QEvent.Type.EnabledChange, QtCore.QEvent.Type.ToolTipChange,
        ):
            self._mirror_save_state()
        return super().eventFilter(obj, event)

    def _sync_actions(self, *_args) -> None:
        available = getattr(self.analysis_panel, "quality_report_available", None)
        ok = bool(available()) if callable(available) else False
        self._report_btn.setEnabled(ok)
        if not ok:
            self._report_btn.setToolTip("Run an analysis first (Analyse → Quick Start)")
        else:
            self._report_btn.setToolTip("How well the latest analysis separated your tracks")

    def _on_quality_report(self) -> None:
        show = getattr(self.analysis_panel, "show_quality_report", None)
        if callable(show):
            show()
