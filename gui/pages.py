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
uses that for its quality report and is where "Save clusters as playlists"
belongs.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.navigation import Page, Section
from gui.themes.manager import get_manager
from gui.widgets.pill import ExperimentalPill


class _SectionBar(QtWidgets.QWidget):
    """The strip above a page's sections: tabs on the left, actions on the
    right. Painted from tokens (content background, a hairline below)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pageSectionBar")
        get_manager().theme_changed.connect(self._repaint_on_theme)

    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(t.content_bg))
        p.fillRect(QtCore.QRect(0, self.height() - 1, self.width(), 1), QtGui.QColor(t.card_border))
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
            bar = _SectionBar()
            row = QtWidgets.QHBoxLayout(bar)
            row.setContentsMargins(12, 10, 16, 0)
            row.setSpacing(8)
            tabs = QtWidgets.QTabBar()
            tabs.setObjectName("pageSectionTabs")
            tabs.setDrawBase(False)
            tabs.setExpanding(False)
            tabs.setUsesScrollButtons(True)
            tabs.setElideMode(QtCore.Qt.TextElideMode.ElideNone)
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
            row.addWidget(tabs, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
            row.addStretch(1)
            self._actions_layout = QtWidgets.QHBoxLayout()
            self._actions_layout.setContentsMargins(0, 0, 0, 6)
            self._actions_layout.setSpacing(8)
            row.addLayout(self._actions_layout)
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
        self._stack.setCurrentWidget(panel)
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
        flow = self.cluster_playlists_flow
        if flow is not None:
            flow.set_show_target(self)
            self.add_action(flow.button)
            analysis = self.analysis_panel
            succeeded = getattr(analysis, "analysis_succeeded", None)
            if succeeded is not None:
                succeeded.connect(flow.offer_after_run)
            pause = getattr(analysis, "set_external_busy", None)
            if callable(pause):
                flow.busy_changed.connect(pause)

        self._report_btn = QtWidgets.QPushButton("Quality report")
        self._report_btn.setObjectName("soundMapReportBtn")
        self._report_btn.setToolTip("How well the latest analysis separated your tracks")
        self._report_btn.clicked.connect(self._on_quality_report)
        self.add_action(self._report_btn)
        analysis = self.analysis_panel
        signal = getattr(analysis, "result_changed", None)
        if signal is not None:
            signal.connect(self._sync_actions)
        self._sync_actions()

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
