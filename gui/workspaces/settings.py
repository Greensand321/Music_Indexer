"""Settings page: metadata services, general settings and appearance.

The controls and their save logic are ``SettingsForm``
(``gui/dialogs/settings_drawer.py``), the same form the old Settings dialog
showed, so every config key is unchanged. Saving emits ``settings_saved``;
the main window then runs ``_on_settings_saved(panel)`` exactly as it did for
the dialog (refresh cached appearance settings, tell every page, switch
library if the default library changed here).

Layout (Liquid Glass): one header action row (the page's purpose in one
muted line, the save state, Revert and Save settings) above the sections,
then one glass card per group of controls (``SECTION_GROUPS``):

* ``services``: Lookup service (the service rows; Spotify and Gracenote
  disabled with their tooltip) · Connection (keys, MusicBrainz identity,
  Test connection);
* ``general``: Default library · Duplicate thresholds · Motion (Reduce
  motion);
* ``appearance``: Theme (the theme picker: Liquid Glass with its accents and
  Reduce transparency, Auto, the classic themes — all apply instantly) ·
  Backdrop (Album art / Gradient / Plain, a segmented pill; applies on Save).

Sections are switched by the page's tabs; the panel's own tab bar is hidden
then.
"""
from __future__ import annotations

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.dialogs.settings_drawer import SECTION_GROUPS, SettingsForm
from gui.workspaces.base import WorkspaceBase


class SettingsWorkspace(WorkspaceBase):
    settings_saved = Signal()

    SECTIONS = ("services", "general", "appearance")

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self.form = SettingsForm(self)
        for name in SettingsForm.WIDGET_NAMES:
            setattr(self, name, getattr(self.form, name))
        self.form.settings_saved.connect(self._on_form_saved)
        self._build_ui()

    @property
    def library_changed(self) -> bool:
        """True after a save that changed the default library."""
        return self.form.library_changed

    # ── UI ────────────────────────────────────────────────────────────────

    def _card(self, title: str, body: QtWidgets.QWidget, subtitle: str = "") -> QtWidgets.QFrame:
        from gui.themes.tokens import SPACE

        card = self._make_card()
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(SPACE.md)
        head = QtWidgets.QVBoxLayout()
        head.setSpacing(SPACE.xxs)
        head.addWidget(self._make_card_title(title))
        if subtitle:
            label = QtWidgets.QLabel(subtitle)
            label.setObjectName("mutedLabel")
            label.setWordWrap(True)
            head.addWidget(label)
        lay.addLayout(head)
        lay.addWidget(body)
        return card

    def _view(self, *cards: QtWidgets.QWidget) -> QtWidgets.QWidget:
        from gui.themes.tokens import SPACE

        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACE.gutter)
        for card in cards:
            lay.addWidget(card)
        lay.addStretch(1)
        return page

    def _group_cards(self, section: str) -> list[QtWidgets.QFrame]:
        return [self._card(title, self.form.group(attr), subtitle)
                for attr, title, subtitle in SECTION_GROUPS[section]]

    def _build_ui(self) -> None:
        from gui.themes.picker import ThemePickerPanel
        from gui.themes.tokens import SPACE

        cl = self.content_layout

        # Header action row: what the page is for, then Revert / Save, in
        # reach on all three sections (Appearance is taller than the window).
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(SPACE.sm)
        subtitle = self._make_subtitle(
            "The look applies as you pick it; everything else when you save."
        )
        bar.addWidget(subtitle, 1)
        self._save_state = QtWidgets.QLabel("")
        self._save_state.setObjectName("statusHint")
        bar.addWidget(self._save_state)
        self._revert_btn = QtWidgets.QPushButton("Revert")
        self._revert_btn.setToolTip("Show the saved settings again (drops unsaved changes)")
        self._revert_btn.clicked.connect(self._on_revert)
        bar.addWidget(self._revert_btn)
        self._save_btn = self._make_primary_button("Save settings")
        self._save_btn.setToolTip("Save every section's settings")
        self._save_btn.clicked.connect(self._on_save)
        bar.addWidget(self._save_btn)
        cl.addLayout(bar)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._view(*self._group_cards("services")), "Metadata services")
        self._tabs.addTab(self._view(*self._group_cards("general")), "General")
        self.theme_picker = ThemePickerPanel()
        self._tabs.addTab(self._view(
            self._card("Theme", self.theme_picker, "Applies as soon as you pick it."),
            *self._group_cards("appearance"),
        ), "Appearance")
        # A tab widget is as tall as its tallest page; size it to the one
        # shown instead, so short sections don't scroll into empty space.
        self._tabs.currentChanged.connect(self._fit_to_current)
        self._fit_to_current(self._tabs.currentIndex())
        cl.addWidget(self._tabs)
        cl.addStretch(1)

    def _fit_to_current(self, index: int) -> None:
        policy = QtWidgets.QSizePolicy.Policy
        for i in range(self._tabs.count()):
            view = self._tabs.widget(i)
            if i == index:
                view.setSizePolicy(policy.Preferred, policy.Preferred)
            else:
                view.setSizePolicy(policy.Ignored, policy.Ignored)
            view.updateGeometry()
        self._tabs.updateGeometry()

    # ── Sections ──────────────────────────────────────────────────────────

    def set_section(self, view: str) -> None:
        if view in self.SECTIONS:
            self._tabs.setCurrentIndex(self.SECTIONS.index(view))

    def current_section(self) -> str:
        return self.SECTIONS[max(self._tabs.currentIndex(), 0)]

    def set_section_tabs_visible(self, visible: bool) -> None:
        self._tabs.tabBar().setVisible(visible)
        # No pane frame either: without its tabs it would frame nothing.
        self._tabs.setDocumentMode(not visible)

    # ── Load / save ───────────────────────────────────────────────────────

    def on_activated(self) -> None:
        """Shown again: pick up settings changed elsewhere (e.g. Change
        Library saved a new default), unless there are unsaved edits here."""
        if not self.form.is_dirty():
            self.form.reload()

    def _on_library_changed(self, path: str) -> None:
        if not self.form.is_dirty():
            self.form.reload()

    @Slot()
    def _on_save(self) -> bool:
        return self.form.save()

    @Slot()
    def _on_revert(self) -> None:
        self.form.reload()
        self._save_state.setText("Showing the saved settings.")

    @Slot()
    def _on_form_saved(self) -> None:
        stamp = QtCore.QTime.currentTime().toString("HH:mm")
        self._save_state.setText(f"Saved at {stamp}.")
        self._log("Settings saved.", "ok")
        self.settings_saved.emit()
