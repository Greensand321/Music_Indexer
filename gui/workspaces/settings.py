"""Settings page: metadata services, general settings and appearance.

The controls and their save logic are ``SettingsForm``
(``gui/dialogs/settings_drawer.py``), the same form the old Settings dialog
showed, so every config key is unchanged. Saving emits ``settings_saved``;
the main window then runs ``_on_settings_saved(panel)`` exactly as it did for
the dialog (refresh cached appearance settings, tell every page, switch
library if the default library changed here).

Sections (switched by the page's tabs; the panel's own tab bar is hidden
then): ``services`` · ``general`` · ``appearance``. Appearance embeds the
theme picker (themes apply instantly) and the background-gradient switch
(applies on Save).
"""
from __future__ import annotations

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.dialogs.settings_drawer import SettingsForm
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

    def _card(self, title: str, *widgets: QtWidgets.QWidget, hint: str = "") -> QtWidgets.QFrame:
        card = self._make_card()
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        lay.addWidget(self._make_card_title(title))
        if hint:
            label = QtWidgets.QLabel(hint)
            label.setObjectName("statusHint")
            label.setWordWrap(True)
            lay.addWidget(label)
        for w in widgets:
            lay.addWidget(w)
        return card

    def _view(self, *cards: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 12, 0, 0)
        lay.setSpacing(16)
        for card in cards:
            lay.addWidget(card)
        lay.addStretch(1)
        return page

    def _build_ui(self) -> None:
        from gui.themes.picker import ThemePickerPanel

        cl = self.content_layout
        cl.addWidget(self._make_section_title("Settings"))
        cl.addWidget(self._make_subtitle(
            "Where metadata comes from, which library opens by default, the duplicate "
            "thresholds, and how AlphaDEX looks. Themes apply as soon as you pick one; "
            "everything else applies when you save."
        ))

        # Save / Revert sit above the sections, so they are in reach on all
        # three (the Appearance section is taller than the window).
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(8)
        self._save_state = QtWidgets.QLabel("")
        self._save_state.setObjectName("statusHint")
        bar.addWidget(self._save_state, 1)
        self._revert_btn = QtWidgets.QPushButton("Revert")
        self._revert_btn.setToolTip("Show the saved settings again (drops unsaved changes)")
        self._revert_btn.clicked.connect(self._on_revert)
        bar.addWidget(self._revert_btn)
        self._save_btn = self._make_primary_button("Save settings")
        self._save_btn.setObjectName("primaryBtn")
        self._save_btn.clicked.connect(self._on_save)
        bar.addWidget(self._save_btn)
        cl.addLayout(bar)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._view(self._card(
            "Metadata services", self.form.services_widget,
            hint="Used by the Tag Fixer and the genre tools. An empty AcoustID key "
                 "uses the built-in key.",
        )), "Metadata services")
        self._tabs.addTab(self._view(self._card(
            "Library and duplicates", self.form.general_widget,
            hint="The default library opens at start-up. The thresholds are the "
                 "Duplicates page's starting values.",
        )), "General")
        self.theme_picker = ThemePickerPanel()
        self._tabs.addTab(self._view(
            self._card("Theme", self.theme_picker, hint="Changes apply instantly."),
            self._card("Background", self.form.appearance_widget,
                       hint="Applies when you save."),
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
