"""Settings controls and their load/save logic — metadata services, default
library, thresholds, appearance.

``SettingsForm`` owns every control and the save logic; the Settings *page*
(``gui/workspaces/settings.py``) lays its widgets out in three sections
(Metadata services · General · Appearance). ``SettingsDrawer`` is the same
form in a modal dialog (the app no longer opens it: Ctrl+, and the top bar's
Settings button go to the page).

Every control persists through ``config.load_config()/save_config()`` under
these keys (other pages read them, so the names must not change):

=================================  ==============================================
Control                            Config key
=================================  ==============================================
Service combo                      ``metadata_service`` ("AcoustID" | "MusicBrainz" | "Last.fm")
AcoustID API key                   ``metadata_api_key`` (empty = built-in key)
Last.fm API key                    ``lastfm_api_key`` (``LASTFM_API_KEY`` env wins)
MusicBrainz app / version / mail   ``musicbrainz_useragent`` = {app, version, contact}
Default library                    ``library_root``
Near-duplicate threshold           ``near_duplicate_threshold``
Exact-duplicate threshold          ``exact_duplicate_threshold``
Backdrop (Album art/Gradient/Plain) ``backdrop_mode`` (+ legacy ``bg_gradient_enabled``)
Reduce motion                      ``reduced_motion``
=================================  ==============================================

The backdrop replaced the old "Background gradient" switch: its key is still
read when ``backdrop_mode`` is absent and is kept in step on save (``False``
means Plain). Theme, accent and Reduce transparency are not here: they apply
instantly from the theme picker (``gui/themes/picker.py``).
"""
from __future__ import annotations

import os

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot

# Services with an API key, and the config key each one is stored under. The
# Settings dialog shows one key field for whichever service is selected; it
# used to save every key into ``metadata_api_key``, so a Last.fm key replaced
# the AcoustID key (and the Last.fm source never read it).
_KEY_CONFIG = {"AcoustID": "metadata_api_key", "Last.fm": "lastfm_api_key"}
_KEYED_SERVICES = tuple(_KEY_CONFIG)


def check_service_connection(
    service: str, api_key: str = "", useragent: tuple[str, str, str] = ("", "", "")
) -> tuple[bool, str]:
    """Try the service with the entered credentials. Blocking (network); call
    it off the GUI thread. Returns (ok, message)."""
    try:
        if service == "AcoustID":
            if not api_key.strip():
                return False, "Enter an AcoustID API key first"
            import requests
            resp = requests.get(
                "https://api.acoustid.org/v2/lookup",
                params={
                    "client": api_key.strip(), "format": "json",
                    "duration": "1", "fingerprint": "AQAA",
                },
                timeout=8,
            )
            try:
                data = resp.json()
            except ValueError:
                data = {}
            err = (data or {}).get("error") or {}
            code = err.get("code")
            message = str(err.get("message", ""))
            if code == 4 or "api key" in message.lower():
                return False, "AcoustID: the API key was rejected"
            if resp.status_code >= 500:
                return False, f"AcoustID: server error ({resp.status_code})"
            return True, "AcoustID: connected, key accepted"
        if service == "Last.fm":
            if not api_key.strip():
                return False, "Enter a Last.fm API key first"
            import requests
            resp = requests.get(
                "https://ws.audioscrobbler.com/2.0/",
                params={
                    "method": "chart.gettopartists", "api_key": api_key.strip(),
                    "format": "json", "limit": "1",
                },
                timeout=8,
            )
            try:
                data = resp.json()
            except ValueError:
                data = {}
            if isinstance(data, dict) and data.get("error"):
                return False, f"Last.fm: {data.get('message') or 'request rejected'}"
            if resp.status_code >= 400:
                return False, f"Last.fm: HTTP {resp.status_code}"
            return True, "Last.fm: connected, key accepted"
        if service == "MusicBrainz":
            import musicbrainzngs
            app, version, contact = useragent
            musicbrainzngs.set_useragent(app or "AlphaDEX", version or "1.0", contact)
            musicbrainzngs.search_artists(query="Beatles", limit=1)
            return True, "MusicBrainz: OK"
        return False, f"No connection test for {service}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc) or type(exc).__name__


class SettingsForm(QtCore.QObject):
    """Every Settings control, plus load / save.

    The controls live in three widgets the host lays out: ``services_widget``
    (service, keys, MusicBrainz user agent, connection test),
    ``general_widget`` (default library, thresholds) and
    ``appearance_widget`` (backdrop, reduce motion). ``save()`` validates, writes
    the keys above with load-modify-save, sets ``library_changed`` and emits
    ``settings_saved``.
    """

    settings_saved = Signal()

    # The controls, by attribute name (hosts expose them under these names).
    WIDGET_NAMES = (
        "_service_combo", "_acoustid_frame", "_api_key_entry", "_mb_frame",
        "_mb_app", "_mb_ver", "_mb_contact", "_test_btn", "_test_status",
        "_library_entry", "_near_dup_spin", "_exact_dup_spin", "_backdrop_combo",
        "_reduce_motion_cb",
    )

    def __init__(self, host: QtWidgets.QWidget | None = None) -> None:
        super().__init__(host)
        self._host = host
        self._loaded_library = ""
        self._loaded_values: dict | None = None
        # True after save() when the default library was changed here.
        self.library_changed = False
        try:
            from gui.themes.manager import get_manager
            t = get_manager().current
            self._ok_colour, self._err_colour = t.success, t.danger
        except Exception:
            self._ok_colour, self._err_colour = "green", "red"
        self.services_widget = self._build_services()
        self.general_widget = self._build_general()
        self.appearance_widget = self._build_appearance()
        self._load_config()

    def _parent_widget(self) -> QtWidgets.QWidget | None:
        return self._host

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_services(self) -> QtWidgets.QWidget:
        meta_w = QtWidgets.QWidget()
        meta_l = QtWidgets.QFormLayout(meta_w)
        meta_l.setContentsMargins(0, 0, 0, 0)
        meta_l.setSpacing(12)
        meta_l.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        try:
            from config import SUPPORTED_SERVICES, UNAVAILABLE_SERVICES
        except ImportError:
            SUPPORTED_SERVICES = ["AcoustID", "MusicBrainz", "Last.fm"]
            UNAVAILABLE_SERVICES = {"Spotify", "Gracenote"}

        self._service_combo = QtWidgets.QComboBox()
        for svc in SUPPORTED_SERVICES:
            if svc in UNAVAILABLE_SERVICES:
                self._service_combo.addItem(f"{svc} (unavailable)")
                idx = self._service_combo.count() - 1
                self._service_combo.model().item(idx).setEnabled(False)
                self._service_combo.model().item(idx).setToolTip(
                    f"{svc} is not yet implemented. See ROADMAP.md."
                )
            else:
                self._service_combo.addItem(svc)
        self._service_combo.currentTextChanged.connect(self._on_service_changed)
        meta_l.addRow("Service:", self._service_combo)

        # API-key section: one field showing the selected service's own key
        self._acoustid_frame = QtWidgets.QFrame()
        af_layout = QtWidgets.QFormLayout(self._acoustid_frame)
        af_layout.setContentsMargins(0, 0, 0, 0)
        self._api_key_entry = QtWidgets.QLineEdit()
        self._api_key_entry.setPlaceholderText("Your AcoustID API key")
        self._api_key_entry.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        af_layout.addRow("API Key:", self._api_key_entry)
        meta_l.addRow(self._acoustid_frame)

        # MusicBrainz section
        self._mb_frame = QtWidgets.QFrame()
        mb_layout = QtWidgets.QFormLayout(self._mb_frame)
        mb_layout.setContentsMargins(0, 0, 0, 0)
        mb_layout.setSpacing(8)
        self._mb_app = QtWidgets.QLineEdit()
        self._mb_ver = QtWidgets.QLineEdit()
        self._mb_contact = QtWidgets.QLineEdit()
        self._mb_contact.setPlaceholderText("your@email.com")
        mb_layout.addRow("App name:", self._mb_app)
        mb_layout.addRow("Version:", self._mb_ver)
        mb_layout.addRow("Contact:", self._mb_contact)
        meta_l.addRow(self._mb_frame)

        # Test + status
        test_row = QtWidgets.QHBoxLayout()
        self._test_btn = QtWidgets.QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test)
        self._test_status = QtWidgets.QLabel("")
        test_row.addWidget(self._test_btn)
        test_row.addWidget(self._test_status)
        test_row.addStretch(1)
        meta_l.addRow("", test_row)
        return meta_w

    def _build_general(self) -> QtWidgets.QWidget:
        gen_w = QtWidgets.QWidget()
        gen_l = QtWidgets.QFormLayout(gen_w)
        gen_l.setContentsMargins(0, 0, 0, 0)
        gen_l.setSpacing(12)
        gen_l.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self._library_entry = QtWidgets.QLineEdit()
        lib_row = QtWidgets.QHBoxLayout()
        lib_row.addWidget(self._library_entry, 1)
        lib_browse = QtWidgets.QPushButton("Browse…")
        lib_browse.clicked.connect(self._on_browse_library)
        lib_row.addWidget(lib_browse)
        gen_l.addRow("Default library:", lib_row)

        # Both ranges match the Duplicates page's spin boxes (0–1), so a saved
        # value is never silently clamped when this form re-saves it.
        self._near_dup_spin = QtWidgets.QDoubleSpinBox()
        self._near_dup_spin.setRange(0.0, 1.0)
        self._near_dup_spin.setSingleStep(0.01)
        self._near_dup_spin.setDecimals(3)
        self._near_dup_spin.setValue(0.1)
        gen_l.addRow("Near-duplicate threshold:", self._near_dup_spin)

        self._exact_dup_spin = QtWidgets.QDoubleSpinBox()
        self._exact_dup_spin.setRange(0.0, 1.0)
        self._exact_dup_spin.setSingleStep(0.005)
        self._exact_dup_spin.setDecimals(4)
        self._exact_dup_spin.setValue(0.02)
        gen_l.addRow("Exact-duplicate threshold:", self._exact_dup_spin)
        for spin in (self._near_dup_spin, self._exact_dup_spin):
            spin.setMinimumWidth(120)   # the size hint clipped "0.0200"
        return gen_w

    def _build_appearance(self) -> QtWidgets.QWidget:
        from gui.themes.appearance import BACKDROP_LABELS, BACKDROP_MODES

        app_w = QtWidgets.QWidget()
        app_l = QtWidgets.QFormLayout(app_w)
        app_l.setContentsMargins(0, 0, 0, 0)
        app_l.setSpacing(12)
        self._backdrop_combo = QtWidgets.QComboBox()
        for mode in BACKDROP_MODES:
            self._backdrop_combo.addItem(BACKDROP_LABELS[mode], mode)
        self._backdrop_combo.setToolTip(
            "What the window shows behind its glass panels.\n"
            "Album art: a blurred wall of covers from your library (Liquid Glass; "
            "classic themes show their gradient).\n"
            "Gradient: soft accent glows.  Plain: a solid colour."
        )
        app_l.addRow("Backdrop:", self._backdrop_combo)
        self._reduce_motion_cb = QtWidgets.QCheckBox(
            "Reduce motion — no slides, fades or fly-ins"
        )
        self._reduce_motion_cb.setToolTip(
            "Animations jump straight to their end (the ALPHADEX_REDUCED_MOTION "
            "environment variable does the same)."
        )
        app_l.addRow("Motion:", self._reduce_motion_cb)
        return app_w

    def backdrop_mode(self) -> str:
        return str(self._backdrop_combo.currentData() or "art")

    # ── Load / save ───────────────────────────────────────────────────────

    def reload(self) -> None:
        """Show what is saved now (drops unsaved edits)."""
        self._load_config()

    def values(self) -> dict:
        """What ``save()`` would write, by config key."""
        self._stash_key()
        keys = dict(self.__dict__.get("_keys", {}))
        values = {
            "metadata_service": self._service_combo.currentText(),
            "musicbrainz_useragent": {
                "app": self._mb_app.text(),
                "version": self._mb_ver.text(),
                "contact": self._mb_contact.text(),
            },
            "library_root": self._library_entry.text().strip(),
            "near_duplicate_threshold": self._near_dup_spin.value(),
            "exact_duplicate_threshold": self._exact_dup_spin.value(),
            "reduced_motion": self._reduce_motion_cb.isChecked(),
        }
        from gui.themes.appearance import config_values
        values.update(config_values(self.backdrop_mode()))
        for svc_name, cfg_key in _KEY_CONFIG.items():
            values[cfg_key] = keys.get(svc_name, "").strip()
        return values

    def is_dirty(self) -> bool:
        """True when a control differs from what was last loaded or saved."""
        return self._loaded_values is not None and self.values() != self._loaded_values

    def _load_config(self) -> None:
        try:
            from config import load_config
            cfg = load_config()
            svc = cfg.get("metadata_service", "AcoustID")
            # Empty AcoustID key = the built-in key (plugins.acoustid_plugin);
            # don't copy the built-in key into the user's config.
            self._keys = {s: str(cfg.get(k) or "") for s, k in _KEY_CONFIG.items()}
            self._key_service = None
            idx = self._service_combo.findText(svc)
            if idx >= 0:
                self._service_combo.setCurrentIndex(idx)
            ua = cfg.get("musicbrainz_useragent", {})
            self._mb_app.setText(ua.get("app", ""))
            self._mb_ver.setText(ua.get("version", ""))
            self._mb_contact.setText(ua.get("contact", ""))
            self._loaded_library = str(cfg.get("library_root", "") or "")
            self._library_entry.setText(self._loaded_library)
            self._near_dup_spin.setValue(float(cfg.get("near_duplicate_threshold", 0.1)))
            self._exact_dup_spin.setValue(float(cfg.get("exact_duplicate_threshold", 0.02)))
            from gui.themes.appearance import mode_from_config
            idx = self._backdrop_combo.findData(mode_from_config(cfg))
            self._backdrop_combo.setCurrentIndex(max(0, idx))
            self._reduce_motion_cb.setChecked(bool(cfg.get("reduced_motion", False)))
        except Exception:
            pass
        self._key_service = None
        self._on_service_changed(self._service_combo.currentText())
        self._loaded_values = self.values()

    @Slot()
    def save(self) -> bool:
        """Validate and save. True when saved (and ``settings_saved`` emitted)."""
        library = self._library_entry.text().strip()
        changed = library != self._loaded_library
        # Only a library typed/browsed here is validated: an unchanged saved
        # path that has gone missing must not block saving other settings.
        if changed and library and not os.path.isdir(os.path.expanduser(library)):
            QtWidgets.QMessageBox.warning(
                self._parent_widget(),
                "Library folder not found",
                f"The default library folder\n{library}\ndoes not exist. "
                "Choose an existing folder, or clear the field.",
            )
            return False
        try:
            from config import load_config, save_config
            cfg = load_config()
            cfg.update(self.values())
            save_config(cfg)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self._parent_widget(), "Save Error", str(exc))
            return False
        self._loaded_library = library
        self._loaded_values = self.values()
        self.library_changed = changed
        self.settings_saved.emit()
        return True

    def _on_save(self) -> bool:
        """The dialog-era name (older callers and tests use it)."""
        return self.save()

    @Slot()
    def _on_test(self) -> None:
        self._test_status.setText("Testing…")
        self._test_btn.setEnabled(False)

        # Read every widget value here, on the GUI thread; the worker thread
        # only gets plain strings.
        svc = self._service_combo.currentText()
        api_key = self._api_key_entry.text()
        useragent = (self._mb_app.text(), self._mb_ver.text(), self._mb_contact.text())

        def _run() -> None:
            ok, msg = check_service_connection(svc, api_key, useragent)
            try:
                QtCore.QMetaObject.invokeMethod(
                    self, "_apply_test_result",
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(bool, ok),
                    QtCore.Q_ARG(str, msg),
                )
            except RuntimeError:
                pass  # the form was closed and deleted meanwhile

        import threading
        threading.Thread(target=_run, daemon=True, name="SettingsConnectionTest").start()

    @Slot(bool, str)
    def _apply_test_result(self, ok: bool, msg: str) -> None:
        self._test_btn.setEnabled(True)
        self._test_status.setText(msg)
        colour = self._ok_colour if ok else self._err_colour
        self._test_status.setStyleSheet(f"color: {colour};")

    def _stash_key(self) -> None:
        """Remember what is typed in the key field for the service it belongs to."""
        keys = self.__dict__.setdefault("_keys", {})
        owner = self.__dict__.get("_key_service")
        if owner in _KEY_CONFIG:
            keys[owner] = self._api_key_entry.text()

    @Slot(str)
    def _on_service_changed(self, svc: str) -> None:
        self._stash_key()
        self._key_service = svc
        self._acoustid_frame.setVisible(svc in _KEYED_SERVICES)
        if svc in _KEY_CONFIG:
            self._api_key_entry.setText(self.__dict__.get("_keys", {}).get(svc, ""))
        self._api_key_entry.setPlaceholderText(
            "Your Last.fm API key" if svc == "Last.fm"
            else "Your AcoustID API key (empty = built-in key)"
        )
        self._mb_frame.setVisible(svc == "MusicBrainz")

    @Slot()
    def _on_browse_library(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self._parent_widget(), "Select Default Library Folder", self._library_entry.text()
        )
        if folder:
            self._library_entry.setText(folder)


class SettingsDrawer(QtWidgets.QDialog):
    """``SettingsForm`` in a modal dialog (tabs: Metadata Services · General)."""

    settings_saved = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self.setMinimumHeight(500)
        self.form = SettingsForm(self)
        for name in SettingsForm.WIDGET_NAMES:
            setattr(self, name, getattr(self.form, name))
        self.form.settings_saved.connect(self.settings_saved)
        self._build_ui()

    @property
    def library_changed(self) -> bool:
        return self.form.library_changed

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header (theme-aware: follows the current palette, light or dark)
        try:
            from gui.themes.manager import get_manager
            t = get_manager().current
            hdr_bg, hdr_fg, hdr_muted = t.sidebar_bg, t.text_primary, t.text_secondary
        except Exception:
            hdr_bg, hdr_fg, hdr_muted = "palette(window)", "palette(text)", "palette(mid)"
        header = QtWidgets.QFrame()
        header.setStyleSheet(f"background: {hdr_bg}; padding: 16px;")
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(20, 16, 20, 16)
        title = QtWidgets.QLabel("Settings")
        title.setStyleSheet(f"color: {hdr_fg}; font-size: 16px; font-weight: 700;")
        h_layout.addWidget(title)
        h_layout.addStretch(1)
        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(
            f"background: transparent; border: none; color: {hdr_muted}; font-size: 16px;"
        )
        close_btn.clicked.connect(self.reject)
        h_layout.addWidget(close_btn)
        root.addWidget(header)

        tabs = QtWidgets.QTabWidget()
        tabs.setContentsMargins(16, 16, 16, 16)

        def padded(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
            page = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(page)
            lay.setContentsMargins(20, 20, 20, 20)
            lay.setSpacing(12)
            for w in widgets:
                lay.addWidget(w)
            lay.addStretch(1)
            return page

        tabs.addTab(padded(self.form.services_widget), "Metadata Services")
        tabs.addTab(padded(self.form.general_widget, self.form.appearance_widget), "General")
        root.addWidget(tabs)

        # Footer buttons
        footer = QtWidgets.QWidget()
        foot_l = QtWidgets.QHBoxLayout(footer)
        foot_l.setContentsMargins(20, 12, 20, 16)
        foot_l.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setObjectName("primaryBtn")
        self._save_btn.clicked.connect(self._on_save)
        foot_l.addWidget(cancel_btn)
        foot_l.addWidget(self._save_btn)
        root.addWidget(footer)

    @Slot()
    def _on_save(self) -> None:
        if self.form.save():
            self.accept()

    @Slot()
    def _on_test(self) -> None:
        self.form._on_test()
