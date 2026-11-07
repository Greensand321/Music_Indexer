"""Settings dialog (modal) — metadata services, default library, thresholds.

Every control persists through ``config.load_config()/save_config()`` under
these keys (other pages read them, so the names must not change):

=================================  ==============================================
Control                            Config key
=================================  ==============================================
Service combo                      ``metadata_service`` ("AcoustID" | "MusicBrainz" | "Last.fm")
API key (AcoustID / Last.fm)       ``metadata_api_key``
MusicBrainz app / version / mail   ``musicbrainz_useragent`` = {app, version, contact}
Default library                    ``library_root``
Near-duplicate threshold           ``near_duplicate_threshold``
Exact-duplicate threshold          ``exact_duplicate_threshold``
Background gradient                ``bg_gradient_enabled``
=================================  ==============================================
"""
from __future__ import annotations

import os

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot

# Services whose API key is the shared ``metadata_api_key`` field.
_KEYED_SERVICES = ("AcoustID", "Last.fm")


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


class SettingsDrawer(QtWidgets.QDialog):
    """Modal settings dialog for metadata services and global preferences."""

    settings_saved = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self.setMinimumHeight(500)
        self._loaded_library = ""
        # True after Save when the default library was changed in this dialog.
        self.library_changed = False
        self._build_ui()
        self._load_config()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header (theme-aware: follows the current palette, light or dark)
        try:
            from gui.themes.manager import get_manager
            t = get_manager().current
            hdr_bg, hdr_fg, hdr_muted = t.sidebar_bg, t.text_primary, t.text_secondary
            self._ok_colour, self._err_colour = t.success, t.danger
        except Exception:
            hdr_bg, hdr_fg, hdr_muted = "palette(window)", "palette(text)", "palette(mid)"
            self._ok_colour, self._err_colour = "green", "red"
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

        # Tabs
        tabs = QtWidgets.QTabWidget()
        tabs.setContentsMargins(16, 16, 16, 16)

        # ── Metadata Services tab ──────────────────────────────────────────
        meta_w = QtWidgets.QWidget()
        meta_l = QtWidgets.QFormLayout(meta_w)
        meta_l.setContentsMargins(20, 20, 20, 20)
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

        # API-key section (AcoustID and Last.fm share ``metadata_api_key``)
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
        tabs.addTab(meta_w, "Metadata Services")

        # ── General tab ────────────────────────────────────────────────────
        gen_w = QtWidgets.QWidget()
        gen_l = QtWidgets.QFormLayout(gen_w)
        gen_l.setContentsMargins(20, 20, 20, 20)
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
        # value is never silently clamped when this dialog re-saves it.
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

        self._bg_gradient_cb = QtWidgets.QCheckBox(
            "Background gradient — subtle accent glow behind workspace panels"
        )
        self._bg_gradient_cb.setChecked(True)
        self._bg_gradient_cb.setToolTip(
            "Paints two faint radial glows (accent colour) in opposing corners "
            "of each workspace.  Uncheck for a flat, solid background."
        )
        gen_l.addRow("Appearance:", self._bg_gradient_cb)

        tabs.addTab(gen_w, "General")

        # ── MusicBrainz User-Agent ─────────────────────────────────────────
        # (already in Metadata Services; stub tab for extended config)
        tabs.addTab(QtWidgets.QWidget(), "Advanced")

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

        self._on_service_changed(self._service_combo.currentText())

    # ── Private ───────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        try:
            from config import load_config
            import tag_fixer
            cfg = load_config()
            svc = cfg.get("metadata_service", "AcoustID")
            idx = self._service_combo.findText(svc)
            if idx >= 0:
                self._service_combo.setCurrentIndex(idx)
            self._api_key_entry.setText(
                cfg.get("metadata_api_key", getattr(tag_fixer, "ACOUSTID_API_KEY", ""))
            )
            ua = cfg.get("musicbrainz_useragent", {})
            self._mb_app.setText(ua.get("app", ""))
            self._mb_ver.setText(ua.get("version", ""))
            self._mb_contact.setText(ua.get("contact", ""))
            self._loaded_library = str(cfg.get("library_root", "") or "")
            self._library_entry.setText(self._loaded_library)
            self._near_dup_spin.setValue(float(cfg.get("near_duplicate_threshold", 0.1)))
            self._exact_dup_spin.setValue(float(cfg.get("exact_duplicate_threshold", 0.02)))
            self._bg_gradient_cb.setChecked(bool(cfg.get("bg_gradient_enabled", True)))
        except Exception:
            pass
        self._on_service_changed(self._service_combo.currentText())

    @Slot()
    def _on_save(self) -> None:
        library = self._library_entry.text().strip()
        changed = library != self._loaded_library
        # Only a library typed/browsed here is validated: an unchanged saved
        # path that has gone missing must not block saving other settings.
        if changed and library and not os.path.isdir(os.path.expanduser(library)):
            QtWidgets.QMessageBox.warning(
                self,
                "Library folder not found",
                f"The default library folder\n{library}\ndoes not exist. "
                "Choose an existing folder, or clear the field.",
            )
            return
        try:
            from config import load_config, save_config
            cfg = load_config()
            cfg["metadata_service"] = self._service_combo.currentText()
            cfg["metadata_api_key"] = self._api_key_entry.text()
            cfg["musicbrainz_useragent"] = {
                "app": self._mb_app.text(),
                "version": self._mb_ver.text(),
                "contact": self._mb_contact.text(),
            }
            cfg["library_root"] = library
            cfg["near_duplicate_threshold"] = self._near_dup_spin.value()
            cfg["exact_duplicate_threshold"] = self._exact_dup_spin.value()
            cfg["bg_gradient_enabled"] = self._bg_gradient_cb.isChecked()
            save_config(cfg)
            self.library_changed = changed
            self.settings_saved.emit()
            self.accept()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save Error", str(exc))

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
                pass  # the dialog was closed and deleted meanwhile

        import threading
        threading.Thread(target=_run, daemon=True, name="SettingsConnectionTest").start()

    @Slot(bool, str)
    def _apply_test_result(self, ok: bool, msg: str) -> None:
        self._test_btn.setEnabled(True)
        self._test_status.setText(msg)
        colour = self._ok_colour if ok else self._err_colour
        self._test_status.setStyleSheet(f"color: {colour};")

    @Slot(str)
    def _on_service_changed(self, svc: str) -> None:
        self._acoustid_frame.setVisible(svc in _KEYED_SERVICES)
        self._api_key_entry.setPlaceholderText(
            "Your Last.fm API key" if svc == "Last.fm" else "Your AcoustID API key"
        )
        self._mb_frame.setVisible(svc == "MusicBrainz")

    @Slot()
    def _on_browse_library(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Default Library Folder", self._library_entry.text()
        )
        if folder:
            self._library_entry.setText(folder)
