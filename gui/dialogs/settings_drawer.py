"""Settings drawer — slides in from the right, covers metadata service and config."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot


class SettingsDrawer(QtWidgets.QDialog):
    """Modal settings dialog for metadata services and global preferences."""

    settings_saved = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self.setMinimumHeight(500)
        self._build_ui()
        self._load_config()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QtWidgets.QFrame()
        header.setStyleSheet("background: #0f172a; padding: 16px;")
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(20, 16, 20, 16)
        title = QtWidgets.QLabel("Settings")
        title.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: 700;")
        h_layout.addWidget(title)
        h_layout.addStretch(1)
        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(
            "background: transparent; border: none; color: #94a3b8; font-size: 16px;"
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
            from config import SUPPORTED_SERVICES
            services = [s for s in SUPPORTED_SERVICES if s in ("AcoustID", "MusicBrainz")]
        except ImportError:
            services = ["AcoustID", "MusicBrainz"]

        self._service_combo = QtWidgets.QComboBox()
        self._service_combo.addItems(services)
        self._service_combo.currentTextChanged.connect(self._on_service_changed)
        meta_l.addRow("Service:", self._service_combo)

        # AcoustID section
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

        self._near_dup_spin = QtWidgets.QDoubleSpinBox()
        self._near_dup_spin.setRange(0.0, 1.0)
        self._near_dup_spin.setSingleStep(0.01)
        self._near_dup_spin.setDecimals(3)
        self._near_dup_spin.setValue(0.1)
        gen_l.addRow("Near-duplicate threshold:", self._near_dup_spin)

        self._exact_dup_spin = QtWidgets.QDoubleSpinBox()
        self._exact_dup_spin.setRange(0.0, 0.1)
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
            self._library_entry.setText(cfg.get("library_root", ""))
            self._near_dup_spin.setValue(cfg.get("near_duplicate_threshold", 0.1))
            self._exact_dup_spin.setValue(cfg.get("exact_duplicate_threshold", 0.02))
            self._bg_gradient_cb.setChecked(bool(cfg.get("bg_gradient_enabled", True)))
        except Exception:
            pass

    @Slot()
    def _on_save(self) -> None:
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
            cfg["library_root"] = self._library_entry.text()
            cfg["near_duplicate_threshold"] = self._near_dup_spin.value()
            cfg["exact_duplicate_threshold"] = self._exact_dup_spin.value()
            cfg["bg_gradient_enabled"] = self._bg_gradient_cb.isChecked()
            save_config(cfg)
            self.settings_saved.emit()
            self.accept()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save Error", str(exc))

    @Slot()
    def _on_test(self) -> None:
        self._test_status.setText("Testing…")
        self._test_btn.setEnabled(False)

        svc = self._service_combo.currentText()

        def _run() -> None:
            try:
                if svc == "AcoustID":
                    import requests
                    requests.get("https://api.acoustid.org/v2/", timeout=5)
                    ok, msg = True, "AcoustID: OK"
                elif svc == "MusicBrainz":
                    import musicbrainzngs
                    musicbrainzngs.set_useragent(
                        self._mb_app.text() or "AlphaDEX",
                        self._mb_ver.text() or "1.0",
                        self._mb_contact.text(),
                    )
                    musicbrainzngs.search_artists(query="Beatles", limit=1)
                    ok, msg = True, "MusicBrainz: OK"
                else:
                    ok, msg = False, "Unknown service"
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, str(exc)

            def _done() -> None:
                self._test_btn.setEnabled(True)
                self._test_status.setText(msg)
                colour = "#22c55e" if ok else "#ef4444"
                self._test_status.setStyleSheet(f"color: {colour};")

            QtCore.QMetaObject.invokeMethod(
                self, "_apply_test_result",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(bool, ok),
                QtCore.Q_ARG(str, msg),
            )

        import threading
        threading.Thread(target=_run, daemon=True).start()

    @Slot(bool, str)
    def _apply_test_result(self, ok: bool, msg: str) -> None:
        self._test_btn.setEnabled(True)
        self._test_status.setText(msg)
        colour = "#22c55e" if ok else "#ef4444"
        self._test_status.setStyleSheet(f"color: {colour};")

    @Slot(str)
    def _on_service_changed(self, svc: str) -> None:
        self._acoustid_frame.setVisible(svc == "AcoustID")
        self._mb_frame.setVisible(svc == "MusicBrainz")

    @Slot()
    def _on_browse_library(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Default Library Folder", self._library_entry.text()
        )
        if folder:
            self._library_entry.setText(folder)
