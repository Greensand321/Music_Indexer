"""Settings controls and their load/save logic — metadata services, default
library, thresholds, motion, backdrop.

``SettingsForm`` owns every control and the save logic; the Settings *page*
(``gui/workspaces/settings.py``) lays its groups out as glass cards in three
sections (Metadata services · General · Appearance). ``SettingsDrawer`` is
the same form in a modal dialog (the app no longer opens it: Ctrl+, and the
top bar's Settings button go to the page).

Every control persists through ``config.load_config()/save_config()`` under
these keys (other pages read them, so the names must not change):

=================================  ==============================================
Control                            Config key
=================================  ==============================================
Service rows                       ``metadata_service`` ("AcoustID" | "MusicBrainz" | "Last.fm")
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

Two controls keep a ``QComboBox``-like API although they are not combo boxes
any more, so callers and tests drive them the same way: the service rows
(``ServiceList``: ``currentText`` / ``setCurrentText`` / ``findText`` /
``currentTextChanged``) and the backdrop's segmented pill control
(``SegmentedChoice``: ``findData`` / ``currentData`` / ``setCurrentIndex``).
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

# One line under each service's name.
SERVICE_NOTES = {
    "AcoustID": "Fingerprints the audio; adds MusicBrainz album and genre",
    "MusicBrainz": "Also looks tracks up by artist and title",
    "Last.fm": "Genre tags; needs your own Last.fm key",
}
# What the Connection card says for the selected service.
CONNECTION_NOTES = {
    "AcoustID": "An empty key uses the built-in key.",
    "Last.fm": "Your own Last.fm API key (the LASTFM_API_KEY environment variable wins).",
    "MusicBrainz": "MusicBrainz asks every app to say who it is.",
}

# Card titles and one-line subtitles for the form's groups, per page section
# (the page and the dialog share them).
SECTION_GROUPS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "services": (
        ("service_widget", "Lookup service",
         "Where the Tag Fixer and the genre tools look tracks up."),
        ("connection_widget", "Connection", ""),
    ),
    "general": (
        ("library_widget", "Default library", "Opens when AlphaDEX starts."),
        ("thresholds_widget", "Duplicate thresholds",
         "Where the Duplicates page starts. Lower is stricter."),
        ("motion_widget", "Motion", ""),
    ),
    "appearance": (
        ("backdrop_widget", "Backdrop", "What shows behind the glass. Applies when you save."),
    ),
}


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


# ── Service rows ──────────────────────────────────────────────────────────────

class _ServiceRow(QtWidgets.QAbstractButton):
    """One selectable service: a radio dot, the name and a muted note, on a
    glass list row (the raised surface when chosen). An unavailable service
    is disabled; its tooltip says why."""

    HEIGHT = 48

    def __init__(self, name: str, note: str, available: bool,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("serviceRow")
        self.setText(name)
        self._note = note
        self.setCheckable(True)
        self.setEnabled(available)
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        if available:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setAccessibleName(name)
        self.setAccessibleDescription(note)

    def note(self) -> str:
        return self._note

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(260, self.HEIGHT)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(160, self.HEIGHT)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, font, paint_surface, qcolor
        from gui.themes.manager import get_manager
        from gui.themes.tokens import RADIUS, Surface

        t = get_manager().current
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = QtCore.QRectF(self.rect())
            enabled = self.isEnabled()
            if self.isChecked():
                paint_surface(p, rect, m.raised, radius=RADIUS.sm)
            else:
                paint_surface(p, rect, Surface(m.well.fill, radius=RADIUS.sm),
                              opacity=1.0 if enabled else 0.6)
                if enabled and self.underMouse():
                    path = QtGui.QPainterPath()
                    path.addRoundedRect(rect, RADIUS.sm, RADIUS.sm)
                    p.fillPath(path, qcolor(m.hover))
            if self.hasFocus():
                p.setPen(QtGui.QPen(qcolor(t.input_focus), 1.5))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), RADIUS.sm - 1, RADIUS.sm - 1)

            # Radio dot
            cy = rect.center().y()
            dot = QtCore.QRectF(14, cy - 9, 18, 18)
            if self.isChecked():
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(qcolor(t.accent))
                p.drawEllipse(dot)
                p.setBrush(qcolor(t.text_inverse))
                p.drawEllipse(dot.adjusted(5.5, 5.5, -5.5, -5.5))
            else:
                p.setPen(QtGui.QPen(qcolor(m.raised.border if m.is_glass else t.input_border), 1.2))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(dot.adjusted(0.5, 0.5, -0.5, -0.5))

            # Name over note
            x = dot.right() + 12
            text_w = rect.width() - x - 14
            name_f, note_f = font("body", weight=600), font("caption")
            nh = QtGui.QFontMetrics(name_f).height()
            ch = QtGui.QFontMetrics(note_f).height()
            top = cy - (nh + ch) / 2
            p.setFont(name_f)
            p.setPen(qcolor(t.text_primary if enabled else t.text_muted))
            p.drawText(QtCore.QRectF(x, top, text_w, nh),
                       int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                       QtGui.QFontMetrics(name_f).elidedText(
                           self.text(), QtCore.Qt.TextElideMode.ElideRight, int(text_w)))
            p.setFont(note_f)
            p.setPen(qcolor(t.text_secondary if enabled else t.text_muted))
            p.drawText(QtCore.QRectF(x, top + nh, text_w, ch),
                       int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                       QtGui.QFontMetrics(note_f).elidedText(
                           self._note, QtCore.Qt.TextElideMode.ElideRight, int(text_w)))
        finally:
            p.end()


class ServiceList(QtWidgets.QWidget):
    """The metadata services as selectable glass rows, with the
    ``QComboBox`` API the form and its callers use (``currentText``,
    ``setCurrentText``, ``findText``, ``setCurrentIndex``, ``currentIndex``,
    ``count``, ``itemText``, ``isItemEnabled`` and ``currentTextChanged``)."""

    currentTextChanged = Signal(str)   # noqa: N815 - mirrors QComboBox

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("serviceList")
        self._rows: list[_ServiceRow] = []
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idToggled.connect(self._on_toggled)
        self._lay = QtWidgets.QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)

    def addItem(self, name: str, note: str = "", available: bool = True,  # noqa: N802
                tooltip: str = "") -> _ServiceRow:
        row = _ServiceRow(name, note, available)
        if tooltip:
            row.setToolTip(tooltip)
        self._group.addButton(row, len(self._rows))
        self._rows.append(row)
        self._lay.addWidget(row)
        return row

    def rows(self) -> list[_ServiceRow]:
        return list(self._rows)

    def count(self) -> int:
        return len(self._rows)

    def itemText(self, index: int) -> str:  # noqa: N802
        return self._rows[index].text() if 0 <= index < len(self._rows) else ""

    def isItemEnabled(self, index: int) -> bool:  # noqa: N802
        return 0 <= index < len(self._rows) and self._rows[index].isEnabled()

    def findText(self, text: str) -> int:  # noqa: N802
        for i, row in enumerate(self._rows):
            if row.text() == text:
                return i
        return -1

    def currentIndex(self) -> int:  # noqa: N802
        return self._group.checkedId()

    def currentText(self) -> str:  # noqa: N802
        return self.itemText(self.currentIndex())

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        if 0 <= index < len(self._rows) and self._rows[index].isEnabled():
            self._rows[index].setChecked(True)

    def setCurrentText(self, text: str) -> None:  # noqa: N802
        self.setCurrentIndex(self.findText(text))

    def _on_toggled(self, index: int, checked: bool) -> None:
        if checked:
            self.currentTextChanged.emit(self.itemText(index))


# ── Segmented pill choice ─────────────────────────────────────────────────────

def _segmented_choice_class():
    from gui.pages import SegmentedTabBar

    class SegmentedChoice(SegmentedTabBar):
        """A few mutually exclusive choices as the app's glass segmented pill
        control (the page tabs' look and sliding thumb), with the
        ``QComboBox`` data API (``addItem(text, data)``, ``findData``,
        ``currentData``; ``setCurrentIndex`` / ``currentIndex`` are the tab
        bar's own)."""

        def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.setObjectName("segmentedChoice")   # not the page's section tabs
            self.setExpanding(False)
            self.setUsesScrollButtons(False)
            self.setElideMode(QtCore.Qt.TextElideMode.ElideNone)
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
            from gui.themes.glass import font
            self.setFont(font("label", weight=500))

        def addItem(self, text: str, data=None) -> int:  # noqa: N802
            index = self.addTab(text)
            self.setTabData(index, data)
            return index

        def findData(self, data) -> int:  # noqa: N802
            for i in range(self.count()):
                if self.tabData(i) == data:
                    return i
            return -1

        def currentData(self):  # noqa: N802
            index = self.currentIndex()
            return self.tabData(index) if index >= 0 else None

    return SegmentedChoice


# ── Small parts ───────────────────────────────────────────────────────────────

class _HelpDot(QtWidgets.QWidget):
    """A small painted "info" glyph that carries a help tooltip (not a
    button: hovering it is the whole interaction)."""

    def __init__(self, tip: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("helpDot")
        self.setFixedSize(20, 20)
        self.setToolTip(tip)
        self.setAccessibleName("Help")
        self.setAccessibleDescription(tip)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager
        from gui.widgets.glyphs import paint_glyph

        p = QtGui.QPainter(self)
        try:
            paint_glyph(p, QtCore.QRectF(2, 2, 16, 16), "info", get_manager().current.text_secondary)
        finally:
            p.end()


def _field_label(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("mutedLabel")
    return lbl


def _vbox(widget: QtWidgets.QWidget, spacing: int = 10) -> QtWidgets.QVBoxLayout:
    lay = QtWidgets.QVBoxLayout(widget)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    return lay


# ── The form ──────────────────────────────────────────────────────────────────

class SettingsForm(QtCore.QObject):
    """Every Settings control, plus load / save.

    The controls live in group widgets the host lays out (``SECTION_GROUPS``
    names them with a card title and subtitle): ``service_widget`` (the
    service rows), ``connection_widget`` (keys, MusicBrainz user agent,
    connection test), ``library_widget``, ``thresholds_widget``,
    ``motion_widget`` and ``backdrop_widget``. ``save()`` validates, writes
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
        self.service_widget = self._build_service()
        self.connection_widget = self._build_connection()
        self.library_widget = self._build_library()
        self.thresholds_widget = self._build_thresholds()
        self.motion_widget = self._build_motion()
        self.backdrop_widget = self._build_backdrop()
        self._load_config()
        from gui.themes.manager import get_manager
        get_manager().theme_changed.connect(self._on_theme_changed)

    def _parent_widget(self) -> QtWidgets.QWidget | None:
        return self._host

    def group(self, name: str) -> QtWidgets.QWidget:
        return getattr(self, name)

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_service(self) -> QtWidgets.QWidget:
        try:
            from config import SUPPORTED_SERVICES, UNAVAILABLE_SERVICES
        except ImportError:
            SUPPORTED_SERVICES = ["AcoustID", "MusicBrainz", "Last.fm"]
            UNAVAILABLE_SERVICES = {"Spotify", "Gracenote"}

        self._service_combo = ServiceList()
        for svc in SUPPORTED_SERVICES:
            if svc in UNAVAILABLE_SERVICES:
                self._service_combo.addItem(
                    svc, "Not available yet", available=False,
                    tooltip=f"{svc} is not yet implemented. See ROADMAP.md.",
                )
            else:
                self._service_combo.addItem(svc, SERVICE_NOTES.get(svc, ""))
        self._service_combo.currentTextChanged.connect(self._on_service_changed)
        return self._service_combo

    def _build_connection(self) -> QtWidgets.QWidget:
        conn = QtWidgets.QWidget()
        lay = _vbox(conn, 12)
        self._connection_hint = QtWidgets.QLabel("")
        self._connection_hint.setObjectName("mutedLabel")
        self._connection_hint.setWordWrap(True)
        lay.addWidget(self._connection_hint)

        # API-key section: one field showing the selected service's own key
        self._acoustid_frame = QtWidgets.QFrame()
        af = QtWidgets.QGridLayout(self._acoustid_frame)
        af.setContentsMargins(0, 0, 0, 0)
        af.setHorizontalSpacing(14)
        self._api_key_label = _field_label("API key")
        self._api_key_entry = QtWidgets.QLineEdit()
        self._api_key_entry.setPlaceholderText("Your AcoustID API key")
        self._api_key_entry.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._api_key_entry.setAccessibleName("API key")
        af.addWidget(self._api_key_label, 0, 0)
        af.addWidget(self._api_key_entry, 0, 1)
        af.setColumnStretch(1, 1)
        af.setColumnMinimumWidth(0, 84)
        lay.addWidget(self._acoustid_frame)

        # MusicBrainz section
        self._mb_frame = QtWidgets.QFrame()
        mb = QtWidgets.QGridLayout(self._mb_frame)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.setHorizontalSpacing(14)
        mb.setVerticalSpacing(10)
        self._mb_app = QtWidgets.QLineEdit()
        self._mb_app.setPlaceholderText("AlphaDEX")
        self._mb_ver = QtWidgets.QLineEdit()
        self._mb_ver.setPlaceholderText("1.0")
        self._mb_contact = QtWidgets.QLineEdit()
        self._mb_contact.setPlaceholderText("your@email.com")
        for row, (label, edit) in enumerate((("App name", self._mb_app),
                                             ("Version", self._mb_ver),
                                             ("Contact", self._mb_contact))):
            edit.setAccessibleName(f"MusicBrainz {label.lower()}")
            mb.addWidget(_field_label(label), row, 0)
            mb.addWidget(edit, row, 1)
        mb.setColumnStretch(1, 1)
        mb.setColumnMinimumWidth(0, 84)
        lay.addWidget(self._mb_frame)

        # Test + status
        test_row = QtWidgets.QHBoxLayout()
        test_row.setSpacing(12)
        self._test_btn = QtWidgets.QPushButton("Test connection")
        self._test_btn.setToolTip("Try the service with what is entered above (nothing is saved)")
        from gui.widgets.glyphs import set_glyph
        set_glyph(self._test_btn, "refresh")
        self._test_btn.clicked.connect(self._on_test)
        self._test_status = QtWidgets.QLabel("")
        self._test_status.setObjectName("settingsTestStatus")
        self._test_status.setWordWrap(True)
        test_row.addWidget(self._test_btn)
        test_row.addWidget(self._test_status, 1)
        lay.addLayout(test_row)
        return conn

    def _build_library(self) -> QtWidgets.QWidget:
        from gui.widgets.path_field import PathField

        self._library_entry = PathField(
            "No default library: AlphaDEX asks at start-up",
            button_tip="Choose the default library folder", clearable=True,
        )
        self._library_entry.setAccessibleName("Default library")
        self._library_entry.browse_requested.connect(self._on_browse_library)
        return self._library_entry

    def _threshold_row(self, grid: QtWidgets.QGridLayout, row: int, label: str,
                       spin: QtWidgets.QDoubleSpinBox, tip: str) -> None:
        name = _field_label(label)
        name.setToolTip(tip)
        spin.setToolTip(tip)
        spin.setAccessibleName(label)
        spin.setAccessibleDescription(tip)
        spin.setMinimumWidth(120)   # the size hint clipped "0.0200"
        unit = _field_label("fingerprint distance, 0–1")
        unit.setToolTip(tip)
        grid.addWidget(name, row, 0)
        grid.addWidget(spin, row, 1)
        grid.addWidget(unit, row, 2)
        grid.addWidget(_HelpDot(tip), row, 3, QtCore.Qt.AlignmentFlag.AlignVCenter)

    def _build_thresholds(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        # Both ranges match the Duplicates page's spin boxes (0–1), so a saved
        # value is never silently clamped when this form re-saves it.
        self._near_dup_spin = QtWidgets.QDoubleSpinBox()
        self._near_dup_spin.setRange(0.0, 1.0)
        self._near_dup_spin.setSingleStep(0.01)
        self._near_dup_spin.setDecimals(3)
        self._near_dup_spin.setValue(0.1)
        self._threshold_row(
            grid, 0, "Near duplicate", self._near_dup_spin,
            "Two files whose fingerprints differ by at most this much are the same song in "
            "a different form (a remaster, another rip or encoding). Default 0.10.")

        self._exact_dup_spin = QtWidgets.QDoubleSpinBox()
        self._exact_dup_spin.setRange(0.0, 1.0)
        self._exact_dup_spin.setSingleStep(0.005)
        self._exact_dup_spin.setDecimals(4)
        self._exact_dup_spin.setValue(0.02)
        self._threshold_row(
            grid, 1, "Exact duplicate", self._exact_dup_spin,
            "At most this much fingerprint distance counts as the identical recording, "
            "just a different file. Default 0.02.")
        grid.setColumnStretch(4, 1)   # unit and help sit next to the value
        return box

    def _build_motion(self) -> QtWidgets.QWidget:
        self._reduce_motion_cb = QtWidgets.QCheckBox("Reduce motion")
        self._reduce_motion_cb.setToolTip(
            "No slides, fades or fly-ins: animations jump straight to their end "
            "(the ALPHADEX_REDUCED_MOTION environment variable does the same). "
            "Applies when you save."
        )
        box = QtWidgets.QWidget()
        lay = _vbox(box, 4)
        lay.addWidget(self._reduce_motion_cb)
        note = QtWidgets.QLabel("No slides, fades or fly-ins. Applies when you save.")
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        note.setContentsMargins(26, 0, 0, 0)   # under the checkbox text
        lay.addWidget(note)
        return box

    def _build_backdrop(self) -> QtWidgets.QWidget:
        from gui.themes.appearance import BACKDROP_LABELS, BACKDROP_MODES

        self._backdrop_combo = _segmented_choice_class()()
        for mode in BACKDROP_MODES:
            self._backdrop_combo.addItem(BACKDROP_LABELS[mode], mode)
        tips = {
            "art": "A blurred wall of covers from your library (Liquid Glass; classic "
                   "themes show their gradient)",
            "gradient": "Soft accent glows",
            "plain": "A solid colour",
        }
        for i, mode in enumerate(BACKDROP_MODES):
            self._backdrop_combo.setTabToolTip(i, tips[mode])
        self._backdrop_combo.setAccessibleName("Backdrop")
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._backdrop_combo)
        lay.addStretch(1)
        return box

    def backdrop_mode(self) -> str:
        return str(self._backdrop_combo.currentData() or "art")

    # ── Theme ─────────────────────────────────────────────────────────────

    def _on_theme_changed(self, tokens=None) -> None:
        from gui.workspaces.label_tone import refresh_tones
        try:
            refresh_tones(self._test_status, tokens)
        except RuntimeError:
            pass  # the host was deleted

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
            if idx < 0 or not self._service_combo.isItemEnabled(idx):
                idx = self._service_combo.findText("AcoustID")
            self._service_combo.setCurrentIndex(max(idx, 0))
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
        from gui.workspaces.label_tone import set_tone

        self._test_status.setText("Testing…")
        set_tone(self._test_status, "secondary")
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
        from gui.workspaces.label_tone import set_tone

        self._test_btn.setEnabled(True)
        self._test_status.setText(msg)
        set_tone(self._test_status, "success" if ok else "danger")

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
        self._api_key_label.setText(f"{svc} key" if svc in _KEY_CONFIG else "API key")
        self._connection_hint.setText(CONNECTION_NOTES.get(svc, ""))
        self._mb_frame.setVisible(svc == "MusicBrainz")
        self._test_status.setText("")

    @Slot()
    def _on_browse_library(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self._parent_widget(), "Select Default Library Folder", self._library_entry.text()
        )
        if folder:
            self._library_entry.setText(folder)


# ── The dialog ────────────────────────────────────────────────────────────────

class SettingsDrawer(QtWidgets.QDialog):
    """``SettingsForm`` in a modal dialog (tabs: Metadata services · General).
    The app shows the Settings page instead; this stays a thin wrapper with
    the same controls, cards and dialog chrome."""

    settings_saved = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        self.setMinimumHeight(520)
        self.form = SettingsForm(self)
        for name in SettingsForm.WIDGET_NAMES:
            setattr(self, name, getattr(self.form, name))
        self.form.settings_saved.connect(self.settings_saved)
        self._build_ui()

    @property
    def library_changed(self) -> bool:
        return self.form.library_changed

    def _card(self, title: str, subtitle: str, body: QtWidgets.QWidget) -> QtWidgets.QWidget:
        from gui.widgets.glass import GlassCard

        card = GlassCard()
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(10)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("cardTitle")
        lay.addWidget(heading)
        if subtitle:
            sub = QtWidgets.QLabel(subtitle)
            sub.setObjectName("mutedLabel")
            sub.setWordWrap(True)
            lay.addWidget(sub)
        lay.addWidget(body)
        return card

    def _page(self, *sections: str) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)
        lay.setContentsMargins(2, 12, 2, 4)
        lay.setSpacing(12)
        for section in sections:
            for attr, title, subtitle in SECTION_GROUPS[section]:
                lay.addWidget(self._card(title, subtitle, self.form.group(attr)))
        lay.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_ui(self) -> None:
        from gui.themes.glass import apply_font

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        title = QtWidgets.QLabel("Settings")
        title.setObjectName("dialogTitle")
        apply_font(title, "title")
        root.addWidget(title)

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._page("services"), "Metadata services")
        tabs.addTab(self._page("general", "appearance"), "General")
        root.addWidget(tabs, 1)

        buttons = QtWidgets.QDialogButtonBox()
        cancel_btn = buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setAutoDefault(False)
        self._save_btn = buttons.addButton("Save", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self._save_btn.setObjectName("primaryBtn")
        self._save_btn.setDefault(True)
        cancel_btn.clicked.connect(self.reject)
        self._save_btn.clicked.connect(self._on_save)
        root.addWidget(buttons)
        # Enter saves: no other button may become the dialog's default when
        # it takes focus (Test connection, the path buttons).
        for btn in self.findChildren(QtWidgets.QPushButton):
            if btn is not self._save_btn:
                btn.setAutoDefault(False)

    @Slot()
    def _on_save(self) -> None:
        if self.form.save():
            self.accept()

    @Slot()
    def _on_test(self) -> None:
        self.form._on_test()
