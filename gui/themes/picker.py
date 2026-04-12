"""ThemePickerDialog and AutoThemeDialog — runtime theme selection UI."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.tokens import THEMES, DARK_THEMES, LIGHT_THEMES, ThemeTokens
from gui.themes.manager import get_manager


# ── Swatch card ───────────────────────────────────────────────────────────────

class _SwatchCard(QtWidgets.QAbstractButton):
    """Single theme swatch: coloured preview block + name label."""

    SIZE = QtCore.QSize(120, 88)

    def __init__(self, tokens: ThemeTokens, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tokens = tokens
        self._selected = False
        self.setFixedSize(self.SIZE)
        self.setCheckable(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setToolTip(tokens.name)

    @property
    def tokens(self) -> ThemeTokens:
        return self._tokens

    def setSelected(self, val: bool) -> None:
        self._selected = val
        self.setChecked(val)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        t = self._tokens
        mgr = get_manager()
        current = mgr.current

        r = self.rect()

        # ── Card background ───────────────────────────────────────────────
        card_r = r.adjusted(4, 4, -4, -4)
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(card_r), 10, 10)

        if self._selected or self.isChecked():
            # Accent border glow
            p.setPen(QtGui.QPen(QtGui.QColor(current.accent), 2.5))
        else:
            p.setPen(QtGui.QPen(QtGui.QColor(current.card_border), 1.0))

        p.setBrush(QtGui.QColor(t.content_bg))
        p.drawPath(path)

        # ── Preview rows (sidebar strip + content area) ───────────────────
        inner = card_r.adjusted(6, 6, -6, -6)

        # Sidebar strip (left 28px)
        sb_rect = QtCore.QRectF(inner.left(), inner.top(), 28, inner.height())
        sb_path = QtGui.QPainterPath()
        sb_path.addRoundedRect(sb_rect, 5, 5)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(t.sidebar_bg))
        p.drawPath(sb_path)

        # Sidebar accent dots
        dot_x = sb_rect.left() + sb_rect.width() / 2
        for i, colour in enumerate([t.sidebar_accent, t.accent, t.text_muted]):
            dot_y = sb_rect.top() + 8 + i * 10
            p.setBrush(QtGui.QColor(colour))
            p.drawEllipse(QtCore.QPointF(dot_x, dot_y), 3, 3)

        # Content area
        ct_rect = QtCore.QRectF(inner.left() + 32, inner.top(), inner.width() - 32, inner.height())

        # Top bar strip
        tb_rect = QtCore.QRectF(ct_rect.left(), ct_rect.top(), ct_rect.width(), 14)
        tb_path = QtGui.QPainterPath()
        tb_path.addRoundedRect(tb_rect, 4, 4)
        p.setBrush(QtGui.QColor(t.card_bg))
        p.drawPath(tb_path)

        # Accent button mock
        btn_w = 24
        btn_rect = QtCore.QRectF(ct_rect.right() - btn_w - 2, ct_rect.top() + 2, btn_w, 10)
        btn_path = QtGui.QPainterPath()
        btn_path.addRoundedRect(btn_rect, 3, 3)
        p.setBrush(QtGui.QColor(t.accent))
        p.drawPath(btn_path)

        # Card mock below top bar
        card_mock = QtCore.QRectF(ct_rect.left(), ct_rect.top() + 18, ct_rect.width(), ct_rect.height() - 18)
        cm_path = QtGui.QPainterPath()
        cm_path.addRoundedRect(card_mock, 4, 4)
        p.setBrush(QtGui.QColor(t.card_bg))
        p.drawPath(cm_path)

        # Text lines mock
        p.setBrush(QtGui.QColor(t.text_primary))
        line_rect = QtCore.QRectF(card_mock.left() + 4, card_mock.top() + 5, card_mock.width() * 0.7, 3)
        lp = QtGui.QPainterPath()
        lp.addRoundedRect(line_rect, 1.5, 1.5)
        p.drawPath(lp)

        p.setBrush(QtGui.QColor(t.text_muted))
        line_rect2 = QtCore.QRectF(card_mock.left() + 4, card_mock.top() + 12, card_mock.width() * 0.5, 3)
        lp2 = QtGui.QPainterPath()
        lp2.addRoundedRect(line_rect2, 1.5, 1.5)
        p.drawPath(lp2)

        # ── Theme name label ──────────────────────────────────────────────
        label_rect = QtCore.QRect(card_r.left(), card_r.bottom() - 20, card_r.width(), 20)
        p.setPen(QtGui.QColor(current.text_primary))
        font = p.font()
        font.setPointSize(8)
        font.setBold(self._selected or self.isChecked())
        p.setFont(font)
        p.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignCenter, t.name)

        # ── Checkmark if selected ─────────────────────────────────────────
        if self._selected or self.isChecked():
            ck_size = 16
            ck_rect = QtCore.QRectF(
                card_r.right() - ck_size - 2,
                card_r.top() + 2,
                ck_size,
                ck_size,
            )
            ck_path = QtGui.QPainterPath()
            ck_path.addEllipse(ck_rect)
            p.setBrush(QtGui.QColor(current.accent))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawPath(ck_path)

            # Draw checkmark
            pen = QtGui.QPen(QtGui.QColor(current.text_inverse), 1.5)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            cx, cy = ck_rect.center().x(), ck_rect.center().y()
            p.drawLine(QtCore.QPointF(cx - 3.5, cy), QtCore.QPointF(cx - 1, cy + 2.5))
            p.drawLine(QtCore.QPointF(cx - 1, cy + 2.5), QtCore.QPointF(cx + 3.5, cy - 2.5))

        p.end()


# ── Section header label ──────────────────────────────────────────────────────

def _section_label(text: str, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text, parent)
    lbl.setObjectName("sectionHeader")
    font = lbl.font()
    font.setPointSize(9)
    font.setBold(True)
    lbl.setFont(font)
    return lbl


# ── AutoThemeDialog ───────────────────────────────────────────────────────────

class AutoThemeDialog(QtWidgets.QDialog):
    """Configure the dark / light pair for Auto mode."""

    pair_selected = Signal(str, str)   # dark_key, light_key

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto Theme — Select Pair")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint
        )

        mgr = get_manager()
        self._dark_key = mgr.auto_dark()
        self._light_key = mgr.auto_light()

        self._dark_swatches: list[_SwatchCard] = []
        self._light_swatches: list[_SwatchCard] = []

        self._build_ui()
        self._refresh_selections()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # Description
        desc = QtWidgets.QLabel(
            "Auto mode switches between your chosen dark and light themes "
            "based on the system day/night cycle (or time of day if unavailable)."
        )
        desc.setWordWrap(True)
        desc.setObjectName("mutedLabel")
        root.addWidget(desc)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(inner)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(12)

        # Dark section
        vbox.addWidget(_section_label("Night theme (dark)"))
        dark_grid = self._make_swatch_grid(DARK_THEMES, self._dark_swatches, is_dark=True)
        vbox.addLayout(dark_grid)

        vbox.addSpacing(8)

        # Light section
        vbox.addWidget(_section_label("Day theme (light)"))
        light_grid = self._make_swatch_grid(LIGHT_THEMES, self._light_swatches, is_dark=False)
        vbox.addLayout(light_grid)

        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        ok_btn = QtWidgets.QPushButton("Apply Auto Theme")
        ok_btn.setObjectName("primaryBtn")
        cancel_btn.clicked.connect(self.reject)
        ok_btn.clicked.connect(self._accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        root.addLayout(btn_row)

    def _make_swatch_grid(
        self,
        keys: list[str],
        swatch_list: list[_SwatchCard],
        is_dark: bool,
    ) -> QtWidgets.QGridLayout:
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        cols = 5
        for i, key in enumerate(keys):
            t = THEMES[key]
            card = _SwatchCard(t)
            card.clicked.connect(lambda checked, k=key, d=is_dark: self._on_swatch_clicked(k, d))
            swatch_list.append(card)
            grid.addWidget(card, i // cols, i % cols)
        return grid

    def _on_swatch_clicked(self, key: str, is_dark: bool) -> None:
        if is_dark:
            self._dark_key = key
            for sw in self._dark_swatches:
                sw.setSelected(sw.tokens.key == key)
        else:
            self._light_key = key
            for sw in self._light_swatches:
                sw.setSelected(sw.tokens.key == key)

    def _refresh_selections(self) -> None:
        for sw in self._dark_swatches:
            sw.setSelected(sw.tokens.key == self._dark_key)
        for sw in self._light_swatches:
            sw.setSelected(sw.tokens.key == self._light_key)

    def _accept(self) -> None:
        self.pair_selected.emit(self._dark_key, self._light_key)
        self.accept()


# ── ThemePickerDialog ─────────────────────────────────────────────────────────

class ThemePickerDialog(QtWidgets.QDialog):
    """Main theme picker: swatch grid with all 14 themes + Auto option."""

    theme_applied = Signal(str)   # key or "auto"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose Theme")
        self.setModal(False)          # allow interacting with the main window
        self.setMinimumWidth(680)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint
        )

        mgr = get_manager()
        self._current_key = "auto" if mgr.is_auto() else mgr.current.key
        self._all_swatches: list[_SwatchCard] = []

        self._build_ui()
        self._refresh_selection()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 20)
        root.setSpacing(16)

        # Title row
        title = QtWidgets.QLabel("Choose Theme")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QtWidgets.QLabel("Changes apply instantly")
        subtitle.setObjectName("mutedLabel")

        title_row = QtWidgets.QVBoxLayout()
        title_row.setSpacing(2)
        title_row.addWidget(title)
        title_row.addWidget(subtitle)
        root.addLayout(title_row)

        # Auto card (special)
        root.addWidget(self._make_auto_card())

        # Dark themes
        root.addWidget(_section_label("Dark Themes"))
        dark_grid = self._make_swatch_grid(DARK_THEMES)
        root.addLayout(dark_grid)

        # Light themes
        root.addWidget(_section_label("Light Themes"))
        light_grid = self._make_swatch_grid(LIGHT_THEMES)
        root.addLayout(light_grid)

        # Close button
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _make_auto_card(self) -> QtWidgets.QWidget:
        """Build the special Auto-mode card row."""
        container = QtWidgets.QWidget()
        container.setObjectName("autoCard")
        hl = QtWidgets.QHBoxLayout(container)
        hl.setContentsMargins(14, 10, 14, 10)
        hl.setSpacing(12)

        icon = QtWidgets.QLabel("◑")
        icon_font = icon.font()
        icon_font.setPointSize(20)
        icon.setFont(icon_font)
        icon.setFixedWidth(32)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setSpacing(1)
        name_lbl = QtWidgets.QLabel("Auto")
        name_font = name_lbl.font()
        name_font.setBold(True)
        name_lbl.setFont(name_font)
        desc_lbl = QtWidgets.QLabel("Switches between dark and light based on system day/night cycle")
        desc_lbl.setObjectName("mutedLabel")

        mgr = get_manager()
        dark_name = mgr.current.name if mgr.is_auto() else (
            THEMES[mgr.auto_dark()].name if mgr.auto_dark() in THEMES else "—"
        )
        light_name = THEMES[mgr.auto_light()].name if mgr.auto_light() in THEMES else "—"
        pair_lbl = QtWidgets.QLabel(f"Night: {dark_name}  ·  Day: {light_name}")
        pair_lbl.setObjectName("mutedLabel")
        self._auto_pair_label = pair_lbl

        text_col.addWidget(name_lbl)
        text_col.addWidget(desc_lbl)
        text_col.addWidget(pair_lbl)

        hl.addWidget(icon)
        hl.addLayout(text_col, 1)

        # Configure button
        cfg_btn = QtWidgets.QPushButton("Configure…")
        cfg_btn.setFixedWidth(110)
        cfg_btn.clicked.connect(self._open_auto_config)
        hl.addWidget(cfg_btn)

        # Select button
        self._auto_select_btn = QtWidgets.QPushButton("Select")
        self._auto_select_btn.setObjectName("primaryBtn")
        self._auto_select_btn.setFixedWidth(80)
        self._auto_select_btn.clicked.connect(self._apply_auto)
        hl.addWidget(self._auto_select_btn)

        return container

    def _make_swatch_grid(self, keys: list[str]) -> QtWidgets.QGridLayout:
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)
        cols = 5
        for i, key in enumerate(keys):
            t = THEMES[key]
            card = _SwatchCard(t)
            card.clicked.connect(lambda checked, k=key: self._apply_theme(k))
            self._all_swatches.append(card)
            grid.addWidget(card, i // cols, i % cols)
        return grid

    # ── Theme application ─────────────────────────────────────────────────

    def _apply_theme(self, key: str) -> None:
        self._current_key = key
        get_manager().apply(key)
        self._refresh_selection()
        self.theme_applied.emit(key)

    def _apply_auto(self) -> None:
        self._current_key = "auto"
        # Check if auto pair is configured, otherwise open config first
        mgr = get_manager()
        if not mgr.is_auto():
            mgr.apply("auto")
        self._refresh_selection()
        self.theme_applied.emit("auto")

    def _open_auto_config(self) -> None:
        dlg = AutoThemeDialog(self)
        dlg.pair_selected.connect(self._on_auto_pair_selected)
        dlg.exec()

    def _on_auto_pair_selected(self, dark_key: str, light_key: str) -> None:
        mgr = get_manager()
        mgr.configure_auto(dark_key, light_key)
        # Update pair label
        dark_name = THEMES[dark_key].name if dark_key in THEMES else dark_key
        light_name = THEMES[light_key].name if light_key in THEMES else light_key
        self._auto_pair_label.setText(f"Night: {dark_name}  ·  Day: {light_name}")
        # Apply auto mode
        self._apply_auto()

    # ── Selection refresh ─────────────────────────────────────────────────

    def _refresh_selection(self) -> None:
        is_auto = self._current_key == "auto"
        self._auto_select_btn.setText("✓ Active" if is_auto else "Select")
        self._auto_select_btn.setEnabled(not is_auto)
        for sw in self._all_swatches:
            sw.setSelected(not is_auto and sw.tokens.key == self._current_key)


# ── Convenience launcher ──────────────────────────────────────────────────────

def open_theme_picker(parent: QtWidgets.QWidget | None = None) -> ThemePickerDialog:
    """Create and show a non-modal ThemePickerDialog.  Returns the dialog."""
    dlg = ThemePickerDialog(parent)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
