import sys

def modify_tools():
    with open("demo_tile_proposals.py", "r", encoding="utf-8") as f:
        demo = f.read()
    
    with open("gui/workspaces/tools.py", "r", encoding="utf-8") as f:
        tools = f.read()

    # Extract _GlassBadge and _GlassResultChip
    badge_start = demo.find("class _GlassBadge")
    chip_end = demo.find("class LiquidGlassTile")
    glass_components = demo[badge_start:chip_end]

    # Create new ToolTile logic based on LiquidGlassTile + original ToolTile layout
    
    # We want ToolTile to look like LiquidGlassTile, so we replace its drawing and event handlers.
    tool_tile_old = """class ToolTile(QtWidgets.QFrame):"""
    tool_tile_end = """    def _on_theme_changed(self, tokens) -> None:\n        self._apply_sep_color(tokens)"""
    idx1 = tools.find(tool_tile_old)
    idx2 = tools.find(tool_tile_end) + len(tool_tile_end)
    
    old_tool_tile_content = tools[idx1:idx2]

    # Let's write the new ToolTile definition manually
    new_tool_tile = """
class ToolTile(QtWidgets.QFrame):
    \"\"\"Self-contained tool card with Liquid Glass aesthetic.\"\"\"

    def __init__(
        self,
        icon: str,
        title: str,
        description: str,
        *,
        log_height: int = 110,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("liquidGlassTile")
        self.setStyleSheet("QFrame#liquidGlassTile { background: transparent; border: none; }")
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self._hover_t = 0.0
        self._time = 0.0
        self._target_pos = QtCore.QPointF(200, 100)
        self._glow_pos = QtCore.QPointF(200, 100)
        self._is_hovered = False

        self._footer_buttons: list[QtWidgets.QPushButton] = []
        self._drawer_open = False

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        header = QtWidgets.QWidget()
        header_l = QtWidgets.QVBoxLayout(header)
        header_l.setContentsMargins(18, 16, 18, 12)
        header_l.setSpacing(4)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(12)
        
        self._badge = _GlassBadge(icon, size=40)
        title_row.addWidget(self._badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(6)
        
        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("cardTitle")
        info.addWidget(title_lbl)

        if description:
            desc_lbl = QtWidgets.QLabel(description)
            desc_lbl.setObjectName("sectionSubtitle")
            desc_lbl.setWordWrap(True)
            desc_lbl.setContentsMargins(0, 2, 0, 2)
            info.addWidget(desc_lbl)
            
        title_row.addLayout(info, 1)

        self.status_chip = _GlassResultChip()
        # Initial status
        self.status_chip.show_neutral("Ready.")
        title_row.addWidget(self.status_chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        header_l.addLayout(title_row)
        outer.addWidget(header)

        # ── Options area ──────────────────────────────────────────────────
        self._options = QtWidgets.QWidget()
        self._opts_l = QtWidgets.QVBoxLayout(self._options)
        self._opts_l.setContentsMargins(18, 0, 18, 12)
        self._opts_l.setSpacing(8)
        outer.addWidget(self._options)

        # ── Footer ────────────────────────────────────────────────────────
        self._footer = QtWidgets.QWidget()
        self._footer_l = QtWidgets.QHBoxLayout(self._footer)
        self._footer_l.setContentsMargins(18, 8, 18, 16)
        self._footer_l.setSpacing(8)
        outer.addWidget(self._footer)

        # ── Separator (hidden until drawer opens) ─────────────────────────
        self._sep = QtWidgets.QFrame()
        self._sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self._sep.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self._sep.setVisible(False)
        outer.addWidget(self._sep)

        # ── Drawer ────────────────────────────────────────────────────────
        self._drawer = QtWidgets.QWidget()
        drawer_l = QtWidgets.QVBoxLayout(self._drawer)
        drawer_l.setContentsMargins(18, 10, 18, 16)
        drawer_l.setSpacing(6)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)

        # Old status_label is now just a hidden label so existing code doesn't break,
        # but we also update status_chip.
        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setVisible(False)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(log_height)
        self.log_box.setObjectName("logBox")
        
        # Transparent background for log box to fit glass theme
        self.log_box.setStyleSheet("QPlainTextEdit { background: rgba(0, 0, 0, 0.1); border-radius: 6px; }")

        drawer_l.addWidget(self.progress_bar)
        drawer_l.addWidget(self.status_label)
        drawer_l.addWidget(self.log_box)

        self._drawer.setMinimumHeight(0)
        self._drawer.setMaximumHeight(0)
        outer.addWidget(self._drawer)

        # ── Theme ─────────────────────────────────────────────────────────
        self._apply_sep_color(get_manager().current)
        get_manager().theme_changed.connect(self._on_theme_changed)

    def mouseMoveEvent(self, e):
        self._target_pos = e.position()
        super().mouseMoveEvent(e)

    def enterEvent(self, e):
        self._is_hovered = True
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._is_hovered = False
        super().leaveEvent(e)

    def _tick(self):
        self._time += 0.016
        target_h = 1.0 if self._is_hovered else 0.0
        self._hover_t += (target_h - self._hover_t) * 0.15

        if not self._is_hovered:
            self._target_pos = QtCore.QPointF(self.width() / 2, self.height() / 2)

        dx = self._target_pos.x() - self._glow_pos.x()
        dy = self._target_pos.y() - self._glow_pos.y()
        self._glow_pos.setX(self._glow_pos.x() + dx * 0.15)
        self._glow_pos.setY(self._glow_pos.y() + dy * 0.15)
        
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        rect = QtCore.QRectF(self.rect())
        inner = rect.adjusted(2, 2, -2, -2)
        radius = 16.0
        ht = self._hover_t
        ac = QtGui.QColor(t.accent)

        path = QtGui.QPainterPath()
        path.addRoundedRect(inner, radius, radius)
        p.save()
        p.setClipPath(path)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        bg_alpha = 0.05 + ht * 0.03
        p.setBrush(QtGui.QColor(255, 255, 255, int(bg_alpha * 255)))
        p.drawRect(inner)

        t_alpha = 0.1 + ht * 0.05
        is_dark = getattr(t, "is_dark", True)
        if is_dark:
            tint_color = QtGui.QColor(0, 0, 0, int(t_alpha * 255))
        else:
            tint_color = QtGui.QColor(255, 255, 255, int(t_alpha * 255))
        p.setBrush(tint_color)
        p.drawRect(inner)

        glow_rad = inner.width() * 0.8
        glow = QtGui.QRadialGradient(self._glow_pos, glow_rad)
        c1 = QtGui.QColor(ac); c1.setAlphaF(0.15 + ht * 0.15)
        c2 = QtGui.QColor(ac); c2.setAlphaF(0.0)
        glow.setColorAt(0.0, c1)
        glow.setColorAt(1.0, c2)
        p.setBrush(glow)
        p.drawRect(inner)

        spec = QtGui.QLinearGradient(0, inner.top(), 0, inner.bottom())
        spec.setColorAt(0.0, QtGui.QColor(255, 255, 255, int((0.15 + ht * 0.05) * 255)))
        spec.setColorAt(0.4, QtGui.QColor(255, 255, 255, 0))
        p.setBrush(spec)
        p.drawRect(inner)
        p.restore()
        
        p.end()

    # ── Public API ────────────────────────────────────────────────────────

    def add_option(self, widget: QtWidgets.QWidget) -> None:
        self._opts_l.addWidget(widget)

    def add_option_layout(self, layout: QtWidgets.QLayout) -> None:
        self._opts_l.addLayout(layout)

    def set_run_button(self, label: str, slot) -> QtWidgets.QPushButton:
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(label)
        btn.setObjectName("primaryBtn")
        btn.setMinimumHeight(34)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.clicked.connect(slot)
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def add_secondary_button(self, label: str) -> QtWidgets.QPushButton:
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(label)
        btn.setMinimumHeight(34)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def finish_footer(self) -> None:
        self._footer_l.addStretch(1)

    def hide_footer(self) -> None:
        self._footer.setVisible(False)

    def open_drawer(self) -> None:
        if self._drawer_open:
            return
        self._drawer_open = True
        self._sep.setVisible(True)
        target = max(self._drawer.sizeHint().height(), 100)
        anim = QtCore.QPropertyAnimation(self._drawer, b"maximumHeight", self)
        anim.setDuration(250)
        anim.setStartValue(0)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self._drawer.setMaximumHeight(16777215))
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def set_running(self, running: bool) -> None:
        for btn in self._footer_buttons:
            btn.setEnabled(not running)

    def flash_status(self, from_hex: str, duration: int = 1600) -> None:
        pass # Replaced by show_success/error on the chip

    # ── Internal ──────────────────────────────────────────────────────────

    def _apply_sep_color(self, tokens) -> None:
        self._sep.setStyleSheet(f"QFrame {{ color: {tokens.card_border}; }}")

    def _on_theme_changed(self, tokens) -> None:
        self._apply_sep_color(tokens)
"""

    # We need to update usages of self._at_status.setText to self._at_status.show_neutral, etc.
    # But wait, self._at_status points to tile.status_label.
    # In the updated ToolTile, tile.status_chip is the real chip. So we should update tools.py:
    # `self._at_status = tile.status_label` -> `self._at_status = tile.status_chip`
    
    tools = tools.replace(old_tool_tile_content, glass_components + new_tool_tile)
    tools = tools.replace("tile.status_label", "tile.status_chip")
    tools = tools.replace("self._at_status.setText(", "self._at_status.show_neutral(")
    tools = tools.replace("self._codec_status.setText(", "self._codec_status.show_neutral(")
    tools = tools.replace("self._cleanup_status.setText(", "self._cleanup_status.show_neutral(")

    # Flash status replacements
    tools = tools.replace("self._at_tile.flash_status(t.success)", "self._at_status.show_success('Done')")
    tools = tools.replace("self._at_tile.flash_status(t.danger)", "self._at_status.show_error('Error')")
    tools = tools.replace("self._codec_tile.flash_status(t.success)", "self._codec_status.show_success('Done')")
    tools = tools.replace("self._codec_tile.flash_status(t.danger)", "self._codec_status.show_error('Error')")
    tools = tools.replace("self._cleanup_tile.flash_status(t.success if success else t.danger)", "self._cleanup_status.show_success('Done') if success else self._cleanup_status.show_error('Error')")
    
    # We must patch _tok to get_manager().current in the glass components if needed.
    # Actually demo_tile_proposals defines `def _tok(): ...` but we didn't include it. We can replace `_tok()` with `get_manager().current`
    tools = tools.replace("_tok()", "get_manager().current")
    
    # Fix the missing set_text_fast function on _GlassResultChip so we don't flash animations repeatedly
    chip_update = """
    def set_text_fast(self, text: str, state: int | None = None):
        self._text = text
        if state is not None:
            self._state = state
        self._resize()
        self.update()

    def show_neutral(self, text: str = "\\u29d6  Idle"):
"""
    tools = tools.replace("def show_neutral(self, text: str = \"\\u29d6  Idle\"):\n", chip_update)
    tools = tools.replace("self._at_status.show_neutral(f\"{done} / {total} processed\")", "self._at_status.set_text_fast(f\"{done}/{total}\")")

    with open("gui/workspaces/tools.py", "w", encoding="utf-8") as f:
        f.write(tools)
    
if __name__ == "__main__":
    modify_tools()