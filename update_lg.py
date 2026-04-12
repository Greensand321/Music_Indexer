import sys
import re

with open("demo_tile_proposals.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add import math if not present
if "import math" not in content:
    content = content.replace("import sys", "import sys\nimport math")

# 2. Replace LiquidGlassTile class
start_marker = "class LiquidGlassTile(QtWidgets.QFrame):"
end_marker = "# ══════════════════════════════════════════════════════════════════════════════\n# Proposal label header row"

idx1 = content.find(start_marker)
idx2 = content.find(end_marker)

if idx1 == -1 or idx2 == -1:
    print("Could not find LiquidGlassTile boundaries.")
    sys.exit(1)

new_classes = """class LiquidGlassAuroraTile(QtWidgets.QFrame):
    \"\"\"
    LG Variant 1: "Aurora"
    Focuses on perfectly smooth, animated ambient background glows moving within the glass.
    Replaces pixelated border glows with a high-quality sweeping conical gradient rim
    and floating internal "orbs" of light.
    \"\"\"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("liquidGlassTile")
        self.setStyleSheet("QFrame#liquidGlassTile { background: transparent; border: none; }")
        self.setMouseTracking(True)
        self._hover_t = 0.0
        self._time = 0.0

        lo = QtWidgets.QVBoxLayout(self)
        lo.setContentsMargins(18, 16, 18, 16)
        lo.setSpacing(12)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self._badge = _GlassBadge(_ICON, size=40)
        row.addWidget(self._badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(3)
        info.addWidget(_card_title(_TITLE))
        info.addWidget(_subtitle(_DESC))
        row.addLayout(info, 1)
        self._chip = _GlassResultChip()
        row.addWidget(self._chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        lo.addLayout(row)
        lo.addWidget(_glass_options())

        btn_row = QtWidgets.QHBoxLayout()
        r, o = _std_buttons()
        r.clicked.connect(lambda: self._chip.show_success("\\u2713  847 entries"))
        btn_row.addWidget(r); btn_row.addWidget(o); btn_row.addStretch(1)
        lo.addLayout(btn_row)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        self._time += 0.016
        target = 1.0 if self.underMouse() else 0.0
        self._hover_t += (target - self._hover_t) * 0.15
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = _tok()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        rect = QtCore.QRectF(self.rect())
        inner = rect.adjusted(2, 2, -2, -2)
        radius = 16.0
        ht = self._hover_t
        ac = QtGui.QColor(t.accent)

        # Subtle smooth outer shadow
        shadow_c = QtGui.QColor(ac)
        shadow_c.setAlphaF(0.15 + ht * 0.2)
        p.setPen(QtGui.QPen(shadow_c, 3.0 + ht * 4.0))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(inner.adjusted(1, 1, -1, -1), radius, radius)

        # Draw glass interior
        path = QtGui.QPainterPath()
        path.addRoundedRect(inner, radius, radius)
        p.save()
        p.setClipPath(path)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(255, 255, 255, int((0.04 + ht * 0.04) * 255)))
        p.drawRect(inner)

        orb1_x = inner.center().x() + math.sin(self._time) * inner.width() * 0.3
        orb1_y = inner.center().y() + math.cos(self._time * 0.8) * inner.height() * 0.3
        orb2_x = inner.center().x() + math.cos(self._time * 1.2) * inner.width() * 0.3
        orb2_y = inner.center().y() + math.sin(self._time * 1.1) * inner.height() * 0.3

        def draw_orb(ox, oy, c, mult):
            rad = inner.width() * mult
            grad = QtGui.QRadialGradient(ox, oy, rad)
            c1 = QtGui.QColor(c); c1.setAlphaF(0.15 + ht * 0.15)
            c2 = QtGui.QColor(c); c2.setAlphaF(0.0)
            grad.setColorAt(0, c1); grad.setColorAt(1, c2)
            p.setBrush(grad)
            p.drawRect(inner)

        draw_orb(orb1_x, orb1_y, ac, 0.6)
        draw_orb(orb2_x, orb2_y, QtGui.QColor(t.success), 0.5)

        spec = QtGui.QLinearGradient(0, inner.top(), 0, inner.bottom())
        spec.setColorAt(0.0, QtGui.QColor(255, 255, 255, int((0.15 + ht * 0.1) * 255)))
        spec.setColorAt(0.5, QtGui.QColor(255, 255, 255, 0))
        p.setBrush(spec)
        p.drawRect(inner)
        p.restore()

        angle = (self._time * 60) % 360
        conic = QtGui.QConicalGradient(inner.center(), angle)
        c_rim1 = QtGui.QColor(ac); c_rim1.setAlphaF(0.6 + ht * 0.4)
        c_rim2 = QtGui.QColor(255, 255, 255); c_rim2.setAlphaF(0.1)
        conic.setColorAt(0.0, c_rim1)
        conic.setColorAt(0.2, c_rim2)
        conic.setColorAt(0.8, c_rim2)
        conic.setColorAt(1.0, c_rim1)

        p.setPen(QtGui.QPen(QtGui.QBrush(conic), 1.5))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(inner.adjusted(0.75, 0.75, -0.75, -0.75), radius, radius)
        p.end()

class LiquidGlassPrismaticTile(QtWidgets.QFrame):
    \"\"\"
    LG Variant 2: "Prismatic Depth"
    Features a dark, deep glass look with parallax shift and an intense specular mouse follow.
    \"\"\"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("liquidGlassTile")
        self.setStyleSheet("QFrame#liquidGlassTile { background: transparent; border: none; }")
        self.setMouseTracking(True)
        self._mouse_pos = QtCore.QPointF(200, 100)
        self._target_pos = QtCore.QPointF(200, 100)
        self._hover_t = 0.0
        self._time = 0.0

        self._content = QtWidgets.QWidget(self)
        self._content.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lo = QtWidgets.QVBoxLayout(self._content)
        lo.setContentsMargins(18, 16, 18, 16)
        lo.setSpacing(12)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self._badge = _GlassBadge(_ICON, size=40)
        row.addWidget(self._badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(3)
        info.addWidget(_card_title(_TITLE))
        info.addWidget(_subtitle(_DESC))
        row.addLayout(info, 1)
        self._chip = _GlassResultChip()
        row.addWidget(self._chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        lo.addLayout(row)
        lo.addWidget(_glass_options())

        btn_row = QtWidgets.QHBoxLayout()
        r, o = _std_buttons()
        r.clicked.connect(lambda: self._chip.show_success("\\u2713  847 entries"))
        btn_row.addWidget(r); btn_row.addWidget(o); btn_row.addStretch(1)
        lo.addLayout(btn_row)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._content.setGeometry(self.rect())

    def mouseMoveEvent(self, e):
        self._target_pos = e.position()
        super().mouseMoveEvent(e)

    def _tick(self):
        self._time += 0.016
        is_hovered = self.underMouse()
        target = 1.0 if is_hovered else 0.0
        self._hover_t += (target - self._hover_t) * 0.15

        if not is_hovered:
            self._target_pos = QtCore.QPointF(self.width() / 2, self.height() / 2)

        dx = self._target_pos.x() - self._mouse_pos.x()
        dy = self._target_pos.y() - self._mouse_pos.y()
        self._mouse_pos.setX(self._mouse_pos.x() + dx * 0.2)
        self._mouse_pos.setY(self._mouse_pos.y() + dy * 0.2)

        cx = self.width() / 2
        cy = self.height() / 2
        px = (self._mouse_pos.x() - cx) * 0.05 * self._hover_t
        py = (self._mouse_pos.y() - cy) * 0.05 * self._hover_t
        self._content.setGeometry(QtCore.QRect(int(px), int(py), self.width(), self.height()))
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = _tok()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(self.rect())
        inner = rect.adjusted(2, 2, -2, -2)
        radius = 16.0
        ht = self._hover_t
        ac = QtGui.QColor(t.accent)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        bg = QtGui.QColor(0, 0, 0, int((0.2 + ht * 0.1) * 255))
        p.setBrush(bg)
        p.drawRoundedRect(inner, radius, radius)

        p.save()
        path = QtGui.QPainterPath()
        path.addRoundedRect(inner, radius, radius)
        p.setClipPath(path)

        shine_rad = inner.width() * 0.8
        grad = QtGui.QRadialGradient(self._mouse_pos, shine_rad)
        c1 = QtGui.QColor(ac); c1.setAlphaF(0.3 * ht)
        c2 = QtGui.QColor(t.success); c2.setAlphaF(0.15 * ht)
        c3 = QtGui.QColor(255, 255, 255, 0)
        grad.setColorAt(0.0, c1)
        grad.setColorAt(0.4, c2)
        grad.setColorAt(1.0, c3)
        p.setBrush(grad)
        p.drawRect(inner)
        p.restore()

        rim = QtGui.QLinearGradient(0, inner.top(), 0, inner.bottom())
        rim.setColorAt(0.0, QtGui.QColor(255, 255, 255, int((0.4 + ht * 0.4) * 255)))
        rim.setColorAt(0.2, QtGui.QColor(255, 255, 255, 0))
        rim.setColorAt(1.0, QtGui.QColor(ac.red(), ac.green(), ac.blue(), int((0.1 + ht * 0.3) * 255)))
        p.setPen(QtGui.QPen(QtGui.QBrush(rim), 1.2))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(inner.adjusted(0.6, 0.6, -0.6, -0.6), radius, radius)
        p.end()

class LiquidGlassCyberTile(QtWidgets.QFrame):
    \"\"\"
    LG Variant 3: "Cyber Fluid"
    Animated liquid wave at the bottom, scanlines overlay, pulsing neon border.
    \"\"\"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("liquidGlassTile")
        self.setStyleSheet("QFrame#liquidGlassTile { background: transparent; border: none; }")
        self.setMouseTracking(True)
        self._hover_t = 0.0
        self._time = 0.0

        lo = QtWidgets.QVBoxLayout(self)
        lo.setContentsMargins(18, 16, 18, 16)
        lo.setSpacing(12)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self._badge = _GlassBadge(_ICON, size=40)
        row.addWidget(self._badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(3)
        info.addWidget(_card_title(_TITLE))
        info.addWidget(_subtitle(_DESC))
        row.addLayout(info, 1)
        self._chip = _GlassResultChip()
        row.addWidget(self._chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        lo.addLayout(row)
        lo.addWidget(_glass_options())

        btn_row = QtWidgets.QHBoxLayout()
        r, o = _std_buttons()
        r.clicked.connect(lambda: self._chip.show_success("\\u2713  847 entries"))
        btn_row.addWidget(r); btn_row.addWidget(o); btn_row.addStretch(1)
        lo.addLayout(btn_row)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        self._time += 0.016
        target = 1.0 if self.underMouse() else 0.0
        self._hover_t += (target - self._hover_t) * 0.15
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = _tok()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(self.rect())
        inner = rect.adjusted(2, 2, -2, -2)
        radius = 12.0
        ht = self._hover_t
        ac = QtGui.QColor(t.accent)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(ac.red(), ac.green(), ac.blue(), 15))
        p.drawRoundedRect(inner, radius, radius)

        p.save()
        path = QtGui.QPainterPath()
        path.addRoundedRect(inner, radius, radius)
        p.setClipPath(path)

        wave_height = 20.0 + ht * 15.0
        base_y = inner.bottom() - wave_height

        wave_path1 = QtGui.QPainterPath()
        wave_path1.moveTo(inner.left(), inner.bottom())
        wave_path1.lineTo(inner.left(), base_y)

        wave_path2 = QtGui.QPainterPath()
        wave_path2.moveTo(inner.left(), inner.bottom())
        wave_path2.lineTo(inner.left(), base_y)

        for x in range(int(inner.left()), int(inner.right()) + 1, 5):
            freq = 0.02
            w1_y = base_y + math.sin(x * freq + self._time * 3) * 6
            w2_y = base_y + math.cos(x * freq * 1.3 - self._time * 2.5) * 8
            wave_path1.lineTo(x, w1_y)
            wave_path2.lineTo(x, w2_y)

        wave_path1.lineTo(inner.right(), inner.bottom())
        wave_path2.lineTo(inner.right(), inner.bottom())
        wave_path1.closeSubpath()
        wave_path2.closeSubpath()

        c_wave1 = QtGui.QColor(ac); c_wave1.setAlphaF(0.2 + ht * 0.15)
        p.setBrush(c_wave1)
        p.drawPath(wave_path1)

        c_wave2 = QtGui.QColor(t.accent); c_wave2.setAlphaF(0.3 + ht * 0.2)
        p.setBrush(c_wave2)
        p.drawPath(wave_path2)

        if ht > 0.01:
            p.setPen(QtGui.QColor(255, 255, 255, int(10 * ht)))
            for y in range(int(inner.top()), int(inner.bottom()), 4):
                p.drawLine(int(inner.left()), y, int(inner.right()), y)
        p.restore()

        pulse = (math.sin(self._time * 4) + 1) / 2.0
        border_c = QtGui.QColor(ac)
        border_c.setAlphaF(0.4 + 0.6 * pulse * ht)
        p.setPen(QtGui.QPen(border_c, 1.5 + ht * 1.0))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(inner.adjusted(0.75, 0.75, -0.75, -0.75), radius, radius)
        p.end()
"""

content = content[:idx1] + new_classes + "\n" + content[idx2:]

# 3. Replace the DemoWindow layout section
demo_start = "        # ── Featured: Liquid Glass ────────────────────────────────────────"
demo_end = "        # ── 2-column grid of remaining proposals ─────────────────────────"

idx3 = content.find(demo_start)
idx4 = content.find(demo_end)

if idx3 == -1 or idx4 == -1:
    print("Could not find DemoWindow boundaries.")
    sys.exit(1)

new_demo = """        # ── Featured: Liquid Glass Variants ──────────────────────────────
        page.addWidget(_proposal_header("LG1", "Liquid Glass V1: Aurora — Ambient animated glows, sweeping border rim"))
        lg1_row = QtWidgets.QHBoxLayout()
        lg1_tile = LiquidGlassAuroraTile()
        lg1_tile.setMinimumWidth(440)
        lg1_row.addWidget(lg1_tile, 1)
        lg1_row.addStretch(1)
        page.addLayout(lg1_row)
        page.addSpacing(20)

        page.addWidget(_proposal_header("LG2", "Liquid Glass V2: Prismatic Depth — Mouse parallax, deep dark glass, specular follow"))
        lg2_row = QtWidgets.QHBoxLayout()
        lg2_tile = LiquidGlassPrismaticTile()
        lg2_tile.setMinimumWidth(440)
        lg2_row.addWidget(lg2_tile, 1)
        lg2_row.addStretch(1)
        page.addLayout(lg2_row)
        page.addSpacing(20)

        page.addWidget(_proposal_header("LG3", "Liquid Glass V3: Cyber Fluid — Animated liquid waves, neon pulse border, scanlines"))
        lg3_row = QtWidgets.QHBoxLayout()
        lg3_tile = LiquidGlassCyberTile()
        lg3_tile.setMinimumWidth(440)
        lg3_row.addWidget(lg3_tile, 1)
        lg3_row.addStretch(1)
        page.addLayout(lg3_row)
        page.addSpacing(28)

"""

content = content[:idx3] + new_demo + content[idx4:]

content = content.replace(
    "Featured: Liquid Glass (P4 + P6 + P7) — frosted glass card, heavy spring toggles, \"\n            \"cycling glass chip. Switch themes to see adaptation. \"\n            \"P2 and the Liquid Glass border animate continuously.",
    "Featured: 3 Liquid Glass Variants — Aurora, Prismatic Depth, and Cyber Fluid. Switch themes to see adaptation."
)

with open("demo_tile_proposals.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated successfully.")
