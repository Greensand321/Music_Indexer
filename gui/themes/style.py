"""AlphaDEXStyle — QProxyStyle subclass that paints every widget from scratch.

Everything is drawn from the active tokens and their *material*
(``gui/themes/tokens.py``), so Liquid Glass (translucent surfaces over the
backdrop) and the classic themes (opaque) share one code path:

* buttons are pills: secondary = the ``raised`` surface (fill, border, top
  highlight), primary / danger / success = a solid colour with dark text;
* line edits, combo boxes and spin boxes are pill fields on the ``well``
  surface with an accent focus ring;
* checkboxes are 18 px rounded squares, radios circles;
* scroll bars are thin rounded thumbs on a transparent track;
* sliders: a 5 px track, accent fill, a 13 px knob with a soft shadow;
* progress bars: a pill track with an accent fill;
* generic tab bars draw pill tabs (the page section tabs are the separate
  ``SegmentedTabBar`` in ``gui/pages.py``).

Rendering pipeline
──────────────────
  drawPrimitive       buttons, inputs, checkboxes, radio buttons, frames
  drawControl         push buttons, check/radio labels, progress bar, tabs,
                      combo-box label, scroll-bar arrows
  drawComplexControl  combo box, spin box, scroll bar, slider
  pixelMetric         spacing / size constants
  sizeFromContents    minimum widget dimensions
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.tokens import ThemeTokens, classic_material, rgba
from gui.themes.effects import R, lerp_color, with_alpha
from gui.themes.glass import qcolor, paint_surface

Qt    = QtCore.Qt
QSO   = QtWidgets.QStyleOption
QP    = QtGui.QPainter
QR    = QtCore.QRect
QRF   = QtCore.QRectF
QC    = QtGui.QColor
QPath = QtGui.QPainterPath
QBrush= QtGui.QBrush
QPen  = QtGui.QPen
QLinGrad = QtGui.QLinearGradient

_S = QtWidgets.QStyle
PE = _S.PrimitiveElement
CE = _S.ControlElement
CC = _S.ComplexControl
SC = _S.SubControl
SM = _S.SubElement
PM = _S.PixelMetric
CT = _S.ContentsType
ST = _S.State

# Qt6/PySide6 removed several enum members that existed in Qt5.
# Use getattr sentinels so the elif branches below are simply skipped.
_PE_FrameComboBox   = getattr(PE, "PE_FrameComboBox",   None)
_CE_GroupBoxLabel   = getattr(CE, "CE_GroupBoxLabel",   None)
_CE_ScrollBarGroove = getattr(CE, "CE_ScrollBarGroove", None)


# Dynamic property holding the font _polish_font assigned (DS-11).
_ROLE_FONT_PROP = "_adxRoleFont"
# Dynamic property: text margins this style gave a line edit (so a re-polish
# never stacks them and never overrides margins set by the widget's owner).
_TEXT_MARGIN_PROP = "_adxTextMargins"

_ACCENT_BUTTONS = ("primaryBtn", "dangerBtn", "successBtn")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _aa(p: QP) -> None:
    p.setRenderHint(QP.RenderHint.Antialiasing, True)
    p.setRenderHint(QP.RenderHint.TextAntialiasing, True)


def _col(value: str | QC | QBrush | QLinGrad) -> QBrush:
    if isinstance(value, QBrush):
        return value
    if isinstance(value, QC):
        return QBrush(value)
    if isinstance(value, QtGui.QGradient):
        return QBrush(value)
    return QBrush(qcolor(value))


def _rounded(p: QP, rect: QR | QRF, r: float,
             fill: str | QC | QBrush | QtGui.QGradient | None = None,
             border: str | QC | None = None,
             bw: float = 1.0) -> None:
    """Draw a single antialiased rounded rect (``r`` clamps to a pill)."""
    _aa(p)
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    rf = QRF(rect).adjusted(bw / 2, bw / 2, -bw / 2, -bw / 2)
    r = max(0.0, min(float(r), rf.height() / 2, rf.width() / 2))
    path = QPath()
    path.addRoundedRect(rf, r, r)
    if fill is not None:
        p.fillPath(path, _col(fill))
    if border is not None:
        c = qcolor(border) if isinstance(border, str) else border
        p.setPen(QPen(c, bw))
        p.drawPath(path)
    p.restore()


def _hover_t(widget: QtWidgets.QWidget | None,
             option: QtWidgets.QStyleOption) -> float:
    """Return hover progress 0→1.  Reads _hover_p attribute if available."""
    if widget is not None and hasattr(widget, "_hover_p"):
        return float(widget._hover_p)
    if option.state & ST.State_MouseOver:
        return 1.0
    return 0.0


def _is_enabled(option: QtWidgets.QStyleOption) -> bool:
    return bool(option.state & ST.State_Enabled)


def _is_sunken(option: QtWidgets.QStyleOption) -> bool:
    return bool(option.state & ST.State_Sunken)


def _is_on(option: QtWidgets.QStyleOption) -> bool:
    return bool(option.state & ST.State_On)


def _mix(a: QC, b: QC, t: float) -> QC:
    t = max(0.0, min(1.0, t))
    return QC(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


def _draw_check_mark(p: QP, rect: QRF, color: str | QC) -> None:
    """Draw a crisp ✓ checkmark inside rect."""
    p.save()
    _aa(p)
    pen = QPen(qcolor(color) if isinstance(color, str) else color, 2.0, Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    cx, cy = rect.center().x(), rect.center().y()
    s = rect.width() * 0.18
    path = QPath()
    path.moveTo(cx - s * 1.6, cy)
    path.lineTo(cx - s * 0.2, cy + s * 1.4)
    path.lineTo(cx + s * 1.8, cy - s * 1.4)
    p.drawPath(path)
    p.restore()


def _draw_chevron(p: QP, rect: QRF, color: str | QC, direction: str = "down",
                  size: float = 4.0) -> None:
    """A stroked chevron (down / up / left / right) centred in ``rect``."""
    p.save()
    _aa(p)
    pen = QPen(qcolor(color) if isinstance(color, str) else color, 1.6, Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    cx, cy = rect.center().x(), rect.center().y()
    s = size
    path = QPath()
    if direction == "down":
        path.moveTo(cx - s, cy - s * 0.5)
        path.lineTo(cx, cy + s * 0.5)
        path.lineTo(cx + s, cy - s * 0.5)
    elif direction == "up":
        path.moveTo(cx - s, cy + s * 0.5)
        path.lineTo(cx, cy - s * 0.5)
        path.lineTo(cx + s, cy + s * 0.5)
    elif direction == "left":
        path.moveTo(cx + s * 0.5, cy - s)
        path.lineTo(cx - s * 0.5, cy)
        path.lineTo(cx + s * 0.5, cy + s)
    else:
        path.moveTo(cx - s * 0.5, cy - s)
        path.lineTo(cx + s * 0.5, cy)
        path.lineTo(cx - s * 0.5, cy + s)
    p.drawPath(path)
    p.restore()


# Kept for callers that drew triangles.
_draw_arrow = _draw_chevron


# Size constants (a module table: pixelMetric runs thousands of times while
# a window is built).
_METRICS: dict = {
    PM.PM_ButtonMargin:               14,
    PM.PM_ButtonDefaultIndicator:     0,
    PM.PM_DefaultFrameWidth:          1,
    PM.PM_ScrollBarExtent:            10,
    PM.PM_ScrollBarSliderMin:         32,
    PM.PM_IndicatorWidth:             18,
    PM.PM_IndicatorHeight:            18,
    PM.PM_ExclusiveIndicatorWidth:    18,
    PM.PM_ExclusiveIndicatorHeight:   18,
    PM.PM_CheckBoxLabelSpacing:       8,
    PM.PM_RadioButtonLabelSpacing:    8,
    PM.PM_TabBarTabHSpace:            28,
    PM.PM_TabBarTabVSpace:            14,
    PM.PM_TabBarTabOverlap:           0,
    PM.PM_TabBarBaseOverlap:          0,
    PM.PM_ComboBoxFrameWidth:         0,
    PM.PM_SpinBoxFrameWidth:          0,
    PM.PM_MenuPanelWidth:             1,
    PM.PM_MenuHMargin:                4,
    PM.PM_MenuVMargin:                4,
    PM.PM_ToolBarFrameWidth:          0,
    PM.PM_ToolBarItemSpacing:         4,
    PM.PM_SliderThickness:            20,
    PM.PM_SliderControlThickness:     16,
    PM.PM_SliderLength:               16,
}


# ── Main style class ──────────────────────────────────────────────────────────

class AlphaDEXStyle(QtWidgets.QProxyStyle):
    """Full QPainter-based style engine for AlphaDEX."""

    def __init__(self, tokens: ThemeTokens) -> None:
        super().__init__("fusion")
        self._t = tokens

    @property
    def t(self) -> ThemeTokens:
        return self._t

    @property
    def m(self):
        return self._t.material or classic_material(self._t)

    def set_tokens(self, tokens: ThemeTokens) -> None:
        """Swap the colour tokens in place (theme switch without setStyle).

        Re-installing a style re-polishes every widget in the app; painting
        reads ``self._t`` on every call, so swapping it and repainting is
        all a colour change needs (DS-04)."""
        self._t = tokens

    # ── Shared looks ──────────────────────────────────────────────────────

    def _field(self, p: QP, rect: QR | QRF, focused: bool, hover: float, enabled: bool,
               radius: float | None = None) -> None:
        """A pill input field on the ``well`` surface; accent ring on focus."""
        m, t = self.m, self._t
        well = m.well
        fill = qcolor(well.fill)
        if hover > 0 and enabled:
            fill = _mix(fill, qcolor(m.hover), hover * 0.6)
        border = qcolor(m.raised.border if m.is_glass else t.input_border)
        if focused:
            border = qcolor(t.input_focus)
        elif hover > 0:
            border = _mix(border, qcolor(t.accent), hover * 0.4)
        r = rect.height() / 2 if radius is None else radius
        _rounded(p, rect, r, fill=fill, border=border, bw=1.5 if focused else 1.0)
        if not enabled:
            _rounded(p, rect, r, fill=with_alpha(t.content_bg, 90))

    def _thumb_colour(self, hover: float) -> QC:
        t = self._t
        return qcolor(rgba(t.text_primary, 0.22 + 0.16 * hover))

    # ── Metrics ───────────────────────────────────────────────────────────

    def pixelMetric(self, metric: PM,
                    option: QtWidgets.QStyleOption | None = None,
                    widget: QtWidgets.QWidget | None = None) -> int:
        value = _METRICS.get(metric)
        return value if value is not None else super().pixelMetric(metric, option, widget)

    def sizeFromContents(self, ct: CT,
                         option: QtWidgets.QStyleOption | None,
                         size: QtCore.QSize,
                         widget: QtWidgets.QWidget | None = None) -> QtCore.QSize:
        s = super().sizeFromContents(ct, option, size, widget)
        if ct == CT.CT_PushButton:
            return QtCore.QSize(max(s.width() + 20, 80), max(s.height(), 36))
        if ct in (CT.CT_ComboBox, CT.CT_SpinBox):
            return QtCore.QSize(s.width() + 12, max(s.height(), 34))
        if ct == CT.CT_LineEdit:
            return QtCore.QSize(s.width(), max(s.height(), 34))
        if ct == CT.CT_CheckBox:
            return QtCore.QSize(s.width() + 4, max(s.height(), 22))
        if ct == CT.CT_TabBarTab:
            return QtCore.QSize(s.width(), max(s.height(), 34))
        return s

    # ── Primitive elements ────────────────────────────────────────────────

    def drawPrimitive(self, element: PE,
                      option: QtWidgets.QStyleOption,
                      painter: QP,
                      widget: QtWidgets.QWidget | None = None) -> None:
        t, m = self._t, self.m
        painter.save()

        # ── Button panel: a pill ───────────────────────────────────────────
        if element == PE.PE_PanelButtonCommand:
            enabled = _is_enabled(option)
            sunken  = _is_sunken(option)
            ht      = _hover_t(widget, option)
            name = widget.objectName() if widget else ""
            flat = (isinstance(option, QtWidgets.QStyleOptionButton)
                    and bool(option.features & QtWidgets.QStyleOptionButton.ButtonFeature.Flat))
            rect = QRF(option.rect)

            if name in _ACCENT_BUTTONS:
                base, hover, press = {
                    "primaryBtn": (t.accent, t.accent_hover, t.accent_pressed),
                    "dangerBtn": (t.danger, t.danger_hover, t.danger_hover),
                    "successBtn": (t.success, lerp_color(t.success, "#ffffff", 0.15).name(),
                                   lerp_color(t.success, "#000000", 0.10).name()),
                }[name]
                if sunken:
                    fill = QC(press)
                else:
                    fill = lerp_color(base, hover, ht)
                _rounded(painter, rect, rect.height() / 2, fill=fill)
                if not enabled:
                    _rounded(painter, rect, rect.height() / 2,
                             fill=with_alpha(m.backdrop_tone, 150))
            elif flat:
                if enabled and (ht > 0 or sunken):
                    over = qcolor(m.pressed if sunken else m.hover)
                    over.setAlphaF(over.alphaF() * (1.0 if sunken else ht))
                    _rounded(painter, rect, rect.height() / 2, fill=over)
            else:
                raised = m.raised
                fill = qcolor(raised.fill)
                if sunken:
                    fill = _mix(fill, qcolor(m.pressed), 1.0) if not m.is_glass else qcolor(
                        rgba(t.text_primary, 0.22))
                elif ht > 0:
                    target = qcolor(rgba(t.text_primary, 0.20)) if m.is_glass else qcolor(m.hover)
                    fill = _mix(fill, target, ht)
                paint_surface(painter, rect, raised, fill=fill,
                              opacity=1.0 if enabled else 0.55)

        # ── Line edit / text area frame ────────────────────────────────────
        elif element in (PE.PE_FrameLineEdit, PE.PE_PanelLineEdit):
            if widget is not None and isinstance(widget.parentWidget(),
                                                 (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
                pass   # the spin box / combo box draws the field
            else:
                focused = bool(option.state & ST.State_HasFocus)
                self._field(painter, QRF(option.rect), focused, _hover_t(widget, option),
                            _is_enabled(option))

        # ── Focus rect → suppress (we draw focus inline) ──────────────────
        elif element == PE.PE_FrameFocusRect:
            pass

        # ── Checkbox indicator ─────────────────────────────────────────────
        elif element == PE.PE_IndicatorCheckBox:
            checked  = _is_on(option)
            partial  = bool(option.state & ST.State_NoChange)
            enabled  = _is_enabled(option)
            ht       = _hover_t(widget, option)
            box_rect = QRF(option.rect)
            sz = min(box_rect.width(), box_rect.height())
            bx = box_rect.left() + (box_rect.width() - sz) / 2
            by = box_rect.top()  + (box_rect.height() - sz) / 2
            box_rf = QRF(bx, by, sz, sz).adjusted(0.5, 0.5, -0.5, -0.5)

            if checked or partial:
                fill = lerp_color(t.accent, t.accent_hover, ht)
                _rounded(painter, box_rf, 6, fill=fill)
                if partial:
                    _rounded(painter, QRF(box_rf.center().x() - 4.5, box_rf.center().y() - 1,
                                          9, 2), 1, fill=t.text_inverse)
                else:
                    _draw_check_mark(painter, box_rf, t.text_inverse)
            else:
                fill = _mix(qcolor(m.well.fill), qcolor(m.hover), ht)
                border = _mix(qcolor(m.raised.border if m.is_glass else t.input_border),
                              qcolor(t.accent), ht * 0.6)
                _rounded(painter, box_rf, 6, fill=fill, border=border, bw=1.2)
            if not enabled:
                _rounded(painter, box_rf, 6, fill=with_alpha(m.backdrop_tone, 130))

        # ── Radio button indicator ─────────────────────────────────────────
        elif element == PE.PE_IndicatorRadioButton:
            checked  = _is_on(option)
            enabled  = _is_enabled(option)
            ht       = _hover_t(widget, option)
            box_rect = QRF(option.rect)
            sz = min(box_rect.width(), box_rect.height())
            bx = box_rect.left() + (box_rect.width() - sz) / 2
            by = box_rect.top()  + (box_rect.height() - sz) / 2
            box_rf = QRF(bx, by, sz, sz).adjusted(0.5, 0.5, -0.5, -0.5)
            _aa(painter)
            if checked:
                fill = lerp_color(t.accent, t.accent_hover, ht)
                _rounded(painter, box_rf, sz, fill=fill)
                inner = box_rf.adjusted(5, 5, -5, -5)
                _rounded(painter, inner, sz, fill=t.text_inverse)
            else:
                fill = _mix(qcolor(m.well.fill), qcolor(m.hover), ht)
                border = _mix(qcolor(m.raised.border if m.is_glass else t.input_border),
                              qcolor(t.accent), ht * 0.6)
                _rounded(painter, box_rf, sz, fill=fill, border=border, bw=1.2)
            if not enabled:
                _rounded(painter, box_rf, sz, fill=with_alpha(m.backdrop_tone, 130))

        # ── Generic frame / panel ──────────────────────────────────────────
        elif element in (PE.PE_Frame, PE.PE_FrameGroupBox):
            _rounded(painter, option.rect, 16, border=m.card.border or t.card_border, bw=1.0)

        elif element == PE.PE_FrameTabWidget:
            _rounded(painter, option.rect, 18, fill=m.view_alt if m.is_glass else None,
                     border=m.hairline if m.is_glass else t.card_border, bw=1.0)

        elif _PE_FrameComboBox is not None and element == _PE_FrameComboBox:
            pass  # drawn in drawComplexControl

        elif element == PE.PE_PanelButtonTool:
            enabled = _is_enabled(option)
            ht = _hover_t(widget, option)
            rect = QRF(option.rect)
            if _is_sunken(option) or _is_on(option):
                _rounded(painter, rect, 10, fill=m.pressed)
            elif enabled and ht > 0:
                over = qcolor(m.hover)
                over.setAlphaF(over.alphaF() * ht)
                _rounded(painter, rect, 10, fill=over)

        elif element == PE.PE_PanelItemViewItem:
            opt = option
            selected = bool(opt.state & ST.State_Selected)
            hover    = bool(opt.state & ST.State_MouseOver)
            if selected:
                painter.fillRect(opt.rect, opt.palette.brush(QtGui.QPalette.ColorRole.Highlight))
            elif hover:
                painter.fillRect(opt.rect, qcolor(m.hover))
            if self._hairline_rows(widget):
                r = opt.rect
                painter.fillRect(QR(r.left(), r.bottom(), r.width(), 1), qcolor(m.hairline))

        else:
            painter.restore()
            super().drawPrimitive(element, option, painter, widget)
            return

        painter.restore()

    @staticmethod
    def _hairline_rows(widget: QtWidgets.QWidget | None) -> bool:
        """Tables without a grid and trees get a hairline under each row."""
        if isinstance(widget, QtWidgets.QTableView):
            return not widget.showGrid()
        return isinstance(widget, QtWidgets.QTreeView)

    # ── Control elements ──────────────────────────────────────────────────

    def drawControl(self, element: CE,
                    option: QtWidgets.QStyleOption,
                    painter: QP,
                    widget: QtWidgets.QWidget | None = None) -> None:
        t, m = self._t, self.m
        painter.save()

        # ── Push button ────────────────────────────────────────────────────
        if element == CE.CE_PushButton:
            if isinstance(option, QtWidgets.QStyleOptionButton):
                self.drawPrimitive(PE.PE_PanelButtonCommand, option, painter, widget)
                self.drawControl(CE.CE_PushButtonLabel, option, painter, widget)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        elif element == CE.CE_PushButtonBevel:
            if isinstance(option, QtWidgets.QStyleOptionButton):
                self.drawPrimitive(PE.PE_PanelButtonCommand, option, painter, widget)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        elif element == CE.CE_PushButtonLabel:
            if not isinstance(option, QtWidgets.QStyleOptionButton):
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return
            opt   = option
            name  = widget.objectName() if widget else ""
            is_accent = name in _ACCENT_BUTTONS
            if not _is_enabled(option):
                col = t.text_muted
            else:
                col = t.text_inverse if is_accent else t.text_primary
            _aa(painter)
            painter.setPen(QPen(QC(col)))
            rect = QRF(opt.rect)

            text_w = QtGui.QFontMetrics(painter.font()).horizontalAdvance(opt.text) if opt.text else 0
            if opt.icon and not opt.icon.isNull():
                icon_sz = opt.iconSize
                gap = 6 if opt.text else 0
                total = icon_sz.width() + gap + text_w
                left = rect.left() + max(8.0, (rect.width() - total) / 2)
                icon_rect = QRF(left, rect.top() + (rect.height() - icon_sz.height()) / 2,
                                icon_sz.width(), icon_sz.height())
                mode = (QtGui.QIcon.Mode.Disabled if not _is_enabled(option)
                        else QtGui.QIcon.Mode.Normal)
                opt.icon.paint(painter, icon_rect.toRect(), Qt.AlignmentFlag.AlignCenter, mode)
                text_rect = QRF(icon_rect.right() + gap, rect.top(),
                                max(0.0, rect.right() - icon_rect.right() - gap), rect.height())
                align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            else:
                text_rect = rect
                align = Qt.AlignmentFlag.AlignCenter

            if opt.text:
                font = painter.font()
                font.setWeight(QtGui.QFont.Weight.DemiBold)
                painter.setFont(font)
                painter.drawText(text_rect.toRect(), int(align), opt.text)

        # ── Check box ──────────────────────────────────────────────────────
        elif element == CE.CE_CheckBox:
            if isinstance(option, QtWidgets.QStyleOptionButton):
                opt = option
                ind_size = self.pixelMetric(PM.PM_IndicatorWidth, option, widget)
                ind_rect = QR(opt.rect.x(),
                              opt.rect.y() + (opt.rect.height() - ind_size) // 2,
                              ind_size, ind_size)
                ind_opt = QtWidgets.QStyleOptionButton(opt)
                ind_opt.rect = ind_rect
                self.drawPrimitive(PE.PE_IndicatorCheckBox, ind_opt, painter, widget)
                if opt.text:
                    sp   = self.pixelMetric(PM.PM_CheckBoxLabelSpacing, option, widget)
                    col  = t.text_muted if not _is_enabled(option) else t.text_primary
                    painter.setPen(QPen(QC(col)))
                    lx   = ind_rect.right() + sp
                    lrect = QR(lx, opt.rect.y(),
                               opt.rect.right() - lx, opt.rect.height())
                    painter.drawText(lrect,
                                     int(Qt.AlignmentFlag.AlignLeft |
                                         Qt.AlignmentFlag.AlignVCenter),
                                     opt.text)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        # ── Radio button ───────────────────────────────────────────────────
        elif element == CE.CE_RadioButton:
            if isinstance(option, QtWidgets.QStyleOptionButton):
                opt = option
                ind_size = self.pixelMetric(PM.PM_ExclusiveIndicatorWidth, option, widget)
                ind_rect = QR(opt.rect.x(),
                              opt.rect.y() + (opt.rect.height() - ind_size) // 2,
                              ind_size, ind_size)
                ind_opt = QtWidgets.QStyleOptionButton(opt)
                ind_opt.rect = ind_rect
                self.drawPrimitive(PE.PE_IndicatorRadioButton, ind_opt, painter, widget)
                if opt.text:
                    sp    = self.pixelMetric(PM.PM_RadioButtonLabelSpacing, option, widget)
                    col   = t.text_muted if not _is_enabled(option) else t.text_primary
                    painter.setPen(QPen(QC(col)))
                    lx    = ind_rect.right() + sp
                    lrect = QR(lx, opt.rect.y(),
                               opt.rect.right() - lx, opt.rect.height())
                    painter.drawText(lrect,
                                     int(Qt.AlignmentFlag.AlignLeft |
                                         Qt.AlignmentFlag.AlignVCenter),
                                     opt.text)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        # ── Progress bar groove: a pill track ──────────────────────────────
        elif element == CE.CE_ProgressBarGroove:
            _rounded(painter, option.rect, option.rect.height() / 2, fill=m.track)

        # ── Progress bar contents: accent fill ─────────────────────────────
        elif element == CE.CE_ProgressBarContents:
            if isinstance(option, QtWidgets.QStyleOptionProgressBar):
                opt  = option
                mn   = opt.minimum
                mx   = opt.maximum
                val  = opt.progress
                rect = option.rect
                if mx == mn:  # indeterminate: the whole track, softly
                    _rounded(painter, rect, rect.height() / 2,
                             fill=with_alpha(t.progress_fg, 150))
                elif mx > mn and val >= mn:
                    ratio = (val - mn) / (mx - mn)
                    filled = QR(rect.left(), rect.top(),
                                int(rect.width() * ratio), rect.height())
                    if filled.width() > 0:
                        filled.setWidth(max(filled.width(), rect.height()))
                        grad = QLinGrad(QRF(filled).topLeft(), QRF(filled).topRight())
                        grad.setColorAt(0.0, QC(t.progress_fg))
                        grad.setColorAt(1.0, QC(t.accent_hover))
                        _rounded(painter, filled, rect.height() / 2, fill=grad)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        # ── Progress bar label ─────────────────────────────────────────────
        elif element == CE.CE_ProgressBarLabel:
            pass  # we suppress the text label; bar height is decorative

        # ── Tab bar tab: pill tabs ─────────────────────────────────────────
        elif element == CE.CE_TabBarTab:
            if isinstance(option, QtWidgets.QStyleOptionTab):
                self.drawControl(CE.CE_TabBarTabShape, option, painter, widget)
                self.drawControl(CE.CE_TabBarTabLabel, option, painter, widget)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        elif element == CE.CE_TabBarTabShape:
            if isinstance(option, QtWidgets.QStyleOptionTab):
                opt     = option
                active  = bool(opt.state & ST.State_Selected)
                ht      = _hover_t(widget, opt)
                rect    = QRF(opt.rect).adjusted(2, 3, -2, -3)
                if active:
                    paint_surface(painter, rect, m.raised, radius=rect.height() / 2)
                elif ht > 0 or (opt.state & ST.State_MouseOver):
                    over = qcolor(m.hover)
                    over.setAlphaF(over.alphaF() * max(ht, 1.0 if opt.state & ST.State_MouseOver else 0))
                    _rounded(painter, rect, rect.height() / 2, fill=over)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        elif element == CE.CE_TabBarTabLabel:
            if isinstance(option, QtWidgets.QStyleOptionTab):
                opt    = option
                active = bool(opt.state & ST.State_Selected)
                col    = t.text_primary if active else t.tab_text
                if not _is_enabled(opt):
                    col = t.text_muted
                _aa(painter)
                painter.setPen(QPen(QC(col)))
                font = painter.font()
                font.setWeight(QtGui.QFont.Weight.DemiBold if active else QtGui.QFont.Weight.Medium)
                painter.setFont(font)
                text_rect = QR(opt.rect)
                # A tab button (e.g. the "Experimental" pill on a section
                # tab) sits at the right edge: centre the label in what is
                # left, so the two never overlap.
                button_w = opt.rightButtonSize.width()
                if button_w > 0:
                    pad = self.pixelMetric(PM.PM_TabBarTabHSpace, opt, widget) // 2
                    text_rect.adjust(pad // 2, 0, -(button_w + pad), 0)
                painter.drawText(text_rect,
                                 int(Qt.AlignmentFlag.AlignCenter), opt.text)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        # ── Group box label ────────────────────────────────────────────────
        elif _CE_GroupBoxLabel is not None and element == _CE_GroupBoxLabel:
            if isinstance(option, QtWidgets.QStyleOptionGroupBox):
                col = t.text_secondary
                _aa(painter)
                painter.setPen(QPen(QC(col)))
                font = painter.font()
                font.setPointSize(max(font.pointSize() - 1, 9))
                font.setWeight(QtGui.QFont.Weight.DemiBold)
                painter.setFont(font)
                painter.drawText(option.rect,
                                 int(Qt.AlignmentFlag.AlignLeft |
                                     Qt.AlignmentFlag.AlignVCenter),
                                 option.text if hasattr(option, "text") else "")
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        # ── Scroll bar arrows ──────────────────────────────────────────────
        elif element in (CE.CE_ScrollBarAddLine, CE.CE_ScrollBarSubLine):
            pass  # suppress arrows — scrollbar uses click-on-track

        elif element == CE.CE_ScrollBarSlider:
            self._paint_scroll_thumb(painter, option, widget, option.rect)

        elif _CE_ScrollBarGroove is not None and element == _CE_ScrollBarGroove:
            pass   # transparent track: the backdrop shows through

        else:
            painter.restore()
            super().drawControl(element, option, painter, widget)
            return

        painter.restore()

    def _paint_scroll_thumb(self, painter: QP, option, widget, rect: QR) -> None:
        ht = _hover_t(widget, option)
        if option.state & ST.State_Sunken:
            ht = 1.0
        horiz = getattr(option, "orientation", Qt.Orientation.Vertical) == Qt.Orientation.Horizontal
        thick = 6.0 + 2.0 * ht
        r = QRF(rect)
        if horiz:
            r = QRF(r.left() + 2, r.center().y() - thick / 2, r.width() - 4, thick)
        else:
            r = QRF(r.center().x() - thick / 2, r.top() + 2, thick, r.height() - 4)
        _rounded(painter, r, thick / 2, fill=self._thumb_colour(ht))

    # ── Complex controls ──────────────────────────────────────────────────

    def drawComplexControl(self, ctrl: CC,
                           option: QtWidgets.QStyleOptionComplex,
                           painter: QP,
                           widget: QtWidgets.QWidget | None = None) -> None:
        t, m = self._t, self.m
        painter.save()

        # ── Combo box: a pill field with a chevron ─────────────────────────
        if ctrl == CC.CC_ComboBox:
            if isinstance(option, QtWidgets.QStyleOptionComboBox):
                opt     = option
                focused = bool(opt.state & ST.State_HasFocus)
                self._field(painter, QRF(opt.rect), focused, _hover_t(widget, opt),
                            _is_enabled(opt))
                arrow_rect = self.subControlRect(CC.CC_ComboBox, opt,
                                                 SC.SC_ComboBoxArrow, widget)
                _draw_chevron(painter, QRF(arrow_rect).adjusted(0, 0, -4, 0),
                              t.text_secondary, "down")
            else:
                painter.restore()
                super().drawComplexControl(ctrl, option, painter, widget)
                return

        # ── Spin box: a pill field with two small chevrons ─────────────────
        elif ctrl == CC.CC_SpinBox:
            if isinstance(option, QtWidgets.QStyleOptionSpinBox):
                opt     = option
                focused = bool(opt.state & ST.State_HasFocus)
                self._field(painter, QRF(opt.rect), focused, _hover_t(widget, opt),
                            _is_enabled(opt))
                if opt.buttonSymbols != QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons:
                    up_rect   = self.subControlRect(CC.CC_SpinBox, opt, SC.SC_SpinBoxUp, widget)
                    down_rect = self.subControlRect(CC.CC_SpinBox, opt, SC.SC_SpinBoxDown, widget)
                    arrow_col = t.text_secondary
                    if up_rect.isValid():
                        _draw_chevron(painter, QRF(up_rect).adjusted(0, 3, -4, 0),
                                      arrow_col, "up", 3.5)
                    if down_rect.isValid():
                        _draw_chevron(painter, QRF(down_rect).adjusted(0, 0, -4, -3),
                                      arrow_col, "down", 3.5)
            else:
                painter.restore()
                super().drawComplexControl(ctrl, option, painter, widget)
                return

        # ── Scroll bar: transparent track, thin rounded thumb ──────────────
        elif ctrl == CC.CC_ScrollBar:
            if isinstance(option, QtWidgets.QStyleOptionSlider):
                slider_rect = self.subControlRect(CC.CC_ScrollBar, option,
                                                  SC.SC_ScrollBarSlider, widget)
                if slider_rect.isValid() and option.maximum > option.minimum:
                    self._paint_scroll_thumb(painter, option, widget, slider_rect)
            else:
                painter.restore()
                super().drawComplexControl(ctrl, option, painter, widget)
                return

        # ── Slider: 5 px track, accent fill, 13 px knob ────────────────────
        elif ctrl == CC.CC_Slider:
            if isinstance(option, QtWidgets.QStyleOptionSlider):
                opt  = option
                groove_rect = self.subControlRect(CC.CC_Slider, opt,
                                                  SC.SC_SliderGroove, widget)
                handle_rect = self.subControlRect(CC.CC_Slider, opt,
                                                  SC.SC_SliderHandle, widget)
                horiz = (opt.orientation == Qt.Orientation.Horizontal)
                track = 5.0

                if horiz:
                    gr = QRF(groove_rect.x() + 2, groove_rect.center().y() - track / 2 + 0.5,
                             groove_rect.width() - 4, track)
                else:
                    gr = QRF(groove_rect.center().x() - track / 2 + 0.5, groove_rect.y() + 2,
                             track, groove_rect.height() - 4)
                _rounded(painter, gr, track / 2, fill=m.track)

                hc = QtCore.QPointF(handle_rect.center())
                if opt.maximum > opt.minimum:
                    if horiz:
                        filled_r = QRF(gr.x(), gr.y(), max(0.0, hc.x() - gr.x()), gr.height())
                    else:
                        filled_r = QRF(gr.x(), hc.y(), gr.width(), max(0.0, gr.bottom() - hc.y()))
                    if filled_r.width() > 0 and filled_r.height() > 0:
                        _rounded(painter, filled_r, track / 2,
                                 fill=t.accent if _is_enabled(opt) else t.text_muted)

                ht  = _hover_t(widget, opt)
                hsz = 13.0 + 2.0 * ht
                hr  = QRF(hc.x() - hsz / 2, hc.y() - hsz / 2, hsz, hsz)
                # Soft shadow under the knob (0 2px 6px rgba(0,0,0,.4)).
                for grow, alpha in ((3.0, 30), (1.5, 45)):
                    _rounded(painter, hr.adjusted(-grow, -grow + 2, grow, grow + 2), hsz,
                             fill=QC(0, 0, 0, alpha))
                _rounded(painter, hr, hsz / 2, fill=m.knob)
            else:
                painter.restore()
                super().drawComplexControl(ctrl, option, painter, widget)
                return

        else:
            painter.restore()
            super().drawComplexControl(ctrl, option, painter, widget)
            return

        painter.restore()

    # ── Sub-control rectangles ────────────────────────────────────────────

    def subControlRect(self, ctrl: CC,
                       option: QtWidgets.QStyleOptionComplex,
                       sc: SC,
                       widget: QtWidgets.QWidget | None = None) -> QR:
        if ctrl == CC.CC_ComboBox and isinstance(option, QtWidgets.QStyleOptionComboBox):
            r  = option.rect
            aw = 30
            pad = max(12, r.height() // 2 - 2)
            if sc == SC.SC_ComboBoxFrame:
                return r
            if sc == SC.SC_ComboBoxArrow:
                return QR(r.right() - aw, r.top(), aw, r.height())
            if sc == SC.SC_ComboBoxEditField:
                return QR(r.left() + pad, r.top(), r.width() - aw - pad, r.height())
            if sc == SC.SC_ComboBoxListBoxPopup:
                return r
        if ctrl == CC.CC_SpinBox and isinstance(option, QtWidgets.QStyleOptionSpinBox):
            r  = option.rect
            bw = 26 if option.buttonSymbols != QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons else 0
            pad = max(12, r.height() // 2 - 2)
            if sc == SC.SC_SpinBoxFrame:
                return r
            if sc == SC.SC_SpinBoxUp:
                return QR(r.right() - bw, r.top(), bw, r.height() // 2)
            if sc == SC.SC_SpinBoxDown:
                return QR(r.right() - bw, r.top() + r.height() // 2,
                          bw, r.height() - r.height() // 2)
            if sc == SC.SC_SpinBoxEditField:
                return QR(r.left() + pad, r.top(), r.width() - bw - pad - 2, r.height())
        return super().subControlRect(ctrl, option, sc, widget)

    # ── Polish / unpolish ─────────────────────────────────────────────────

    def polish(self, obj: QtWidgets.QWidget | QtGui.QPalette | QtWidgets.QApplication) -> None:  # type: ignore[override]
        if isinstance(obj, QtWidgets.QWidget):
            # Install hover tracking on interactive widgets
            if isinstance(obj, (QtWidgets.QAbstractButton,
                                 QtWidgets.QComboBox,
                                 QtWidgets.QAbstractSlider,
                                 QtWidgets.QAbstractSpinBox,
                                 QtWidgets.QLineEdit,
                                 QtWidgets.QTabBar)):
                obj.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

            if isinstance(obj, QtWidgets.QLineEdit):
                self._polish_line_edit(obj)

            if (isinstance(obj, QtWidgets.QMenu)
                    and not obj.testAttribute(Qt.WidgetAttribute.WA_WState_Created)):
                # Rounded menus need a translucent window behind the corners
                # (set before the native window exists).
                obj.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

            # Per-role font assignments
            self._polish_font(obj)

        super().polish(obj)

    @staticmethod
    def _polish_line_edit(edit: QtWidgets.QLineEdit) -> None:
        """Pill fields need room at the ends; editors inside spin / combo
        boxes are positioned by ``subControlRect`` instead."""
        if isinstance(edit.parentWidget(), (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
            return
        current = edit.textMargins()
        ours = edit.property(_TEXT_MARGIN_PROP)
        if (current.left(), current.right()) not in ((0, 0), (ours, ours)):
            return   # the owner set its own margins
        edit.setTextMargins(10, 0, 10, 0)
        edit.setProperty(_TEXT_MARGIN_PROP, 10)

    def _polish_font(self, widget: QtWidgets.QWidget) -> None:
        """Assign the type-ramp role fonts (``gui.themes.glass.font``) to
        specific objectName roles and to controls."""
        try:
            from gui.themes.glass import font as role_font
        except ImportError:  # pragma: no cover
            return

        # Never override a font someone set explicitly (DS-11). Fonts this
        # method assigned are remembered in a dynamic property, so a later
        # re-polish may refresh them, but a caller's setFont() always wins.
        if widget.testAttribute(Qt.WidgetAttribute.WA_SetFont):
            ours = widget.property(_ROLE_FONT_PROP)
            if not isinstance(ours, QtGui.QFont) or ours != widget.font():
                return

        name = widget.objectName()

        def _assign(f: QtGui.QFont) -> None:
            widget.setFont(f)
            widget.setProperty(_ROLE_FONT_PROP, widget.font())

        if name == "appTitle":
            _assign(role_font("title", weight=700))
        elif name == "sectionTitle":
            _assign(role_font("title"))
        elif name in ("sectionSubtitle", "mutedLabel"):
            _assign(role_font("label"))
        elif name == "sidebarSectionLabel":
            _assign(role_font("micro"))
        elif name == "navButton":
            _assign(role_font("body"))
        elif name in ("libraryPath", "libStats", "statUnit"):
            _assign(role_font("label"))
        elif name == "statNumber":
            _assign(role_font("label", weight=600))
        elif name == "logText":
            _assign(role_font("mono"))
        elif isinstance(widget, (QtWidgets.QAbstractButton,
                                  QtWidgets.QComboBox,
                                  QtWidgets.QAbstractSpinBox,
                                  QtWidgets.QLineEdit,
                                  QtWidgets.QTabBar)):
            _assign(role_font("body"))

    def unpolish(self, obj: QtWidgets.QWidget | QtWidgets.QApplication) -> None:  # type: ignore[override]
        super().unpolish(obj)
