"""AlphaDEXStyle — QProxyStyle subclass that paints every widget from scratch.

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
from gui.themes.tokens import ThemeTokens
from gui.themes.effects import R, lerp_color, with_alpha

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _aa(p: QP) -> None:
    p.setRenderHint(QP.RenderHint.Antialiasing, True)
    p.setRenderHint(QP.RenderHint.TextAntialiasing, True)


def _rounded(p: QP, rect: QR | QRF, r: float,
             fill: str | QC | None = None,
             border: str | QC | None = None,
             bw: float = 1.0) -> None:
    """Draw a single antialiased rounded rect."""
    _aa(p)
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    rf = QRF(rect).adjusted(bw / 2, bw / 2, -bw / 2, -bw / 2)
    path = QPath()
    path.addRoundedRect(rf, r, r)
    if fill is not None:
        c = QC(fill) if isinstance(fill, str) else fill
        p.fillPath(path, QBrush(c))
    if border is not None:
        c = QC(border) if isinstance(border, str) else border
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


def _draw_check_mark(p: QP, rect: QRF, color: str) -> None:
    """Draw a crisp ✓ checkmark inside rect."""
    p.save()
    _aa(p)
    pen = QPen(QC(color), 2.0, Qt.PenStyle.SolidLine,
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


def _draw_arrow(p: QP, rect: QRF, color: str, direction: str = "down") -> None:
    """Draw a small triangle arrow: direction ∈ {down, up, left, right}."""
    p.save()
    _aa(p)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QC(color)))
    cx, cy = rect.center().x(), rect.center().y()
    s = 4.0
    path = QPath()
    if direction == "down":
        path.moveTo(cx - s, cy - s * 0.5)
        path.lineTo(cx + s, cy - s * 0.5)
        path.lineTo(cx, cy + s * 0.7)
    elif direction == "up":
        path.moveTo(cx - s, cy + s * 0.5)
        path.lineTo(cx + s, cy + s * 0.5)
        path.lineTo(cx, cy - s * 0.7)
    elif direction == "left":
        path.moveTo(cx + s * 0.5, cy - s)
        path.lineTo(cx + s * 0.5, cy + s)
        path.lineTo(cx - s * 0.7, cy)
    elif direction == "right":
        path.moveTo(cx - s * 0.5, cy - s)
        path.lineTo(cx - s * 0.5, cy + s)
        path.lineTo(cx + s * 0.7, cy)
    path.closeSubpath()
    p.drawPath(path)
    p.restore()


# ── Main style class ──────────────────────────────────────────────────────────

class AlphaDEXStyle(QtWidgets.QProxyStyle):
    """Full QPainter-based style engine for AlphaDEX."""

    def __init__(self, tokens: ThemeTokens) -> None:
        super().__init__("fusion")
        self._t = tokens

    @property
    def t(self) -> ThemeTokens:
        return self._t

    # ── Metrics ───────────────────────────────────────────────────────────

    def pixelMetric(self, metric: PM,
                    option: QtWidgets.QStyleOption | None = None,
                    widget: QtWidgets.QWidget | None = None) -> int:
        m: dict[PM, int] = {
            PM.PM_ButtonMargin:               8,
            PM.PM_ButtonDefaultIndicator:     0,
            PM.PM_DefaultFrameWidth:          1,
            PM.PM_ScrollBarExtent:            8,
            PM.PM_ScrollBarSliderMin:         28,
            PM.PM_IndicatorWidth:             16,
            PM.PM_IndicatorHeight:            16,
            PM.PM_ExclusiveIndicatorWidth:    16,
            PM.PM_ExclusiveIndicatorHeight:   16,
            PM.PM_CheckBoxLabelSpacing:       6,
            PM.PM_RadioButtonLabelSpacing:    6,
            PM.PM_TabBarTabHSpace:            20,
            PM.PM_TabBarTabVSpace:            10,
            PM.PM_TabBarTabOverlap:           0,
            PM.PM_ComboBoxFrameWidth:         0,
            PM.PM_SpinBoxFrameWidth:          0,
            PM.PM_MenuPanelWidth:             1,
            PM.PM_MenuHMargin:                4,
            PM.PM_MenuVMargin:                4,
            PM.PM_ToolBarFrameWidth:          0,
            PM.PM_ToolBarItemSpacing:         4,
        }
        return m.get(metric, super().pixelMetric(metric, option, widget))

    def sizeFromContents(self, ct: CT,
                         option: QtWidgets.QStyleOption | None,
                         size: QtCore.QSize,
                         widget: QtWidgets.QWidget | None = None) -> QtCore.QSize:
        s = super().sizeFromContents(ct, option, size, widget)
        if ct == CT.CT_PushButton:
            return QtCore.QSize(max(s.width() + 24, 80), max(s.height(), 34))
        if ct in (CT.CT_ComboBox, CT.CT_LineEdit, CT.CT_SpinBox):
            return QtCore.QSize(s.width(), max(s.height(), 34))
        if ct == CT.CT_CheckBox:
            return QtCore.QSize(s.width() + 4, max(s.height(), 20))
        return s

    # ── Primitive elements ────────────────────────────────────────────────

    def drawPrimitive(self, element: PE,
                      option: QtWidgets.QStyleOption,
                      painter: QP,
                      widget: QtWidgets.QWidget | None = None) -> None:
        t = self._t
        painter.save()

        # ── Button panel ───────────────────────────────────────────────────
        if element == PE.PE_PanelButtonCommand:
            enabled = _is_enabled(option)
            sunken  = _is_sunken(option)
            ht      = _hover_t(widget, option)

            # Detect special button names via objectName
            name = widget.objectName() if widget else ""

            if name == "primaryBtn":
                base  = t.accent
                hover = t.accent_hover
                press = t.accent_pressed
            elif name == "dangerBtn":
                base  = t.danger
                hover = t.danger_hover
                press = t.danger_hover
            elif name == "successBtn":
                base  = t.success
                hover = lerp_color(t.success, "#ffffff", 0.15).name()
                press = lerp_color(t.success, "#000000", 0.10).name()
            else:
                base  = t.card_bg
                hover = t.sidebar_hover
                press = t.card_border

            if not enabled:
                fill_col = lerp_color(base, t.content_bg, 0.50)
            elif sunken:
                fill_col = QC(press)
            else:
                fill_col = lerp_color(base, hover, ht)

            border_col = None if name in ("primaryBtn", "dangerBtn", "successBtn") \
                         else t.card_border

            _rounded(painter, option.rect, R.button,
                     fill=fill_col, border=border_col, bw=1.0)

        # ── Line edit / text area frame ────────────────────────────────────
        elif element == PE.PE_FrameLineEdit:
            focused = bool(option.state & ST.State_HasFocus)
            border  = t.input_focus if focused else t.input_border
            _rounded(painter, option.rect, R.input,
                     fill=t.input_bg, border=border, bw=1.5 if focused else 1.0)

        # ── Focus rect → suppress (we draw focus inline) ──────────────────
        elif element == PE.PE_FrameFocusRect:
            pass

        # ── Checkbox indicator ─────────────────────────────────────────────
        elif element == PE.PE_IndicatorCheckBox:
            checked  = _is_on(option)
            enabled  = _is_enabled(option)
            ht       = _hover_t(widget, option)
            box_rect = QRF(option.rect)
            sz = min(box_rect.width(), box_rect.height())
            bx = box_rect.left() + (box_rect.width() - sz) / 2
            by = box_rect.top()  + (box_rect.height() - sz) / 2
            box_rf = QRF(bx, by, sz, sz).adjusted(1, 1, -1, -1)

            if checked:
                fill = lerp_color(t.accent, t.accent_hover, ht)
                _rounded(painter, box_rf, R.checkbox, fill=fill)
                _draw_check_mark(painter, box_rf, t.text_inverse)
            else:
                fill = lerp_color(t.input_bg, t.sidebar_hover, ht * 0.4)
                border = lerp_color(t.input_border, t.accent, ht * 0.6)
                _rounded(painter, box_rf, R.checkbox, fill=fill, border=border, bw=1.5)
            if not enabled:
                _rounded(painter, box_rf, R.checkbox,
                         fill=with_alpha(t.content_bg, 120))

        # ── Radio button indicator ─────────────────────────────────────────
        elif element == PE.PE_IndicatorRadioButton:
            checked  = _is_on(option)
            enabled  = _is_enabled(option)
            ht       = _hover_t(widget, option)
            box_rect = QRF(option.rect)
            sz = min(box_rect.width(), box_rect.height())
            bx = box_rect.left() + (box_rect.width() - sz) / 2
            by = box_rect.top()  + (box_rect.height() - sz) / 2
            box_rf = QRF(bx, by, sz, sz).adjusted(1, 1, -1, -1)
            _aa(painter)
            if checked:
                fill = lerp_color(t.accent, t.accent_hover, ht)
                _rounded(painter, box_rf, R.radio, fill=fill)
                inner = box_rf.adjusted(4, 4, -4, -4)
                _rounded(painter, inner, R.radio, fill=t.text_inverse)
            else:
                fill   = lerp_color(t.input_bg, t.sidebar_hover, ht * 0.4)
                border = lerp_color(t.input_border, t.accent, ht * 0.6)
                _rounded(painter, box_rf, R.radio,
                         fill=fill, border=border, bw=1.5)
            if not enabled:
                _rounded(painter, box_rf, R.radio, fill=with_alpha(t.content_bg, 120))

        # ── Generic frame / panel ──────────────────────────────────────────
        elif element in (PE.PE_Frame, PE.PE_FrameGroupBox):
            _rounded(painter, option.rect, R.card,
                     border=t.card_border, bw=1.0)

        elif element == PE.PE_PanelLineEdit:
            focused = bool(option.state & ST.State_HasFocus)
            border  = t.input_focus if focused else t.input_border
            _rounded(painter, option.rect, R.input,
                     fill=t.input_bg, border=border,
                     bw=1.5 if focused else 1.0)

        elif _PE_FrameComboBox is not None and element == _PE_FrameComboBox:
            pass  # drawn in drawComplexControl

        elif element == PE.PE_PanelItemViewItem:
            opt = option
            selected = bool(opt.state & ST.State_Selected)
            hover    = bool(opt.state & ST.State_MouseOver)
            if selected:
                painter.fillRect(opt.rect, QC(t.accent))
            elif hover:
                painter.fillRect(opt.rect, QC(t.sidebar_hover))

        else:
            painter.restore()
            super().drawPrimitive(element, option, painter, widget)
            return

        painter.restore()

    # ── Control elements ──────────────────────────────────────────────────

    def drawControl(self, element: CE,
                    option: QtWidgets.QStyleOption,
                    painter: QP,
                    widget: QtWidgets.QWidget | None = None) -> None:
        t = self._t
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
            is_accent = name in ("primaryBtn", "dangerBtn", "successBtn")
            col   = t.text_inverse if is_accent else (
                    t.text_muted if not _is_enabled(option) else t.text_primary)
            _aa(painter)
            painter.setPen(QPen(QC(col)))
            rect = QRF(opt.rect)

            if opt.icon and not opt.icon.isNull():
                icon_sz  = opt.iconSize
                icon_rect = QRF(rect.left() + 8,
                                rect.top() + (rect.height() - icon_sz.height()) / 2,
                                icon_sz.width(), icon_sz.height())
                mode = (QtGui.QIcon.Mode.Disabled if not _is_enabled(option)
                        else QtGui.QIcon.Mode.Normal)
                opt.icon.paint(painter, icon_rect.toRect(), Qt.AlignmentFlag.AlignCenter, mode)
                text_rect = QRF(icon_rect.right() + 6, rect.top(),
                                rect.right() - icon_rect.right() - 6, rect.height())
            else:
                text_rect = rect

            if opt.text:
                font = painter.font()
                font.setWeight(QtGui.QFont.Weight.Medium)
                painter.setFont(font)
                painter.drawText(text_rect.toRect(),
                                 int(Qt.AlignmentFlag.AlignCenter),
                                 opt.text)

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

        # ── Progress bar groove ────────────────────────────────────────────
        elif element == CE.CE_ProgressBarGroove:
            _rounded(painter, option.rect, R.scroll,
                     fill=t.progress_bg)

        # ── Progress bar contents ──────────────────────────────────────────
        elif element == CE.CE_ProgressBarContents:
            if isinstance(option, QtWidgets.QStyleOptionProgressBar):
                opt  = option
                mn   = opt.minimum
                mx   = opt.maximum
                val  = opt.progress
                if mx == mn:  # indeterminate — draw pulsing segment
                    grad = QLinGrad(
                        QRF(option.rect).topLeft(),
                        QRF(option.rect).topRight()
                    )
                    grad.setColorAt(0.0, with_alpha(t.progress_fg, 30))
                    grad.setColorAt(0.5, QC(t.progress_fg))
                    grad.setColorAt(1.0, with_alpha(t.progress_fg, 30))
                    _rounded(painter, option.rect, R.scroll, fill=QC(t.progress_fg))
                else:
                    if mx > mn and val >= mn:
                        ratio = (val - mn) / (mx - mn)
                        filled = QR(option.rect.left(),
                                    option.rect.top(),
                                    int(option.rect.width() * ratio),
                                    option.rect.height())
                        if filled.width() > 0:
                            grad = QLinGrad(
                                QRF(filled).topLeft(),
                                QRF(filled).topRight()
                            )
                            grad.setColorAt(0.0, QC(t.progress_fg))
                            grad.setColorAt(1.0, QC(t.accent_hover))
                            _rounded(painter, filled, R.scroll, fill=grad)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        # ── Progress bar label ─────────────────────────────────────────────
        elif element == CE.CE_ProgressBarLabel:
            pass  # we suppress the text label; bar height is decorative

        # ── Tab bar tab ────────────────────────────────────────────────────
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
                fill    = t.tab_active_bg if active else lerp_color(t.tab_bg, t.card_bg, ht)
                rect    = QRF(opt.rect)
                _rounded(painter, rect.adjusted(0, 2, 0, 0), R.tab,
                         fill=fill,
                         border=t.card_border if not active else None)
                # Bottom accent line on active tab
                if active:
                    line_rect = QRF(rect.left() + 4, rect.bottom() - 2.5,
                                    rect.width() - 8, 2.5)
                    _rounded(painter, line_rect, 1.5, fill=t.accent)
            else:
                painter.restore()
                super().drawControl(element, option, painter, widget)
                return

        elif element == CE.CE_TabBarTabLabel:
            if isinstance(option, QtWidgets.QStyleOptionTab):
                opt    = option
                active = bool(opt.state & ST.State_Selected)
                col    = t.tab_active_text if active else t.tab_text
                _aa(painter)
                painter.setPen(QPen(QC(col)))
                font = painter.font()
                if active:
                    font.setWeight(QtGui.QFont.Weight.SemiBold)
                painter.setFont(font)
                painter.drawText(opt.rect,
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
                font.setWeight(QtGui.QFont.Weight.SemiBold)
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
            ht = _hover_t(widget, option)
            fill = lerp_color(t.text_muted, t.text_secondary, ht)
            _rounded(painter, option.rect.adjusted(2, 2, -2, -2),
                     R.scroll, fill=fill)

        elif _CE_ScrollBarGroove is not None and element == _CE_ScrollBarGroove:
            painter.fillRect(option.rect, QC(t.content_bg))

        else:
            painter.restore()
            super().drawControl(element, option, painter, widget)
            return

        painter.restore()

    # ── Complex controls ──────────────────────────────────────────────────

    def drawComplexControl(self, ctrl: CC,
                           option: QtWidgets.QStyleOptionComplex,
                           painter: QP,
                           widget: QtWidgets.QWidget | None = None) -> None:
        t = self._t
        painter.save()

        # ── Combo box ──────────────────────────────────────────────────────
        if ctrl == CC.CC_ComboBox:
            if isinstance(option, QtWidgets.QStyleOptionComboBox):
                opt     = option
                focused = bool(opt.state & ST.State_HasFocus)
                ht      = _hover_t(widget, opt)
                border  = t.input_focus if focused else lerp_color(
                    t.input_border, t.accent, ht * 0.5)
                _rounded(painter, opt.rect, R.input,
                         fill=t.input_bg, border=border,
                         bw=1.5 if focused else 1.0)
                arrow_rect = self.subControlRect(CC.CC_ComboBox, opt,
                                                 SC.SC_ComboBoxArrow, widget)
                _draw_arrow(painter, QRF(arrow_rect), t.text_secondary, "down")
            else:
                painter.restore()
                super().drawComplexControl(ctrl, option, painter, widget)
                return

        # ── Spin box ───────────────────────────────────────────────────────
        elif ctrl == CC.CC_SpinBox:
            if isinstance(option, QtWidgets.QStyleOptionSpinBox):
                opt     = option
                focused = bool(opt.state & ST.State_HasFocus)
                ht      = _hover_t(widget, opt)
                border  = t.input_focus if focused else lerp_color(
                    t.input_border, t.accent, ht * 0.5)
                _rounded(painter, opt.rect, R.input,
                         fill=t.input_bg, border=border,
                         bw=1.5 if focused else 1.0)
                up_rect   = self.subControlRect(CC.CC_SpinBox, opt,
                                               SC.SC_SpinBoxUp, widget)
                down_rect = self.subControlRect(CC.CC_SpinBox, opt,
                                               SC.SC_SpinBoxDown, widget)
                arrow_col = t.text_secondary
                if up_rect.isValid():
                    _draw_arrow(painter, QRF(up_rect), arrow_col, "up")
                if down_rect.isValid():
                    _draw_arrow(painter, QRF(down_rect), arrow_col, "down")
            else:
                painter.restore()
                super().drawComplexControl(ctrl, option, painter, widget)
                return

        # ── Scroll bar ─────────────────────────────────────────────────────
        elif ctrl == CC.CC_ScrollBar:
            if isinstance(option, QtWidgets.QStyleOptionSlider):
                opt  = option
                # Track
                painter.fillRect(opt.rect, QC(t.content_bg))
                # Slider
                slider_rect = self.subControlRect(CC.CC_ScrollBar, opt,
                                                  SC.SC_ScrollBarSlider, widget)
                if slider_rect.isValid():
                    ht   = _hover_t(widget, opt)
                    fill = lerp_color(t.text_muted, t.text_secondary, ht)
                    pad  = 2
                    _rounded(painter,
                             slider_rect.adjusted(pad, pad, -pad, -pad),
                             R.scroll, fill=fill)
            else:
                painter.restore()
                super().drawComplexControl(ctrl, option, painter, widget)
                return

        # ── Slider ─────────────────────────────────────────────────────────
        elif ctrl == CC.CC_Slider:
            if isinstance(option, QtWidgets.QStyleOptionSlider):
                opt  = option
                groove_rect = self.subControlRect(CC.CC_Slider, opt,
                                                  SC.SC_SliderGroove, widget)
                handle_rect = self.subControlRect(CC.CC_Slider, opt,
                                                  SC.SC_SliderHandle, widget)
                horiz = (opt.orientation == Qt.Orientation.Horizontal)

                # Groove background
                if horiz:
                    gr = QRF(groove_rect.x(),
                             groove_rect.center().y() - 3,
                             groove_rect.width(), 6)
                else:
                    gr = QRF(groove_rect.center().x() - 3,
                             groove_rect.y(),
                             6, groove_rect.height())
                _rounded(painter, gr, 3, fill=t.progress_bg)

                # Filled portion
                if opt.maximum > opt.minimum:
                    ratio = (opt.sliderValue - opt.minimum) / (opt.maximum - opt.minimum)
                    if horiz:
                        fw = gr.width() * ratio
                        filled_r = QRF(gr.x(), gr.y(), fw, gr.height())
                    else:
                        fh = gr.height() * (1 - ratio)
                        filled_r = QRF(gr.x(), gr.y() + fh, gr.width(), gr.height() - fh)
                    if (horiz and filled_r.width() > 0) or \
                       (not horiz and filled_r.height() > 0):
                        _rounded(painter, filled_r, 3, fill=t.accent)

                # Handle
                ht  = _hover_t(widget, opt)
                hsz = 16
                hcx = handle_rect.center()
                hr  = QRF(hcx.x() - hsz / 2, hcx.y() - hsz / 2, hsz, hsz)
                fill_h = lerp_color(t.accent, t.accent_hover, ht)
                _rounded(painter, hr, hsz / 2, fill=fill_h)
                # Inner white dot
                _rounded(painter, hr.adjusted(5, 5, -5, -5), 3,
                         fill=t.text_inverse)
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
            aw = 24
            if sc == SC.SC_ComboBoxFrame:
                return r
            if sc == SC.SC_ComboBoxArrow:
                return QR(r.right() - aw, r.top(), aw, r.height())
            if sc == SC.SC_ComboBoxEditField:
                return QR(r.left() + 8, r.top(), r.width() - aw - 8, r.height())
            if sc == SC.SC_ComboBoxListBoxPopup:
                return r
        if ctrl == CC.CC_SpinBox and isinstance(option, QtWidgets.QStyleOptionSpinBox):
            r  = option.rect
            bw = 20
            if sc == SC.SC_SpinBoxFrame:
                return r
            if sc == SC.SC_SpinBoxUp:
                return QR(r.right() - bw, r.top(), bw, r.height() // 2)
            if sc == SC.SC_SpinBoxDown:
                return QR(r.right() - bw, r.top() + r.height() // 2,
                          bw, r.height() - r.height() // 2)
            if sc == SC.SC_SpinBoxEditField:
                return QR(r.left() + 8, r.top(), r.width() - bw - 8, r.height())
        return super().subControlRect(ctrl, option, sc, widget)

    # ── Polish / unpolish ─────────────────────────────────────────────────

    def polish(self, obj: QtWidgets.QWidget | QtGui.QPalette | QtWidgets.QApplication) -> None:  # type: ignore[override]
        if isinstance(obj, QtWidgets.QWidget):
            # Install hover tracking on interactive widgets
            if isinstance(obj, (QtWidgets.QAbstractButton,
                                 QtWidgets.QComboBox,
                                 QtWidgets.QAbstractSlider,
                                 QtWidgets.QSpinBox,
                                 QtWidgets.QDoubleSpinBox)):
                obj.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

            # Per-role font assignments
            self._polish_font(obj)

        super().polish(obj)

    def _polish_font(self, widget: QtWidgets.QWidget) -> None:
        """Assign Inter / JetBrains Mono with precise weight + letter-spacing
        to specific objectName roles.  Falls back gracefully if fonts are not
        registered (the system sans-serif will be used instead)."""
        try:
            from gui.fonts.loader import UI_FAMILY, MONO_FAMILY, TypeScale
        except ImportError:
            return

        W  = QtGui.QFont.Weight
        SP = QtGui.QFont.SpacingType
        HP = QtGui.QFont.HintingPreference

        def _soft(f: QtGui.QFont) -> QtGui.QFont:
            """Apply no-hinting for smooth, easy-on-the-eyes rendering."""
            f.setHintingPreference(HP.PreferNoHinting)
            return f

        name = widget.objectName()

        if name == "appTitle":
            f = QtGui.QFont(UI_FAMILY, TypeScale.APP_TITLE)
            f.setWeight(W(TypeScale.W_BOLD))
            f.setLetterSpacing(SP.PercentageSpacing, 98.0)   # -0.02em
            widget.setFont(_soft(f))

        elif name == "sectionTitle":
            f = QtGui.QFont(UI_FAMILY, TypeScale.SECTION_HDR)
            f.setWeight(W(TypeScale.W_SEMIBOLD))
            widget.setFont(_soft(f))

        elif name in ("sectionSubtitle", "mutedLabel"):
            f = QtGui.QFont(UI_FAMILY, TypeScale.LABEL)
            f.setWeight(W(TypeScale.W_REGULAR))
            widget.setFont(_soft(f))

        elif name == "sidebarSectionLabel":
            f = QtGui.QFont(UI_FAMILY, TypeScale.NAV_SECTION)
            f.setWeight(W(TypeScale.W_SEMIBOLD))
            f.setLetterSpacing(SP.PercentageSpacing, 108.0)  # +0.08em
            widget.setFont(_soft(f))

        elif name == "navButton":
            f = QtGui.QFont(UI_FAMILY, TypeScale.NAV_ITEM)
            f.setWeight(W(TypeScale.W_REGULAR))
            widget.setFont(_soft(f))

        elif name in ("libraryPath", "libStats"):
            f = QtGui.QFont(UI_FAMILY, TypeScale.SMALL)
            f.setWeight(W(TypeScale.W_REGULAR))
            widget.setFont(_soft(f))

        elif name == "statNumber":
            f = QtGui.QFont(UI_FAMILY, TypeScale.SMALL)
            f.setWeight(W(TypeScale.W_SEMIBOLD))
            widget.setFont(_soft(f))

        elif name == "statUnit":
            f = QtGui.QFont(UI_FAMILY, TypeScale.SMALL)
            f.setWeight(W(TypeScale.W_REGULAR))
            widget.setFont(_soft(f))

        elif name == "logText":
            f = QtGui.QFont(MONO_FAMILY, TypeScale.MONO)
            f.setWeight(W(TypeScale.W_REGULAR))
            widget.setFont(_soft(f))

        elif isinstance(widget, (QtWidgets.QAbstractButton,
                                  QtWidgets.QComboBox,
                                  QtWidgets.QSpinBox,
                                  QtWidgets.QDoubleSpinBox,
                                  QtWidgets.QLineEdit)):
            f = QtGui.QFont(UI_FAMILY, TypeScale.BUTTON)
            f.setWeight(W(TypeScale.W_REGULAR))
            widget.setFont(_soft(f))

    def unpolish(self, obj: QtWidgets.QWidget | QtWidgets.QApplication) -> None:  # type: ignore[override]
        super().unpolish(obj)
