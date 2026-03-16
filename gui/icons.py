"""AlphaDEX application icon — rendered programmatically, no external assets.

The icon is a dark rounded-square with five equalizer bars in an
indigo → violet gradient, communicating "music" instantly at every size.
Multiple resolutions are baked in so Windows picks the sharpest one for the
taskbar, title bar, and Alt-Tab switcher.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui


def make_app_icon() -> QtGui.QIcon:
    """Return a multi-resolution QIcon for the taskbar / title bar."""
    icon = QtGui.QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(_render(sz))
    return icon


def _render(sz: int) -> QtGui.QPixmap:
    pix = QtGui.QPixmap(sz, sz)
    pix.fill(QtCore.Qt.GlobalColor.transparent)

    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

    # ── Background: dark rounded square with subtle gradient ────────────────
    radius = sz * 0.22
    bg_path = QtGui.QPainterPath()
    bg_path.addRoundedRect(QtCore.QRectF(0, 0, sz, sz), radius, radius)

    bg_grad = QtGui.QLinearGradient(0.0, 0.0, float(sz), float(sz))
    bg_grad.setColorAt(0.0, QtGui.QColor("#1e2235"))
    bg_grad.setColorAt(1.0, QtGui.QColor("#0d1117"))
    p.fillPath(bg_path, QtGui.QBrush(bg_grad))

    # ── Equalizer bars ──────────────────────────────────────────────────────
    # Use 3 bars at tiny sizes (≤24 px) to stay legible; 5 bars otherwise.
    if sz <= 24:
        bar_heights = [0.52, 0.82, 0.42]
    else:
        bar_heights = [0.40, 0.68, 0.90, 0.58, 0.30]

    n       = len(bar_heights)
    pad     = sz * 0.17          # left / right padding
    bot_y   = sz * 0.78          # baseline
    top_y   = sz * 0.14          # ceiling of tallest bar
    avail_h = bot_y - top_y
    slot_w  = (sz - pad * 2) / n
    bar_w   = slot_w * 0.56      # bar takes 56 % of each slot; rest is gap

    for i, h_frac in enumerate(bar_heights):
        x       = pad + i * slot_w + (slot_w - bar_w) / 2.0
        bar_h   = avail_h * h_frac
        bar_top = bot_y - bar_h

        grad = QtGui.QLinearGradient(0.0, bar_top, 0.0, bot_y)
        grad.setColorAt(0.0, QtGui.QColor("#a78bfa"))  # violet at top
        grad.setColorAt(1.0, QtGui.QColor("#6366f1"))  # indigo at base

        corner_r = min(bar_w * 0.45, sz * 0.055)
        bar_path = QtGui.QPainterPath()
        bar_path.addRoundedRect(
            QtCore.QRectF(x, bar_top, bar_w, bar_h), corner_r, corner_r
        )
        p.fillPath(bar_path, QtGui.QBrush(grad))

    p.end()
    return pix
