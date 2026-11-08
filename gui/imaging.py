"""QImage helpers for pre-blurred glass: worker threads only (QImage + numpy).

On the web, glass blurs what is behind it at paint time. Qt has no cheap
live blur, so AlphaDEX blurs *images* once, off the GUI thread, and paints
the result: the window backdrop (``gui/widgets/backdrop.py``) and the
landing card's frost (``gui/widgets/landing.py``).

``gaussian_blur`` approximates a Gaussian with three box blurs (cumulative
sums in numpy), the standard fast trick; ``blur_image`` runs it on a
downscaled copy and scales the result back up smoothly, which is both much
faster and visually identical for large radii.
"""
from __future__ import annotations

import math

from gui.compat import QtCore, QtGui


def _np():
    try:
        import numpy as np
        return np
    except ImportError:  # pragma: no cover - numpy is a core requirement
        return None


def to_array(img: QtGui.QImage):
    """QImage → H×W×3 float32 array in B, G, R order (alpha dropped)."""
    np = _np()
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    raw = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes())
    return raw.reshape(h, bpl)[:, : w * 4].reshape(h, w, 4)[..., :3].astype(np.float32)


def from_array(bgr) -> QtGui.QImage:
    """H×W×3 float array (B, G, R) → an opaque, self-owned QImage."""
    np = _np()
    h, w, _ = bgr.shape
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = np.clip(bgr + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return QtGui.QImage(out.tobytes(), w, h, w * 4, QtGui.QImage.Format.Format_RGB32).copy()


def box_radii(sigma: float, passes: int = 3) -> list[int]:
    """Box radii whose ``passes`` successive box blurs approximate a Gaussian."""
    w_ideal = math.sqrt(12.0 * sigma * sigma / passes + 1.0)
    wl = int(math.floor(w_ideal))
    if wl % 2 == 0:
        wl -= 1
    wu = wl + 2
    m = round((12.0 * sigma * sigma - passes * wl * wl - 4 * passes * wl - 3 * passes)
              / (-4.0 * wl - 4.0))
    return [max(0, ((wl if i < m else wu) - 1) // 2) for i in range(passes)]


def _box(a, r: int, axis: int):
    np = _np()
    if r <= 0:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / float(2 * r + 1)


def gaussian_blur(a, sigma: float):
    """Fast approximate Gaussian blur of an H×W×C float array (3 box passes)."""
    for r in box_radii(sigma):
        a = _box(_box(a, r, 1), r, 0)
    return a


def saturate(bgr, s: float):
    """CSS ``saturate(s)`` on a B,G,R float array."""
    np = _np()
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    nr = (0.213 + 0.787 * s) * r + (0.715 - 0.715 * s) * g + (0.072 - 0.072 * s) * b
    ng = (0.213 - 0.213 * s) * r + (0.715 + 0.285 * s) * g + (0.072 - 0.072 * s) * b
    nb = (0.213 - 0.213 * s) * r + (0.715 - 0.715 * s) * g + (0.072 + 0.928 * s) * b
    return np.clip(np.stack([nb, ng, nr], axis=-1), 0.0, 255.0)


def blur_image(img: QtGui.QImage, sigma: float, *, scale: int = 4,
               saturation: float = 1.0) -> QtGui.QImage:
    """``img`` blurred by ``sigma`` px (and optionally saturated), same size.

    Blurs a 1/``scale`` copy, then scales back up smoothly. Falls back to a
    down/up-scale blur without numpy."""
    w, h = img.width(), img.height()
    if w <= 0 or h <= 0:
        return QtGui.QImage(img)
    sw, sh = max(1, w // scale), max(1, h // scale)
    small = img.scaled(sw, sh, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                       QtCore.Qt.TransformationMode.SmoothTransformation)
    if _np() is None:  # pragma: no cover
        tiny = small.scaled(max(1, sw // 4), max(1, sh // 4),
                            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                            QtCore.Qt.TransformationMode.SmoothTransformation)
        small = tiny
    else:
        arr = gaussian_blur(to_array(small), sigma / scale)
        if saturation != 1.0:
            arr = saturate(arr, saturation)
        small = from_array(arr)
    out = small.scaled(w, h, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                       QtCore.Qt.TransformationMode.SmoothTransformation)
    out.setDevicePixelRatio(img.devicePixelRatio())
    return out
