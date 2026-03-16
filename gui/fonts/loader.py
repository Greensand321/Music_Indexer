"""Font loader — registers Inter and JetBrains Mono with QFontDatabase.

Call ``load_fonts()`` once, before creating the main window.  After that,
refer to fonts by their family names:

    "Inter"            — UI font (all weights via InterVariable.ttf)
    "JetBrains Mono"   — Monospace font for the log drawer and path labels

If the bundled files are missing the function logs a warning and falls back
gracefully; the rest of the application continues with the system sans-serif.
"""
from __future__ import annotations

import logging
from pathlib import Path

_FONTS_DIR = Path(__file__).parent
_log = logging.getLogger(__name__)

# Files to register, in priority order.
# InterVariable.ttf is a single variable-weight file that covers all weights;
# the static files are registered as well so Qt can find exact weight matches
# on platforms that do not support variable fonts.
_FONT_FILES = [
    "InterVariable.ttf",
    "Inter-Regular.ttf",
    "Inter-Medium.ttf",
    "Inter-SemiBold.ttf",
    "Inter-Bold.ttf",
    "JetBrainsMono-Regular.ttf",
    "JetBrainsMono-Medium.ttf",
]

_loaded = False


def load_fonts() -> dict[str, bool]:
    """Register all bundled fonts with ``QFontDatabase``.

    Returns a dict mapping each file name to True (loaded) / False (missing).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _loaded
    if _loaded:
        return {}

    from gui.compat import QtGui

    results: dict[str, bool] = {}
    for name in _FONT_FILES:
        path = _FONTS_DIR / name
        if not path.exists():
            _log.warning("Bundled font missing: %s", path)
            results[name] = False
            continue
        fid = QtGui.QFontDatabase.addApplicationFont(str(path))
        if fid == -1:
            _log.warning("QFontDatabase failed to load: %s", path)
            results[name] = False
        else:
            results[name] = True

    _loaded = True

    loaded = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    if loaded:
        _log.debug("Fonts loaded: %s", ", ".join(loaded))
    if failed:
        _log.warning("Fonts failed: %s", ", ".join(failed))

    return results


def inter_available() -> bool:
    """Return True if Inter was successfully registered."""
    from gui.compat import QtGui
    return "Inter" in QtGui.QFontDatabase.families()


def jetbrains_mono_available() -> bool:
    """Return True if JetBrains Mono was successfully registered."""
    from gui.compat import QtGui
    return "JetBrains Mono" in QtGui.QFontDatabase.families()


# ── Font role constants ────────────────────────────────────────────────────────
# Used by AlphaDEXStyle and ThemeManager to build consistent type scales.

UI_FAMILY = "Inter"
MONO_FAMILY = "JetBrains Mono"

# Fallback stacks — used in QSS / QFont if the primary is not available
UI_STACK = '"Inter", "Segoe UI Variable", "SF Pro Text", "Helvetica Neue", Arial, sans-serif'
MONO_STACK = '"JetBrains Mono", "Cascadia Code", "Consolas", "Menlo", monospace'


class TypeScale:
    """Per-role font size and weight constants.

    Sizes are in points (pt).  Qt renders point sizes with DPI awareness on
    all platforms.
    """

    # ── Sizes (pt) ────────────────────────────────────────────────────────
    APP_TITLE   = 16    # "AlphaDEX" brand in top bar
    SECTION_HDR = 14    # Workspace section headings
    BODY        = 11    # Default widget text
    LABEL       = 10    # Secondary / support text
    SMALL       = 9     # Muted captions, badge text, sub-labels
    MONO        = 10    # Log drawer, path labels

    # ── Sidebar ───────────────────────────────────────────────────────────
    NAV_ITEM    = 10
    NAV_SECTION = 8     # Section headers in ALL CAPS

    # ── Controls ─────────────────────────────────────────────────────────
    BUTTON      = 11
    INPUT       = 11
    TAB         = 10

    # ── Weights (QFont.Weight enum values) ────────────────────────────────
    W_THIN      = 100
    W_LIGHT     = 300
    W_REGULAR   = 400
    W_MEDIUM    = 500
    W_SEMIBOLD  = 600
    W_BOLD      = 700

    # ── Letter-spacing (em) ───────────────────────────────────────────────
    # Expressed as fractions; multiply by point size × 4/3 for QFont::setLetterSpacing
    LS_TIGHT    = -0.02   # Headings / titles
    LS_NORMAL   =  0.00   # Body
    LS_WIDE     =  0.08   # All-caps section labels
