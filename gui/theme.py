"""Backward-compatibility shim — theming is now handled by gui.themes."""
# The full theme engine lives in gui/themes/.
# This file is kept so that any legacy import of build_stylesheet() does not
# crash; it returns an empty string because AlphaDEXStyle + QPalette + the
# residual QSS in ThemeManager cover everything.


def build_stylesheet() -> str:  # noqa: D103
    return ""
