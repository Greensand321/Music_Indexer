"""Backward-compatibility shim — theming is now handled by gui.themes."""
# The full theme engine lives in gui/themes/.
# This file is kept so that any legacy import of build_stylesheet() does not
# crash; it returns an empty string because AlphaDEXStyle + QPalette + the
# residual QSS in ThemeManager cover everything.

# Log-drawer colour constants (default dark / Midnight theme values).
# These are used by gui.widgets.log_drawer; the live theme overrides them
# via stylesheet but these fallbacks must exist for the import to succeed.
LOG_BG       = "#0d1117"
LOG_TEXT     = "#8b949e"
LOG_TEXT_OK  = "#4ade80"
LOG_TEXT_WARN = "#fbbf24"
LOG_TEXT_ERR  = "#f87171"


def build_stylesheet() -> str:  # noqa: D103
    return ""
