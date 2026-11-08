"""Cached appearance settings that painting code reads (never config I/O in paint).

Config keys (read and written through ``config.load_config/save_config``):

==========================  ===================================================
``backdrop_mode``           "art" (album covers, Liquid Glass's default) ·
                            "gradient" · "plain"
``bg_gradient_enabled``     legacy switch, still honoured: when
                            ``backdrop_mode`` is absent, ``False`` means
                            "plain" and ``True`` the theme's default. Saving
                            Settings keeps it in step (``False`` == "plain").
``glass_accent``            Liquid Glass accent: blue · peach · mint · lilac
                            (read by ``ThemeManager``)
``reduce_transparency``     opaque surfaces (read by ``ThemeManager``)
``reduced_motion``          see ``gui.themes.motion``
==========================  ===================================================

What a theme actually shows depends on its material: classic themes have no
album-art backdrop, so "art" falls back to their gradient there.

The value is read once and cached; ``refresh()`` drops the cache (Settings
calls it after saving, the theme manager on every switch).
"""
from __future__ import annotations

BACKDROP_MODES: tuple[str, ...] = ("art", "gradient", "plain")
BACKDROP_LABELS: dict[str, str] = {
    "art": "Album art",
    "gradient": "Gradient",
    "plain": "Plain",
}
MODE_KEY = "backdrop_mode"
LEGACY_GRADIENT_KEY = "bg_gradient_enabled"

_mode_cache: str | None = None


def mode_from_config(cfg: dict) -> str:
    """The saved backdrop mode ("art" when nothing says otherwise)."""
    mode = cfg.get(MODE_KEY)
    if mode in BACKDROP_MODES:
        return mode
    if cfg.get(LEGACY_GRADIENT_KEY, True) is False:
        return "plain"
    return "art"


def backdrop_mode() -> str:
    """The saved backdrop mode, cached (safe to call from paint code)."""
    global _mode_cache
    if _mode_cache is None:
        try:
            from config import load_config
            _mode_cache = mode_from_config(load_config())
        except Exception:
            _mode_cache = "art"
    return _mode_cache


def effective_mode(tokens, mode: str | None = None) -> str:
    """What the backdrop shows for ``tokens``: classic themes have no album
    art, so "art" becomes their gradient."""
    mode = mode or backdrop_mode()
    material = getattr(tokens, "material", None)
    if mode == "art" and (material is None or not material.is_glass):
        return "gradient"
    return mode


def invalidate() -> None:
    """Forget the cached value; the next read loads it once."""
    global _mode_cache
    _mode_cache = None


def refresh() -> str:
    """Re-read now. Returns the mode."""
    invalidate()
    return backdrop_mode()


def config_values(mode: str) -> dict:
    """The keys a Settings save writes for ``mode`` (legacy key in step)."""
    mode = mode if mode in BACKDROP_MODES else "art"
    return {MODE_KEY: mode, LEGACY_GRADIENT_KEY: mode != "plain"}
