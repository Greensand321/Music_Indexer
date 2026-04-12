"""Theme token definitions — the single source of truth for every colour in the app."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeTokens:
    # ── Identity ──────────────────────────────────────────────────────────
    key:     str
    name:    str
    variant: str   # "dark" | "light"
    icon:    str   # emoji shown in swatch card

    # ── Sidebar ───────────────────────────────────────────────────────────
    sidebar_bg:      str
    sidebar_hover:   str
    sidebar_active:  str   # fill rect behind active item
    sidebar_accent:  str   # 3-px left border strip on active item
    sidebar_text:    str
    sidebar_hdr:     str   # section-header label colour

    # ── Content / workspace ───────────────────────────────────────────────
    content_bg:      str   # workspace/scroll-area background
    card_bg:         str
    card_border:     str
    card_shadow:     str   # rgba string for QGraphicsDropShadowEffect

    # ── Typography ────────────────────────────────────────────────────────
    text_primary:    str
    text_secondary:  str
    text_muted:      str
    text_inverse:    str   # text on accent-coloured surfaces

    # ── Interactive ───────────────────────────────────────────────────────
    accent:          str
    accent_hover:    str
    accent_pressed:  str
    success:         str
    warning:         str
    danger:          str
    danger_hover:    str

    # ── Inputs ────────────────────────────────────────────────────────────
    input_bg:        str
    input_border:    str
    input_focus:     str   # border colour when focused

    # ── Progress ──────────────────────────────────────────────────────────
    progress_bg:     str
    progress_fg:     str   # solid colour; gradient built in style.py

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_bg:          str   # inactive tab
    tab_active_bg:   str
    tab_text:        str
    tab_active_text: str

    # ── Log drawer ────────────────────────────────────────────────────────
    log_bg:          str
    log_handle:      str   # the dark strip at the top of the drawer
    log_text:        str
    log_ok:          str
    log_warn:        str
    log_err:         str

    # ── Swatch preview (theme picker card) ────────────────────────────────
    swatch_bg:       str   # large area (= sidebar_bg for dark, card_bg for light)
    swatch_accent:   str   # small accent circle


# ── Helper ────────────────────────────────────────────────────────────────────

def _d(**kw) -> dict:
    """Shared dark-theme defaults — overridden per theme."""
    base = dict(
        variant="dark",
        success="#22c55e",
        warning="#f59e0b",
        danger="#ef4444",
        danger_hover="#dc2626",
        log_ok="#4ade80",
        log_warn="#fbbf24",
        log_err="#f87171",
    )
    base.update(kw)
    return base


def _l(**kw) -> dict:
    """Shared light-theme defaults."""
    base = dict(
        variant="light",
        success="#16a34a",
        warning="#d97706",
        danger="#dc2626",
        danger_hover="#b91c1c",
        log_ok="#15803d",
        log_warn="#b45309",
        log_err="#b91c1c",
    )
    base.update(kw)
    return base


# ── Dark themes ───────────────────────────────────────────────────────────────

MIDNIGHT = ThemeTokens(**_d(
    key="midnight",      name="Midnight",    icon="🌙",
    sidebar_bg="#0d1117",  sidebar_hover="#1c2128",  sidebar_active="#1c2a3a",
    sidebar_accent="#58a6ff", sidebar_text="#c9d1d9", sidebar_hdr="#484f58",
    content_bg="#161b22",  card_bg="#1c2128",  card_border="#30363d",
    card_shadow="rgba(0,0,0,0.45)",
    text_primary="#e6edf3", text_secondary="#8b949e", text_muted="#484f58",
    text_inverse="#0d1117",
    accent="#58a6ff",      accent_hover="#79b8ff",   accent_pressed="#388bfd",
    input_bg="#0d1117",    input_border="#30363d",   input_focus="#58a6ff",
    progress_bg="#21262d", progress_fg="#58a6ff",
    tab_bg="#1c2128",      tab_active_bg="#0d1117",
    tab_text="#8b949e",    tab_active_text="#e6edf3",
    log_bg="#0d1117",      log_handle="#161b22",
    log_text="#8b949e",
    swatch_bg="#0d1117",   swatch_accent="#58a6ff",
))

OBSIDIAN = ThemeTokens(**_d(
    key="obsidian",      name="Obsidian",    icon="◼",
    sidebar_bg="#18181b",  sidebar_hover="#27272a",  sidebar_active="#3f3f46",
    sidebar_accent="#a78bfa", sidebar_text="#d4d4d8", sidebar_hdr="#52525b",
    content_bg="#1e1e24",  card_bg="#27272a",  card_border="#3f3f46",
    card_shadow="rgba(0,0,0,0.50)",
    text_primary="#fafafa",  text_secondary="#a1a1aa", text_muted="#52525b",
    text_inverse="#ffffff",
    accent="#a78bfa",      accent_hover="#c4b5fd",   accent_pressed="#7c3aed",
    input_bg="#18181b",    input_border="#3f3f46",   input_focus="#a78bfa",
    progress_bg="#27272a", progress_fg="#a78bfa",
    tab_bg="#27272a",      tab_active_bg="#18181b",
    tab_text="#71717a",    tab_active_text="#fafafa",
    log_bg="#18181b",      log_handle="#1e1e24",
    log_text="#71717a",
    swatch_bg="#18181b",   swatch_accent="#a78bfa",
))

GRAPHITE = ThemeTokens(**_d(
    key="graphite",      name="Graphite",    icon="◈",
    sidebar_bg="#1c1c1c",  sidebar_hover="#2a2a2a",  sidebar_active="#383838",
    sidebar_accent="#94a3b8", sidebar_text="#d4d4d4", sidebar_hdr="#5a5a5a",
    content_bg="#242424",  card_bg="#2e2e2e",  card_border="#404040",
    card_shadow="rgba(0,0,0,0.40)",
    text_primary="#f0f0f0",  text_secondary="#9a9a9a", text_muted="#5a5a5a",
    text_inverse="#1c1c1c",
    accent="#94a3b8",      accent_hover="#b0bec5",   accent_pressed="#607d8b",
    input_bg="#1c1c1c",    input_border="#404040",   input_focus="#94a3b8",
    progress_bg="#2e2e2e", progress_fg="#94a3b8",
    tab_bg="#2e2e2e",      tab_active_bg="#1c1c1c",
    tab_text="#707070",    tab_active_text="#f0f0f0",
    log_bg="#1c1c1c",      log_handle="#242424",
    log_text="#707070",
    swatch_bg="#1c1c1c",   swatch_accent="#94a3b8",
))

NAVY = ThemeTokens(**_d(
    key="navy",          name="Navy",        icon="⚓",
    sidebar_bg="#0a192f",  sidebar_hover="#112240",  sidebar_active="#1d3a5f",
    sidebar_accent="#64ffda", sidebar_text="#ccd6f6", sidebar_hdr="#495670",
    content_bg="#0e2040",  card_bg="#112240",  card_border="#233554",
    card_shadow="rgba(0,0,0,0.50)",
    text_primary="#e6f1ff",  text_secondary="#8892b0", text_muted="#495670",
    text_inverse="#0a192f",
    accent="#64ffda",      accent_hover="#a8ffee",   accent_pressed="#00b4d8",
    input_bg="#0a192f",    input_border="#233554",   input_focus="#64ffda",
    progress_bg="#112240", progress_fg="#64ffda",
    tab_bg="#112240",      tab_active_bg="#0a192f",
    tab_text="#8892b0",    tab_active_text="#e6f1ff",
    log_bg="#0a192f",      log_handle="#0e2040",
    log_text="#8892b0",
    swatch_bg="#0a192f",   swatch_accent="#64ffda",
))

TWILIGHT = ThemeTokens(**_d(
    key="twilight",      name="Twilight",    icon="🌆",
    sidebar_bg="#1a0a2e",  sidebar_hover="#2d1654",  sidebar_active="#3d1f6e",
    sidebar_accent="#c084fc", sidebar_text="#ddd6fe", sidebar_hdr="#6b4fa0",
    content_bg="#1e0f35",  card_bg="#2a1450",  card_border="#4a2580",
    card_shadow="rgba(0,0,0,0.55)",
    text_primary="#ede9fe",  text_secondary="#a78bfa", text_muted="#6b4fa0",
    text_inverse="#1a0a2e",
    accent="#c084fc",      accent_hover="#d8b4fe",   accent_pressed="#9333ea",
    input_bg="#1a0a2e",    input_border="#4a2580",   input_focus="#c084fc",
    progress_bg="#2a1450", progress_fg="#c084fc",
    tab_bg="#2a1450",      tab_active_bg="#1a0a2e",
    tab_text="#7c3aed",    tab_active_text="#ede9fe",
    log_bg="#1a0a2e",      log_handle="#1e0f35",
    log_text="#7c3aed",
    swatch_bg="#1a0a2e",   swatch_accent="#c084fc",
))

AURORA = ThemeTokens(**_d(
    key="aurora",        name="Aurora",      icon="🌌",
    sidebar_bg="#0d1f2d",  sidebar_hover="#14303f",  sidebar_active="#1a4a5a",
    sidebar_accent="#34d399", sidebar_text="#d1fae5", sidebar_hdr="#3d7a6a",
    content_bg="#0f2535",  card_bg="#14303f",  card_border="#1e4d5e",
    card_shadow="rgba(0,0,0,0.50)",
    text_primary="#ecfdf5",  text_secondary="#6ee7b7", text_muted="#3d7a6a",
    text_inverse="#0d1f2d",
    accent="#34d399",      accent_hover="#6ee7b7",   accent_pressed="#059669",
    input_bg="#0d1f2d",    input_border="#1e4d5e",   input_focus="#34d399",
    progress_bg="#14303f", progress_fg="#34d399",
    tab_bg="#14303f",      tab_active_bg="#0d1f2d",
    tab_text="#3d7a6a",    tab_active_text="#ecfdf5",
    log_bg="#0d1f2d",      log_handle="#0f2535",
    log_text="#3d7a6a",
    swatch_bg="#0d1f2d",   swatch_accent="#34d399",
))

FOREST = ThemeTokens(**_d(
    key="forest",        name="Forest",      icon="🌲",
    sidebar_bg="#0d1a0d",  sidebar_hover="#132413",  sidebar_active="#1e3a1e",
    sidebar_accent="#6ee7b7", sidebar_text="#d1fae5", sidebar_hdr="#3a5a3a",
    content_bg="#111f11",  card_bg="#162416",  card_border="#2a402a",
    card_shadow="rgba(0,0,0,0.50)",
    text_primary="#f0fdf4",  text_secondary="#86efac", text_muted="#3a5a3a",
    text_inverse="#0d1a0d",
    accent="#6ee7b7",      accent_hover="#a7f3d0",   accent_pressed="#059669",
    input_bg="#0d1a0d",    input_border="#2a402a",   input_focus="#6ee7b7",
    progress_bg="#162416", progress_fg="#6ee7b7",
    tab_bg="#162416",      tab_active_bg="#0d1a0d",
    tab_text="#3a5a3a",    tab_active_text="#f0fdf4",
    log_bg="#0d1a0d",      log_handle="#111f11",
    log_text="#3a5a3a",
    swatch_bg="#0d1a0d",   swatch_accent="#6ee7b7",
))

EMBER = ThemeTokens(**_d(
    key="ember",         name="Ember",       icon="🔥",
    sidebar_bg="#1a0d0d",  sidebar_hover="#2d1515",  sidebar_active="#4a2020",
    sidebar_accent="#fb923c", sidebar_text="#fed7aa", sidebar_hdr="#7a3a2a",
    content_bg="#1f1010",  card_bg="#2a1515",  card_border="#4a2828",
    card_shadow="rgba(0,0,0,0.55)",
    text_primary="#fff7ed",  text_secondary="#fdba74", text_muted="#7a3a2a",
    text_inverse="#1a0d0d",
    accent="#fb923c",      accent_hover="#fdba74",   accent_pressed="#ea580c",
    input_bg="#1a0d0d",    input_border="#4a2828",   input_focus="#fb923c",
    progress_bg="#2a1515", progress_fg="#fb923c",
    tab_bg="#2a1515",      tab_active_bg="#1a0d0d",
    tab_text="#7a3a2a",    tab_active_text="#fff7ed",
    log_bg="#1a0d0d",      log_handle="#1f1010",
    log_text="#7a3a2a",
    swatch_bg="#1a0d0d",   swatch_accent="#fb923c",
))


# ── Light themes ──────────────────────────────────────────────────────────────

PEARL = ThemeTokens(**_l(
    key="pearl",         name="Pearl",       icon="🪩",
    sidebar_bg="#f0f2f8",  sidebar_hover="#e2e6f0",  sidebar_active="#dce2f5",
    sidebar_accent="#6366f1", sidebar_text="#374151", sidebar_hdr="#9ca3af",
    content_bg="#f8f9fc",  card_bg="#ffffff",  card_border="#e5e7eb",
    card_shadow="rgba(0,0,0,0.08)",
    text_primary="#111827",  text_secondary="#6b7280", text_muted="#9ca3af",
    text_inverse="#ffffff",
    accent="#6366f1",      accent_hover="#4f46e5",   accent_pressed="#4338ca",
    input_bg="#ffffff",    input_border="#d1d5db",   input_focus="#6366f1",
    progress_bg="#e5e7eb", progress_fg="#6366f1",
    tab_bg="#f3f4f6",      tab_active_bg="#ffffff",
    tab_text="#9ca3af",    tab_active_text="#6366f1",
    log_bg="#1e1e2e",      log_handle="#2a2a3e",
    log_text="#8b949e",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#f0f2f8",   swatch_accent="#6366f1",
))

AZURE = ThemeTokens(**_l(
    key="azure",         name="Azure",       icon="💙",
    sidebar_bg="#e8f0fe",  sidebar_hover="#dce8fd",  sidebar_active="#c7d9fb",
    sidebar_accent="#3b82f6", sidebar_text="#1e3a5f", sidebar_hdr="#93c5fd",
    content_bg="#f0f6ff",  card_bg="#ffffff",  card_border="#dbeafe",
    card_shadow="rgba(59,130,246,0.10)",
    text_primary="#1e3a5f",  text_secondary="#3b6ea5", text_muted="#93c5fd",
    text_inverse="#ffffff",
    accent="#3b82f6",      accent_hover="#2563eb",   accent_pressed="#1d4ed8",
    input_bg="#ffffff",    input_border="#bfdbfe",   input_focus="#3b82f6",
    progress_bg="#dbeafe", progress_fg="#3b82f6",
    tab_bg="#eff6ff",      tab_active_bg="#ffffff",
    tab_text="#93c5fd",    tab_active_text="#3b82f6",
    log_bg="#1e293b",      log_handle="#273449",
    log_text="#8b949e",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#e8f0fe",   swatch_accent="#3b82f6",
))

BLOSSOM = ThemeTokens(**_l(
    key="blossom",       name="Blossom",     icon="🌸",
    sidebar_bg="#fce7f3",  sidebar_hover="#fbd5e8",  sidebar_active="#f9a8d4",
    sidebar_accent="#ec4899", sidebar_text="#831843", sidebar_hdr="#f9a8d4",
    content_bg="#fff0f6",  card_bg="#ffffff",  card_border="#fce7f3",
    card_shadow="rgba(236,72,153,0.10)",
    text_primary="#500724",  text_secondary="#9d174d", text_muted="#f9a8d4",
    text_inverse="#ffffff",
    accent="#ec4899",      accent_hover="#db2777",   accent_pressed="#be185d",
    input_bg="#ffffff",    input_border="#fce7f3",   input_focus="#ec4899",
    progress_bg="#fce7f3", progress_fg="#ec4899",
    tab_bg="#fdf2f8",      tab_active_bg="#ffffff",
    tab_text="#f9a8d4",    tab_active_text="#ec4899",
    log_bg="#2d1b24",      log_handle="#3d2030",
    log_text="#8b949e",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#fce7f3",   swatch_accent="#ec4899",
))

MEADOW = ThemeTokens(**_l(
    key="meadow",        name="Meadow",      icon="🌿",
    sidebar_bg="#dcfce7",  sidebar_hover="#bbf7d0",  sidebar_active="#a7f3d0",
    sidebar_accent="#10b981", sidebar_text="#064e3b", sidebar_hdr="#6ee7b7",
    content_bg="#f0fdf4",  card_bg="#ffffff",  card_border="#d1fae5",
    card_shadow="rgba(16,185,129,0.10)",
    text_primary="#064e3b",  text_secondary="#065f46", text_muted="#6ee7b7",
    text_inverse="#ffffff",
    accent="#10b981",      accent_hover="#059669",   accent_pressed="#047857",
    input_bg="#ffffff",    input_border="#d1fae5",   input_focus="#10b981",
    progress_bg="#d1fae5", progress_fg="#10b981",
    tab_bg="#ecfdf5",      tab_active_bg="#ffffff",
    tab_text="#6ee7b7",    tab_active_text="#10b981",
    log_bg="#1a2e1a",      log_handle="#223322",
    log_text="#8b949e",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#dcfce7",   swatch_accent="#10b981",
))

LAVENDER = ThemeTokens(**_l(
    key="lavender",      name="Lavender",    icon="💜",
    sidebar_bg="#f3e8ff",  sidebar_hover="#e9d5ff",  sidebar_active="#ddd6fe",
    sidebar_accent="#8b5cf6", sidebar_text="#3b0764", sidebar_hdr="#c4b5fd",
    content_bg="#faf5ff",  card_bg="#ffffff",  card_border="#ede9fe",
    card_shadow="rgba(139,92,246,0.10)",
    text_primary="#2e1065",  text_secondary="#6d28d9", text_muted="#c4b5fd",
    text_inverse="#ffffff",
    accent="#8b5cf6",      accent_hover="#7c3aed",   accent_pressed="#6d28d9",
    input_bg="#ffffff",    input_border="#ede9fe",   input_focus="#8b5cf6",
    progress_bg="#ede9fe", progress_fg="#8b5cf6",
    tab_bg="#f5f3ff",      tab_active_bg="#ffffff",
    tab_text="#c4b5fd",    tab_active_text="#8b5cf6",
    log_bg="#1e0f35",      log_handle="#2d1654",
    log_text="#8b949e",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#f3e8ff",   swatch_accent="#8b5cf6",
))

SUNSET = ThemeTokens(**_l(
    key="sunset",        name="Sunset",      icon="🌅",
    sidebar_bg="#fff7ed",  sidebar_hover="#ffedd5",  sidebar_active="#fed7aa",
    sidebar_accent="#f97316", sidebar_text="#431407", sidebar_hdr="#fdba74",
    content_bg="#fffbf5",  card_bg="#ffffff",  card_border="#ffedd5",
    card_shadow="rgba(249,115,22,0.10)",
    text_primary="#431407",  text_secondary="#9a3412", text_muted="#fdba74",
    text_inverse="#ffffff",
    accent="#f97316",      accent_hover="#ea580c",   accent_pressed="#c2410c",
    input_bg="#ffffff",    input_border="#ffedd5",   input_focus="#f97316",
    progress_bg="#ffedd5", progress_fg="#f97316",
    tab_bg="#fff7ed",      tab_active_bg="#ffffff",
    tab_text="#fdba74",    tab_active_text="#f97316",
    log_bg="#1f1008",      log_handle="#2d1810",
    log_text="#8b949e",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#fff7ed",   swatch_accent="#f97316",
))


# ── Registry ──────────────────────────────────────────────────────────────────

THEMES: dict[str, ThemeTokens] = {
    t.key: t for t in [
        MIDNIGHT, OBSIDIAN, GRAPHITE, NAVY, TWILIGHT, AURORA, FOREST, EMBER,
        PEARL, AZURE, BLOSSOM, MEADOW, LAVENDER, SUNSET,
    ]
}

DARK_THEMES  = [k for k, t in THEMES.items() if t.variant == "dark"]
LIGHT_THEMES = [k for k, t in THEMES.items() if t.variant == "light"]

DEFAULT_DARK  = "midnight"
DEFAULT_LIGHT = "pearl"
