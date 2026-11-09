"""Theme token definitions — the single source of truth for every colour in the app.

Three layers live here:

* **Colour tokens** (``ThemeTokens``): one opaque hex per role, per theme. Every
  theme has them; they are what the palette, the contrast tests and older
  widgets read.
* **Material** (``ThemeTokens.material``): how surfaces are painted, per
  elevation — ``sidebar``, ``card``, ``raised`` (pill buttons, chips, the
  active nav item, the segmented-control thumb), ``capsule`` (the floating
  Now Playing bar) and ``well`` (inset fields, stat tiles, list rows) — plus
  the backdrop. Liquid Glass is a translucent material over an album-art
  backdrop; the classic themes get an opaque material derived from their
  colour tokens, so both render through the same components.
* **Scales** (theme-independent): the type ramp (``TYPE``), spacing
  (``SPACE``) and radii (``RADIUS``), so pages ask for a role, never a literal.

Surface colours are CSS-style strings: ``"#rrggbb"`` or ``"rgba(r,g,b,a)"``
with ``a`` in 0..1; ``gui.themes.glass.qcolor`` parses both.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

# Fully rounded corners ("pill"): painters clamp a radius to half the height.
PILL = 999


@dataclass(frozen=True)
class Surface:
    """How one elevation of a material is painted (see ``gui.themes.glass``)."""
    fill: str
    border: str = ""          # 1 px border; "" = none
    highlight: str = ""       # 1 px inner top highlight; "" = none
    shadow: str = ""          # soft shadow colour; "" = none
    blur: int = 0             # shadow blur, px (CSS box-shadow blur radius)
    offset_y: int = 0         # shadow drop, px
    radius: int = 26          # corner radius; PILL = fully rounded


@dataclass(frozen=True)
class Material:
    """Surfaces per elevation, overlays, and the backdrop of one theme."""
    kind: str                 # "glass" | "classic"
    sidebar: Surface
    card: Surface
    raised: Surface
    capsule: Surface
    well: Surface
    hover: str                # hovered rows / idle items
    pressed: str              # pressed overlay
    selection: str            # selected rows (palette Highlight)
    selection_text: str       # text on a selection
    hairline: str             # row separators, dividers
    knob: str                 # slider knobs
    track: str                # slider / progress tracks
    view: str                 # item views (tables, trees, lists)
    view_alt: str             # alternate rows
    backdrop_base: str        # solid colour under everything (also dialogs)
    backdrop_tone: str        # average colour of the finished backdrop
    backdrop_scrim: str = ""  # laid over the album-art wall ("" = none)
    backdrop: str = "gradient"   # default backdrop mode: "art" | "gradient" | "plain"
    # Album-art wall (Liquid Glass): 8 columns, gap 18, radius 24, blurred
    # ~46 px, saturation 150 %, opacity .95, bleeding 80 px past the edges.
    wall_columns: int = 8
    wall_gap: int = 18
    wall_radius: int = 24
    blur_px: int = 46
    saturation: float = 1.5
    art_opacity: float = 0.95
    bleed: int = 80

    @property
    def is_glass(self) -> bool:
        return self.kind == "glass"

    def surface(self, name: str) -> Surface:
        return getattr(self, name)


SURFACE_NAMES = ("sidebar", "card", "raised", "capsule", "well")


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

    # ── Material (surfaces + backdrop) ────────────────────────────────────
    # None only while a theme is being declared: the registry below fills a
    # classic (opaque) material for every theme that does not bring its own.
    material: Material | None = field(default=None, compare=True)

    @property
    def is_glass(self) -> bool:
        return self.material is not None and self.material.is_glass

    @property
    def accent_ink(self) -> str:
        """The accent for *text on an accent tint* (the playing row, a
        proposed value): the accent itself under Liquid Glass, its darker
        shade on light themes and its lighter shade on dark classic themes,
        so it keeps 4.5:1 where the plain accent drops to ~3.7:1."""
        if self.is_glass:
            return self.accent
        return self.accent_pressed if self.variant == "light" else self.accent_hover


# ── Helper ────────────────────────────────────────────────────────────────────

def _d(**kw) -> dict:
    """Shared dark-theme defaults — overridden per theme.

    ``danger`` is red-400, not red-500: red-500 fell to 3.6:1 as text on the
    darker cards (Graphite, Aurora), and the dark label of a danger button
    to 3.5:1 on its hover fill (tests/test_material_contrast.py)."""
    base = dict(
        variant="dark",
        success="#22c55e",
        warning="#f59e0b",
        danger="#f87171",
        danger_hover="#fca5a5",
        log_ok="#4ade80",
        log_warn="#fbbf24",
        log_err="#f87171",
    )
    base.update(kw)
    return base


def _l(**kw) -> dict:
    """Shared light-theme defaults.

    The status colours are the 700 shades: as text on white cards (a
    "Flagged" count, a layout warning) and under the white label of a
    success button, the 600 shades only reached 2.9-3.3:1."""
    base = dict(
        variant="light",
        success="#15803d",
        warning="#b45309",
        danger="#b91c1c",
        danger_hover="#991b1b",
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
    text_primary="#e6edf3", text_secondary="#959da7", text_muted="#848b93",
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
    text_primary="#fafafa",  text_secondary="#a1a1aa", text_muted="#909096",
    text_inverse="#18181b",
    accent="#a78bfa",      accent_hover="#c4b5fd",   accent_pressed="#9467f4",
    input_bg="#18181b",    input_border="#3f3f46",   input_focus="#a78bfa",
    progress_bg="#27272a", progress_fg="#a78bfa",
    tab_bg="#27272a",      tab_active_bg="#18181b",
    tab_text="#8f8f96",    tab_active_text="#fafafa",
    log_bg="#18181b",      log_handle="#1e1e24",
    log_text="#87878e",
    swatch_bg="#18181b",   swatch_accent="#a78bfa",
))

GRAPHITE = ThemeTokens(**_d(
    key="graphite",      name="Graphite",    icon="◈",
    sidebar_bg="#1c1c1c",  sidebar_hover="#2a2a2a",  sidebar_active="#383838",
    sidebar_accent="#94a3b8", sidebar_text="#d4d4d4", sidebar_hdr="#5a5a5a",
    content_bg="#242424",  card_bg="#2e2e2e",  card_border="#404040",
    card_shadow="rgba(0,0,0,0.40)",
    text_primary="#f0f0f0",  text_secondary="#a6a6a6", text_muted="#989898",
    text_inverse="#1c1c1c",
    accent="#94a3b8",      accent_hover="#b0bec5",   accent_pressed="#708999",
    input_bg="#1c1c1c",    input_border="#404040",   input_focus="#94a3b8",
    progress_bg="#2e2e2e", progress_fg="#94a3b8",
    tab_bg="#2e2e2e",      tab_active_bg="#1c1c1c",
    tab_text="#989898",    tab_active_text="#f0f0f0",
    log_bg="#1c1c1c",      log_handle="#242424",
    log_text="#8c8c8c",
    swatch_bg="#1c1c1c",   swatch_accent="#94a3b8",
))

NAVY = ThemeTokens(**_d(
    key="navy",          name="Navy",        icon="⚓",
    sidebar_bg="#0a192f",  sidebar_hover="#112240",  sidebar_active="#1d3a5f",
    sidebar_accent="#64ffda", sidebar_text="#ccd6f6", sidebar_hdr="#495670",
    content_bg="#0e2040",  card_bg="#112240",  card_border="#233554",
    card_shadow="rgba(0,0,0,0.50)",
    text_primary="#e6f1ff",  text_secondary="#949dba", text_muted="#808ca2",
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
    text_primary="#ede9fe",  text_secondary="#a78bfa", text_muted="#9580be",
    text_inverse="#1a0a2e",
    accent="#c084fc",      accent_hover="#d8b4fe",   accent_pressed="#a655f2",
    input_bg="#1a0a2e",    input_border="#4a2580",   input_focus="#c084fc",
    progress_bg="#2a1450", progress_fg="#c084fc",
    tab_bg="#2a1450",      tab_active_bg="#1a0a2e",
    tab_text="#9f70f2",    tab_active_text="#ede9fe",
    log_bg="#1a0a2e",      log_handle="#1e0f35",
    log_text="#9864f1",
    swatch_bg="#1a0a2e",   swatch_accent="#c084fc",
))

AURORA = ThemeTokens(**_d(
    key="aurora",        name="Aurora",      icon="🌌",
    sidebar_bg="#0d1f2d",  sidebar_hover="#14303f",  sidebar_active="#1a4a5a",
    sidebar_accent="#34d399", sidebar_text="#d1fae5", sidebar_hdr="#3d7a6a",
    content_bg="#0f2535",  card_bg="#14303f",  card_border="#1e4d5e",
    card_shadow="rgba(0,0,0,0.50)",
    text_primary="#ecfdf5",  text_secondary="#6ee7b7", text_muted="#6e9f91",
    text_inverse="#0d1f2d",
    accent="#34d399",      accent_hover="#6ee7b7",   accent_pressed="#07996b",
    input_bg="#0d1f2d",    input_border="#1e4d5e",   input_focus="#34d399",
    progress_bg="#14303f", progress_fg="#34d399",
    tab_bg="#14303f",      tab_active_bg="#0d1f2d",
    tab_text="#6e9f91",    tab_active_text="#ecfdf5",
    log_bg="#0d1f2d",      log_handle="#0f2535",
    log_text="#629588",
    swatch_bg="#0d1f2d",   swatch_accent="#34d399",
))

FOREST = ThemeTokens(**_d(
    key="forest",        name="Forest",      icon="🌲",
    sidebar_bg="#0d1a0d",  sidebar_hover="#132413",  sidebar_active="#1e3a1e",
    sidebar_accent="#6ee7b7", sidebar_text="#d1fae5", sidebar_hdr="#3a5a3a",
    content_bg="#111f11",  card_bg="#162416",  card_border="#2a402a",
    card_shadow="rgba(0,0,0,0.50)",
    text_primary="#f0fdf4",  text_secondary="#86efac", text_muted="#769077",
    text_inverse="#0d1a0d",
    accent="#6ee7b7",      accent_hover="#a7f3d0",   accent_pressed="#059669",
    input_bg="#0d1a0d",    input_border="#2a402a",   input_focus="#6ee7b7",
    progress_bg="#162416", progress_fg="#6ee7b7",
    tab_bg="#162416",      tab_active_bg="#0d1a0d",
    tab_text="#769077",    tab_active_text="#f0fdf4",
    log_bg="#0d1a0d",      log_handle="#111f11",
    log_text="#718c72",
    swatch_bg="#0d1a0d",   swatch_accent="#6ee7b7",
))

EMBER = ThemeTokens(**_d(
    key="ember",         name="Ember",       icon="🔥",
    sidebar_bg="#1a0d0d",  sidebar_hover="#2d1515",  sidebar_active="#4a2020",
    sidebar_accent="#fb923c", sidebar_text="#fed7aa", sidebar_hdr="#7a3a2a",
    content_bg="#1f1010",  card_bg="#2a1515",  card_border="#4a2828",
    card_shadow="rgba(0,0,0,0.55)",
    text_primary="#fff7ed",  text_secondary="#fdba74", text_muted="#a77a6c",
    text_inverse="#1a0d0d",
    accent="#fb923c",      accent_hover="#fdba74",   accent_pressed="#ea580c",
    input_bg="#1a0d0d",    input_border="#4a2828",   input_focus="#fb923c",
    progress_bg="#2a1515", progress_fg="#fb923c",
    tab_bg="#2a1515",      tab_active_bg="#1a0d0d",
    tab_text="#a77a6c",    tab_active_text="#fff7ed",
    log_bg="#1a0d0d",      log_handle="#1f1010",
    log_text="#a27364",
    swatch_bg="#1a0d0d",   swatch_accent="#fb923c",
))


# ── Light themes ──────────────────────────────────────────────────────────────

PEARL = ThemeTokens(**_l(
    key="pearl",         name="Pearl",       icon="🪩",
    sidebar_bg="#f0f2f8",  sidebar_hover="#e2e6f0",  sidebar_active="#dce2f5",
    sidebar_accent="#6366f1", sidebar_text="#374151", sidebar_hdr="#9ca3af",
    content_bg="#f8f9fc",  card_bg="#ffffff",  card_border="#e5e7eb",
    card_shadow="rgba(0,0,0,0.08)",
    text_primary="#111827",  text_secondary="#666d7b", text_muted="#6a717e",
    text_inverse="#ffffff",
    accent="#5e5feb",      accent_hover="#4f46e5",   accent_pressed="#4338ca",
    input_bg="#ffffff",    input_border="#d1d5db",   input_focus="#6366f1",
    progress_bg="#e5e7eb", progress_fg="#6366f1",
    tab_bg="#f3f4f6",      tab_active_bg="#ffffff",
    tab_text="#676e7b",    tab_active_text="#5e5feb",
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
    text_primary="#1e3a5f",  text_secondary="#3b6ea5", text_muted="#4c709d",
    text_inverse="#ffffff",
    accent="#2c67e7",      accent_hover="#2563eb",   accent_pressed="#1d4ed8",
    input_bg="#ffffff",    input_border="#bfdbfe",   input_focus="#3b82f6",
    progress_bg="#dbeafe", progress_fg="#3b82f6",
    tab_bg="#eff6ff",      tab_active_bg="#ffffff",
    tab_text="#4c709d",    tab_active_text="#2c67e7",
    log_bg="#1e293b",      log_handle="#273449",
    log_text="#959ea7",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#e8f0fe",   swatch_accent="#3b82f6",
))

BLOSSOM = ThemeTokens(**_l(
    key="blossom",       name="Blossom",     icon="🌸",
    sidebar_bg="#fce7f3",  sidebar_hover="#fbd5e8",  sidebar_active="#f9a8d4",
    sidebar_accent="#ec4899", sidebar_text="#831843", sidebar_hdr="#f9a8d4",
    content_bg="#fff0f6",  card_bg="#ffffff",  card_border="#fce7f3",
    card_shadow="rgba(236,72,153,0.10)",
    text_primary="#500724",  text_secondary="#9d174d", text_muted="#a15478",
    text_inverse="#ffffff",
    accent="#cd2871",      accent_hover="#b42363",   accent_pressed="#991e54",
    input_bg="#ffffff",    input_border="#fce7f3",   input_focus="#ec4899",
    progress_bg="#fce7f3", progress_fg="#ec4899",
    tab_bg="#fdf2f8",      tab_active_bg="#ffffff",
    tab_text="#a3567a",    tab_active_text="#cd2871",
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
    text_primary="#064e3b",  text_secondary="#065f46", text_muted="#277f63",
    text_inverse="#ffffff",
    accent="#06825d",      accent_hover="#057252",   accent_pressed="#046146",
    input_bg="#ffffff",    input_border="#d1fae5",   input_focus="#10b981",
    progress_bg="#d1fae5", progress_fg="#10b981",
    tab_bg="#ecfdf5",      tab_active_bg="#ffffff",
    tab_text="#277f63",    tab_active_text="#06825d",
    log_bg="#1a2e1a",      log_handle="#223322",
    log_text="#9098a2",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#dcfce7",   swatch_accent="#10b981",
))

LAVENDER = ThemeTokens(**_l(
    key="lavender",      name="Lavender",    icon="💜",
    sidebar_bg="#f3e8ff",  sidebar_hover="#e9d5ff",  sidebar_active="#ddd6fe",
    sidebar_accent="#8b5cf6", sidebar_text="#3b0764", sidebar_hdr="#c4b5fd",
    content_bg="#faf5ff",  card_bg="#ffffff",  card_border="#ede9fe",
    card_shadow="rgba(139,92,246,0.10)",
    text_primary="#2e1065",  text_secondary="#6d28d9", text_muted="#7962b1",
    text_inverse="#ffffff",
    accent="#824dee",      accent_hover="#7c3aed",   accent_pressed="#6d28d9",
    input_bg="#ffffff",    input_border="#ede9fe",   input_focus="#8b5cf6",
    progress_bg="#ede9fe", progress_fg="#8b5cf6",
    tab_bg="#f5f3ff",      tab_active_bg="#ffffff",
    tab_text="#7861af",    tab_active_text="#824dee",
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
    text_primary="#431407",  text_secondary="#9a3412", text_muted="#9e653c",
    text_inverse="#ffffff",
    accent="#c9480d",      accent_hover="#b13f0b",   accent_pressed="#963609",
    input_bg="#ffffff",    input_border="#ffedd5",   input_focus="#f97316",
    progress_bg="#ffedd5", progress_fg="#f97316",
    tab_bg="#fff7ed",      tab_active_bg="#ffffff",
    tab_text="#9c643b",    tab_active_text="#c9480d",
    log_bg="#1f1008",      log_handle="#2d1810",
    log_text="#8b949e",    log_ok="#4ade80", log_warn="#fbbf24", log_err="#f87171",
    swatch_bg="#fff7ed",   swatch_accent="#f97316",
))


# ── Liquid Glass ──────────────────────────────────────────────────────────────
#
# The owner's chosen direction (docs/redesign: "Refined Liquid Glass"). The
# hex tokens are the *opaque equivalents* of the glass surfaces over the
# scrimmed album-art backdrop (what the contrast tests and older widgets
# read); the translucent truth is in the material.

GLASS_MATERIAL = Material(
    kind="glass",
    sidebar=Surface("rgba(22,26,38,0.46)", "rgba(255,255,255,0.14)", "rgba(255,255,255,0.24)",
                    "rgba(0,0,0,0.38)", blur=70, offset_y=24, radius=30),
    card=Surface("rgba(26,30,44,0.42)", "rgba(255,255,255,0.14)", "rgba(255,255,255,0.22)",
                 "rgba(0,0,0,0.30)", blur=50, offset_y=18, radius=26),
    raised=Surface("rgba(255,255,255,0.14)", "rgba(255,255,255,0.22)", "rgba(255,255,255,0.28)",
                   radius=PILL),
    capsule=Surface("rgba(28,32,46,0.56)", "rgba(255,255,255,0.18)", "rgba(255,255,255,0.30)",
                    "rgba(0,0,0,0.45)", blur=60, offset_y=20, radius=PILL),
    well=Surface("rgba(255,255,255,0.08)", "rgba(255,255,255,0.10)", radius=16),
    hover="rgba(255,255,255,0.08)",
    pressed="rgba(255,255,255,0.14)",
    selection="rgba(169,199,255,0.28)",
    selection_text="#ffffff",
    hairline="rgba(255,255,255,0.08)",
    knob="#ffffff",
    track="rgba(255,255,255,0.16)",
    view="rgba(255,255,255,0.03)",
    view_alt="rgba(255,255,255,0.025)",
    backdrop_base="#0b0d14",
    backdrop_tone="#141826",
    backdrop_scrim="rgba(8,10,18,0.58)",
    backdrop="art",
)

GLASS = ThemeTokens(**_d(
    key="glass",         name="Liquid Glass", icon="◇",
    sidebar_bg="#141824",  sidebar_hover="#232838",  sidebar_active="#2b3142",
    sidebar_accent="#a9c7ff", sidebar_text="#d5dae4", sidebar_hdr="#9ca4b6",
    content_bg="#0b0d14",  card_bg="#171b29",  card_border="#363a49",
    card_shadow="rgba(0,0,0,0.30)",
    text_primary="#f4f6fb",  text_secondary="#bcc3d2", text_muted="#a2a9ba",
    text_inverse="#101320",
    accent="#a9c7ff",      accent_hover="#d3e2ff",   accent_pressed="#8fb3f5",
    success="#9be3b5",     warning="#ffbe78",
    danger="#ff8f8f",      danger_hover="#ff7373",
    input_bg="#1f2331",    input_border="#3c4152",   input_focus="#a9c7ff",
    progress_bg="#2a2f3f", progress_fg="#a9c7ff",
    tab_bg="#171b29",      tab_active_bg="#2b3142",
    tab_text="#bcc3d2",    tab_active_text="#ffffff",
    log_bg="#0f121b",      log_handle="#151926",
    log_text="#aab1c2",    log_ok="#9be3b5", log_warn="#ffcf8a", log_err="#ff9d9d",
    swatch_bg="#0b0d14",   swatch_accent="#a9c7ff",
    material=GLASS_MATERIAL,
))


# ── Accents (Liquid Glass) ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Accent:
    key: str
    name: str
    base: str
    hover: str
    pressed: str


ACCENTS: dict[str, Accent] = {
    a.key: a for a in (
        Accent("blue", "Blue", "#a9c7ff", "#d3e2ff", "#8fb3f5"),
        Accent("peach", "Peach", "#ffb37a", "#ffd2ae", "#f59a5c"),
        Accent("mint", "Mint", "#9be3b5", "#c6f1d5", "#7dd09c"),
        Accent("lilac", "Lilac", "#e8a6ff", "#f3ceff", "#d68bf2"),
    )
}
DEFAULT_ACCENT = "blue"


# ── Colour arithmetic (pure Python: no Qt needed to read the tokens) ──────────

def parse_colour(value: str) -> tuple[int, int, int, float]:
    """``"#rrggbb"`` / ``"rgba(r,g,b,a)"`` → ``(r, g, b, a)`` with a in 0..1."""
    s = value.strip()
    if s.startswith("rgba(") and s.endswith(")"):
        r, g, b, a = (p.strip() for p in s[5:-1].split(","))
        return int(r), int(g), int(b), float(a)
    h = s.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0


def rgba(colour: str, alpha: float) -> str:
    """``colour`` with ``alpha`` (0..1), as an ``rgba(...)`` string."""
    r, g, b, _ = parse_colour(colour)
    return f"rgba({r},{g},{b},{alpha:g})"


def flatten(colour: str, under: str) -> str:
    """``colour`` composited over the opaque ``under``, as ``#rrggbb``."""
    r, g, b, a = parse_colour(colour)
    ur, ug, ub, _ = parse_colour(under)
    mix = (round(ur + (r - ur) * a), round(ug + (g - ug) * a), round(ub + (b - ub) * a))
    return "#" + "".join(f"{c:02x}" for c in mix)


def mix(a: str, b: str, amount: float) -> str:
    """``b`` laid over ``a`` at ``amount`` (0..1), as ``#rrggbb``."""
    return flatten(rgba(b, amount), a)


def is_opaque(colour: str) -> bool:
    return not colour or parse_colour(colour)[3] >= 1.0


# ── Classic material: the opaque counterpart, derived from the colour tokens ──

def classic_material(t: ThemeTokens) -> Material:
    """Classic themes render through the same components as Liquid Glass with
    opaque fills, no top highlight and a plain or gradient backdrop."""
    return Material(
        kind="classic",
        sidebar=Surface(t.sidebar_bg, t.card_border, "", t.card_shadow,
                        blur=40, offset_y=12, radius=30),
        card=Surface(t.card_bg, t.card_border, "", t.card_shadow, blur=30, offset_y=8, radius=26),
        raised=Surface(mix(t.card_bg, t.text_primary, 0.07), t.card_border, radius=PILL),
        capsule=Surface(t.sidebar_bg, t.card_border, "", t.card_shadow,
                        blur=40, offset_y=12, radius=PILL),
        well=Surface(t.input_bg, t.input_border, radius=16),
        # Like the glass overlays (white at .08 / .14 over a dark card): a
        # faint tint of the text colour over the card. The sidebar tokens
        # used before vanished on dark cards (Midnight's hover *is* its card
        # colour) and were heavy on light ones (secondary text at 4.2:1).
        hover=mix(t.card_bg, t.text_primary, 0.06),
        pressed=mix(t.card_bg, t.text_primary, 0.11),
        selection=t.accent,
        selection_text=t.text_inverse,
        hairline=t.card_border,
        knob=t.accent,
        track=t.progress_bg,
        view=t.card_bg,
        view_alt=t.content_bg,
        backdrop_base=t.content_bg,
        backdrop_tone=t.content_bg,
        backdrop="gradient",
    )


def with_accent(t: ThemeTokens, accent_key: str) -> ThemeTokens:
    """Liquid Glass in one of the four accents (classic themes keep theirs)."""
    accent = ACCENTS.get(accent_key)
    if accent is None or not t.is_glass or t.accent == accent.base:
        return t
    material = replace(t.material, selection=rgba(accent.base, 0.28))
    return replace(
        t, accent=accent.base, accent_hover=accent.hover, accent_pressed=accent.pressed,
        sidebar_accent=accent.base, input_focus=accent.base, progress_fg=accent.base,
        swatch_accent=accent.base, material=material,
    )


def _flat_surface(s: Surface, under: str) -> Surface:
    return replace(s, fill=flatten(s.fill, under),
                   border=flatten(s.border, under) if s.border else "")


def reduced_transparency(t: ThemeTokens) -> ThemeTokens:
    """Opaque surfaces (Settings → Appearance → Reduce transparency): every
    translucent fill is composited over what it usually sits on, so nothing
    shows through any more. Classic themes are already opaque."""
    m = t.material
    if m is None or all(is_opaque(m.surface(n).fill) for n in SURFACE_NAMES):
        return t
    tone = m.backdrop_tone
    sidebar = _flat_surface(m.sidebar, tone)
    card = _flat_surface(m.card, tone)
    flat = replace(
        m,
        sidebar=sidebar, card=card,
        capsule=_flat_surface(m.capsule, tone),
        raised=_flat_surface(m.raised, card.fill),
        well=_flat_surface(m.well, card.fill),
        hover=flatten(m.hover, card.fill),
        pressed=flatten(m.pressed, card.fill),
        selection=flatten(m.selection, card.fill),
        hairline=flatten(m.hairline, card.fill),
        track=flatten(m.track, card.fill),
        view=flatten(m.view, card.fill),
        view_alt=flatten(m.view_alt, card.fill),
    )
    return replace(t, material=flat)


# ── Type ramp, spacing and radii (theme-independent) ──────────────────────────

@dataclass(frozen=True)
class TypeRole:
    px: int
    weight: int
    tracking: float = 0.0     # em; negative = tighter
    family: str = "ui"        # "ui" | "display" | "mono"
    tabular: bool = False     # tabular figures (numbers that line up)


# One px ramp. Pages ask for a role (``gui.themes.glass.font("title")``),
# never a size. "label" (13) is the mockup's subtitle / chip / stats size.
TYPE: dict[str, TypeRole] = {
    "display": TypeRole(34, 700, -0.03, family="display"),
    "title":   TypeRole(20, 650, -0.01, family="display"),
    "heading": TypeRole(16, 650),
    "body":    TypeRole(14, 400),
    "label":   TypeRole(13, 400),
    "caption": TypeRole(12, 400),
    "micro":   TypeRole(11, 600, 0.12),
    "mono":    TypeRole(12, 400, family="mono"),
    "stat":    TypeRole(24, 700, -0.02, tabular=True),
}

# Preferred families, best first: Segoe UI Variable on Windows 11 (the
# owner's platform), then Segoe UI, SF Pro, the bundled Inter, system sans.
UI_FAMILIES: tuple[str, ...] = (
    "Segoe UI Variable Text", "Segoe UI Variable", "Segoe UI", "SF Pro Text", "Inter",
    "Helvetica Neue", "Arial", "sans-serif",
)
DISPLAY_FAMILIES: tuple[str, ...] = (
    "Segoe UI Variable Display", "Segoe UI Variable", "Segoe UI", "SF Pro Display", "Inter",
    "Helvetica Neue", "Arial", "sans-serif",
)
MONO_FAMILIES: tuple[str, ...] = (
    "JetBrains Mono", "Cascadia Mono", "Consolas", "SF Mono", "Menlo", "monospace",
)


@dataclass(frozen=True)
class Spacing:
    hair: int = 2
    xxs: int = 4
    xs: int = 6
    sm: int = 10
    md: int = 14
    lg: int = 18
    xl: int = 24
    xxl: int = 34
    gutter: int = 18          # outer padding and the gap between panels
    card_x: int = 24          # card padding, horizontal
    card_y: int = 22          # card padding, vertical
    page_bottom: int = 104    # room under page content for the Now Playing capsule
    capsule_bottom: int = 22  # capsule distance from the bottom of the page area


@dataclass(frozen=True)
class Radii:
    check: int = 6
    xs: int = 8
    sm: int = 12
    nav: int = 14             # sidebar items
    md: int = 16              # wells, stat tiles
    lg: int = 20              # album tiles
    card: int = 26
    hero: int = 30
    sidebar: int = 30
    pill: int = PILL


SPACE = Spacing()
RADIUS = Radii()


# ── Album-tile placeholders ───────────────────────────────────────────────────
# Diagonal gradient pairs shown on landing tiles until (or unless) real album
# art arrives.
PLACEHOLDER_GRADIENTS: tuple[tuple[str, str], ...] = (
    ("#6366f1", "#a78bfa"), ("#0ea5e9", "#38bdf8"), ("#10b981", "#34d399"),
    ("#f59e0b", "#fbbf24"), ("#ef4444", "#f87171"), ("#8b5cf6", "#c4b5fd"),
    ("#06b6d4", "#67e8f9"), ("#ec4899", "#f9a8d4"), ("#84cc16", "#a3e635"),
    ("#f97316", "#fb923c"), ("#14b8a6", "#2dd4bf"), ("#a855f7", "#d8b4fe"),
    ("#3b82f6", "#93c5fd"), ("#d946ef", "#f0abfc"), ("#22c55e", "#86efac"),
    ("#eab308", "#fde047"), ("#64748b", "#94a3b8"), ("#1e40af", "#60a5fa"),
)


# ── Registry ──────────────────────────────────────────────────────────────────

def _with_material(t: ThemeTokens) -> ThemeTokens:
    return t if t.material is not None else replace(t, material=classic_material(t))


THEMES: dict[str, ThemeTokens] = {
    t.key: _with_material(t) for t in [
        GLASS,
        MIDNIGHT, OBSIDIAN, GRAPHITE, NAVY, TWILIGHT, AURORA, FOREST, EMBER,
        PEARL, AZURE, BLOSSOM, MEADOW, LAVENDER, SUNSET,
    ]
}

DARK_THEMES  = [k for k, t in THEMES.items() if t.variant == "dark"]
LIGHT_THEMES = [k for k, t in THEMES.items() if t.variant == "light"]
# The 14 themes from before Liquid Glass, grouped as "Classic" in Settings.
CLASSIC_THEMES = [k for k, t in THEMES.items() if not t.is_glass]

GLASS_KEY     = "glass"
DEFAULT_THEME = GLASS_KEY
DEFAULT_DARK  = GLASS_KEY
DEFAULT_LIGHT = "pearl"
