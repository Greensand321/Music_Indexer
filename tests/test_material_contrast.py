"""WCAG contrast of the text the pages draw on the *surfaces they draw it on*.

``tests/test_design_contrast.py`` checks the opaque colour tokens pairwise.
Since Liquid Glass, pages paint on materials: translucent cards over the
album-art backdrop, wells (stat tiles, list rows), raised pills (buttons,
chips), hover and selection overlays, tinted tone pills. This test
composites each surface the way it is painted (glass over the backdrop's
average tone; classic themes are opaque) and checks the text that sits on
it, in every theme, in Liquid Glass with each accent and with Reduce
transparency.

Thresholds (WCAG 2.x): 4.5:1 for body text, 3:1 for large text and for the
secondary, muted text roles (hints, captions, meta lines).

Pure token arithmetic: no Qt needed.
"""
from __future__ import annotations

import pytest

from gui.themes.tokens import (
    ACCENTS, THEMES, flatten, mix, reduced_transparency, rgba, with_accent,
)

BODY = 4.5
SECONDARY = 3.0
# How much of a tone colour tints a pill / a playing row (gui/widgets/pill.py
# FILL_STRENGTH, gui/widgets/dupes_parts.pill_colours, the Player's rows).
TINT = 0.18


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    assert len(h) == 6, hex_colour
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _variants():
    """Every theme, plus Liquid Glass in each accent and with Reduce
    transparency (the effective tokens ``ThemeManager`` would apply)."""
    out = {}
    for key, t in THEMES.items():
        out[key] = t
        if t.is_glass:
            out[f"{key}+reduce_transparency"] = reduced_transparency(t)
            for accent in ACCENTS:
                tinted = with_accent(t, accent)
                out[f"{key}:{accent}"] = tinted
                out[f"{key}:{accent}+reduce_transparency"] = reduced_transparency(tinted)
    return out


VARIANTS = _variants()


def surfaces(t, backdrop: str | None = None) -> dict[str, str]:
    """Opaque colours of the painted surfaces: ``card`` over the backdrop,
    the rest over the card (or the backdrop, for header pills)."""
    m = t.material
    under = backdrop or m.backdrop_tone
    card = flatten(m.card.fill, under)
    return {
        "backdrop": under,
        "card": card,
        "sidebar": flatten(m.sidebar.fill, under),
        "capsule": flatten(m.capsule.fill, under),
        "well": flatten(m.well.fill, card),
        "raised": flatten(m.raised.fill, card),
        "raised on backdrop": flatten(m.raised.fill, under),
        "hover row": flatten(m.hover, card),
        "pressed row": flatten(m.pressed, card),
        "item view": flatten(m.view, card),
        "selection": flatten(m.selection, card),
    }


def pairs(t) -> list[tuple[str, str, str, float]]:
    """``(label, text, background, required ratio)`` for one theme."""
    s = surfaces(t)
    m = t.material
    out: list[tuple[str, str, str, float]] = []

    # Text roles on the surfaces they appear on.
    for surface in ("card", "backdrop", "well", "raised", "raised on backdrop", "hover row",
                    "item view", "sidebar", "capsule"):
        out.append((f"text on {surface}", t.text_primary, s[surface], BODY))
        out.append((f"secondary text on {surface}", t.text_secondary, s[surface], BODY))
        out.append((f"muted text on {surface}", t.text_muted, s[surface], SECONDARY))
    out.append(("text on pressed row", t.text_primary, s["pressed row"], BODY))

    # Accent text: links, active steps, accent stat values.
    out.append(("accent text on card", t.accent, s["card"], BODY))
    out.append(("accent text on backdrop", t.accent, s["backdrop"], BODY))
    out.append(("accent text on well", t.accent, s["well"], BODY))
    # Accent text on an accent tint (the playing row, proposed values).
    tint = mix(s["card"], t.accent, TINT)
    out.append(("accent ink on accent tint", t.accent_ink, tint, BODY))
    out.append(("text on accent tint", t.text_primary, tint, BODY))

    # Selected rows (palette Highlight / the delegates' selection fill).
    out.append(("selected text on selection", m.selection_text, s["selection"], BODY))

    # Button labels: primary (accent), danger, success; at rest and hovered.
    for fill in ("accent", "accent_hover", "accent_pressed", "danger", "danger_hover", "success"):
        out.append((f"button label on {fill}", t.text_inverse, getattr(t, fill), BODY))

    # Status colours as text on a card (warnings, errors, "Flagged" counts).
    for tone in ("success", "warning", "danger"):
        out.append((f"{tone} text on card", getattr(t, tone), s["card"], BODY))

    # Pills: tone pills (fill = the tone at 18 % over the surface, text
    # primary) on cards and wells, the Experimental pill (over card_bg), and
    # the neutral pill (a well, secondary text).
    for tone in ("accent", "success", "warning", "danger"):
        colour = getattr(t, tone)
        for surface in ("card", "well"):
            out.append((f"{tone} pill text on {surface}", t.text_primary,
                        flatten(rgba(colour, TINT), s[surface]), BODY))
        out.append((f"{tone} badge text", t.text_primary, mix(t.card_bg, colour, TINT), BODY))
    out.append(("neutral pill text", t.text_secondary, s["well"], BODY))

    # The log inset (strip, Activity, Playlists details).
    for role in ("log_text", "log_ok", "log_warn", "log_err"):
        out.append((f"{role} on the log", getattr(t, role), t.log_bg, BODY))
    return out


@pytest.mark.parametrize("key", sorted(VARIANTS))
def test_text_on_painted_surfaces_meets_wcag(key):
    t = VARIANTS[key]
    failures = [
        f"{label}: {contrast(fg, bg):.2f}:1 < {need} ({fg} on {bg})"
        for label, fg, bg, need in pairs(t)
        if contrast(fg, bg) < need
    ]
    assert not failures, f"{key}:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("key", sorted(k for k in VARIANTS if VARIANTS[k].is_glass
                                       and "reduce" not in k))
def test_glass_text_survives_a_bright_album_wall(key):
    """The backdrop tone is an average; a mostly white cover wall is the
    worst case the scrim must still carry. Card text stays readable over it:
    body text at 4.5:1, the secondary and muted roles at 3:1."""
    t = VARIANTS[key]
    m = t.material
    art = flatten(rgba("#ffffff", m.art_opacity), m.backdrop_base)
    bright = flatten(m.backdrop_scrim, art)
    s = surfaces(t, backdrop=bright)
    checks = [
        ("text on card", t.text_primary, s["card"], BODY),
        ("secondary text on card", t.text_secondary, s["card"], SECONDARY),
        ("muted text on card", t.text_muted, s["card"], SECONDARY),
        ("text on sidebar", t.text_primary, s["sidebar"], BODY),
        ("secondary text on sidebar", t.text_secondary, s["sidebar"], SECONDARY),
        ("text on capsule", t.text_primary, s["capsule"], BODY),
        ("text on backdrop (page title)", t.text_primary, s["backdrop"], SECONDARY),
    ]
    failures = [f"{label}: {contrast(fg, bg):.2f}:1 < {need}"
                for label, fg, bg, need in checks if contrast(fg, bg) < need]
    assert not failures, f"{key} over a bright wall ({bright}):\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("key", sorted(THEMES))
def test_hover_and_pressed_rows_are_visible(key):
    """A hovered or pressed row (and the resting glass rows that use the
    hover overlay) must differ from the card it sits on. Classic themes used
    the sidebar's hover token, which *is* Midnight's card colour."""
    t = THEMES[key]
    s = surfaces(t)
    assert contrast(s["hover row"], s["card"]) >= 1.08, f"{key}: hover row invisible"
    assert contrast(s["pressed row"], s["card"]) > contrast(s["hover row"], s["card"]), \
        f"{key}: pressed must read stronger than hover"


def test_contrast_formula_reference_points():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)
