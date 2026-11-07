"""WCAG AA contrast for every text role in every theme (DS-19).

Pure token arithmetic: no Qt needed. Each pair is a text token and the
backgrounds it is actually drawn on:

* text_primary / text_secondary / text_muted on cards, the workspace
  background and inputs (muted carries status hints, phase descriptions,
  inactive stepper steps and placeholders, so it is functional text);
* text_secondary on the sidebar (idle nav items);
* text_inverse on accent, accent_hover and accent_pressed (primary button
  labels at rest, hovered and pressed; also selected rows and badges);
* accent on cards and the workspace background (active stepper step, links,
  active tab text);
* tab text on its tab background;
* the log drawer's text and status colours on the log surfaces.
"""
from __future__ import annotations

import pytest

from gui.themes.tokens import THEMES

AA = 4.5

PAIRS = [
    ("text_primary", ("card_bg", "content_bg", "input_bg", "sidebar_bg")),
    ("text_secondary", ("card_bg", "content_bg", "input_bg", "sidebar_bg")),
    ("text_muted", ("card_bg", "content_bg", "input_bg")),
    ("text_inverse", ("accent", "accent_hover", "accent_pressed")),
    ("accent", ("card_bg", "content_bg")),
    ("tab_text", ("tab_bg",)),
    ("tab_active_text", ("tab_active_bg",)),
    ("log_text", ("log_bg", "log_handle")),
    ("log_ok", ("log_bg", "log_handle")),
    ("log_warn", ("log_bg", "log_handle")),
    ("log_err", ("log_bg", "log_handle")),
]


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    assert len(h) == 6, hex_colour
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_contrast_formula_matches_wcag_reference_points():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)


@pytest.mark.parametrize("key", sorted(THEMES))
def test_text_roles_meet_wcag_aa(key):
    t = THEMES[key]
    failures = []
    for fg, backgrounds in PAIRS:
        for bg in backgrounds:
            ratio = contrast(getattr(t, fg), getattr(t, bg))
            if ratio < AA:
                failures.append(f"{fg} on {bg}: {ratio:.2f}:1")
    assert not failures, f"{key}: " + "; ".join(failures)


@pytest.mark.parametrize("key", sorted(THEMES))
def test_muted_stays_quieter_than_secondary(key):
    """The fix must not flatten the hierarchy: muted is still the least
    prominent text role on cards (closest to the card colour)."""
    t = THEMES[key]
    assert contrast(t.text_muted, t.card_bg) <= contrast(t.text_secondary, t.card_bg) + 0.5
    assert contrast(t.text_secondary, t.card_bg) < contrast(t.text_primary, t.card_bg)
