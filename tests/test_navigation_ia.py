"""The information architecture as data (gui/navigation.py) — no Qt needed.

The adopted IA (docs/redesign/README.md §5): sidebar groups → pages →
sections, every section backed by a panel; old per-panel keys keep routing.
"""
from __future__ import annotations

import pytest

from gui import navigation as nav


def test_sidebar_groups_and_pages_follow_the_adopted_structure():
    groups = [(title, [p.key for p in pages]) for title, pages in nav.nav_groups()]
    assert groups == [
        ("LIBRARY", ["home"]),
        ("ORGANIZE", ["indexer", "library_sync", "file_cleanup"]),
        ("CLEAN UP", ["duplicates", "metadata"]),
        ("LISTEN", ["player", "sound_map", "playlists"]),
        ("TAKE AWAY", ["mirror", "exports"]),
        ("", ["activity", "settings", "help"]),
    ]
    assert nav.nav_order() == [key for _title, keys in groups for key in keys]


def test_sections_of_the_composed_pages():
    sections = {p.key: [s.key for s in p.sections] for p in nav.PAGES if p.has_sections}
    assert sections == {
        "duplicates": ["find", "compare", "pair_review", "fuzzy_finder", "scan_engine", "bucketing"],
        "metadata": ["tag_fixer", "fill_genres", "normalize_genres", "undo_tags"],
        "sound_map": ["analyse", "map"],
        "mirror": ["build", "verify"],
        "settings": ["services", "general", "appearance"],
    }


def test_keys_are_unique_and_do_not_collide():
    pages = [p.key for p in nav.PAGES]
    sections = [s.key for p in nav.PAGES for s in p.sections]
    assert len(set(pages)) == len(pages)
    assert len(set(sections)) == len(sections), "section keys are unique across the app"
    # A section key that is also a page key must be that page's own section.
    for page in nav.PAGES:
        for section in page.sections:
            if section.key in nav.PAGE_BY_KEY:
                assert section.key == page.key
    # A panel may serve several sections only within one page.
    for panel in nav.panel_keys():
        assert len({p.key for p, _s in nav.sections_of_panel(panel)}) == 1, panel


def test_exactly_the_three_prototypes_are_experimental():
    experimental = {(p.key, s.key) for p in nav.PAGES for s in p.sections if s.experimental}
    assert experimental == {
        ("duplicates", "fuzzy_finder"), ("duplicates", "scan_engine"), ("duplicates", "bucketing"),
    }


@pytest.mark.parametrize("key, expected", [
    # old panel keys → their page and section
    ("similarity", ("duplicates", "compare")),
    ("duplicates", ("duplicates", "find")),
    ("tag_fixer", ("metadata", "tag_fixer")),
    ("genres", ("metadata", "fill_genres")),
    ("clustered", ("sound_map", "analyse")),
    ("graph", ("sound_map", "map")),
    ("compression", ("mirror", "build")),
    ("indexer", ("indexer", "indexer")),
    ("library_sync", ("library_sync", "library_sync")),
    ("playlists", ("playlists", "playlists")),
    ("player", ("player", "player")),
    ("help", ("help", "help")),
    # the retired Utilities page
    ("tools", ("exports", "exports")),
    # page keys that are not panels: the page, in its current section
    ("metadata", ("metadata", None)),
    ("sound_map", ("sound_map", None)),
    ("mirror", ("mirror", None)),
    # section keys and explicit page/section
    ("compare", ("duplicates", "compare")),
    ("appearance", ("settings", "appearance")),
    ("settings/appearance", ("settings", "appearance")),
    ("duplicates/bucketing", ("duplicates", "bucketing")),
])
def test_resolve(key, expected):
    assert nav.resolve(key) == expected


@pytest.mark.parametrize("key", ["", "nope", "settings/nope", "nope/find", "duplicates/map"])
def test_resolve_rejects_unknown_keys(key):
    assert nav.resolve(key) is None


def test_status_labels():
    assert nav.label("sound_map", "map") == "Sound Map · Map"
    assert nav.label("duplicates", "compare") == "Duplicates · Compare two files"
    assert nav.label("player", "player") == "Player"      # one section: no suffix
    assert nav.label("home") == "Home"
