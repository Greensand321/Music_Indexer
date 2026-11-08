"""The shared Liquid Glass parts: stat tiles and empty states."""
from __future__ import annotations

from gui.widgets.glass_parts import EmptyState, StatRow, StatTile


def test_stat_tile_updates_in_place_and_ignores_unknown_tones(qtbot):
    tile = StatTile("–", "Tracks")
    qtbot.addWidget(tile)
    tile.set_value(4812, tone="success")
    assert (tile.value(), tile.label(), tile.tone()) == ("4812", "Tracks", "success")
    assert tile.accessibleName() == "Tracks: 4812"
    tile.set_tone("neon")
    assert tile.tone() == ""
    tile.resize(tile.sizeHint())
    tile.grab()  # paints without error
    assert tile.sizeHint().height() > 30


def test_stat_row_builds_named_tiles():
    row = StatRow([("tracks", "Tracks"), ("artists", "Artists")])
    assert row.keys() == ["tracks", "artists"]
    row.tile("artists").set_value(391)
    assert row.tile("artists").value() == "391"
    assert row.tile("tracks").objectName() == "stat_tracks"
    row.deleteLater()


def test_empty_state_hides_its_button_until_something_listens(qtbot):
    empty = EmptyState("No runs yet", "Reports appear here.", "Open the Indexer")
    qtbot.addWidget(empty)
    empty.show()
    assert not empty.action_button().isVisible()
    hits = []
    empty.action_requested.connect(lambda: hits.append(1))
    assert empty.action_button().isVisible()
    empty.action_button().click()
    assert hits == [1]
    empty.set_action("")
    assert not empty.action_button().isVisible()
    empty.set_text("Nothing here", "")
    assert (empty.title(), empty.message()) == ("Nothing here", "")


def test_empty_state_set_action_with_callback(qtbot):
    empty = EmptyState("Empty")
    qtbot.addWidget(empty)
    empty.show()
    hits = []
    empty.set_action("Scan", lambda: hits.append("scan"))
    assert empty.action_button().isVisible()
    empty.action_button().click()
    assert hits == ["scan"]


def test_every_glyph_paints_and_unknown_names_fail_loudly(qtbot):
    import pytest
    from gui.compat import QtWidgets
    from gui.widgets.glyphs import GLYPHS, glyph_icon, set_glyph

    for name in GLYPHS:
        icon = glyph_icon(name, "#ffffff", 16)
        img = icon.pixmap(16, 16).toImage()
        assert any(img.pixelColor(x, y).alpha() for x in range(16) for y in range(16)), name
    with pytest.raises(KeyError):
        glyph_icon("emoji")
    btn = QtWidgets.QPushButton("Browse")
    qtbot.addWidget(btn)
    with pytest.raises(KeyError):
        set_glyph(btn, "nope")


def test_set_glyph_follows_the_theme(qtbot):
    from gui.compat import QtWidgets
    from gui.themes.manager import get_manager
    from gui.widgets.glyphs import set_glyph

    btn = QtWidgets.QPushButton("Open folder")
    qtbot.addWidget(btn)
    set_glyph(btn, "folder", role="muted")
    before = btn.icon().cacheKey()
    mgr = get_manager()
    original = mgr.current.key
    try:
        mgr.apply("pearl" if original != "pearl" else "glass")
        assert btn.icon().cacheKey() != before
    finally:
        mgr.apply(original)
