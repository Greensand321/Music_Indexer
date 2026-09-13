"""Stage 2.5 — display-string parsing.

Every row in the "real rows" section is an actual line from the user's YouTube
Music export. They are the acceptance cases for this module.
"""
from __future__ import annotations

import pytest

from playlist_gap_lexicon import load_lexicon
from playlist_gap_parse import (
    candidate_splits,
    compare_key,
    fold_text,
    strip_accents,
)
from playlist_gap_types import ModifierClass


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon()


def best(display, lexicon, **kwargs):
    splits = candidate_splits(display, lexicon=lexicon, **kwargs)
    assert splits, f"no split produced for {display!r}"
    return splits[0]


# ── real rows ────────────────────────────────────────────────────────────────


def test_plain_artist_title(lexicon):
    split = best("The Sways - Someday We Will Dream About Today", lexicon)
    assert split.artist == "The Sways"
    assert split.title == "Someday We Will Dream About Today"
    assert split.modifiers == ()


def test_title_parenthetical_is_kept(lexicon):
    """"(A Deal With God)" is part of the song's name, not a modifier.

    It is also a trap: "With God" trips the `feat` rule, so a trailing scan that
    reads inside kept brackets amputates the title.
    """
    split = best("Kate Bush - Running Up That Hill (A Deal With God)", lexicon)
    assert split.artist == "Kate Bush"
    assert split.title == "Running Up That Hill (A Deal With God)"
    assert split.modifiers == ()


def test_version_parenthetical_is_a_substantive_modifier(lexicon):
    split = best("FIFTY FIFTY - Cupid (Twin Version)", lexicon)
    assert split.artist == "FIFTY FIFTY"
    assert split.title == "Cupid"
    assert [m.cls for m in split.modifiers] == [ModifierClass.SUBSTANTIVE]


def test_uploader_prefix_and_fan_edit_tail(lexicon):
    split = best("(Triple Vibe) HOME - Resonance but it's beats 3,3", lexicon)
    assert split.artist == "HOME"
    assert split.title == "Resonance"
    assert [m.cls for m in split.modifiers] == [ModifierClass.SUBSTANTIVE]


def test_two_delimiters_produce_several_candidates(lexicon):
    """Two delimiters, no way to know which marks the artist boundary."""
    splits = candidate_splits(
        "French 79 · New Constellations - Colors Collide", lexicon=lexicon
    )
    readings = {(s.artist, s.title) for s in splits}
    assert ("French 79 · New Constellations", "Colors Collide") in readings
    assert ("French 79", "New Constellations - Colors Collide") in readings


def test_mashup_is_never_split_on_x(lexicon):
    """`music_indexer_api.extract_primary_and_collabs` lists " x " as an artist
    separator, which would read this as *Self Aware* featuring *Babydoll*."""
    splits = candidate_splits("Self Aware x Babydoll", lexicon=lexicon)
    assert splits
    assert all(s.artist is None for s in splits)
    assert splits[0].origin == "mashup"
    assert "Self Aware x Babydoll" == splits[0].title


def test_blank_display_yields_nothing(lexicon):
    assert candidate_splits("", lexicon=lexicon) == []
    assert candidate_splits("   ", lexicon=lexicon) == []


# ── mechanics ────────────────────────────────────────────────────────────────


def test_feat_credit_is_cosmetic(lexicon):
    split = best("ODESZA - Say My Name (feat. Zyra)", lexicon)
    assert split.title == "Say My Name"
    assert split.modifiers[0].cls is ModifierClass.COSMETIC
    assert split.modifiers[0].actor == "Zyra"


def test_multiple_trailing_modifiers_all_survive(lexicon):
    """Taking only the first match would lose the "Radio Edit" half."""
    split = best("Tinlicker - Because You Move Me - 2021 Remastered Radio Edit", lexicon)
    assert split.title == "Because You Move Me"
    classes = {m.cls for m in split.modifiers}
    assert ModifierClass.COSMETIC in classes
    assert ModifierClass.AMBIGUOUS in classes


def test_both_orderings_are_offered(lexicon):
    splits = candidate_splits("Wavebeatmaker - Resonance", lexicon=lexicon)
    origins = {s.origin.split(":")[0] for s in splits}
    assert {"delimiter", "reversed", "whole"} <= origins


def test_forward_reading_outranks_reversed(lexicon):
    splits = candidate_splits("Wavebeatmaker - Resonance", lexicon=lexicon)
    assert splits[0].artist == "Wavebeatmaker"
    forward = next(s for s in splits if s.origin.startswith("delimiter"))
    reverse = next(s for s in splits if s.origin.startswith("reversed"))
    assert forward.prior > reverse.prior


def test_known_artist_raises_the_prior(lexicon):
    plain = candidate_splits("Wavebeatmaker - Resonance", lexicon=lexicon)[0]
    boosted = candidate_splits(
        "Wavebeatmaker - Resonance",
        lexicon=lexicon,
        known_artists=frozenset({"wavebeatmaker"}),
    )[0]
    assert boosted.prior > plain.prior


def test_provided_fields_lead_but_do_not_suppress(lexicon):
    splits = candidate_splits(
        "Some Channel - Cupid",
        lexicon=lexicon,
        provided_artist="FIFTY FIFTY",
        provided_title="Cupid",
    )
    assert splits[0].origin == "provided"
    assert splits[0].artist == "FIFTY FIFTY"
    # The parsed reading survives too, because a "provided" artist is sometimes
    # only a channel name.
    assert any(s.artist == "Some Channel" for s in splits)


def test_candidate_limit_is_respected(lexicon):
    splits = candidate_splits("a - b - c - d - e - f", lexicon=lexicon, max_candidates=3)
    assert len(splits) <= 3


def test_duplicate_readings_are_collapsed(lexicon):
    splits = candidate_splits("Artist - Title", lexicon=lexicon)
    keys = [(s.artist, s.title, s.modifier_keys()) for s in splits]
    assert len(keys) == len(set(keys))


# ── normalization ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Don’t Stop", "Don't Stop"),
        ("A — B", "A - B"),
        ("  lots   of   space ", "lots of space"),
        ("nbsp here", "nbsp here"),
    ],
)
def test_fold_text(raw, expected):
    assert fold_text(raw) == expected


def test_strip_accents():
    assert strip_accents("Beyoncé") == "Beyonce"
    assert strip_accents("Sigur Rós") == "Sigur Ros"


def test_compare_key_folds_accents_by_default():
    assert compare_key("Beyoncé") == compare_key("Beyonce")
    assert compare_key("Beyoncé", fold_accents=False) != compare_key("Beyonce")


def test_compare_key_is_punctuation_insensitive():
    assert compare_key("Don't Stop!") == compare_key("dont stop")


def test_looks_like_mashup_is_a_property_of_the_string(lexicon):
    from playlist_gap_parse import looks_like_mashup

    assert looks_like_mashup("Self Aware x Babydoll")
    assert looks_like_mashup("Artist A vs Artist B")
    assert not looks_like_mashup("Kate Bush - Running Up That Hill")
    assert not looks_like_mashup("Malcolm X Tribute")
