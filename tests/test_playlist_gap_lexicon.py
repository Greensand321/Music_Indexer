"""The modifier lexicon: classification and the directional difference."""
from __future__ import annotations

import json

import pytest

from playlist_gap_lexicon import Lexicon, load_lexicon, modifier_delta
from playlist_gap_types import ModifierClass


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    return load_lexicon()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("feat. Zyra", ModifierClass.COSMETIC),
        ("featuring Someone", ModifierClass.COSMETIC),
        ("2011 Remastered", ModifierClass.COSMETIC),
        ("Chris Lake Remix", ModifierClass.SUBSTANTIVE),
        ("Remix", ModifierClass.SUBSTANTIVE),
        ("Live at Wembley", ModifierClass.SUBSTANTIVE),
        ("Acoustic", ModifierClass.SUBSTANTIVE),
        ("Extended Mix", ModifierClass.SUBSTANTIVE),
        ("Twin Version", ModifierClass.SUBSTANTIVE),
        ("sped up", ModifierClass.SUBSTANTIVE),
        ("slowed + reverb", ModifierClass.SUBSTANTIVE),
        ("nightcore", ModifierClass.SUBSTANTIVE),
        ("but it's beats 3,3", ModifierClass.SUBSTANTIVE),
        ("Original Mix", ModifierClass.AMBIGUOUS),
        ("Radio Edit", ModifierClass.AMBIGUOUS),
        ("A Deal With God", ModifierClass.TITLE_PART),
    ],
)
def test_classification(lexicon, text, expected):
    modifier = lexicon.classify(text, bracketed=True)
    assert modifier is not None, f"{text!r} matched no rule"
    assert modifier.cls is expected


def test_remix_captures_the_remixer(lexicon):
    assert lexicon.classify("Chris Lake Remix").actor == "Chris Lake"


def test_unknown_text_is_not_a_modifier(lexicon):
    assert lexicon.classify("Colors Collide") is None


def test_broad_rules_are_bracket_only(lexicon):
    """Bare "audio" or "mix" would otherwise eat words out of real titles."""
    assert lexicon.classify("Audio Video Disco") is None
    assert lexicon.classify("Official Audio", bracketed=True) is not None


def test_with_is_only_a_feat_synonym_inside_brackets(lexicon):
    """Unbracketed "with" appears in ordinary titles far too often."""
    assert lexicon.classify("With God") is None
    assert lexicon.classify("with Someone", bracketed=True).cls is ModifierClass.COSMETIC


def test_scan_finds_every_rule_in_a_run(lexicon):
    keys = {m.key for m in lexicon.scan("2011 Remastered Radio Edit", bracketed=True)}
    assert "remaster" in keys
    assert "radio_edit" in keys


# ── the directional difference ───────────────────────────────────────────────


def test_identical_modifier_sets_have_no_delta(lexicon):
    mods = (lexicon.classify("Chris Lake Remix"),)
    assert modifier_delta(mods, mods) == ((), ())


def test_wanted_has_a_modifier_the_candidate_lacks(lexicon):
    """Wanting the remix while owning only the original is still missing."""
    wanted = (lexicon.classify("Chris Lake Remix"),)
    only_wanted, only_candidate = modifier_delta(wanted, ())
    assert only_wanted and not only_candidate


def test_candidate_has_a_modifier_the_wanted_lacks(lexicon):
    """The mirror case: owning only a remix is not owning the original."""
    candidate = (lexicon.classify("Chris Lake Remix"),)
    only_wanted, only_candidate = modifier_delta((), candidate)
    assert only_candidate and not only_wanted


def test_different_remixers_are_a_two_sided_delta(lexicon):
    a = (lexicon.classify("Chris Lake Remix"),)
    b = (lexicon.classify("Kaskade Remix"),)
    only_a, only_b = modifier_delta(a, b)
    assert only_a and only_b


def test_title_parts_are_not_a_modifier_difference(lexicon):
    """Title parts belong to the name and are compared as part of the title."""
    part = (lexicon.classify("A Deal With God"),)
    assert modifier_delta(part, ()) == ((), ())


# ── loading ──────────────────────────────────────────────────────────────────


def test_user_overrides_replace_by_id(tmp_path):
    user = tmp_path / "lex.json"
    user.write_text(json.dumps({
        "rules": [{"id": "twin", "pattern": "\\btwin\\s+version\\b", "class": "cosmetic"}],
        "title_parts": ["my special title"],
    }), encoding="utf-8")
    merged = load_lexicon(user_path=str(user))
    assert merged.classify("Twin Version").cls is ModifierClass.COSMETIC
    assert merged.is_title_part("My Special Title")
    # Untouched rules survive the merge.
    assert merged.classify("Chris Lake Remix").cls is ModifierClass.SUBSTANTIVE


def test_a_malformed_user_rule_does_not_break_the_lexicon(tmp_path):
    user = tmp_path / "lex.json"
    user.write_text(json.dumps({"rules": [{"id": "bad", "pattern": "([", "class": "cosmetic"}]}),
                    encoding="utf-8")
    merged = load_lexicon(user_path=str(user))
    assert merged.classify("Chris Lake Remix") is not None


def test_missing_user_file_is_fine(tmp_path):
    assert load_lexicon(user_path=str(tmp_path / "nope.json")).rules
