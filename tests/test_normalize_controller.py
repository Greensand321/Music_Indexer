import pytest

from controllers.normalize_controller import normalize_genres


def test_normalize_genres_trims_and_dedupes_case_insensitive():
    mapping = {"rock": "Rock"}

    result = normalize_genres([" Rock ", "ROCK"], mapping)

    assert result == ["Rock"]


def test_normalize_genres_splits_combined_entries():
    genres = ["Hip-Hop/Rap; Indie "]

    result = normalize_genres(genres, {})

    assert result == ["Hip-Hop", "Rap", "Indie"]


def test_normalize_genres_handles_mapping_lists_and_invalid_entries():
    mapping = {
        "electronic": ["Electronic", " electronic ", None],
        "skip": None,
        5: "ignored key",
        "alt": ["Alt", "Alt/Rock"],
    }

    result = normalize_genres(["Electronic", "skip", None, "Alt"], mapping)

    assert result == ["Electronic", "Alt", "Rock"]

