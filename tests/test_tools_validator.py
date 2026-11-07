"""Library Validator (TOOL-22): an indexed library need not have By Year/."""
from __future__ import annotations

import pytest

from validator import validate_soundvault_structure


@pytest.mark.parametrize("dirs", [
    ["By Artist"],                     # no "rare" artists → the Indexer never made By Year/
    ["By Year"],                       # only rare artists
    ["By Artist", "By Year"],
    ["Music/By Artist"],
    ["Music/By Year"],
])
def test_indexed_layouts_are_valid(tmp_path, dirs):
    for d in dirs:
        (tmp_path / d).mkdir(parents=True)
    assert validate_soundvault_structure(str(tmp_path)) == (True, [])


@pytest.mark.parametrize("dirs", [[], ["Artist/Album"], ["Music/Artist"], ["By Album"]])
def test_unindexed_folders_are_invalid(tmp_path, dirs):
    for d in dirs:
        (tmp_path / d).mkdir(parents=True)
    valid, errors = validate_soundvault_structure(str(tmp_path))
    assert valid is False
    assert any("Indexer" in e for e in errors)


def test_a_file_named_by_artist_is_not_a_folder(tmp_path):
    (tmp_path / "By Artist").write_text("not a folder")
    assert validate_soundvault_structure(str(tmp_path))[0] is False
