from pathlib import Path

from utils.reserved_folders import (
    NON_LIBRARY_FOLDERS, RESERVED_FOLDERS, is_reserved_name, is_reserved_path,
    iter_library_files, reserved_component,
)


def _touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def test_reserved_names_match_claude_md():
    assert set(RESERVED_FOLDERS) == {
        "Not Sorted", "Playlists", "Manual Review", "Docs", "Trash", "Quarantine"}
    assert set(NON_LIBRARY_FOLDERS) < set(RESERVED_FOLDERS)
    assert is_reserved_name("quarantine") and not is_reserved_name("Quarantined")


def test_only_top_level_folders_are_reserved(tmp_path):
    lib = tmp_path / "Music"
    assert reserved_component(lib / "Quarantine" / "a.mp3", lib) == "Quarantine"
    assert not is_reserved_path(lib / "Artist" / "Docs" / "a.mp3", lib)
    assert not is_reserved_path(tmp_path / "Elsewhere" / "Trash" / "a.mp3", lib)
    assert not is_reserved_path(lib, lib)


def test_iter_library_files_skips_reserved(tmp_path):
    lib = tmp_path / "Music"
    keep = [_touch(lib, "A/B/01.mp3"), _touch(lib, "A/Docs/02.flac"), _touch(lib, "loose.MP3")]
    for folder in RESERVED_FOLDERS:
        _touch(lib, f"{folder}/x.mp3")
    _touch(lib, "A/B/cover.jpg")
    found = sorted(iter_library_files(lib, {".mp3", ".flac"}))
    assert found == sorted(keep)


def test_custom_reserved_subset(tmp_path):
    lib = tmp_path / "Music"
    review = _touch(lib, "Manual Review/t.mp3")
    _touch(lib, "Quarantine/q.mp3")
    found = list(iter_library_files(lib, {".mp3"}, reserved=NON_LIBRARY_FOLDERS))
    assert found == [review]
