"""Every duplicate scanner skips the library's reserved top-level folders.

The legacy simple finder, the fingerprint generator and the two
proof-of-concept engines only skipped "Not Sorted"/"Playlists" (at any
depth), so quarantined losers in ``Quarantine/``, ``Trash/`` leftovers and
``Docs/`` came back as "duplicates" of the winners they had just been
consolidated into. They now use ``utils/reserved_folders`` — top-level only,
like every other workflow, so ``Artist/Playlists/…`` is ordinary content.
"""
from __future__ import annotations

import os
import types
from pathlib import Path

from _helpers.module_isolation import isolated_modules
from _helpers.dupes_plans import write

RESERVED = ("Quarantine", "Trash", "Docs", "Not Sorted", "Playlists", "Manual Review")


def _library(root: Path) -> tuple[Path, list[Path]]:
    keep = write(root / "Artist" / "Album" / "Song.flac")
    nested = write(root / "Artist" / "Playlists" / "Song (nested).mp3")
    reserved = [write(root / folder / "Song.mp3") for folder in RESERVED]
    return keep, [nested, *reserved]


def test_simple_finder_skips_reserved_folders(tmp_path, monkeypatch):
    import simple_duplicate_finder as sdf

    keep, others = _library(tmp_path)
    seen = []

    def fake_get_fingerprint(path, db_path, compute, log_callback=None, **_kw):
        seen.append(path)
        return "1 2 3 4 5 6 7 8"

    monkeypatch.setattr(sdf, "get_fingerprint", fake_get_fingerprint)
    monkeypatch.setattr(sdf, "flush_fingerprint_writes", lambda *a, **k: None)
    duplicates, _missing = sdf.find_duplicates(
        str(tmp_path), db_path=str(tmp_path / "fp.db"), log_callback=lambda m: None, max_workers=1
    )

    nested = others[0]
    assert sorted(seen) == sorted([str(keep), str(nested)])
    assert [sorted(pair) for pair in duplicates] == [sorted([str(keep), str(nested)])]
    assert sdf._count_audio_files(str(tmp_path)) == 2


def test_fingerprint_generator_skips_reserved_folders(tmp_path):
    import fingerprint_generator as fg

    keep, others = _library(tmp_path)
    found = fg._gather_audio_files(str(tmp_path))
    assert sorted(found) == sorted([str(keep), str(others[0])])


def test_scan_engine_skips_reserved_folders(tmp_path):
    keep, others = _library(tmp_path)
    with isolated_modules({"librosa": types.ModuleType("librosa")}, fresh=["duplicate_scan_engine"]) as mods:
        found = mods["duplicate_scan_engine"]._list_audio_files(str(tmp_path))
    assert sorted(found) == sorted([str(keep), str(others[0])])


def test_bucketing_poc_skips_reserved_folders(tmp_path):
    import duplicate_bucketing_poc as poc

    keep, others = _library(tmp_path)
    kept = [p for p in [keep, *others] if not poc._is_excluded_path(str(p), str(tmp_path))]
    assert sorted(kept) == sorted([keep, others[0]])
    assert poc._is_excluded_path(os.path.join(str(tmp_path), "quarantine", "x.mp3"), str(tmp_path))
