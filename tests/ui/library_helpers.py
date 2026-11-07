"""Helpers for the Indexer / Library Sync GUI tests (area: library).

Filesystem snapshots (to prove nothing outside a plan moved and nothing was
overwritten), a worker wait helper, and fake fingerprints for Library Sync
(there is no fpcalc in CI).
"""
from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

from library_factory import TrackSpec, build_library

RESERVED = ("Not Sorted", "Playlists", "Manual Review", "Docs", "Trash", "Quarantine")


def file_hash(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def tree(root: Path) -> dict[str, str]:
    """``{relative posix path: sha1}`` for every file under ``root``."""
    root = Path(root)
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            out[p.relative_to(root).as_posix()] = file_hash(p)
    return out


def reserved_tree(root: Path, folders=RESERVED) -> dict[str, str]:
    full = tree(root)
    return {k: v for k, v in full.items() if k.split("/", 1)[0] in folders}


def content_multiset(snapshot: dict[str, str]) -> Counter:
    return Counter(snapshot.values())


def wait_for(qtbot, predicate, timeout: int = 30_000) -> None:
    qtbot.waitUntil(predicate, timeout=timeout)


def wait_idle(qtbot, ws, attrs=("_worker",), timeout: int = 30_000) -> None:
    """Wait until the page has handled its workers' results (the finished
    slots reset these attributes to None)."""
    qtbot.waitUntil(lambda: all(getattr(ws, a, None) is None for a in attrs), timeout=timeout)


def add_reserved_content(root: Path, template_track: Path) -> None:
    """Populate every reserved folder with audio and non-audio files."""
    root = Path(root)
    (root / "Quarantine").mkdir(exist_ok=True)
    shutil.copyfile(template_track, root / "Quarantine" / "dupe_loser.mp3")
    (root / "Trash").mkdir(exist_ok=True)
    shutil.copyfile(template_track, root / "Trash" / "old.mp3")
    (root / "Trash" / "cover.jpg").write_bytes(b"trash-cover-already-here")
    (root / "Manual Review").mkdir(exist_ok=True)
    shutil.copyfile(template_track, root / "Manual Review" / "pending.mp3")
    (root / "Not Sorted").mkdir(exist_ok=True)
    shutil.copyfile(template_track, root / "Not Sorted" / "mine.mp3")
    (root / "Docs").mkdir(exist_ok=True)
    (root / "Docs" / "keep.txt").write_text("user doc")


# ── Library Sync ──────────────────────────────────────────────────────────────

def _fp_for(title: str) -> str:
    seed = int(hashlib.sha1(title.encode()).hexdigest()[:6], 16)
    return ",".join(str(seed + i) for i in range(16))


def install_fake_fingerprints(monkeypatch, *, title_for=None) -> None:
    """Fingerprint = function of the track title, so the same song in two
    formats is an exact match and different songs never match."""
    import library_sync
    from utils.audio_metadata_reader import read_tags

    def fake(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kw):
        out = []
        for i, p in enumerate(paths, 1):
            title = (title_for(p) if title_for else None) or read_tags(p).get("title") or Path(p).stem
            out.append((p, 1, _fp_for(str(title))))
            if progress_callback:
                progress_callback(i, len(paths), p, "fp")
        return out

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", fake)


def build_incoming(root: Path) -> dict[str, Path]:
    """An incoming folder: one brand-new track, one exact duplicate (same
    format) of a library track, one lossless upgrade of a library MP3."""
    lib = build_library(root, [
        TrackSpec("Zephyr Quay", "Harbour Lights", "Moorings", 1, "2024", "Pop", ext=".flac",
                  freq=700, relpath="drop/new_track.flac"),
        TrackSpec("Echo Park", "Loose Wire", None, None, None, None, ext=".mp3", freq=622,
                  relpath="drop/loose_wire_copy.mp3"),
        TrackSpec("Aurora Lines", "Northbound", "Dawn Signals", 2, "2019", "Electronic",
                  ext=".flac", freq=349, relpath="drop/northbound.flac"),
    ])
    by_title = {spec.title: path for path, spec in lib.tracks.items()}
    return {"new": by_title["Harbour Lights"], "dupe": by_title["Loose Wire"],
            "upgrade": by_title["Northbound"]}
