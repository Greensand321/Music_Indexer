"""A mixed-codec library + incoming folder for Library Sync matching tests.

The library holds 73 real, tagged audio files (FLAC, MP3, M4A, Ogg Vorbis,
Opus) whose fingerprints are faked per title tag (there is no fpcalc in CI):

* 10 FLACs and 55 MP3s that sort *before* everything that matters, each a
  different recording (synthetic Chromaprint prints);
* ``M Artist/Mid Album/01 - Song A.mp3`` — the real fingerprint of songA as
  a 128 kbit/s MP3, somewhere in the middle (not first);
* ``Z Artist/Late Album/01 - Late Song.mp3`` — sorted *last*; the incoming
  ``late.mp3`` is a re-encode of it (60 flipped bits, distance ≈ 0.008).

The incoming folder holds, in ``drop/``:

* ``song_a.flac``  — songA as FLAC: its MP3 copy is in the library;
* ``late.mp3``     — the same-codec duplicate of the last library track;
* ``shifted.mp3``  — songA starting 0.3 s later (distance ≈ 0.074);
* ``brand_new.flac`` — a recording the library does not have.

Before the shortlist fix only ``brand_new`` came out right: the FLAC was
compared with the first 50 FLACs only, the MP3s with the first 50 MP3s.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from _helpers.dupes_chromaprint import fake_fingerprint, real_fingerprints
from library_factory import TrackSpec, build_library

_REAL = real_fingerprints()

# title tag -> fingerprint
FINGERPRINTS: dict[str, str] = {
    "Song A": _REAL["songA.mp3_128"],
    "Late Song": fake_fingerprint("late-song", flips=60),   # ~0.008: a transcode
    "Song A FLAC": _REAL["songA.flac"],
    "Late Song Copy": fake_fingerprint("late-song"),
    "Song A Shifted": _REAL["songA_cut.mp3_128"],
    "Brand New": fake_fingerprint("brand-new"),
}


def fingerprint_for_title(title: str) -> str:
    return FINGERPRINTS.get(title) or fake_fingerprint(f"other:{title}")


@dataclass
class SyncScenario:
    library: Path
    incoming: Path
    lib_song_a: Path
    lib_late: Path
    song_a_flac: Path
    late_copy: Path
    shifted: Path
    brand_new: Path


def _filler() -> list[TrackSpec]:
    specs = []
    for i in range(10):
        specs.append(TrackSpec(f"A{i:02d} Artist", f"Flac Song {i}", "Lossless", 1, "2020", "Rock",
                               ext=".flac", freq=300 + i))
    for i in range(55):
        specs.append(TrackSpec(f"B{i:02d} Artist", f"Mp3 Song {i}", "Lossy", 1, "2020", "Pop",
                               ext=".mp3", freq=300 + i))
    for i, ext in enumerate([".m4a", ".m4a", ".ogg", ".ogg", ".opus", ".opus"]):
        specs.append(TrackSpec(f"C{i:02d} Artist", f"Other Song {i}", "Mixed", 1, "2020", "Jazz",
                               ext=ext, freq=400 + i))
    return specs


def build_sync_scenario(root: Path) -> SyncScenario:
    root = Path(root)
    lib = build_library(root / "library", _filler() + [
        TrackSpec("M Artist", "Song A", "Mid Album", 1, "2021", "Pop", ext=".mp3", freq=440),
        TrackSpec("Z Artist", "Late Song", "Late Album", 1, "2022", "Pop", ext=".mp3", freq=500),
    ])
    inc = build_library(root / "incoming", [
        TrackSpec("M Artist", "Song A FLAC", "Mid Album", 1, "2021", "Pop", ext=".flac", freq=440,
                  relpath="drop/song_a.flac"),
        TrackSpec("Z Artist", "Late Song Copy", "Late Album", 1, "2022", "Pop", ext=".mp3", freq=500,
                  relpath="drop/late.mp3"),
        TrackSpec("M Artist", "Song A Shifted", "Mid Album", 2, "2021", "Pop", ext=".mp3", freq=441,
                  relpath="drop/shifted.mp3"),
        TrackSpec("N Artist", "Brand New", "Fresh", 1, "2024", "Pop", ext=".flac", freq=600,
                  relpath="drop/brand_new.flac"),
    ])
    by_title = {spec.title: path for path, spec in {**lib.tracks, **inc.tracks}.items()}
    return SyncScenario(
        library=lib.root, incoming=inc.root,
        lib_song_a=by_title["Song A"], lib_late=by_title["Late Song"],
        song_a_flac=by_title["Song A FLAC"], late_copy=by_title["Late Song Copy"],
        shifted=by_title["Song A Shifted"], brand_new=by_title["Brand New"],
    )


def install_title_fingerprints(monkeypatch) -> list[list[str]]:
    """Fake Library Sync's fingerprinting from each file's title tag."""
    import library_sync
    from utils.audio_metadata_reader import read_tags

    calls: list[list[str]] = []

    def fake(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kw):
        calls.append(list(paths))
        out = []
        for i, p in enumerate(paths, 1):
            title = str(read_tags(p).get("title") or Path(p).stem)
            out.append((p, 30, fingerprint_for_title(title)))
            if progress_callback:
                progress_callback(i, len(paths), p, "fp")
        return out

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", fake)
    return calls
