"""Synthetic music-library factory for tests and GUI harness runs.

Builds a small, *real* music library on disk: genuine audio files (short sine
tones encoded by ffmpeg) carrying genuine tags (written with mutagen), laid out
the way AlphaDEX expects (``Artist/Album/NN - Title.ext``), plus the awkward
cases real libraries contain — an untagged file, a cross-codec duplicate pair,
non-audio leftovers, a playlist.

Why real files rather than mocks: the GUI redesign needs end-to-end harness
runs (scan → preview → confirm) that exercise the actual metadata readers,
and several historic bugs only reproduced with real containers.

Usage (pytest)::

    def test_something(synthetic_library):          # fixture in tests/conftest.py
        lib = synthetic_library                     # -> SyntheticLibrary
        assert lib.root.is_dir()
        assert len(lib.audio_paths) >= 8

Usage (script)::

    python tests/library_factory.py /tmp/demo_library

Encoded clips are cached per (format, duration, frequency) inside the cache
directory, so generating the same library repeatedly costs one ffmpeg call per
distinct clip and is then just file copies + tagging.

The module deliberately has no dependency on Qt or on any AlphaDEX module.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Formats this factory can produce, and the ffmpeg arguments for each.
_FFMPEG_CODEC_ARGS: dict[str, list[str]] = {
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "96k"],
    ".flac": ["-c:a", "flac"],
    ".m4a": ["-c:a", "aac", "-b:a", "96k"],
    ".ogg": ["-c:a", "libvorbis", "-q:a", "2"],
    ".opus": ["-c:a", "libopus", "-b:a", "48k"],
    ".wav": ["-c:a", "pcm_s16le"],
}
SUPPORTED_EXTENSIONS = tuple(_FFMPEG_CODEC_ARGS)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ── Real mutagen, even when a test has stubbed it ─────────────────────────────
#
# Many legacy test modules install an inline fake ``mutagen`` into
# ``sys.modules``. The factory must always write tags with the real library,
# so it imports the real modules once, temporarily lifting any stub.

_REAL_MUTAGEN: dict[str, object] | None = None


def _real_mutagen() -> dict[str, object]:
    global _REAL_MUTAGEN
    if _REAL_MUTAGEN is not None:
        return _REAL_MUTAGEN
    stashed = {
        name: mod
        for name, mod in list(sys.modules.items())
        if (name == "mutagen" or name.startswith("mutagen."))
        and getattr(mod, "__file__", None) is None
    }
    for name in stashed:
        del sys.modules[name]
    try:
        mods = {
            "id3": importlib.import_module("mutagen.id3"),
            "easyid3": importlib.import_module("mutagen.easyid3"),
            "flac": importlib.import_module("mutagen.flac"),
            "mp4": importlib.import_module("mutagen.mp4"),
            "oggvorbis": importlib.import_module("mutagen.oggvorbis"),
            "oggopus": importlib.import_module("mutagen.oggopus"),
            "wave": importlib.import_module("mutagen.wave"),
        }
    finally:
        # Put the stubs back exactly as the test left them.
        sys.modules.update(stashed)
    _REAL_MUTAGEN = mods
    return mods


# ── Specs ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrackSpec:
    """One audio file to create.

    ``None`` for a tag field means "leave that tag out" (use it to build
    under-tagged files). ``relpath`` overrides the default
    ``Artist/Album/NN - Title.ext`` placement (relative to the library root).
    """

    artist: str | None
    title: str | None
    album: str | None = None
    track: int | None = None
    year: str | None = None
    genre: str | None = None
    albumartist: str | None = None
    ext: str = ".mp3"
    seconds: float = 1.0
    freq: float = 440.0
    relpath: str | None = None
    write_tags: bool = True

    def default_relpath(self) -> str:
        artist = self.albumartist or self.artist or "Unknown Artist"
        album = self.album or "Unknown Album"
        stem = self.title or f"Track {self.freq:g}Hz"
        if self.track is not None:
            stem = f"{self.track:02d} - {stem}"
        return f"{artist}/{album}/{stem}{self.ext}"


@dataclass(frozen=True)
class ExtraFile:
    """A non-audio file (artwork, notes, playlists...)."""

    relpath: str
    content: bytes = b""


@dataclass
class SyntheticLibrary:
    root: Path
    tracks: dict[Path, TrackSpec] = field(default_factory=dict)
    extras: list[Path] = field(default_factory=list)

    @property
    def audio_paths(self) -> list[Path]:
        return sorted(self.tracks)

    def paths_with_ext(self, ext: str) -> list[Path]:
        return [p for p in self.audio_paths if p.suffix.lower() == ext.lower()]

    def find(self, title: str) -> Path:
        for path, spec in self.tracks.items():
            if spec.title == title:
                return path
        raise KeyError(title)


# ── Encoding ──────────────────────────────────────────────────────────────────

def _default_cache_dir() -> Path:
    base = os.environ.get("ALPHADEX_TEST_AUDIO_CACHE")
    if base:
        return Path(base)
    return Path(tempfile.gettempdir()) / "alphadex_test_audio_cache"


def _write_wav_pure_python(dest: Path, seconds: float, freq: float, rate: int = 22050) -> None:
    """Fallback encoder when ffmpeg is unavailable (WAV only)."""
    frames = int(seconds * rate)
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        data = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(frames)
        )
        w.writeframes(data)


def encode_clip(ext: str, seconds: float, freq: float, cache_dir: Path | None = None) -> Path:
    """Return a cached, untagged clip for (ext, seconds, freq), encoding it if needed."""
    ext = ext.lower()
    if ext not in _FFMPEG_CODEC_ARGS:
        raise ValueError(f"Unsupported extension {ext!r}; choose from {SUPPORTED_EXTENSIONS}")
    cache_dir = Path(cache_dir or _default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{ext}|{seconds:.3f}|{freq:.3f}".encode()).hexdigest()[:16]
    dest = cache_dir / f"clip_{key}{ext}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    tmp = dest.with_name(dest.stem + f".{os.getpid()}.tmp{ext}")
    if ffmpeg_available():
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}:sample_rate=44100",
            "-ac", "2", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
            *_FFMPEG_CODEC_ARGS[ext], str(tmp),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    elif ext == ".wav":
        _write_wav_pure_python(tmp, seconds, freq)
    else:
        raise RuntimeError(f"ffmpeg is required to synthesise {ext} files")
    os.replace(tmp, dest)  # atomic: concurrent test workers never see half a file
    return dest


# ── Tagging ───────────────────────────────────────────────────────────────────

def _tag_fields(spec: TrackSpec) -> dict[str, str]:
    fields = {
        "artist": spec.artist,
        "title": spec.title,
        "album": spec.album,
        "tracknumber": str(spec.track) if spec.track is not None else None,
        "date": spec.year,
        "genre": spec.genre,
        "albumartist": spec.albumartist,
    }
    return {k: v for k, v in fields.items() if v is not None}


def write_tags(path: Path, spec: TrackSpec) -> None:
    fields = _tag_fields(spec)
    if not fields:
        return
    m = _real_mutagen()
    ext = path.suffix.lower()
    if ext == ".mp3":
        try:
            audio = m["easyid3"].EasyID3(str(path))
        except m["id3"].ID3NoHeaderError:
            audio = m["easyid3"].EasyID3()
        for k, v in fields.items():
            audio[k] = v
        audio.save(str(path))
    elif ext in (".flac", ".ogg", ".opus"):
        cls = {".flac": m["flac"].FLAC, ".ogg": m["oggvorbis"].OggVorbis,
               ".opus": m["oggopus"].OggOpus}[ext]
        audio = cls(str(path))
        for k, v in fields.items():
            audio[k] = [v]
        audio.save()
    elif ext == ".m4a":
        mp4_keys = {"artist": "\xa9ART", "title": "\xa9nam", "album": "\xa9alb",
                    "date": "\xa9day", "genre": "\xa9gen", "albumartist": "aART"}
        audio = m["mp4"].MP4(str(path))
        for k, v in fields.items():
            if k == "tracknumber":
                audio["trkn"] = [(int(v), 0)]
            else:
                audio[mp4_keys[k]] = [v]
        audio.save()
    elif ext == ".wav":
        id3 = m["id3"]
        audio = m["wave"].WAVE(str(path))
        if audio.tags is None:
            audio.add_tags()
        frame_map = {"artist": id3.TPE1, "title": id3.TIT2, "album": id3.TALB,
                     "tracknumber": id3.TRCK, "date": id3.TDRC, "genre": id3.TCON,
                     "albumartist": id3.TPE2}
        for k, v in fields.items():
            audio.tags.add(frame_map[k](encoding=3, text=[v]))
        audio.save()
    else:  # pragma: no cover - guarded by encode_clip
        raise ValueError(ext)


# ── Building ──────────────────────────────────────────────────────────────────

def build_library(
    root: Path | str,
    tracks: Iterable[TrackSpec],
    extras: Iterable[ExtraFile] = (),
    cache_dir: Path | None = None,
) -> SyntheticLibrary:
    """Materialise ``tracks`` and ``extras`` under ``root`` and return the result."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lib = SyntheticLibrary(root=root)
    for spec in tracks:
        dest = root / (spec.relpath or spec.default_relpath())
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(encode_clip(spec.ext, spec.seconds, spec.freq, cache_dir), dest)
        if spec.write_tags:
            write_tags(dest, spec)
        lib.tracks[dest] = spec
    for extra in extras:
        dest = root / extra.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(extra.content)
        lib.extras.append(dest)
    return lib


def standard_tracks() -> list[TrackSpec]:
    """A compact library touching every code path the workflows care about."""
    return [
        # Fully tagged MP3 album
        TrackSpec("Aurora Lines", "First Light", "Dawn Signals", 1, "2019", "Electronic", freq=330),
        TrackSpec("Aurora Lines", "Northbound", "Dawn Signals", 2, "2019", "Electronic", freq=349),
        TrackSpec("Aurora Lines", "Polar Night", "Dawn Signals", 3, "2019", "Electronic", freq=392),
        # Lossless FLAC album
        TrackSpec("Basalt", "Fault Line", "Granite", 1, "2021", "Rock", ext=".flac", freq=262),
        TrackSpec("Basalt", "Magma", "Granite", 2, "2021", "rock", ext=".flac", freq=294),
        # Cross-codec duplicate of "Fault Line" (same tone, lossy copy elsewhere)
        TrackSpec("Basalt", "Fault Line", "Granite", 1, "2021", "Rock", ext=".mp3", freq=262,
                  relpath="Basalt/Singles/Fault Line.mp3"),
        # Other containers
        TrackSpec("Cobalt Choir", "Vespers", "Hymns", 1, "2015", "Classical", ext=".m4a", freq=440),
        TrackSpec("Cobalt Choir", "Matins", "Hymns", 2, "2015", "Classical", ext=".ogg", freq=466),
        TrackSpec("Delta Static", "Low Tide", "Coastal", 1, "2023", "Ambient", ext=".opus", freq=523),
        # Under-tagged file in the root (Manual Review candidate)
        TrackSpec(None, None, ext=".mp3", freq=587, relpath="untitled_track.mp3", write_tags=False),
        # Artist + title but no album/genre
        TrackSpec("Echo Park", "Loose Wire", None, None, None, None, freq=622,
                  relpath="Echo Park - Loose Wire.mp3"),
    ]


def standard_extras() -> list[ExtraFile]:
    return [
        ExtraFile("Aurora Lines/Dawn Signals/cover.jpg", b"\xff\xd8\xff\xe0fake-jpeg"),
        ExtraFile("Basalt/Granite/notes.txt", b"liner notes\n"),
        ExtraFile(
            "Playlists/Favourites.m3u",
            b"#EXTM3U\n../Aurora Lines/Dawn Signals/01 - First Light.mp3\n"
            b"../Basalt/Granite/02 - Magma.flac\n",
        ),
    ]


def build_standard_library(root: Path | str, cache_dir: Path | None = None) -> SyntheticLibrary:
    tracks: Sequence[TrackSpec] = standard_tracks()
    if not ffmpeg_available():
        # Degrade gracefully: same layout, WAV everywhere.
        tracks = [
            dataclasses.replace(
                t, ext=".wav",
                relpath=(t.relpath.rsplit(".", 1)[0] + ".wav") if t.relpath else None,
            )
            for t in tracks
        ]
    return build_library(root, tracks, standard_extras(), cache_dir)


if __name__ == "__main__":  # pragma: no cover - manual helper
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_library")
    built = build_standard_library(target)
    print(f"Built {len(built.audio_paths)} tracks + {len(built.extras)} extras under {built.root}")
