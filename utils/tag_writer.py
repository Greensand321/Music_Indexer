"""Format-aware tag writing for the metadata-repair tools (Tag Fixer, Genres).

Every value is written as a *list*, so multi-value fields (several genres)
round-trip faithfully: FLAC/Ogg/Opus get one Vorbis comment per value, MP3 and
ID3-in-RIFF files one multi-value text frame, M4A one atom with several
strings.

Why a module of its own: mutagen's "easy" interface covers MP3, FLAC, Ogg,
Opus and M4A, but a WAV/AIFF file with ID3 tags comes back as a plain ID3
container whose ``audio["artist"] = ...`` raises ``TypeError`` ("not a Frame
instance"). Before this module that error escaped the Tag Fixer's writer and
aborted the whole apply batch at the first tagged WAV, and the Genre
Normalizer could never write a WAV at all. Here those files get real ID3
frames instead.

Backend only: no Qt, no Tk.
"""
from __future__ import annotations

import os
import stat
from typing import Callable, Mapping, Optional, Sequence

from utils.path_helpers import ensure_long_path

try:
    from mutagen import File as MutagenFile  # type: ignore
except Exception:  # noqa: BLE001 - ImportError, or ValueError from a spec-less stub
    MutagenFile = None  # type: ignore

#: Audio extensions the tag tools (Tag Fixer, both Genre Normalizer tabs) scan.
#: One shared set, so the tabs agree on which files exist.
SUPPORTED_TAG_EXTS = frozenset({".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"})

# Easy-interface field name -> ID3 frame id, for ID3 containers that mutagen
# does not wrap in EasyID3 (WAV, AIFF).
_ID3_FRAMES = {
    "artist": "TPE1",
    "albumartist": "TPE2",
    "title": "TIT2",
    "album": "TALB",
    "genre": "TCON",
    "date": "TDRC",
    "tracknumber": "TRCK",
    "discnumber": "TPOS",
}


class TagWriteError(Exception):
    """A tag could not be written; the message says why, in plain words."""


def _is_raw_id3(tags: object) -> bool:
    return hasattr(tags, "getall") and hasattr(tags, "setall")


def _as_list(values: object) -> list[str]:
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)]


def assign_fields(audio, values: Mapping[str, Sequence[str] | str]) -> None:
    """Put ``values`` (field -> list of strings) into a loaded mutagen object.

    Does not save. Raises :class:`TagWriteError` when the format cannot hold
    the tags (e.g. raw ADTS ``.aac``).
    """
    if getattr(audio, "tags", None) is None:
        add_tags = getattr(audio, "add_tags", None)
        if callable(add_tags):
            try:
                add_tags()
            except Exception as exc:  # noqa: BLE001
                raise TagWriteError(f"this file format can't hold tags ({exc})") from exc
    tags = getattr(audio, "tags", None)
    if tags is not None and _is_raw_id3(tags):
        try:
            from mutagen import id3  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise TagWriteError(f"ID3 support unavailable ({exc})") from exc
        for key, raw in values.items():
            frame_id = _ID3_FRAMES.get(key)
            if frame_id is None:
                raise TagWriteError(f"can't write {key!r} to this file's ID3 tags")
            frame_cls = getattr(id3, frame_id)
            tags.setall(frame_id, [frame_cls(encoding=3, text=_as_list(raw))])
        return
    for key, raw in values.items():
        try:
            audio[key] = _as_list(raw)
        except Exception as exc:  # noqa: BLE001
            raise TagWriteError(f"can't write {key!r} to this file ({exc})") from exc


def write_fields(
    path: str,
    values: Mapping[str, Sequence[str] | str],
    *,
    opener: Optional[Callable[..., object]] = None,
) -> None:
    """Open ``path``, write ``values`` and save. Raises :class:`TagWriteError`."""
    open_file = opener or MutagenFile
    if open_file is None:
        raise TagWriteError("mutagen is not installed")
    try:
        audio = open_file(ensure_long_path(path), easy=True)
    except Exception as exc:  # noqa: BLE001
        raise TagWriteError(f"could not open the file ({exc})") from exc
    if audio is None:
        raise TagWriteError("unsupported audio format")
    assign_fields(audio, values)
    try:
        audio.save()
    except Exception as exc:  # noqa: BLE001
        raise TagWriteError(f"could not save ({exc})") from exc


def is_read_only(path: str) -> bool:
    """True when the file is marked read-only (permission bits), even for root."""
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return False
    return not (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)) or not os.access(path, os.W_OK)


def check_writable(path: str) -> Optional[str]:
    """Why tags can't be written to ``path`` — or ``None`` if they can.

    Used by previews so files that would fail are shown up front instead of
    failing (and reappearing) at apply time. Reads only; never saves.
    """
    if not os.path.exists(path):
        return "file not found"
    if is_read_only(path):
        return "file is read-only"
    if MutagenFile is None:
        return "mutagen is not installed"
    try:
        audio = MutagenFile(ensure_long_path(path), easy=True)
    except Exception as exc:  # noqa: BLE001
        return f"unreadable ({exc})"
    if audio is None:
        return "unsupported audio format"
    if getattr(audio, "tags", None) is None:
        add_tags = getattr(audio, "add_tags", None)
        if callable(add_tags):
            try:
                add_tags()  # in memory only — nothing is saved
            except Exception:  # noqa: BLE001
                return "this file format can't hold tags"
    return None
