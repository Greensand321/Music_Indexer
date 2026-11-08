"""Execute duplicate consolidation plans with safe ordering and reporting.

This module consumes the dry-run plan produced by ``duplicate_consolidation``
and performs the requested playlist rewrites, artwork transfers, metadata
normalization, and loser disposition handling (quarantine/delete). The
execution pipeline is intentionally conservative:

- Playlists are backed up before edits and validated after rewrites.
- A cancellation request stops processing after the current operation.
- Errors stop subsequent steps while preserving any backups already written.
- Every run produces machine-readable and human-readable reports.
"""
from __future__ import annotations

import base64
import concurrent.futures
import datetime
import errno
import hashlib
import html
import importlib.util
import json
import logging
import platform
import os
import re
import shutil
import tempfile
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

from duplicate_consolidation import (
    ARTWORK_VASTLY_DIFFERENT_THRESHOLD,
    ArtworkDirective,
    ConsolidationPlan,
    _hamming_distance,
    _capture_library_state,
    _iter_playlists as _planner_iter_playlists,
    consolidation_plan_from_dict,
    planned_winner_rename,
    playlist_path_key,
    playlist_write_encoding,
    read_playlist_lines,
)
from indexer_control import IndexCancelled, cancel_event as global_cancel_event
from utils.audio_metadata_reader import _extract_cover_payloads, read_sidecar_artwork_bytes
from utils.path_helpers import ensure_long_path, match_separators, playlist_entry, safe_relpath
from utils.year_normalizer import UNKNOWN_YEAR

try:
    _MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
except ValueError:  # pragma: no cover - defensive: broken mutagen installs
    _MUTAGEN_AVAILABLE = False
if _MUTAGEN_AVAILABLE:
    from mutagen import File as MutagenFile  # type: ignore
else:  # pragma: no cover - optional dependency
    MutagenFile = None  # type: ignore

logger = logging.getLogger(__name__)

def _default_log(msg: str) -> None:
    return None


@dataclass(frozen=True)
class PlanLoadResult:
    plan: ConsolidationPlan
    source: str
    path: str | None
    input_type: str


@dataclass
class ExecutionConfig:
    """Configuration for consolidation execution."""

    library_root: str
    reports_dir: str
    playlists_dir: str | None = None
    quarantine_dir: str | None = None
    cancel_event: threading.Event | None = None
    log_callback: Callable[[str], None] | None = None
    apply_metadata: bool = True
    apply_artwork: bool = True
    allow_review_required: bool = False
    operation_limit: int | None = 20_000
    confirm_operation_overage: bool = False
    allow_deletion: bool = False
    confirm_deletion: bool = False
    dry_run_execute: bool = False
    retain_losers: bool = False
    quarantine_flatten: bool = False
    show_artwork_variants: bool = True


@dataclass
class ExecutionAction:
    """Individual action recorded in the audit log."""

    step: str
    target: str
    status: str
    detail: str
    planned: bool = False
    attempted: bool = False
    verified: bool = False
    verification_detail: str | None = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "target": self.target,
            "status": self.status,
            "detail": self.detail,
            "planned": self.planned,
            "attempted": self.attempted,
            "verified": self.verified,
            "verification_detail": self.verification_detail,
            "metadata": dict(self.metadata),
        }


@dataclass
class PlaylistRewriteResult:
    """Outcome of rewriting a single playlist."""

    playlist: str
    backup_path: str
    replaced_entries: int
    status: str
    detail: str = ""
    original_playlist: str | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "playlist": self.playlist,
            "backup_path": self.backup_path,
            "replaced_entries": self.replaced_entries,
            "status": self.status,
            "detail": self.detail,
            "original_playlist": self.original_playlist,
        }


@dataclass
class ExecutionResult:
    """Aggregate execution outcome and artifact locations."""

    success: bool
    actions: List[ExecutionAction]
    playlist_reports: List[PlaylistRewriteResult]
    quarantine_index: Dict[str, str]
    report_paths: Dict[str, str]


EXECUTION_REPORT_FILENAME = "execution_report.html"
INTERNAL_TAG_KEYS = {"album_type"}


def _is_placeholder_tag(key: str, value: object) -> bool:
    """The normalized "Unknown" year means *no* year; never write it to a file."""
    return key == "year" and value == UNKNOWN_YEAR
_NUMERIC_SUFFIX_RE = re.compile(r"\s*\((\d+)\)$")


def _timestamped_dir(base: str) -> str:
    os.makedirs(base, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    for attempt in range(1000):
        suffix = f"_{attempt}" if attempt else ""
        path = os.path.join(base, f"consolidation_run_{ts}{suffix}")
        try:
            os.makedirs(path)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("Unable to create unique consolidation report directory.")


def _atomic_write_text(path: str, content: str, *, encoding: str = "utf-8") -> None:
    directory = os.path.dirname(path) or "."
    safe_directory = ensure_long_path(directory)
    safe_path = ensure_long_path(path)
    os.makedirs(safe_directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=safe_directory, encoding=encoding) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        os.replace(ensure_long_path(tmp_path), safe_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _atomic_write_json(path: str, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    _atomic_write_text(path, text)


def _iter_playlists(playlists_dir: str) -> Iterable[str]:
    return _planner_iter_playlists(playlists_dir)


def _read_playlist(path: str) -> List[str]:
    """A playlist's lines in whatever encoding it uses (FileNotFoundError if absent)."""
    return read_playlist_lines(path)[0]


def _playlist_format(path: str) -> tuple[str, str]:
    """``(encoding, newline)`` a rewrite of ``path`` must keep."""
    try:
        with open(ensure_long_path(path), "rb") as handle:
            data = handle.read()
        _lines, encoding = read_playlist_lines(path)
    except OSError:
        return "utf-8", "\n"
    return playlist_write_encoding(path, encoding), ("\r\n" if b"\r\n" in data else "\n")


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(32, cpu_count + 4)


def _capture_state_with_cache(path: str) -> Dict[str, object]:
    return _capture_library_state(path, quick=True)


def _normalize_playlist_entry(entry: str, playlist_dir: str) -> str:
    if not entry or entry.startswith("#"):
        return entry
    if os.path.isabs(entry):
        return os.path.normpath(entry)
    return os.path.normpath(os.path.join(playlist_dir, entry))


def _rewrite_playlist(path: str, mapping: Mapping[str, str], *, base_dir: str | None = None) -> tuple[int, List[str]]:
    playlist_dir = base_dir or os.path.dirname(path)
    try:
        lines = _read_playlist(path)
    except FileNotFoundError:
        return 0, []
    return _rewrite_playlist_lines(lines, mapping, playlist_dir=playlist_dir)


def _rewrite_playlist_lines(
    lines: Sequence[str],
    mapping: Mapping[str, str],
    *,
    playlist_dir: str,
) -> tuple[int, List[str]]:
    replaced = 0
    keyed = {playlist_path_key(k): v for k, v in mapping.items()}
    new_lines: List[str] = []
    for ln in lines:
        entry = ln.strip()
        normalized = _normalize_playlist_entry(entry, playlist_dir)
        new_abs = keyed.get(playlist_path_key(normalized)) if entry and not entry.startswith("#") else None
        if new_abs is not None:
            replaced += 1
            # Absolute stays absolute, relative stays relative (absolute
            # across Windows drives), in the entry's own separator style.
            new_lines.append(playlist_entry(new_abs, playlist_dir, like=entry))
        else:
            new_lines.append(ln)
    return replaced, new_lines


def _write_playlist_atomic(path: str, lines: Sequence[str]) -> None:
    """Replace ``path`` with ``lines`` in the file's own encoding and line endings.

    Raises ``UnicodeEncodeError`` (before anything is written) when a new
    entry cannot be represented in that encoding — the playlist is then left
    alone and the run stops before any loser is touched.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    encoding, newline = _playlist_format(path)
    payload = "".join(ln + newline for ln in lines).encode(encoding)
    tmp = tempfile.NamedTemporaryFile("wb", delete=False, dir=directory)
    tmp_path = tmp.name
    try:
        with tmp:
            tmp.write(payload)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _verify_playlist_write(path: str, expected_lines: Sequence[str]) -> tuple[bool, str]:
    try:
        actual_lines = _read_playlist(path)
    except FileNotFoundError:
        return False, "Playlist missing after rewrite."
    if list(expected_lines) != actual_lines:
        return False, "Playlist contents do not match expected rewrite."
    return True, "Playlist rewrite verified."


def _validate_playlist_lines(lines: Sequence[str], *, base_dir: str | None = None) -> List[str]:
    playlist_dir = base_dir or os.getcwd()
    missing: List[str] = []
    for ln in lines:
        normalized = _normalize_playlist_entry(ln.strip(), playlist_dir)
        if not normalized or normalized.startswith("#"):
            continue
        if not os.path.exists(ensure_long_path(normalized)):
            missing.append(normalized)
    return missing


def _validate_playlist(path: str, *, base_dir: str | None = None) -> List[str]:
    playlist_dir = base_dir or os.path.dirname(path)
    try:
        lines = _read_playlist(path)
    except FileNotFoundError:
        return ["Playlist missing after rewrite."]

    return _validate_playlist_lines(lines, base_dir=playlist_dir)


def _backup_playlist(path: str, backup_root: str) -> str:
    try:
        rel = os.path.relpath(path, os.path.commonpath([path, backup_root]))
    except ValueError:  # another drive on Windows
        rel = os.path.basename(path)
    dest = os.path.join(backup_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def _rename_no_overwrite(src: str, dest: str) -> None:
    """Rename ``src`` to ``dest``; raise ``FileExistsError`` rather than replace a file.

    ``os.replace`` (and ``os.rename`` off Windows) silently overwrite. A hard
    link refuses an existing name atomically; where links are unsupported
    (FAT/exFAT, some shares) a checked rename is the fallback.
    """
    safe_src = ensure_long_path(src)
    safe_dest = ensure_long_path(dest)
    if os.name == "nt":
        os.rename(safe_src, safe_dest)  # never replaces an existing file on Windows
        return
    try:
        os.link(safe_src, safe_dest, follow_symlinks=False)
    except FileExistsError:
        raise
    except (OSError, NotImplementedError):
        if os.path.lexists(safe_dest):
            raise FileExistsError(errno.EEXIST, "Destination exists", dest)
        os.rename(safe_src, safe_dest)
        return
    os.unlink(safe_src)


def _strip_trailing_numeric_suffix(path: str) -> str | None:
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    match = _NUMERIC_SUFFIX_RE.search(stem)
    if not match:
        return None
    trimmed = stem[: match.start()].rstrip()
    if not trimmed:
        return None
    return os.path.join(os.path.dirname(path), f"{trimmed}{ext}")


def _extract_artwork_bytes(path: str) -> bytes | None:
    payload = read_sidecar_artwork_bytes(path)
    if payload is not None:
        return payload
    if MutagenFile is None:
        return None
    audio = None
    try:
        audio = MutagenFile(ensure_long_path(path))
        if audio is None:
            return None
        tags = getattr(audio, "tags", {}) or {}
        ext = os.path.splitext(path)[1].lower()
        if ext in {".m4a", ".mp4"}:
            covr_payloads = tags.get("covr") if isinstance(tags, Mapping) else None
            if covr_payloads:
                if covr_payloads[0]:
                    logger.debug("Found M4A cover art in covr atom for %s", path)
                else:
                    logger.debug("M4A covr atom present but empty for %s", path)
            else:
                logger.debug("M4A covr atom missing for %s", path)
        payloads = _extract_cover_payloads(audio)
        if payloads:
            return bytes(payloads[0])
    except Exception:
        return None
    finally:
        _close_mutagen_audio(audio)
    return None


def _payload_mime(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _edit_copy_then_replace(target: str, edit: Callable[[str], tuple]) -> tuple:
    """Run ``edit`` on a copy of ``target``; swap the copy in only if it worked.

    ``edit(copy_path)`` returns a tuple whose first item says whether it
    succeeded. The winner — the copy the user keeps — is therefore never left
    half-written by a crash, a full disk or a tag library error mid-save: it
    is either untouched or atomically replaced by the fully written copy. A
    symlinked winner is edited at its real location.
    """
    real = os.path.realpath(ensure_long_path(target))
    directory = os.path.dirname(real)
    ext = os.path.splitext(real)[1]
    fd, tmp_path = tempfile.mkstemp(prefix=".alphadex-edit-", suffix=ext, dir=directory)
    os.close(fd)
    try:
        shutil.copy2(real, tmp_path)
        outcome = edit(tmp_path)
        if outcome and outcome[0]:
            os.replace(tmp_path, real)
            tmp_path = ""
        return outcome
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


_VORBIS_COMMENT_MODULES = ("mutagen.oggvorbis", "mutagen.oggopus", "mutagen.oggflac", "mutagen.oggspeex")
_ID3_CHUNK_MODULES = ("mutagen.mp3", "mutagen.wave", "mutagen.aiff", "mutagen.trueaudio", "mutagen.dsf")


def _embedded_front_covers(audio: object) -> List[bytes]:
    """Front-cover payloads in a loaded mutagen file, for every format we write."""
    covers: List[bytes] = []
    module = type(audio).__module__
    if module == "mutagen.flac":
        covers.extend(bytes(p.data) for p in getattr(audio, "pictures", []) if getattr(p, "type", None) == 3)
    tags = getattr(audio, "tags", None)
    if tags is None:
        return covers
    if module.startswith(_VORBIS_COMMENT_MODULES):
        from mutagen.flac import Picture  # type: ignore

        for value in tags.get("metadata_block_picture", []) or []:
            try:
                picture = Picture(base64.b64decode(value))
            except Exception:
                continue
            if picture.type == 3:
                covers.append(bytes(picture.data))
    elif module.startswith("mutagen.mp4"):
        covers.extend(bytes(c) for c in tags.get("covr", []) or [])
    elif hasattr(tags, "getall"):
        covers.extend(bytes(f.data) for f in tags.getall("APIC") if getattr(f, "type", None) == 3)
    return covers


def _embed_front_cover(path: str, payload: bytes) -> tuple[bool, str]:
    """Embed ``payload`` as the front cover of ``path`` in that format's own way.

    Other pictures are kept. Formats without a known cover container are
    refused rather than guessed: the old fallback wrote an ID3 header onto the
    front of OGG/Opus/WAV/AIFF files, which made them unreadable.
    """
    mime = _payload_mime(payload)
    audio = MutagenFile(path)
    try:
        if audio is None:
            return False, "Unsupported audio format for embedding."
        module = type(audio).__module__
        if module == "mutagen.flac":
            from mutagen.flac import Picture  # type: ignore

            kept = [p for p in audio.pictures if getattr(p, "type", None) != 3]
            pic = Picture()
            pic.data, pic.type, pic.mime, pic.desc = payload, 3, mime, "Front cover"
            audio.clear_pictures()
            for p in kept:
                audio.add_picture(p)
            audio.add_picture(pic)
            audio.save()
            return True, "Embedded FLAC picture."
        if module.startswith("mutagen.mp4"):
            from mutagen.mp4 import MP4Cover  # type: ignore

            if audio.tags is None:
                audio.add_tags()
            cover_format = MP4Cover.FORMAT_PNG if mime == "image/png" else MP4Cover.FORMAT_JPEG
            audio.tags["covr"] = [MP4Cover(payload, imageformat=cover_format)]
            audio.save()
            return True, "Embedded MP4 cover atom."
        if module.startswith(_VORBIS_COMMENT_MODULES):
            from mutagen.flac import Picture  # type: ignore

            if audio.tags is None:
                audio.add_tags()
            kept: List[str] = []
            for value in audio.tags.get("metadata_block_picture", []) or []:
                try:
                    if Picture(base64.b64decode(value)).type == 3:
                        continue
                except Exception:
                    pass
                kept.append(value)
            pic = Picture()
            pic.data, pic.type, pic.mime, pic.desc = payload, 3, mime, "Front cover"
            audio.tags["metadata_block_picture"] = kept + [base64.b64encode(pic.write()).decode("ascii")]
            audio.save()
            return True, "Embedded Vorbis comment picture."
        if module.startswith(_ID3_CHUNK_MODULES):
            from mutagen.id3 import APIC  # type: ignore

            if audio.tags is None:
                audio.add_tags()
            preserved = [f for f in audio.tags.getall("APIC") if getattr(f, "type", None) != 3]
            audio.tags.delall("APIC")
            for frame in preserved:
                audio.tags.add(frame)
            audio.tags.add(APIC(encoding=3, mime=mime, type=3, desc="Front cover", data=payload))
            audio.save()
            return True, "Embedded ID3 cover art."
        return False, f"Embedding artwork into {type(audio).__name__} files is not supported; file left unchanged."
    finally:
        _close_mutagen_audio(audio)


def _apply_artwork(action: ArtworkDirective) -> tuple[bool, str]:
    payload = _extract_artwork_bytes(action.source)
    if payload is None:
        return False, "No artwork payload available to embed."

    sidecar = f"{action.target}.artwork"
    directory = os.path.dirname(sidecar)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=directory) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name
    try:
        os.replace(tmp_path, sidecar)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if MutagenFile is None:
        return True, "Mutagen unavailable; wrote artwork sidecar only."

    try:
        return _edit_copy_then_replace(action.target, lambda copy: _embed_front_cover(copy, payload))
    except Exception as exc:
        return False, f"Artwork embed failed: {exc}"


def _verify_artwork_write(action: ArtworkDirective) -> tuple[bool, str]:
    if not os.path.exists(ensure_long_path(action.target)):
        return False, "Target missing after artwork update."
    sidecar = f"{action.target}.artwork"
    if MutagenFile is None:
        if not os.path.exists(sidecar):
            return False, "Artwork sidecar missing after update."
        try:
            if os.path.getsize(sidecar) <= 0:
                return False, "Artwork sidecar is empty."
        except OSError:
            return False, "Artwork sidecar could not be read."
        return True, "Artwork write verified."
    payload = _extract_artwork_bytes(action.source)
    audio = None
    try:
        audio = MutagenFile(ensure_long_path(action.target))
        if audio is None:
            return False, "Target is no longer a readable audio file after the artwork update."
        if payload is not None and payload not in _embedded_front_covers(audio):
            return False, "Embedded front cover does not match the planned artwork."
    except Exception as exc:
        return False, f"Target could not be read back after the artwork update ({exc})."
    finally:
        _close_mutagen_audio(audio)
    return True, "Artwork write verified."


# Planner-internal keys: normalised duplicates of a real tag (``track`` and
# ``disc`` mirror ``tracknumber``/``discnumber``, ``year`` mirrors ``date``) or
# planning context (``album_type``). Writing them added stray TRACK=, DISC=,
# YEAR= fields to FLAC/Vorbis files.
_INTERNAL_PLAN_TAGS = {"album_type", "track", "disc", "year", "cover_hash", "artwork_hash"}


def _tags_to_write(planned: Mapping[str, object], changes: Mapping[str, object]) -> Dict[str, object]:
    """The tag values to write for one winner: exactly the previewed changes.

    The preview shows ``metadata_changes``; the executor used to write the whole
    ``planned_winner_tags`` dict instead, including internal keys. Values that
    are missing or the "Unknown" year placeholder are never written, and no
    tag is ever removed.
    """
    out: Dict[str, object] = {}
    for key in changes:
        if key in _INTERNAL_PLAN_TAGS:
            continue
        value = planned.get(key)
        if value in (None, "", []) or _is_placeholder_tag(key, value):
            continue
        out[key] = value
    if "year" in changes and "date" not in out and not planned.get("date"):
        year = planned.get("year")
        if year not in (None, "") and not _is_placeholder_tag("year", year):
            out["date"] = str(year)
    return out


def _normalized_tag_value(value: object) -> List[str]:
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


def _write_easy_tags(path: str, planned_tags: Mapping[str, object]) -> tuple[bool, str, List[str]]:
    audio = MutagenFile(path, easy=True)
    try:
        if audio is None:
            return True, "Unsupported audio format; metadata not written.", []
        changed = False
        skipped: List[str] = []
        for key, value in planned_tags.items():
            if value is None or _is_placeholder_tag(key, value):
                continue
            if key in INTERNAL_TAG_KEYS:
                skipped.append(key)
                continue
            normalized = _normalized_tag_value(value)
            try:
                current = audio.tags.get(key) if audio.tags else None
                if current != normalized:
                    audio[key] = normalized
                    changed = True
            except Exception:
                skipped.append(key)
        if changed:
            try:
                audio.save()
            except Exception as exc:
                return False, f"Failed to save metadata tags ({exc}).", skipped
        if skipped:
            skipped_list = ", ".join(sorted(set(skipped)))
            return True, f"Metadata normalized; skipped unsupported keys: {skipped_list}.", skipped
        return True, "Metadata normalized.", skipped
    finally:
        _close_mutagen_audio(audio)


def _apply_metadata(target: str, planned_tags: Mapping[str, object]) -> tuple[bool, str, List[str]]:
    if MutagenFile is None:
        return True, "Mutagen unavailable; metadata not written.", []
    if not planned_tags:
        return True, "No previewed tag changes to write.", []
    try:
        return _edit_copy_then_replace(target, lambda copy: _write_easy_tags(copy, planned_tags))
    except Exception as exc:
        return False, f"Failed to write metadata tags ({exc}).", []


def _verify_metadata_write(target: str, planned_tags: Mapping[str, object], skipped_keys: Iterable[str]) -> tuple[bool, str]:
    if not os.path.exists(ensure_long_path(target)):
        return False, "Target missing after metadata update."
    if MutagenFile is None:
        return False, "Mutagen unavailable; metadata not verified."
    if not planned_tags:
        return True, "Nothing to verify."
    audio = MutagenFile(ensure_long_path(target), easy=True)
    try:
        if audio is None or audio.tags is None:
            return False, "Metadata unavailable after update."
        skipped = set(skipped_keys)
        mismatched: List[str] = []
        for key, value in planned_tags.items():
            if value is None or key in INTERNAL_TAG_KEYS or key in skipped:
                continue
            if _is_placeholder_tag(key, value):
                continue
            current = audio.tags.get(key)
            if current != _normalized_tag_value(value):
                mismatched.append(key)
        if mismatched:
            return False, f"Metadata mismatch after write: {', '.join(sorted(mismatched))}."
        return True, "Metadata write verified."
    finally:
        _close_mutagen_audio(audio)


def _close_mutagen_audio(audio: object | None) -> None:
    if audio is None:
        return
    closer = getattr(audio, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass
    fileobj = getattr(audio, "fileobj", None)
    if fileobj is not None:
        file_close = getattr(fileobj, "close", None)
        if callable(file_close):
            try:
                file_close()
            except Exception:
                pass


def _name_key(path: str) -> str:
    """Comparison key for a destination path, as a case-insensitive file system sees it.

    Windows and macOS treat ``Song.mp3``/``song.mp3`` — and the NFC/NFD forms
    of ``Café.mp3`` — as one file. Two losers whose names differ only that way
    must never be given "different" quarantine names that are really the same.
    """
    return unicodedata.normalize("NFC", os.path.normcase(os.path.abspath(path))).casefold()


def _same_path(left: str, right: str) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _unique_destination(dest: str, taken: set[str]) -> str:
    """Return ``dest``, or ``name (n).ext`` if that file exists or is ``taken``.

    The chosen path is added to ``taken`` so concurrent cleanup tasks in one
    run never pick the same destination. (Moving onto an existing path made
    ``shutil.move`` silently replace an earlier quarantined loser.) Names in
    ``taken`` are compared case- and Unicode-normalisation-insensitively.
    """
    taken_keys = {_name_key(t) for t in taken}
    stem, ext = os.path.splitext(dest)
    candidate = dest
    counter = 1
    while _name_key(candidate) in taken_keys or os.path.lexists(ensure_long_path(candidate)):
        candidate = f"{stem} ({counter}){ext}"
        counter += 1
    taken.add(candidate)
    return candidate


_MAX_DESTINATION_ATTEMPTS = 100_000


def _reserve_destination(dest: str, taken_keys: set[str]) -> str:
    """Claim a free quarantine path for one loser and return it.

    The claim is an empty placeholder file created with ``O_EXCL``, so the file
    system itself refuses a name that already exists — including one that only
    differs in case or Unicode form on Windows/macOS — and the loser is then
    renamed *over its own placeholder*, never over anybody else's file.
    ``taken_keys`` (see ``_name_key``) keeps one run's concurrent moves apart.
    """
    stem, ext = os.path.splitext(dest)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    for counter in range(_MAX_DESTINATION_ATTEMPTS):
        candidate = dest if counter == 0 else f"{stem} ({counter}){ext}"
        key = _name_key(candidate)
        if key in taken_keys:
            continue
        try:
            fd = os.open(ensure_long_path(candidate), flags, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        taken_keys.add(key)
        return candidate
    raise RuntimeError(f"No free quarantine name for {dest}")


def _discard_placeholder(path: str | None) -> None:
    """Remove a destination claimed by ``_reserve_destination`` if it is still empty."""
    if not path:
        return
    safe = ensure_long_path(path)
    try:
        if os.path.isfile(safe) and not os.path.islink(safe) and os.path.getsize(safe) == 0:
            os.remove(safe)
    except OSError:
        pass


class _OriginalNotRemoved(OSError):
    """A cross-device move copied the loser but could not delete the original."""


def _move_onto_placeholder(src: str, dest: str) -> None:
    """Move ``src`` onto ``dest``, an empty placeholder this run created.

    Same file system: one atomic rename. Across file systems (a Quarantine on
    another drive): copy to a temporary file beside ``dest``, rename it over
    the placeholder, then delete ``src``. If ``src`` cannot be deleted (open
    in a player, read-only, …) the copy is removed again and the error is
    raised, so the loser is simply still where it was — never in two places
    while the report claims one. A failed or partial copy is removed too.
    """
    safe_src = ensure_long_path(src)
    safe_dest = ensure_long_path(dest)
    try:
        os.replace(safe_src, safe_dest)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    fd, tmp_path = tempfile.mkstemp(prefix=".alphadex-move-", suffix=".part", dir=os.path.dirname(safe_dest))
    os.close(fd)
    try:
        shutil.copy2(safe_src, tmp_path)
        os.replace(tmp_path, safe_dest)
        tmp_path = ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    try:
        os.remove(safe_src)
    except OSError as exc:
        try:
            os.remove(safe_dest)
            undo = "the quarantine copy was removed again; the file is still in the library"
        except OSError as undo_exc:
            undo = f"the quarantine copy at {dest} could not be removed either ({undo_exc})"
        raise _OriginalNotRemoved(
            exc.errno, f"copied to quarantine but the original could not be deleted ({exc}); {undo}"
        ) from exc


def _quarantine_path(path: str, quarantine_root: str, library_root: str, *, flatten: bool) -> str:
    if flatten:
        os.makedirs(quarantine_root, exist_ok=True)
        return os.path.join(quarantine_root, os.path.basename(path))
    try:
        rel = os.path.relpath(path, library_root)
        if rel.startswith(".."):
            rel = os.path.basename(path)
    except ValueError:
        rel = os.path.basename(path)
    dest = os.path.join(quarantine_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return dest


def _compute_plan_signature(plan: ConsolidationPlan) -> str:
    canonical = {
        "generated_at": plan.generated_at.isoformat(),
        "groups": [g.to_dict() for g in plan.groups],
        "snapshot": {k: dict(v) for k, v in plan.source_snapshot.items()},
        "fingerprint_settings": dict(plan.fingerprint_settings),
        "threshold_settings": dict(plan.threshold_settings),
        "impacted_playlists": list(plan.impacted_playlists or []),
    }
    text = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_state_drift(expected: Mapping[str, object], actual: Mapping[str, object]) -> str | None:
    if not expected:
        return "Missing expected snapshot."
    if not actual.get("exists"):
        if expected.get("exists"):
            return "File missing since preview."
        return None
    if expected.get("exists") is False and actual.get("exists"):
        return "File appeared after preview."
    for key in ("size", "mtime"):
        if key in expected and key in actual and expected[key] != actual[key]:
            return f"{key} mismatch (expected {expected.get(key)}, found {actual.get(key)})."
    return None


def _plan_structure_problems(plan: ConsolidationPlan) -> List[str]:
    """Reasons the plan's groups are not a safe partition, or [] when they are.

    Every loser must belong to exactly one group, and no path may be both a
    winner and a loser: otherwise one group's loser cleanup removes another
    group's winner, and a group can end with no surviving copy at all.
    """
    problems: List[str] = []
    winners: Dict[str, str] = {}
    losers_seen: Dict[str, str] = {}
    for group in plan.groups:
        winners[os.path.normcase(os.path.abspath(group.winner_path))] = group.group_id
    for group in plan.groups:
        winner_key = os.path.normcase(os.path.abspath(group.winner_path))
        for loser in group.losers:
            key = os.path.normcase(os.path.abspath(loser))
            if key == winner_key:
                problems.append(f"group {group.group_id}: the winner is also listed as a loser ({loser})")
            elif key in winners:
                problems.append(
                    f"group {group.group_id}: loser {loser} is the winner of group {winners[key]}"
                )
            if key in losers_seen and losers_seen[key] != group.group_id:
                problems.append(f"{loser} is a loser in groups {losers_seen[key]} and {group.group_id}")
            losers_seen.setdefault(key, group.group_id)
    return problems


def _run_in_pool(
    items: Sequence[object],
    fn: Callable[[object], object],
    *,
    max_workers: int,
    on_done: Callable[[object, "concurrent.futures.Future[object]"], None],
    stop_requested: Callable[[], bool],
) -> List[object]:
    """Run ``fn(item)`` on a thread pool, at most ``max_workers`` at a time.

    ``on_done(item, future)`` runs on the calling thread for every task that
    was started, in completion order — so every file operation that happened
    is reported, even when the run stops early. ``stop_requested`` is checked
    before each submission: once it is true (cancel, or a failure ``on_done``
    noted) nothing new starts, and the tasks already running finish and are
    reported. (Submitting everything up front used to let every queued move
    and delete run after a cancel or an error, unrecorded.)

    Returns the items that were never started, in order.
    """
    queue = list(items)
    position = 0
    pending: Dict["concurrent.futures.Future[object]", object] = {}
    workers = max(1, int(max_workers or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            while position < len(queue) and len(pending) < workers and not stop_requested():
                item = queue[position]
                position += 1
                pending[executor.submit(fn, item)] = item
            if not pending:
                break
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                on_done(pending.pop(future), future)
    return queue[position:]


def _coerce_consolidation_plan(
    plan_input: ConsolidationPlan | Mapping[str, object] | str,
    log: Callable[[str], None],
) -> PlanLoadResult:
    if isinstance(plan_input, ConsolidationPlan):
        return PlanLoadResult(plan=plan_input, source="in-memory plan", path=None, input_type="in-memory")
    payload: object = plan_input
    source = "plan payload"
    plan_path: str | None = None
    input_type = "mapping"
    if isinstance(plan_input, str):
        if os.path.exists(plan_input):
            if not os.path.isfile(plan_input):
                raise ValueError(f"Plan path is not a file: {plan_input}")
            try:
                with open(plan_input, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except OSError as exc:
                raise ValueError(f"Plan file could not be read: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"Plan JSON could not be decoded: {exc}") from exc
            source = f"preview output {plan_input}"
            plan_path = plan_input
            input_type = "file"
        else:
            try:
                payload = json.loads(plan_input)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Plan JSON could not be decoded: {exc}") from exc
            source = "json payload"
            input_type = "json string"
    if isinstance(payload, Mapping):
        if "plan" in payload:
            inner = payload.get("plan")
            if not isinstance(inner, Mapping):
                raise ValueError("Plan payload has a non-mapping 'plan' field.")
            payload = inner
            source = f"{source} (wrapped)"
        return PlanLoadResult(
            plan=consolidation_plan_from_dict(payload),
            source=source,
            path=plan_path,
            input_type=input_type,
        )
    raise ValueError("Plan input must be a ConsolidationPlan, mapping, or JSON string.")


def execute_consolidation_plan(
    plan_input: ConsolidationPlan | Mapping[str, object] | str,
    config: ExecutionConfig,
) -> ExecutionResult:
    """Execute the provided consolidation plan with safety guarantees."""

    cancel = config.cancel_event or global_cancel_event
    log = config.log_callback or _default_log
    actions: List[ExecutionAction] = []
    playlist_results: List[PlaylistRewriteResult] = []
    quarantine_index: Dict[str, str] = {}
    group_lookup: Dict[str, object] = {}
    plan: ConsolidationPlan | None = None
    computed_signature = ""
    max_workers = _default_max_workers()
    io_workers = min(16, max_workers)

    run_root = _timestamped_dir(config.reports_dir)
    reports_dir = os.path.join(run_root, "reports")
    backups_dir = os.path.join(run_root, "playlist_backups")
    validation_playlists_dir = os.path.join(run_root, "playlist_validation")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(backups_dir, exist_ok=True)
    playlists_dir = (
        config.playlists_dir
        if config.playlists_dir is not None
        else os.path.join(config.library_root, "Playlists")
    )
    update_playlists = bool(playlists_dir)
    quarantine_dir = config.quarantine_dir or os.path.join(config.library_root, "Quarantine")

    def _record(
        step: str,
        target: str,
        status: str,
        detail: str,
        *,
        planned: bool = False,
        attempted: bool = False,
        verified: bool = False,
        verification_detail: str | None = None,
        **metadata: object,
    ) -> None:
        actions.append(
            ExecutionAction(
                step=step,
                target=target,
                status=status,
                detail=detail,
                planned=planned,
                attempted=attempted,
                verified=verified,
                verification_detail=verification_detail,
                metadata=metadata,
            )
        )

    def _check_cancel(stage: str) -> None:
        if cancel.is_set():
            _record(stage, "execution", "cancelled", "Execution cancelled by user request.")
            raise IndexCancelled()

    loser_to_winner: Dict[str, str] = {}
    loser_disposition: Dict[str, str] = {}
    artwork_actions: List[ArtworkDirective] = []
    planned_tags: Dict[str, Dict[str, object]] = {}
    playlist_rewrite_map: Dict[str, str] = {}
    path_to_group: Dict[str, str] = {}
    playlist_targets: Dict[str, str] = {}
    playlist_base_dirs: Dict[str, str] = {}

    def _find_loser_references(
        working_playlists: Mapping[str, str],
        playlist_base_dirs: Mapping[str, str] | None,
        *,
        playlist_lines: Mapping[str, Sequence[str]] | None = None,
    ) -> Dict[str, List[str]]:
        refs: Dict[str, List[str]] = {}
        for playlist, working_path in working_playlists.items():
            lines = None
            if playlist_lines is not None:
                lines = list(playlist_lines.get(working_path, [])) or None
            if lines is None:
                try:
                    lines = _read_playlist(working_path)
                except FileNotFoundError:
                    continue
            playlist_dir = (playlist_base_dirs or {}).get(playlist) or os.path.dirname(working_path)
            normalized = {
                playlist_path_key(_normalize_playlist_entry(ln.strip(), playlist_dir)) for ln in lines if ln.strip()
            }
            hits = sorted(
                [loser for loser in playlist_rewrite_map.keys() if playlist_path_key(loser) in normalized]
            )
            if hits:
                refs[playlist] = hits
        return refs

    success = True

    try:
        try:
            plan_result = _coerce_consolidation_plan(plan_input, log)
            plan = plan_result.plan
            plan_source = plan_result.source
            plan_path = plan_result.path
            plan_input_type = plan_result.input_type
            log(f"Execute plan input resolved from {plan_source}.")
        except ValueError as exc:
            success = False
            _record("preflight", "plan", "blocked", f"Invalid consolidation plan: {exc}")
            raise RuntimeError(f"Execution blocked: invalid consolidation plan ({exc})") from exc

        group_lookup = {g.group_id: g for g in plan.groups}
        for group in plan.groups:
            if group.metadata_changes:
                to_write = _tags_to_write(group.planned_winner_tags, group.metadata_changes)
                if to_write:
                    planned_tags[group.winner_path] = to_write
            for loser in group.losers:
                loser_to_winner[loser] = group.winner_path
                loser_disposition[loser] = group.loser_disposition.get(loser, "quarantine")
                path_to_group[loser] = group.group_id
            path_to_group[group.winner_path] = group.group_id
            artwork_actions.extend(group.artwork)
            playlist_rewrite_map.update(group.playlist_rewrites)

        computed_signature = _compute_plan_signature(plan)
        _check_cancel("start")

        if plan.plan_signature and plan.plan_signature != computed_signature:
            generated_at_iso = plan.generated_at.isoformat()
            snapshot_count = len(plan.source_snapshot)
            canonical_snapshot = {k: dict(v) for k, v in plan.source_snapshot.items()}
            snapshot_id = hashlib.sha256(
                json.dumps(canonical_snapshot, sort_keys=True).encode("utf-8")
            ).hexdigest()
            drift_examples: List[str] = []
            for path, expected in plan.source_snapshot.items():
                if len(drift_examples) >= 5:
                    break
                actual = _capture_library_state(path)
                delta = _detect_state_drift(expected, actual)
                if delta:
                    drift_examples.append(f"{path}: {delta}")
            plan_path_mtime = None
            plan_path_size = None
            if plan_path:
                try:
                    plan_path_mtime = datetime.datetime.fromtimestamp(
                        os.path.getmtime(plan_path), datetime.timezone.utc
                    ).isoformat()
                    plan_path_size = os.path.getsize(plan_path)
                except OSError:
                    plan_path_mtime = None
                    plan_path_size = None
            log(
                "Plan signature mismatch summary: "
                + json.dumps(
                    {
                        "plan_path": plan_path,
                        "plan_source": plan_source,
                        "plan_input_type": plan_input_type,
                        "plan_path_mtime": plan_path_mtime,
                        "plan_path_size": plan_path_size,
                        "plan_signature": plan.plan_signature,
                        "computed_signature": computed_signature,
                        "generated_at": generated_at_iso,
                        "snapshot_id": snapshot_id,
                        "snapshot_count": snapshot_count,
                        "group_count": len(plan.groups),
                        "drift_examples": drift_examples,
                    },
                    sort_keys=True,
                )
            )
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Plan signature mismatch — preview plan is stale or not the one being executed. Rerun Preview.",
                expected_signature=plan.plan_signature,
                computed_signature=computed_signature,
                plan_path=plan_path or "unknown",
                plan_source=plan_source,
                plan_input_type=plan_input_type,
                plan_path_mtime=plan_path_mtime,
                plan_path_size=plan_path_size,
                generated_at=generated_at_iso,
                snapshot_id=snapshot_id,
                snapshot_count=snapshot_count,
                group_count=len(plan.groups),
                drift_examples=drift_examples,
            )
            raise RuntimeError("Execution blocked: plan signature mismatch.")

        if not plan.source_snapshot:
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Plan missing source snapshot; rerun preview to capture library state.",
            )
            raise RuntimeError("Execution blocked: missing source snapshot.")

        structure_problems = _plan_structure_problems(plan)
        if structure_problems:
            success = False
            for problem in structure_problems:
                _record(
                    "preflight",
                    "plan",
                    "blocked",
                    f"Plan groups overlap — {problem}. Rerun Preview.",
                )
            raise RuntimeError("Execution blocked: plan groups overlap.")

        if plan.requires_review and not config.allow_review_required:
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Review-required groups present; enable override to continue.",
                review_required=plan.review_required_count,
            )
            raise RuntimeError("Execution blocked: review-required groups present.")

        planned_operations = (
            len(loser_to_winner) + len(artwork_actions) + len(planned_tags) + len(playlist_rewrite_map)
        )
        if config.operation_limit and planned_operations > config.operation_limit and not config.confirm_operation_overage:
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Planned operations exceed configured limit.",
                planned_operations=planned_operations,
                operation_limit=config.operation_limit,
            )
            raise RuntimeError("Execution blocked: operation limit exceeded.")

        drift: Dict[str, str] = {}
        if plan.source_snapshot:
            affected_paths: set[str] = set(loser_to_winner.keys())
            affected_paths.update(loser_to_winner.values())
            affected_paths.update(planned_tags.keys())
            for art in artwork_actions:
                affected_paths.add(art.source)
                affected_paths.add(art.target)
            if affected_paths:
                work_items = [
                    (path, expected)
                    for path, expected in plan.source_snapshot.items()
                    if path in affected_paths
                ]
            else:
                work_items = list(plan.source_snapshot.items())
            log(
                "Preflight drift check: "
                f"{len(work_items)} of {len(plan.source_snapshot)} snapshot paths."
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_capture_state_with_cache, path)
                    for path, _expected in work_items
                ]
                for (path, expected), future in zip(work_items, futures):
                    _check_cancel("preflight")
                    actual = future.result()
                    delta = _detect_state_drift(expected, actual)
                    if delta:
                        drift[path] = delta
        if drift:
            success = False
            for path, reason in drift.items():
                _record(
                    "preflight",
                    path,
                    "failed",
                    f"Library changed since preview: {reason}",
                    group_id=path_to_group.get(path),
                )
            raise RuntimeError("Execution blocked: library drift detected.")

        has_deletions = any(disposition == "delete" for disposition in loser_disposition.values())
        if has_deletions and not config.dry_run_execute:
            if not (config.allow_deletion and config.confirm_deletion):
                success = False
                _record(
                    "preflight",
                    "execution",
                    "blocked",
                    "Deletion requested but explicit confirmation was not provided.",
                    deletions_requested=sum(1 for d in loser_disposition.values() if d == "delete"),
                )
                raise RuntimeError("Execution blocked: deletions require confirmation.")

        # Step 1: backup playlists that will change
        impacted: List[str] = []
        playlist_groups: Dict[str, set[str]] = {}
        playlist_source_lines: Dict[str, List[str]] = {}
        if plan and plan.impacted_playlists:
            candidate_playlists = list(plan.impacted_playlists)
        else:
            candidate_playlists = list(_iter_playlists(playlists_dir))
        rewrite_keys = {playlist_path_key(loser): loser for loser in playlist_rewrite_map}
        for playlist in candidate_playlists:
            try:
                lines = _read_playlist(playlist)
            except FileNotFoundError:
                continue
            playlist_source_lines[playlist] = lines
            playlist_dir = os.path.dirname(playlist)
            normalized = {
                playlist_path_key(_normalize_playlist_entry(ln.strip(), playlist_dir)) for ln in lines if ln.strip()
            }
            affected = {rewrite_keys[key] for key in normalized & rewrite_keys.keys()}
            if affected:
                impacted.append(playlist)
                playlist_groups[playlist] = {path_to_group.get(path, "") for path in affected if path in path_to_group}

        playlist_backups: Dict[str, str] = {}
        playlist_expected_lines: Dict[str, List[str]] = {}
        for playlist in impacted:
            _check_cancel("backup_playlists")
            try:
                if config.dry_run_execute:
                    working_copy = os.path.join(validation_playlists_dir, os.path.relpath(playlist, playlists_dir))
                    os.makedirs(os.path.dirname(working_copy), exist_ok=True)
                    shutil.copy2(playlist, working_copy)
                    playlist_backups[playlist] = working_copy
                    playlist_targets[playlist] = working_copy
                    playlist_base_dirs[playlist] = os.path.dirname(playlist)
                    verified = os.path.exists(working_copy)
                    verification_detail = (
                        "Playlist copy verified." if verified else "Playlist copy missing after write."
                    )
                    _record(
                        "backup_playlists",
                        playlist,
                        "success" if verified else "failed",
                        "Playlist copied for dry-run validation.",
                        planned=True,
                        attempted=True,
                        verified=verified,
                        verification_detail=verification_detail,
                        backup_path=working_copy,
                        group_ids=sorted(g for g in playlist_groups.get(playlist, []) if g),
                    )
                    if not verified:
                        raise RuntimeError(f"Playlist copy verification failed for {playlist}")
                else:
                    backup_path = _backup_playlist(playlist, backups_dir)
                    playlist_backups[playlist] = backup_path
                    playlist_targets[playlist] = playlist
                    playlist_base_dirs[playlist] = os.path.dirname(playlist)
                    verified = os.path.exists(backup_path)
                    verification_detail = (
                        "Playlist backup verified." if verified else "Playlist backup missing after write."
                    )
                    _record(
                        "backup_playlists",
                        playlist,
                        "success" if verified else "failed",
                        "Playlist backed up.",
                        planned=True,
                        attempted=True,
                        verified=verified,
                        verification_detail=verification_detail,
                        backup_path=backup_path,
                        group_ids=sorted(g for g in playlist_groups.get(playlist, []) if g),
                    )
                    if not verified:
                        raise RuntimeError(f"Playlist backup verification failed for {playlist}")
            except Exception as exc:
                success = False
                _record(
                    "backup_playlists",
                    playlist,
                    "failed",
                    f"Failed to backup playlist: {exc}",
                    planned=True,
                    attempted=True,
                )
                raise

        # Step 2: rewrite playlists
        for playlist in impacted:
            _check_cancel("rewrite_playlists")
            target_playlist = playlist_targets.get(playlist, playlist)
            base_dir = playlist_base_dirs.get(playlist, os.path.dirname(target_playlist))
            source_lines = playlist_source_lines.get(playlist)
            if source_lines is not None:
                replaced, new_lines = _rewrite_playlist_lines(
                    source_lines,
                    playlist_rewrite_map,
                    playlist_dir=base_dir,
                )
            else:
                replaced, new_lines = _rewrite_playlist(target_playlist, playlist_rewrite_map, base_dir=base_dir)
            playlist_expected_lines[target_playlist] = list(new_lines)
            if replaced == 0:
                playlist_results.append(
                    PlaylistRewriteResult(
                        playlist=target_playlist,
                        backup_path=playlist_backups.get(playlist, _backup_playlist(playlist, backups_dir)),
                        replaced_entries=0,
                        status="skipped",
                        detail="No matching loser entries found." + (" (dry-run copy)" if config.dry_run_execute else ""),
                        original_playlist=playlist if target_playlist != playlist else None,
                    )
                )
                if target_playlist not in playlist_expected_lines and source_lines is not None:
                    playlist_expected_lines[target_playlist] = list(source_lines)
                _record(
                    "rewrite_playlists",
                    target_playlist,
                    "skipped",
                    "No entries required updates.",
                    planned=True,
                    attempted=False,
                    verified=False,
                    verification_detail="No rewrite needed.",
                    group_ids=sorted(g for g in playlist_groups.get(playlist, []) if g),
                    original_playlist=playlist if target_playlist != playlist else None,
                )
                continue
            try:
                _write_playlist_atomic(target_playlist, new_lines)
                verified, verification_detail = _verify_playlist_write(target_playlist, new_lines)
                status = "success" if verified else "failed"
                if not verified:
                    success = False
                playlist_results.append(
                    PlaylistRewriteResult(
                        playlist=target_playlist,
                        backup_path=playlist_backups.get(
                            playlist,
                            os.path.join(
                                backups_dir,
                                os.path.relpath(playlist, playlists_dir),
                            ),
                        ),
                        replaced_entries=replaced,
                        status=status,
                        original_playlist=playlist if target_playlist != playlist else None,
                    )
                )
                _record(
                    "rewrite_playlists",
                    target_playlist,
                    status,
                    "Playlist rewritten.",
                    planned=True,
                    attempted=True,
                    verified=verified,
                    verification_detail=verification_detail,
                    replaced=replaced,
                    group_ids=sorted(g for g in playlist_groups.get(playlist, []) if g),
                    original_playlist=playlist if target_playlist != playlist else None,
                )
                if not verified:
                    raise RuntimeError(f"Playlist rewrite verification failed for {target_playlist}")
            except Exception as exc:
                success = False
                _record(
                    "rewrite_playlists",
                    target_playlist,
                    "failed",
                    f"Failed to rewrite playlist: {exc}",
                    planned=True,
                    attempted=True,
                )
                raise

        # Step 3: validate rewritten playlists
        for result in playlist_results:
            _check_cancel("validate_playlists")
            base_dir = playlist_base_dirs.get(result.original_playlist or result.playlist, os.path.dirname(result.playlist))
            expected_lines = playlist_expected_lines.get(result.playlist)
            if expected_lines is not None:
                missing = _validate_playlist_lines(expected_lines, base_dir=base_dir)
            else:
                missing = _validate_playlist(result.playlist, base_dir=base_dir)
            # Entries that were already dead before the rewrite are the user's
            # business, not this run's: they used to abort every execution
            # whose losers appeared in that playlist.
            source_lines = playlist_source_lines.get(result.original_playlist or result.playlist)
            already_missing = (
                set(_validate_playlist_lines(source_lines, base_dir=base_dir)) if source_lines is not None else set()
            )
            if already_missing:
                log(
                    f"Playlist {result.playlist}: {len(already_missing)} entr(y/ies) already pointed at "
                    "missing files before this run; left as they were."
                )
            missing = [m for m in missing if m not in already_missing]
            if missing:
                success = False
                try:
                    if os.path.exists(result.backup_path):
                        shutil.copy2(result.backup_path, result.playlist)
                finally:
                    _record(
                        "validate_playlists",
                        result.playlist,
                        "failed",
                        "Playlist validation failed; restored backup.",
                        attempted=True,
                        missing=missing,
                        group_ids=sorted(g for g in playlist_groups.get(result.original_playlist or result.playlist, []) if g),
                        original_playlist=result.original_playlist,
                    )
                raise RuntimeError(f"Playlist validation failed for {result.playlist}")
            _record(
                "validate_playlists",
                result.playlist,
                "success",
                "Playlist validated.",
                attempted=True,
                group_ids=sorted(g for g in playlist_groups.get(result.original_playlist or result.playlist, []) if g),
                original_playlist=result.original_playlist,
            )

        lingering_refs = _find_loser_references(
            playlist_targets or {},
            playlist_base_dirs,
            playlist_lines=playlist_expected_lines,
        )
        if lingering_refs:
            success = False
            for playlist, losers in lingering_refs.items():
                _record(
                    "validate_playlists",
                    playlist,
                    "failed",
                    "Loser references remain after rewrite; aborting cleanup.",
                    losers=losers,
                    group_ids=sorted({path_to_group.get(l, "") for l in losers if l in path_to_group}),
                )
            raise RuntimeError("Playlist rewrites incomplete; loser references remain.")

        def _apply_artwork_task(art: ArtworkDirective) -> tuple[ArtworkDirective, bool, str, bool, str | None]:
            ok, detail = _apply_artwork(art)
            verified = False
            verification_detail = None
            if ok:
                verified, verification_detail = _verify_artwork_write(art)
            return art, ok, detail, verified, verification_detail

        def _apply_metadata_task(
            target: str,
            tags: Dict[str, object],
        ) -> tuple[str, bool, str, bool, str | None, set[str]]:
            ok, detail, skipped_keys = _apply_metadata(target, tags)
            verified = False
            verification_detail = None
            if ok:
                verified, verification_detail = _verify_metadata_write(target, tags, skipped_keys)
            return target, ok, detail, verified, verification_detail, skipped_keys

        quarantine_lock = threading.Lock()
        quarantine_taken: set[str] = set()

        def _loser_refusal(loser: str) -> tuple[str, str] | None:
            """(skip_reason, detail) when ``loser`` must not be touched, else None.

            Checked immediately before the file operation, not only at
            preflight: steps 1-5 can take long, and a loser is only ever
            removed while its winner is still there to replace it.
            """
            winner = loser_to_winner.get(loser)
            if not winner:
                return "no_winner", "Not removed: the plan names no winner for this loser."
            safe_winner = ensure_long_path(winner)
            if not os.path.exists(safe_winner):
                return "winner_missing", f"Not removed: its winner is missing ({winner})."
            try:
                if os.path.samefile(safe_winner, ensure_long_path(loser)):
                    return (
                        "same_file",
                        f"Not removed: it is the same file as its winner {winner} (a link or alias).",
                    )
            except OSError as exc:
                return "winner_unreadable", f"Not removed: could not compare with its winner ({exc})."
            expected = plan.source_snapshot.get(loser) if plan else None
            if not expected:
                return "no_snapshot", "Not removed: the preview recorded no state for this file. Rerun Preview."
            drift = _detect_state_drift(expected, _capture_library_state(loser))
            if drift:
                return "changed_since_preview", f"Not removed: changed since the preview ({drift})."
            return None

        def _cleanup_loser_task(item: object) -> Dict[str, object]:
            loser, disposition = item  # type: ignore[misc]
            outcome: Dict[str, object] = {
                "loser": loser,
                "action": disposition,
                "status": "failed",
                "verified": False,
                "verification_detail": None,
                "dest": None,
                "detail": "",
                "skip_reason": None,
            }
            if not os.path.lexists(ensure_long_path(loser)):
                outcome.update(
                    status="skipped",
                    detail="Missing source; loser not moved or deleted.",
                    verification_detail="Source missing.",
                    skip_reason="missing_source",
                )
                return outcome
            refusal = _loser_refusal(loser)
            if refusal:
                outcome.update(
                    status="failed",
                    detail=refusal[1],
                    verification_detail="Refused before touching the file.",
                    skip_reason=refusal[0],
                    refused=True,
                )
                return outcome
            if disposition == "delete":
                os.remove(ensure_long_path(loser))
                verified = not os.path.lexists(ensure_long_path(loser))
                outcome.update(
                    status="success" if verified else "failed",
                    verified=verified,
                    detail="Loser deleted.",
                    verification_detail=(
                        "Loser deletion verified." if verified else "Loser still present after delete."
                    ),
                )
                return outcome
            dest = _quarantine_path(
                loser,
                quarantine_dir,
                config.library_root,
                flatten=config.quarantine_flatten,
            )
            with quarantine_lock:
                dest = _reserve_destination(dest, quarantine_taken)
            outcome["dest"] = dest
            try:
                _move_onto_placeholder(loser, dest)
            except BaseException:
                _discard_placeholder(dest)
                raise
            verified = os.path.exists(ensure_long_path(dest)) and not os.path.lexists(ensure_long_path(loser))
            outcome.update(
                status="success" if verified else "failed",
                verified=verified,
                detail="Loser quarantined.",
                verification_detail=(
                    "Loser quarantine verified." if verified else "Quarantine move not confirmed."
                ),
            )
            return outcome

        # Step 4: apply artwork transfers
        if config.dry_run_execute and config.apply_artwork:
            for art in artwork_actions:
                _record(
                    "artwork",
                    art.target,
                    "skipped",
                    f"Dry-run execute enabled; artwork not applied (source: {art.source}).",
                    planned=True,
                    attempted=False,
                    verified=False,
                    verification_detail="Dry-run execute enabled.",
                    source=art.source,
                    group_id=path_to_group.get(art.target),
                    skip_reason="dry_run",
                )
        elif config.apply_artwork:
            artwork_errors: List[str] = []

            def _artwork_done(item: object, future: "concurrent.futures.Future[object]") -> None:
                nonlocal success
                art = item  # type: ignore[assignment]
                try:
                    _art, ok, detail, verified, verification_detail = future.result()  # type: ignore[misc]
                except Exception as exc:
                    success = False
                    artwork_errors.append(f"Artwork copy failed for {art.target}: {exc}")
                    _record(
                        "artwork",
                        art.target,
                        "failed",
                        f"Artwork copy failed: {exc}",
                        planned=True,
                        attempted=True,
                        verified=False,
                        verification_detail="Artwork task failed.",
                        source=art.source,
                        group_id=path_to_group.get(art.target),
                    )
                    return
                status = "success" if ok and verified else "failed"
                if status == "failed":
                    success = False
                    if ok and verification_detail:
                        detail = f"{detail} Verification failed: {verification_detail}"
                    artwork_errors.append(
                        f"Artwork {'verification' if ok else 'copy'} failed for {art.target}"
                    )
                _record(
                    "artwork",
                    art.target,
                    status,
                    f"{detail or art.reason} (source: {art.source})",
                    planned=True,
                    attempted=True,
                    verified=verified,
                    verification_detail=verification_detail,
                    source=art.source,
                    group_id=path_to_group.get(art.target),
                )

            not_started = _run_in_pool(
                artwork_actions,
                _apply_artwork_task,  # type: ignore[arg-type]
                max_workers=io_workers,
                on_done=_artwork_done,
                stop_requested=lambda: cancel.is_set() or bool(artwork_errors),
            )
            for art in not_started:
                _record(
                    "artwork",
                    art.target,  # type: ignore[attr-defined]
                    "skipped",
                    "Execution stopped; artwork not applied.",
                    planned=True,
                    attempted=False,
                    verified=False,
                    verification_detail="Not attempted.",
                    source=art.source,  # type: ignore[attr-defined]
                    group_id=path_to_group.get(art.target),  # type: ignore[attr-defined]
                    skip_reason="not_attempted",
                )
            if artwork_errors:
                raise RuntimeError(artwork_errors[0])
            _check_cancel("artwork")

        # Step 5: apply metadata normalization
        if config.dry_run_execute and config.apply_metadata:
            for target, _tags in planned_tags.items():
                _record(
                    "metadata",
                    target,
                    "skipped",
                    "Dry-run execute enabled; metadata not written.",
                    planned=True,
                    attempted=False,
                    verified=False,
                    verification_detail="Dry-run execute enabled.",
                    group_id=path_to_group.get(target),
                    skip_reason="dry_run",
                )
        elif config.apply_metadata:
            if MutagenFile is None:
                for target in planned_tags:
                    _record(
                        "metadata",
                        target,
                        "skipped",
                        "Mutagen unavailable; metadata not written.",
                        planned=True,
                        attempted=False,
                        verified=False,
                        verification_detail="Mutagen unavailable.",
                        group_id=path_to_group.get(target),
                        skip_reason="mutagen_unavailable",
                    )
            else:
                metadata_items: List[tuple[str, Dict[str, object]]] = []
                for target, tags in planned_tags.items():
                    if not os.path.exists(ensure_long_path(target)):
                        _record(
                            "metadata",
                            target,
                            "skipped",
                            "Missing source; metadata not written.",
                            planned=True,
                            attempted=False,
                            verified=False,
                            verification_detail="Source missing.",
                            group_id=path_to_group.get(target),
                            skip_reason="missing_source",
                        )
                        continue
                    metadata_items.append((target, tags))
                metadata_errors: List[str] = []

                def _metadata_done(item: object, future: "concurrent.futures.Future[object]") -> None:
                    nonlocal success
                    target = item[0]  # type: ignore[index]
                    try:
                        (
                            _target,
                            ok,
                            detail,
                            verified,
                            verification_detail,
                            _skipped_keys,
                        ) = future.result()  # type: ignore[misc]
                    except Exception as exc:
                        success = False
                        metadata_errors.append(f"Metadata normalization failed for {target}: {exc}")
                        _record(
                            "metadata",
                            target,
                            "failed",
                            f"Metadata normalization failed: {exc}",
                            planned=True,
                            attempted=True,
                            verified=False,
                            verification_detail="Metadata task failed.",
                            group_id=path_to_group.get(target),
                        )
                        return
                    status = "success" if ok and verified else "failed"
                    if status == "failed":
                        success = False
                        if ok and verification_detail:
                            detail = f"{detail} Verification failed: {verification_detail}"
                        metadata_errors.append(
                            f"Metadata {'verification' if ok else 'normalization'} failed for {target}"
                        )
                    _record(
                        "metadata",
                        target,
                        status,
                        detail,
                        planned=True,
                        attempted=True,
                        verified=verified,
                        verification_detail=verification_detail,
                        group_id=path_to_group.get(target),
                    )

                not_started = _run_in_pool(
                    metadata_items,
                    lambda item: _apply_metadata_task(*item),  # type: ignore[misc]
                    max_workers=io_workers,
                    on_done=_metadata_done,
                    stop_requested=lambda: cancel.is_set() or bool(metadata_errors),
                )
                for target, _tags in not_started:  # type: ignore[misc]
                    _record(
                        "metadata",
                        target,
                        "skipped",
                        "Execution stopped; metadata not written.",
                        planned=True,
                        attempted=False,
                        verified=False,
                        verification_detail="Not attempted.",
                        group_id=path_to_group.get(target),
                        skip_reason="not_attempted",
                    )
                if metadata_errors:
                    raise RuntimeError(metadata_errors[0])
                _check_cancel("metadata")

        # Step 6: quarantine or delete losers
        if not config.dry_run_execute:
            cleanup_items: List[tuple[str, str]] = []
            for loser, disposition in loser_disposition.items():
                effective = disposition if disposition in ("retain", "quarantine", "delete") else None
                if effective is None:
                    effective = "retain" if config.retain_losers else "quarantine"
                if effective == "retain":
                    quarantine_index[loser] = "retained"
                    _record(
                        "loser_cleanup",
                        loser,
                        "skipped",
                        "Loser retained by plan disposition.",
                        planned=True,
                        attempted=False,
                        verified=False,
                        verification_detail="Loser retained.",
                        disposition=effective,
                        group_id=path_to_group.get(loser),
                        skip_reason="retain",
                    )
                    continue
                cleanup_items.append((loser, effective))

            cleanup_errors: List[str] = []
            refused_losers: List[str] = []

            def _cleanup_done(item: object, future: "concurrent.futures.Future[object]") -> None:
                nonlocal success
                loser, effective = item  # type: ignore[misc]
                try:
                    outcome = future.result()
                except Exception as exc:
                    success = False
                    verb = "delete" if effective == "delete" else "quarantine"
                    cleanup_errors.append(f"Failed to {verb} {loser}: {exc}")
                    _record(
                        "loser_cleanup",
                        loser,
                        "failed",
                        f"Failed to clean up loser ({verb}): {exc}",
                        planned=True,
                        attempted=True,
                        disposition=effective,
                        group_id=path_to_group.get(loser),
                    )
                    return
                assert isinstance(outcome, dict)
                status = str(outcome["status"])
                action = str(outcome["action"])
                dest = outcome.get("dest")
                verified = bool(outcome.get("verified"))
                if status == "skipped":
                    _record(
                        "loser_cleanup",
                        loser,
                        "skipped",
                        str(outcome["detail"]),
                        planned=True,
                        attempted=False,
                        verified=False,
                        verification_detail=outcome.get("verification_detail"),
                        disposition=effective,
                        group_id=path_to_group.get(loser),
                        skip_reason=outcome.get("skip_reason"),
                    )
                    return
                if outcome.get("refused"):
                    # The file was not touched: a per-file safety refusal, not
                    # an I/O error, so independent groups carry on.
                    success = False
                    refused_losers.append(loser)
                    log(f"Loser left in place: {loser} — {outcome['detail']}")
                    _record(
                        "loser_cleanup",
                        loser,
                        "failed",
                        str(outcome["detail"]),
                        planned=True,
                        attempted=False,
                        verified=False,
                        verification_detail=outcome.get("verification_detail"),
                        disposition=effective,
                        group_id=path_to_group.get(loser),
                        skip_reason=outcome.get("skip_reason"),
                    )
                    return
                if not verified:
                    success = False
                    cleanup_errors.append(f"{outcome.get('verification_detail')} ({loser})")
                if action == "delete":
                    if verified:
                        quarantine_index[loser] = "deleted"
                    _record(
                        "loser_cleanup",
                        loser,
                        status,
                        "Loser deleted.",
                        planned=True,
                        attempted=True,
                        verified=verified,
                        verification_detail=outcome.get("verification_detail"),
                        disposition="delete",
                        group_id=path_to_group.get(loser),
                    )
                else:
                    if verified and dest:
                        quarantine_index[loser] = str(dest)
                    _record(
                        "loser_cleanup",
                        loser,
                        status,
                        "Loser quarantined.",
                        planned=True,
                        attempted=True,
                        verified=verified,
                        verification_detail=outcome.get("verification_detail"),
                        quarantine_path=dest,
                        disposition="quarantine",
                        group_id=path_to_group.get(loser),
                    )

            not_started = _run_in_pool(
                cleanup_items,
                _cleanup_loser_task,
                max_workers=io_workers,
                on_done=_cleanup_done,
                stop_requested=lambda: cancel.is_set() or bool(cleanup_errors),
            )
            stop_detail = (
                "Execution cancelled by user request"
                if cancel.is_set() and not cleanup_errors
                else "Execution stopped after an error"
            )
            for loser, effective in not_started:  # type: ignore[misc]
                _record(
                    "loser_cleanup",
                    loser,
                    "skipped",
                    f"{stop_detail}; loser not moved or deleted.",
                    planned=True,
                    attempted=False,
                    verified=False,
                    verification_detail="Not attempted.",
                    disposition=effective,
                    group_id=path_to_group.get(loser),
                    skip_reason="not_attempted",
                )
            if cleanup_errors:
                raise RuntimeError(cleanup_errors[0])
            _check_cancel("loser_cleanup")
            if refused_losers:
                log(f"{len(refused_losers)} loser(s) were left in place; see the report.")
        else:
            for loser, disposition in loser_disposition.items():
                _check_cancel("loser_cleanup")
                effective = disposition if disposition in ("retain", "quarantine", "delete") else None
                if effective is None:
                    effective = "retain" if config.retain_losers else "quarantine"
                quarantine_index[loser] = "retained"
                if effective == "retain":
                    detail = "Dry-run execute enabled; loser retained in place."
                    skip_reason = "retain"
                else:
                    detail = "Dry-run execute enabled; loser not moved or deleted."
                    skip_reason = "dry_run"
                _record(
                    "loser_cleanup",
                    loser,
                    "skipped",
                    detail,
                    planned=True,
                    attempted=False,
                    verified=False,
                    verification_detail="Dry-run execute enabled.",
                    disposition=effective,
                    group_id=path_to_group.get(loser),
                    skip_reason=skip_reason,
                )

        # Step 7: cleanup redundant numeric suffixes on winners
        # Only the renames the preview showed (``planned_winner_rename``), and
        # only once the same-named loser really left this run.
        cleanup_candidates: List[tuple[str, str]] = []
        for group in plan.groups if plan else []:
            candidate = planned_winner_rename(group)
            if not candidate:
                continue
            if not config.dry_run_execute:
                freed = [
                    loser
                    for loser in group.losers
                    if _same_path(loser, candidate)
                    and quarantine_index.get(loser) not in (None, "retained")
                ]
                if not freed:
                    continue
            cleanup_candidates.append((group.winner_path, candidate))

        rename_map: Dict[str, str] = {}
        if cleanup_candidates:
            if config.dry_run_execute:
                for original, candidate in cleanup_candidates:
                    _record(
                        "filename_cleanup",
                        original,
                        "skipped",
                        "Dry-run execute enabled; filename cleanup not applied.",
                        planned=True,
                        attempted=False,
                        verified=False,
                        verification_detail="Dry-run execute enabled.",
                        proposed_path=candidate,
                        group_id=path_to_group.get(original),
                        skip_reason="dry_run",
                    )
            else:
                for original, candidate in cleanup_candidates:
                    _check_cancel("filename_cleanup")
                    if not os.path.exists(original):
                        _record(
                            "filename_cleanup",
                            original,
                            "skipped",
                            "Missing source; filename cleanup not applied.",
                            planned=True,
                            attempted=False,
                            verified=False,
                            verification_detail="Source missing.",
                            proposed_path=candidate,
                            group_id=path_to_group.get(original),
                            skip_reason="missing_source",
                        )
                        continue
                    if os.path.lexists(ensure_long_path(candidate)):
                        _record(
                            "filename_cleanup",
                            original,
                            "skipped",
                            "Filename cleanup skipped to avoid collision.",
                            planned=True,
                            attempted=False,
                            verified=False,
                            verification_detail="Destination already exists.",
                            proposed_path=candidate,
                            group_id=path_to_group.get(original),
                            skip_reason="collision",
                        )
                        continue
                    try:
                        _rename_no_overwrite(original, candidate)
                        verified = os.path.exists(ensure_long_path(candidate)) and not os.path.lexists(
                            ensure_long_path(original)
                        )
                        verification_detail = (
                            "Filename cleanup verified."
                            if verified
                            else "Cleanup rename not confirmed."
                        )
                        status = "success" if verified else "failed"
                        if verified:
                            rename_map[original] = candidate
                        else:
                            success = False
                        _record(
                            "filename_cleanup",
                            original,
                            status,
                            "Removed redundant numeric suffix from filename.",
                            planned=True,
                            attempted=True,
                            verified=verified,
                            verification_detail=verification_detail,
                            new_path=candidate,
                            group_id=path_to_group.get(original),
                        )
                        if not verified:
                            raise RuntimeError(f"Filename cleanup verification failed for {original}")
                    except Exception as exc:
                        success = False
                        _record(
                            "filename_cleanup",
                            original,
                            "failed",
                            f"Failed to cleanup filename: {exc}",
                            planned=True,
                            attempted=True,
                            proposed_path=candidate,
                            group_id=path_to_group.get(original),
                        )
                        raise

        if rename_map:
            for old_path, new_path in rename_map.items():
                group_id = path_to_group.pop(old_path, None)
                if group_id:
                    path_to_group[new_path] = group_id
            if not update_playlists:
                _record(
                    "filename_cleanup_playlists",
                    "playlists",
                    "skipped",
                    "Playlist updates disabled; filename cleanup not applied to playlists.",
                    planned=True,
                    attempted=False,
                    verified=False,
                    verification_detail="Playlist updates disabled.",
                    skip_reason="playlists_disabled",
                )
            else:
                for playlist in _iter_playlists(playlists_dir):
                    _check_cancel("filename_cleanup_playlists")
                    base_dir = os.path.dirname(playlist)
                    try:
                        before_lines = _read_playlist(playlist)
                    except FileNotFoundError:
                        continue
                    replaced, new_lines = _rewrite_playlist_lines(before_lines, rename_map, playlist_dir=base_dir)
                    if replaced == 0:
                        continue
                    already_missing = set(_validate_playlist_lines(before_lines, base_dir=base_dir))
                    if playlist not in playlist_backups:
                        try:
                            backup_path = _backup_playlist(playlist, backups_dir)
                            playlist_backups[playlist] = backup_path
                            _record(
                                "backup_playlists",
                                playlist,
                                "success",
                                "Playlist backed up for filename cleanup.",
                                planned=True,
                                attempted=True,
                                verified=os.path.exists(backup_path),
                                verification_detail="Backup created for filename cleanup.",
                                backup_path=backup_path,
                            )
                        except Exception as exc:
                            success = False
                            _record(
                                "backup_playlists",
                                playlist,
                                "failed",
                                f"Failed to backup playlist for cleanup: {exc}",
                                planned=True,
                                attempted=True,
                            )
                            raise
                    try:
                        _write_playlist_atomic(playlist, new_lines)
                        verified, verification_detail = _verify_playlist_write(playlist, new_lines)
                        status = "success" if verified else "failed"
                        if not verified:
                            success = False
                        _record(
                            "filename_cleanup_playlists",
                            playlist,
                            status,
                            "Playlist rewritten for filename cleanup.",
                            planned=True,
                            attempted=True,
                            verified=verified,
                            verification_detail=verification_detail,
                            replaced=replaced,
                        )
                        if not verified:
                            raise RuntimeError(f"Playlist cleanup rewrite verification failed for {playlist}")
                        missing = [
                            m for m in _validate_playlist(playlist, base_dir=base_dir) if m not in already_missing
                        ]
                        if missing:
                            success = False
                            _record(
                                "filename_cleanup_validate",
                                playlist,
                                "failed",
                                "Playlist validation failed after filename cleanup.",
                                attempted=True,
                                missing=missing,
                            )
                            raise RuntimeError(f"Playlist cleanup validation failed for {playlist}")
                        _record(
                            "filename_cleanup_validate",
                            playlist,
                            "success",
                            "Playlist validated after filename cleanup.",
                            attempted=True,
                        )
                    except Exception as exc:
                        success = False
                        _record(
                            "filename_cleanup_playlists",
                            playlist,
                            "failed",
                            f"Failed to rewrite playlist for cleanup: {exc}",
                            planned=True,
                            attempted=True,
                        )
                        raise
    except IndexCancelled:
        success = False
        _record("execution", "execution", "cancelled", "Execution cancelled by user request.")
    except Exception as exc:
        success = False
        _record("execution", "execution", "failed", f"Execution failed: {exc}")
    finally:
        consolidated_report_path = os.path.join(reports_dir, "execution_report.json")
        html_report_path = os.path.join(reports_dir, EXECUTION_REPORT_FILENAME)

        plan_signature = plan.plan_signature if plan else None
        source_snapshot = plan.source_snapshot if plan else {}
        rollup = {
            "planned": sum(1 for a in actions if a.planned),
            "attempted": sum(1 for a in actions if a.attempted),
            "verified": sum(1 for a in actions if a.verified),
            "skipped": sum(1 for a in actions if a.status == "skipped"),
            "failed": sum(1 for a in actions if a.status in {"failed", "blocked"}),
        }
        consolidated_payload = {
            "success": success,
            "actions": [a.to_dict() for a in actions],
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "plan_signature": plan_signature or computed_signature,
            "computed_signature": computed_signature,
            "source_snapshot": {k: dict(v) for k, v in source_snapshot.items()},
            "plan": plan.to_dict() if plan else None,
            "playlists": [p.to_dict() for p in playlist_results],
            "quarantine_index": dict(quarantine_index),
            "metadata_plans": {path: dict(tags) for path, tags in planned_tags.items()},
            "action_rollup": dict(rollup),
            "execution_context": {
                "dry_run_execute": config.dry_run_execute,
                "library_root": os.path.abspath(config.library_root),
                "quarantine_dir": os.path.abspath(quarantine_dir) if quarantine_dir else None,
            },
            "run_paths": {
                "reports_dir": reports_dir,
                "backups_dir": backups_dir,
            },
        }

        try:
            _atomic_write_json(consolidated_report_path, consolidated_payload)
        except Exception:
            pass

        try:
            actions_by_group: Dict[str, List[ExecutionAction]] = {}
            for action in actions:
                meta = action.metadata or {}
                group_ids: set[str] = set()
                gid = meta.get("group_id")
                if isinstance(gid, str) and gid:
                    group_ids.add(gid)
                meta_groups = meta.get("group_ids")
                if isinstance(meta_groups, (list, tuple, set)):
                    group_ids.update([g for g in meta_groups if isinstance(g, str) and g])
                if not group_ids:
                    actions_by_group.setdefault("__general__", []).append(action)
                else:
                    for gid in group_ids:
                        actions_by_group.setdefault(gid, []).append(action)

            def _basename(path: str) -> str:
                return os.path.basename(path) or path

            def _group_state(group_actions: List[ExecutionAction]) -> str:
                statuses = {act.status for act in group_actions}
                for state in ("failed", "blocked", "cancelled"):
                    if state in statuses:
                        return state
                if statuses == {"skipped"}:
                    return "skipped"
                return "success" if "success" in statuses else "unknown"

            def _action_badges(group_actions: List[ExecutionAction]) -> List[str]:
                badges: List[str] = []
                meta_actions = [act for act in group_actions if act.step == "metadata"]
                meta_success = sum(1 for act in meta_actions if act.status == "success")
                meta_skipped = sum(1 for act in meta_actions if act.status == "skipped")
                meta_failed = sum(1 for act in meta_actions if act.status == "failed")
                if meta_success:
                    badges.append(f"metadata normalized ({meta_success})")
                if meta_skipped:
                    badges.append(f"metadata skipped ({meta_skipped})")
                if meta_failed:
                    badges.append(f"metadata failed ({meta_failed})")

                cleanup_actions = [act for act in group_actions if act.step == "loser_cleanup"]
                quarantined = sum(
                    1
                    for act in cleanup_actions
                    if act.metadata.get("disposition") == "quarantine"
                )
                deleted = sum(
                    1
                    for act in cleanup_actions
                    if act.metadata.get("disposition") == "delete"
                )
                retained = sum(
                    1
                    for act in cleanup_actions
                    if act.metadata.get("disposition") == "retain"
                )
                if quarantined:
                    badges.append(f"quarantined ({quarantined})")
                if deleted:
                    badges.append(f"deleted ({deleted})")
                if retained:
                    badges.append(f"retained ({retained})")

                art_actions = [act for act in group_actions if act.step == "artwork"]
                art_success = sum(1 for act in art_actions if act.status == "success")
                if art_success:
                    badges.append(f"artwork copied ({art_success})")
                return badges

            def _album_art_src(
                group: object,
                track_path: str,
                include_group_chosen: bool = True,
            ) -> str | None:
                candidates: List[str] = []
                if include_group_chosen and group is not None:
                    chosen = getattr(group, "chosen_artwork_source", None)
                    if isinstance(chosen, Mapping):
                        chosen_path = chosen.get("path")
                        if isinstance(chosen_path, str) and chosen_path:
                            candidates.append(chosen_path)
                if track_path:
                    candidates.append(track_path)
                for path in candidates:
                    payload = _extract_artwork_bytes(path)
                    if payload:
                        return _image_data_uri(payload)
                return None

            def _image_data_uri(payload: bytes) -> str:
                mime = _image_mime(payload)
                encoded = base64.b64encode(payload).decode("ascii")
                return f"data:{mime};base64,{encoded}"

            def _image_mime(payload: bytes) -> str:
                if payload.startswith(b"\x89PNG\r\n\x1a\n"):
                    return "image/png"
                if payload.startswith(b"\xff\xd8"):
                    return "image/jpeg"
                if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
                    return "image/webp"
                return "image/jpeg"

            def _format_metadata_notes(meta: Dict[str, object]) -> str:
                notes = []
                for key, value in meta.items():
                    if key in {"group_id", "group_ids"}:
                        continue
                    if isinstance(value, (list, tuple, set)):
                        rendered = ", ".join(str(v) for v in value)
                    else:
                        rendered = str(value)
                    notes.append(f"{key}: {rendered}")
                return " | ".join(notes)

            def _format_trace_value(value: object) -> str:
                if isinstance(value, (list, tuple, set)):
                    return ", ".join(str(v) for v in value) if value else "—"
                if isinstance(value, dict):
                    return json.dumps(value, sort_keys=True)
                if value in (None, ""):
                    return "—"
                return str(value)

            def _trace_panel(group: object) -> List[str]:
                trace = getattr(group, "pipeline_trace", {})
                if not isinstance(trace, Mapping):
                    trace = {}
                summary = trace.get("summary") if isinstance(trace.get("summary"), Mapping) else {}
                tracks = trace.get("tracks") if isinstance(trace.get("tracks"), Mapping) else {}
                trace_id = f"trace-{getattr(group, 'group_id', '')}"
                lines: List[str] = []
                lines.append(f"<details class='trace' id='{html.escape(trace_id)}'>")
                lines.append("<summary class='trace-summary'>Pipeline Trace</summary>")
                lines.append("<div class='trace-grid'>")
                lines.append("<div class='k'>Bucket ID</div>")
                lines.append(
                    f"<div class='v mono'>{html.escape(_format_trace_value(summary.get('bucket_id')))}</div>"
                )
                lines.append("<div class='k'>Formation</div>")
                lines.append(
                    f"<div class='v'>{html.escape(_format_trace_value(summary.get('formation')))}</div>"
                )
                lines.append("<div class='k'>Metadata key</div>")
                lines.append(
                    f"<div class='v'>{html.escape(_format_trace_value(summary.get('metadata_key')))}</div>"
                )
                lines.append("<div class='k'>Bucket sources</div>")
                lines.append(
                    f"<div class='v tiny'>{html.escape(_format_trace_value(summary.get('bucket_sources')))}</div>"
                )
                lines.append("<div class='k'>Album keys</div>")
                lines.append(
                    f"<div class='v'>{html.escape(_format_trace_value(summary.get('album_keys')))}</div>"
                )
                lines.append("<div class='k'>Missing album</div>")
                lines.append(
                    f"<div class='v'>{html.escape(_format_trace_value(summary.get('missing_album')))}</div>"
                )
                lines.append("<div class='k'>Artwork variants</div>")
                lines.append(
                    f"<div class='v tiny'>{html.escape(_format_trace_value(summary.get('artwork_variants')))}</div>"
                )
                lines.append("<div class='k'>Artwork unknown</div>")
                lines.append(
                    f"<div class='v tiny'>{html.escape(_format_trace_value(summary.get('artwork_unknown')))}</div>"
                )
                lines.append("<div class='k'>Thresholds</div>")
                lines.append(
                    f"<div class='v tiny'>{html.escape(_format_trace_value(summary.get('thresholds')))}</div>"
                )
                lines.append("</div>")

                comparisons = (
                    summary.get("comparisons") if isinstance(summary.get("comparisons"), list) else []
                )
                lines.append("<div class='trace-block'>")
                lines.append(
                    f"<div class='tiny muted'>Comparisons attempted ({len(comparisons)})</div>"
                )
                if comparisons:
                    lines.append("<ul class='edge-list'>")
                    for item in sorted(comparisons, key=lambda e: float(e.get("distance", 0.0))):
                        left = str(item.get("left"))
                        right = str(item.get("right"))
                        distance = float(item.get("distance", 0.0))
                        threshold = float(item.get("threshold", 0.0))
                        verdict = str(item.get("verdict", ""))
                        mixed_codec = " mixed-codec" if item.get("mixed_codec") else ""
                        lines.append(
                            "<li>"
                            f"{html.escape(_basename(left))} ↔ {html.escape(_basename(right))} "
                            f"({distance:.4f} ≤ {threshold:.4f}, {html.escape(verdict)}{mixed_codec})"
                            "</li>"
                        )
                    lines.append("</ul>")
                else:
                    lines.append("<div class='tiny muted'>No comparisons recorded.</div>")
                lines.append("</div>")

                outcome = summary.get("outcome") if isinstance(summary.get("outcome"), Mapping) else {}
                preserve = (
                    outcome.get("preserve_different_art")
                    if isinstance(outcome.get("preserve_different_art"), Mapping)
                    else {}
                )
                lines.append("<div class='trace-block'>")
                lines.append("<div class='tiny muted'>Final outcome</div>")
                lines.append("<div class='trace-grid'>")
                lines.append("<div class='k'>Winner</div>")
                lines.append(
                    f"<div class='v'>{html.escape(_format_trace_value(outcome.get('winner')))}</div>"
                )
                lines.append("<div class='k'>Losers</div>")
                lines.append(
                    f"<div class='v'>{html.escape(_format_trace_value(outcome.get('losers')))}</div>"
                )
                lines.append("<div class='k'>Dispositions</div>")
                lines.append(
                    f"<div class='v tiny'>{html.escape(_format_trace_value(outcome.get('dispositions')))}</div>"
                )
                lines.append("<div class='k'>Preserve (different art)</div>")
                lines.append(
                    f"<div class='v tiny'>{html.escape(_format_trace_value(preserve))}</div>"
                )
                lines.append("</div>")
                lines.append("</div>")

                lines.append("<div class='trace-block'>")
                lines.append("<div class='tiny muted'>Per-file trace</div>")
                for path, payload in sorted(tracks.items(), key=lambda item: str(item[0]).lower()):
                    trace_payload = payload if isinstance(payload, Mapping) else {}
                    lines.append("<details class='trace-track'>")
                    lines.append(
                        "<summary class='trace-track-summary'>"
                        f"{html.escape(_basename(str(path)))}"
                        "</summary>"
                    )
                    lines.append("<div class='trace-grid'>")
                    discovery = (
                        trace_payload.get("discovery")
                        if isinstance(trace_payload.get("discovery"), Mapping)
                        else {}
                    )
                    lines.append("<div class='k'>Discovery</div>")
                    lines.append(
                        f"<div class='v tiny'>{html.escape(_format_trace_value(discovery))}</div>"
                    )
                    metadata_read = (
                        trace_payload.get("metadata_read")
                        if isinstance(trace_payload.get("metadata_read"), Mapping)
                        else {}
                    )
                    lines.append("<div class='k'>Metadata read</div>")
                    lines.append(
                        f"<div class='v tiny'>{html.escape(_format_trace_value(metadata_read))}</div>"
                    )
                    album_art = (
                        trace_payload.get("album_art")
                        if isinstance(trace_payload.get("album_art"), Mapping)
                        else {}
                    )
                    lines.append("<div class='k'>Album art</div>")
                    lines.append(
                        f"<div class='v tiny'>{html.escape(_format_trace_value(album_art))}</div>"
                    )
                    artwork_hashing = (
                        trace_payload.get("artwork_hashing")
                        if isinstance(trace_payload.get("artwork_hashing"), Mapping)
                        else {}
                    )
                    lines.append("<div class='k'>Artwork hashing</div>")
                    lines.append(
                        f"<div class='v tiny'>{html.escape(_format_trace_value(artwork_hashing))}</div>"
                    )
                    artwork_status = (
                        trace_payload.get("artwork_status")
                        if isinstance(trace_payload.get("artwork_status"), Mapping)
                        else {}
                    )
                    lines.append("<div class='k'>Artwork status</div>")
                    lines.append(
                        f"<div class='v tiny'>{html.escape(_format_trace_value(artwork_status))}</div>"
                    )
                    artwork_comp = (
                        trace_payload.get("artwork_comparison")
                        if isinstance(trace_payload.get("artwork_comparison"), Mapping)
                        else {}
                    )
                    if artwork_comp:
                        lines.append("<div class='k'>Artwork compare</div>")
                        lines.append(
                            f"<div class='v tiny'>{html.escape(_format_trace_value(artwork_comp))}</div>"
                        )
                    fingerprint = (
                        trace_payload.get("fingerprint")
                        if isinstance(trace_payload.get("fingerprint"), Mapping)
                        else {}
                    )
                    lines.append("<div class='k'>Fingerprint</div>")
                    lines.append(
                        f"<div class='v tiny'>{html.escape(_format_trace_value(fingerprint))}</div>"
                    )
                    bucketing = (
                        trace_payload.get("bucketing")
                        if isinstance(trace_payload.get("bucketing"), Mapping)
                        else {}
                    )
                    lines.append("<div class='k'>Bucketing</div>")
                    lines.append(
                        f"<div class='v tiny'>{html.escape(_format_trace_value(bucketing))}</div>"
                    )
                    lines.append("</div>")
                    lines.append("</details>")
                if not tracks:
                    lines.append("<div class='tiny muted'>No per-file trace data recorded.</div>")
                lines.append("</div>")
                lines.append("</details>")
                return lines

            metadata_actions = [act for act in actions if act.step == "metadata"]
            metadata_success = sum(1 for act in metadata_actions if act.status == "success")
            metadata_failed = sum(1 for act in metadata_actions if act.status == "failed")
            metadata_skipped = sum(1 for act in metadata_actions if act.status == "skipped")
            quarantined_count = sum(
                1 for dest in quarantine_index.values() if dest not in {"retained", "deleted"}
            )
            deleted_count = sum(1 for dest in quarantine_index.values() if dest == "deleted")
            groups_processed = len(group_lookup)
            winners_kept = groups_processed
            generated_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            host_name = platform.node() or "unknown-host"
            planned_count = rollup.get("planned", 0)
            attempted_count = rollup.get("attempted", 0)
            verified_count = rollup.get("verified", 0)
            skipped_count = rollup.get("skipped", 0)
            failed_count = rollup.get("failed", 0)
            execution_mode = "dry-run" if config.dry_run_execute else "real"
            resolved_library_root = os.path.abspath(config.library_root)
            resolved_quarantine_dir = os.path.abspath(quarantine_dir) if quarantine_dir else "n/a"

            def _status_badge_class(status: str) -> str:
                if status == "success":
                    return "ok"
                if status == "failed":
                    return "fail"
                return "warn"

            def _format_notes(detail: str, meta_notes: str) -> str:
                detail_html = html.escape(detail)
                if meta_notes:
                    meta_html = html.escape(meta_notes)
                    return f"{detail_html}<div class='tiny muted'>{meta_html}</div>"
                return detail_html

            path_counter = 0

            def _path_row(path_value: str) -> str:
                nonlocal path_counter
                path_counter += 1
                el_id = f"path-{path_counter}"
                safe = html.escape(path_value)
                return (
                    "<div class='path-row'>"
                    f"<span class='path path-truncate' id='{el_id}'>{safe}</span>"
                    f"<button class='copy tiny' data-copy-text='{safe}'>Copy</button>"
                    f"<button class='copy tiny' data-expand='{el_id}'>Expand</button>"
                    "</div>"
                )

            def _pairwise_stats(group) -> Dict[str, object]:
                thresholds = getattr(group, "grouping_thresholds", {}) or {}
                exact = float(thresholds.get("exact", 0.0))
                near = float(thresholds.get("near", exact))
                boost = float(thresholds.get("mixed_codec_boost", 0.0))
                distances = getattr(group, "fingerprint_distances", {}) or {}
                track_quality = getattr(group, "track_quality", {}) or {}
                exact_count = 0
                near_count = 0
                no_match_count = 0
                best_distance = None
                edges: List[Dict[str, object]] = []
                no_match_edges: List[Dict[str, object]] = []
                seen: set[tuple[str, str]] = set()
                for left, neighbors in distances.items():
                    for right, dist in neighbors.items():
                        key = tuple(sorted((left, right)))
                        if left == right or key in seen:
                            continue
                        seen.add(key)
                        left_lossless = bool(track_quality.get(left, {}).get("is_lossless"))
                        right_lossless = bool(track_quality.get(right, {}).get("is_lossless"))
                        effective_near = near + boost if left_lossless != right_lossless else near
                        verdict = "no match"
                        if dist <= exact:
                            verdict = "exact"
                            exact_count += 1
                        elif dist <= effective_near:
                            verdict = "near"
                            near_count += 1
                        else:
                            no_match_count += 1
                        if best_distance is None or dist < best_distance:
                            best_distance = dist
                        payload = {
                            "left": left,
                            "right": right,
                            "distance": dist,
                            "threshold": effective_near,
                            "verdict": verdict,
                        }
                        if verdict == "no match":
                            no_match_edges.append(payload)
                        else:
                            edges.append(payload)
                return {
                    "exact": exact_count,
                    "near": near_count,
                    "no_match": no_match_count,
                    "best": best_distance,
                    "edges": edges,
                    "no_match_edges": no_match_edges,
                    "exact_threshold": exact,
                    "near_threshold": near,
                    "mixed_boost": boost,
                }

            def _group_type_hint(group_actions: List[ExecutionAction]) -> str:
                if not group_actions:
                    return "unknown"
                if all(act.step == "metadata" for act in group_actions):
                    return "metadata-only"
                return "mixed"

            def _group_quarantine_count(group_actions: List[ExecutionAction]) -> int:
                return sum(
                    1
                    for act in group_actions
                    if act.step == "loser_cleanup"
                    and act.metadata.get("disposition") == "quarantine"
                )

            def _group_has_quarantine(group_actions: List[ExecutionAction]) -> bool:
                return _group_quarantine_count(group_actions) > 0

            def _group_has_failure(group_actions: List[ExecutionAction]) -> bool:
                return any(
                    act.status in {"failed", "blocked", "cancelled"} for act in group_actions
                )

            def _run_status() -> tuple[str, str, str]:
                status = "success" if success else "failed"
                status_class = "ok" if success else "fail"
                emoji = "✅" if success else "❌"
                return status, status_class, f"{emoji} {status}"

            settings_entries: List[tuple[str, object]] = []
            if plan and plan.threshold_settings:
                settings_entries.extend(
                    sorted(plan.threshold_settings.items(), key=lambda item: str(item[0]))
                )
            if plan and plan.fingerprint_settings:
                settings_entries.extend(
                    sorted(plan.fingerprint_settings.items(), key=lambda item: str(item[0]))
                )

            def _format_setting(value: object) -> str:
                if isinstance(value, float):
                    return f"{value:.3f}"
                return str(value)

            show_artwork_variants = config.show_artwork_variants
            group_pair_stats = {
                group.group_id: _pairwise_stats(group) for group in (plan.groups if plan else [])
            }
            total_exact = sum(stats["exact"] for stats in group_pair_stats.values())
            total_near = sum(stats["near"] for stats in group_pair_stats.values())
            art_threshold = float(
                (plan.threshold_settings if plan else {}).get(
                    "artwork_vastly_different_threshold", ARTWORK_VASTLY_DIFFERENT_THRESHOLD
                )
            )

            def _is_artwork_variant(group: object, loser_path: str, threshold: float) -> bool:
                artwork_hashes = getattr(group, "artwork_hashes", {}) or {}
                winner_path = getattr(group, "winner_path", "")
                winner_hash = artwork_hashes.get(winner_path)
                loser_hash = artwork_hashes.get(loser_path)
                if winner_hash is None or loser_hash is None:
                    return False
                return _hamming_distance(winner_hash, loser_hash) > threshold

            html_lines = [
                "<!doctype html>",
                "<html lang='en'>",
                "<head>",
                "<meta charset='utf-8' />",
                "<meta name='viewport' content='width=device-width,initial-scale=1' />",
                "<title>Execution Report</title>",
                "<style>",
                ":root{",
                "--bg:#ffffff;",
                "--text:#111;",
                "--muted:#666;",
                "--border:#e6e6e6;",
                "--card:#fafafa;",
                "--good:#1b7a1b;",
                "--bad:#a30000;",
                "--warn:#8a5b00;",
                "--chip:#f2f2f2;",
                "--mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;",
                "--sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, \"Apple Color Emoji\",\"Segoe UI Emoji\";",
                "}",
                "* { box-sizing: border-box; }",
                "body{",
                "margin: 18px;",
                "font-family: var(--sans);",
                "background: var(--bg);",
                "color: var(--text);",
                "line-height: 1.35;",
                "}",
                "header{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:12px 16px;",
                "align-items:baseline;",
                "justify-content:space-between;",
                "margin-bottom: 12px;",
                "}",
                "h1{",
                "font-size: 20px;",
                "margin: 0;",
                "letter-spacing: .2px;",
                "}",
                ".row{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:10px;",
                "align-items:center;",
                "}",
                ".card{",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "padding: 12px;",
                "background: var(--card);",
                "}",
                ".context-card{",
                "display:flex;",
                "gap:12px;",
                "align-items:flex-start;",
                "}",
                ".context-card-art{",
                "flex: 0 0 auto;",
                "}",
                ".context-card-details{",
                "flex: 1;",
                "}",
                ".card-stack{",
                "display:grid;",
                "gap:10px;",
                "}",
                ".summary{",
                "display:grid;",
                "grid-template-columns: 1fr;",
                "gap: 10px;",
                "margin-bottom: 14px;",
                "}",
                "@media (min-width: 900px){",
                ".summary{ grid-template-columns: 2fr 1fr; }",
                "}",
                ".kv{",
                "display:grid;",
                "grid-template-columns: 160px 1fr;",
                "gap:6px 10px;",
                "align-items:start;",
                "}",
                ".k{ color: var(--muted); font-size: 12px; }",
                ".v{ font-size: 13px; }",
                ".mono{ font-family: var(--mono); font-size: 12px; }",
                ".path{",
                "font-family: var(--mono);",
                "font-size: 12px;",
                "word-break: break-all;",
                "}",
                ".path-row{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:8px;",
                "align-items:center;",
                "}",
                ".path-truncate{",
                "overflow:hidden;",
                "text-overflow: ellipsis;",
                "white-space: nowrap;",
                "max-width: 60ch;",
                "}",
                ".path-expanded{",
                "white-space: normal;",
                "word-break: break-all;",
                "max-width: none;",
                "}",
                ".status{",
                "font-weight: 600;",
                "display:inline-flex;",
                "align-items:center;",
                "gap:6px;",
                "}",
                ".ok{ color: var(--good); }",
                ".fail{ color: var(--bad); }",
                ".warn{ color: var(--warn); }",
                ".pill{",
                "display:inline-flex;",
                "align-items:center;",
                "gap:8px;",
                "padding: 6px 10px;",
                "border-radius: 999px;",
                "background: var(--chip);",
                "border: 1px solid var(--border);",
                "font-size: 12px;",
                "white-space: nowrap;",
                "}",
                ".pill strong{ font-weight: 700; }",
                ".controls{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:10px;",
                "align-items:center;",
                "justify-content:space-between;",
                "margin: 14px 0 10px;",
                "}",
                ".controls .left, .controls .right{",
                "display:flex; flex-wrap:wrap; gap:10px; align-items:center;",
                "}",
                "input[type='search']{",
                "padding: 9px 10px;",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "width: min(520px, 92vw);",
                "font-size: 13px;",
                "background: #fff;",
                "}",
                "select{",
                "padding: 9px 10px;",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "font-size: 13px;",
                "background: #fff;",
                "}",
                "button.copy{",
                "border:1px solid var(--border);",
                "background:#fff;",
                "border-radius: 10px;",
                "padding: 8px 10px;",
                "font-size: 12px;",
                "cursor:pointer;",
                "}",
                "button.copy.tiny{",
                "padding: 3px 7px;",
                "font-size: 11px;",
                "border-radius: 8px;",
                "}",
                "button.copy:active{ transform: translateY(1px); }",
                ".settings-panel{",
                "margin-top: 10px;",
                "}",
                "details.group{",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "background: #fff;",
                "margin-bottom: 10px;",
                "overflow:hidden;",
                "}",
                "details.group[open]{ box-shadow: 0 1px 0 rgba(0,0,0,.03); }",
                "summary.group-summary{",
                "list-style:none;",
                "cursor:pointer;",
                "padding: 12px 12px;",
                "display:grid;",
                "grid-template-columns: auto 1fr;",
                "grid-template-rows: auto auto;",
                "column-gap: 12px;",
                "row-gap: 6px;",
                "background: #fff;",
                "}",
                "summary.group-summary::-webkit-details-marker{ display:none; }",
                ".album-art{",
                "width: 42px;",
                "height: 42px;",
                "display: inline-block;",
                "grid-column: 1;",
                "grid-row: 1 / span 2;",
                "align-self: center;",
                "border-radius: 8px;",
                "border: 1px solid var(--border);",
                "position: relative;",
                "overflow: hidden;",
                "background: #f7f7f7;",
                "}",
                ".album-art.album-art-thumb{",
                "width: 64px;",
                "height: 64px;",
                "border-radius: 6px;",
                "}",
                ".album-art::after{",
                "content: \"\";",
                "position: absolute;",
                "inset: 0;",
                "pointer-events: none;",
                "box-shadow: inset 0 0 0 1px rgba(0,0,0,.06), inset 0 0 10px rgba(0,0,0,.20);",
                "}",
                ".album-art img{",
                "width: 100%;",
                "height: 100%;",
                "object-fit: cover;",
                "display: block;",
                "}",
                ".group-top{",
                "grid-column: 2;",
                "grid-row: 1;",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:10px 12px;",
                "align-items:center;",
                "justify-content:space-between;",
                "}",
                "summary.group-summary .tiny{",
                "grid-column: 2;",
                "grid-row: 2;",
                "}",
                ".group-title{",
                "display:flex;",
                "align-items:baseline;",
                "gap:10px;",
                "min-width: 240px;",
                "}",
                ".gid{",
                "font-family: var(--mono);",
                "font-size: 12px;",
                "color: var(--muted);",
                "}",
                ".winner-short{",
                "font-weight: 650;",
                "font-size: 13px;",
                "max-width: 74ch;",
                "overflow:hidden;",
                "text-overflow: ellipsis;",
                "white-space: nowrap;",
                "}",
                ".chips{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:8px;",
                "justify-content:flex-end;",
                "}",
                "button.trace-btn{",
                "border:1px solid var(--border);",
                "background:#fff;",
                "border-radius: 999px;",
                "padding: 3px 10px;",
                "font-size: 12px;",
                "font-weight: 650;",
                "cursor:pointer;",
                "white-space: nowrap;",
                "}",
                "button.trace-btn:hover{ background:#f7f7f7; }",
                ".group-body{",
                "border-top: 1px solid var(--border);",
                "background: var(--card);",
                "padding: 12px;",
                "display:grid;",
                "grid-template-columns: 1fr;",
                "gap: 10px;",
                "}",
                "@media (min-width: 980px){",
                ".group-body{ grid-template-columns: 1fr 1.2fr; }",
                "}",
                ".section h3{",
                "font-size: 12px;",
                "margin: 0 0 8px;",
                "color: var(--muted);",
                "text-transform: uppercase;",
                "letter-spacing: .08em;",
                "}",
                "table.ops{",
                "width:100%;",
                "border-collapse: collapse;",
                "background:#fff;",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "overflow:hidden;",
                "}",
                "table.ops th, table.ops td{",
                "border-bottom:1px solid var(--border);",
                "padding: 9px 10px;",
                "font-size: 12px;",
                "vertical-align: top;",
                "}",
                "table.ops th{",
                "text-align:left;",
                "color: var(--muted);",
                "background: #fcfcfc;",
                "font-weight: 650;",
                "}",
                "table.ops tr:last-child td{ border-bottom: none; }",
                ".op-key{",
                "font-family: var(--mono);",
                "font-size: 12px;",
                "color:#222;",
                "white-space:nowrap;",
                "}",
                ".note{",
                "color: var(--muted);",
                "}",
                ".badge{",
                "display:inline-flex;",
                "align-items:center;",
                "padding: 3px 8px;",
                "border-radius: 999px;",
                "border:1px solid var(--border);",
                "font-size: 12px;",
                "font-weight: 650;",
                "background: #fff;",
                "white-space: nowrap;",
                "}",
                ".badge.ok{ border-color: rgba(27,122,27,.25); }",
                ".badge.fail{ border-color: rgba(163,0,0,.25); }",
                ".badge.keep{ border-color: rgba(27,122,27,.35); background:#f2fff2; }",
                "details.bucket{",
                "border:1px dashed var(--border);",
                "border-radius: 10px;",
                "background:#fff;",
                "padding: 10px;",
                "}",
                "details.trace{",
                "border:1px dashed var(--border);",
                "border-radius: 10px;",
                "background:#fff;",
                "padding: 10px;",
                "margin-top: 8px;",
                "}",
                "summary.trace-summary{",
                "cursor:pointer;",
                "list-style:none;",
                "font-weight: 650;",
                "font-size: 12px;",
                "}",
                "summary.trace-summary::-webkit-details-marker{ display:none; }",
                ".trace-grid{",
                "display:grid;",
                "grid-template-columns: 180px 1fr;",
                "gap:6px 10px;",
                "font-size: 12px;",
                "margin-top: 8px;",
                "}",
                ".trace-block{ margin-top: 8px; }",
                "details.trace-track{",
                "border:1px solid var(--border);",
                "border-radius: 8px;",
                "padding: 8px;",
                "background:#fafafa;",
                "margin-top: 6px;",
                "}",
                "summary.trace-track-summary{",
                "cursor:pointer;",
                "list-style:none;",
                "font-weight: 600;",
                "font-size: 12px;",
                "}",
                "summary.trace-track-summary::-webkit-details-marker{ display:none; }",
                "summary.bucket-summary{",
                "cursor:pointer;",
                "list-style:none;",
                "font-weight: 650;",
                "font-size: 12px;",
                "}",
                "summary.bucket-summary::-webkit-details-marker{ display:none; }",
                ".bucket-grid{",
                "display:grid;",
                "grid-template-columns: 150px 1fr;",
                "gap:6px 10px;",
                "font-size: 12px;",
                "margin-top: 8px;",
                "}",
                ".edge-list{",
                "margin: 6px 0 0 18px;",
                "padding: 0;",
                "font-size: 12px;",
                "}",
                ".edge-list li{ margin: 4px 0; }",
                "details.quarantine{",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "background:#fff;",
                "margin-top: 14px;",
                "overflow:hidden;",
                "}",
                "summary.quarantine-summary{",
                "cursor:pointer;",
                "padding: 12px;",
                "list-style:none;",
                "}",
                "summary.quarantine-summary::-webkit-details-marker{ display:none; }",
                "ul.q-list{",
                "margin:0;",
                "padding: 0 12px 12px 28px;",
                "color:#222;",
                "}",
                "ul.q-list li{",
                "font-family: var(--mono);",
                "font-size: 12px;",
                "word-break: break-all;",
                "margin: 6px 0;",
                "}",
                ".muted{ color: var(--muted); }",
                ".tiny{ font-size: 12px; }",
                ".hidden{ display:none !important; }",
                "</style>",
                "</head>",
                "<body>",
                "<header>",
                "<h1>Execution Report</h1>",
                f"<div class='status {_run_status()[1]}' id='runStatus' data-status='{_run_status()[0]}'>"
                f"{_run_status()[2]}</div>",
                "</header>",
                "<section class='summary'>",
                "<div class='card'>",
                "<div class='row' style='justify-content:space-between;'>",
                "<div class='row' style='gap:8px;'>",
                "<span class='pill'><strong>Groups</strong> <span id='statGroups'>0</span></span>",
                "<span class='pill'><strong>Winners</strong> <span id='statWinners'>0</span></span>",
                "<span class='pill'><strong>Quarantined</strong> <span id='statQuarantined'>0</span></span>",
                "<span class='pill'><strong>Ops</strong> <span id='statOpsOk'>0</span>/<span id='statOpsTotal'>0</span> ok</span>",
                "</div>",
                "<div class='row'>",
                "<button class='copy' data-copy='#planSignature'>Copy plan signature</button>",
                "<button class='copy' data-copy='#reportsDir'>Copy reports dir</button>",
                "</div>",
                "</div>",
                "<div class='row' style='gap:8px; margin-top:6px;'>",
                f"<span class='pill'><strong>Planned</strong> {planned_count}</span>",
                f"<span class='pill'><strong>Attempted</strong> {attempted_count}</span>",
                f"<span class='pill'><strong>Verified</strong> {verified_count}</span>",
                f"<span class='pill'><strong>Skipped</strong> {skipped_count}</span>",
                f"<span class='pill'><strong>Failed</strong> {failed_count}</span>",
                "</div>",
                "<div class='row' style='gap:8px; margin-top:6px;'>",
                f"<span class='pill'><strong>Exact</strong> <span id='statExact'>{total_exact}</span></span>",
                f"<span class='pill'><strong>Near</strong> <span id='statNear'>{total_near}</span></span>",
                "</div>",
                "<div style='height:10px'></div>",
                "<div class='kv'>",
                "<div class='k'>Plan signature</div>",
                f"<div class='v mono' id='planSignature'>{html.escape(plan_signature or computed_signature)}</div>",
                "<div class='k'>Reports directory</div>",
                f"<div class='v path' id='reportsDir'>{html.escape(reports_dir)}</div>",
                "<div class='k'>Execution mode</div>",
                f"<div class='v tiny'>{html.escape(execution_mode)}</div>",
                "<div class='k'>Library root</div>",
                f"<div class='v path'>{html.escape(resolved_library_root)}</div>",
                "<div class='k'>Quarantine dir</div>",
                f"<div class='v path'>{html.escape(resolved_quarantine_dir)}</div>",
                "</div>",
                "</div>",
                "<div class='card'>",
                "<div class='kv'>",
                "<div class='k'>Generated</div>",
                f"<div class='v tiny'>{html.escape(generated_at_iso)}</div>",
                "<div class='k'>Host</div>",
                f"<div class='v tiny'>{html.escape(host_name)}</div>",
                "<div class='k'>Notes</div>",
                "<div class='v tiny muted'>",
                "Groups are collapsed by default. Full paths and verbose messages are in “Context”.",
                "</div>",
                "</div>",
                "</div>",
                "</section>",
                "<section class='controls'>",
                "<div class='left'>",
                "<input id='search' type='search' placeholder='Search groups, winner filename, paths, notes…' />",
                "<select id='filter'>",
                "<option value='all'>All groups</option>",
                "<option value='has-quarantine'>Has quarantine</option>",
                "<option value='metadata-only'>Metadata only</option>",
                "<option value='failed'>Any failures</option>",
                "</select>",
                "</div>",
                "<div class='right'>",
                "<span class='tiny muted' id='visibleCount'>0 visible</span>",
                "<button class='copy' id='expandAll'>Expand all</button>",
                "<button class='copy' id='collapseAll'>Collapse all</button>",
                "</div>",
                "</section>",
                "<main id='groups'>",
            ]

            if settings_entries:
                insert_at = html_lines.index(
                    "<button class='copy' data-copy='#reportsDir'>Copy reports dir</button>"
                ) + 1
                html_lines.insert(insert_at, "<button class='copy' id='toggleSettings'>Show settings</button>")
                panel = [
                    "<div class='card hidden settings-panel' id='settingsPanel'>",
                    "<div class='kv'>",
                    "<div class='k'>Active settings</div>",
                    "<div class='v tiny muted'>Thresholds + fingerprint inputs used for this plan.</div>",
                ]
                for key, value in settings_entries:
                    panel.append(f"<div class='k'>{html.escape(str(key))}</div>")
                    panel.append(f"<div class='v tiny'>{html.escape(_format_setting(value))}</div>")
                panel.extend(["</div>", "</div>"])
                section_end = html_lines.index("</section>")
                html_lines[section_end:section_end] = panel

            for gid in sorted([g for g in actions_by_group.keys() if g != "__general__"]):
                grp = group_lookup.get(gid)
                winner_path = getattr(grp, "winner_path", "Unknown winner")
                group_actions = actions_by_group.get(gid, [])
                visible_losers = list(getattr(grp, "losers", []) or [])
                if not show_artwork_variants and grp:
                    hidden_losers = {
                        loser
                        for loser in getattr(grp, "losers", [])
                        if _is_artwork_variant(grp, loser, art_threshold)
                    }
                    if hidden_losers:
                        visible_losers = [
                            loser for loser in getattr(grp, "losers", []) if loser not in hidden_losers
                        ]
                        group_actions = [
                            act
                            for act in group_actions
                            if not (act.step == "loser_cleanup" and act.target in hidden_losers)
                        ]
                if not show_artwork_variants and not group_actions and not visible_losers:
                    continue
                state = _group_state(group_actions)
                badges = _action_badges(group_actions)
                group_quarantine_count = _group_quarantine_count(group_actions)
                group_type_hint = _group_type_hint(group_actions)
                group_has_quarantine = _group_has_quarantine(group_actions)
                group_has_failure = _group_has_failure(group_actions)
                search_tokens = [gid, winner_path, _basename(winner_path), state]
                search_tokens.extend(badges)
                for act in group_actions:
                    search_tokens.extend([act.step, act.target, act.status, act.detail])
                    meta_notes = _format_metadata_notes(act.metadata)
                    if meta_notes:
                        search_tokens.append(meta_notes)
                search_text = " ".join(str(token) for token in search_tokens if token)
                summary_line = (
                    f"Status: {state}. " + "; ".join(badges) if badges else f"Status: {state}."
                )
                group_paths = {winner_path, *visible_losers}
                bucket_stats = group_pair_stats.get(gid, {})
                best_distance = bucket_stats.get("best")
                best_distance_label = (
                    f"{best_distance:.4f}" if isinstance(best_distance, float) else "n/a"
                )
                exact_count = int(bucket_stats.get("exact", 0) or 0)
                near_count = int(bucket_stats.get("near", 0) or 0)
                no_match_count = int(bucket_stats.get("no_match", 0) or 0)
                html_lines.append(
                    "<details class='group' "
                    f"data-group-id='{html.escape(gid)}' "
                    f"data-has-quarantine='{str(group_has_quarantine).lower()}' "
                    f"data-has-failure='{str(group_has_failure).lower()}' "
                    f"data-type='{html.escape(group_type_hint)}' "
                    f"data-search='{html.escape(search_text.lower())}' "
                    f"data-quarantine-count='{group_quarantine_count}'>"
                )
                album_art_src = _album_art_src(grp, winner_path)
                html_lines.append("<summary class='group-summary'>")
                html_lines.append("<span class='album-art' title='Album Art'>")
                if album_art_src:
                    html_lines.append(f"<img src='{album_art_src}' alt='' />")
                html_lines.append("</span>")
                html_lines.append("<div class='group-top'>")
                html_lines.append("<div class='group-title'>")
                html_lines.append(f"<span class='gid'>Group {html.escape(gid)}</span>")
                html_lines.append(
                    "<span class='winner-short' "
                    f"title='{html.escape(winner_path)}'>{html.escape(_basename(winner_path))}</span>"
                )
                html_lines.append("</div>")
                html_lines.append("<div class='chips'>")
                html_lines.append(
                    f"<span class='badge {_status_badge_class(state)}' data-status='{html.escape(state)}'>"
                    f"{html.escape(state)}</span>"
                )
                for badge in badges:
                    html_lines.append(f"<span class='pill'>{html.escape(badge)}</span>")
                trace_target = f"trace-{gid}"
                html_lines.append(
                    f"<button class='trace-btn' type='button' data-trace-target='{html.escape(trace_target)}'>Trace</button>"
                )
                html_lines.append("</div>")
                html_lines.append("</div>")
                html_lines.append(f"<div class='tiny muted'>{html.escape(summary_line)}</div>")
                html_lines.append("</summary>")
                html_lines.append("<div class='group-body'>")
                html_lines.append("<div class='section'>")
                html_lines.append("<h3>Context</h3>")
                cleanup_actions = [act for act in group_actions if act.step == "loser_cleanup"]
                winner_art_src = _album_art_src(grp, winner_path)
                winner_art_hash = getattr(grp, "artwork_hashes", {}).get(winner_path)
                html_lines.append("<div class='card-stack'>")
                html_lines.append("<div class='card context-card' style='background:#fff;'>")
                html_lines.append("<div class='context-card-art'>")
                html_lines.append("<span class='album-art album-art-thumb' title='Album Art'>")
                if winner_art_src:
                    html_lines.append(f"<img src='{winner_art_src}' alt='' />")
                html_lines.append("</span>")
                html_lines.append("</div>")
                html_lines.append("<div class='kv context-card-details'>")
                html_lines.append("<div class='k'>Kept file (winner)</div>")
                html_lines.append(f"<div class='v'>{_path_row(winner_path)}</div>")
                html_lines.append("<div class='k'>Group ID</div>")
                html_lines.append(f"<div class='v mono'>{html.escape(gid)}</div>")
                html_lines.append("<div class='k'>Metadata bucket key</div>")
                html_lines.append(
                    f"<div class='v mono'>{html.escape(str(getattr(grp, 'grouping_metadata_key', '—')))}</div>"
                )
                html_lines.append("<div class='k'>Audio identity</div>")
                html_lines.append(
                    "<div class='v tiny'>"
                    f"{html.escape(str(getattr(grp, 'group_match_type', '')))} · best {html.escape(best_distance_label)} "
                    f"(exact {exact_count} · near {near_count} · no-match {no_match_count})"
                    "</div>"
                )
                html_lines.append("<div class='k'>Artwork variant</div>")
                html_lines.append(
                    f"<div class='v tiny'>{html.escape(str(getattr(grp, 'artwork_variant_label', '')))}</div>"
                )
                if getattr(grp, "artwork_variant_total", 1) > 1:
                    html_lines.append("<div class='k'>Artwork variants</div>")
                    html_lines.append(
                        "<div class='v tiny muted'>Audio-identical but different artwork variants preserved.</div>"
                    )
                unknown_reasons = getattr(grp, "artwork_unknown_reasons", {}) or {}
                if isinstance(unknown_reasons, Mapping) and unknown_reasons:
                    missing_notes = ", ".join(
                        f"{_basename(path)} ({reason})"
                        for path, reason in sorted(unknown_reasons.items(), key=lambda item: str(item[0]).lower())
                    )
                    html_lines.append("<div class='k'>Artwork missing/unreadable</div>")
                    html_lines.append(f"<div class='v tiny'>{html.escape(missing_notes)}</div>")
                html_lines.append("<div class='k'>Debug</div>")
                html_lines.append(
                    "<div class='v tiny muted'>Shown for traceability; safe to ignore for normal review.</div>"
                )
                html_lines.append("</div>")
                html_lines.append("</div>")
                if cleanup_actions:
                    loser_entries = cleanup_actions
                elif show_artwork_variants or visible_losers:
                    loser_entries = [None]
                else:
                    loser_entries = []
                for loser_action in loser_entries:
                    loser_path = loser_action.target if loser_action else ""
                    loser_disposition = ""
                    loser_reason = ""
                    if loser_action:
                        raw_disposition = loser_action.metadata.get("disposition")
                        loser_disposition = str(raw_disposition) if raw_disposition else ""
                        loser_reason = loser_action.detail or ""
                    art_distance = None
                    is_art_variant = False
                    if loser_path:
                        loser_hash = getattr(grp, "artwork_hashes", {}).get(loser_path)
                        if winner_art_hash is not None and loser_hash is not None:
                            art_distance = _hamming_distance(winner_art_hash, loser_hash)
                            is_art_variant = art_distance > art_threshold
                    loser_art_src = (
                        _album_art_src(grp, loser_path, include_group_chosen=False)
                        if loser_path
                        else None
                    )
                    html_lines.append("<div class='card context-card' style='background:#fff;'>")
                    html_lines.append("<div class='context-card-art'>")
                    html_lines.append("<span class='album-art album-art-thumb' title='Album Art'>")
                    if loser_art_src:
                        html_lines.append(f"<img src='{loser_art_src}' alt='' />")
                    html_lines.append("</span>")
                    html_lines.append("</div>")
                    html_lines.append("<div class='kv context-card-details'>")
                    html_lines.append("<div class='k'>Loser file</div>")
                    if loser_path:
                        html_lines.append(f"<div class='v'>{_path_row(loser_path)}</div>")
                    else:
                        html_lines.append("<div class='v path'>—</div>")
                    html_lines.append("<div class='k'>Disposition</div>")
                    if loser_disposition:
                        if is_art_variant:
                            html_lines.append(
                                "<div class='v'>Preserve (different art) "
                                "<span class='badge keep'>different art → keep</span></div>"
                            )
                        else:
                            html_lines.append(
                                f"<div class='v'>{html.escape(loser_disposition)}</div>"
                            )
                    else:
                        html_lines.append("<div class='v'>—</div>")
                    if loser_reason:
                        html_lines.append("<div class='k'>Reason</div>")
                        html_lines.append(
                            f"<div class='v tiny muted'>{html.escape(loser_reason)}</div>"
                        )
                    if art_distance is not None:
                        html_lines.append("<div class='k'>Artwork distance</div>")
                        html_lines.append(
                            "<div class='v tiny'>"
                            "<details class='tiny'><summary>details</summary>"
                            f"<div class='muted'>Winner ↔ loser distance: {art_distance} "
                            f"(threshold {art_threshold:.0f})</div>"
                            "</details></div>"
                        )
                    html_lines.append("</div>")
                    html_lines.append("</div>")
                html_lines.append("</div>")
                bucket_diag = getattr(grp, "bucket_diagnostics", {}) or {}
                bucket_sources = (
                    bucket_diag.get("sources")
                    if isinstance(bucket_diag.get("sources"), Mapping)
                    else {}
                )
                bucket_size = len(bucket_sources) if bucket_sources else len(group_paths)
                bucket_id = bucket_diag.get("bucket_id", "n/a")
                has_metadata = bool(bucket_diag.get("metadata_seeded")) or (
                    isinstance(bucket_sources, Mapping)
                    and any(source == "metadata" for source in bucket_sources.values())
                )
                if has_metadata:
                    formation = "metadata-seeded"
                else:
                    formation = "fallback"
                edges = bucket_stats.get("edges", []) if isinstance(bucket_stats.get("edges"), list) else []
                no_match_edges = (
                    bucket_stats.get("no_match_edges", [])
                    if isinstance(bucket_stats.get("no_match_edges"), list)
                    else []
                )
                html_lines.append("<details class='bucket'>")
                html_lines.append("<summary class='bucket-summary'>Bucket diagnostics</summary>")
                html_lines.append("<div class='bucket-grid'>")
                html_lines.append("<div class='k'>Bucket ID</div>")
                html_lines.append(f"<div class='v mono'>{html.escape(str(bucket_id))}</div>")
                html_lines.append("<div class='k'>Bucket size</div>")
                html_lines.append(f"<div class='v'>{bucket_size}</div>")
                html_lines.append("<div class='k'>Formation</div>")
                html_lines.append(f"<div class='v'>{html.escape(formation)}</div>")
                html_lines.append("<div class='k'>Exact / near / no-match</div>")
                html_lines.append(f"<div class='v'>{exact_count} / {near_count} / {no_match_count}</div>")
                html_lines.append("<div class='k'>Best distance</div>")
                html_lines.append(f"<div class='v'>{best_distance_label}</div>")
                html_lines.append("<div class='k'>Thresholds</div>")
                html_lines.append(
                    f"<div class='v tiny'>exact {bucket_stats.get('exact_threshold', 0.0):.4f} · "
                    f"near {bucket_stats.get('near_threshold', 0.0):.4f} · "
                    f"mixed boost +{bucket_stats.get('mixed_boost', 0.0):.4f}</div>"
                )
                html_lines.append("</div>")
                html_lines.append("<div class='tiny muted' style='margin-top:6px;'>Match edges</div>")
                if edges:
                    html_lines.append("<ul class='edge-list'>")
                    for edge in sorted(edges, key=lambda e: float(e.get("distance", 0.0))):
                        left = str(edge.get("left"))
                        right = str(edge.get("right"))
                        distance = float(edge.get("distance", 0.0))
                        verdict = str(edge.get("verdict"))
                        html_lines.append(
                            "<li>"
                            f"{html.escape(_basename(left))} ↔ {html.escape(_basename(right))} "
                            f"({distance:.4f}, {html.escape(verdict)})"
                            "</li>"
                        )
                    html_lines.append("</ul>")
                else:
                    html_lines.append("<div class='tiny muted'>No exact/near edges in this group.</div>")
                html_lines.append("<details class='tiny' style='margin-top:6px;'>")
                html_lines.append(f"<summary>No-match edges ({len(no_match_edges)})</summary>")
                if no_match_edges:
                    html_lines.append("<ul class='edge-list'>")
                    for edge in sorted(no_match_edges, key=lambda e: float(e.get("distance", 0.0))):
                        left = str(edge.get("left"))
                        right = str(edge.get("right"))
                        distance = float(edge.get("distance", 0.0))
                        html_lines.append(
                            "<li>"
                            f"{html.escape(_basename(left))} ↔ {html.escape(_basename(right))} "
                            f"({distance:.4f})"
                            "</li>"
                        )
                    html_lines.append("</ul>")
                else:
                    html_lines.append("<div class='tiny muted'>No no-match edges.</div>")
                html_lines.append("</details>")
                html_lines.append("</details>")
                html_lines.extend(_trace_panel(grp))
                html_lines.append("</div>")
                html_lines.append("<div class='section'>")
                html_lines.append("<h3>Operations</h3>")
                html_lines.append("<table class='ops'>")
                html_lines.append("<thead>")
                html_lines.append("<tr>")
                html_lines.append("<th style='width: 160px;'>Operation</th>")
                html_lines.append("<th>Target</th>")
                html_lines.append("<th style='width: 110px;'>Status</th>")
                html_lines.append("<th>Notes</th>")
                html_lines.append("</tr>")
                html_lines.append("</thead>")
                html_lines.append("<tbody>")
                for act in group_actions:
                    meta_notes = _format_metadata_notes(act.metadata)
                    notes = _format_notes(act.detail, meta_notes)
                    status_class = _status_badge_class(act.status)
                    html_lines.append(
                        "<tr>"
                        f"<td class='op-key'>{html.escape(act.step)}</td>"
                        f"<td class='path' title='{html.escape(act.target)}'>"
                        f"{html.escape(_basename(act.target))}</td>"
                        f"<td><span class='badge {status_class}'>{html.escape(act.status)}</span></td>"
                        f"<td class='note'>{notes}</td>"
                        "</tr>"
                    )
                html_lines.append("</tbody>")
                html_lines.append("</table>")
                html_lines.append("</div>")
                html_lines.append("</div>")
                html_lines.append("</details>")

            html_lines.append("</main>")
            html_lines.append("<details class='quarantine' id='quarantineIndex'>")
            html_lines.append("<summary class='quarantine-summary'>")
            html_lines.append("<div class='row' style='justify-content:space-between;'>")
            html_lines.append(
                "<div><strong>Quarantined Files</strong> <span class='muted'>(<span id='qCount'>0</span>)</span></div>"
            )
            html_lines.append("<div class='muted tiny'>Collapsed by default</div>")
            html_lines.append("</div>")
            html_lines.append("</summary>")
            html_lines.append("<ul class='q-list' id='qList'>")
            for loser, dest in quarantine_index.items():
                html_lines.append(
                    f"<li>{html.escape(loser)} → {html.escape(dest)}</li>"
                )
            html_lines.append("</ul>")
            html_lines.append("</details>")

            missing_sections: List[str] = []
            missing_sections.append(
                f"Metadata operations: {metadata_success} ok / {metadata_failed} failed / {metadata_skipped} skipped"
            )
            if deleted_count:
                missing_sections.append(
                    f"Files quarantined: {quarantined_count} (deleted: {deleted_count})"
                )
            else:
                missing_sections.append(f"Files quarantined: {quarantined_count}")

            general_actions = actions_by_group.get("__general__", [])
            playlist_results = playlist_results or []
            if general_actions or playlist_results:
                missing_sections.append("")
            if general_actions:
                missing_sections.append("General Actions")
                missing_sections.append("<table class='ops'>")
                missing_sections.append(
                    "<thead><tr><th>Operation</th><th>Target</th><th>Status</th><th>Notes</th></tr></thead>"
                )
                missing_sections.append("<tbody>")
                for act in general_actions:
                    meta_notes = _format_metadata_notes(act.metadata)
                    notes = _format_notes(act.detail, meta_notes)
                    status_class = _status_badge_class(act.status)
                    missing_sections.append(
                        "<tr>"
                        f"<td class='op-key'>{html.escape(act.step)}</td>"
                        f"<td class='path' title='{html.escape(act.target)}'>"
                        f"{html.escape(_basename(act.target))}</td>"
                        f"<td><span class='badge {status_class}'>{html.escape(act.status)}</span></td>"
                        f"<td class='note'>{notes}</td>"
                        "</tr>"
                    )
                missing_sections.append("</tbody></table>")
            if playlist_results:
                missing_sections.append("Playlist Changes")
                missing_sections.append("<table class='ops'>")
                missing_sections.append(
                    "<thead><tr><th>Playlist</th><th>Status</th><th>Details</th></tr></thead>"
                )
                missing_sections.append("<tbody>")
                for res in playlist_results:
                    label = res.playlist
                    if res.original_playlist:
                        label = f"{res.original_playlist} (validated copy: {res.playlist})"
                    status_class = _status_badge_class(res.status)
                    missing_sections.append(
                        "<tr>"
                        f"<td class='path'>{html.escape(label)}</td>"
                        f"<td><span class='badge {status_class}'>{html.escape(res.status)}</span></td>"
                        f"<td class='note'>replaced {res.replaced_entries}; backup: {html.escape(res.backup_path)}</td>"
                        "</tr>"
                    )
                missing_sections.append("</tbody></table>")

            if missing_sections:
                html_lines.append(
                    "<div style='margin-top:14px;'>"
                    "<strong>NEED TO INCORPORATE AT LATER DATE</strong>"
                    "</div>"
                )
                for section in missing_sections:
                    if not section:
                        html_lines.append("<div style='height:10px'></div>")
                        continue
                    html_lines.append(f"<div class='tiny'>{section}</div>")
            html_lines.append(
                "<script>"
                "(function(){"
                "const $ = (s, r=document) => r.querySelector(s);"
                "const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));"
                "const groupsEl = $('#groups');"
                "const groups = () => $$('#groups details.group');"
                "const searchEl = $('#search');"
                "const filterEl = $('#filter');"
                "const visibleCountEl = $('#visibleCount');"
                "const expandAllBtn = $('#expandAll');"
                "const collapseAllBtn = $('#collapseAll');"
                "function computeStats(){"
                "const g = groups();"
                "$('#statGroups').textContent = g.length.toString();"
                "$('#statWinners').textContent = g.length.toString();"
                "let qTotal = 0;"
                "for (const d of g){"
                "const chip = d.querySelector('.pill');"
                "const attr = d.getAttribute('data-quarantine-count');"
                "if (attr) qTotal += Number(attr) || 0;"
                "}"
                "$('#statQuarantined').textContent = qTotal.toString();"
                "let opsTotal = 0, opsOk = 0;"
                "for (const d of g){"
                "const rows = $$('table.ops tbody tr', d);"
                "opsTotal += rows.length;"
                "for (const r of rows){"
                "const badge = r.querySelector('.badge');"
                "if (badge && badge.textContent.trim().toLowerCase() === 'success') opsOk += 1;"
                "}"
                "}"
                "$('#statOpsTotal').textContent = opsTotal.toString();"
                "$('#statOpsOk').textContent = opsOk.toString();"
                "const qItems = $$('#qList li').filter(li => li.textContent.trim().length > 0);"
                "$('#qCount').textContent = qItems.length.toString();"
                "}"
                "function applyFilters(){"
                "const term = (searchEl.value || '').trim().toLowerCase();"
                "const mode = filterEl.value;"
                "let visible = 0;"
                "for (const d of groups()){"
                "const hay = (d.getAttribute('data-search') || '').toLowerCase();"
                "const matchesSearch = !term || hay.includes(term);"
                "const hasQuarantine = (d.getAttribute('data-has-quarantine') || 'false') === 'true';"
                "const hasFailure = (d.getAttribute('data-has-failure') || 'false') === 'true';"
                "const typeHint = (d.getAttribute('data-type') || '').toLowerCase();"
                "let matchesFilter = true;"
                "if (mode === 'has-quarantine') matchesFilter = hasQuarantine;"
                "if (mode === 'metadata-only') matchesFilter = typeHint === 'metadata-only';"
                "if (mode === 'failed') matchesFilter = hasFailure;"
                "const show = matchesSearch && matchesFilter;"
                "d.classList.toggle('hidden', !show);"
                "if (show) visible += 1;"
                "}"
                "visibleCountEl.textContent = `${visible} visible`;"
                "}"
                "for (const btn of $$('button.copy[data-copy], button.copy[data-copy-text]')){"
                "btn.addEventListener('click', async () => {"
                "const explicit = btn.getAttribute('data-copy-text');"
                "const sel = btn.getAttribute('data-copy');"
                "const el = sel ? $(sel) : null;"
                "const text = explicit || (el ? el.textContent.trim() : '');"
                "if (!text) return;"
                "try{"
                "await navigator.clipboard.writeText(text);"
                "const old = btn.textContent;"
                "btn.textContent = 'Copied';"
                "setTimeout(()=>btn.textContent=old, 900);"
                "}catch(e){"
                "const ta = document.createElement('textarea');"
                "ta.value = text;"
                "document.body.appendChild(ta);"
                "ta.select();"
                "document.execCommand('copy');"
                "ta.remove();"
                "}"
                "});"
                "}"
                "for (const btn of $$('button.copy[data-expand]')){"
                "btn.addEventListener('click', () => {"
                "const id = btn.getAttribute('data-expand');"
                "const el = id ? document.getElementById(id) : null;"
                "if (!el) return;"
                "const expanded = el.classList.toggle('path-expanded');"
                "btn.textContent = expanded ? 'Collapse' : 'Expand';"
                "});"
                "}"
                "for (const btn of $$('button.trace-btn[data-trace-target]')){"
                "btn.addEventListener('click', (event) => {"
                "event.preventDefault();"
                "const id = btn.getAttribute('data-trace-target');"
                "const panel = id ? document.getElementById(id) : null;"
                "if (!panel) return;"
                "panel.open = !panel.open;"
                "panel.scrollIntoView({behavior: 'smooth', block: 'start'});"
                "});"
                "}"
                "expandAllBtn?.addEventListener('click', () => groups().forEach(d => d.open = true));"
                "collapseAllBtn?.addEventListener('click', () => groups().forEach(d => d.open = false));"
                "const settingsBtn = $('#toggleSettings');"
                "const settingsPanel = $('#settingsPanel');"
                "if (settingsBtn && settingsPanel){"
                "settingsBtn.addEventListener('click', () => {"
                "const hidden = settingsPanel.classList.toggle('hidden');"
                "settingsBtn.textContent = hidden ? 'Show settings' : 'Hide settings';"
                "});"
                "}"
                "searchEl?.addEventListener('input', applyFilters);"
                "filterEl?.addEventListener('change', applyFilters);"
                "computeStats();"
                "applyFilters();"
                "})();"
                "</script>"
            )
            html_lines.append("</body></html>")
            _atomic_write_text(html_report_path, "\n".join(html_lines))
            if not os.path.exists(ensure_long_path(html_report_path)):
                log(f"Report write failed: HTML report not found at {html_report_path}")
        except Exception as exc:
            log(f"Report write failed: unable to create HTML report at {html_report_path} ({exc})")

    report_paths = {
        "audit": consolidated_report_path,
        "json_report": consolidated_report_path,
        "html_report": html_report_path,
        "reports_root": reports_dir,
        "backups_root": backups_dir,
    }

    return ExecutionResult(
        success=success,
        actions=actions,
        playlist_reports=playlist_results,
        quarantine_index=quarantine_index,
        report_paths=report_paths,
    )
