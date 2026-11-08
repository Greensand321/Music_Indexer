#!/usr/bin/env python3
"""
update_genres.py

Walk a chosen folder and all subfolders for audio files. For each file, read
its “artist” and “title” tags, query MusicBrainz for a matching recording, pull
the MB recording’s tags (genres), pick the top 3 by popularity, and add them
to the file’s own “genre” tag (using Mutagen).

Rules, shared with the Qt app's "Fill from MusicBrainz" tab:

* files that already carry two or more genre values are left untouched;
* otherwise the MusicBrainz tags are **added after** the existing genre —
  nothing already on the file is removed (see
  ``controllers.normalize_controller.add_genres``).

Preview first, in the app and here: every file is looked up first
(``normalize_controller.plan_genre_fill`` — nothing is written), the planned
changes are listed, and only after the user confirms are exactly those files
written (``apply_genre_changes``, which skips a file whose genres changed
since the preview). The command line used to write each file the moment its
lookup returned.

Run ``python update_genres.py [folder] [--yes]``; without a folder a folder
picker opens, without ``--yes`` the script asks before writing.

Produces ``Docs/genre_update_log.txt`` (the plan and each file's outcome) and
``Docs/tag_changes_genre_fill_cli_<time>.jsonl`` (before/after of every
write, failure and skip).

Prerequisites:
    pip install mutagen musicbrainzngs
"""

import os
import sys
import time

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import musicbrainzngs
from utils.audio_metadata_reader import read_tags, read_genres
from utils.tag_writer import SUPPORTED_TAG_EXTS, TagWriteError, write_fields
from utils.path_helpers import safe_relpath

try:
    from mutagen import File as MutagenFile  # noqa: F401 - kept for importers
except Exception:  # pragma: no cover - optional dependency
    MutagenFile = None  # type: ignore

# ─── CONFIGURATION ─────────────────────────────────────────────────────
#
# The MusicBrainz user agent is no longer set as an import side effect (with
# a placeholder e-mail address). Every lookup sets it from Settings first —
# see metadata_service.ensure_musicbrainz_useragent.

RATE_LIMIT_SECONDS = 1.0
#: One extension set for every tag tool (this used to omit ``.opus``).
SUPPORTED_EXTS = set(SUPPORTED_TAG_EXTS)

# ─── UTILITY FUNCTIONS ──────────────────────────────────────────────────

def choose_top_genres(mb_tag_list, top_n=3):
    """
    Return a list of up to top_n genre names sorted by 'count' descending.
    """
    if not mb_tag_list:
        return []

    valid = [t for t in mb_tag_list if "name" in t and t["name"].strip()]
    if not valid:
        return []

    for t in valid:
        if not isinstance(t.get("count"), int):
            try:
                t["count"] = int(t.get("count"))
            except (TypeError, ValueError):
                t["count"] = 0

    sorted_tags = sorted(valid, key=lambda t: (-t["count"], t["name"].lower()))
    return [t["name"].strip() for t in sorted_tags[:top_n]]


def extract_artist_title(filepath):
    """
    Return (artist, title) from file’s metadata, or (None, None).
    """
    try:
        tags = read_tags(filepath)
        artist = tags.get("artist")
        title = tags.get("title")
        return (artist.strip() if isinstance(artist, str) and artist else None,
                title.strip() if isinstance(title, str) and title else None)
    except Exception:
        return None, None


def extract_genre_list(filepath):
    """
    Return every genre value in the file (not just the first), or [] if none.
    """
    try:
        return list(read_genres(filepath))
    except Exception:
        return []


def write_genre_tag(filepath, new_genres):
    """Write ``new_genres`` into the file's genre tag; raise on failure.

    Raises :class:`utils.tag_writer.TagWriteError` whose message says why
    (a locked or read-only file, an unsupported format, ...), so a caller can
    log the reason instead of a bare "failed".
    """
    try:
        write_fields(filepath, {"genre": [str(g) for g in new_genres]})
    except TagWriteError:
        raise
    except Exception as exc:  # noqa: BLE001 - one plain error type for callers
        raise TagWriteError(str(exc) or type(exc).__name__) from exc


def update_genre_tag(filepath, new_genres):
    """
    Write new_genres (list of strings) into file’s 'genre' tag. Return True/False.

    Writes a real multi-value genre list in every supported format, including
    WAV/AIFF files with ID3 tags (which the easy interface can't write). Use
    :func:`write_genre_tag` to learn why a write failed.
    """
    try:
        write_genre_tag(filepath, new_genres)
        return True
    except TagWriteError:
        return False


def log_line(logfile, message):
    """
    Write a line to the logfile and flush immediately.
    """
    logfile.write(message + "\n")
    logfile.flush()


def lookup_musicbrainz_genres(filepath, top_n=3):
    """Look ``filepath`` up on MusicBrainz by artist + title. Never writes.

    Returns a ``normalize_controller.FillLookup``. No rate-limit sleep here;
    callers space out their requests.
    """
    from controllers.normalize_controller import FillLookup
    from metadata_service import ensure_musicbrainz_useragent

    artist, title = extract_artist_title(filepath)
    if not artist or not title:
        return FillLookup(status="missing_artist_title", detail="skipped (missing artist/title)", queried=False)
    try:
        ensure_musicbrainz_useragent(musicbrainzngs)
        result = musicbrainzngs.search_recordings(
            recording=title,
            artist=artist,
            limit=1,
            strict=True,
        )
    except Exception as e:  # noqa: BLE001
        return FillLookup(status="error", detail=f"MusicBrainz search error: {e}")
    recordings = (result or {}).get("recording-list", [])
    if not recordings:
        return FillLookup(status="no_match", detail="no MusicBrainz match")
    rec = recordings[0]
    top_genres = choose_top_genres(rec.get("tag-list", []), top_n=top_n)
    if not top_genres:
        return FillLookup(status="no_tags", detail="matched on MusicBrainz, but it has no tags")
    match = " – ".join(x for x in (artist, rec.get("title") or title) if x)
    return FillLookup(genres=top_genres, status="matched", detail=f"MusicBrainz: {match}")


# ─── CORE PROCESSING ───────────────────────────────────────────────────

def process_file(filepath, logfile):
    """Look up and write one file immediately — no preview.

    Kept for scripts that call it; :func:`main` no longer does (it previews
    every change and asks first). Same rules as the app: skip files with 2+
    genres, otherwise add the MusicBrainz tags after the existing genre. A
    file whose genres cannot be read is never written: its existing genres
    are unknown, and writing would replace them.
    """
    from controllers.normalize_controller import FILL_SKIP_AT, add_genres
    from utils.audio_metadata_reader import read_genres_checked

    # Relative to the working directory for a short log line; another drive
    # on Windows has no relative path, so the absolute path is logged then.
    relative = safe_relpath(filepath)
    existing = extract_genre_list(filepath)
    if not existing:
        _genres, read_error = read_genres_checked(filepath)
        if read_error:
            log_line(logfile, f"{relative}  → skipped (could not read its tags: {read_error}).")
            return
    if len(existing) >= FILL_SKIP_AT:
        log_line(logfile, f"{relative}  → skipped (already has genres {existing}).")
        return

    found = lookup_musicbrainz_genres(filepath)
    if found.queried:
        time.sleep(RATE_LIMIT_SECONDS)
    if not found.genres:
        log_line(logfile, f"{relative}  → {found.detail}.")
        return

    after = add_genres(existing, found.genres)
    if after == existing:
        log_line(logfile, f"{relative}  → already has [{', '.join(existing)}].")
        return
    try:
        write_genre_tag(filepath, after)
    except TagWriteError as exc:
        log_line(logfile, f"{relative}  → failed to write genres [{', '.join(after)}]: {exc}.")
    else:
        log_line(logfile, f"{relative}  → updated genres to [{', '.join(after)}].")


def _pick_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.5)
    except Exception:
        pass
    root.withdraw()
    return filedialog.askdirectory(title="Select Music Folder to Update Genres")


def run_cli(folder, *, assume_yes=False, ask=input, out=print, plan_kwargs=None, apply_kwargs=None):
    """Preview-first genre fill for ``folder``; returns the number of files written.

    1. Look every file up (nothing is written) and list each planned change.
    2. Ask ``ask(prompt)`` — unless ``assume_yes`` — and stop on anything but
       "y"/"yes".
    3. Write exactly the listed changes; a file whose genres changed since the
       preview is skipped. Every outcome goes to ``Docs/`` (see the module
       docstring).

    ``plan_kwargs`` / ``apply_kwargs`` are passed on to ``plan_genre_fill`` /
    ``apply_genre_changes`` (tests inject the lookup and the writer there).
    """
    from controllers.normalize_controller import apply_genre_changes, plan_genre_fill
    from controllers.tagfix_controller import discover_files
    from utils.tag_change_log import TagChangeLog

    docs_dir = os.path.join(folder, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    logpath = os.path.join(docs_dir, "genre_update_log.txt")
    with open(logpath, "w", encoding="utf-8") as logfile:
        log_line(logfile, f"Genre Update Log for: {folder}")
        log_line(logfile, f"Started: {time.ctime()}")
        log_line(logfile, "--------------------------------------")

        # Same discovery as the app: reserved folders (Not Sorted,
        # Quarantine…) are never entered.
        files = discover_files(folder)
        total = len(files)

        def _progress(index, count):
            if index:
                rel = safe_relpath(files[index - 1], folder)
                out(f"[{index}/{count}] Looking up: {rel}")

        plan = plan_genre_fill(
            folder, _progress, files=files,
            log_callback=lambda m: log_line(logfile, f"lookup: {m}"),
            **(plan_kwargs or {}),
        )
        out("")
        out(
            f"{len(plan.changes)} of {total} file(s) would get new genres "
            f"({len(plan.skipped_multi)} already have 2+, {len(plan.no_match)} without a match, "
            f"{len(plan.blocked)} not writable, {len(plan.errors)} lookup errors)."
        )
        for change in plan.changes:
            rel = safe_relpath(change.path, folder)
            line = f"{rel}: [{', '.join(change.before)}] → [{', '.join(change.after)}]"
            out("  " + line)
            log_line(logfile, f"planned: {line}")
        if not plan.changes:
            log_line(logfile, "Nothing to write.")
            out("Nothing to write.")
            return 0
        if not assume_yes:
            answer = ask(f"Write these genres to {len(plan.changes)} file(s)? [y/N] ")
            if str(answer).strip().lower() not in ("y", "yes"):
                log_line(logfile, "Not confirmed; nothing was written.")
                out("Nothing was written.")
                return 0

        with TagChangeLog(folder, "genre_fill_cli") as change_log:
            result = apply_genre_changes(
                plan.changes, change_log=change_log, **(apply_kwargs or {})
            )
        for path in result.applied:
            log_line(logfile, f"written: {safe_relpath(path, folder)}")
        for path in result.skipped:
            log_line(logfile, f"skipped (genres changed since the preview): {safe_relpath(path, folder)}")
        for path in result.failed:
            log_line(logfile, f"failed: {safe_relpath(path, folder)}")
        log_line(logfile, "--------------------------------------")
        log_line(logfile, f"Finished: {time.ctime()}")
        log_line(logfile, f"Total files checked: {total}")
        out(
            f"Wrote {len(result.applied)} file(s); {len(result.skipped)} skipped, "
            f"{len(result.failed)} failed. Log: {logpath}"
        )
        return len(result.applied)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    assume_yes = "--yes" in args
    args = [a for a in args if a != "--yes"]
    folder = args[0] if args else _pick_folder()
    if not folder:
        print("No folder selected; exiting.")
        sys.exit(0)
    if not os.path.isdir(folder):
        print(f"Not a folder: {folder}")
        sys.exit(1)
    run_cli(folder, assume_yes=assume_yes)


if __name__ == "__main__":
    main()
