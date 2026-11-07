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

The Qt app looks everything up first, shows the result, and writes only after
the user confirms (``normalize_controller.plan_genre_fill`` +
``apply_genre_changes``). This script's ``main()`` is the legacy command-line
version and writes as it goes.

While processing, the script prints simple progress messages like
``[index/total] Processing <file>``.

Produces a log file “genre_update_log.txt” summarizing each file’s result.

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


def update_genre_tag(filepath, new_genres):
    """
    Write new_genres (list of strings) into file’s 'genre' tag. Return True/False.

    Writes a real multi-value genre list in every supported format, including
    WAV/AIFF files with ID3 tags (which the easy interface can't write).
    """
    try:
        write_fields(filepath, {"genre": [str(g) for g in new_genres]})
        return True
    except TagWriteError:
        return False
    except Exception:
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
    """Legacy one-shot path (CLI): look up and write immediately.

    Same rules as the app's preview: skip files with 2+ genres, otherwise add
    the MusicBrainz tags after the existing genre.
    """
    from controllers.normalize_controller import FILL_SKIP_AT, add_genres

    relative = os.path.relpath(filepath)
    existing = extract_genre_list(filepath)
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
    success = update_genre_tag(filepath, after)
    if success:
        log_line(logfile, f"{relative}  → updated genres to [{', '.join(after)}].")
    else:
        log_line(logfile, f"{relative}  → failed to write genres [{', '.join(after)}].")


def main():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.5)
    except Exception:
        pass
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Music Folder to Update Genres")
    if not folder:
        print("No folder selected; exiting.")
        sys.exit(0)

    docs_dir = os.path.join(folder, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    logpath = os.path.join(docs_dir, "genre_update_log.txt")
    try:
        logfile = open(logpath, "w", encoding="utf-8")
    except Exception as e:
        print(f"Cannot open log file: {e}")
        sys.exit(1)

    log_line(logfile, f"Genre Update Log for: {folder}")
    log_line(logfile, f"Started: {time.ctime()}")
    log_line(logfile, "--------------------------------------")

    # Same discovery as the app: reserved folders (Not Sorted, Quarantine…)
    # are never entered.
    from controllers.tagfix_controller import discover_files

    all_files = discover_files(folder)

    total = len(all_files)

    for index, fullpath in enumerate(all_files, 1):
        rel = os.path.relpath(fullpath, folder)
        print(f"[{index}/{total}] Processing: {rel}")
        process_file(fullpath, logfile)

    log_line(logfile, "--------------------------------------")
    log_line(logfile, f"Finished: {time.ctime()}")
    log_line(logfile, f"Total files checked: {total}")
    logfile.close()
    print(f"Done. Check log at: {logpath}")


if __name__ == "__main__":
    main()
