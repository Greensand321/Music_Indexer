#!/usr/bin/env python3
"""
update_genres.py

Walk a chosen folder and all subfolders for audio files. For each file, read
its “artist” and “title” tags, query MusicBrainz for a matching recording, pull
the MB recording’s tags (genres), pick the top 3 by popularity, and write them
back into the file’s own “genre” tag field (using Mutagen).

While processing, the script prints simple progress messages like
``[index/total] Processing <file>``. Install ``tqdm`` if you’d prefer a fancy
progress bar instead.

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

import tkinter as tk
from tkinter import filedialog

import musicbrainzngs
try:
    from mutagen import File as MutagenFile
    from mutagen.easyid3 import EasyID3
    from mutagen.flac import FLAC
except Exception:  # pragma: no cover - optional dependency
    class _DummyAudio:
        def __init__(self, *a, **k):
            self.tags = {}

        def get(self, key, default=None):
            return self.tags.get(key, default)

    def MutagenFile(*_a, **_k):
        return _DummyAudio()

    class EasyID3(_DummyAudio):
        pass

    class FLAC(_DummyAudio):
        pass

# ─── CONFIGURATION ─────────────────────────────────────────────────────

musicbrainzngs.set_useragent(
    "SoundVaultGenreUpdater",
    "1.0",
    "youremail@example.com"
)

RATE_LIMIT_SECONDS = 1.0
SUPPORTED_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".wav", ".ogg"}

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
            t["count"] = 0

    sorted_tags = sorted(valid, key=lambda t: (-t["count"], t["name"].lower()))
    return [t["name"].strip() for t in sorted_tags[:top_n]]


def extract_artist_title(filepath):
    """
    Return (artist, title) from file’s metadata, or (None, None).
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".mp3":
            audio = EasyID3(filepath)
            artist = audio.get("artist", [None])[0]
            title = audio.get("title", [None])[0]
        elif ext == ".flac":
            audio = FLAC(filepath)
            artist = audio.get("artist", [None])[0]
            title = audio.get("title", [None])[0]
        else:
            audio = MutagenFile(filepath, easy=True)
            if not audio or not audio.tags:
                return None, None
            artist = audio.tags.get("artist", [None])[0]
            title = audio.tags.get("title", [None])[0]
        return (artist.strip() if artist else None,
                title.strip() if title else None)
    except Exception:
        return None, None


def extract_genre_list(filepath):
    """
    Return a list of existing genres from the file, or [] if none.
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".mp3":
            audio = EasyID3(filepath)
            return audio.get("genre", [])
        elif ext == ".flac":
            audio = FLAC(filepath)
            return audio.get("genre", [])
        else:
            audio = MutagenFile(filepath, easy=True)
            if not audio or not audio.tags:
                return []
            return audio.tags.get("genre", [])
    except Exception:
        return []


def update_genre_tag(filepath, new_genres):
    """
    Write new_genres (list of strings) into file’s 'genre' tag. Return True/False.
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".mp3":
            audio = EasyID3(filepath)
            audio["genre"] = new_genres
            audio.save()
        elif ext == ".flac":
            audio = FLAC(filepath)
            audio["genre"] = new_genres
            audio.save()
        else:
            audio = MutagenFile(filepath, easy=True)
            if not audio or not audio.tags:
                return False
            audio.tags["genre"] = new_genres
            audio.save()
        return True
    except Exception:
        return False


def log_line(logfile, message):
    """
    Write a line to the logfile and flush immediately.
    """
    logfile.write(message + "\n")
    logfile.flush()


# ─── CORE PROCESSING ───────────────────────────────────────────────────

def process_file(filepath, logfile):
    relative = os.path.relpath(filepath)
    artist, title = extract_artist_title(filepath)
    if not artist or not title:
        log_line(logfile, f"{relative}  → skipped (missing artist/title).")
        return

    try:
        result = musicbrainzngs.search_recordings(
            recording=title,
            artist=artist,
            limit=1,
            strict=True
        )
    except Exception as e:
        log_line(logfile, f"{relative}  → MB search error: {e}")
        time.sleep(RATE_LIMIT_SECONDS)
        return

    recordings = result.get("recording-list", [])
    if not recordings:
        log_line(logfile, f"{relative}  → no MB match.")
        time.sleep(RATE_LIMIT_SECONDS)
        return

    rec = recordings[0]
    mb_tags = rec.get("tag-list", [])
    top_genres = choose_top_genres(mb_tags, top_n=3)

    if not top_genres:
        log_line(logfile, f"{relative}  → matched MB but no tags.")
        time.sleep(RATE_LIMIT_SECONDS)
        return

    existing = extract_genre_list(filepath)
    if existing and len(existing) >= 2:
        log_line(logfile, f"{relative}  → skipped (already has genres {existing}).")
        time.sleep(RATE_LIMIT_SECONDS)
        return

    success = update_genre_tag(filepath, top_genres)
    if success:
        log_line(logfile, f"{relative}  → updated genres to [{', '.join(top_genres)}].")
    else:
        log_line(logfile, f"{relative}  → failed to write genres [{', '.join(top_genres)}].")
    time.sleep(RATE_LIMIT_SECONDS)


def main():
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

    logpath = os.path.join(folder, "genre_update_log.txt")
    try:
        logfile = open(logpath, "w", encoding="utf-8")
    except Exception as e:
        print(f"Cannot open log file: {e}")
        sys.exit(1)

    log_line(logfile, f"Genre Update Log for: {folder}")
    log_line(logfile, f"Started: {time.ctime()}")
    log_line(logfile, "--------------------------------------")

    all_files = []
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                all_files.append(os.path.join(dirpath, fname))

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
