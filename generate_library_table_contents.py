#!/usr/bin/env python3
"""
generate_library_table_contents.py

Pops up a folder picker, scans for audio files (including .ogg), reads tags
(artist, title, album, year, track, genre), and writes a single HTML file
named 'library_index.html' in that folder for easy reference.

Prerequisites:
    pip install mutagen
"""

import os, sys, tkinter as tk
from tkinter import filedialog
from mutagen import File as MutagenFile
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC

SUPPORTED_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".wav", ".ogg"}

def sanitize(text: str) -> str:
    if not text:
        return ""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def extract_tags(path):
    """
    Return a dict with keys: artist, title, album, year, track, genre (comma-separated).
    """
    ext = os.path.splitext(path)[1].lower()
    artist = title = album = year = track = None
    genres = []

    try:
        if ext == ".mp3":
            audio = EasyID3(path)
            artist = audio.get("artist", [None])[0]
            title  = audio.get("title",  [None])[0]
            album  = audio.get("album",  [None])[0]
            year   = audio.get("date",   [None])[0] or audio.get("year", [None])[0]
            track  = audio.get("tracknumber", [None])[0] or audio.get("track", [None])[0]
            genres = audio.get("genre", [])
        elif ext == ".flac":
            audio = FLAC(path)
            artist = audio.get("artist", [None])[0]
            title  = audio.get("title",  [None])[0]
            album  = audio.get("album",  [None])[0]
            year   = audio.get("date",   [None])[0] or audio.get("year", [None])[0]
            track  = audio.get("tracknumber", [None])[0] or audio.get("track", [None])[0]
            genres = audio.get("genre", [])
        else:
            audio = MutagenFile(path, easy=True)
            if audio and audio.tags:
                artist = audio.tags.get("artist", [None])[0]
                title  = audio.tags.get("title",  [None])[0]
                album  = audio.tags.get("album",  [None])[0]
                year   = audio.tags.get("date",   [None])[0] or audio.tags.get("year", [None])[0]
                track  = audio.tags.get("tracknumber", [None])[0] or audio.tags.get("track", [None])[0]
                genres = audio.tags.get("genre", [])
    except Exception:
        pass

    # Clean up fields
    artist = artist.strip() if isinstance(artist, str) else ""
    title  = title.strip()  if isinstance(title, str)  else ""
    album  = album.strip()  if isinstance(album, str)  else ""
    year   = year.strip()   if isinstance(year, str)   else ""
    if track:
        try:
            track = str(int(track.split("/")[0]))
        except:
            track = track.strip()
    else:
        track = ""
    genres = [g.strip() for g in genres if isinstance(g, str) and g.strip()]
    return {
        "artist": artist,
        "title":  title,
        "album":  album,
        "year":   year[:4] if year else "",
        "track":  track,
        "genres": ", ".join(genres)
    }


def main():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Music Folder to Index")
    if not folder:
        print("No folder selected; exiting.")
        sys.exit(0)

    entries = []
    for dirpath, _, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                full = os.path.join(dirpath, fname)
                tags = extract_tags(full)
                entries.append((os.path.relpath(full, folder), tags))

    html_lines = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html lang='en'><head><meta charset='utf-8'>")
    html_lines.append(f"<title>Library Index – {sanitize(os.path.basename(folder))}</title>")
    html_lines.append("<style>")
    html_lines.append("  body { font-family: Arial, sans-serif; background: #2e3440; color: #d8dee9; }")
    html_lines.append("  table { border-collapse: collapse; width: 100%; }")
    html_lines.append("  th, td { border: 1px solid #4c566a; padding: 6px; }")
    html_lines.append("  th { background: #4c566a; }")
    html_lines.append("  tr:nth-child(even) { background: #3b4252; }")
    html_lines.append("  tr:nth-child(odd)  { background: #434c5e; }")
    html_lines.append("</style></head><body>")
    html_lines.append(f"<h1>Music Library Index: {sanitize(os.path.basename(folder))}</h1>")
    html_lines.append("<table>")
    html_lines.append("<tr>")
    html_lines.append("<th>Path</th><th>Artist</th><th>Title</th><th>Album</th><th>Year</th><th>Track</th><th>Genres</th>")
    html_lines.append("</tr>")

    for relpath, t in sorted(entries, key=lambda x: x[0].lower()):
        html_lines.append("<tr>")
        html_lines.append(f"<td>{sanitize(relpath)}</td>")
        html_lines.append(f"<td>{sanitize(t['artist'])}</td>")
        html_lines.append(f"<td>{sanitize(t['title'])}</td>")
        html_lines.append(f"<td>{sanitize(t['album'])}</td>")
        html_lines.append(f"<td>{sanitize(t['year'])}</td>")
        html_lines.append(f"<td>{sanitize(t['track'])}</td>")
        html_lines.append(f"<td>{sanitize(t['genres'])}</td>")
        html_lines.append("</tr>")

    html_lines.append("</table>")
    html_lines.append("</body></html>")

    out_path = os.path.join(folder, "library_index.html")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))
        print(f"✓ Wrote library index to: {out_path}")
    except Exception as e:
        print(f"❌ Failed to write HTML: {e}")


if __name__ == "__main__":
    main()
