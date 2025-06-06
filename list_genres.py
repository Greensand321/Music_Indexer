#!/usr/bin/env python3
"""
list_genres.py

Pops up a Windows “Select Folder” dialog, scans that folder (and subfolders)
for audio files, extracts every file’s “genre” tag, then writes out a simple
HTML file called 'genres.html' in the same folder—listing each unique genre once.

Usage:
    python list_genres.py
    (or double-click it on Windows)
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
from mutagen import File as MutagenFile

# ─── CONFIGURATION ─────────────────────────────────────────────────────
SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}

# ─── UTILITY FUNCTIONS ──────────────────────────────────────────────────
def sanitize(text: str) -> str:
    """
    Escape HTML‐special characters in a string so it can be safely embedded
    in an HTML page.
    """
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

def extract_genre(filepath: str) -> str:
    """
    Try to read the 'genre' tag from an audio file using Mutagen (Easy).
    Returns the genre string if found (first entry), or None if no genre.
    """
    try:
        audio = MutagenFile(filepath, easy=True)
        if not audio or not audio.tags:
            return None
        raw_genre = audio.tags.get("genre", [None])[0]
        if raw_genre:
            return raw_genre.strip()
        return None
    except Exception:
        return None

def gather_unique_genres(root_path: str) -> set:
    """
    Walk all subfolders of root_path, find audio files, extract their genre tags,
    and return a set of unique, non‐empty genre strings.
    """
    unique = set()
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                fullpath = os.path.join(dirpath, fname)
                g = extract_genre(fullpath)
                if g:
                    unique.add(g)
    return unique

def write_genres_html(root_path: str, genres: set, output_filename: str = "genres.html"):
    """
    Write out a simple HTML page into root_path/output_filename listing each
    genre (one per line) in alphabetized order.
    """
    sorted_genres = sorted(genres, key=lambda s: s.lower())
    lines = []

    lines.append("<!DOCTYPE html>")
    lines.append("<html lang=\"en\">")
    lines.append("<head>")
    lines.append("  <meta charset=\"UTF-8\">")
    lines.append(f"  <title>Unique Genres in “{sanitize(os.path.basename(root_path) or root_path)}”</title>")
    lines.append("  <style>")
    lines.append("    body { background:#2e3440; color:#d8dee9; font-family:'Courier New', monospace; }")
    lines.append("    h1 { font-size:24px; }")
    lines.append("    ul { list-style-type: none; padding-left: 0; }")
    lines.append("    li { margin: 4px 0; }")
    lines.append("  </style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append(f"  <h1>Unique Genres in “{sanitize(os.path.basename(root_path) or root_path)}”</h1>")
    lines.append("  <ul>")

    if sorted_genres:
        for genre in sorted_genres:
            lines.append(f"    <li>{sanitize(genre)}</li>")
    else:
        lines.append("    <li><em>No genre tags found</em></li>")

    lines.append("  </ul>")
    lines.append("</body>")
    lines.append("</html>")

    output_path = os.path.join(root_path, output_filename)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✓ Wrote unique genres list to: {output_path}")
    except Exception as e:
        print(f"❌ Error writing HTML file: {e}")

def main():
    # Hide the main Tkinter window; we only want the folder dialog
    root = tk.Tk()
    root.withdraw()

    # Pop up “Select Folder” dialog
    folder_selected = filedialog.askdirectory(title="Select Folder to Scan for Genres")
    if not folder_selected:
        print("No folder selected. Exiting.")
        sys.exit(0)

    print(f"Scanning '{folder_selected}' for audio genres…")
    genres = gather_unique_genres(folder_selected)
    write_genres_html(folder_selected, genres, output_filename="genres.html")

if __name__ == "__main__":
    main()
