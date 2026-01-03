import os
from tkinter import messagebox
from utils.audio_metadata_reader import read_tags

SUPPORTED_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".wav", ".ogg"}


def sanitize(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def extract_tags(path: str) -> dict:
    """Return basic metadata for an audio file."""
    tags = read_tags(path)
    artist = tags.get("artist")
    title = tags.get("title")
    album = tags.get("album")
    year = tags.get("year")
    track = tags.get("track")
    raw_genres = tags.get("genre")

    artist = artist.strip() if isinstance(artist, str) else ""
    title = title.strip() if isinstance(title, str) else ""
    album = album.strip() if isinstance(album, str) else ""
    year = str(year).strip() if year is not None else ""
    if track is not None:
        track = str(track).strip()
    else:
        track = ""
    if raw_genres in (None, ""):
        genres = []
    elif isinstance(raw_genres, (list, tuple)):
        genres = [g.strip() for g in raw_genres if isinstance(g, str) and g.strip()]
    else:
        genres = [str(raw_genres).strip()] if str(raw_genres).strip() else []
    return {
        "artist": artist,
        "title": title,
        "album": album,
        "year": year[:4] if year else "",
        "track": track,
        "genres": ", ".join(genres),
    }


def generate_index(folder_path: str) -> str:
    """Generate library_index.html for ``folder_path`` and return its path."""
    entries = []
    for dirpath, _, files in os.walk(folder_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                full = os.path.join(dirpath, fname)
                tags = extract_tags(full)
                entries.append((os.path.relpath(full, folder_path), tags))

    html_lines: list[str] = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html lang='en'><head><meta charset='utf-8'>")
    html_lines.append(f"<title>Library Index – {sanitize(os.path.basename(folder_path))}</title>")
    html_lines.append("<style>")
    html_lines.append("  body { font-family: Arial, sans-serif; background: #2e3440; color: #d8dee9; }")
    html_lines.append("  table { border-collapse: collapse; width: 100%; }")
    html_lines.append("  th, td { border: 1px solid #4c566a; padding: 6px; }")
    html_lines.append("  th { background: #4c566a; }")
    html_lines.append("  tr:nth-child(even) { background: #3b4252; }")
    html_lines.append("  tr:nth-child(odd)  { background: #434c5e; }")
    html_lines.append("</style></head><body>")
    html_lines.append(f"<h1>Music Library Index: {sanitize(os.path.basename(folder_path))}</h1>")
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

    out_path = os.path.join(folder_path, "library_index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    messagebox.showinfo(
        "Library Index Generated",
        f"Your library index has been saved to:\n{out_path}"
    )
    return out_path
