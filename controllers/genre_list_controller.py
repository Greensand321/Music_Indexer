import os
from typing import Set
from mutagen import File as MutagenFile

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}


def sanitize(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def extract_genre(filepath: str) -> str | None:
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


def gather_unique_genres(root_path: str) -> Set[str]:
    unique: Set[str] = set()
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                fullpath = os.path.join(dirpath, fname)
                g = extract_genre(fullpath)
                if g:
                    unique.add(g)
    return unique


def write_genres_html(root_path: str, genres: Set[str], output_filename: str = "genres.html") -> str:
    sorted_genres = sorted(genres, key=lambda s: s.lower())
    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang='en'>")
    lines.append("<head>")
    lines.append("  <meta charset='UTF-8'>")
    lines.append(f"  <title>Unique Genres in {sanitize(os.path.basename(root_path) or root_path)}</title>")
    lines.append("  <style>")
    lines.append("    body { background:#2e3440; color:#d8dee9; font-family:'Courier New', monospace; }")
    lines.append("    h1 { font-size:24px; }")
    lines.append("    ul { list-style-type: none; padding-left: 0; }")
    lines.append("    li { margin: 4px 0; }")
    lines.append("  </style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append(f"  <h1>Unique Genres in {sanitize(os.path.basename(root_path) or root_path)}</h1>")
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
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def list_unique_genres(folder_path: str) -> str:
    """Scan ``folder_path`` and write genres.html listing unique genres."""
    genres = gather_unique_genres(folder_path)
    return write_genres_html(folder_path, genres)
