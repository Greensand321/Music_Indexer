# music_indexer_api.py

import os
import re
import shutil
import hashlib
from collections import defaultdict
from mutagen import File as MutagenFile
from mutagen.id3 import ID3NoHeaderError

# ─── CONFIGURATION ─────────────────────────────────────────────────────
COMMON_ARTIST_THRESHOLD = 10
REMIX_FOLDER_THRESHOLD = 3   # minimum number of tracks for a “(Remixes)” album folder

# ─── Shared Utility Functions ─────────────────────────────────────────
def sanitize(name: str) -> str:
    if name is None:
        return "Unknown"
    invalid = r'<>:"/\\|?*'
    cleaned = "".join(c for c in name if c not in invalid).strip()
    return cleaned or "Unknown"

def collapse_repeats(name: str) -> str:
    """
    If name is something like 'DROELOEDROELOE', collapse to 'DROELOE'.
    For strings shorter than 4 characters, return unchanged.
    """
    if not name or len(name) < 4:
        return name or "Unknown"
    for length in range(1, len(name)//2 + 1):
        if len(name) % length == 0:
            chunk = name[:length]
            if chunk * (len(name)//length) == name:
                return chunk
    return name

def is_repeated(name: str) -> bool:
    """
    Detect if the entire artist tag is a repeated substring,
    e.g. 'DROELOEDROELOE'. For strings shorter than 4, return False.
    """
    if not name or len(name) < 4:
        return False
    for length in range(1, len(name)//2 + 1):
        if len(name) % length == 0:
            chunk = name[:length]
            if chunk * (len(name)//length) == name:
                return True
    return False

def get_tags(path: str):
    """
    Read basic tags using Mutagen (artist, title, album, year, track, genre).
    Return a dict with those fields (or None if missing).
    """
    try:
        audio = MutagenFile(path, easy=True)
        if not audio or not audio.tags:
            return {"artist": None, "title": None, "album": None,
                    "year": None, "track": None, "genre": None}
        tags = audio.tags
        artist = tags.get("artist", [None])[0]
        title  = tags.get("title",  [None])[0]
        album  = tags.get("album",  [None])[0]
        raw_date = tags.get("date", [None])[0] or tags.get("year", [None])[0]
        year = raw_date[:4] if raw_date else None
        raw_track = tags.get("tracknumber", [None])[0] or tags.get("track", [None])[0]
        track = None
        if raw_track:
            try:
                track = int(raw_track.split("/")[0])
            except Exception:
                track = None
        genre = tags.get("genre", [None])[0]
        return {"artist": artist, "title": title, "album": album,
                "year": year, "track": track, "genre": genre}
    except Exception:
        return {"artist": None, "title": None, "album": None,
                "year": None, "track": None, "genre": None}

def extract_primary_and_collabs(raw_artist: str):
    """
    Use collapse_repeats on raw_artist. Then:
      - If raw_artist contains "/" → split on "/" → first part is primary, rest are collabs.
      - Else look for separators: " feat.", " ft.", " & ", " x ", ", ", ";"
        First segment is primary, remainder split by commas/& is collabs.
      - Else split on uppercase boundary, treat first as primary and rest as a single collab string.
      - Otherwise, primary = raw_artist, collabs = [].
    """
    if not raw_artist:
        return ("Unknown", [])
    text = raw_artist.strip()
    text = collapse_repeats(text)

    if "/" in text:
        parts = [collapse_repeats(p.strip()) for p in text.split("/") if p.strip()]
        return (parts[0], parts[1:])

    lowered = text.lower()
    separators = [" feat.", " ft.", " & ", " x ", ", ", ";"]
    for sep in separators:
        idx = lowered.find(sep)
        if idx != -1:
            primary = collapse_repeats(text[:idx].strip())
            rest = text[idx + len(sep):].strip()
            subparts = [collapse_repeats(p.strip()) for p in re.split(r"\s*&\s*|,\s*", rest) if p.strip()]
            return (primary, subparts)

    parts = re.split(r'(?<=[a-z])(?=[A-Z])', text)
    if len(parts) > 1:
        primary = collapse_repeats(parts[0].strip())
        collabs = [" & ".join(collapse_repeats(p.strip()) for p in parts[1:])]
        return (primary, collabs)

    return (text, [])

def derive_tags_from_path(filepath: str, music_root: str):
    """
    If a file lives in subfolders under music_root, gather those folder names
    as “inherited tags.” E.g. /…/By Artist/ArtistName/AlbumX/Track.mp3 → {"ArtistName", "AlbumX"}.
    """
    rel = os.path.relpath(os.path.dirname(filepath), music_root)
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    return set(parts)


# ─── A. HELPER: COMPUTE MOVES & TAG INDEX ───────────────────────────────

def compute_moves_and_tag_index(root_path, log_callback=None):
    """
    1) Determine MUSIC_ROOT: if root_path/Music exists, use that; otherwise root_path itself.
    2) Scan for all audio files under MUSIC_ROOT.
    3) Deduplicate by (primary, title, album) and delete lower-priority duplicates.
    4) Read metadata into `songs` dict, build:
       - primary_counts: how many kept files per artist
       - album_counts: how many genuine (non-remix) tracks per (artist, album)
       - remix_counts: how many remix‐tagged tracks per (artist, album)
       - cover_counts: how many files share each embedded cover
    5) Phase 4: For each entry in `songs`, decide new folder & filename,
       with special‐case for small “(Remixes)” albums.
    Returns:
      - moves: { old_path: new_path, ... }
      - tag_index: { new_path: { "leftover_tags": [...], "old_paths": [...] }, ... }
      - decision_log: list of strings explaining each track’s decision
    """
    if log_callback is None:
        def log_callback(msg): pass

    # ─── 1) Determine MUSIC_ROOT ──────────────────────────────────────────────
    MUSIC_ROOT = os.path.join(root_path, "Music") \
        if os.path.isdir(os.path.join(root_path, "Music")) \
        else root_path

    SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}

    # ─── Phase 1: Scan for audio files ─────────────────────────────────────────
    log_callback("1/6: Scanning for all audio files…")
    all_audio = []
    for dirpath, _, files in os.walk(MUSIC_ROOT):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                all_audio.append(os.path.join(dirpath, fname))
    total_audio = len(all_audio)
    log_callback(f"   → Found {total_audio} audio files.")

    # ─── Phase 2: Deduplicate by (primary, title, album) ────────────────────────
    log_callback("2/6: Deduplicating by (primary, title, album)…")
    dup_groups = defaultdict(list)
    for idx, fullpath in enumerate(all_audio, start=1):
        if idx % 50 == 0 or idx == total_audio:
            log_callback(f"   • Processing file {idx}/{total_audio} for dedupe")
        data = get_tags(fullpath)
        raw_artist = data["artist"] or os.path.splitext(os.path.basename(fullpath))[0]
        raw_artist = collapse_repeats(raw_artist)
        title  = data["title"]  or os.path.splitext(os.path.basename(fullpath))[0]
        album  = data["album"]  or ""
        primary, _ = extract_primary_and_collabs(raw_artist)
        key = (primary.lower(), title.lower(), album.lower())
        dup_groups[key].append(fullpath)

    kept_files = set()
    EXT_PRIORITY = {".flac": 0, ".m4a": 1, ".aac": 1, ".mp3": 2, ".wav": 3, ".ogg": 4}

    for key, paths in dup_groups.items():
        if len(paths) == 1:
            kept_files.add(paths[0])
        else:
            paths_sorted = sorted(
                paths,
                key=lambda p: EXT_PRIORITY.get(os.path.splitext(p)[1].lower(), 999)
            )
            best = paths_sorted[0]
            kept_files.add(best)
            for loser in paths_sorted[1:]:
                try:
                    os.remove(loser)
                    log_callback(f"   - Deleted duplicate: {loser}")
                except Exception as e:
                    log_callback(f"   ! Failed to delete {loser}: {e}")

    # ─── Phase 3: Read metadata & build counters ─────────────────────────────────
    log_callback("3/6: Reading metadata and building counters…")
    songs = {}
    primary_counts = defaultdict(int)
    album_counts   = defaultdict(int)
    remix_counts   = defaultdict(int)
    cover_counts   = defaultdict(int)
    total_kept = len(kept_files)

    for idx, fullpath in enumerate(kept_files, start=1):
        if idx % 50 == 0 or idx == total_kept:
            log_callback(f"   • Reading metadata {idx}/{total_kept}")

        data = get_tags(fullpath)
        raw_artist = data["artist"] or os.path.splitext(os.path.basename(fullpath))[0]
        raw_artist = collapse_repeats(raw_artist)
        title   = data["title"] or os.path.splitext(os.path.basename(fullpath))[0]
        album   = data["album"]
        year    = data["year"]
        genre   = data["genre"]
        track   = data["track"]

        # 1) Primary & collabs (normalize case)
        primary, collabs = extract_primary_and_collabs(raw_artist)
        primary = primary.upper()
        p_lower = primary.lower()

        # 2) Count how often each artist appears
        primary_counts[p_lower] += 1

        # 3) album_counts for genuine (non-remix) tracks under each (artist, album)
        if album and "remix" not in title.lower() and album.strip().lower() != title.strip().lower():
            album_counts[(p_lower, album.lower())] += 1

        # 4) remix_counts for tracks whose album tag contains "remix"
        if album and "remix" in album.lower():
            remix_counts[(p_lower, album.lower())] += 1

        # 5) EXTRACT EMBEDDED COVER DATA → compute SHA-1 and store first 10 hex digits
        cover_hash = None
        try:
            audio_file = MutagenFile(fullpath)
            img_data = None
            if hasattr(audio_file, "tags") and audio_file.tags is not None:
                for key in audio_file.tags.keys():
                    if key.startswith("APIC"):
                        img_data = audio_file.tags[key].data
                        break
            if img_data is None and audio_file.__class__.__name__ == "FLAC":
                pics = audio_file.pictures
                if pics:
                    img_data = pics[0].data
            if img_data:
                sha1 = hashlib.sha1(img_data).hexdigest()
                cover_hash = sha1[:10]
                cover_counts[cover_hash] += 1
        except ID3NoHeaderError:
            pass
        except Exception:
            pass

        folder_tags = derive_tags_from_path(fullpath, MUSIC_ROOT)

        songs[fullpath] = {
            "raw_artist": raw_artist,
            "primary":    primary,
            "collabs":    collabs,
            "title":      title,
            "album":      album,
            "year":       year,
            "genre":      genre,
            "track":      track,
            "cover_hash": cover_hash,
            "folder_tags": folder_tags
        }

    log_callback(f"   → Collected metadata for {total_kept} files.")

    # ─── Phase 4: Determine destination for each file (with logging) ─────────────
    log_callback("4/6: Determining destination paths for each file…")
    moves = {}
    tag_index = {}
    decision_log = []
    total_songs = len(songs)

    for idx, (old_path, info) in enumerate(songs.items(), start=1):
        if idx % 50 == 0 or idx == total_songs:
            log_callback(f"   • Determining destination {idx}/{total_songs}")

        raw_artist = info["raw_artist"]
        primary    = info["primary"]
        collabs    = info["collabs"]
        title      = info["title"]
        album      = info["album"]
        year       = info["year"] or "Unknown Year"
        genre      = info["genre"]
        track      = info["track"]
        cover_hash = info.get("cover_hash")
        folders    = info["folder_tags"]

        p_lower = primary.lower()
        decision_log.append(f"\n---\nSong: {os.path.basename(old_path)}")
        decision_log.append(f"  Raw artist tag: “{raw_artist}” → initial primary = “{primary}”")

        # ─── 4.A) SPECIAL CASE: Album ends with “(Remixes)” but only if enough tracks exist ───
        if album and album.strip().lower().endswith("(remixes)"):
            rcount = remix_counts.get((p_lower, album.lower()), 0)
            decision_log.append(f"  Album '{album}' has {rcount} remix‐tagged tracks (threshold={REMIX_FOLDER_THRESHOLD})")
            if rcount >= REMIX_FOLDER_THRESHOLD:
                # Force primary to the first segment of raw_artist (before “/”)
                main_artist = raw_artist.split("/", 1)[0].upper()
                p_lower = main_artist.lower()
                decision_log.append(f"  → Enough remixes ({rcount} ≥ {REMIX_FOLDER_THRESHOLD}); force primary = '{main_artist}'")
                artist_folder = os.path.join(MUSIC_ROOT, "By Artist", sanitize(main_artist))
                base_folder = os.path.join(artist_folder, sanitize(album))

                # Build filename (skip all other ranking/album logic)
                basename = os.path.basename(old_path)
                if "/" in raw_artist or is_repeated(raw_artist):
                    new_filename = basename
                    decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping filename '{basename}'")
                else:
                    ext = os.path.splitext(old_path)[1].lower()
                    filename_artist = sanitize(raw_artist)
                    track_str = f"{track:02d}" if track is not None else "00"
                    title_str = sanitize(title)
                    new_filename = f"{filename_artist}_{track_str}_{title_str}{ext}"
                    decision_log.append(f"  Renaming to '{new_filename}' (using raw artist)")

                new_path = os.path.join(base_folder, new_filename)
                moves[old_path] = new_path
                decision_log.append(
                    f"  → (Remixes folder) placed under '{main_artist}/{album}' → Final: '{os.path.relpath(new_path, MUSIC_ROOT)}'"
                )
                continue
            else:
                decision_log.append(
                    f"  → Only {rcount} remixes (< {REMIX_FOLDER_THRESHOLD}); treat as singles/year."
                )

        # ─── 4.0) BROAD COLLABORATOR RANKING ───────────────────────────────────
        all_candidates = [primary] + collabs
        counts = {artist: primary_counts.get(artist.lower(), 0) for artist in all_candidates}
        best_artist = max(counts, key=lambda a: counts[a])
        decision_log.append(f"  Candidates: {all_candidates}, counts: {counts}")
        primary = best_artist
        decision_log.append(f"  → After ranking, primary = '{primary}'")

        # Normalize to uppercase so “DROELOE” vs “Droeloe” collapse
        primary = primary.upper()
        p_lower = primary.lower()
        decision_log.append(f"  → Normalized to uppercase primary = '{primary}'")

        # ─── 4.1) COVER-FIRST GROUPING (Threshold ≥ 5) ──────────────────────────
        goto_build_filename = False
        if cover_hash and cover_counts.get(cover_hash, 0) >= 5:
            artist_folder = os.path.join(MUSIC_ROOT, "By Artist", sanitize(primary))
            if album:
                base_folder = os.path.join(artist_folder, sanitize(album))
                decision_log.append(
                    f"  Cover‐first: cover_hash={cover_hash} appears {cover_counts[cover_hash]} times; group under Album '{album}'"
                )
            else:
                base_folder = os.path.join(artist_folder, f"{sanitize(primary)} – Cover_{cover_hash}")
                decision_log.append(
                    f"  Cover‐first: cover_hash={cover_hash} appears {cover_counts[cover_hash]} times; "
                    f"group under '{primary} – Cover_{cover_hash}'"
                )
            goto_build_filename = True

        # ─── 4.2) (If NOT cover-first) REMIX & ALBUM vs. SINGLES vs. YEAR ──────
        if not goto_build_filename:
            # 4.2A) Title ends with "(XYZ Remix)"?
            is_remix = False
            remix_match = re.search(r"\(([^)]+)\s+Remix\)\s*$", title, flags=re.IGNORECASE)
            if remix_match:
                remixer_name = collapse_repeats(remix_match.group(1).strip())
                if primary_counts.get(remixer_name.lower(), 0) > 0:
                    decision_log.append(
                        f"  Title ends with '({remixer_name} Remix)': promoting to remixer "
                        f"(count={primary_counts[remixer_name.lower()]})"
                    )
                    primary = remixer_name.upper()
                    p_lower = primary.lower()
                    decision_log.append(f"  → New primary = '{primary}'")
                else:
                    decision_log.append(
                        f"  Title ends with '({remixer_name} Remix)', but remixer count="
                        f"{primary_counts.get(remixer_name.lower(), 0)} → no promotion"
                    )
                is_remix = True

            # 4.2B) If album tag literally contains "remix"
            if album and "remix" in album.lower():
                c_count = album_counts.get((p_lower, album.lower()), 0)
                artist_folder = os.path.join(MUSIC_ROOT, "By Artist", sanitize(primary))
                if c_count > 3:
                    base_folder = os.path.join(artist_folder, sanitize(album))
                    decision_log.append(
                        f"  Album tag '{album}' contains 'remix' and album_count={c_count} > 3 → "
                        f"group under Album '{album}'"
                    )
                else:
                    base_folder = os.path.join(artist_folder, f"{sanitize(primary)} - Singles")
                    decision_log.append(
                        f"  Album tag '{album}' contains 'remix' but album_count={c_count} ≤ 3 → "
                        f"group under '{primary} - Singles'"
                    )

            else:
                # 4.2C) Common vs. Rare artist (using COMMON_ARTIST_THRESHOLD)
                count_now = primary_counts.get(p_lower, 0)
                decision_log.append(
                    f"  No cover‐first, no remix/'remix'‐album. primary_counts[{primary}] = {count_now}"
                )
                if count_now >= COMMON_ARTIST_THRESHOLD:
                    artist_folder = os.path.join(MUSIC_ROOT, "By Artist", sanitize(primary))
                    if not album or album.strip().lower() == title.strip().lower():
                        base_folder = os.path.join(artist_folder, f"{sanitize(primary)} - Singles")
                        decision_log.append(
                            f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, no album or album=title "
                            f"→ group under '{primary} - Singles'"
                        )
                    else:
                        c2 = album_counts.get((p_lower, album.lower()), 0)
                        if c2 > 3:
                            base_folder = os.path.join(artist_folder, sanitize(album))
                            decision_log.append(
                                f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, album_count={c2} > 3 "
                                f"→ group under Album '{album}'"
                            )
                        else:
                            base_folder = os.path.join(artist_folder, f"{sanitize(primary)} - Singles")
                            decision_log.append(
                                f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, but album_count={c2} ≤ 3 "
                                f"→ group under '{primary} - Singles'"
                            )
                else:
                    base_folder = os.path.join(MUSIC_ROOT, "By Year", sanitize(year))
                    decision_log.append(
                        f"  Count {count_now} < {COMMON_ARTIST_THRESHOLD} → group under By Year/{year}"
                    )

        # ─── 4.3) Build the new filename (preserving raw_artist order) ───────────
        basename = os.path.basename(old_path)
        if "/" in raw_artist or is_repeated(raw_artist):
            new_filename = basename
            decision_log.append(
                f"  Raw artist '{raw_artist}' malformed → keeping filename '{basename}'"
            )
        else:
            ext = os.path.splitext(old_path)[1].lower()
            filename_artist = sanitize(raw_artist)
            track_str = f"{track:02d}" if track is not None else "00"
            title_str = sanitize(title)
            new_filename = f"{filename_artist}_{track_str}_{title_str}{ext}"
            decision_log.append(f"  Renaming to '{new_filename}' (using raw artist)")

        new_path = os.path.join(base_folder, new_filename)
        moves[old_path] = new_path
        decision_log.append(f"  Final destination: '{os.path.relpath(new_path, MUSIC_ROOT)}'")

        # ─── 4.4) Build “leftover_tags” for HTML preview (optional) ───────────────
        leftover = set(folders)
        leftover.discard(primary)
        if album:
            leftover.discard(album)
        leftover.discard(year)
        if genre:
            leftover.add(genre)

        tag_index[new_path] = {
            "leftover_tags": sorted(leftover),
            "old_paths": sorted(folders)
        }

    log_callback("   → Destination paths determined for all files.")
    return moves, tag_index, decision_log


# ─── B. BUILD DRY-RUN HTML ─────────────────────────────────────────────

def build_dry_run_html(root_path, output_html_path, log_callback=None):
    """
    1) Call compute_moves_and_tag_index() to get (moves, tag_index, decision_log).
    2) Write a dry-run HTML to output_html_path showing the proposed tree.
    Does NOT move any files.
    """
    if log_callback is None:
        def log_callback(msg): pass

    moves, tag_index, _ = compute_moves_and_tag_index(root_path, log_callback)

    log_callback("5/6: Writing dry-run HTML…")
    with open(output_html_path, "w", encoding="utf-8") as out:
        out.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Music Index (Dry Run) – {sanitize(os.path.basename(root_path))}</title>
  <style>
    body {{ background:#2e3440; color:#d8dee9; font-family:'Courier New', monospace; }}
    pre  {{ font-size:14px; }}
    .folder {{ color:#81a1c1; }}
    .song   {{ color:#a3be8c; }}
    .tags   {{ color:#88c0d0; font-size:12px; margin-left:1em; }}
  </style>
</head>
<body>
<pre>
""")
        out.write(f"<span class=\"folder\">{sanitize(os.path.basename(root_path))}/</span>\n\n")
        tree_nodes = set()
        for new_path in moves.values():
            parts = os.path.relpath(new_path, root_path).split(os.sep)
            for i in range(1, len(parts) + 1):
                subtree = os.path.join(root_path, *parts[:i])
                tree_nodes.add(subtree)

        for node in sorted(tree_nodes, key=lambda p: os.path.relpath(p, root_path)):
            rel = os.path.relpath(node, root_path)
            depth = rel.count(os.sep)
            indent = "    " * depth
            _, ext = os.path.splitext(node)
            if os.path.isdir(node) or ext.lower() not in {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}:
                out.write(f"{indent}<span class=\"folder\">{sanitize(os.path.basename(node))}/</span>\n")
            else:
                fname = os.path.basename(node)
                leftover = tag_index.get(node, {}).get("leftover_tags", [])
                tags_str = ", ".join(leftover) if leftover else ""
                out.write(f"{indent}<span class=\"song\">- {sanitize(fname)}</span>")
                if tags_str:
                    out.write(f"  <span class=\"tags\">[{sanitize(tags_str)}]</span>")
                out.write("\n")

        out.write("</pre>\n</body>\n</html>\n")

    log_callback(f"✓ Dry-run HTML written to: {output_html_path}")
    return {"moved": 0, "html": output_html_path, "dry_run": True}


# ─── C. APPLY MOVES ─────────────────────────────────────────────────────

def apply_indexer_moves(root_path, log_callback=None):
    """
    1) Call compute_moves_and_tag_index() to get (moves, tag_index, decision_log).
    2) Move/rename each file in `moves`.
    3) Move any leftover non-audio or album cover images into Trash or into the correct folder.
    4) Remove empty directories.
    Returns summary: {"moved": <count>, "errors": [<error strings>]}.
    """
    if log_callback is None:
        def log_callback(msg): pass

    moves, _, _ = compute_moves_and_tag_index(root_path, log_callback)

    MUSIC_ROOT = os.path.join(root_path, "Music") \
        if os.path.isdir(os.path.join(root_path, "Music")) \
        else root_path
    SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
    IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif"}

    summary = {"moved": 0, "errors": []}
    total_moves = len(moves)

    # Build a mapping: old_dir → set of target_dirs for audio files in that folder
    olddir_to_newdirs = defaultdict(set)
    for old_path, new_path in moves.items():
        old_dir = os.path.dirname(old_path)
        new_dir = os.path.dirname(new_path)
        olddir_to_newdirs[old_dir].add(new_dir)

    # Phase 5: Move audio files
    for idx, (old_path, new_path) in enumerate(moves.items(), start=1):
        if idx % 50 == 0 or idx == total_moves:
            log_callback(f"   • Moving file {idx}/{total_moves}")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        try:
            if os.path.abspath(old_path) != os.path.abspath(new_path):
                shutil.move(old_path, new_path)
                summary["moved"] += 1
        except Exception as e:
            err = f"Failed to move {old_path} → {new_path}: {e}"
            summary["errors"].append(err)
            log_callback(f"   ! {err}")

    # Phase 6: Handle non-audio leftovers and album covers
    TRASH_ROOT = os.path.join(root_path, "Trash")
    os.makedirs(TRASH_ROOT, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(MUSIC_ROOT, topdown=False):
        target_dirs = olddir_to_newdirs.get(dirpath, set())

        for fname in filenames:
            full = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()

            # 6.1) If this was not an audio file (ext not in SUPPORTED_EXTS)
            if full not in moves and ext not in SUPPORTED_EXTS:
                # 6.1.a) If it’s an image and all audio from this folder went to one album folder,
                #       move the image into that album folder so the cover stays with that album.
                if ext in IMAGE_EXTS and len(target_dirs) == 1:
                    dest_folder = next(iter(target_dirs))
                    try:
                        shutil.move(full, os.path.join(dest_folder, fname))
                    except Exception:
                        try:
                            shutil.move(full, os.path.join(TRASH_ROOT, fname))
                        except Exception:
                            base, ext2 = os.path.splitext(fname)
                            count = 1
                            newname = f"{base}_{count}{ext2}"
                            while os.path.exists(os.path.join(TRASH_ROOT, newname)):
                                count += 1
                                newname = f"{base}_{count}{ext2}"
                            shutil.move(full, os.path.join(TRASH_ROOT, newname))
                else:
                    # 6.1.b) Otherwise, send it to Trash
                    try:
                        shutil.move(full, os.path.join(TRASH_ROOT, fname))
                    except Exception:
                        base, ext2 = os.path.splitext(fname)
                        count = 1
                        newname = f"{base}_{count}{ext2}"
                        while os.path.exists(os.path.join(TRASH_ROOT, newname)):
                            count += 1
                            newname = f"{base}_{count}{ext2}"
                        shutil.move(full, os.path.join(TRASH_ROOT, newname))

        # 6.2) Remove empty directory if it is now empty
        if not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
            except Exception:
                pass

    return summary


# ─── D. HIGH-LEVEL “RUN FULL INDEXER” ───────────────────────────────────

def run_full_indexer(root_path, output_html_path, dry_run_only=False, log_callback=None):
    """
    1) Call compute_moves_and_tag_index() to get (moves, tag_index, decision_log).
    2) Write a detailed log file `indexer_log.txt` under root_path.
    3) Write dry-run HTML via build_dry_run_html().
    4) If dry_run_only=False, call apply_indexer_moves() to move files.
    Returns summary: {"moved": <count>, "html": <path>, "dry_run": <True/False>}.
    """
    if log_callback is None:
        def log_callback(msg): pass

    # Phase 1–4: Generate moves, tags, and collect decision log
    moves, tag_index, decision_log = compute_moves_and_tag_index(root_path, log_callback)

    # Write the detailed log file
    log_path = os.path.join(root_path, "indexer_log.txt")
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("Indexing Decision Log\n")
        lf.write("======================\n")
        lf.write(f"Library root: {root_path}\n")
        lf.write(f"Generated on: {__import__('datetime').datetime.now()}\n\n")
        for line in decision_log:
            lf.write(line + "\n")
    log_callback(f"✓ Detailed log written to: {log_path}")

    # Build dry-run HTML
    log_callback("5/6: Writing dry-run HTML…")
    build_dry_run_html(root_path, output_html_path, log_callback)

    if not dry_run_only:
        # Phase 5–6: Actually move audio files and cover images
        actual_summary = apply_indexer_moves(root_path, log_callback)
        summary = {"moved": actual_summary["moved"], "html": output_html_path, "dry_run": False}
        return summary
    else:
        return {"moved": 0, "html": output_html_path, "dry_run": True}
