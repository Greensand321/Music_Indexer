# music_indexer_api.py

import os
import re
import shutil  # used for relocating special folders
import hashlib
import sqlite3
from collections import defaultdict
from mutagen import File as MutagenFile
from mutagen.id3 import ID3NoHeaderError

# ─── CONFIGURATION ─────────────────────────────────────────────────────
COMMON_ARTIST_THRESHOLD = 10
REMIX_FOLDER_THRESHOLD  = 3
SUPPORTED_EXTS          = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}

# ─── Helper: build primary_counts for the entire vault ─────────────────────
def build_primary_counts(root_path):
    """
    Walk the entire vault (By Artist, By Year, Incoming, etc.) and
    return a dict mapping each lowercase-normalized artist → total file count.
    Falls back to filename if metadata “artist” is missing.
    """
    counts = {}
    for dirpath, _, files in os.walk(root_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue

            path = os.path.join(dirpath, fname)
            tags = get_tags(path)

            # Normalize artist identically to Phase 4
            raw = (tags.get("artist") or "").strip()

            if not raw:
                name_only = os.path.splitext(fname)[0]
                if "_" in name_only:
                    raw = name_only.split("_", 1)[0]
                elif " - " in name_only:
                    raw = name_only.split(" - ", 1)[0]
                else:
                    raw = name_only

            primary, _ = extract_primary_and_collabs(raw)
            p_lower = primary.lower()
            counts[p_lower] = counts.get(p_lower, 0) + 1

    return counts

# ─── Shared Utility Functions ─────────────────────────────────────────
def sanitize(name: str) -> str:
    invalid = r'<>:"/\\|?*'
    return "".join(c for c in (name or "Unknown") if c not in invalid).strip() or "Unknown"

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
    3) Deduplicate by (primary, title, album, fingerprint) and delete lower-priority duplicates.
    4) Read metadata into `songs` dict, build:
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

    docs_dir = os.path.join(root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    db_path = os.path.join(docs_dir, ".soundvault.db")
    db_folder = os.path.dirname(db_path)
    os.makedirs(db_folder, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
          path TEXT PRIMARY KEY,
          duration INT,
          fingerprint TEXT
        );
        """
    )

    # --- Phase 0: Pre-scan entire vault (under MUSIC_ROOT) ---
    global_counts = build_primary_counts(MUSIC_ROOT)
    log_callback(f"   → Pre-scan: found {len(global_counts)} unique artists")
    log_callback(f"   → DEBUG: MUSIC_ROOT = {MUSIC_ROOT}")
    log_callback(f"   → DEBUG: droeloe count = {global_counts.get('droeloe', 0)}")

    # --- Phase 1: Scan for audio files ---
    all_audio = []
    for dirpath, _, files in os.walk(MUSIC_ROOT):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                all_audio.append(os.path.join(dirpath, fname))
    total_audio = len(all_audio)
    log_callback(f"   → Found {total_audio} audio files.")

    # ─── Phase 2: Deduplicate by (primary, title, album, fingerprint) ────────────────────────
    log_callback("2/6: Deduplicating by (primary, title, album, fingerprint)…")
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
        row = db.execute(
            "SELECT fingerprint FROM fingerprints WHERE path=?",
            (fullpath,),
        ).fetchone()
        fp = row[0] if row else None
        key = (primary.lower(), title.lower(), album.lower(), fp or "")
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

        # 2) album_counts for genuine (non-remix) tracks under each (artist, album)
        if album and "remix" not in title.lower() and album.strip().lower() != title.strip().lower():
            album_counts[(p_lower, album.lower())] += 1

        # 3) remix_counts for tracks whose album tag contains "remix"
        if album and "remix" in album.lower():
            remix_counts[(p_lower, album.lower())] += 1

        # 4) EXTRACT EMBEDDED COVER DATA → compute SHA-1 and store first 10 hex digits
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
    tag_index = {}
    moves = {}
    decision_log = []
    index = 0
    total = len(songs)

    # Precompute any necessary lookups from songs data
    # (album_counts, remix_counts, etc. were built in Phase 3)

    for old_path, info in songs.items():
        index += 1
        if index % 50 == 0 or index == total:
            log_callback(f"   • Determining destination {index}/{total}")

        raw_artist = info["raw_artist"]
        title      = info["title"] or ""
        album      = info["album"] or ""
        year       = info["year"] or "Unknown"
        track      = info["track"]
        folders    = info["folder_tags"]

        # Normalize again for consistent global lookup
        primary_norm, _ = extract_primary_and_collabs(raw_artist)
        p_lower = primary_norm.lower()

        # Lookup how many total tracks this artist has across the entire vault
        count_now = global_counts.get(p_lower, 0)
        decision_log.append(f"Song: {raw_artist} – {title} ({album}) → global_counts[{raw_artist}] = {count_now}")

        # ─── 4.1) EARLY BAIL-OUT: “Rare” artists go straight to By Year ─────────────────
        if count_now < COMMON_ARTIST_THRESHOLD:
            base_folder = os.path.join(MUSIC_ROOT, "By Year", sanitize(year))
            decision_log.append(
                f"  Early-exit: Count {count_now} < {COMMON_ARTIST_THRESHOLD} → group under By Year/{year}"
            )

            # Build the new filename as usual (rename invalid artists or use sanitized tags)
            if "/" in raw_artist or is_repeated(raw_artist):
                new_filename = os.path.basename(old_path)
                decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping filename '{new_filename}'")
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
                f"  → Final: '{os.path.relpath(new_path, MUSIC_ROOT)}'"
            )
            # Skip all other logic (no remix or album/singles checks)
            continue

        # At this point, count_now ≥ COMMON_ARTIST_THRESHOLD
        # ─── 4.2) SPECIAL CASE: Album ends with “(Remixes)” ──────────────────────────
        if album and album.strip().lower().endswith("(remixes)"):
            rcount = remix_counts.get((p_lower, album.lower()), 0)
            decision_log.append(
                f"  Album '{album}' has {rcount} remix-tagged tracks (threshold={REMIX_FOLDER_THRESHOLD})"
            )

            # Only force into By Artist if both remix-count AND artist-count thresholds are met
            if rcount >= REMIX_FOLDER_THRESHOLD and count_now >= COMMON_ARTIST_THRESHOLD:
                # Promote primary to remixer (first part of raw_artist before “/”)
                main_artist = raw_artist.split("/", 1)[0]
                p_lower = main_artist.lower()
                decision_log.append(
                    f"  → Enough remixes ({rcount} ≥ {REMIX_FOLDER_THRESHOLD}) "
                    f"AND count_now ({count_now}) ≥ {COMMON_ARTIST_THRESHOLD}; force primary = '{main_artist}'"
                )
                artist_folder = os.path.join(MUSIC_ROOT, "By Artist", sanitize(main_artist))
                base_folder = os.path.join(artist_folder, sanitize(album))

                # Build filename (skip normal album logic)
                basename = os.path.basename(old_path)
                if "/" in raw_artist or is_repeated(raw_artist):
                    new_filename = basename
                    decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping '{basename}'")
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
                    f"  → (Remixes folder) placed under By Artist/{main_artist}/{album} "
                    f"→ Final: '{os.path.relpath(new_path, MUSIC_ROOT)}'"
                )
                continue
            else:
                decision_log.append(
                    f"  → Either only {rcount} remixes (< {REMIX_FOLDER_THRESHOLD}) "
                    f"or count_now ({count_now}) < {COMMON_ARTIST_THRESHOLD}; "
                    f"fall back to album/singles logic"
                )

        # ─── 4.3) BROAD COLLABORATOR RANKING (unchanged) ────────────────────────────
        all_candidates = [info["primary"]] + info["collabs"]
        counts = {artist: global_counts.get(artist.lower(), 0) for artist in all_candidates}
        best_artist = max(counts, key=lambda a: counts[a])
        decision_log.append(f"  Candidates: {all_candidates}, counts: {counts}")
        primary = best_artist
        p_lower = primary.lower()
        decision_log.append(f"  → After ranking, primary = '{primary}'")

        # Build “By Artist/<primary>” base path
        artist_folder = os.path.join(MUSIC_ROOT, "By Artist", sanitize(primary))

        # ─── 4.4) COMMON ARTIST: decide between “Album” vs. “Singles” ───────────────
        # (At this point, count_now ≥ COMMON_ARTIST_THRESHOLD is guaranteed.)
        if not album or album.strip().lower() == title.strip().lower():
            # No album tag (or album == title) → put into “<Artist> - Singles”
            base_folder = os.path.join(artist_folder, f"{sanitize(primary)} - Singles")
            decision_log.append(
                f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, no album or album=title "
                f"→ group under '{primary} - Singles'"
            )
        else:
            c2 = album_counts.get((p_lower, album.lower()), 0)
            if c2 > 3:
                # Enough tracks in this album → put into “<Artist>/<Album>”
                base_folder = os.path.join(artist_folder, sanitize(album))
                decision_log.append(
                    f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, album_count={c2} > 3 "
                    f"→ group under Album '{album}'"
                )
            else:
                # Few tracks in album → treat as a “Singles” release
                base_folder = os.path.join(artist_folder, f"{sanitize(primary)} - Singles")
                decision_log.append(
                    f"  Count {count_now} ≥ {COMMON_ARTIST_THRESHOLD}, but "
                    f"album_count={c2} ≤ 3 → group under '{primary} - Singles'"
                )

        # Build filename for all “common” tracks
        basename = os.path.basename(old_path)
        if "/" in raw_artist or is_repeated(raw_artist):
            new_filename = basename
            decision_log.append(f"  Raw artist '{raw_artist}' malformed → keeping '{basename}'")
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
            f"  → Final: '{os.path.relpath(new_path, MUSIC_ROOT)}'"
        )

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
    db.close()
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

def apply_indexer_moves(root_path, log_callback=None, progress_callback=None):
    """
    1) Call compute_moves_and_tag_index() to get (moves, tag_index, decision_log).
    2) Move/rename each file in `moves`.
    3) Move any leftover non-audio or album cover images into Trash or into the correct folder.
    4) Remove empty directories.
    Returns summary: {"moved": <count>, "errors": [<error strings>]}.
    """
    if log_callback is None:
        def log_callback(msg):
            pass
    if progress_callback is None:
        def progress_callback(current, total, path):
            pass

    # ─── Phase 0: Relocate Holidays & Soundtracks to root ────────────────
    for special in ("Holidays", "Soundtracks"):
        dest = os.path.join(root_path, special)
        for dirpath, _, _ in os.walk(root_path):
            if os.path.basename(dirpath) == special and os.path.abspath(dirpath) != os.path.abspath(dest):
                os.makedirs(root_path, exist_ok=True)
                try:
                    shutil.move(dirpath, dest)
                    log_callback(f"✓ Moved {special} to root: {dest}")
                except Exception as e:
                    log_callback(f"! Could not move {special}: {e}")
                break

    # ─── Phase 0.5: Compute and cache fingerprints ─────────────────
    docs_dir = os.path.join(root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    db_path = os.path.join(docs_dir, ".soundvault.db")
    from fingerprint_generator import compute_fingerprints
    compute_fingerprints(root_path, db_path, log_callback)

    moves, _, _ = compute_moves_and_tag_index(root_path, log_callback)

    # ─── Phase 5: Exclude Holidays & Soundtracks from indexing ───────────
    filtered = {}
    for old, new in moves.items():
        first = os.path.relpath(old, root_path).split(os.sep, 1)[0]
        if first in ("Holidays", "Soundtracks"):
            log_callback(f"⚠ Skipping in {first}: {old}")
            continue
        filtered[old] = new
    moves = filtered

    total_moves = len(moves)

    MUSIC_ROOT = os.path.join(root_path, "Music") \
        if os.path.isdir(os.path.join(root_path, "Music")) \
        else root_path
    SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
    IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif"}

    summary = {"moved": 0, "errors": []}

    # Build a mapping: old_dir → set of target_dirs for audio files in that folder
    olddir_to_newdirs = defaultdict(set)
    for old_path, new_path in moves.items():
        old_dir = os.path.dirname(old_path)
        new_dir = os.path.dirname(new_path)
        olddir_to_newdirs[old_dir].add(new_dir)

    # Phase 5: Move audio files
    for idx, (old_path, new_path) in enumerate(moves.items(), start=1):
        if progress_callback:
            progress_callback(idx, total_moves, old_path)
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

    # Phase 6: Handle non-audio leftovers…
    docs_dir = os.path.join(root_path, "Docs")
    trash_dir = os.path.join(root_path, "Trash")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(trash_dir, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(MUSIC_ROOT, topdown=False):
        # 1) Don’t recurse back into Trash/ or Docs/
        dirnames[:] = [d for d in dirnames if d.lower() not in ("trash", "docs")]

        for fname in filenames:
            full = os.path.join(dirpath, fname)
            ext  = os.path.splitext(fname)[1].lower()

            # 2) Skip files already moved or valid audio
            if full in moves or ext in SUPPORTED_EXTS:
                continue

            target = docs_dir if ext in (".txt", ".html", ".db") else trash_dir
            if target is docs_dir:
                log_callback(f"Moving doc to Docs/: {full}")
            else:
                log_callback(f"Moving to Trash/: {full}")
            try:
                shutil.move(full, os.path.join(target, fname))
            except Exception as e:
                log_callback(f"   ! Failed to move leftover {full}: {e}")

    # ─── Phase 7: Generate Playlists with edge-case handling ────────────
    try:
        from playlist_generator import generate_playlists
        log_callback("7/7: Generating safe playlists…")

        def plog(msg):
            log_callback(msg)
            if progress_callback and msg.startswith("→ Writing playlist:"):
                playlist_file = msg.split(":", 1)[1].strip()
                progress_callback(total_moves, total_moves, playlist_file)

        generate_playlists(
            moves,
            root_path,
            output_dir=None,        # default: root_path/Playlists
            valid_exts=None,        # default extensions
            overwrite=True,
            log_callback=plog,
        )
        log_callback("✓ Robust playlists created.")
    except ImportError:
        log_callback("! playlist_generator.py missing; skipping playlists.")
    except Exception as e:
        log_callback(f"! Playlist generation error: {e}")

    return summary


# ─── D. HIGH-LEVEL “RUN FULL INDEXER” ───────────────────────────────────

def run_full_indexer(root_path, output_html_path, dry_run_only=False, log_callback=None, progress_callback=None):
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
    docs_dir = os.path.join(root_path, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    log_path = os.path.join(docs_dir, "indexer_log.txt")
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
        actual_summary = apply_indexer_moves(
            root_path,
            log_callback,
            progress_callback,
        )
        summary = {"moved": actual_summary["moved"], "html": output_html_path, "dry_run": False}
        return summary
    else:
        return {"moved": 0, "html": output_html_path, "dry_run": True}


# ─── E. DUPLICATE DETECTION HELPER ─────────────────────────────────────

def find_duplicates(root_path, log_callback=None):
    """Return list of (original_path, duplicate_path) that would be marked as
    duplicates by the indexer."""

    moves, _, _ = compute_moves_and_tag_index(root_path, log_callback)
    dup_indicator = os.path.join("Duplicates", "")
    return [
        (old, new)
        for old, new in moves.items()
        if dup_indicator in new
    ]
