import os
import sys
import argparse
import acoustid
import musicbrainzngs
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, Callable, List, Optional
from mutagen import File as MutagenFile

import sqlite3

# ─── Database Helpers ─────────────────────────────────────────────────────
def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
          path        TEXT PRIMARY KEY,
          status      TEXT,
          score       REAL,
          old_artist  TEXT, new_artist TEXT,
          old_title   TEXT, new_title  TEXT,
          old_album   TEXT, new_album  TEXT,
          old_genres  TEXT, new_genres TEXT
        )
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_status ON files(status)")
    conn.commit()
    return conn

# ─── Configuration ────────────────────────────────────────────────────────
ACOUSTID_API_KEY       = "eBOqCZhyAx"
ACOUSTID_APP_NAME      = "SoundVaultTagFixer"
ACOUSTID_APP_VERSION   = "1.0.0"
SUPPORTED_EXTS         = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav"}

musicbrainzngs.set_useragent(
    "SoundVaultTagFixer",
    "1.0",
    "youremail@example.com",
)

# ─── Data Classes ────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    """Representation of a single audio file and proposed tags."""

    path: Path
    status: str
    score: Optional[float] = None
    old_artist: Optional[str] = None
    new_artist: Optional[str] = None
    old_title: Optional[str] = None
    new_title: Optional[str] = None
    old_album: Optional[str] = None
    new_album: Optional[str] = None
    old_genres: List[str] = field(default_factory=list)
    new_genres: List[str] = field(default_factory=list)

# Score thresholds
MIN_AUTOMATIC_SCORE   = 0.90   # ≥90% → apply automatically
MIN_INTERACTIVE_SCORE = 0.75   # ≥75% & <90% → prompt user

# ─── Utility Functions ────────────────────────────────────────────────────
def is_remix(audio_path):
    """Return True if filename or existing title suggests a remix."""
    if "remix" in os.path.basename(audio_path).lower():
        return True
    audio = MutagenFile(audio_path, easy=True)
    if audio and audio.tags and "title" in audio.tags:
        title = " ".join(audio.tags["title"]).lower()
        if "remix" in title:
            return True
    return False

def find_files(root):
    """Return a list of audio files under `root` (file or directory)."""
    if os.path.isfile(root):
        return [root]
    audio_files = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                audio_files.append(os.path.join(dirpath, fname))
    return audio_files

from itertools import islice

def query_acoustid(path, log_callback):
    try:
        match_gen = acoustid.match(ACOUSTID_API_KEY, path)
        # Grab first 5 for debugging, then put them back into a list
        peek = list(islice(match_gen, 5))
        if not peek:
            log_callback("  No matches at all")
            return None

        # Debug: show those 5
        for i, (score, rid, title, artist) in enumerate(peek, start=1):
            log_callback(f"  [{i}] score={score:.4f} → “{artist} – {title}”")

        # Now get the very first result for your decision
        best_score, best_rid, best_title, best_artist = peek[0]
        return {
            "title": best_title,
            "artist": best_artist,
            "score": best_score,
            "recording_id": best_rid,
        }

    except acoustid.NoBackendError:
        log_callback("Chromaprint library/tool not found")
    except acoustid.FingerprintGenerationError:
        log_callback(f"Failed to fingerprint: {path}")
    except acoustid.WebServiceError as exc:
        log_callback(f"AcoustID request failed: {exc}")
    return None

def update_tags(path: str, proposal: FileRecord, fields: List[str], log_callback):
    """Write selected tags from ``proposal`` into ``path``. Return True if saved."""
    audio = MutagenFile(path, easy=True)
    if audio is None:
        return False
    changed = False
    if "artist" in fields and proposal.new_artist is not None and proposal.new_artist != proposal.old_artist:
        audio["artist"] = [proposal.new_artist]
        changed = True
    if "title" in fields and proposal.new_title is not None and proposal.new_title != proposal.old_title:
        audio["title"] = [proposal.new_title]
        changed = True
    if "album" in fields and proposal.new_album is not None and proposal.new_album != proposal.old_album:
        audio["album"] = [proposal.new_album]
        changed = True
    if "genres" in fields:
        existing = audio.tags.get("genre", []) if audio.tags else []
        old = proposal.old_genres or existing
        new = proposal.new_genres or []
        merged = []
        seen: set[str] = set()
        for g in list(old) + list(new):
            if g not in seen:
                merged.append(g)
                seen.add(g)
        if merged != existing:
            audio["genre"] = ["; ".join(merged)]
            changed = True
    if changed:
        try:
            audio.save()
            log_callback(f"Updated tags for {path}")
            return True
        except Exception as e:
            log_callback(f"Failed to save {path}: {e}")
    return False

def prompt_user_about_tags(f, old_artist, old_title, new_tags):
    """Show side-by-side old vs new and ask Y/N."""
    new_artist = new_tags["artist"]
    new_title  = new_tags["title"]
    print(f"\nFile: {f}")
    print(f"{'Field':10} │ {'Current':30} │ {'New from AcoustID':30}")
    print("-" * 75)
    print(f"{'Artist':10} │ {old_artist or '—':30} │ {new_artist:30}")
    print(f"{'Title':10} │ {old_title  or '—':30} │ {new_title:30}")
    print()
    resp = input("Apply these changes? [y/N]: ").strip().lower()
    return (resp == "y")

# ─── Main Tag-Fixing Logic ────────────────────────────────────────────────
def build_file_records(
    root: str,
    *,
    db: sqlite3.Connection,
    show_all: bool = False,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> List[FileRecord]:
    """Return a list of ``FileRecord`` objects for ``root``."""

    if log_callback is None:
        def log_callback(msg: str):
            print(msg)

    records: List[FileRecord] = []

    existing_status = dict(db.execute("SELECT path, status FROM files"))

    files = find_files(root)
    for idx, f in enumerate(files, start=1):
        status = existing_status.get(f, "new")

        if status == "applied" and not show_all:
            if progress_callback:
                progress_callback(idx)
            continue

        if is_remix(f) and not show_all:
            if progress_callback:
                progress_callback(idx)
            continue

        log_callback(f"Fingerprinting {f}")
        try:
            result = query_acoustid(f, log_callback)
        except Exception as e:
            log_callback(f"  lookup error: {e}")
            result = None

        if not result or not result.get("title"):
            if not result:
                log_callback(f"  no fingerprint match")

            audio = MutagenFile(f, easy=True)
            old_artist = (audio.tags.get("artist") or [None])[0] if audio and audio.tags else None
            old_title = (audio.tags.get("title") or [None])[0] if audio and audio.tags else None
            old_album = (audio.tags.get("album") or [None])[0] if audio and audio.tags else None
            old_genres = audio.tags.get("genre", []) if audio and audio.tags else []

            rec = FileRecord(
                path=Path(f),
                status="unmatched",
                score=None,
                old_artist=old_artist,
                new_artist=None,
                old_title=old_title,
                new_title=None,
                old_album=old_album,
                new_album=None,
                old_genres=old_genres,
                new_genres=[],
            )
            records.append(rec)
            genres_old = ";".join(rec.old_genres)
            genres_new = ";".join(rec.new_genres)
            vals = (
                str(rec.path), rec.status, rec.score,
                rec.old_artist, rec.new_artist,
                rec.old_title, rec.new_title,
                rec.old_album, rec.new_album,
                genres_old, genres_new,
            )
            db.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                vals,
            )
            if progress_callback:
                progress_callback(idx)
            continue

        score = result.get("score")
        if progress_callback:
            progress_callback(idx)

        mb_album = None
        mb_genres: List[str] = []
        mbid = result.get("recording_id")
        if mbid:
            try:
                rec = musicbrainzngs.get_recording_by_id(
                    mbid,
                    includes=["releases", "tags"],
                )["recording"]
                rels = rec.get("releases", [])
                if rels:
                    mb_album = rels[0].get("title")
                mb_tags = rec.get("tag-list", [])
                mb_genres = [t["name"] for t in mb_tags if "name" in t]
            except Exception as e:
                log_callback(f"  MB lookup failed: {e}")

        audio = MutagenFile(f, easy=True)
        old_artist = (audio.tags.get("artist") or [None])[0] if audio and audio.tags else None
        old_title = (audio.tags.get("title") or [None])[0] if audio and audio.tags else None
        old_album = (audio.tags.get("album") or [None])[0] if audio and audio.tags else None
        old_genres = audio.tags.get("genre", []) if audio and audio.tags else []

        rec = FileRecord(
            path=Path(f),
            status=status,
            score=score,
            old_artist=old_artist,
            new_artist=result.get("artist"),
            old_title=old_title,
            new_title=result.get("title"),
            old_album=old_album,
            new_album=mb_album,
            old_genres=old_genres,
            new_genres=mb_genres,
        )

        all_match = (
            rec.old_artist == rec.new_artist
            and rec.old_title == rec.new_title
            and rec.old_album == rec.new_album
            and sorted(rec.old_genres or []) == sorted(rec.new_genres or [])
        )
        if all_match:
            rec.status = "no_diff"
        records.append(rec)
        genres_old = ";".join(rec.old_genres)
        genres_new = ";".join(rec.new_genres)
        vals = (
            str(rec.path), rec.status, rec.score,
            rec.old_artist, rec.new_artist,
            rec.old_title, rec.new_title,
            rec.old_album, rec.new_album,
            genres_old, genres_new,
        )
        db.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            vals,
        )

    db.commit()
    return records


def apply_tag_proposals(
    selected: Iterable[FileRecord],
    *,
    fields: List[str] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> int:
    """Apply ``selected`` proposals and return number of files updated."""
    if log_callback is None:
        def log_callback(msg: str):
            print(msg)

    if fields is None:
        fields = ["artist", "title"]

    updated = 0
    for p in selected:
        if update_tags(str(p.path), p, fields, log_callback):
            updated += 1
    return updated


def fix_tags(target, log_callback=None, interactive=False):
    """Fill missing tags for files in target using AcoustID."""
    if log_callback is None:
        def log_callback(msg):
            print(msg)

    files = find_files(target)
    if not files:
        log_callback("No audio files found.")
        return {"processed": 0, "updated": 0}

    db_path = os.path.join(
        target if os.path.isdir(target) else os.path.dirname(target),
        ".soundvault.db",
    )
    db = init_db(db_path)

    records = build_file_records(
        target,
        db=db,
        show_all=False,
        log_callback=log_callback,
    )
    selected = [p for p in records if p.status != "no_diff"]

    if interactive:
        selected = []
        for p in [r for r in records if r.status != "no_diff"]:
            apply_change = prompt_user_about_tags(
                str(p.path),
                p.old_artist,
                p.old_title,
                {"artist": p.new_artist, "title": p.new_title}
            )
            if apply_change:
                selected.append(p)

    root_path = target if os.path.isdir(target) else os.path.dirname(target)
    updated = apply_tag_proposals(
        selected,
        fields=["artist", "title"],
        log_callback=log_callback,
    )

    selected_set = {rec.path for rec in selected}
    for rec in records:
        if rec.path in selected_set:
            status = "applied"
        elif rec.status == "no_diff":
            status = "no_diff"
        else:
            status = "skipped"
        rec.status = status
        db.execute(
            "UPDATE files SET status=? WHERE path=?",
            (status, str(rec.path)),
        )

    db.commit()
    return {"processed": len(files), "updated": updated}

# ─── CLI Entry Point ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fill missing audio tags using AcoustID"
    )
    parser.add_argument("target", help="file or folder to process")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="show and confirm tag changes for medium-confidence matches",
    )
    args = parser.parse_args()

    try:
        summary = fix_tags(args.target, interactive=args.interactive)
        print(f"\nProcessed {summary['processed']} files, updated {summary['updated']}.")
    except RuntimeError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
