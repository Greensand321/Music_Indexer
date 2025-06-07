import os
import sys
import argparse
import acoustid
import musicbrainzngs
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, Callable, List, Tuple, Dict, Optional
from mutagen import File as MutagenFile

import log_manager

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
class TagProposal:
    file: Path
    old_artist: Optional[str] = None
    new_artist: Optional[str] = None
    old_title: Optional[str] = None
    new_title: Optional[str] = None
    old_album: Optional[str] = None
    new_album: Optional[str] = None
    old_genres: List[str] = field(default_factory=list)
    new_genres: List[str] = field(default_factory=list)
    score: float = 0.0

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

def update_tags(path: str, proposal: TagProposal, fields: List[str], log_callback):
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
def collect_tag_proposals(
    files: Iterable[str],
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    *,
    show_all: bool = False,
) -> Tuple[List[TagProposal], List[TagProposal]]:
    """Return (diff_proposals, no_diff_files) for the given ``files``."""
    if log_callback is None:
        def log_callback(msg: str):
            print(msg)

    if not ACOUSTID_API_KEY:
        raise RuntimeError("ACOUSTID_API_KEY not configured")

    diff: List[TagProposal] = []
    no_diff: List[TagProposal] = []
    for idx, f in enumerate(files, start=1):
        if is_remix(f):
            if progress_callback:
                progress_callback(idx)
            continue

        log_callback(f"Fingerprinting {f}")
        result = query_acoustid(f, log_callback)
        if progress_callback:
            progress_callback(idx)
        if not result:
            continue

        score = result["score"]
        if not show_all and score < MIN_INTERACTIVE_SCORE:
            continue

        mb_album = None
        mb_genres = []
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

        entry = TagProposal(
            file=Path(f),
            old_artist=old_artist,
            new_artist=result["artist"],
            old_title=old_title,
            new_title=result["title"],
            old_album=old_album,
            new_album=mb_album,
            old_genres=old_genres,
            new_genres=mb_genres,
            score=score,
        )

        all_match = (
            entry.old_artist == entry.new_artist
            and entry.old_title == entry.new_title
            and entry.old_album == entry.new_album
            and sorted(entry.old_genres or []) == sorted(entry.new_genres or [])
        )
        if all_match:
            no_diff.append(entry)
        else:
            diff.append(entry)

    return diff, no_diff


def apply_tag_proposals(
    selected: Iterable[TagProposal],
    diff_proposals: Iterable[TagProposal],
    no_diff_files: Iterable[TagProposal],
    library_root: str,
    fields: List[str] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> int:
    """Apply ``selected`` proposals and log all scan results."""
    if log_callback is None:
        def log_callback(msg: str):
            print(msg)

    if fields is None:
        fields = ["artist", "title"]

    selected_paths = {str(p.file) for p in selected}

    updated = 0
    for p in selected:
        if update_tags(str(p.file), p, fields, log_callback):
            updated += 1

    # ─── Update log only after successful writes ──────────────────────────
    log_data = log_manager.load_log(library_root)

    def record(entry: TagProposal, status: str):
        rel = os.path.relpath(entry.file, library_root)
        log_data[rel] = {
            "status": status,
            "old_artist": entry.old_artist,
            "old_title": entry.old_title,
            "new_artist": entry.new_artist,
            "new_title": entry.new_title,
            "old_album": entry.old_album,
            "new_album": entry.new_album,
            "old_genres": entry.old_genres,
            "new_genres": entry.new_genres,
        }

    for p in diff_proposals:
        if str(p.file) in selected_paths:
            record(p, "applied")
        else:
            record(p, "skipped")

    for p in no_diff_files:
        record(p, "no_diff")

    log_manager.save_log(log_data, library_root)
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

    diff_props, no_diff = collect_tag_proposals(files, log_callback)
    selected = diff_props

    if interactive:
        selected = []
        for p in diff_props:
            apply_change = prompt_user_about_tags(
                str(p.file),
                p.old_artist,
                p.old_title,
                {"artist": p.new_artist, "title": p.new_title}
            )
            if apply_change:
                selected.append(p)

    root_path = target if os.path.isdir(target) else os.path.dirname(target)
    updated = apply_tag_proposals(
        selected,
        diff_props,
        no_diff,
        root_path,
        fields=["artist", "title"],
        log_callback=log_callback,
    )

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
