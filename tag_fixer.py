import os
import sys
import argparse
import acoustid
from mutagen import File as MutagenFile

# ─── Configuration ────────────────────────────────────────────────────────
ACOUSTID_API_KEY       = "eBOqCZhyAx"
ACOUSTID_APP_NAME      = "SoundVaultTagFixer"
ACOUSTID_APP_VERSION   = "1.0.0"
SUPPORTED_EXTS         = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav"}

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
        best_score, _, best_title, best_artist = peek[0]
        return {"title": best_title, "artist": best_artist, "score": best_score}

    except acoustid.NoBackendError:
        log_callback("Chromaprint library/tool not found")
    except acoustid.FingerprintGenerationError:
        log_callback(f"Failed to fingerprint: {path}")
    except acoustid.WebServiceError as exc:
        log_callback(f"AcoustID request failed: {exc}")
    return None

def update_tags(path, new_tags, log_callback):
    """Write missing artist/title tags to the file. Returns True if saved."""
    audio = MutagenFile(path, easy=True)
    if audio is None:
        return False
    changed = False
    if new_tags.get("artist") and not audio.tags.get("artist"):
        audio.tags["artist"] = [new_tags["artist"]]
        changed = True
    if new_tags.get("title") and not audio.tags.get("title"):
        audio.tags["title"] = [new_tags["title"]]
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
def collect_tag_proposals(target, log_callback=None):
    """Walk files under ``target`` and return tagging proposals."""
    if log_callback is None:
        def log_callback(msg):
            print(msg)

    if not ACOUSTID_API_KEY:
        raise RuntimeError("ACOUSTID_API_KEY not configured")

    proposals = []
    for f in find_files(target):
        if is_remix(f):
            continue

        log_callback(f"Fingerprinting {f}")
        result = query_acoustid(f, log_callback)
        if not result:
            continue

        score = result["score"]
        if score < MIN_INTERACTIVE_SCORE:
            continue

        audio = MutagenFile(f, easy=True)
        old_artist = (audio.tags.get("artist") or [None])[0] if audio and audio.tags else None
        old_title = (audio.tags.get("title") or [None])[0] if audio and audio.tags else None

        proposals.append({
            "file": f,
            "old_artist": old_artist,
            "old_title": old_title,
            "new_artist": result["artist"],
            "new_title": result["title"],
            "score": score,
        })

    return proposals


def apply_tag_proposals(proposals, log_callback=None):
    """Write the tags for each proposal and return the count updated."""
    if log_callback is None:
        def log_callback(msg):
            print(msg)

    updated = 0
    for p in proposals:
        path = p["file"]
        tags = {"artist": p["new_artist"], "title": p["new_title"]}
        if update_tags(path, tags, log_callback):
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

    proposals = collect_tag_proposals(target, log_callback)

    if interactive:
        filtered = []
        for p in proposals:
            apply_change = prompt_user_about_tags(
                p["file"], p["old_artist"], p["old_title"],
                {"artist": p["new_artist"], "title": p["new_title"]}
            )
            if apply_change:
                filtered.append(p)
        proposals = filtered

    updated = apply_tag_proposals(proposals, log_callback)

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
