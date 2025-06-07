import os
import sys
import argparse
import acoustid
from mutagen import File as MutagenFile

# AcoustID credentials and client identification
ACOUSTID_API_KEY     = "eBOqCZhyAx"
ACOUSTID_APP_NAME    = "SoundVaultTagFixer"
ACOUSTID_APP_VERSION = "1.0.0"

SUPPORTED_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav"}

def is_remix(audio_path):
    """Return True if the filename or existing title suggests a remix."""
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
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                audio_files.append(os.path.join(dirpath, fname))
    return audio_files

def query_acoustid(path, log_callback):
    """
    Return tags from AcoustID if the best fingerprint match meets your threshold.
    Also logs the top‐5 candidate scores for debugging.
    """
    try:
        status, results = acoustid.match(
            ACOUSTID_API_KEY,
            path
        )
        if not results:
            log_callback("  No matches at all")
            return None

        # Debug: show top‐5 scores
        for i, (score, rid, title, artist) in enumerate(results[:5], start=1):
            log_callback(f"  [{i}] score={score:.4f} → “{artist} – {title}”")

        best_score, _, best_title, best_artist = results[0]
        # Change 1.0 → 0.95 here if you want a looser match
        if best_score >= 1.0:
            return {"title": best_title, "artist": best_artist}

        log_callback(f"  Best score {best_score:.4f} < 1.00 → skipping")
        return None

    except acoustid.NoBackendError:
        log_callback("Chromaprint library/tool not found")
    except acoustid.FingerprintGenerationError:
        log_callback(f"Failed to fingerprint: {path}")
    except acoustid.WebServiceError as exc:
        log_callback(f"AcoustID request failed: {exc}")

    return None

def update_tags(path, new_tags, log_callback):
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

def fix_tags(target, log_callback=None, interactive=False):
    """Fill missing tags for files in target using AcoustID."""
    if log_callback is None:
        def log_callback(msg):
            print(msg)

    if not ACOUSTID_API_KEY:
        raise RuntimeError("ACOUSTID_API_KEY not configured")

    files = find_files(target)
    if not files:
        log_callback("No audio files found.")
        return {"processed": 0, "updated": 0}

    updated = 0
    for f in files:
        if is_remix(f):
            log_callback(f"Skipping remix {f}")
            continue
        log_callback(f"Processing {f}...")
        tags = query_acoustid(f, log_callback)
        if not tags:
            log_callback("  No perfect match found.")
            continue

        audio = MutagenFile(f, easy=True)
        old_artist = None
        old_title  = None
        if audio and audio.tags:
            old_artist = (audio.tags.get("artist") or [None])[0]
            old_title  = (audio.tags.get("title")  or [None])[0]

        new_artist = tags["artist"]
        new_title  = tags["title"]

        if interactive:
            print(f"\nFile: {f}")
            print(f"{'Field':10} │ {'Current':30} │ {'New from AcoustID':30}")
            print("-" * 75)
            print(f"{'Artist':10} │ {old_artist or '—':30} │ {new_artist:30}")
            print(f"{'Title':10} │ {old_title  or '—':30} │ {new_title:30}")
            print()
            resp = input("Apply these changes? [y/N]: ").strip().lower()
            if resp != 'y':
                log_callback("  Skipped.")
                continue

        if update_tags(f, tags, log_callback):
            updated += 1

    return {"processed": len(files), "updated": updated}

def main():
    parser = argparse.ArgumentParser(
        description="Fill missing audio tags using AcoustID"
    )
    parser.add_argument("target", help="file or folder to process")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="show proposed tag changes and ask before writing",
    )
    args = parser.parse_args()

    try:
        summary = fix_tags(args.target, interactive=args.interactive)
        print(
            f"\nProcessed {summary['processed']} files, updated {summary['updated']}."
        )
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    main()
