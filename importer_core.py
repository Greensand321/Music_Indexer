"""Importer entry point.

This function scans a folder of new audio files and places them into an
existing SoundVault.  It mirrors the decision logic of the music indexer so the
new files land in the correct ``By Artist``/``By Year`` structure.

Parameters
----------
vault_root : str
    Path to an existing SoundVault root.
import_folder : str
    Folder containing new audio files to import.
dry_run : bool, optional
    If True, nothing is moved and an HTML preview of the resulting library
    layout is produced.
estimate_bpm : bool, optional
    If True, attempt to fill in missing BPM information using ``librosa``.
log_callback : callable, optional
    Function that accepts a single string for progress logging.  If ``None`` a
    no-op logger is used.

Returns
-------
dict
    Summary dictionary with keys ``moved`` (number of files moved), ``html``
    (path to the dry-run preview if generated) and ``dry_run`` (boolean).
"""

import os
import shutil
import tempfile
import hashlib
from mutagen import File as MutagenFile
from mutagen.id3 import ID3NoHeaderError

from validator import validate_soundvault_structure
import music_indexer_api as idx


def scan_and_import(vault_root, import_folder, dry_run=False, estimate_bpm=False, log_callback=None):
    if log_callback is None:
        def log_callback(msg):
            pass

    # ─── 1) Validate the vault ───────────────────────────────────────────────
    valid, errors = validate_soundvault_structure(vault_root)
    if not valid:
        raise ValueError("Invalid SoundVault root:\n" + "\n".join(errors))

    music_root = os.path.join(vault_root, "Music") if os.path.isdir(os.path.join(vault_root, "Music")) else vault_root
    supported_exts = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}

    # ─── 2) Scan import folder for audio files ───────────────────────────────
    new_files = []
    for dirpath, _, files in os.walk(import_folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_exts:
                new_files.append(os.path.join(dirpath, fname))

    if not new_files:
        log_callback("No audio files found to import.")
        return {"moved": 0, "html": None, "dry_run": dry_run}

    log_callback(f"Found {len(new_files)} new audio files to import.")

    # ─── 3) Collect metadata and cover hashes ────────────────────────────────
    file_info = {}
    for path in new_files:
        tags = idx.get_tags(path)

        # Extract embedded cover art hash (same logic as indexer)
        cover_hash = None
        try:
            audio_file = MutagenFile(path)
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
        except ID3NoHeaderError:
            pass
        except Exception:
            pass

        tags["cover_hash"] = cover_hash

        # ─── 4) Optionally estimate BPM if missing ───────────────────────────
        if estimate_bpm and not tags.get("bpm"):
            try:
                import librosa  # heavy but optional dependency
                y, sr = librosa.load(path, mono=True)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tags["bpm"] = int(round(float(tempo)))
            except Exception:
                tags["bpm"] = None

        file_info[path] = tags

    # ─── 5) Determine final destinations using indexer logic ─────────────────
    temp_dir = tempfile.mkdtemp(dir=music_root, prefix="import_tmp_")
    orig_to_temp = {}
    for src in new_files:
        temp_path = os.path.join(temp_dir, os.path.basename(src))
        shutil.copy2(src, temp_path)
        orig_to_temp[src] = temp_path

    moves, tag_index, decision_log = idx.compute_moves_and_tag_index(vault_root, log_callback)

    # Filter moves so we only act on our imported temp files
    import_moves = {}
    for orig, tmp in orig_to_temp.items():
        if tmp in moves:
            import_moves[orig] = moves[tmp]

    preview_html = os.path.join(import_folder, "import_preview.html")

    # ─── 6) Handle dry-run preview ───────────────────────────────────────────
    if dry_run:
        idx.build_dry_run_html(vault_root, preview_html, log_callback)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"moved": 0, "html": preview_html, "dry_run": True}

    # ─── 7) Move files to destination and log ────────────────────────────────
    moved = 0
    errors = []
    for src, dest in import_moves.items():
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)
            moved += 1
        except Exception as e:
            errors.append(f"Failed to move {src} → {dest}: {e}")

    shutil.rmtree(temp_dir, ignore_errors=True)

    # Write import log
    log_path = os.path.join(vault_root, "import_log.txt")
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            for src, dest in import_moves.items():
                lf.write(f"{os.path.basename(src)} → {os.path.relpath(dest, music_root)}\n")
    except Exception:
        pass

    if errors:
        for err in errors:
            log_callback(f"! {err}")

    # Generate preview HTML of the final library state after import
    idx.build_dry_run_html(vault_root, preview_html, log_callback)

    return {"moved": moved, "html": preview_html, "dry_run": False, "errors": errors}
