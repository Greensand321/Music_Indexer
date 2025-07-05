import os
import shutil
import tempfile
import hashlib
from typing import Callable, Dict, Any
from mutagen import File as MutagenFile
from mutagen.id3 import ID3NoHeaderError

from validator import validate_soundvault_structure
import music_indexer_api as idx

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}


def import_new_files(
    vault_root: str,
    import_folder: str,
    dry_run: bool = False,
    estimate_bpm: bool = False,
    log_callback: Callable[[str], None] | None = None,
    enable_phase_c: bool = False,
) -> Dict[str, Any]:
    """Import new audio files into a SoundVault library."""
    if log_callback is None:
        def log_callback(msg: str) -> None:
            pass

    valid, errors = validate_soundvault_structure(vault_root)
    if not valid:
        raise ValueError("Invalid SoundVault root:\n" + "\n".join(errors))

    new_files = []
    for dirpath, _, files in os.walk(import_folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                new_files.append(os.path.join(dirpath, fname))

    if not new_files:
        log_callback("No audio files found to import.")
        return {"moved": 0, "html": None, "dry_run": dry_run}

    log_callback(f"Found {len(new_files)} new audio files to import.")

    file_info = {}
    for path in new_files:
        tags = idx.get_tags(path)
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
                pics = getattr(audio_file, "pictures", [])
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

        if estimate_bpm and not tags.get("bpm"):
            try:
                import librosa
                y, sr = librosa.load(path, mono=True)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tags["bpm"] = int(round(float(tempo)))
            except Exception:
                tags["bpm"] = None

        file_info[path] = tags

    music_root = vault_root
    temp_dir = tempfile.mkdtemp(dir=music_root, prefix="import_tmp_")
    orig_to_temp = {}
    for src in new_files:
        temp_path = os.path.join(temp_dir, os.path.basename(src))
        shutil.copy2(src, temp_path)
        orig_to_temp[src] = temp_path

    moves, tag_index, decision_log = idx.compute_moves_and_tag_index(vault_root, log_callback, coord=None)

    import_moves = {}
    for orig, tmp in orig_to_temp.items():
        if tmp in moves:
            import_moves[orig] = moves[tmp]

    preview_html = os.path.join(import_folder, "import_preview.html")

    if dry_run:
        idx.build_dry_run_html(
            vault_root, preview_html, log_callback, enable_phase_c=enable_phase_c
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"moved": 0, "html": preview_html, "dry_run": True}

    moved = 0
    errors = []
    for src, dest in import_moves.items():
        parent_dir = os.path.dirname(dest)
        try:
            os.makedirs(parent_dir, exist_ok=True)
            shutil.move(src, dest)
            moved += 1
        except Exception as e:
            errors.append(f"Failed to move {src} → {dest}: {e}")

    if errors:
        for err in errors:
            log_callback(f"! {err}")

    shutil.rmtree(temp_dir, ignore_errors=True)

    log_path = os.path.join(vault_root, "import_log.txt")
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            for src, dest in import_moves.items():
                lf.write(f"{os.path.basename(src)} → {os.path.relpath(dest, music_root)}\n")
    except Exception:
        pass

    idx.build_dry_run_html(
        vault_root, preview_html, log_callback, enable_phase_c=enable_phase_c
    )

    return {"moved": moved, "html": preview_html, "dry_run": False, "errors": errors}
