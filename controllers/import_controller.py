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


def build_import_preview_html(
    root_path: str, moves: Dict[str, str], output_html_path: str
) -> None:
    """Write a minimal HTML preview listing only newly added tracks."""

    lines = ["<h2>Import Preview</h2>", "<pre>"]
    lines.append(
        f"<span class=\"folder\">{idx.sanitize(os.path.basename(root_path))}/</span>"
    )
    dests = set(moves.values())
    tree_nodes = set()
    for dest in dests:
        parts = os.path.relpath(dest, root_path).split(os.sep)
        for i in range(1, len(parts) + 1):
            tree_nodes.add(os.path.join(root_path, *parts[:i]))

    for node in sorted(tree_nodes, key=lambda p: os.path.relpath(p, root_path)):
        rel = os.path.relpath(node, root_path)
        depth = rel.count(os.sep)
        indent = "    " * depth
        if node in dests:
            lines.append(
                f"{indent}<span class=\"song\">- {idx.sanitize(os.path.basename(node))}</span>"
            )
        else:
            lines.append(
                f"{indent}<span class=\"folder\">{idx.sanitize(os.path.basename(node))}/</span>"
            )
    lines.append("</pre>")
    html_body = "\n".join(lines)

    with open(output_html_path, "w", encoding="utf-8") as out:
        out.write(
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n"
        )
        out.write(
            f"  <title>Import Preview – {idx.sanitize(os.path.basename(root_path))}</title>\n"
        )
        out.write(
            "  <style>\n    body { background:#2e3440; color:#d8dee9; font-family:'Courier New', monospace; }\n"
            "    pre  { font-size:14px; }\n    .folder { color:#81a1c1; }\n    .song   { color:#a3be8c; }\n  </style>\n"
            "</head>\n<body>\n"
        )
        out.write(html_body)
        if html_body and not html_body.endswith("\n"):
            out.write("\n")
        out.write("</body>\n</html>\n")


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

    docs_dir = os.path.join(vault_root, "Docs")
    db_path = os.path.join(docs_dir, ".soundvault.db")
    from cache_prewarmer import prewarm_cache

    def _compute(path: str) -> tuple[int | None, str | None]:
        try:
            import acoustid
            return acoustid.fingerprint_file(path)
        except Exception:
            return None, None

    prewarm_cache(list(orig_to_temp.values()), db_path, _compute)

    moves, tag_index, decision_log = idx.compute_moves_and_tag_index(vault_root, log_callback, coord=None)

    import_moves = {}
    for orig, tmp in orig_to_temp.items():
        if tmp in moves:
            import_moves[orig] = moves[tmp]

    preview_html = os.path.join(import_folder, "import_preview.html")

    if dry_run:
        build_import_preview_html(vault_root, import_moves, preview_html)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"moved": 0, "html": preview_html, "dry_run": True}

    moved = 0
    errors = []
    successful_moves: Dict[str, str] = {}
    for src, dest in import_moves.items():
        parent_dir = os.path.dirname(dest)
        try:
            os.makedirs(parent_dir, exist_ok=True)
            shutil.move(src, dest)
            moved += 1
            successful_moves[src] = dest
        except Exception as e:
            errors.append(f"Failed to move {src} → {dest}: {e}")

    if errors:
        for err in errors:
            log_callback(f"! {err}")

    shutil.rmtree(temp_dir, ignore_errors=True)

    log_path = os.path.join(vault_root, "import_log.txt")
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            for src, dest in successful_moves.items():
                lf.write(
                    f"{os.path.basename(src)} → {os.path.relpath(dest, music_root)}\n"
                )
    except Exception:
        pass

    build_import_preview_html(vault_root, successful_moves, preview_html)

    return {
        "moved": moved,
        "html": preview_html,
        "dry_run": False,
        "errors": errors,
    }
