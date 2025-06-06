"""Core logic for importing new songs into a SoundVault library."""

import os
import shutil

from validator import validate_soundvault_structure


def scan_and_import(vault_root, import_folder, dry_run=False, estimate_bpm=False,
                    log_callback=None):
    """Scan ``import_folder`` and move any supported audio files into the
    SoundVault at ``vault_root``.

    This is a minimal placeholder implementation. It validates the target
    ``vault_root`` and collects all audio files from ``import_folder``. When
    ``dry_run`` is ``True`` the function simply reports how many files would be
    imported. Otherwise the files are moved into an ``Incoming`` folder within
    the SoundVault. The return value is a summary dictionary.
    """

    if log_callback is None:
        def log_callback(msg):
            pass

    # Validate the vault structure
    valid, errors = validate_soundvault_structure(vault_root)
    if not valid:
        raise ValueError("Invalid SoundVault root:\n" + "\n".join(errors))

    SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}

    new_files = []
    for dirpath, _, files in os.walk(import_folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                new_files.append(os.path.join(dirpath, fname))

    summary = {"found": len(new_files), "imported": 0, "dry_run": dry_run}

    if dry_run:
        log_callback(f"Found {len(new_files)} importable files (dry run).")
        return summary

    incoming = os.path.join(vault_root, "Incoming")
    os.makedirs(incoming, exist_ok=True)

    for src in new_files:
        dest = os.path.join(incoming, os.path.basename(src))
        try:
            shutil.move(src, dest)
            summary["imported"] += 1
        except Exception as e:
            log_callback(f"Failed to move {src} → {dest}: {e}")

    log_callback(f"Imported {summary['imported']} files to {incoming}.")
    return summary
