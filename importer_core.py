def scan_and_import(vault_root, import_folder, dry_run=False, estimate_bpm=False, log_callback=None):
    # 1) validate vault_root
    # 2) scan import_folder → new_files list
    # 3) read metadata & cover_hash for each → build a dict
    # 4) (optional) estimate missing BPM if enabled
    # 5) decide destination for each using the same logic as MusicIndexer
    # 6) if dry_run: build HTML preview & return summary without moving
    # 7) else: move/rename files transactionally, update logs, return summary
