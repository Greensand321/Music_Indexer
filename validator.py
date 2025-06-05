# validator.py

import os

def validate_soundvault_structure(root_path):
    """
    Check that the selected folder is already in SoundVault format, ignoring what
    the folder itself is named. Return (True, []) if valid, or (False, [errors]) if not.

    Valid if either:
      1) root_path/By Artist  AND  root_path/By Year
      2) root_path/Music/By Artist  AND  root_path/Music/By Year

    Otherwise, invalid.
    """
    errors = []

    # Case A: Does root_path directly contain "By Artist" and "By Year"?
    by_artist_direct = os.path.join(root_path, "By Artist")
    by_year_direct   = os.path.join(root_path, "By Year")
    if os.path.isdir(by_artist_direct) and os.path.isdir(by_year_direct):
        return True, []

    # Case B: Does root_path/Music contain "By Artist" and "By Year"?
    music_sub = os.path.join(root_path, "Music")
    by_artist_music = os.path.join(music_sub, "By Artist")
    by_year_music   = os.path.join(music_sub, "By Year")
    if os.path.isdir(music_sub) and os.path.isdir(by_artist_music) and os.path.isdir(by_year_music):
        return True, []

    # Otherwise, it’s not a valid SoundVault
    errors.append("Missing required subfolders for a SoundVault.")
    errors.append("Either:")
    errors.append("  • Select the folder that directly contains 'By Artist/' and 'By Year/',")
    errors.append("  • Or select a parent folder whose 'Music/' child contains 'By Artist/' & 'By Year/'.")
    return False, errors
