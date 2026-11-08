# validator.py

import os

# The top-level folders the Indexer sorts tracks into (music_indexer_api):
# most artists go to "By Artist/", artists with only a few tracks go to
# "By Year/". A library indexed by AlphaDEX therefore has at least one of
# them, but not necessarily both — a library with no "rare" artists never
# gets a "By Year/" folder.
INDEXED_LAYOUT_FOLDERS = ("By Artist", "By Year")


def _has_layout(folder):
    return any(os.path.isdir(os.path.join(folder, name)) for name in INDEXED_LAYOUT_FOLDERS)


def validate_soundvault_structure(root_path):
    """
    Check that the selected folder is already in SoundVault (AlphaDEX-indexed)
    format, ignoring what the folder itself is named. Return (True, []) if
    valid, or (False, [errors]) if not.

    Valid if either:
      1) root_path contains 'By Artist/' or 'By Year/' (or both)
      2) root_path/Music contains 'By Artist/' or 'By Year/' (or both)

    Otherwise, invalid.
    """
    errors = []

    # No folder at all: "" used to be joined onto the working directory (so
    # a "By Artist/" there made it "valid") and None raised a TypeError.
    if root_path is None or not str(root_path).strip():
        return False, ["No library folder was selected."]
    root_path = str(root_path)
    if not os.path.isdir(root_path):
        return False, [f"The folder does not exist (or is not a folder): {root_path}"]

    # Case A: does root_path directly contain the indexed layout?
    if _has_layout(root_path):
        return True, []

    # Case B: does root_path/Music contain it?
    music_sub = os.path.join(root_path, "Music")
    if os.path.isdir(music_sub) and _has_layout(music_sub):
        return True, []

    # Otherwise, it’s not a valid SoundVault
    errors.append("This folder has not been organised by the AlphaDEX Indexer yet.")
    errors.append("An indexed library contains 'By Artist/' and/or 'By Year/'. Either:")
    errors.append("  • Select the folder that directly contains them,")
    errors.append("  • Or select a parent folder whose 'Music/' child contains them,")
    errors.append("  • Or run the Indexer on this folder to create that layout.")
    return False, errors
