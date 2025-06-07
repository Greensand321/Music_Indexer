cat > README.md << 'EOF'
# Music Indexer / SoundVault

This project contains utilities and a GUI for organizing, tagging, and managing music files. Under the hood, it uses:

- [`mutagen`](https://mutagen.readthedocs.io/) to read/write audio metadata
- [`musicbrainzngs`](https://musicbrainzngs.readthedocs.io/) to query MusicBrainz for tags
- [`pydub`](https://github.com/jiaaro/pydub) + FFmpeg for audio previewing (“Sample Song Highlight”)
- [`pyacoustid`](https://github.com/beetbox/pyacoustid) for AcoustID-based tag fixing
- [`librosa`](https://librosa.org/) (optional) for BPM estimation

## Prerequisites

1. **Python 3.10 – 3.12** (Python 3.13 lacks `audioop`, so “Sample Song Highlight” will be disabled)  
2. **FFmpeg** on your PATH (for `pydub` to work)  
   - On Windows, you can install via:  
     - **Winget**:  
       ```powershell
       winget install "FFmpeg (Essentials Build)"
       ```  
     - Or download an Essentials build from https://www.gyan.dev/ffmpeg/builds/, unzip, and add its `bin\` folder to your system PATH.  
   - Verify with:  
     ```bash
     ffmpeg -version
     ```  
3. **Chromaprint’s `fpcalc`** on your PATH (for the tag-fixer)  
4. **Git** (to clone/pull the repository)

## Installation

Install the Python dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Starting the GUI

From the project folder run:

```bash
python main_gui.py
```

A window will open. Use **File → Open Library…** to choose the folder you want to index or import music into.

## Recommended folder layout

The HTML documentation proposes collecting all tools under a single `soundvault/` directory. The structure looks like:

```
soundvault/
  indexer/
    music_indexer_api.py
  html_utils/
    folder_to_html.py
    folder_to_html_with_genre.py
    list_genres.py
    generate_library_index.py
  genre_updater.py
  playlist_generator.py  (future)
  soundvault.py          (CLI driver)
  requirements.txt
  docs/
    project_documentation.html
    part2_project_documentation.html (if split)
    README.md
    ...
```

This repository currently contains the early versions of these scripts along with the project documentation. See [`docs/project_documentation.html`](docs/project_documentation.html) for detailed information on each component.
EOF
