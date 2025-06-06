# Music Indexer / SoundVault

This project contains utilities for organizing and tagging music files. The programs rely on the
[mutagen](https://mutagen.readthedocs.io/) library to read and write audio metadata and
[musicbrainzngs](https://musicbrainzngs.readthedocs.io/) to query MusicBrainz.

## Installation

Install the Python dependencies in your environment:

```bash
pip install mutagen musicbrainzngs
```

## Starting the GUI

Navigate to the project folder and run:

```bash
python main_gui.py
```

A simple window will open that lets you choose your SoundVault folder and run the importer.

## Recommended folder layout

The HTML documentation proposes collecting all tools under a single `soundvault/` directory.
The structure looks like:

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
  soundvault.py         (CLI driver)
  requirements.txt
  docs/
    project_documentation.html
    part2_project_documentation.html (if split)
    README.md
    ...
```

This repository currently contains the early versions of these scripts along with the project
documentation.
