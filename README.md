# SoundVault Music Indexer

SoundVault organizes large music libraries. It deduplicates tracks, fixes tags via AcoustID, normalizes genres, and generates playlists while keeping your folder structure intact.

## Prerequisites

- **Python 3.11+** (use [conda](https://docs.conda.io/en/latest/miniconda.html) or `venv`)
- **Git** command line
  ```bash
  git clone --recurse-submodules https://example.com/yourrepo.git
  ```
- **FFmpeg** installed and on your `PATH`
- **llama.exe** command line for LLM features

## Installation

Create and activate a virtual environment then install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # or "Scripts\activate" on Windows
pip install -r requirements.txt
```

The indexer will exit with an error if the real `mutagen` package is missing,
so ensure all dependencies are installed before running.

## Quickstart

```bash
python main_gui.py
```

1. **Open** your library folder
2. Use the **Indexer** tab to dedupe, detect near duplicates, and move files
3. **Fix Tags** via the AcoustID menu (now supports multiple metadata services)
4. **Generate Playlists** from your folder structure
5. **Clustered Playlists** (interactive K-Means/HDBSCAN) via the Tools ▸ Clustered Playlists menu
6. **Tidal-dl Sync** can upgrade low-quality files to FLAC
7. Use the **Theme** dropdown and **Help** tab for assistance

Cluster generation writes progress into `<method>_log.txt` inside your library so you can review the steps later.

### Windows Long Paths

The indexer automatically prefixes file paths with `\\?\` on Windows, allowing it to work with directories deeper than the classic 260-character limit.

## Threading

Long running actions such as indexing, tag fixing and tidal-dl comparison are executed in daemon threads. GUI updates from these background tasks are scheduled using Tkinter's `after` method so message boxes and progress indicators always run on the main thread.

## Configuration

User settings are stored in `~/.soundvault_config.json`. To tweak the fuzzy fingerprint
threshold used during deduplication, add a value like:

```json
{
  "fuzzy_fp_threshold": 0.1
}
```

Lower values require more similar fingerprints.

Per-format fingerprint thresholds used by the tidal-dl matcher can also be
configured. Add a `format_fp_thresholds` section with extension keys:

```json
{
  "format_fp_thresholds": {
    "default": 0.3,
    ".flac": 0.3,
    ".mp3": 0.2,
    ".aac": 0.25
  }
}
```

Values are floating point distances – lower numbers require closer matches.

The configuration file also stores your selected metadata service and API key.
You can update these via **Settings → Metadata Services** in the GUI:

```json
{
  "metadata_service": "AcoustID",
  "metadata_api_key": "YOUR_KEY"
}
```
These values are updated whenever you save the Metadata Services settings.
Testing the connection or saving will persist your selections for future runs.

MusicBrainz requests require a valid User-Agent string containing your
application name, version and contact email.

## File Overview

The codebase is organized into a handful of key modules:

```
main_gui.py               - Tkinter entry point for the desktop app
music_indexer_api.py      - Core scanning, dedupe and relocation logic
playlist_generator.py     - `.m3u` playlist creation helpers
clustered_playlists.py    - Feature extraction and clustering algorithms
cluster_graph_panel.py    - Interactive scatter plot for clustered playlists
fingerprint_generator.py  - Build AcoustID fingerprint database
fingerprint_cache.py      - Persistent fingerprint cache
near_duplicate_detector.py - Fuzzy near-duplicate detection helpers
tag_fixer.py              - Tag fixing engine using plugin metadata
update_genres.py          - Batch genre tag updater via MusicBrainz
tidal_sync.py             - Sync tidal-dl downloads to upgrade your library
validator.py              - Verify SoundVault folder layout
config.py                 - Read/write persistent configuration
mutagen_stub/             - Minimal fallback used by the tests

controllers/
  library_controller.py        - Handle library selection and persistence
  import_controller.py         - Import new audio files
  tagfix_controller.py         - Apply tag proposals and update DB
  normalize_controller.py      - AI genre normalization workflow
  cluster_controller.py        - Gather tracks and run clustering
  library_index_controller.py  - Build HTML index of your library
  highlight_controller.py      - Play short audio snippets
  genre_list_controller.py     - Scan library for unique genres
  playlist_controller.py       - Playlist export placeholder

plugins/
  base.py               - Metadata plugin interface
  acoustid_plugin.py    - Metadata lookup via selected service
  assistant_plugin.py   - LLM helper integration
  discogs.py            - Discogs metadata stub
  lastfm.py             - Fetch genres from Last.fm
  spotify.py            - Spotify metadata stub
  test_plugin.py        - Example plugin

bindings/    - C++/pybind11 wrapper for llama binaries
docs/        - Additional project documentation
third_party/ - Prebuilt llama executables
```

## Roadmap (Upcoming Features)

These items are currently under development and not yet part of the stable release.

- Smarter Playlist Engine
- Tempo/Energy Buckets
- "More Like This" suggestions
- Deej-AI / Auto-DJ Integration
- Metadata Plugins (Discogs, Spotify)

See [`docs/project_documentation.html`](docs/project_documentation.html) for technical details.
