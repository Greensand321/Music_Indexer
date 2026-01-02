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

The **Duplicate Finder** tab now opens the redesigned shell for spotting
duplicates. Keep the Music Indexer package installed (e.g. `pip install .`) so
the duplicate detection backend remains available as the new UI rolls out.

### Optional: Essentia audio engine

Essentia can be used instead of `librosa` for tempo and feature extraction. It
is optional—stick with `librosa` if you don't need it—but enables faster C++
implementations when available.

- **Prerequisites** (Linux builds compile C++ code and can take several
  minutes):
  - Debian/Ubuntu: `sudo apt-get install build-essential libfftw3-dev liblapack-dev libblas-dev libyaml-dev libtag1-dev libchromaprint-dev libsamplerate0-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev`
  - macOS: `brew install essentia` (installs prebuilt formula with dependencies)
  - Windows: no official wheel; use WSL/Linux if you need Essentia.
- **Install** (after prerequisites):
  ```bash
  pip install essentia==2.1b6
  ```

Expect longer build times on Linux the first time you install Essentia. If you
prefer the pure-Python stack, you can continue using `librosa` without this
extra dependency.

## Quickstart

```bash
python main_gui.py
```

1. **Open** your library folder
2. Use the **Indexer** tab to dedupe, detect near duplicates, and move files
3. **Fix Tags** via the AcoustID menu (now supports multiple metadata services)
4. **Generate Playlists** from your folder structure
5. **Clustered Playlists** (interactive K-Means/HDBSCAN) via the Tools ▸ Clustered Playlists menu
6. **Smart Playlist Engine** with tempo/energy buckets
7. **Auto‑DJ** mode builds seamless playlists starting from any song
8. **Tidal-dl Sync** can upgrade low-quality files to FLAC
9. **Library Duplicate Scan** finds duplicate tracks after you drop new songs directly into your library
10. **Cross-Album Scan** optionally finds duplicates appearing on multiple albums
11. Use the **Theme** dropdown and **Help** tab for assistance
12. Launch the **Duplicate Finder** tab to open the updated shell for spotting duplicates
13. Use **Tools → Similarity Inspector** to compare two files and see fingerprint distance details

### Playlist generator feedback

When you start a playlist job (tempo/energy buckets, Auto‑DJ, or auto‑creating clustered playlists), the app automatically switches to the **Log** tab. The tab shows timestamped messages from the playlist helpers (feature gathering, similarity calculations, and playlist writes) so you can see that background work is running without waiting for a popup.

Cluster generation writes progress into `<method>_log.txt` inside your library so you can review the steps later.

### Windows Long Paths

The indexer automatically prefixes file paths with `\\?\` on Windows, allowing it to work with directories deeper than the classic 260-character limit.

## Threading

Long running actions such as indexing, tag fixing and library sync operations are executed in daemon threads. GUI updates from these background tasks are scheduled using Tkinter's `after` method so message boxes and progress indicators always run on the main thread.

## Configuration

User settings are stored in `~/.soundvault_config.json`. To tweak the fuzzy fingerprint
threshold used during deduplication, add a value like:

```json
{
  "fuzzy_fp_threshold": 0.1
}
```

Lower values require more similar fingerprints.

Per-format fingerprint thresholds used by the sync matcher can also be
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

The same dictionary can be provided to ``library_sync.compare_libraries`` via
the optional ``thresholds`` parameter to control how strictly fingerprints are
matched when scanning two libraries.

You can also store the path to your library for automatic scanning:

```json
{
  "library_root": "/path/to/your/Music"
}
```

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

## Duplicate Finder (Redesigned)

The Duplicate Finder has been rebuilt into a review-first workflow that makes it
easy to preview and execute deduplication safely.

- **Scan Library** builds a fingerprint plan and summarizes duplicate groups.
- **Preview** writes `Docs/duplicate_preview.json` and
  `Docs/duplicate_preview.html` so you can review every group before changes.
- **Execute** applies the plan, writes a detailed HTML report under
  `Docs/duplicate_execution_reports/`, and updates playlists when enabled.
- **Group dispositions** let you retain, quarantine, or delete losers per group
  while keeping global defaults for everything else.
- **Review-required groups** block execution until resolved or overridden.
- **Thresholds** controls let you tune exact/near matching as well as
  fingerprint windowing and silence trimming for tough cases.

Duplicates are quarantined into `Quarantine/` by default; you can switch to
retain-in-place or delete (with confirmation) from the main controls.

## Similarity Inspector

The Similarity Inspector is a targeted tool for understanding why two tracks
match (or do not match) during duplicate detection.

- Launch from **Tools → Similarity Inspector…**.
- Select two files, optionally override fingerprint offsets, trimming, and
  thresholds, then run the inspection.
- The report shows codec, duration, raw fingerprint distance, the effective
  near-duplicate threshold (including mixed-codec adjustments), and the verdict.
- Every run writes a timestamped report to `Docs/` inside the selected library.

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

- Metadata Plugins (Discogs, Spotify)

See [`docs/project_documentation.html`](docs/project_documentation.html) for technical details.
