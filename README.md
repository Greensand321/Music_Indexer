<img width="1024" height="432" alt="AlphaDEX" src="https://github.com/user-attachments/assets/d8eb11c2-6a50-47d1-8932-77a53cee9b17" />

# AlphaDEX (formerly SoundVault) a Music Indexer

Welcome to AlphaDEX (formerly SoundVault)! If you have a library of songs with mixed codecs, duplicates, messy metadata, and disorganized folders, AlphaDEX is built to fix all of it. It is designed for casual users and enthusiasts alike, and the tools inside are the result of years of frustration born from too many songs, too little space, and no desire to manually sort thousands of files. The current feature set represents 500+ hours of focused work over roughly six months, with the original vision first outlined more than five years ago when the music collection began. The advent of AI made it possible to extend those ideas and ship the program you see here.

This README mirrors the project documentation and provides:
- A friendly overview of the core programs.
- A quick path to get running.
- Supporting tools and configuration details for deeper work.

## Core programs (start here)

These are the main workflows described in the documentation. Each is followed by
where it lives in the app (written **Page → Section** when a page has several
sections):

1. **Music Indexer** (**Indexer**): preview-first workflows to organize, move, and rename your library with a full HTML report before committing changes.
2. **Duplicate Finder** (**Duplicates → Find duplicates**): review-first deduplication with fingerprinting, group-by-group decisions, and safety-focused execution reports.
3. **Compare two files** (**Duplicates → Compare two files**, formerly the Similarity Inspector): a diagnostic tool to understand why two tracks match (or do not match) during duplicate detection.
4. **Library Sync** (**Library Sync**): compare an incoming folder with your library, review the matches, build a plan, and preview or execute copy/move actions.

## Supporting tools (roughly simple → advanced)

1. **Tag Fixer** (**Metadata → Tag Fixer**): repair metadata using AcoustID and other services.
2. **Genre tools** (**Metadata → Fill genres** and **Metadata → Normalize genres**, formerly the Genre Normalizer): fill missing genres from MusicBrainz, or collapse messy genre spellings with a mapping you build; **Metadata → Undo changes** rolls back any tag apply.
3. **File Cleanup** (**File Cleanup**): remove " (1)" / " - Copy" suffixes from file names, previewed and undoable.
4. **Playlist Generator** (**Playlists**): build `.m3u` playlists by folder, tempo/energy and Auto‑DJ flow, and repair broken playlists.
5. **Clustering** (**Sound Map → Analyse**, formerly Clustered Playlists): run K-Means/HDBSCAN clustering; **Save cluster playlists…** previews and saves one playlist per cluster.
6. **The map** (**Sound Map → Map**, formerly the Visual Music Graph): an interactive scatter plot of the clusters that lets you explore your library as a map, plus a 3-D view in your browser.
7. **Opus mirror** (**Mirror → Build the mirror** and **Mirror → Verify the mirror**): a portable FLAC → Opus copy of the library, and a check that converted files kept their tags and art.
8. **Exports** (**Exports**): artist/title and codec lists of the library, written to `Docs/`.

## Prerequisites

- **Python 3.11+** (use [conda](https://docs.conda.io/en/latest/miniconda.html) or `venv`)
- **Git** command line for cloning the repo
- **FFmpeg** installed and on your `PATH` (required for audio analysis)
- **VLC / libVLC** for in-app playback (recommended; required for audio preview features)

## Installation

Create and activate a virtual environment then install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # or "Scripts\activate" on Windows
pip install -r requirements.txt
```

The indexer will exit with an error if the real `mutagen` package is missing,
so ensure all dependencies are installed before running.

If you run from source, keep this repo on your Python path (run `python
alpha_dex_gui.py`, or the legacy `python main_gui.py`, from the repo root, or
install an editable package) so the backend modules remain importable.

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

## Quickstart (basic flow)

```bash
# New PySide6 GUI (recommended)
python alpha_dex_gui.py

# Legacy Tkinter GUI (still functional)
python main_gui.py
```

The Qt app opens on an album-art landing screen, then on **Home**. The sidebar
groups its pages by job: **Library** (Home), **Organize** (Indexer, Library Sync,
File Cleanup), **Clean up** (Duplicates, Metadata), **Listen** (Player, Sound Map,
Playlists) and **Take away** (Mirror, Exports), with Activity, Settings and Help at
the foot. Ctrl+1…9 open the first nine pages.

1. **Choose** your library folder on the landing screen (later: **Change library**
   in the page header, or Ctrl+O).
2. On the **Indexer** page, keep **Dry run** ticked and click **Run Preview**; open
   the HTML plan with **Open Report**.
3. Untick Dry run and click **Execute Previewed Plan** once the preview looks right.
4. Open **Duplicates → Find duplicates** to scan, preview, and execute duplicate groups.
5. Use **Duplicates → Compare two files** to understand tricky duplicate matches.
6. Fix tags in **Metadata → Tag Fixer**, and genres in **Metadata → Fill genres** or
   **Metadata → Normalize genres**.
7. Explore **Playlists** (folder playlists, tempo/energy, Auto‑DJ, repair) and the
   **Sound Map** (Analyse, then Map) once the library is cleaned.
8. Use **Library Sync** to merge an incoming folder when needed.

### Where to find things

- **The library at a glance, its health, the latest runs:** Home
- **File organization / rename:** Indexer; copy-suffix cleanup: File Cleanup
- **Merging new music:** Library Sync
- **Duplicates:** Duplicates → Find duplicates (also Compare two files and Pair review;
  the Fuzzy finder, Scan engine and Bucketing sections are Experimental prototypes)
- **Tags and genres:** Metadata → Tag Fixer / Fill genres / Normalize genres / Undo changes
- **Playback:** Player, plus the Now Playing capsule on every page
- **Playlist tools:** Playlists; one playlist per cluster: Sound Map → Save cluster playlists…
- **Opus mirror:** Mirror → Build the mirror / Verify the mirror
- **Artist/title and codec lists:** Exports
- **Running jobs, the full log, the crash log:** Activity
- **Settings:** Settings (Ctrl+,) → Metadata services / General / Appearance

The legacy Tkinter app (`python main_gui.py`) keeps its own layout of tabs and a
Tools menu.

### Following a long job

Every page reports progress to the log drawer at the bottom of the window (Ctrl+L
toggles it; it opens by itself when a warning or error arrives), and the **Activity**
page lists each running job with a Cancel button and keeps the session's full log.
A clustering run also shows a live log on the Results tab of **Sound Map → Analyse**.
(The legacy Tk app switches to its **Log** tab when a playlist job starts, and writes
cluster progress to `Docs/<method>_log.txt`.)

### Windows Long Paths

The indexer automatically prefixes file paths with `\\?\` on Windows, allowing it to work with directories deeper than the classic 260-character limit.

### Indexer outputs and behavior

- **Preview output:** A preview writes `Docs/MusicIndex.html` under the selected library root; **Open Report** opens it in your browser.
- **Decision log:** A detailed `Docs/indexer_log.txt` captures routing decisions and metadata fallbacks.
- **Missing metadata:** Tracks missing core tags land in `Manual Review/` so you can fix them before re-running.
- **Excluded folders:** the reserved folders (`Not Sorted/`, `Playlists/`, `Manual Review/`, `Docs/`, `Trash/`, `Quarantine/`) are skipped during scans, so you can stash exceptions safely in `Not Sorted/`.
- **Leftover files:** Non-audio leftovers are moved into `Docs/` (`.txt`, `.html`, `.db`) or `Trash/` for everything else.
- **Playlists:** Full runs generate playlists in `Playlists/` by default; uncheck **Create playlists after execution** to only update existing playlists' paths.

## Threading

Long running actions such as indexing, tag fixing and library sync operations run in background `QThread`s, each started through one job registry (`gui/workers.py`) that keeps it alive until it has really finished and cancels it cooperatively when you quit. GUI updates from these background tasks are delivered via Qt signals and scheduled on the main thread — worker threads never touch widgets directly. Only one job at a time may change library files.

## Configuration

User settings are stored in `~/.soundvault_config.json` (legacy filename from SoundVault). To tweak the near-duplicate
fingerprint threshold used during deduplication, add a value like:

```json
{
  "near_duplicate_threshold": 0.1
}
```

Lower values require more similar fingerprints.

Per-format fingerprint thresholds used by the sync matcher can also be
configured. Add a `format_fp_thresholds` section with extension keys:

```json
{
  "format_fp_thresholds": {
    "default": 0.2,
    ".flac": 0.2,
    ".mp3": 0.2,
    ".m4a": 0.2,
    ".aac": 0.2
  }
}
```

Values are floating point distances – lower numbers require closer matches. They
are on the decoded-Chromaprint scale (the same recording in any codec scores below
about 0.01, different recordings 0.43 or more), and 0.2 is the default. Older
saved defaults of 0.3 / 0.35 were tuned for a previous metric.

The same dictionary can be provided to ``library_sync.compare_libraries`` via
the optional ``thresholds`` parameter to control how strictly fingerprints are
matched when scanning two libraries.

You can also store the path to your library for automatic scanning:

```json
{
  "library_root": "/path/to/your/Music"
}
```

The configuration file also stores your selected metadata service and API keys.
You can update these on the **Settings** page, in its **Metadata services**
section:

```json
{
  "metadata_service": "AcoustID",
  "metadata_api_key": "YOUR_ACOUSTID_KEY",
  "lastfm_api_key": "YOUR_LASTFM_KEY"
}
```
`metadata_api_key` is the AcoustID key (leave it empty to use the built-in one) and
`lastfm_api_key` the Last.fm key (the `LASTFM_API_KEY` environment variable wins
when it is set). They are written when you click **Save settings**; **Test
Connection** only checks the key you typed.

MusicBrainz requests require a valid User-Agent string containing your
application name, version and contact email; set them in the same section
(`musicbrainz_useragent`).

## Duplicate Finder

**Duplicates → Find duplicates** is a review-first workflow that makes it easy to
preview and execute deduplication safely.

- **Scan Library** fingerprints the library, groups duplicates and writes the
  preview, `Docs/duplicate_preview.html` (plus `Docs/duplicate_preview.json`, which
  **Duplicates → Pair review** reads); **Preview** opens it.
- **Execute** applies exactly that plan, writes a detailed HTML report under
  `Docs/duplicate_execution_reports/`, and points playlist entries at the kept files.
- **Group dispositions** let you retain, quarantine, or delete losers per group
  while keeping a global default for everything else.
- **Review-flagged groups** are listed before execution: you either override
  (execute every group exactly as previewed) or stop with nothing moved.
- **Thresholds** can be adjusted for a scan: exact and near start from
  **Settings → General**, the mixed-codec boost from the config file
  (`mixed_codec_threshold_boost`).

Duplicates are quarantined into `Quarantine/` by default; you can switch to
retain-in-place, or to delete, which asks you to type DELETE before anything is
deleted.

## Compare two files

**Duplicates → Compare two files** (formerly the Similarity Inspector, and still
headed that way) is a targeted tool for understanding why two tracks match (or do
not match) during duplicate detection.

- Select two files and run the comparison. By default it fingerprints them exactly
  like the Duplicate Finder (reusing its cache) with the thresholds from Settings;
  **Advanced Overrides** lets you trim silence, change the fingerprint offset and
  duration, and set the thresholds yourself.
- The report shows codecs, durations, the fingerprint distance (and the
  overlap-only distance beside it), the thresholds that applied (including the
  mixed-codec adjustment), the verdict, and which fingerprint source was used.

## File Overview

The codebase is organized into backend modules plus a PySide6 GUI layer:

```
alpha_dex_gui.py          - PySide6 entry point (current GUI)
main_gui.py               - Legacy Tkinter entry point (still functional)

── Backend ──────────────────────────────────────────────────────────────
music_indexer_api.py      - Core scanning and relocation logic
duplicate_consolidation.py - Duplicate plan builder (dry-run)
duplicate_consolidation_executor.py - Plan executor
library_sync.py           - Library comparison and plan execution
fingerprint_generator.py  - Build AcoustID fingerprint database
fingerprint_cache.py      - Persistent SQLite fingerprint cache
near_duplicate_detector.py - Fuzzy near-duplicate detection helpers
tag_fixer.py              - Tag fixing engine using plugin metadata
update_genres.py          - Batch genre tag updater via MusicBrainz
playlist_generator.py     - .m3u playlist creation helpers
playlist_engine.py        - Tempo/energy/Auto-DJ logic
clustered_playlists.py    - Feature extraction and K-Means/HDBSCAN clustering
cluster_graph_data.py     - Qt-free data layer for the Sound Map's map (load, validate, export)
cluster_graph_3d.py       - Writes the standalone 3-D browser view (Docs/cluster_graph.html)
cluster_playlist_export.py - Save cluster playlists: summary, planner and writer
cluster_graph_panel.py    - Legacy Tkinter cluster graph panel (not used by the Qt app)
validator.py              - Verify AlphaDEX folder layout (Home's "Library health")
config.py                 - Read/write persistent configuration (~/.soundvault_config.json)
chromaprint_utils.py      - fpcalc wrapper
audio_norm.py             - Audio normalization helpers

── PySide6 GUI (gui/) ───────────────────────────────────────────────────
gui/main_window.py        - AlphaDEXWindow: backdrop, sidebar, page header, pages,
                             Now Playing capsule, log drawer
gui/navigation.py         - The information architecture as data: sidebar groups →
                             pages → sections, Ctrl+1..9 order, navigation keys
gui/pages.py              - The section host (segmented section tabs) and the Sound Map page
gui/workers.py            - JobRegistry: starts, cancels and outlives every QThread
gui/compat.py             - PySide6/PyQt6 compatibility shim

gui/themes/               - Custom QPainter theme engine
  tokens.py               - ThemeTokens + 15 named themes (Liquid Glass, 8 classic dark, 6 classic light)
  glass.py                - The glass painter every surface goes through
  style.py                - AlphaDEXStyle(QProxyStyle): full QPainter rendering
  manager.py              - ThemeManager singleton: apply/persist/auto OS day-night switch
  motion.py               - Motion tokens and safe animation helpers (reduced motion)
  appearance.py           - Cached appearance settings (the backdrop) for painting code
  effects.py              - Colour and shadow helpers, radius constants
  animations.py           - HoverMixin, AnimatedNavButton (badge), AnimatedButton
  picker.py               - ThemePickerPanel (Settings → Appearance) + AutoThemeDialog

gui/widgets/
  backdrop.py             - The window backdrop: a blurred wall of the library's album covers
  sidebar.py              - Floating glass sidebar: grouped pages, Activity · Settings · Help
  top_bar.py              - Page header: page title, library chip with stats, search, Change library
  now_playing_bar.py      - Floating Now Playing capsule
  log_drawer.py           - Slide-up log panel with colour-coded levels
  landing.py              - Album-art landing shown at startup
  interactive_scatter_plot.py, cluster_legend.py, track_details_panel.py
                          - The map, its legend and its details panel (Sound Map → Map)

gui/workspaces/           - One panel per section (WorkspaceBase subclasses), by page
  base.py                 - WorkspaceBase: scroll surface, card/title/button helpers, job helpers
  home.py                 - Home: library stats, Library health (the Validator), latest runs
  indexer.py              - Indexer: 3-phase progress, dry-run/execute, report
  library_sync.py         - Library Sync: scan, review and flag, plan, copy/move execution
  tools.py                - File Cleanup page and Exports page (they replaced Utilities)
  duplicates.py           - Duplicates → Find duplicates: fingerprint scan, groups + inspector
  similarity.py           - Duplicates → Compare two files: two-file threshold breakdown
  launchers.py            - Duplicates → Pair review, and the Experimental Fuzzy finder,
                             Scan engine and Bucketing sections (each opens a dialog)
  tag_fixer.py            - Metadata → Tag Fixer: proposals table with checkboxes
  genres.py               - Metadata → Fill genres and Normalize genres (one panel):
                             MusicBrainz gap-fill + canonical mapping (scan → LLM mapping
                             → preview → apply)
  tag_history.py          - Metadata → Undo changes: roll back a tag apply, previewed first
  player.py               - Player: libVLC transport controls, library table, queue
  clustered_enhanced.py   - Sound Map → Analyse: Quick Start, the clustering wizard, results
  graph.py                - Sound Map → Map: the in-app 2-D map, plus the browser-based 3-D view
  playlists.py            - Playlists: Folder / Tempo+Energy / Auto-DJ / Repair
  compression.py          - Mirror → Build the mirror: FLAC → Opus library mirror
  mirror_verify.py        - Mirror → Verify the mirror: the Opus and M4A testers
  activity.py             - Activity: running jobs (with Cancel), the log, problems and crash log
  settings.py             - Settings → Metadata services / General / Appearance
  help.py                 - Help: doc links, keyboard shortcuts, About

gui/dialogs/              - Clustering wizard, quality report, Save cluster playlists preview,
                             File Cleanup preview, Pair Review and the prototype dialogs

── Other ─────────────────────────────────────────────────────────────────
controllers/              - Thin wrappers wiring backend to the GUI
plugins/
  base.py               - Metadata plugin interface
  api_service.py        - Base class for Settings-UI "Test Connection" services
  acoustid_plugin.py    - Metadata lookup via AcoustID, plus a MusicBrainz
                           service class (not itself a scan plugin — see
                           docs/ROADMAP.md)
  musicbrainz_plugin.py - MusicBrainz text search, used by the Tag Fixer scan
                           when MusicBrainz is the service chosen in Settings
  lastfm.py              - Fetch genres from Last.fm
  test_plugin.py         - A test fixture; never loaded by a real scan

docs/                     - Project documentation and design notes
tests/                    - pytest suite (headless GUI tests in tests/ui/)
```

> Note: `bindings/`, `third_party/`, and `plugins/assistant_plugin.py`
> (an optional local-LLM assistant) do not exist in this repository and
> never have, despite being mentioned in some older material. There is no
> dormant LLM feature to resume — it would be new work from scratch.

## Roadmap (Upcoming Features)

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the maintained, detailed list of
what's planned, partially built, or known to be broken. Highlights:

- Library Sync: flagging many tracks at once, and keeping flags across restarts.
- The Sound Map: live re-tuning, in-map cluster editing, and a choice of
  map-flattening methods.
- Expanded metadata plugins beyond AcoustID/MusicBrainz/Last.fm (Spotify,
  Gracenote, Discogs).
- One known bug remains, in the legacy Tkinter app only — see `docs/ROADMAP.md`'s
  "Known bugs" section. (The duplicate-pair report and the cluster quality report
  dialog, once listed here, are fixed, and the in-app 2-D map is wired in as
  Sound Map → Map.)

See [`docs/README.md`](docs/README.md) for the full documentation suite,
starting with `docs/overview.md`.

## Known gaps

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the maintained, detailed version
of this list. In brief:

- **Tidal-dl sync**: referenced in older material, but there is no UI or workflow (and `tidal-dl` is not a dependency).
- **Metadata provider breadth**: only AcoustID, MusicBrainz, and Last.fm are fully wired end-to-end; Spotify/Gracenote are shown, clearly disabled, in Settings, but not implemented.
- **Library Sync per-item flags**: ✅ IMPLEMENTED — right-click incoming tracks to flag for copy/replace or add notes. Flags override auto-decisions during plan building.
- **Library Sync Export Report**: ✅ IMPLEMENTED — a real Export Report button saves HTML, JSON, or CSV.
