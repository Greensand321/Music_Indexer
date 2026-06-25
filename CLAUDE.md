# CLAUDE.md — AlphaDEX (Music Indexer)

Project context and working conventions for Claude Code sessions.

---

## What this project is

**AlphaDEX** is a Python desktop application for organizing large music
libraries. It is a single-user tool; there is no server, no API surface, and
no database other than SQLite caches.

**Two GUI entry points exist (read this carefully):**

- `python alpha_dex_gui.py` — **PySide6 / Qt rebuild. This is the active,
  primary app** and where new feature work lands. It uses an Option A
  Navigator + Workspace layout (`gui/` package: `main_window.py` plus ~14
  user-facing workspaces under `gui/workspaces/`). Most "implemented" features
  described below — clustering wizard, interactive graph, Library Sync per-item
  flags, splash/landing — live here.
- `python main_gui.py` — **Tkinter app (~11,600 lines). Legacy.** Still
  functional and kept as a fallback/reference, but no longer the development
  target. (`alpha_dex_gui.py`'s own docstring notes "the original Tkinter app
  remains available at `python main_gui.py`.")

When a doc or task says "the app" without qualifying, assume the **Qt** app.
Plain-English, concept-driven documentation for the whole system lives in
`docs/` — start at `docs/overview.md` and `docs/architecture.md`.

The core workflows, in order of user importance:

1. **Indexer** — scans a library folder, normalizes file names and folder
   structure, and writes an HTML preview before touching any files.
2. **Duplicate Finder** — fingerprints audio with AcoustID/Chromaprint, groups
   near-duplicates, lets the user review groups, then executes a
   quarantine/delete plan.
3. **Library Sync** — compares an existing library against an incoming folder,
   produces a copy/move plan, previews it, then executes.
4. **Tag Fixer** — looks up AcoustID / MusicBrainz metadata and proposes tag
   corrections for review before writing.
5. **Playlist Creator** — tempo/energy bucketing, Auto-DJ chaining, K-Means
   and HDBSCAN clustering, genre normalization, year-gap assistant.

---

## Repository layout

```
alpha_dex_gui.py             # PySide6/Qt entry point — ACTIVE app (landing → main window)
main_gui.py                  # Tkinter entry point (~11 600 lines) — LEGACY
gui/                         # Qt app: main_window.py, qt_launcher.py, theme/icons
gui/workspaces/              # ~14 user-facing workspaces (indexer, library_sync,
                             #   duplicates, similarity, tag_fixer, genres, playlists,
                             #   clustered, clustered_enhanced, graph, player,
                             #   compression, tools, help); base.py is the shared template
gui/widgets/                 # Qt widgets (interactive scatter plots, cluster legend, etc.)
gui/dialogs/                 # Qt dialogs (clustering wizard, quality report, settings)
music_indexer_api.py         # Core scan / relocation logic
duplicate_consolidation.py   # Duplicate plan builder (dry-run)
duplicate_consolidation_executor.py  # Plan executor
library_sync.py              # Library comparison and plan execution
library_sync_review.py       # Library Sync review-first UI panel
fingerprint_generator.py     # AcoustID fingerprint generation
fingerprint_cache.py         # SQLite fingerprint cache
near_duplicate_detector.py   # Fingerprint distance helpers
playlist_generator.py        # .m3u playlist helpers
playlist_engine.py           # Tempo/energy/Auto-DJ logic
clustered_playlists.py       # K-Means / HDBSCAN clustering
cluster_graph_panel.py       # Interactive scatter-plot widget
tag_fixer.py                 # Tag-fix engine
update_genres.py             # Batch genre updater
config.py                    # Load / save ~/.soundvault_config.json
validator.py                 # Validate library folder structure
chromaprint_utils.py         # fpcalc wrapper
audio_norm.py                # Audio normalization helpers

controllers/                 # Thin wrappers wiring backend to GUI
plugins/                     # Metadata service integrations
utils/                       # Metadata readers, path helpers
tests/                       # pytest suite (42 modules)
docs/                        # HTML docs, design notes, GUI inventory
mutagen_stub/                # Minimal mutagen fallback for tests
library_sync_indexer_engine/ # Alternate indexer used by Library Sync
third_party/                 # Prebuilt llama binaries (LLM assistant)
bindings/                    # C++/pybind11 llama wrappers
```

---

## Key constants and configuration

**Config file:** `~/.soundvault_config.json` (legacy name from the project's
former "SoundVault" identity — do not rename it).

**Audio extensions** (from `simple_duplicate_finder.py`):
`.flac .m4a .aac .mp3 .wav .ogg .opus`

**Lossless extensions** (from `duplicate_consolidation.py`):
`.flac .wav .alac .ape .aiff .aif`

**Codec priority** (higher = preferred winner in dedup):
`.flac` (3) > `.wav` (2) > `.mp3` (1)

**Default fingerprint thresholds** (from `config.py`):
- Exact duplicate: `0.02`
- Near duplicate: `0.1`
- Mixed-codec boost: `0.03`
- Library sync default: `0.3`

**Reserved library folders** — the indexer and duplicate finder skip these
during scans:
- `Not Sorted/` — user-managed exclusion zone
- `Playlists/` — playlist storage
- `Manual Review/` — tracks missing required metadata
- `Docs/` — HTML reports and logs
- `Trash/` — non-audio file leftovers
- `Quarantine/` — duplicate losers awaiting review

---

## Running the application

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main_gui.py
```

FFmpeg must be on `PATH` for audio analysis. VLC / libVLC is required for the
in-app Player tab. Optional: `essentia==2.1b6` for faster C++ feature
extraction (see README for platform-specific build steps).

---

## Running the tests

```bash
# From the repo root, with the venv active:
pytest

# Run a single file:
pytest tests/test_duplicate_consolidation_hardening.py

# Run with verbose output:
pytest -v
```

**pytest config** (`pytest.ini`):
- `testpaths = tests`
- `pythonpath = .`

The `tests/conftest.py` injects lightweight stubs for `pydub`, `tkinter`, and
other heavy dependencies so the suite can run without the full GUI stack or
audio libraries installed. Do not remove or weaken these stubs.

Test files follow the `test_<module_or_feature>.py` naming convention.
Individual tests use `monkeypatch` and `tmp_path` (pytest fixtures) — no
custom base classes.

---

## Architecture rules to keep in mind

### Preview-first, never destructive by default
Every major workflow (indexing, deduplication, library sync) must produce a
dry-run preview before modifying any files. Do not add code paths that skip
the preview stage or mutate the library without an explicit user confirmation
step.

### GUI ↔ backend separation
- Business logic lives in the backend modules (`music_indexer_api.py`,
  `duplicate_consolidation.py`, `library_sync.py`, etc.).
- `main_gui.py` and `library_sync_review.py` are UI-only. They call backend
  functions and schedule results back to the main thread via `after()`.
- Do not import `tkinter` in backend modules.

### Threading model
Long operations run in daemon threads. GUI state is only mutated from the main
thread via `widget.after(0, callback)`. Never call `.configure()`, `.insert()`,
or any other widget method directly from a worker thread.

### No silent data loss
Operations that move or delete files must log every action to `Docs/` and
respect the user's chosen disposition (retain / quarantine / delete). When in
doubt, default to quarantine, not delete.

### Config persistence
All user settings are read/written through `config.load_config()` /
`config.save_config()`. Do not read `~/.soundvault_config.json` directly.

---

## Common patterns

**Adding a new Tools menu item**
1. Add the `tools_menu.add_command(...)` entry in `_build_menu()` (~line 8103
   of `main_gui.py`).
2. Implement the handler as a method on `SoundVaultImporterApp`.
3. If it's a significant workflow, put the UI in a new `tk.Toplevel` subclass
   (follow the pattern of `FileCleanupDialog`, `PlaylistRepairDialog`, etc.).

**Adding a new backend module**
- Place logic in a top-level `.py` file.
- Add a corresponding `tests/test_<name>.py`.
- Import it in `main_gui.py` only if a GUI hook is needed; keep the module
  importable without Tkinter.

**Adding a Playlist Creator sub-panel**
- Add the panel name to the `plugin_list.insert` block (~line 8326).
- Add an `elif name == "Your Panel Name":` branch in
  `create_panel_for_plugin()` (~line 216).

---

## Docs to check before making changes

The `docs/` suite was rewritten (2026-06-25) as plain-English, concept-driven
documentation. Start here:

| File | When to read it |
|---|---|
| `docs/README.md` | Index / reading order for the whole docs suite |
| `docs/overview.md` | First read — what the app is, core principles, key concepts, the two GUIs |
| `docs/architecture.md` | How the layers, workers, preview contract, cache, and workspaces fit together |
| `docs/features/indexer.md` | Before touching the Indexer |
| `docs/features/duplicate_finder.md` | Before touching the Duplicate Finder |
| `docs/features/library_sync.md` | Before touching Library Sync |
| `docs/features/playlists_and_clustering.md` | Before touching playlists, clustering, or the graph |
| `docs/ROADMAP.md` | What's planned but not yet built (single source of truth for gaps) |
| `docs/gui_inventory.md` | Detailed map of every screen and control in the Qt app |
| `docs/archive/` | Superseded planning notes / audit snapshots — history only, not current |

---

## Known gaps (do not assume these are complete)

See `docs/ROADMAP.md` for the full, maintained list. Summary:

- **Metadata provider breadth:** Only AcoustID, MusicBrainz, and Last.fm are fully
  wired end-to-end. Spotify and Gracenote are listed in `config.SUPPORTED_SERVICES`
  and have stub functions in `metadata_service.py` that return `{}` (placeholders,
  not implementations). Discogs is roadmap-only.
- **Tidal-dl sync:** referenced historically but has no UI or workflow. Dormant.
- **Library Sync per-item flags:** ✅ IMPLEMENTED and wired in the Qt app. Right-click
  an incoming track in `gui/workspaces/library_sync.py` to flag copy/replace/clear or
  add a note; `ReviewStateStore` holds flags and they override auto-decisions during
  plan building. (Older archived docs claiming this is "not exposed in the UI" are
  stale/false.) Remaining limitation: one-at-a-time flagging; no bulk flag yet.
- **Library Sync Export Report:** Export helpers exist and work in
  `library_sync_review_report.py` (`export_report()`, `export_review_report_html()`),
  but the Export Report button is not wired to a user-accessible control. Smallest
  outstanding win in the project.
- **Clustering features:** the engine currently clusters on timbre (MFCC) + tempo;
  the wizard's chroma/spectral/onset checkboxes are UI scaffolding not yet wired into
  feature extraction. Interactive-graph Phases 4–8 (live tuning, advanced selection,
  in-map cluster editing, report export) are unbuilt — see `docs/ROADMAP.md`.
