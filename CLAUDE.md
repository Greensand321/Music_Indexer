# CLAUDE.md — AlphaDEX (Music Indexer)

Project context and working conventions for Claude Code sessions.

---

## What this project is

**AlphaDEX** is a Python desktop application for organizing large music
libraries. It is a single-user tool; there is no server, no API surface, and
no database other than SQLite caches.

**Two GUI entry points exist (read this carefully):**

- `python alpha_dex_gui.py` — **PySide6 / Qt rebuild. This is the active,
  primary app** and where new feature work lands. Sidebar **pages**, each made of
  one or more **sections** (tabs), each section backed by a **panel** (a
  `WorkspaceBase` subclass under `gui/workspaces/`). `gui/navigation.py` is the
  single source for pages, sections and key routing (old per-panel keys such as
  `similarity`, `graph`, `genres` still route). Most "implemented" features
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
4. **Tag Fixer** — looks up AcoustID metadata (MusicBrainz enrichment nested
   inside an AcoustID match; MusicBrainz is also a search source of its own when
   selected in Settings — see `docs/features/tag_fixer.md`) and proposes per-field
   tag corrections for review before writing exactly what is ticked.
5. **Playlist Creator** — tempo/energy bucketing, Auto-DJ chaining, K-Means
   and HDBSCAN clustering, genre tag-filling. (A "year-gap assistant" is
   referenced in some older material; no such feature exists — see
   `docs/ROADMAP.md`.)

---

## Repository layout

```
alpha_dex_gui.py             # PySide6/Qt entry point — ACTIVE app (landing → main window)
main_gui.py                  # Tkinter entry point (~11 600 lines) — LEGACY
gui/                         # Qt app: main_window.py, qt_launcher.py, theme/icons,
                             #   navigation.py (pages → sections → panels), pages.py
                             #   (the section host, SoundMapPage)
gui/workspaces/              # Panels: home, indexer, library_sync, tools (File Cleanup,
                             #   Exports), duplicates, similarity, launchers (Pair review
                             #   + the Experimental prototypes), tag_fixer, genres,
                             #   tag_history (Undo changes), player, clustered_enhanced
                             #   (Analyse), graph (Map), playlists, compression,
                             #   mirror_verify, activity, settings, help; base.py is the
                             #   shared template
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
library_sync_indexer_engine/ # A SEPARATELY MAINTAINED, DRIFTED COPY of the
                             #   indexer/fingerprint-cache engine, used only
                             #   by library_sync.py. Not a thin re-export —
                             #   see docs/architecture.md's "two indexer
                             #   engines" note before assuming a fix to the
                             #   root-level modules also applies here.
```

`bindings/` and `third_party/` (an optional local-LLM assistant, referenced in
older notes) **do not exist in this repository and never have** — confirmed
against the full git history. If you're looking for that feature, there is
nothing to resume; it would be new work from zero.

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
- Library sync default: `0.2` (decoded-Chromaprint bit error rate; old saved 0.3/0.35 are migrated once — see `docs/CONTRADICTING_UPDATES.md`)

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
python alpha_dex_gui.py        # the Qt app (primary); legacy Tk app: python main_gui.py
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

Install the dev/test dependencies with `pip install -r requirements-dev.txt`
(PySide6, pytest-qt, pytest-timeout, mutagen, ...). ffmpeg on `PATH` lets the
synthetic test library use real MP3/FLAC/M4A/OGG/Opus files.

**pytest config** (`pytest.ini`):
- `testpaths = tests`, `pythonpath = .`, `qt_api = pyside6`
- `timeout = 300` with `timeout_method = thread` (a test hung inside a Qt
  event loop is killed instead of hanging CI)

**Every test runs with an isolated HOME** (`tests/conftest.py::_isolated_home`
also repoints `config.CONFIG_PATH`), so the suite can never read or overwrite
the real `~/.soundvault_config.json`. `tests/conftest.py` also injects a
lightweight `pydub` stub. Do not remove or weaken either. The
`synthetic_library` fixture (from `tests/library_factory.py`) gives each test a
fresh library of real, tagged audio files.

**Module stubs must be scoped to one test.** Use
`tests/_helpers/module_isolation.py` (`isolated_modules()`,
`stub_if_unavailable()`) or `monkeypatch.setitem(sys.modules, ...)` — never
assign into `sys.modules` at module level. Ad-hoc global stubs (mostly fake
`mutagen` modules) used to make the result depend on file order; the suite is
now verified green in default, reversed and shuffled order. Keep it that way.
(`plugins/acoustid_plugin.py` no longer imports `tkinter`; there is still no
tkinter stub, and none is needed by the Qt app.)

**GUI tests** live in `tests/ui/` and run headless (`QT_QPA_PLATFORM=offscreen`).
Read `tests/ui/README.md` before writing one. The harness automatically fails
a test on: an unplanned modal dialog (queue answers through the `dialogs`
fixture instead), a `QThread` left running, cross-thread/painting/animation Qt
warnings, or any exception raised inside a slot or paint event. Fixtures:
`main_window`, `make_workspace`, `library`, `dialogs`, `threads`, `motion`.
`scripts/gui_snapshots.py` renders every page per theme to PNG for visual
before/after comparison.

**Static ratchets.** `tests/test_gui_hygiene_ratchets.py` caps the counts of
hard-coded hex colours, inline `setStyleSheet` calls, `font-size` literals,
direct `QThread.start()` calls and unparented animations in `gui/`. A change
may lower a count, never raise it; lower the ceiling when you remove some.
`tests/ui/test_ui_static_qt_api.py` fails on any `QtCore`/`QtGui`/`QtWidgets`
name that doesn't exist in PySide6 (it caught `QFont.Weight.SemiBold` and the
Qt5-era `QtWidgets.QShortcut`).

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
Long operations run in worker threads; GUI state is only mutated on the main
thread (Qt: emit signals from the worker; legacy Tk: `widget.after(0, ...)`).
Never call widget methods directly from a worker thread.

**Qt app: start every `QThread` through `WorkspaceBase._start_worker(worker,
"Name")`** (or `gui.workers.start_worker(...)` outside a workspace). The
`JobRegistry` in `gui/workers.py` keeps each thread alive until the real C++
`QThread::finished()` arrives. Before it existed, dropping `self._worker` in a
"finished" slot, re-running a job, or quitting mid-run destroyed a running
thread — `QThread: Destroyed while thread is still running`, a hard abort of
the whole process, sometimes mid file-move. The main window's `closeEvent`
asks before quitting with jobs running, cancels them cooperatively (calls the
worker's `cancel()` and `requestInterruption()`), and waits. So: give long
workers a real `cancel()`, disable Run buttons while running, and decide what
a library change mid-run does.

### Reserved library folders
Use `utils/reserved_folders.py` (`RESERVED_FOLDERS`, `NON_LIBRARY_FOLDERS`,
`walk_library`, `iter_library_files`, `is_reserved_path`) for any scan that
treats files as "the library". Don't keep private folder lists — several
workflows used to scan `Quarantine/`, `Trash/` and `Docs/` as music (the
Indexer even moved quarantined duplicate losers back into the library).

### No silent data loss
Operations that move or delete files must log every action to `Docs/` and
respect the user's chosen disposition (retain / quarantine / delete). When in
doubt, default to quarantine, not delete.

### Config persistence
All user settings are read/written through `config.load_config()` /
`config.save_config()`. Do not read `~/.soundvault_config.json` directly.

---

## Common patterns

**Adding a page or a section (Qt app)**
1. Write the panel as a `WorkspaceBase` subclass in `gui/workspaces/` (use
   `_make_card()`, `_make_section_title()`, theme tokens; no hex colours or inline
   style sheets — the ratchets count them).
2. Register it in `gui/main_window._WORKSPACE_MAP` under a panel key.
3. Add a `Section(...)` to its page (or a new `Page`) in `gui/navigation.PAGES`.
   Prototypes get `experimental=True`, which shows the "Experimental" pill.
4. Update `docs/gui_inventory.md` and `tests/test_navigation_ia.py`.

**Adding a new Tools menu item (legacy Tk app only)**
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

**Adding a Playlist Creator sub-panel (legacy Tk app only)**
- Add the panel name to the `plugin_list.insert` block (~line 8326).
- Add an `elif name == "Your Panel Name":` branch in
  `create_panel_for_plugin()` (~line 216).

---

## Docs to check before making changes

The `docs/` suite was rewritten (2026-06-25) as plain-English, concept-driven
documentation, then fully re-verified against the code (2026-07-22) — the
second pass found and corrected substantial drift, especially in
`gui_inventory.md`. Start here:

| File | When to read it |
|---|---|
| `docs/README.md` | Index / reading order for the whole docs suite, plus a scannable known-gaps summary |
| `docs/overview.md` | First read — what the app is, core principles, key concepts, the two GUIs |
| `docs/architecture.md` | How the layers, workers, preview contract, cache, and workspaces fit together — includes the "two indexer engines" architecture note |
| `docs/features/indexer.md` | Before touching the Indexer |
| `docs/features/duplicate_finder.md` | Before touching the Duplicate Finder |
| `docs/features/library_sync.md` | Before touching Library Sync |
| `docs/features/tag_fixer.md` | Before touching the Tag Fixer or Genre Normalizer |
| `docs/features/playlists_and_clustering.md` | Before touching playlists, clustering, or the graph |
| `docs/ROADMAP.md` | What's planned but not yet built, plus a "Known bugs" section (single source of truth for gaps) |
| `docs/gui_inventory.md` | Detailed map of every screen and control in the Qt app, including which controls are currently decorative |
| `docs/redesign/README.md` | **GUI redesign groundwork:** the intent of every page, how pages connect, systemic problems, proposed IA, design-system/motion plan, open decisions, bug status. Per-area detail (with bug tables) in `docs/redesign/pages/` |
| `tests/ui/README.md` | The headless Qt test harness: fixtures, guards, ratchets, snapshots |
| `docs/CONTRADICTING_UPDATES.md` | Every change that contradicts older docs, code comments or the original design (kept so the owner can walk any of them back) |
| `docs/specs/` | Feature specs (e.g. `cluster_playlists.md`) |
| `docs/archive/` | Superseded planning notes / audit snapshots — history only, not current |

---

## Known gaps and bugs (do not assume these are complete)

See `docs/ROADMAP.md` for the full, maintained list, including a "Known bugs"
section kept separate from planned-but-unbuilt features. Summary:

- **Metadata provider breadth:** Only AcoustID, MusicBrainz, and Last.fm are fully
  wired end-to-end. Spotify and Gracenote are listed in `config.SUPPORTED_SERVICES`
  and have stub functions in `metadata_service.py` that return `{}` (placeholders,
  not implementations) — but the Settings UI now shows them visibly disabled with an
  explanatory tooltip rather than silently omitting or misrepresenting them.
  Discogs is roadmap-only. MusicBrainz contributes via enrichment inside an
  AcoustID match and, when chosen in Settings, as its own search source
  (`plugins/musicbrainz_plugin.py`). Keys: `metadata_api_key` is the AcoustID key
  (empty = built-in key), `lastfm_api_key` the Last.fm key.
- **Tidal-dl sync:** referenced historically but has no UI or workflow. Dormant.
- **Library Sync per-item flags:** ✅ IMPLEMENTED and wired in the Qt app. Right-click
  an incoming track in `gui/workspaces/library_sync.py` to flag copy/replace/clear or
  add a note; `ReviewStateStore` holds flags and they override auto-decisions during
  plan building. Remaining limitation: one-at-a-time flagging; no bulk flag yet.
- **Library Sync Export Report:** ✅ IMPLEMENTED. A real "Export Report" button in
  the Qt workspace writes HTML, JSON, or CSV via `library_sync_review_report.py`.
- **Library Sync has two separately-maintained engine copies** — and this is
  *settled, not debt to pay down.* `library_sync.py` imports its
  scanning/fingerprinting logic from a vendored copy under
  `library_sync_indexer_engine/`, a leftover from the project's earlier
  (Tkinter-era) generation; the root-level `music_indexer_api.py` /
  `fingerprint_cache.py` are the ones that kept evolving. **The root-level engine
  is canonical for new work; leave the vendored copy alone unless you are
  specifically changing it.** Unifying them is explicitly not planned. The one
  consequence to remember: a fix to a root module does not reach Library Sync
  automatically — port it across deliberately if Library Sync needs it (the
  decoded-Chromaprint fingerprint distance was ported this way). See
  `docs/architecture.md`.
- **Canonical genre mapping is in the Qt app** (`gui/workspaces/genres.py`,
  "Normalize with a mapping" tab), backed by `controllers/normalize_controller.py`
  — `plan_genre_normalization` (dry run) then `apply_genre_changes` (writes only
  the plan). Discovery/reader/writer are injectable; keep it that way so it stays
  Qt-free and testable. `["invalid"]` and `null` mapping values mean "drop."
- **Clustering features are selectable and all wired.** `clustered_playlists.FEATURE_ORDER`
  defines six blocks (mfcc 26, tempo 1, chroma 12, spectral 2, energy 2, onset_rate 1);
  `generate_clustered_playlists(features=..., normalization=...)` computes only the
  selected ones. The feature cache filename is keyed by selection and engine
  (`feature_cache_name`) — do not reintroduce a single shared `features.npy` for all
  selections. `features=None` keeps the legacy 27-dim MFCC+tempo vector that Auto-DJ
  (`playlist_engine`) still relies on.
- **The in-app 2-D interactive graph is built and wired.** `gui/workspaces/graph.py`
  embeds `InteractiveScatterPlot` + `ClusterLegendWidget` + `TrackDetailsPanel`;
  data loading/validation/export lives in the Qt-free `cluster_graph_data.py` (tested
  in `tests/test_cluster_graph_data.py`). The unused `Interactive3DScatterPlot` widget was deleted —
  the 3-D view users reach is the browser page from `cluster_graph_3d.py`.
- **Cluster payload arrays are index-aligned, and must stay that way.**
  `X_2d`/`X_3d`/`labels`/`tracks` in `Docs/cluster_info.json` are consumed
  positionally. `clustered_playlists` previously subset the coordinates when
  downsampling (>5,000 tracks) without subsetting labels/tracks, which broke the 3-D
  generator outright and would have mislabelled points. If you touch that block,
  keep every array subset together.
- **Decorative controls were wired or switched off** in the 2026-11 fix phase (the
  Duplicates per-group disposition, the Playlist Generator's tempo/energy ranges and
  "Prefer Opus" now work; controls the engine ignores — e.g. the Indexer's Phase C /
  Flush cache — are disabled with an explanatory tooltip; Max workers became real,
  default 1). Don't add
  new controls that nothing reads; `docs/ROADMAP.md` "Controls that are still
  switched off" lists the rest.
- **Every file-changing job takes the library-write lock.** Start it with
  `self._start_worker(..., library_write=True)` after checking
  `self._library_writes_blocked("Action name")`, so two pages never change library
  files at once. Read-only jobs pass `ask_before_quit=False`.
- **GUI redesign groundwork lives in `docs/redesign/`** — the intent of every page,
  the proposed information architecture, design-system and motion plan, and the
  owner's open decisions. Read `docs/redesign/README.md` before restructuring pages.
