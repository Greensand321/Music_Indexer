# AlphaDEX GUI Inventory

Plain-English reference for every screen, panel, control, and dialog in the
**PySide6 GUI** (`alpha_dex_gui.py` → `gui/main_window.py`).

> The legacy Tkinter interface (`main_gui.py`) is still functional but is no
> longer the primary GUI. This document describes the PySide6 replacement only.

---

## 1. Application Window (`AlphaDEXWindow`)

`QMainWindow`, default size 1300 × 860 px, minimum 900 × 600 px.

Layout (top → bottom, left → right):

```
┌─────────────────────────────────────────────────────────────┐
│  Top Bar                                                     │
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│   Sidebar    │           Workspace (stacked)                │
│   (220 px)   │                                              │
│              ├──────────────────────────────────────────────┤
│              │  Log Drawer (slide-up, collapsed by default) │
├──────────────┴──────────────────────────────────────────────┤
│  Status Bar                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Top Bar (`gui/widgets/top_bar.py`)

Fixed height 64 px. Always visible.

| Element | Type | Purpose |
|---|---|---|
| App title "AlphaDEX" | Label (`#appTitle`) | Brand / identity |
| Library path | Label (`#libraryPath`) | Full path of the currently open library |
| Library stats | Label (`#libStats`) | Track count · GB · artist count (populated after background scan) |
| 📁 Change Library | Button | Opens a `QFileDialog` folder picker; emits `library_changed(path)` |
| 🎨 Theme | Button | Opens `ThemePickerDialog` (non-modal swatch grid) |
| ⚙ Settings | Button | Opens `SettingsDrawer` dialog |

**Signals emitted**

| Signal | When |
|---|---|
| `library_changed(str)` | User selects a new library folder |
| `settings_requested()` | User clicks ⚙ Settings |
| `theme_requested()` | User clicks 🎨 Theme |

---

## 3. Sidebar (`gui/widgets/sidebar.py`)

Fixed width 220 px. Dark background (token `sidebar_bg`). Contains a `QScrollArea`
so the nav list can scroll on small displays.

### 3.1 Logo label

Static "AlphaDEX" label at the top.

### 3.2 Navigation items

Grouped into five sections. Each item is an `AnimatedNavButton` (150 ms hover
easing, badge support). Clicking emits `nav_changed(key)` to the main window
which switches the active workspace.

| Section | Key | Label |
|---|---|---|
| **ORGANIZE** | `indexer` | Indexer |
| | `library_sync` | Library Sync |
| **CLEAN UP** | `duplicates` | Duplicates |
| | `similarity` | Similarity Inspector |
| | `tag_fixer` | Tag Fixer |
| | `genres` | Genre Normalizer |
| **PLAYLISTS** | `playlists` | Playlist Generator |
| | `clustered` | Clustered Playlists |
| | `graph` | Visual Music Graph |
| **PLAYER** | `player` | Player |
| | `compression` | Compression |
| **TOOLS** | `tools` | Export & Utilities |
| | `help` | Help |

Active item renders with accent fill (`sidebar_active` token). Badges (numeric
counts) appear at the trailing edge and are set via `Sidebar.set_badge(key, n)`.

**Keyboard shortcuts** (set in `main_window.py`):

| Shortcut | Action |
|---|---|
| Ctrl+1 … Ctrl+9 | Switch to workspace 1–9 in sidebar order |
| Ctrl+O | Change library (same as 📁 button) |
| Ctrl+, | Open Settings |
| Ctrl+L | Toggle log drawer |
| Ctrl+W | Clear log |

---

## 4. Log Drawer (`gui/widgets/log_drawer.py`)

A slide-up panel anchored to the bottom of the content area. Collapsed by
default (0 px); expands to ~200 px on toggle or automatically when an
error/warning message arrives.

| Element | Type | Purpose |
|---|---|---|
| Handle bar | Clickable strip | Shows current status chip; click to toggle open/close |
| Status chip | Label (coloured) | One-line summary of the last operation status |
| Log text area | `QPlainTextEdit` (read-only) | Colour-coded log: green = ok, amber = warn, red = err |
| Clear button | Button (in handle bar) | Clears the log text |

`append(message, level)` — called by workspaces via the `log_message` signal.
Levels: `"info"`, `"ok"`, `"warn"`, `"err"`.

---

## 5. Theme Engine (`gui/themes/`)

### 5.1 ThemeManager (`manager.py`)

Singleton `QObject`. Exposes:

- `apply(key)` — switch to a named theme or `"auto"`.
- `load_persisted()` — reads `~/.soundvault_config.json` on startup.
- `configure_auto(dark_key, light_key)` — sets the OS day/night pair.
- `theme_changed` signal — emitted after every switch; `main_window.py` listens
  to refresh card drop-shadows.
- Auto mode: monitors `QGuiApplication.styleHints().colorSchemeChanged` (Qt 6.5+);
  falls back to time-of-day (07:00–20:00 = light).

### 5.2 Themes

14 named themes persisted to config as `"theme"`.

**Dark (8)**: Midnight, Obsidian, Graphite, Navy, Twilight, Aurora, Forest, Ember

**Light (6)**: Pearl, Azure, Blossom, Meadow, Lavender, Sunset

### 5.3 ThemePickerDialog (`picker.py`)

Non-modal dialog opened by the 🎨 Theme button.

| Element | Purpose |
|---|---|
| Auto card | Special row with Configure… and Select buttons; shows current day/night pair |
| Dark Themes grid | Swatch cards for all 8 dark themes |
| Light Themes grid | Swatch cards for all 6 light themes |
| Swatch card | Mini preview (sidebar strip + content area mock); checkmark when active |

Theme changes apply instantly (no Apply button needed).

### 5.4 AutoThemeDialog

Modal child of `ThemePickerDialog`.

| Element | Purpose |
|---|---|
| Description label | Explains OS day/night switching |
| Night theme section | Swatch grid — select one dark theme |
| Day theme section | Swatch grid — select one light theme |
| Apply Auto Theme button | Saves pair and activates auto mode |

---

## 6. Workspaces

All workspaces inherit `WorkspaceBase` (`gui/workspaces/base.py`). Common
features:

- Wrapped in a `QScrollArea` so tall content is always reachable.
- `log_message(str, level)` signal routed to the log drawer.
- `status_changed(str, colour)` signal updates the log drawer handle and status bar.
- `_make_card()` returns a `QFrame` with rounded corners and a drop-shadow effect
  (shadow colour updates on theme change via `refresh_shadows()`).

### 6.1 Indexer (`indexer.py`)

Controls the file-organisation and rename pipeline.

| Element | Type | Purpose |
|---|---|---|
| Configuration card | Card | Contains all run options |
| Dry Run | Checkbox | Preview only — writes HTML report, no files moved |
| Cross-Album Scan | Checkbox | Enables Phase C (across album boundaries) |
| Flush Cache | Checkbox | Clears fingerprint cache before run |
| Create Playlists | Checkbox | Generates `.m3u` playlists in `Playlists/` on full run |
| Max Workers | SpinBox | Thread pool size (1–32) |
| Start Indexer / Cancel | Button (primary) | Launches or cancels the background indexer |
| Progress card | Card | Three labeled progress bars: Phase A / Phase B / Phase C |
| Status label | Label | Current step description |
| Open Report | Button | Opens `Docs/MusicIndex.html` in the system browser |

Worker: `IndexerWorker(QThread)` calls `music_indexer_api.run_full_indexer()`.

### 6.2 Duplicates (`duplicates.py`)

Review-first duplicate detection and disposal.

| Element | Type | Purpose |
|---|---|---|
| Scan options card | Card | Library path entry, threshold sliders (exact / near / mixed-codec) |
| Start Scan / Cancel | Button (primary) | Launches `DupeScanWorker` |
| Scan progress bar | Progress bar | Fingerprint generation progress |
| Groups table | `QTreeWidget` | Lists duplicate groups; columns: Group, Files, Sizes, Codecs |
| Inspector panel | Card | Shows selected group's file list with per-item details |
| Disposition row | Radio buttons | Retain / Quarantine / Delete |
| Execute Plan | Button (danger) | Applies dispositions — writes execution report to `Docs/` |

### 6.3 Library Sync (`library_sync.py`)

Compare two libraries and execute a copy/move plan.

| Element | Type | Purpose |
|---|---|---|
| Source path entry + Browse | Row | Pick the existing library folder |
| Incoming path entry + Browse | Row | Pick the incoming library folder |
| Threshold override | Input field | Fingerprint matching threshold (0.0–1.0) |
| Scan progress | Two progress bars | Existing lib scan + Incoming lib scan |
| Start Scan | Button (primary) | Launches `SyncScanWorker` |
| **Incoming Tracks table** | Tree | Displays incoming files with metadata |
| Table columns | | Track name, Status, Distance, **Flag, Note** |
| **Right-click context menu** | Menu | 📋 Copy, ↻ Replace, ✕ Clear flag, 📝 Add note |
| Plan summary card | Card | Shows counts and status distribution |
| Build Plan | Button (primary) | Computes move/copy plan |
| Preview Plan | Button | Opens HTML preview in browser |
| Execute Plan | Button (primary) | Runs the plan, writes execution report |
| Copy / Move toggle | Toggle | Whether to copy or move files |

**User interactions:**
- Right-click incoming track to flag for copy/replace
- Add notes to explain flagging decisions
- Flags override auto-decisions when plan is built
- Preview HTML shows how flags affect the plan

Worker calls `library_sync.compare_libraries()` and `library_sync.build_library_sync_preview()`.

### 6.4 Similarity Inspector (`similarity.py`)

Targeted two-file duplicate diagnostic.

| Element | Type | Purpose |
|---|---|---|
| File A path + Browse | Row | First audio file |
| File B path + Browse | Row | Second audio file |
| Advanced options | Collapsible card | Fingerprint offset, trimming, threshold overrides |
| Run Inspection | Button (primary) | Launches `SimilarityWorker` |
| Report card | Card | Full threshold breakdown: codec, duration, raw distance, verdict |

### 6.5 Tag Fixer (`tag_fixer.py`)

AcoustID / MusicBrainz tag correction workflow.

| Element | Type | Purpose |
|---|---|---|
| Service selector | ComboBox | AcoustID / MusicBrainz / Last.fm |
| Scan Library | Button (primary) | Launches `TagFixWorker` |
| Proposals table | `QTableWidget` | Columns: File, Field, Current, Proposed; checkbox per row |
| Select All / Deselect All | Buttons | Bulk selection |
| Apply Selected | Button (primary) | Writes accepted proposals to file tags |

### 6.6 Genre Normalizer (`genres.py`)

Batch genre tag update via MusicBrainz or Last.fm.

| Element | Type | Purpose |
|---|---|---|
| Service selector | ComboBox | MusicBrainz / Last.fm |
| Dry Run | Checkbox | Preview without writing tags |
| Overwrite Existing | Checkbox | Replace genres already tagged |
| Run Genre Update | Button (primary) | Launches `GenreWorker` |
| Progress bar | Progress bar | File processing progress |
| Results card | Card | Count summary: updated / skipped / errors |

### 6.7 Playlist Generator (`playlists.py`)

Four sub-panels in a tab bar:

| Tab | Purpose |
|---|---|
| Folder Playlists | Generates one `.m3u` per album/artist folder |
| Tempo + Energy | Buckets tracks by BPM + energy level; writes named playlists |
| Auto-DJ | Chains tracks for smooth transitions using similarity scoring |
| Playlist Repair | Finds and fixes broken paths inside existing `.m3u` files |

Each tab has a Run button that launches a `PlaylistWorker(QThread)`.

### 6.8 Clustered Playlists (`clustered.py`)

K-Means and HDBSCAN clustering over audio features.

| Element | Type | Purpose |
|---|---|---|
| Algorithm selector | ComboBox | K-Means / HDBSCAN |
| Feature checkboxes | Checkboxes | Tempo, Energy, Danceability, Valence, Loudness, … |
| Cluster count | SpinBox | Target cluster count (K-Means only) |
| Run Clustering | Button (primary) | Launches `ClusterWorker` |
| Open Visual Graph | Button | Switches to the Graph workspace with cluster data loaded |

### 6.9 Visual Music Graph (`graph.py`)

Status-only workspace. Checks whether cluster data is available and launches
`ClusterGraphPanel` (the interactive matplotlib scatter plot) in a new window.

### 6.10 Player (`player.py`)

In-app audio playback via libVLC.

| Element | Type | Purpose |
|---|---|---|
| Track metadata card | Card | Title, artist, album, codec, duration |
| Waveform / album art area | Label | Placeholder for artwork or waveform display |
| Transport controls | Buttons | ⏮ Previous, ⏪ Seek back, ⏯ Play/Pause, ⏩ Seek forward, ⏭ Next |
| Progress slider | Slider | Scrub position; updates in real time |
| Volume slider | Slider | 0–100 % |
| Open File | Button | Load a single audio file |

### 6.11 Compression (`compression.py`)

Library-wide format conversion and archiving.

| Element | Type | Purpose |
|---|---|---|
| Target format | ComboBox | MP3 / AAC / FLAC / OGG |
| Bitrate | SpinBox | Output bitrate (kbps) |
| Archive originals | Checkbox | Zip source files after conversion |
| Start | Button (primary) | Launches compression worker |
| Progress | Progress bar | Per-file progress |

### 6.12 Export & Utilities (`tools.py`)

Five-tab panel of miscellaneous tools.

| Tab | Contents |
|---|---|
| Artist/Title Export | Path entry, delimiter selector, Export button — writes text file of all artist · title pairs |
| Codec List Export | Export button — writes text file of all file paths grouped by codec |
| File Cleanup | Finds and lists non-audio files in the library; Move to Trash / Delete buttons |
| Diagnostics | Library health check: missing tags, unexpected folder structures, orphaned files |
| Validator | Runs `validator.py` against the library and displays the result report |

### 6.13 Help (`help.py`)

| Element | Purpose |
|---|---|
| Documentation links | Buttons linking to `docs/project_documentation.html`, `docs/gui_inventory.md`, etc. |
| Keyboard shortcuts table | Full reference of all Ctrl+key shortcuts |
| About card | Version, author, GitHub link |

---

## 7. Dialogs

### 7.1 SettingsDrawer

Modal `QDialog`. Opened by ⚙ Settings or Ctrl+,.

Sections (not yet fully implemented — placeholder structure):

- Metadata Services (API key, service selector)
- Library defaults (reserved folder names)
- Fingerprint thresholds
- Playback (VLC path)

Emits `settings_saved()` when accepted; `AlphaDEXWindow` reloads the library
path if it changed.

### 7.2 ThemePickerDialog

See §5.3.

### 7.3 AutoThemeDialog

See §5.4.

---

## 8. Signals flow summary

```
TopBar.library_changed ──► AlphaDEXWindow._on_library_changed
                              └─► ws.set_library_path() for all workspaces
                              └─► config.save_config()
                              └─► _StatsWorker.start()

TopBar.theme_requested ──► ThemePickerDialog (non-modal)
                              └─► ThemeManager.apply(key)
                                   └─► AlphaDEXStyle applied app-wide
                                   └─► QPalette updated
                                   └─► theme_changed signal
                                        └─► ws.refresh_shadows() for all workspaces

Sidebar.nav_changed ─────► AlphaDEXWindow._on_nav_changed
                              └─► QStackedWidget.setCurrentWidget(ws)

WorkspaceBase.log_message ──► LogDrawer.append()
WorkspaceBase.status_changed ─► LogDrawer.set_status() + QStatusBar
```
