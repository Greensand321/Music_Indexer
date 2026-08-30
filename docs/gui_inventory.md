# AlphaDEX GUI Inventory

Plain-English reference for every screen, panel, control, and dialog in the
**PySide6 GUI** (`alpha_dex_gui.py` → `gui/main_window.py`).

> The legacy Tkinter interface (`main_gui.py`) is still functional but is no
> longer the primary GUI. This document describes the PySide6 replacement only.

> **Verified against the code 2026-07-22.** The previous version of this document
> had drifted significantly from the actual app in several sections (startup
> sequence, Player, Compression, Utilities, Duplicates, Clustered Playlists,
> Settings). Every section below has been re-checked against the source; places
> where a control looks wired but currently has no effect are called out
> explicitly rather than silently omitted, since knowing a checkbox is decorative
> is as useful as knowing what a real one does.

---

## 0. Startup Sequence

Handles the initial application loading and library selection before showing the
main window.

**The `SplashScreen` class (`gui/widgets/splash.py`) is dead code.** It's fully
built — a branded two-phase loading bar with a spinner — but nothing in the app
instantiates or shows it any more. A comment left in the landing screen's code
confirms this was intentional, not an oversight: a short pause between the
landing window's fade-in and its tiles beginning to animate is described as
"the merged 'logo moment' that replaces the old SplashScreen." The splash
screen was superseded by the mosaic landing described below and simply never
removed from the repository.

**What actually runs**, in order:

| Component | Purpose |
|---|---|
| `MosaicLanding` (`gui/widgets/landing.py`) | The real first thing you see. Presents an animated grid of album art tiles built from your previously selected library (or lets you pick a new one if none is saved). Validates the library path, builds the mosaic with a "logo moment" pause before the tiles fly in, and shows a call-to-action button. |
| `AlphaDEXWindow` (`gui/main_window.py`) | Constructed on a deferred timer tick (scheduled for the very next event-loop cycle) specifically so the landing screen can appear and start its fade-in before the heavier main-window import runs. The two windows then cross-fade into each other. |

The landing's call-to-action button is **not** labeled "Continue" — it reads
**"Initialize"** if you have a previously saved library, or **"Choose Library
Folder"** if you don't. Clicking directly on one of the mosaic's individual
album-art tiles skips the button entirely: it accepts that library *and*
navigates straight into the Player workspace, already playing that folder — a
"click the art to play it" shortcut with no equivalent mention anywhere else in
the app.

The genuinely expensive work (scanning and fingerprinting your library) is
deliberately postponed until whichever comes first: 30 seconds after the
landing screen first appears, or 3 seconds after you press the call-to-action
button — so the opening animation never stutters behind a frozen progress bar.

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
│   (360 px)   │                                              │
│              ├──────────────────────────────────────────────┤
│              │  Now Playing bar (hidden until a track loads)│
│              ├──────────────────────────────────────────────┤
│              │  Log Drawer (slide-up, collapsed by default) │
├──────────────┴──────────────────────────────────────────────┤
│  Status Bar                                                  │
└─────────────────────────────────────────────────────────────┘
```

The **Now Playing bar** (`gui/widgets/now_playing_bar.py`) is a persistent
transport strip — art thumbnail, title/artist, prev/play-pause/next, seek
slider, volume slider — that sits above the log drawer and stays visible no
matter which workspace you're in once a track has started playing. It is
two-way wired to the Player workspace (now-playing/position/state signals one
direction, transport commands and volume the other), including a
shared-volume-slider trick so the two volume controls never fight each other
when one is adjusted programmatically.

---

## 2. Top Bar (`gui/widgets/top_bar.py`)

Fixed height 64 px. Always visible.

| Element | Type | Purpose |
|---|---|---|
| App title "AlphaDEX" | Label (`#appTitle`) | Brand / identity |
| Library path | Label (`#libraryPath`) | Full path of the currently open library |
| Library stats | Label (`#libStats`) | Track count · GB · artist count (populated after background scan) |
| "Change Library" | Button | Opens a `QFileDialog` folder picker; emits `library_changed(path)` |
| "Theme" | Button | Opens `ThemePickerDialog` (non-modal swatch grid) |
| "Settings" | Button | Opens `SettingsDrawer` dialog |

The three action buttons are plain text labels — there are no emoji glyphs
baked into them, unlike the sidebar's nav items. (A small `⌂` glyph does
appear as a separate prefix on the library-path label itself, but not on any
button.)

**Signals emitted**

| Signal | When |
|---|---|
| `library_changed(str)` | User selects a new library folder |
| `settings_requested()` | User clicks Settings |
| `theme_requested()` | User clicks Theme |

---

## 3. Sidebar (`gui/widgets/sidebar.py`)

Fixed width **360 px** (not 220 px as an earlier version of this document
claimed). Dark background (token `sidebar_bg`). Contains a `QScrollArea` so
the nav list can scroll on small displays.

### 3.1 Logo label

Static "AlphaDEX" label at the top.

### 3.2 Navigation items

Grouped into five sections, plus a separate **Exit** control pinned below a
divider at the bottom of the list — not part of the five workspace sections,
and easy to miss if you're only looking at the `NAV_STRUCTURE` groups. Each
item is an `AnimatedNavButton` (150 ms hover easing, badge support — though
see the note below). Clicking a workspace item emits `nav_changed(key)` to
the main window which switches the active workspace; clicking Exit emits
`exit_requested` and closes the app.

| Section | Key | Sidebar label |
|---|---|---|
| **ORGANIZE** | `indexer` | Indexer |
| | `library_sync` | Library Sync |
| **CLEAN UP** | `duplicates` | Duplicates |
| | `similarity` | **Similarity** |
| | `tag_fixer` | Tag Fixer |
| | `genres` | Genre Normalizer |
| **PLAYLISTS** | `playlists` | **Playlists** |
| | `clustered` | **Clustered** |
| | `graph` | **Music Graph** |
| **PLAYER** | `player` | Player |
| | `compression` | Compression |
| **TOOLS** | `tools` | **Utilities** |
| | `help` | Help |
| *(below divider)* | `exit` | Exit |

A handful of these labels are shorter than the underlying feature name might
suggest (e.g. "Similarity" rather than "Similarity Inspector," "Utilities"
rather than "Export & Utilities") — worth knowing since other documents in
this suite, and the app's own Help screen, sometimes refer to the fuller
feature name.

Active item renders with accent fill (`sidebar_active` token). `Sidebar.set_badge(key, n)`
exists and is built to show a numeric badge at an item's trailing edge, but
nothing in the app currently calls it — no workspace posts a badge count
today, so this is a built, currently-unused capability rather than a visible
feature.

**Keyboard shortcuts** (set in `main_window.py`):

| Shortcut | Action |
|---|---|
| Ctrl+1 … Ctrl+9 | Switch to the first 9 workspaces in sidebar order (Indexer → Music Graph). Player, Compression, Utilities, and Help have no shortcut. |
| Ctrl+O | Change library (same as the Top Bar button) |
| Ctrl+, | Open Settings |
| Ctrl+L | Toggle log drawer |
| Ctrl+W | Clear log |

---

## 4. Log Drawer (`gui/widgets/log_drawer.py`)

A slide-up panel anchored to the bottom of the content area, below the Now
Playing bar. Collapsed by default (0 px); expands to **220 px** on toggle or
automatically when an error/warning message arrives.

| Element | Type | Purpose |
|---|---|---|
| Handle bar | Clickable strip | Shows current status chip; click to toggle open/close |
| Status chip | Label (coloured) | One-line summary of the last operation status |
| Log text area | `QPlainTextEdit` (read-only) | Colour-coded log |
| Clear button | Button (in handle bar) | Clears the log text |

`append(message, level)` — called by workspaces via the `log_message` signal.
The four levels actually used across the codebase are `"info"`, `"ok"`,
`"warn"`, and **`"error"`** (not the abbreviation `"err"` an earlier version
of this document listed — every workspace's own logging call uses the full
word).

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

A design note left in the theme engine explains why a residual stylesheet
still exists alongside the custom `QProxyStyle`: "AlphaDEXStyle handles ALL
widget painting. This residual [stylesheet] is only for the handful of things
QProxyStyle cannot reach: tooltip sizing, plain-text area font, and explicit
selection-colour anchoring."

### 5.2 Themes

14 named themes persisted to config as `"theme"`.

**Dark (8)**: Midnight, Obsidian, Graphite, Navy, Twilight, Aurora, Forest, Ember

**Light (6)**: Pearl, Azure, Blossom, Meadow, Lavender, Sunset

### 5.3 ThemePickerDialog (`picker.py`)

Non-modal dialog opened by the Theme button.

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

### 5.5 The shared background glow

Every workspace and the sidebar paint against the same two-corner accent-glow
background (`gui/widgets/gradient_bg.py`, `GradientWidget`), togglable in
Settings → General → "Background gradient." A comment on the sidebar's own
paint routine explains the intent behind this precisely: *"The
window-spanning gradient is painted at the same coordinates used by
GradientWidget, so the two glows read as a single canvas that stretches
across both the sidebar and the workspace... just enough to hint at the
panel boundary without breaking the illusion of one unified background."*

---

## 6. Workspaces

All workspaces inherit `WorkspaceBase` (`gui/workspaces/base.py`). Common
features:

- Wrapped in a `QScrollArea` so tall content is always reachable.
- `log_message(str, level)` signal routed to the log drawer.
- `status_changed(str, colour)` signal updates the log drawer handle and status bar.
- `navigate_requested(key)` signal asking the main window to switch to another
  workspace. Workspaces never change the active workspace themselves — they raise
  this and the main window performs the move, so navigation stays in one place.
- `_make_card()` returns a `QFrame` with rounded corners and a drop-shadow effect
  (shadow colour updates on theme change via `refresh_shadows()`).
- The scroll area's inner content widget is specifically a `GradientWidget`
  (see §5.5), not a plain `QWidget` — a comment in `base.py` notes plainly why:
  a plain `QWidget` would be transparent and the shared background glow
  wouldn't render.

### 6.1 Indexer (`indexer.py`)

Controls the file-organisation and rename pipeline.

| Element | Type | Purpose |
|---|---|---|
| Workflow stepper | Header row | Shows the three-step flow (Configure → Preview → Execute) as active/inactive labels |
| Configuration card | Card | Contains all run options |
| Dry Run | Checkbox | Preview only — writes HTML report, no files moved |
| Cross-Album Scan | Checkbox | Enables Phase C (across album boundaries) |
| Flush Cache | Checkbox | Clears fingerprint cache before run |
| Create Playlists | Checkbox | Generates `.m3u` playlists in `Playlists/` on full run |
| Max Workers | SpinBox | Thread pool size (1–32) |
| Run button | Button (primary) | Label changes dynamically: **"▶ Run Preview"** while Dry Run is checked, **"▶ Execute"** when it isn't |
| Cancel | Button | Cancels the background indexer |
| Progress card | Card | Three labeled progress bars: Phase A / Phase B / Phase C |
| Status label | Label | Current step description |
| Open Report | Button | Opens `Docs/MusicIndex.html` in the system browser |

Unchecking Dry Run and clicking the now-labeled "Execute" button triggers a
confirmation `QMessageBox` before anything actually runs — an extra safety
step not obvious from the button alone.

Worker: `IndexerWorker(QThread)` calls `music_indexer_api.run_full_indexer()`.

### 6.2 Duplicates (`duplicates.py`)

Review-first duplicate detection and disposal. Also has its own workflow
stepper header ("1. Scan & fingerprint → 2. Review groups → 3. Execute plan").

| Element | Type | Purpose |
|---|---|---|
| Scan options card | Card | Library path entry, threshold fields (`QDoubleSpinBox`, not sliders — exact / near / mixed-codec) |
| Start Scan / Cancel | Button (primary) | Launches `DupeScanWorker` |
| Scan progress bar | Progress bar | Fingerprint generation progress |
| Groups table | `QTreeWidget` | Columns are **`#`, Winner, Losers, Review** (not "Group, Files, Sizes, Codecs") |
| Inspector panel | Card | Shows selected group's file list with per-item details |
| Global disposition | Radio buttons | Two options: "Quarantine duplicates (safe default)" / "Delete losers permanently" — there is no separate "Retain" radio |
| Per-group disposition | `QComboBox` | Enabled once a group is selected, offering `Default (global) / Retain / Quarantine / Delete` — **this control currently has no change handler wired to it; selecting a value has no effect on the plan.** Treat it as not-yet-functional rather than a per-group override. |
| Execute Plan | Button (danger) | Applies the *global* disposition — writes execution report to `Docs/` |

### 6.3 Library Sync (`library_sync.py`)

Compare two libraries and execute a copy/move plan. This is one of the larger
and richer workspaces in the app — an earlier version of this document
significantly undersold it.

| Element | Type | Purpose |
|---|---|---|
| Source path entry + Browse | Row | Pick the existing library folder |
| Incoming path entry + Browse | Row | Pick the incoming library folder |
| Preset name field | Input | Names a saved threshold/config preset |
| Global threshold override | Input field | Fingerprint matching threshold (0.0–1.0) |
| Format overrides | Multi-line text box | Per-extension threshold overrides, `ext=value` syntax (e.g. `.flac=0.3`) |
| Recompute Matches | Button | Re-runs matching against the current thresholds without a full rescan |
| Save Session | Button | Persists the current scan session |
| Scan progress | Two progress bars | Existing lib scan + Incoming lib scan |
| Start Scan | Button (primary) | Launches `SyncScanWorker` |
| **Incoming Tracks table** | Tree | Displays incoming files with metadata |
| **Existing Tracks table** | Tree | A second panel — columns Track / Status / Best Matches — showing your existing library side |
| Match Inspector | Card | Text area + one-line summary describing the currently selected match in detail |
| Table columns (Incoming) | | Track name, Status, Distance, **Flag, Note** |
| **Right-click context menu** | Menu | Copy, Replace, Clear flag, Add note |
| Plan summary card | Card | Shows counts and status distribution |
| Build Plan | Button (primary) | Computes move/copy plan |
| Preview Plan | Button | Opens HTML preview in browser |
| Execute Plan | Button (primary) | Runs the plan, writes execution report |
| Copy / Move toggle | Toggle | Whether to copy or move files |
| **Export Report** | Button | Saves a report of every match, flag, and note as HTML, JSON, or CSV, once a scan has completed |
| Export Logs… | Button | Saves the session's activity log as a plain text file |

**User interactions:**
- Right-click incoming track to flag for copy/replace
- Add notes to explain flagging decisions
- Flags override auto-decisions when plan is built
- Preview HTML shows how flags affect the plan

Worker calls `library_sync.compare_libraries()` and `library_sync.build_library_sync_preview()`.

### 6.4 Similarity (`similarity.py`)

Sidebar label is "Similarity"; a targeted two-file duplicate diagnostic.

| Element | Type | Purpose |
|---|---|---|
| File A path + Browse | Row | First audio file |
| File B path + Browse | Row | Second audio file |
| Advanced options | Collapsible card | Fingerprint offset, trimming, threshold overrides |
| Run Inspection | Button (primary) | Launches `SimilarityWorker` |
| Report card | Card | Full threshold breakdown: codec, duration, raw distance, verdict |

### 6.5 Tag Fixer (`tag_fixer.py`)

AcoustID-driven tag correction workflow. Notably, **there is no service
selector in this workspace** — the automatic scan always uses the same lookup
path regardless; see **features/tag_fixer.md** for why MusicBrainz and
Last.fm aren't independently selectable here despite being real, working
services elsewhere in the app.

| Element | Type | Purpose |
|---|---|---|
| Dry run checkbox | Checkbox | Checked by default; blocks the Apply button with a message rather than implementing a separate write path |
| Fix fields checkboxes | Checkboxes | Title / Artist / Album / Genre |
| Scan Library | Button (primary) | Launches `TagFixWorker` |
| Proposals table | `QTableWidget` | Columns: ✓, File, Field, Current Value, Proposed Value, **Score** |
| Select All / Deselect All | Buttons | Bulk selection |
| Apply Selected | Button (primary) | Writes accepted proposals to file tags, after a confirmation prompt |

### 6.6 Genre Normalizer (`genres.py`)

Batch genre-tag update — see **features/tag_fixer.md** for the important
caveat that this label describes something different from the legacy app's
"Genre Normalizer."

| Element | Type | Purpose |
|---|---|---|
| Service selector | ComboBox | MusicBrainz / Last.fm / Both — **currently decorative; the backend only ever queries MusicBrainz regardless of this selection** |
| Dry Run | Checkbox | When checked, files are logged as "would process" without ever being looked up — the dry-run preview does not show what genres would actually be proposed |
| Overwrite Existing | Checkbox | **Currently decorative; has no effect.** The real (fixed) rule is: skip any file that already has two or more genre tags |
| Run Genre Update | Button (primary) | Launches `GenreWorker` |
| Progress bar | Progress bar | File processing progress |
| Results card | Card | Count summary: updated / skipped / errors |

### 6.7 Playlists (`playlists.py`)

Four sub-panels in a tab bar:

| Tab | Purpose |
|---|---|
| Folder Playlists | Generates one `.m3u` per album/artist folder |
| Tempo + Energy | Buckets tracks by BPM + energy level; writes named playlists |
| Auto-DJ | Chains tracks for smooth transitions using similarity scoring |
| Playlist Repair | Finds and fixes broken paths inside existing `.m3u` files |

Two controls on the Tempo + Energy tab are currently decorative: the **Tempo
range** and **Energy range** spin boxes are captured but never passed to the
bucketing function, which always uses its own fixed tiers (see
**features/playlists_and_clustering.md**) regardless of what you set here.
The **"Prefer Opus when searching for missing FLAC"** checkbox is similarly
built but not read anywhere.

Each tab has a Run button that launches a `PlaylistWorker(QThread)`.

### 6.8 Clustered (`clustered_enhanced.py`)

The workspace the sidebar's "Clustered" entry actually opens. (An older,
simpler workspace, `clustered.py`, still exists in the repository but is dead
code — nothing imports it any more, and it isn't reachable from the app.)

Three tabs:

| Tab | Contents |
|---|---|
| 🚀 Quick Start | A single fixed "recommended settings" run — one button, no configuration. |
| ⚙ Advanced | Opens the **Clustering Wizard** dialog (see §7.2 below); shows the resulting configuration as read-only text; has its own Run button. |
| 📊 Results | Progress bar and live log during the run, then a results panel — cluster count, silhouette score, and three buttons: "View Quality Report," "View Playlists," "Open Visual Graph." |

The feature checkboxes exposed via the wizard are `tempo`, `mfcc`, `chroma`,
`spectral`, `energy`, `onset_rate` — see **features/playlists_and_clustering.md**
for which of these the clustering engine actually computes today.

**"Open Visual Graph"** navigates to the Music Graph workspace by emitting the
shared `WorkspaceBase.navigate_requested(key)` signal, which the main window
connects for every workspace — workspaces never switch themselves. (It used
to pop a message box telling you to click the sidebar yourself.)

Still worth knowing: **"View Playlists"** opens the generated-playlists folder
via a Linux-specific `xdg-open` call with no Windows or macOS fallback — it
will not work as described on those platforms.

### 6.9 Music Graph (`graph.py`)

A status-and-launch panel, not an interactive graph in itself. It checks
whether cluster data is available (inspecting `Docs/cluster_info.json`) and,
via its "Open 3D Graph" button, opens a **browser-based** 3-D visualization
(Three.js/WebGL) generated by `cluster_graph_3d.py`. Also offers "Regenerate
HTML" and "Test with Demo Data" buttons.

Controls on the generated page itself: orbit/zoom, hover tooltip (cluster
swatch, track, metadata), click-to-select with CSV / `.m3u` export, a legend
whose rows toggle per-cluster visibility, a spread slider with 1×/10×/30×/50×/100×
presets, axes / grid / orbit-ring toggles, auto-rotate, and an **Import JSON**
button for loading a different cluster file into the same viewer. The page is
self-contained: the library's data is inlined at generation time, and when the
template is opened directly with no data injected it falls back to a built-in
sample set.

This workspace does **not** use the in-app `InteractiveScatterPlot`,
`Interactive3DScatterPlot`, `ClusterLegendWidget`, or `TrackDetailsPanel`
widgets described in §8 — those are fully built but currently orphaned; see
**features/playlists_and_clustering.md** for the detail.

### 6.10 Player (`player.py`)

A full-featured, in-app audio player — considerably more elaborate than
"transport controls plus a waveform placeholder." There is, in fact, no
waveform anywhere in the current implementation.

| Element | Type | Purpose |
|---|---|---|
| Library table | Sortable/filterable `QTableWidget` | Search box; right-click menu (Play now / Add to queue / Open folder / Save selection as M3U) |
| Queue panel | List + buttons | Add Selected / Clear / Save Queue as M3U; its own right-click menu |
| Recently Played | Collapsible list | Up to 30 entries, de-duplicated |
| Now-playing covers | Three-panel display | Previous / current / next album art in a cover-flow arrangement |
| Blurred background art | Full-panel backdrop | A blurred, cross-fading rendition of the current track's art behind the library table, for ambience without hurting readability |
| Hover popup | Tooltip-like panel | Appears after a 1.5-second dwell over a table row, showing art and metadata |
| Transport controls | Buttons | Previous, seek back, play/pause, seek forward, next |
| Progress slider | Slider | Scrub position; updates in real time |
| Volume slider | Slider | 0–100%, kept in sync with the Now Playing bar's own volume control |
| Shuffle / Repeat | Toggles | Off/track/all repeat modes, with shuffle history |
| 30-second preview mode | Toggle | Seeks to the 30-second mark and auto-advances after 15 seconds — a "highlight reel" listening mode |
| Load M3U / Save Queue as M3U | Buttons | Playlist import/export |
| Open File | Button | Load a single audio file |

**Keyboard shortcuts** (scoped to this workspace, not in the app-wide table in
§3.2): `Space` play/pause, `N`/`P` next/previous, `Left`/`Right` seek ±5s,
`Shift+Left`/`Shift+Right` seek ±30s.

One piece of vestigial code worth knowing about if you're reading the source:
a background art-loading thread still fires on every single click in the
library table, but the handler it feeds is an explicit no-op — a comment in
the code notes it as "old cover-flow back-cover logic removed." The work
still runs; nothing uses the result any more.

### 6.11 Compression (`compression.py`)

**A fixed FLAC → Opus library mirror tool** — not a general format
converter. An earlier version of this document described a Target
Format/Bitrate/Archive combination that does not exist in the current
implementation; output is always Opus, and there is no zip/archive option.

| Element | Type | Purpose |
|---|---|---|
| Source folder | Path entry + Browse | The library to mirror |
| Destination folder | Path entry + Browse | Where the Opus copy is written — always a separate folder, never in-place |
| Opus bitrate | SpinBox | 48–320 kbps |
| Overwrite existing files in destination | Checkbox | Whether to re-convert files already present at the destination |
| Estimate Space Savings | Button | Pre-computes total FLAC byte count and predicted Opus size before you commit to running anything |
| Start Compression | Button (primary) | Gated on FFmpeg being found on `PATH`; launches the mirror worker |
| Progress | Progress bar | Per-file progress |
| Open Report | Button | Opens a generated HTML summary of the mirror run |

### 6.12 Utilities (`tools.py`)

A grid of five "liquid glass" tile cards (`TileGrid`/`ToolTile` — hover-glow,
animated drawer), **not** a `QTabWidget` as an earlier version of this
document stated. Minimum tile width is 500 px.

| Tile | Actual contents |
|---|---|
| Artist/Title Export | Two toggle switches — "Exclude FLAC files" and "Include per-album duplicate titles" — and an Export button. There is no path entry or delimiter selector; output is always written to `Docs/artist_title_list.txt`. |
| Codec List Export | Per-extension checkboxes (`.flac`/`.mp3`/`.m4a`/`.aac`/`.wav`/`.opus`/`.ogg`), a "Filenames only (no full paths)" toggle, and an Export button. |
| **File Cleanup** | **Not** a non-audio-file trash tool. It strips duplicate-download artifacts (trailing `" (1)"`, `" (2)"`, `" copy"`) from **audio** filenames and repairs any playlist references afterward. |
| Diagnostics | A responsive 4-column grid of buttons launching diagnostic dialogs: M4A Tester, Opus Tester, Bucketing POC, Scan Engine, Fuzzy Finder, Pair Review, View Crash Log. |
| Validator | Runs `validator.py` against the library and displays the result report. |

### 6.13 Help (`help.py`)

| Element | Purpose |
|---|---|
| Documentation links | Buttons linking to `docs/project_documentation.html`, `docs/gui_inventory.md`, and other reference docs |
| Keyboard shortcuts table | Full reference of all Ctrl+key shortcuts |
| About card | Version, author, GitHub link; also states the app's former name ("AlphaDEX (formerly SoundVault)") and the location of the config file |

One of the documentation links currently points at a file that doesn't
exist (`docs/library_sync_redesign.md`, which was archived and superseded by
`docs/features/library_sync.md`) — clicking it produces a "Not Found"
message box rather than opening anything.

---

## 7. Dialogs

### 7.1 SettingsDrawer

Modal `QDialog`. Opened by the Settings button or Ctrl+,.

**This is not a placeholder** — an earlier version of this document described
it as "not yet fully implemented," which is no longer accurate. It has three
real tabs:

1. **Metadata Services** — a service dropdown populated from the app's list
   of known services, with the not-yet-implemented ones (Spotify, Gracenote)
   shown but disabled and tooltipped rather than hidden; an AcoustID API key
   field; MusicBrainz app name/version/contact fields; and a "Test
   Connection" button that runs a real background connectivity check.
2. **General** — default library path, near-duplicate and exact-duplicate
   threshold spin boxes, and the "Background gradient" toggle described in
   §5.5.
3. **Advanced** — genuinely still a stub (an empty tab), with a comment
   noting the relevant settings currently live in the Metadata Services tab
   instead.

There is no "Library defaults (reserved folder names)" section and no VLC
playback-path field anywhere in the current dialog, despite an earlier
version of this document listing both.

Emits `settings_saved()` when accepted; `AlphaDEXWindow` reloads the library
path if it changed.

### 7.2 ClusteringWizardDialog

A genuine 5-step wizard (`gui/dialogs/clustering_wizard_dialog.py`), reached
from the Clustered workspace's Advanced tab, not previously documented here:

| Step | Contents |
|---|---|
| 1. Feature Selection | Checkboxes for tempo / mfcc / chroma / spectral / energy / onset_rate, each with an explanatory tooltip, plus Fast / Balanced / Complete presets. |
| 2. Normalization & Preprocessing | Standard / MinMax / Robust normalization choice, plus None / PCA / t-SNE / UMAP dimensionality-reduction choice. |
| 3. Algorithm Selection | K-Means (with a cluster-count spin box) or HDBSCAN (with min-cluster-size and min-samples spin boxes). |
| 4. Post-Processing | Optional removal of clusters below a size threshold; optional merging of small clusters into a "Miscellaneous" bucket. |
| 5. Output Options | Create M3U playlists / generate a quality report / open the interactive graph — checkboxes. |

Back/Next/Cancel navigation with a progress bar; refuses to proceed if you
uncheck every feature.

### 7.3 ClusterQualityReportDialog

Shows Silhouette Score / Davies-Bouldin Index / Calinski-Harabasz Score with
color-coded verdicts, plus a per-cluster breakdown and heuristic
improvement suggestions. (This dialog previously failed to render its
per-cluster section against any real result — it called a helper method
under the wrong name. Fixed.)

### 7.4 ThemePickerDialog

See §5.3.

### 7.5 AutoThemeDialog

See §5.4.

### 7.6 Diagnostics Dialogs

Various dialogs launched from the Utilities workspace.

| Dialog | Purpose |
|---|---|
| `MediaTesterDialog` | Tests file decoding for specific codecs (e.g., M4A, Opus). |
| `BucketingPocDialog` | Minimal UI for testing duplicate bucketing strategies. |
| `ScanEngineDialog` | Exposes all ten duplicate-scan tuning fields as text inputs (sample rate, analysis seconds, tolerances, fingerprint bands/thresholds). |
| `FuzzyDupeDialog` | Fuzzy duplicate finder UI. Several of its controls are currently decorative — the "Search fields" checkboxes and the entire "Fingerprint Filter" group (exact/near thresholds, mixed-codec boost, two more checkboxes) are rendered but not read by the actual search. Its "Send Matches to Duplicate Pair Review" button is a stub that logs the pairs and shows a "coming soon" message rather than opening the Pair Review dialog. |
| `PairReviewDialog` | A fuller tool than a one-line summary suggests: two-panel side-by-side comparison (cover art, winner badge, tags) with Prev/Next/Switch/"Prefer MP3"/jump-to-pair navigation, keyboard shortcuts (Enter = confirm delete, Backspace = skip, arrows = navigate), and automatic playlist repair when a file is deleted. |
| `CrashLogDialog` | Displays the most recent application crash log. |

---

## 8. Widgets (`gui/widgets/`)

| Widget | Status |
|---|---|
| `top_bar.py` (`TopBar`) | Live — see §2. |
| `sidebar.py` (`Sidebar`) | Live — see §3. |
| `log_drawer.py` (`LogDrawer`) | Live — see §4. |
| `now_playing_bar.py` (`NowPlayingBar`) | Live — see §1. |
| `gradient_bg.py` (`GradientWidget`) | Live — see §5.5. |
| `landing.py` (`MosaicLanding`) | Live — see §0. |
| `splash.py` (`SplashScreen`) | **Dead code** — built, never invoked. See §0. |
| `interactive_scatter_plot.py` (`InteractiveScatterPlot`) | **Orphaned** — a real, working PyQtGraph 2D cluster scatter with lasso/rect selection and hover tooltips, imported nowhere except a stale integration script. See **features/playlists_and_clustering.md**. |
| `interactive_3d_scatter.py` (`Interactive3DScatterPlot`) | **Orphaned** — a real, working OpenGL 3D cluster scatter with camera presets, imported nowhere in the live app. |
| `cluster_legend.py` (`ClusterLegendWidget`) | **Orphaned** — a per-cluster show/hide checklist with color swatches, not wired into any live workspace. |
| `track_details_panel.py` (`TrackDetailsPanel`) | **Orphaned** — an album-art + metadata panel meant to accompany the scatter plot above; same status. |
| `audio_preview.py` | Lives directly under `gui/`, not `gui/widgets/`, despite the sibling naming — worth knowing if you go looking for it by directory. |

`gui/qt_launcher.py` is worth a separate mention: it's a standalone,
self-contained "AlphaDEX Qt Preview" window with placeholder/no-op behavior
throughout (its own log even labels one action "(placeholder)"). It is not
imported by the real entry point and is not part of the shipped app — more a
leftover proof-of-concept than a hidden feature.

---

## 9. Signals flow summary

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

Sidebar.exit_requested ──► AlphaDEXWindow.close()

WorkspaceBase.log_message ──► LogDrawer.append()
WorkspaceBase.status_changed ─► LogDrawer.set_status() + QStatusBar
WorkspaceBase.navigate_requested ─► AlphaDEXWindow._on_nav_changed
                              └─► activates sidebar item + switches workspace
                                   (e.g. Clustered "Open Visual Graph" → graph)

PlayerWorkspace ⇄ NowPlayingBar   (two-way: now-playing/position/state one way,
                                    play-pause/next/prev/seek/volume the other;
                                    volume sliders kept in sync without echoing)

MosaicLanding.tile_clicked ──► AlphaDEXWindow.play_directory()
                              └─► navigates to Player workspace, begins playback
```
