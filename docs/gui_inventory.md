# AlphaDEX GUI Inventory

Plain-English reference for every screen, panel, control, and dialog in the
**PySide6 GUI** (`alpha_dex_gui.py` → `gui/main_window.py`).

> The legacy Tkinter interface (`main_gui.py`) is still functional but is no
> longer the primary GUI. This document describes the PySide6 app only.

> **Verified against the code 2026-11-07**, after the fix phase that closed about
> 200 bugs across every page (see `docs/redesign/README.md` for the bug inventory).
> Many pages changed behaviour: most now preview first, several controls that used
> to do nothing are now wired, and the ones that still have no backend are
> **disabled with a tooltip that says so** instead of looking live. Where a control
> is still decorative or a feature is still partial, this document says so plainly.

---

## 0. Startup Sequence

Handles the initial loading and library selection before the main window appears.
The sequence lives in `alpha_dex_gui.py` (`Startup` / `build_startup()`, which the
test harness drives without `app.exec()`).

**The `SplashScreen` class (`gui/widgets/splash.py`) is dead code.** It is fully
built but nothing shows it. The landing's short "logo moment" pause replaced it.

**What actually runs**, in order:

| Step | What happens |
|---|---|
| Crash log | `main()` first installs the Qt crash log (`crash_logger.install_for_qt()`; see architecture.md). |
| `MosaicLanding` (`gui/widgets/landing.py`) | Shown immediately. The window fades in with only the call-to-action card (the logo moment), then album-art tiles fly in, filled by a background art scanner (`_ArtScanner`, cancellable, registered with the JobRegistry). |
| `AlphaDEXWindow` (`gui/main_window.py`) | Built once the landing has **fully faded in** (`faded_in` signal), during the logo pause, so the first landing frames are never blocked. If the user clicks before that, it is built on demand. |
| Cross-fade | When a library is chosen, the landing fades out while the main window fades in over the same duration. |
| Library applied | The chosen library is applied **once**, after the cross-fade finishes (`window.set_library(path)`). The main window never applies the saved library by itself. |

The call-to-action button reads **"Initialize"** when the saved library exists, or
**"Choose Library Folder"** when there is none. A saved library that no longer exists
is shown as **"Saved library not found"** with "Choose Library Folder" (the folder
dialog opens at the nearest folder that still exists).

Clicking an album-art tile skips the button: it accepts the saved library *and* opens
the Player already playing that album folder. The first click wins; later clicks
during the exit animation are ignored. Tiles only carry a folder when the art scanner
found a cover in a saved library.

If `AlphaDEXWindow()` raises, the app shows an error dialog, writes the traceback to
stderr and the crash log, and exits with status 1.

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

**One validated library path.** The Top Bar's Change Library (and Ctrl+O), the
Settings dialog's Default library, and the startup sequence all end in
`_apply_library()`. It rejects a folder that does not exist (with a message; the
current library is kept), does nothing for the library that is already active, and
only then fans the path out to every workspace, refreshes the top-bar statistics, and
saves `library_root`.

**Closing the window** is safe while jobs run (see architecture.md, "Background jobs").
Read-only jobs are cancelled silently. If any job may be writing files, a "Background
tasks running" question lists them; on Yes the window disables its UI, cancels every
job cooperatively, and closes once they stop. If a job ignores cancellation for 20 s,
the user is asked once whether to quit anyway.

### 1.1 Now Playing bar (`gui/widgets/now_playing_bar.py`)

A slim transport strip above the log drawer, hidden until the first track loads and
hidden again when nothing is loaded any more.

| Element | Purpose |
|---|---|
| Art thumbnail | 44 px cover; a ♪ placeholder when the track has no art |
| Title / artist | Current track |
| ⏮ / ▶ / ⏭ | Previous, play/pause, next (routed to the Player workspace) |
| Seek slider | Drag, wheel, keys and groove clicks all seek |
| Volume slider | Kept in sync with the Player page's slider in both directions, without echo |
| ⌨ toggle | Turns **keyboard mode** on and off (see §6.10) |

---

## 2. Top Bar (`gui/widgets/top_bar.py`)

Fixed height 64 px. Always visible.

| Element | Type | Purpose |
|---|---|---|
| App title "AlphaDEX" | Label (`#appTitle`) | Brand |
| Library path | Label (`#libraryPath`) with a `⌂` glyph | The last two path segments (`…  Music  /  Library`); the tooltip holds the full path. Muted colour when no library is set. |
| Library stats | Labels | Tracks · GB · artists, from a background scan. The scan skips the reserved folders and unreadable files; "artists" counts `Artist/Album/track` folders. A newer library's scan cancels and replaces an older one. Shows "–" until it finishes. |
| Change Library | Button | Folder picker; emits `library_changed(path)` |
| Theme | Button | Opens the theme picker (one non-modal picker, reused) |
| Settings | Button | Opens the Settings dialog |

**Signals:** `library_changed(str)`, `settings_requested()`, `theme_requested()`.

---

## 3. Sidebar (`gui/widgets/sidebar.py`)

Fixed width 360 px, token `sidebar_bg`, inside a `QScrollArea`.

### 3.1 Logo

"AlphaDEX" label painted in the theme's primary text colour (it used to be a fixed
near-white that vanished on light themes).

### 3.2 Navigation items

Five sections from `NAV_STRUCTURE`, plus an **Exit** item pinned below a divider. Each
item is an `AnimatedNavButton` whose hover tint eases through the motion tokens; the
active item has an accent "pill" that slides between items. Clicking emits
`nav_changed(key)`; Exit emits `exit_requested`, which closes the window (with the
close protocol above).

| Section | Key | Sidebar label |
|---|---|---|
| **ORGANIZE** | `indexer` | Indexer |
| | `library_sync` | Library Sync |
| **CLEAN UP** | `duplicates` | Duplicates |
| | `similarity` | Similarity |
| | `tag_fixer` | Tag Fixer |
| | `genres` | Genre Normalizer |
| **PLAYLISTS** | `playlists` | Playlists |
| | `clustered` | Clustered |
| | `graph` | Music Graph |
| **PLAYER** | `player` | Player |
| | `compression` | Compression |
| **TOOLS** | `tools` | Utilities |
| | `help` | Help |
| *(below divider)* | `exit` | Exit |

`Sidebar.set_badge(key, n)` is built but nothing calls it.

**Keyboard shortcuts** (set in `main_window.py`):

| Shortcut | Action |
|---|---|
| Ctrl+1 … Ctrl+9 | The first nine sidebar items, in sidebar order (Indexer → Music Graph). The order and the status-bar labels are derived from `NAV_STRUCTURE`, so they can't drift from the sidebar. Player, Compression, Utilities and Help have no shortcut. |
| Ctrl+O | Change library |
| Ctrl+, | Open Settings |
| Ctrl+L | Toggle log drawer |
| Ctrl+W | Clear log |

When a page is shown, the window calls its optional `on_activated()` hook (no page
defines one yet; the Music Graph refreshes itself from `showEvent` instead).

---

## 4. Log Drawer (`gui/widgets/log_drawer.py`)

A slide-up panel at the bottom of the content area. Collapsed by default; expands to
**220 px** on toggle, or by itself when a `warn` or `error` line arrives.

| Element | Purpose |
|---|---|
| Handle bar | Click to toggle; shows the status chip |
| Status chip | One-line status from the active page (`status_changed`) |
| Log text | Timestamped, colour-coded lines (`info`, `ok`, `warn`, `error`) |
| Clear | Clears the log |

Colours come from the theme's `log_*` tokens and follow theme switches, including the
colour of lines already shown. The slide moves a sheet inside a layout-free body, so
the window is laid out once per toggle rather than on every animation frame.

Messages a page logs while it is still being built are buffered and delivered once
the window has connected the page (`WorkspaceBase.flush_pending_logs`).

---

## 5. Theme Engine (`gui/themes/`)

### 5.1 ThemeManager (`manager.py`)

Singleton `QObject`.

- `apply(key)` — switch to a named theme or `"auto"`, and persist it as `theme`.
- `load_persisted()` — read the saved theme at startup.
- `configure_auto(dark_key, light_key)` — the Auto pair (`auto_dark_theme`, `auto_light_theme`).
- `theme_changed(tokens)` — emitted after every switch. Widgets that colour
  themselves (Top Bar, Player, log drawer, card shadows, …) listen to it.
- **Auto mode** uses the platform's colour scheme when it reports Light or Dark, and
  falls back to the clock (07:00–20:00 = light) when it reports Unknown, which many
  Linux desktops do. It re-checks every 15 minutes while Auto is on.

A theme switch takes about 60–80 ms (it used to freeze the window for over a second);
see architecture.md, "Themes and motion", for how.

### 5.2 Themes

14 named themes, persisted as `theme`. Every text role meets WCAG AA contrast on the
surfaces it is drawn on, in every theme (`tests/test_design_contrast.py`).

**Dark (8)**: Midnight, Obsidian, Graphite, Navy, Twilight, Aurora, Forest, Ember

**Light (6)**: Pearl, Azure, Blossom, Meadow, Lavender, Sunset

### 5.3 ThemePickerDialog (`picker.py`)

Non-modal; the Theme button raises the open picker instead of opening a second one.
Changes apply instantly.

| Element | Purpose |
|---|---|
| Auto card | Configure… and Select buttons; always names the configured night and day themes |
| Dark / Light grids | Swatch cards for all 14 themes |
| Swatch card | Mini preview with its name drawn in the theme's own text colour; checkmark when active |

### 5.4 AutoThemeDialog

Modal child of the picker ("Auto Theme — Select Pair"): a night-theme grid, a
day-theme grid, Cancel, and **Apply Auto Theme**, which saves the pair and turns Auto on.

### 5.5 The shared background glow and card shadows

Every workspace and the sidebar paint the same two-corner accent glow
(`gui/widgets/gradient_bg.py`, `GradientWidget`), switchable in Settings → General →
"Background gradient". The setting is read once and cached; saving Settings refreshes
it.

The same `GradientWidget` also **paints the soft shadow under every card** (from a
cached 9-slice pixmap) instead of giving each card a graphics effect. Until this fix
no card had a visible shadow at all.

---

## 6. Workspaces

All workspaces inherit `WorkspaceBase` (`gui/workspaces/base.py`):

- Content sits in a `QScrollArea` whose inner widget is a `GradientWidget` (a plain
  `QWidget` would be transparent and the glow wouldn't show).
- Signals: `log_message(str, level)` → log drawer; `status_changed(str, colour)` →
  status chip and status bar; `navigate_requested(key)` → the main window switches
  page (pages never switch themselves); `play_tracks_requested(paths, label)` → the
  main window hands tracks to the Player.
- `_make_card()` returns a themed card and registers it for a painted shadow.
- **Background jobs:** `_start_worker(worker, name, ask_before_quit=..., library_write=...)`
  starts every `QThread` under the JobRegistry; `_library_writes_blocked(action)` is
  checked before any confirm dialog for a job that changes library files, and refuses
  ("Library Busy") while another such job runs. See architecture.md.
- Hooks the main window calls: `set_library_path()` (→ `_on_library_changed`),
  `on_settings_saved()` after Settings saves, and `shutdown()` when the app closes.

A pattern most pages now share: a **library change cancels read-only work** (scans,
previews, lookups) and ignores its late result, while a **write the user already
confirmed is left to finish** on the library it started on. (Duplicates is the
exception; see §6.2.) Most pages also have a workflow stepper that now advances with
the flow, and a neutral "Cancelled" state instead of reporting a cancel as an error.

### 6.1 Indexer (`indexer.py`)

Preview first; Execute applies **that** preview.

| Element | Type | Purpose |
|---|---|---|
| Workflow stepper | Header | 1. Preview (dry run) → 2. Review HTML report → 3. Execute changes |
| Dry run | Checkbox | Checked by default. Unchecking switches the Run button to Execute. |
| Create playlists after execution | Checkbox | Checked: regenerate playlists after the moves. Unchecked: only update existing playlists' paths. |
| Cross-album scan (Phase C) | Checkbox | **Disabled**, tooltip: the current indexer engine ignores this option. |
| Flush fingerprint cache before scan | Checkbox | **Disabled**, same tooltip. |
| Max workers | SpinBox | **Disabled**, same tooltip. |
| Run | Button (primary) | "▶ Run Preview" while Dry run is checked; "▶ Execute Previewed Plan" otherwise, enabled only while a preview of the current library exists. |
| Open Report | Button | Opens `Docs/MusicIndex.html` |
| Cancel | Button | Cancels the running preview or execute (the worker has its own cancel event) |
| Progress | Three bars | Phase A — Scanning, Phase B — Planning, Phase C — Applying changes (Execute only) |
| Status label + Run log | Label + text | Current step and the run's log lines |

Execute shows a confirmation with the plan's counts (files moved or renamed, how many
go to Manual Review, leftovers moved to Trash/Docs, folders removed, playlist mode).
The backend refuses a plan whose library changed since the preview. A plan runs at
most once; afterwards the page needs a fresh preview. A missing library folder is
refused rather than created.

Worker: `IndexerWorker` computes `music_indexer_api.IndexerPlan` (preview) or calls
`apply_indexer_plan()` (execute; a `LIBRARY_WRITES` job).

### 6.2 Duplicates (`duplicates.py`)

Scan → Preview → Execute. Stepper: 1. Scan & fingerprint → 2. Review groups → 3. Execute plan.
The page always works on the current library (there is no separate path field).

| Element | Type | Purpose |
|---|---|---|
| Exact / Near / Mixed-codec boost | `QDoubleSpinBox` ×3 | Start from the Settings keys (`exact_duplicate_threshold`, `near_duplicate_threshold`, `mixed_codec_threshold_boost`) and follow later Settings saves |
| Update playlists after execution | Checkbox | **Shown checked and disabled on purpose:** the executor always rewrites playlist entries that point at a moved loser, so it is not a choice. |
| Quarantine duplicates (safe default) / Delete losers permanently | Radio buttons | The global disposition |
| Scan Library | Button (primary) | Fingerprints and groups. If fpcalc or FFmpeg is missing, asks once (with install links) whether to scan using cached fingerprints only. |
| Preview | Button | Opens `Docs/duplicate_preview.html` |
| Execute | Button (danger) | See below. Disabled once the plan has run, until the next scan. |
| Open Report | Button | Opens the newest execution report, including one from an earlier session |
| Cancel | Button | Cancels the scan, or asks the execution to stop at the next safe point |
| Progress | Bar + label | |
| Groups table | `QTreeWidget` | Columns `#`, Winner, Losers, Review (the Review cell's tooltip lists the flags) |
| Group details + Disposition | Text + `QComboBox` | **Wired:** Default (global choice) / Keep loser in place (retain) / Quarantine / Delete permanently, per group |
| Log | Text | Per-file failures are capped at 20 lines |

**Execute** in order: refuses if another library write is running; if any group is
flagged for review, lists the flagged groups and why, and asks whether to **override**
(execute all groups exactly as previewed) or stop with nothing moved; writes the chosen
dispositions into the plan and re-exports the preview so the preview matches what will
run; confirms with counts (quarantined, **permanently deleted**, retained, tag and
artwork changes; a plan over 20,000 operations is called out); and, if anything will
be deleted, asks you to **type DELETE**, which opens the executor's double deletion
gate. Preflight refusals are reported with their reason.

The scan also writes `Docs/duplicate_preview.json`, which Utilities → Pair Review reads.

**Still different from the other pages:** a library change *cancels* a running
duplicate execution (it stops at the next safe point and writes its report to the old
library's `Docs/`), where the Indexer, Library Sync and the tag tools let a confirmed
write finish. See ROADMAP.md.

### 6.3 Library Sync (`library_sync.py`)

Scan both folders → review matches and flag → build plan → preview → execute that
plan. Stepper: 1. Configure paths → 2. Scan & match → 3. Build & execute plan.

| Element | Type | Purpose |
|---|---|---|
| Existing library / Incoming folder | Entry + Browse… | Validated before a scan: both set, both exist, not the same, not nested. Folders are never created. |
| Global threshold | Input | 0–1, validated |
| Preset name | Input | **Disabled**, tooltip "Not used yet: presets are not implemented." |
| Format overrides | Text | `.ext=value` per line; validated, extensions normalised to `.ext` |
| Recompute Matches | Button | **Wired:** re-matches the last scan with the current thresholds, without re-reading or re-fingerprinting files |
| Save Session | Button | Saves the existing/incoming paths and threshold (`sync_existing`, `sync_incoming`, `sync_threshold`); they are **restored** next time. Flags are not saved. |
| Scan Both Libraries / Cancel | Buttons | `SyncScanWorker`; Cancel stops a scan, plan build or execute |
| State label | Label | Idle / Scanning… / Re-matching… / Cancelling… |
| Existing Library / Incoming Folder | Two progress bars + status | Real progress from the scan |
| Incoming Tracks | Tree | Track, Status, Distance, **Flag**, **Note**; right-click: 📋 Copy, ↻ Replace, ✕ Clear flag, 📝 Add note |
| Matched Existing Tracks | Tree | **Populated:** each library track an incoming one matched — Track, Status ("Matched" / "Upgrade available"), Best Matches |
| Match Inspector | Text + summary | The selected match: paths, status, distance, confidence, quality scores, candidates, your flag and note, and its planned decision once a plan exists |
| Build Plan | Button | Builds the plan **from the reviewed matches and flags**, plus its HTML preview (`Docs/LibrarySyncPreview.html`) |
| Preview Plan | Button | Opens that preview |
| Copy Originals / Move | Toggle | Transfer mode; changing it requires rebuilding the plan |
| Execute Plan | Button (danger) | Runs the plan verbatim (a `LIBRARY_WRITES` job) |
| Export Report | Button | Matches, flags and notes as HTML, JSON or CSV |
| Write transfer playlist (.m3u8) | Checkbox | Passed to execute |
| Plan status + plan bar | Label + bar | Build and execute progress |
| Log + Export Logs… | Text + button | Humanised, batched log; lines from abandoned workers are dropped |

**The plan is tied to its inputs.** Changing a path, the transfer mode, a flag, or
re-scanning or switching library marks the plan out of date and disables Execute,
with the reason. The confirmation states library, incoming folder, copy vs move and
the counts (added, replaced with a backup, skipped, left for review). The result
names the run's own audit file and how many items were skipped because their files
changed since the plan was built. Flags survive a re-scan where they still apply; the
rest are listed in a notice.

### 6.4 Similarity (`similarity.py`)

A two-file duplicate diagnostic that uses **the Duplicate Finder's own pipeline and
settings**.

| Element | Type | Purpose |
|---|---|---|
| File A / File B | Entry + Browse… | The two audio files |
| Advanced Overrides | Checkable group (off by default) | Off: fingerprint exactly like the Duplicate Finder (reusing its cache) with the thresholds from Settings. On: trim silence, fingerprint offset / duration, silence threshold / min length, and exact / near / mixed-codec thresholds are used. An invalid field is named in a warning. |
| ⚖ Run Comparison / Clear | Buttons | Clear also discards a late result from a running comparison |
| Comparison Report | Text | Codecs, durations, distance, thresholds, verdict, and which fingerprint source was used (the Duplicate Finder cache at `<library>/Docs/.duplicate_fingerprints.db`, fpcalc, or pyacoustid as a last resort) |

Thresholds are re-read when Settings saves (`on_settings_saved`).

### 6.5 Tag Fixer (`tag_fixer.py`)

Scan (lookups only) → tick field changes → Apply Selected → confirm. There is no
service selector on this page; the sources come from Settings (see
features/tag_fixer.md).

| Element | Type | Purpose |
|---|---|---|
| Dry run (propose only, do not write tags) | Checkbox | Checked by default; while checked, Apply only shows a message |
| Fix fields: Title / Artist / Album / Genre | Checkboxes | Show or hide that field's rows; hidden rows are never written |
| 🏷 Scan for Tag Issues | Button (primary) | `TagFixWorker`; never writes |
| ✓ Apply Selected | Button | Writes exactly the visible, ticked rows (a `LIBRARY_WRITES` job that also takes the shared tag-write guard) |
| Select All / Deselect All / Cancel | Buttons | Cancel stops a scan or an apply |
| Progress | Bar + label | A real percentage |
| Tag Proposals | Tree | ✓, File, Field, Current Value, Proposed Value, Score. **One row per field change**; each tick selects that one field. |
| Summary label | Label | Counts |
| Reason label | Label | Which lookup sources ran, and why nothing matched when nothing did |
| No match found | List | Files nothing could identify, each with its reason |
| Log | Text | |

After an apply, the written rows drop out, the log is kept, and the change-log path
(`Docs/tag_changes_tagfixer_<timestamp>.jsonl`) is shown. There is no automatic
re-scan. A library change cancels a scan and clears the grid; a confirmed apply
finishes.

### 6.6 Genre Normalizer (`genres.py`)

Two tabs. See features/tag_fixer.md. Both read every genre value of a file, write real
multi-value lists, skip files changed since the preview, log every write to
`Docs/tag_changes_*.jsonl`, and share one tag-write guard with the Tag Fixer.

**Tab — Fill from MusicBrainz** (preview-first)

| Element | Type | Purpose |
|---|---|---|
| What this does | Card | States the rules: top three MusicBrainz community tags, existing genres kept, files with 2+ genres skipped |
| 🔍 Look up genres (preview) | Button (primary) | Lookups only; builds the proposals table, writes nothing |
| ✎ Apply N additions | Button (danger) | Enabled once proposals exist; confirms, then writes exactly the table (a `LIBRARY_WRITES` job) |
| Cancel | Button | Stops the lookup or the apply |
| Progress + status | | |
| Proposed genre additions | Table | File, Current genres, After, MusicBrainz match |
| Log | Text | |

(The old dry-run checkbox is gone; the preview replaced it.)

**Tab — Normalize with a mapping** (backed by `controllers/normalize_controller.py`)

| Element | Type | Purpose |
|---|---|---|
| 1 · Scan library / Copy list | Buttons | Every distinct raw genre string |
| 2 · Copy prompt / Load saved mapping / Save mapping | Buttons | Prompt for an external LLM; mapping file `Docs/.genre_mapping.json`. Load asks before replacing unsaved edits. |
| Mapping editor | Text | JSON, validated live. Unsaved text is kept per library across library switches. |
| 3 · Preview changes | Button (primary) | Table of File / Current genres / After, plus counts and files that can't be written |
| 4 · ✎ Apply N changes / Cancel | Buttons | Enabled only for a preview of the current mapping text; confirms; writes exactly the preview |

Preview is disabled while any apply runs. A library change cancels scans, previews and
lookups and drops their results.

### 6.7 Playlists (`playlists.py`)

Every mode is **preview-first**: a background job works out exactly which `.m3u`
files would be written and which existing ones replaced, a confirmation lists them,
and only then is exactly that plan written, with a timestamped `.bak` of anything it
replaces and a line in `Docs/playlist_writes.log`.

| Tab | Controls |
|---|---|
| Folder Playlists | Output folder (blank = `Playlists/`), Generate. One playlist per album folder; reserved folders skipped. |
| Tempo / Energy | Tempo range (BPM) and Energy range, **now used**: tracks outside them are left out. Generate. |
| Auto-DJ | Track count, Seed track (must be a library track outside the reserved folders), Generate → `Playlists/autodj.m3u` |
| Playlist Repair | Add Playlists… / Remove Selected / Clear (empty list = every playlist in `Playlists/`), **"Prefer Opus when searching for missing FLAC" (now read)**, Run Repair |

A shared Cancel button (neutral "Cancelled" status), Run buttons disabled while busy,
and a Log. Writes check the library-write lock before the confirmation. A library
change cancels planning but not a confirmed write.

### 6.8 Clustered (`clustered_enhanced.py`)

The page the sidebar's "Clustered" opens. **Analysis only:** a run writes the feature
cache, `Docs/cluster_info.json` and `Docs/cluster_graph.html`; it does **not** write
playlists. (The older `clustered.py` is dead code, marked as such in its docstring.)

| Tab | Contents |
|---|---|
| 🚀 Quick Start | States the fixed settings (8 K-Means clusters; tempo, MFCC and energy; standard normalization) and a Run Quick Start button. |
| ⚙ Advanced | Configure with Wizard… (§7.2), the chosen configuration as read-only text, and Run. Kept separate from Quick Start. |
| 📊 Results | Progress bar, status, live log and a **Cancel** button during the run (cooperative; finishes the files being analysed). Then: clusters created, total tracks, unclustered (noise) count, tracks that could not be analysed, silhouette score, a hint that playlists are exported from the Music Graph, **View Quality Report** and **Open Visual Graph**. |

Noise is not counted as a cluster and is excluded from the scores. Reserved folders
are not clustered. A library change cancels a run and resets the results. "Open
Visual Graph" emits `navigate_requested("graph")`. (The old "View Playlists" button,
which used a Linux-only `xdg-open` on a folder with no cluster playlists in it, is gone.)

### 6.9 Music Graph (`graph.py`)

The in-app interactive cluster map, plus the browser-based 3-D view. Loading,
validation and export live in the Qt-free `cluster_graph_data.py`.

| Element | Type | Purpose |
|---|---|---|
| Pan / Rectangle / Lasso | Exclusive toggles | Drag behaviour; Ctrl or Shift adds to the selection |
| Fit view / Clear selection | Buttons | |
| ↻ Reload | Button | Re-read the cluster data |
| Open 3D view | Button | Opens (generating if needed) the browser 3-D page |
| Scatter plot | `InteractiveScatterPlot` | One dot per track, coloured by cluster; noise grey. Hover identifies; **a click pins** the track in the details panel (hover no longer replaces it). |
| Cluster legend | `ClusterLegendWidget` | Colour key and counts; checkbox hides a cluster (and drops its tracks from the selection); clicking a row selects it; Show all / Hide all |
| Track details | `TrackDetailsPanel` | Art and tags of the hovered or pinned track, with Play and Open-folder |
| Selection bar | Label + buttons | "N tracks selected", ▶ Play selection, Export CSV…, Export playlist… |

The page **reloads itself when shown** if `Docs/cluster_info.json` changed; a failed
reload clears the plot and legend. The plot background, selection outline and marquee
follow theme switches. Without a working PyQtGraph, the real import error is shown and
the graph controls stay disabled. Selections go to the Player via
`play_tracks_requested`.

The browser page: orbit/zoom, hover tooltip, click-to-select with CSV / `.m3u` export,
legend visibility toggles, a spread slider with 1×/10×/30×/50×/100× presets, axes /
grid / orbit-ring toggles, auto-rotate, and Import JSON. It is self-contained (data
inlined at generation; a built-in sample set when opened with none).

### 6.10 Player (`player.py`)

An in-app audio player behind a **playback backend seam** (`gui/playback_backend.py`).
With no working backend (python-vlc missing, or installed without libVLC) the page
says **"Playback unavailable"** next to the controls (the tooltip gives the reason),
disables transport, and never pretends to play.

| Element | Type | Purpose |
|---|---|---|
| Search | Field | Filters the library table |
| ⟳ Reload | Button | Re-scan the library (reserved folders Playlists/Docs/Trash/Quarantine are skipped; Not Sorted and Manual Review are listed) |
| Browse… | Button | Open and play a single file |
| Load M3U | Button | Load a playlist into the queue (cp1252 / latin-1 fallback for non-UTF-8 files) |
| 30s Preview | Checkbox | Seek to 0:30 and auto-advance after 15 s; pauses and resumes with playback |
| Library table | Sortable table | #, Title, Artist, Album, Time (default sort Artist / Album / Track); right-click: Play now, Add to queue, Open folder, Save selection as M3U… |
| Now-playing covers | Display | Previous / current / next art |
| Blurred background art | Backdrop | Cross-fading blurred art behind the table |
| Hover popup | Tooltip-like panel | After a 1.5 s dwell over a row: art and tags (loaded asynchronously) |
| Queue | List + buttons | + Add Selected, Clear, 💾 Save Queue as M3U…; right-click: Play now, Remove from queue |
| Recently Played | Collapsible list | Up to 30 entries |
| Seek slider + times | Slider | |
| ⏮ ▶ ⏭ ■ | Buttons | Previous, play/pause, next, **stop** |
| 🔀 / 🔁 | Toggles | Shuffle (with history); repeat cycles Off / Track / All |
| Volume | Slider | Kept in sync with the Now Playing bar |
| Backend status | Label | "VLC: ready" or "Playback unavailable" |

There are **no** seek-back / seek-forward buttons and no "Open File" button (Browse…
does that).

**Shortcuts** (active while the Player page is showing): `Space` play/pause, `N` next,
`P` previous, `Shift+Right` / `Shift+Left` seek ±30 s. There is no plain Left/Right
±5 s seek.

**Keyboard mode** (the ⌨ toggle on the Now Playing bar) maps the plain arrow keys to
Down = preview the next track, Right = next, Left = previous, Up = restart. It applies
to the library table or the queue, whichever you last clicked. It is **scoped**: the
app-wide key filter is installed only while the mode is on, and it only takes arrow
keys when the Player page is showing or the Now Playing bar has focus, never inside
text fields, spin boxes or combo boxes, and never while a modal dialog or popup is open.

Decode errors are logged and the track skipped, with a streak limit so a queue of bad
files stops. Scans, tag reads and art loads are cancellable workers whose stale results
are dropped; queue, M3U and graph adds reuse the scan cache and read missing tags in a
worker.

### 6.11 Compression (`compression.py`)

**A FLAC → Opus library mirror.** Output is always Opus in a separate folder; other
files are copied unchanged.

| Element | Type | Purpose |
|---|---|---|
| Source / Destination | Entry + Browse… | The destination may not be inside the source; this is checked before anything is created |
| Opus bitrate (kbps) | SpinBox | 48–320, default 96; disabled during a run |
| Overwrite existing files in destination | Checkbox | The confirmation states the choice |
| 🔍 Estimate Space Savings | Button (primary) | Runs in a worker; never creates the destination |
| ▶ Start Compression | Button | Enabled only by an estimate for the current source / destination / bitrate; changing any of them invalidates the estimate. Needs FFmpeg on `PATH`. |
| ■ Cancel | Button | Real cancel: terminates ffmpeg and removes partial output |
| Progress + counter | Bar + label | |
| Open Report | Button | `<destination>/Docs/opus_library_mirror_reports/opus_library_mirror_report_<timestamp>.html` |

Quarantine/, Trash/ and Docs/ are not mirrored. Output is encoded and tagged under a
temporary name and renamed when complete. Two sources that would land on the same
destination name (`Song.flac` and `Song.opus`) are resolved before any worker runs. A
library change during a run is applied after the run.

### 6.12 Utilities (`tools.py`)

A grid of "liquid glass" tiles (`TileGrid` / `ToolTile`). Each tile's status chip
reports running / done / error / cancelled. The export and cleanup jobs run in
workers with a Cancel button, and a library change cancels the read-only ones (a
confirmed File Cleanup or undo finishes).

| Tile | Contents |
|---|---|
| Artist / Title Export | "Exclude FLAC files" and "Include per-album duplicate titles" toggles; Export, Cancel, Open File, open-folder. Writes `Docs/artist_title_list.txt`. |
| Codec List Export | Per-extension chips, "Filenames only (no full paths)"; Export (in a worker), Cancel, Open File, open-folder |
| **File Cleanup** | **Preview Cleanup…** plans in a worker, then opens a **preview dialog** (folder, current name → new name, collisions flagged and skipped); on confirm it renames exactly the previewed rows, writes `Docs/file_cleanup_<timestamp>.jsonl` and updates playlists. **Undo Last Cleanup…** previews the newest cleanup log in the same dialog, confirms, restores the original names and points the changed playlists back. Only true OS copy artifacts are stripped (` (1)`…` (99)`, ` - Copy`, Finder ` copy N`); reserved folders are skipped; nothing is ever overwritten. Both are library writes. |
| Diagnostics | Launchers for M4A Tester, Opus Tester, Bucketing POC, Scan Engine, Fuzzy Finder, Pair Review, View Crash Log (§7.6). A dialog whose worker is still running is reopened rather than duplicated. |
| Library Validator | Runs `validator.py` and shows the report (a library without `By Year/` is accepted) |

Export jobs skip Quarantine/, Trash/, Docs/ and Playlists/. An export that finishes
after a library change no longer re-enables Open buttons for the old library.

### 6.13 Help (`help.py`)

| Element | Purpose |
|---|---|
| Documentation | One button per maintained doc: `docs/README.md`, overview, architecture, every `docs/features/` page, `gui_inventory.md` and `ROADMAP.md` (a test checks that each exists) |
| Keyboard Shortcuts | The app-wide shortcuts (Ctrl+O, Ctrl+, , Ctrl+L, Ctrl+1–9, Ctrl+W). The Player's own shortcuts are not listed here. |
| About AlphaDEX | Former name, config file location, supported formats, and a Report an Issue (GitHub) button |

---

## 7. Dialogs

### 7.1 SettingsDrawer (`gui/dialogs/settings_drawer.py`)

Modal. Opened by Settings or Ctrl+,. A fresh dialog each time (deleted after it closes).
**Every control persists under its config key:**

| Tab | Control | Config key |
|---|---|---|
| Metadata Services | Service (AcoustID / MusicBrainz / Last.fm; Spotify and Gracenote shown disabled with a tooltip) | `metadata_service` |
| | API key, shown for **AcoustID and Last.fm** (one shared field) | `metadata_api_key` |
| | MusicBrainz app name / version / contact, shown for MusicBrainz | `musicbrainz_useragent` |
| | Test Connection | Real checks: AcoustID and Last.fm test the entered key; MusicBrainz runs a search. Runs off the GUI thread. |
| General | Default library (validated if you change it) | `library_root` |
| | Near-duplicate threshold | `near_duplicate_threshold` |
| | Exact-duplicate threshold (range wide enough that saved values aren't clamped) | `exact_duplicate_threshold` |
| | Background gradient | `bg_gradient_enabled` |
| Advanced | **Empty stub tab** | — |

**Known gap:** the API key field is shared. A key typed there for Last.fm is saved as
`metadata_api_key`, which the AcoustID source also reads, while the Last.fm lookup
source only reads the `LASTFM_API_KEY` environment variable. See ROADMAP.md.

There is no Settings control for the mixed-codec boost (edited on the Duplicates page)
or for reduced motion (config key `reduced_motion` or `ALPHADEX_REDUCED_MOTION=1`).

On save the window refreshes the cached gradient and reduced-motion flags, calls every
page's `on_settings_saved()` (Duplicates and Similarity re-read their thresholds there),
and applies a changed default library through `_apply_library()`.

### 7.2 ClusteringWizardDialog (`gui/dialogs/clustering_wizard_dialog.py`)

A 5-step wizard from the Clustered page's Advanced tab. It keeps every choice when you
go Back and Next.

| Step | Contents |
|---|---|
| 1. Feature Selection | tempo / mfcc / chroma / spectral / energy / onset_rate checkboxes with tooltips, plus Fast / Balanced / Complete presets. All six are computed when ticked. |
| 2. Normalization & Preprocessing | Standard / MinMax / Robust (used). Dimensionality reduction (None / PCA / t-SNE / UMAP) is **disabled: not available yet**. |
| 3. Algorithm Selection | K-Means (cluster count) or HDBSCAN (min cluster size, min samples); the choice is passed through correctly. |
| 4. Post-Processing | Remove small clusters / merge into "Miscellaneous": **disabled, not available yet**. |
| 5. Output Options | Create M3U playlists / generate quality report / open graph: **disabled**, with tooltips pointing at what to use instead (Music Graph export; the report is offered when scores exist; "Open Visual Graph"). |

Refuses to proceed if every feature is unticked.

### 7.3 ClusterQualityReportDialog

Silhouette / Davies-Bouldin / Calinski-Harabasz with verdicts, a per-cluster breakdown
and suggestions. Renders correctly and follows the theme (it used to render empty).

### 7.4 ThemePickerDialog / 7.5 AutoThemeDialog

See §5.3 and §5.4.

### 7.6 Diagnostics dialogs (from Utilities)

Each is deleted when closed; closing one mid-run cancels its worker.

| Dialog | Purpose |
|---|---|
| `MediaTesterDialog` (M4A / Opus) | Reads a file through `utils.audio_metadata_reader.read_metadata`, the same reader the app uses, and shows its tags and cover art. |
| `BucketingPocDialog` | Duplicate bucketing proof of concept. Asks before running without fpcalc and then says "Completed without fingerprints". |
| `ScanEngineDialog` | Ten duplicate-scan tuning fields. **Run is disabled** without numpy/librosa (the tooltip says what to install). Its output database has no reader in the app yet. |
| `FuzzyDupeDialog` | Proof of concept that now runs the real `near_duplicate_detector` over the Duplicate Finder's cached fingerprints; thresholds default to Settings; results list verdict, distance and the pair. **Still not implemented, and disabled with a tooltip:** the "Metadata Match Settings" group, "Include pairs without fingerprints", "Skip fingerprint check", "Include non-matching FP pairs in review", and "Send Matches to Duplicate Pair Review". |
| `PairReviewDialog` | Side-by-side review of pairs from the Duplicate Finder's plan (`Docs/duplicate_preview.json`), or else files with the same name in the **same folder** (reserved folders skipped). Prev / Next / Switch / Prefer MP3 / Jump to. **"Quarantine inferior…"** confirms with both file names, moves the loser to `Quarantine/` under a unique name (never deleting, never overwriting), points playlist entries at the kept file, and logs to `Docs/duplicate_pair_review_log.txt`. "No — skip" skips; "Tag" appends the pair to `Docs/duplicate_pair_tags.txt`. Shortcuts: Backspace = skip, Left/Right = navigate. Enter no longer triggers the action. Refuses while another library write runs. |
| `CrashLogDialog` | Shows the tail of `~/.alphadex/crash.log` and `~/.alphadex/faults.log` and names both paths. |

---

## 8. Widgets (`gui/widgets/`)

| Widget | Status |
|---|---|
| `top_bar.py` (`TopBar`) | Live — §2 |
| `sidebar.py` (`Sidebar`) | Live — §3 |
| `log_drawer.py` (`LogDrawer`) | Live — §4 |
| `now_playing_bar.py` (`NowPlayingBar`) | Live — §1.1 |
| `gradient_bg.py` (`GradientWidget`) | Live — §5.5 (background glow and painted card shadows) |
| `landing.py` (`MosaicLanding`) | Live — §0 |
| `splash.py` (`SplashScreen`) | **Dead code** — built, never shown |
| `interactive_scatter_plot.py` (`InteractiveScatterPlot`) | Live — the Music Graph plot |
| `interactive_3d_scatter.py` (`Interactive3DScatterPlot`) | **Orphaned** — not used anywhere; the 3-D view users reach is the browser page from `cluster_graph_3d.py` |
| `cluster_legend.py` (`ClusterLegendWidget`) | Live — Music Graph legend |
| `track_details_panel.py` (`TrackDetailsPanel`) | Live — Music Graph details |

Elsewhere under `gui/`: `playback_backend.py` (the Player's backend seam), `workers.py`
(JobRegistry), `themes/motion.py` (motion tokens and helpers), `workspaces/label_tone.py`
(theme tones and a reusable flash for status labels), and `compat.py` (Qt imports;
QtWebEngine is imported lazily on first use). `gui/audio_preview.py` is used only by
the legacy Tk app, and `gui/qt_launcher.py` is a placeholder "Qt Preview" window
launched only from the legacy app; neither is part of the Qt app.

---

## 9. Signals flow summary

```
TopBar.library_changed ─┐
SettingsDrawer (saved) ─┼─► AlphaDEXWindow._apply_library(path)   (rejects missing, no-op if unchanged)
Startup (after fade) ───┘      └─► ws.set_library_path() for every workspace
                               └─► _StatsWorker (cancels the previous one) → TopBar.set_stats
                               └─► config.save_config()  (library_root)

SettingsDrawer.settings_saved ─► refresh gradient + reduced-motion caches
                               └─► ws.on_settings_saved() for every workspace

TopBar.theme_requested ──► ThemePickerDialog (non-modal, reused)
                              └─► ThemeManager.apply(key)
                                   └─► style tokens + palette swapped, themed widgets re-polished
                                   └─► theme_changed(tokens) → self-coloured widgets refresh

Sidebar.nav_changed ─────► AlphaDEXWindow._on_nav_changed
                              └─► QStackedWidget.setCurrentWidget(ws), status bar label
                              └─► ws.on_activated() if the page defines it

Sidebar.exit_requested ──► AlphaDEXWindow.close()  (closeEvent job protocol)

WorkspaceBase.log_message ──► LogDrawer.append()   (buffered until connected)
WorkspaceBase.status_changed ─► LogDrawer.set_status() + QStatusBar
WorkspaceBase.navigate_requested ─► AlphaDEXWindow._on_nav_changed
WorkspaceBase.play_tracks_requested ─► AlphaDEXWindow.play_tracks
                              └─► PlayerWorkspace.load_tracks_and_play(paths, label)

PlayerWorkspace ⇄ NowPlayingBar   (now-playing / position / state / volume_changed one way;
                                    play-pause / next / prev / seek / volume / keyboard mode the other)

MosaicLanding.tile_clicked ──► Startup.play_dir ──► AlphaDEXWindow.play_directory()
                              └─► Player workspace, begins playback
```
