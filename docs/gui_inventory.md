# AlphaDEX GUI Inventory

Plain-English reference for every page, section, control, and dialog in the
**PySide6 GUI** (`alpha_dex_gui.py` → `gui/main_window.py`).

> The legacy Tkinter interface (`main_gui.py`) is still functional but is no
> longer the primary GUI. This document describes the PySide6 app only.

> **Verified against the code 2026-11-07**, after the fix phase that closed about
> 200 bugs across every page (see `docs/redesign/README.md` for the bug inventory),
> and **updated the same day for the adopted information architecture** (§3): the
> sidebar is now grouped by the user's jobs, several old pages are *sections* of a
> new page, Utilities is gone, and Home, Activity and a Settings page are new.
> Where a control is still decorative or a feature is still partial, this document
> says so plainly.

---

## 0. Startup Sequence

Handles the initial loading and library selection before the main window appears.
The sequence lives in `alpha_dex_gui.py` (`Startup` / `build_startup()`, which the
test harness drives without `app.exec()`). The album-art landing is shown on every
launch.

**What actually runs**, in order:

| Step | What happens |
|---|---|
| Crash log | `main()` first installs the Qt crash log (`crash_logger.install_for_qt()`; see architecture.md). |
| `MosaicLanding` (`gui/widgets/landing.py`) | Shown immediately. The window fades in with only the glass card (the logo moment), then a full-window 7 × 5 grid of album tiles flies in, filled by a background art scanner (`_ArtScanner`, cancellable, registered with the JobRegistry). |
| `AlphaDEXWindow` (`gui/main_window.py`) | Built once the landing has **fully faded in** (`faded_in` signal), during the logo pause, so the first landing frames are never blocked. If the user clicks before that, it is built on demand. It opens on **Home**. |
| Cross-fade | When a library is chosen, the landing fades out while the main window fades in over the same duration. |
| Library applied | The chosen library is applied **once**, after the cross-fade finishes (`window.set_library(path)`). The main window never applies the saved library by itself. |

The landing is Liquid Glass (redesign R1): a frosted glass card in the middle of the
album grid holds the brand, the saved library (path, and its track and artist counts
from a read-only background count), the main button, and — when a library is saved — a
**Choose another folder** link and "Or click any album to start playing it.". The
card's frost is the tiles behind it, blurred off the GUI thread. A glass **✕** at the top
right quits (`quit_requested`; the app exits without building the main window).
Hovering a tile lifts it (scale 1.06) with a white ring and a glass caption chip (album
and artist), painted by a mouse-transparent overlay; tiles never move for it. With
reduced motion the fly-in and the exit scatter are skipped.

The main button reads **"Initialize"** when the saved library exists, or
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

Layout (Liquid Glass shell, redesign R1). The central widget is the **backdrop**
(`gui/widgets/backdrop.py`); everything else floats on it as glass:

```
┌──────────────────────────────────────────────────────────────┐
│ ╭────────────╮   Page title          [ search ] [Change lib.] │
│ │  Sidebar   │   (library chip · tracks · GB · artists)       │
│ │  (glass,   │   ( segmented section tabs, when a page has    │
│ │   250 px,  │     several sections )          [page actions] │
│ │   floats   │                                                │
│ │   18 px    │   Page (stacked; transparent, cards are glass) │
│ │   in)      │                                                │
│ │            │        ╭ Now Playing capsule (floating) ╮      │
│ ╰────────────╯   ╭ Activity Log strip (glass) ──────────────╮ │
└──────────────────────────────────────────────────────────────┘
```

The status bar still receives "Page" / "Page · Section" messages but is not shown
(the page header and the log strip's status chip say the same). Page and section
changes cross-fade (opacity only; instant with reduced motion).

**Panels and pages.** The workspace classes are *panels*. The window builds every
panel once (`self._workspaces[panel_key]`, signals wired as before) and composes them
into pages (`self._pages[page_key]`, `gui/pages.py`). `current_page()` and
`current_panel()` say what is on screen.

**One validated library path.** The Top Bar's Change Library (and Ctrl+O), the
Settings page's Default library, and the startup sequence all end in
`_apply_library()`. It rejects a folder that does not exist (with a message; the
current library is kept), does nothing for the library that is already active, and
only then fans the path out to every panel, refreshes the top-bar statistics (which
Home shows too), and saves `library_root`.

**Closing the window** is safe while jobs run (see architecture.md, "Background jobs").
Read-only jobs are cancelled silently. If any job may be writing files, a "Background
tasks running" question lists them; on Yes the window disables its UI, cancels every
job cooperatively, and closes once they stop. If a job ignores cancellation for 20 s,
the user is asked once whether to quit anyway.

### 1.1 Now Playing capsule (`gui/widgets/now_playing_bar.py`)

A floating glass capsule at the bottom centre of the page area (22 px up, at most
880 px wide), with its shadow over the page. Hidden until the first track loads (it
fades in) and hidden again when nothing is loaded any more. Pages keep 104 px of room
under their content so it never covers anything (the Player page reserves it too).

| Element | Purpose |
|---|---|
| Cover | 52 px round cover; a ♪ disc when the track has no art |
| Title / artist · album | Current track (elided to fit) |
| ⏮ / ▶ / ⏭ | Previous, play/pause, next (routed to the Player panel) |
| Seek slider | Drag, wheel, keys and groove clicks all seek |
| Volume slider | Kept in sync with the Player page's slider in both directions, without echo |
| ⌨ toggle | Turns **keyboard mode** on and off (see §6.7) |

---

## 2. Page header (`gui/widgets/top_bar.py`, class `TopBar`)

Above the page stack, on the backdrop (no card). Always visible.

| Element | Type | Purpose |
|---|---|---|
| Page title | Label (`#pageTitle`, display 34 px) | The sidebar label of the page on screen |
| Library chip | Pill with the path (`#libraryPath`, elided in the middle) | The tooltip holds the full path. Muted colour when no library is set. |
| Library stats | Labels | Tracks · GB · artists, from a background scan. The scan skips the reserved folders and unreadable files; "artists" counts `Artist/Album/track` folders. A newer library's scan cancels and replaces an older one. Shows "–" until it finishes. |
| Search | Pill line edit | Enter shows the **Player** filtered to the text (the Player's own track filter) |
| Change library | Pill button | Folder picker; emits `library_changed(path)` (also Ctrl+O) |

The brand moved to the sidebar; the Theme and Settings buttons are gone (Settings is in
the sidebar and on Ctrl+,; the theme is in Settings → Appearance).

**Signals:** `library_changed(str)`, `search_requested(str)`, and — kept for callers —
`settings_requested()`, `theme_requested()`.

---

## 3. Sidebar and the information architecture

### 3.1 Sidebar (`gui/widgets/sidebar.py`)

A floating glass panel, 250 px wide (220 px minimum), radius 30, 18 px in from the
window edges; its shadow is painted by the backdrop. A brand tile and "AlphaDEX" at
the top, the groups below (11 px headings), text-only items 40 px high (they shrink to
no less than 32 px in a short window; below that the list scrolls). Each item is an
`AnimatedNavButton` whose hover wash eases through the motion tokens; the active item
sits on a raised glass pill that slides between items (a painter offset, never widget
geometry). Each item's tooltip is the page's one-line intent. Clicking emits
`nav_changed(page_key)`. There is no Exit item any more: quit with the window's close
button or **Ctrl+Q** (same close protocol).

The groups come from `gui/navigation.py` (`NAV_STRUCTURE` is derived from it). The
untitled last group (Activity · Settings · Help) sits at the foot of the panel.

| Group | Page key | Sidebar label | What it is for |
|---|---|---|---|
| **LIBRARY** | `home` | Home | The library at a glance (§6.1) |
| **ORGANIZE** | `indexer` | Indexer | Normalize folder structure and names, preview first (§6.2) |
| | `library_sync` | Library Sync | Merge an incoming folder (§6.3) |
| | `file_cleanup` | File Cleanup | Remove copy suffixes, previewed and undoable (§6.4) |
| **CLEAN UP** | `duplicates` | Duplicates | Six sections (§6.5) |
| | `metadata` | Metadata | Tag Fixer · Fill genres · Normalize genres · Undo changes (§6.6) |
| **LISTEN** | `player` | Player | §6.7 |
| | `sound_map` | Sound Map | Analyse · Map (§6.8) |
| | `playlists` | Playlists | §6.9 |
| **TAKE AWAY** | `mirror` | Mirror | Build the mirror · Verify the mirror (§6.10) |
| | `exports` | Exports | Artist/Title and codec lists (§6.11) |
| *(foot)* | `activity` | Activity | Jobs, log, problems (§6.12). A "N running" pill shows the running jobs (`Sidebar.set_badge`; housekeeping jobs such as the backdrop build are not counted). |
| | `settings` | Settings | Metadata services · General · Appearance (§6.13) |
| | `help` | Help | §6.14 |

**Keyboard shortcuts** (set in `main_window.py`):

| Shortcut | Action |
|---|---|
| Ctrl+1 … Ctrl+9 | The first nine pages, in sidebar order (Home → Playlists). The order and the status-bar labels come from `gui/navigation.py`, so they can't drift from the sidebar. Mirror, Exports, Activity, Settings and Help have no number shortcut. |
| Ctrl+O | Change library |
| Ctrl+, | The Settings page |
| Ctrl+L | Toggle log drawer |
| Ctrl+W | Clear log |
| Ctrl+Q | Quit (the close protocol) |

### 3.2 Pages and sections (`gui/navigation.py`, `gui/pages.py`)

A page with more than one section shows its **section tabs** above its panels, as a
glass segmented control (`SegmentedTabBar`: a raised thumb slides to the chosen
section); a page with one section just shows its panel. A page remembers the section it showed
last. When a page or section becomes visible, the window calls the visible panel's
optional `on_activated()` hook (Home re-reads `Docs/` if its last check is older than
3 s; Settings re-reads the config unless there are unsaved edits; Activity refreshes;
the Map reloads itself from `showEvent`). Page-level actions sit to the right of the
tabs (`SectionedPage.add_action`).

| Page | Section key | Tab label | Panel key → class |
|---|---|---|---|
| Duplicates | `find` | Find duplicates | `duplicates` → `DuplicatesWorkspace` |
| | `compare` | Compare two files | `similarity` → `SimilarityWorkspace` |
| | `pair_review` | Pair review | `pair_review` → `PairReviewWorkspace` (opens a dialog) |
| | `fuzzy_finder` | Fuzzy finder **[Experimental]** | `fuzzy_finder` → `FuzzyFinderWorkspace` (opens a dialog) |
| | `scan_engine` | Scan engine **[Experimental]** | `scan_engine` → `ScanEngineWorkspace` (opens a dialog) |
| | `bucketing` | Bucketing **[Experimental]** | `bucketing` → `BucketingWorkspace` (opens a dialog) |
| Metadata | `tag_fixer` | Tag Fixer | `tag_fixer` → `TagFixerWorkspace` |
| | `fill_genres` | Fill genres | `genres` → `GenresWorkspace`, view `fill` |
| | `normalize_genres` | Normalize genres | `genres` → `GenresWorkspace`, view `normalize` |
| | `undo_tags` | Undo changes | `tag_history` → `TagHistoryWorkspace` |
| Sound Map | `analyse` | Analyse | `clustered` → `EnhancedClusteredWorkspace` |
| | `map` | Map | `graph` → `GraphWorkspace` |
| Mirror | `build` | Build the mirror | `compression` → `CompressionWorkspace` |
| | `verify` | Verify the mirror | `mirror_verify` → `MirrorVerifyWorkspace` |
| Settings | `services` | Metadata services | `settings` → `SettingsWorkspace`, view `services` |
| | `general` | General | `settings`, view `general` |
| | `appearance` | Appearance | `settings`, view `appearance` |

Every other page has one section whose key, label and panel key equal the page's
(`home` → `HomeWorkspace`, `file_cleanup` → `FileCleanupWorkspace`, `exports` →
`ExportsWorkspace`, `activity` → `ActivityWorkspace`, …). A panel that serves several
sections (Genres, Settings) hides its own tab bar and is switched by the page's tabs
(`set_section(view)`).

**Navigation keys.** `AlphaDEXWindow._on_nav_changed(key)` (and so
`navigate_requested(key)` from any panel) accepts an explicit `"page/section"`, a
panel key (the old per-workspace keys keep working), a section key, a page key, or a
retired key. The sidebar and Ctrl+1..9 call `select_page()`, which keeps the page's
last section.

| Old key | Now shows |
|---|---|
| `similarity` | Duplicates → Compare two files |
| `duplicates` | Duplicates → Find duplicates (a page *and* panel key: the panel wins) |
| `tag_fixer` | Metadata → Tag Fixer |
| `genres` | Metadata → whichever genre tool the panel shows (Fill genres at first) |
| `clustered` | Sound Map → Analyse |
| `graph` | Sound Map → Map (so Analyse's "Open the map" still works) |
| `compression` | Mirror → Build the mirror |
| `tools` | Exports (Utilities is gone; see §6.4, §6.11 and below) |
| `indexer`, `library_sync`, `player`, `playlists`, `help` | the page of the same name |

**Where Utilities went:** the exports → Exports; File Cleanup → its own page; the
library Validator → Home's "Library health"; the M4A / Opus testers → Mirror → Verify
the mirror (embedded); Bucketing POC, Scan Engine, Fuzzy Finder and Pair Review →
Duplicates sections; the crash log viewer → Activity → Problems.

**The Experimental pill** (`gui/widgets/pill.py`, `ExperimentalPill`) marks exactly the
three prototypes, on their section tab and next to their section title, with a one-line
"Experimental — not ready to test" note inside the section. It is painted from tokens
only (fill = `warning` blended over `card_bg`, border = `warning`, text =
`text_primary`); `tests/test_design_contrast.py` checks it in every theme.

---

## 4. Log Drawer (`gui/widgets/log_drawer.py`)

A glass strip at the bottom of the content area (the material's card surface). Collapsed by default; expands to
**220 px** on toggle, or by itself when a `warn` or `error` line arrives. It keeps the
last 500 lines; the Activity page keeps the session's full history (§6.12).

| Element | Purpose |
|---|---|
| Handle bar | Click to toggle; shows the status chip |
| Status chip | One-line status from the active page (`status_changed`) |
| Log text | Timestamped, colour-coded lines (`info`, `ok`, `warn`, `error`) |
| Clear | Clears the log |

Colours come from the theme's `log_*` tokens and follow theme switches, including the
colour of lines already shown. The slide moves a sheet inside a layout-free body, so
the window is laid out once per toggle rather than on every animation frame.

Messages a panel logs while it is still being built are buffered and delivered once
the window has connected the panel (`WorkspaceBase.flush_pending_logs`).

---

## 5. Theme Engine (`gui/themes/`)

### 5.1 ThemeManager (`manager.py`)

Singleton `QObject`.

- `apply(key)` — switch to a named theme or `"auto"`, and persist it as `theme`.
- `load_persisted()` — read the saved theme at startup.
- `configure_auto(dark_key, light_key)` — the Auto pair (`auto_dark_theme`, `auto_light_theme`).
- `set_accent(key)` / `set_reduce_transparency(on)` — Liquid Glass accent and opaque
  surfaces; both apply at once through the same fast path (`glass_accent`,
  `reduce_transparency`). `current` is the *effective* tokens (theme + accent +
  transparency).
- **One-time migration**: a saved theme other than Liquid Glass is switched to it once
  (`theme_glass_migrated` marker, the old key kept as `theme_before_glass`); the main
  window logs it. Choosing a classic theme afterwards sticks.
- `theme_changed(tokens)` — emitted after every switch. Widgets that colour
  themselves (Top Bar, Player, log drawer, card shadows, pills, …) listen to it.
- **Auto mode** uses the platform's colour scheme when it reports Light or Dark, and
  falls back to the clock (07:00–20:00 = light) when it reports Unknown, which many
  Linux desktops do. It re-checks every 15 minutes while Auto is on.

A theme switch takes about 60–110 ms (it used to freeze the window for over a second);
see architecture.md, "Themes and motion", for how.

### 5.2 Themes

15 named themes, persisted as `theme`. Every text role meets WCAG AA contrast on the
surfaces it is drawn on, in every theme (`tests/test_design_contrast.py`).

**Liquid Glass** (the default): translucent surfaces over the album-art backdrop, in
four accents (Blue, Peach, Mint, Lilac).

**Classic, dark (8)**: Midnight, Obsidian, Graphite, Navy, Twilight, Aurora, Forest, Ember

**Classic, light (6)**: Pearl, Azure, Blossom, Meadow, Lavender, Sunset

Every theme carries a *material* (`ThemeTokens.material`: sidebar, card, raised,
capsule and well surfaces, plus the backdrop). Classic themes get an opaque one
derived from their colours, so they render through the same glass components.

### 5.3 Theme picker (`picker.py`)

`ThemePickerPanel` is embedded in **Settings → Appearance** (the "Theme" card). Changes
apply instantly, and the selection follows switches made elsewhere. Order: Liquid Glass,
then Auto, then the classic themes.

| Element | Purpose |
|---|---|
| Liquid Glass row | Its swatch, the four accent discs, and **Reduce transparency** |
| Auto row (on a well) | A painted half-disc mark, the configured night and day themes, Configure… and Select (primary, check glyph; "Active" when on) |
| Classic themes | "Dark" and "Light" groups, each a responsive grid (`_GridFlow`): as many equal columns as fit (cells ≥ 104 px), the same cell width in both groups, more rows when narrow |
| Swatch card | A miniature of the theme (sidebar strip, top bar, card) above a name strip, both on the theme's own background, so the name (in the theme's own text colour) never overlaps the preview; accent ring and checkmark when active |

`ThemePickerDialog` (the same panel in a non-modal window) still exists but the app no
longer opens it.

### 5.4 AutoThemeDialog

Modal child of the picker ("Auto Theme — Select Pair"): a night-theme grid, a
day-theme grid (the same responsive swatch grid), and on the footer band Cancel and
**Apply Auto Theme** (primary, the default button), which saves the pair and turns Auto on.

### 5.5 The backdrop, glass surfaces and card shadows

The window's central widget is the **backdrop** (`gui/widgets/backdrop.py`), chosen in
Settings → Appearance → **Backdrop**: *Album art* (a blurred wall of the library's own
covers; Liquid Glass only, classic themes show their gradient), *Gradient*, or
*Plain*. It is composed off the GUI thread, cached per library and size bucket, and
painted by blitting the cached image. With no covers it falls back to the gradient.

Pages, scroll areas and the section strip are transparent. Cards are `GlassCard`s
(`gui/widgets/glass.py`) painted with `gui.themes.glass.paint_surface`; the
`GradientWidget` scroll surface **paints the soft shadow under every card** (a cached
9-slice, never under the card itself) instead of giving each card a graphics effect.

---

## 6. Pages

All panels inherit `WorkspaceBase` (`gui/workspaces/base.py`):

- Content sits in a `QScrollArea` whose inner widget is a `GradientWidget` (a plain
  `QWidget` would be transparent and the glow wouldn't show).
- Signals: `log_message(str, level)` → log drawer and Activity; `status_changed(str,
  colour)` → status chip and status bar; `navigate_requested(key)` → the main window
  switches page/section (panels never switch themselves; any key from §3.2);
  `play_tracks_requested(paths, label)` → the main window hands tracks to the Player.
- `_make_card()` returns a themed card and registers it for a painted shadow.
- **Background jobs:** `_start_worker(worker, name, ask_before_quit=..., library_write=...)`
  starts every `QThread` under the JobRegistry; `_library_writes_blocked(action)` is
  checked before any confirm dialog for a job that changes library files, and refuses
  ("Library Busy") while another such job runs. See architecture.md.
- Hooks the main window calls: `set_library_path()` (→ `_on_library_changed`),
  `on_settings_saved()` after Settings saves, `on_activated()` when shown, and
  `shutdown()` when the app closes. Panels serving several sections also implement
  `set_section(view)`, `current_section()` and `set_section_tabs_visible(bool)`.

A pattern most pages share: a **library change cancels read-only work** (scans,
previews, lookups) and ignores its late result, while a **write the user already
confirmed is left to finish** on the library it started on. (Duplicates is the
exception; see §6.5.1.) Most pages also have a workflow stepper that advances with
the flow, and a neutral "Cancelled" state instead of reporting a cancel as an error.

### 6.1 Home (`home.py`)

The library at a glance. **Read-only and cheap:** it never walks the whole library.
Its statistics are the top bar's (the main window's existing stats scan hands them
over), and its own check (`scan_library_health`) only runs the Validator, lists
`Docs/` and walks the three folders below, in one read-only JobRegistry job
(`ask_before_quit=False`) started on a library change, by Refresh, or when the page
is shown again more than 3 s after the last check. A result for an older library is
ignored.

| Card | Contents |
|---|---|
| Library | Full path (selectable), tracks · GB · artists, "Checked at HH:MM", Refresh |
| Library health | The Validator (formerly a Utilities tile): ✓ "Organized" when `By Artist/` / `By Year/` exist, or its explanation. Then **Manual Review/**, **Quarantine/** and **Not Sorted/** with their audio-file counts (or "not created yet") and an "Open folder" button each. |
| Latest runs | The newest of each kind in `Docs/`, newest first, each with **Open**: Indexer run (`indexer_executed_*.txt`), Duplicate cleanup (`duplicate_execution_reports/consolidation_run_*/reports/execution_report.html`), Library Sync run (`LibrarySyncExecuted_*.html`, else `LibrarySyncExecuted.html`, else `LibrarySyncAudit_*.json`), Tag changes (newest `tag_changes_<tool>_*.jsonl` per tool), File cleanup (`file_cleanup_*.jsonl`). "No runs yet…" when there are none. |
| Quick actions | Organize folders (Indexer), Merge new music (Library Sync), Clean up file names (File Cleanup), Find duplicates, Fix tags (Metadata → Tag Fixer), Explore the Sound Map, Make a playlist. |

Without a library it starts no job and says "No library selected".

### 6.2 Indexer (`indexer.py`)

Preview first; Execute applies **that** preview.

| Element | Type | Purpose |
|---|---|---|
| Workflow stepper | Header | 1. Preview (dry run) → 2. Review HTML report → 3. Execute changes |
| Dry run | Checkbox | Checked by default. Unchecking switches the Run button to Execute. |
| Create playlists after execution | Checkbox | Checked: regenerate playlists after the moves. Unchecked: only update existing playlists' paths. |
| Cross-album scan (Phase C) | Checkbox | **Disabled**, tooltip: the engine has no cross-album phase. |
| Flush fingerprint cache before scan | Checkbox | **Disabled**, tooltip: the Indexer keeps no cache, so there is nothing to flush. |
| Max workers | SpinBox | 1–32, default **1**. How many files have their tags read at once during a preview. 1 is fastest on a local disk; more helps on a network share or slow drive. Locked while a run is going. |
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

### 6.4 File Cleanup (`tools.py`, `FileCleanupWorkspace`)

Moved out of Utilities onto its own page. One full-width tile (`ToolTile`) whose status
chip reports running / done / error / cancelled.

**Preview Cleanup…** plans in a worker, then opens a **preview dialog** (folder, current
name → new name, collisions flagged and skipped); on confirm it renames exactly the
previewed rows, writes `Docs/file_cleanup_<timestamp>.jsonl` and updates playlists.
**Undo Last Cleanup…** previews the newest cleanup log in the same dialog, confirms,
restores the original names and points the changed playlists back. Only true OS copy
artifacts are stripped (` (1)`…` (99)`, ` - Copy`, Finder ` copy N`); reserved folders
are skipped; nothing is ever overwritten. Both are library writes. **Cancel** stops the
preview or the renames. A library change cancels a preview; a confirmed cleanup or undo
finishes.

### 6.5 Duplicates

Six sections (§3.2). The first two are the real workflow; Pair review and the three
prototypes open their dialogs (§7.5).

#### 6.5.1 Find duplicates (`duplicates.py`)

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

The scan also writes `Docs/duplicate_preview.json`, which the Pair review section reads.

**Still different from the other pages:** a library change *cancels* a running
duplicate execution (it stops at the next safe point and writes its report to the old
library's `Docs/`), where the Indexer, Library Sync and the tag tools let a confirmed
write finish. See ROADMAP.md.

#### 6.5.2 Compare two files (`similarity.py`)

Formerly the Similarity page. A two-file duplicate diagnostic that uses **the Duplicate Finder's own pipeline and
settings**.

| Element | Type | Purpose |
|---|---|---|
| File A / File B | Entry + Browse… | The two audio files |
| Advanced Overrides | Checkable group (off by default) | Off: fingerprint exactly like the Duplicate Finder (reusing its cache) with the thresholds from Settings. On: trim silence, fingerprint offset / duration, silence threshold / min length, and exact / near / mixed-codec thresholds are used. An invalid field is named in a warning. |
| ⚖ Run Comparison / Clear | Buttons | Clear also discards a late result from a running comparison |
| Comparison Report | Text | Codecs, durations, distance, thresholds, verdict, and which fingerprint source was used (the Duplicate Finder cache at `<library>/Docs/.duplicate_fingerprints.db`, fpcalc, or pyacoustid as a last resort) |

Thresholds are re-read when Settings saves (`on_settings_saved`).

#### 6.5.3 Pair review (`launchers.py`, `PairReviewWorkspace`)

Explains the tool (pairs from the duplicate preview in `Docs/`; tags, format and art side
by side; a rejected file is moved to `Quarantine/`, never deleted; decisions logged to
`Docs/`) and **Open Pair Review…** opens `PairReviewDialog` (§7.5). Without a library it
warns; without a duplicate preview it says "No duplicate preview found — run Duplicate
Finder first".

#### 6.5.4 Fuzzy finder · Scan engine · Bucketing — **Experimental**

Each section shows its name with the **Experimental** pill, the note *"Experimental —
not ready to test. A prototype of another way to find duplicates: its results are not
used by Find duplicates and have not been checked."*, what the prototype does, and an
**Open …** button for its dialog (`FuzzyDupeDialog`, `ScanEngineDialog`,
`BucketingPocDialog`; §7.5). There is at most one live dialog per tool: closing one
whose worker still runs keeps it, and the next Open re-shows it instead of starting a
second run.

### 6.6 Metadata

#### 6.6.1 Tag Fixer (`tag_fixer.py`)

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

#### 6.6.2 Fill genres · Normalize genres (`genres.py`)

One panel (still titled "Genre Normalizer") serves both sections: the Metadata page's tabs switch it, and its own tab bar is hidden. See features/tag_fixer.md. Both read every genre value of a file, write real
multi-value lists, skip files changed since the preview, log every write to
`Docs/tag_changes_*.jsonl`, and share one tag-write guard with the Tag Fixer.

**Fill genres** (the panel's "Fill from MusicBrainz" view; preview-first)

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

**Normalize genres** ("Normalize with a mapping"; backed by `controllers/normalize_controller.py`)

| Element | Type | Purpose |
|---|---|---|
| 1 · Scan library / Copy list | Buttons | Every distinct raw genre string |
| 2 · Copy prompt / Load saved mapping / Save mapping | Buttons | Prompt for an external LLM; mapping file `Docs/.genre_mapping.json`. Load asks before replacing unsaved edits. |
| Mapping editor | Text | JSON, validated live. Unsaved text is kept per library across library switches. |
| 3 · Preview changes | Button (primary) | Table of File / Current genres / After, plus counts and files that can't be written |
| 4 · ✎ Apply N changes / Cancel | Buttons | Enabled only for a preview of the current mapping text; confirms; writes exactly the preview |

Preview is disabled while any apply runs. A library change cancels scans, previews and
lookups and drops their results.

#### 6.6.3 Undo changes (`tag_history.py`)

Rolls back one Tag Fixer or genre apply from its change log (`controllers/tag_revert.py`).

| Element | Type | Purpose |
|---|---|---|
| Tag applies in this library | List | Every change log in `Docs/` that wrote something, newest first: When, Tool, Files, Changes, Status ("Undone" in muted text once reverted; an undo is listed as "Undo"). Re-read on Refresh, on showing the tab, and after an undo. |
| Preview undo | Button (primary) | Read-only job: per file and tag, what it holds now and what it gets back. Files edited since the apply, missing or unreadable are listed as "Left alone". Writes nothing. |
| Undo these changes | Button (primary) | Enabled only with a preview that restores something. Confirms, then writes exactly the preview, re-checking each file first. Holds the tag-write guard and the library-write lock. Logged to `Docs/tag_changes_revert_*.jsonl`, so it can be undone too. |
| Cancel | Button | Stops the preview or the undo at the next file |

Selecting another apply drops the shown preview. A library change drops the preview; a confirmed undo finishes on its own library.

### 6.7 Player (`player.py`)

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

### 6.8 Sound Map

Analyse → map → select → play / save. `SoundMapPage` (`gui/pages.py`) adds page
actions right of the tabs, visible in both sections:

| Action | Purpose |
|---|---|
| Save cluster playlists… | Previews one playlist per cluster of the latest analysis (§7.6), then saves them to `Playlists/Clusters/<date time> <method>/`. The button belongs to a `ClusterPlaylistFlow` (`gui/cluster_playlist_jobs.py`) owned by the Map panel; the page places it. Enabled when the library has `Docs/cluster_members.json` (it does not need the map, or PyQtGraph); disabled with a tooltip without one ("Analyse your library first…") and while an analysis runs anywhere ("Wait for the analysis to finish."). While the read-only preview job runs it reads **Cancel preview**. The preview job is read-only (`ask_before_quit=False`); the save job takes the library-write lock. Both check the lock first ("Library Busy"). A result line ("Saved N cluster playlists to … (K replaced, backed up as .bak; E failed).") goes to the log and the Map's status line. |
| Quality report | The latest analysis' cluster quality report (§7.3); enabled once an analysis with scores finished in this session (`EnhancedClusteredWorkspace.result_changed` / `quality_report_available()`). |

While Save cluster playlists runs, Analyse's Run buttons wait for it. When the
wizard's "Offer to save a playlist per cluster" was ticked, a successful run opens the
preview (`analysis_succeeded` → `offer_after_run`); if the Sound Map is not on screen
then, the offer waits until it is shown.

#### 6.8.1 Analyse (`clustered_enhanced.py`)

Formerly the Clustered page; titled "Analyse your music". **Analysis only:** a run writes the feature
cache, `Docs/cluster_members.json` (full membership), `Docs/cluster_info.json` and
`Docs/cluster_graph.html`; it does **not** write playlists (Save cluster playlists…
does, after its preview). The analysis job is named `Clustering`
(`gui.cluster_playlist_jobs.ANALYSIS_JOB_NAME`).

| Tab | Contents |
|---|---|
| 🚀 Quick Start | States the fixed settings (8 K-Means clusters; tempo, MFCC and energy; standard normalization) and a Run Quick Start button. |
| ⚙ Advanced | Configure with Wizard… (§7.2), the chosen configuration as read-only text, and Run. Kept separate from Quick Start. |
| 📊 Results | Progress bar, status, live log and a **Cancel** button during the run (cooperative; finishes the files being analysed). Then: clusters created, total tracks, unclustered (noise) count, tracks that could not be analysed, silhouette score, a hint pointing at **Save cluster playlists…** (or a lasso export from the map), **View Quality Report** and **Open the map**. |

Noise is not counted as a cluster and is excluded from the scores. Reserved folders
are not clustered. A library change cancels a run and resets the results (and
disables the Sound Map's Quality report). "Open the map" emits
`navigate_requested("graph")`, which switches to the Map section. (The old "View Playlists" button,
which used a Linux-only `xdg-open` on a folder with no cluster playlists in it, is gone.)

#### 6.8.2 Map (`graph.py`)

Formerly the Music Graph page; titled "Explore the map". The in-app interactive
cluster map, plus the browser-based 3-D view. With no cluster data it says to run an
analysis first (the Analyse tab). Loading,
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
| Selection bar | Label + buttons | "N tracks selected", ▶ Play selection, Export CSV…, Export playlist… (checks the library-write lock first; written through `playlist_generator`, so a replaced playlist gets a `.bak` beside it and a line in `Docs/playlist_writes.log`) |

The page **reloads itself when shown** if `Docs/cluster_info.json` changed; a failed
reload clears the plot and legend. The plot background, selection outline and marquee
follow theme switches. Without a working PyQtGraph, the real import error is shown and
the graph controls stay disabled. Selections go to the Player via
`play_tracks_requested`.

The browser page: orbit/zoom, hover tooltip, click-to-select with CSV / `.m3u` export,
legend visibility toggles, a spread slider with 1×/10×/30×/50×/100× presets, axes /
grid / orbit-ring toggles, auto-rotate, and Import JSON. It is self-contained (data
inlined at generation; a built-in sample set when opened with none).

### 6.9 Playlists (`playlists.py`)

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

### 6.10 Mirror

#### 6.10.1 Build the mirror (`compression.py`)

Formerly the Compression page. **A FLAC → Opus library mirror.** Output is always Opus
in a separate folder; other files are copied, and copied playlists have their FLAC
entries pointed at the `.opus`. No in-page title (the shell header says "Mirror", the
segmented control "Build the mirror"); one muted purpose line, then two cards.

**Hero card "Folders and format"** (`headerCard`):

| Element | Type | Purpose |
|---|---|---|
| Source / Destination | `PathField` rows (`gui/widgets/path_field.py`): the path middle-elided on a well, full path in the tooltip, a folder glyph button (and a click anywhere on the row) to choose | Refused, before anything is created: the same folder, a destination inside the source, or a source inside the destination (`check_mirror_folders`) |
| Format: Opus at N kbps | SpinBox with a " kbps" suffix | 48–320, default 96; disabled during a run |
| Update mirror files whose source has changed | Checkbox | Checked by default. A mirrored file older than its changed source is redone; unchecked, it is kept and reported as out of date |
| Overwrite every existing file in the destination | Checkbox | Redo the whole mirror. The confirmation states the choice |
| Estimate savings | Button (search glyph) | The preview. Runs in a worker; never creates the destination |
| Build mirror | Button (primary, play glyph) | Enabled only by an estimate for the current source / destination / bitrate; changing any of them invalidates the estimate. Asks first ("Build the Mirror"). Needs FFmpeg on `PATH` (the "Needs FFmpeg" note) |
| Cancel | Button (stop glyph) | Real cancel: terminates ffmpeg and removes partial output |

**Results card:**

| Element | Type | Purpose |
|---|---|---|
| Empty state | `EmptyState` | "No estimate yet": what to do first. Hidden once anything has happened |
| Status line | Label (toned) | What happened, plus only the explanations that need action: out-of-date files kept (tick "Update mirror files…"), files with no source left. The counts are in the tiles; the full summary still goes to the Activity log |
| Progress | Bar + "N of M files" | While running; the first row of run tiles updates live |
| Estimate | `StatRow`: FLAC to convert (with size) · Opus size at N kbps · Estimated saving · Other files to copy (with size) | |
| Run summary | Two `StatRow`s: Converted · Copied · Skipped · Errors / Updated · Out of date, kept · Partials removed · Orphans (tones: errors danger, kept and orphans warning; each tile has a tooltip) | |
| Open report | Button (external glyph) | `<destination>/Docs/opus_library_mirror_reports/opus_library_mirror_report_<timestamp>.html` |
| Remove N orphaned file(s)… | Button (`dangerBtn`, trash glyph) | Shown after a complete run that found mirror files with no source any more (deleted or renamed in the library). Lists up to 15 of them, asks (default No), then deletes only those, only inside the mirror (never one whose source came back), in a worker; every outcome is logged in the mirror's `Docs/`. Changing the folders hides it. |

Quarantine/, Trash/ and Docs/ are not mirrored. Output is encoded and tagged under a
temporary name and renamed when complete. Two sources that would land on the same
destination name (`Song.flac` and `Song.opus`), or folders differing only in case, are
resolved before any worker runs. Temporary files left by an interrupted earlier run are
removed. A library change during a run is applied after the run.

#### 6.10.2 Verify the mirror (`mirror_verify.py`)

The Opus and M4A testers (formerly Utilities → Diagnostics dialogs), **embedded** as
`MediaTesterPanel`s in two cards, side by side (stacked below ~960 px). Open a file and
see what `utils.audio_metadata_reader.read_metadata` (the reader the Indexer, Tag Fixer
and Player use) finds. Read-only.

| Element | Purpose |
|---|---|
| File row | `PathField` (folder glyph button, "Open a .opus file") |
| Empty state | "No file opened" until a file is open |
| Stat row | Tags found (of 10) · Cover art (Yes / No) · Problems (toned) |
| Problem rows | One glass row per problem (`find_problems`): could not read tags (danger), a reader warning, no title / artist / album tag, no cover art, cover not a readable image, cover only in the MP4 `covr` atom; or one "Tags and cover art came through." row |
| Cover + tags | The cover on a rounded well; the tags as text on a well |

### 6.11 Exports (`tools.py`, `ExportsWorkspace`)

Two tiles from the old Utilities page. Each tile's status chip reports running / done /
error / cancelled; both jobs run in workers with Cancel, skip Quarantine/, Trash/, Docs/
and Playlists/, and are cancelled by a library change (an export that finishes after a
library change does not re-enable Open buttons for the old library).

| Tile | Contents |
|---|---|
| Artist / Title Export | "Exclude FLAC files" and "Include per-album duplicate titles" toggles; Export, Cancel, Open File, open-folder. Writes `Docs/artist_title_list.txt`. |
| Codec List Export | Per-extension chips, "Filenames only (no full paths)"; Export, Cancel, Open File, open-folder. Writes `Docs/codec_file_list.txt`. |

### 6.12 Activity (`activity.py`)

Read-only.

| Card | Contents |
|---|---|
| Running jobs | Every running job in the JobRegistry, live (`jobs_changed`): Job, Started by (the page · section that owns it), Running for, and **Cancel** (`gui.workers.cancel_thread`: the job stops at its next safe point, as on quit; the button then reads "Stopping…"). "Nothing is running." when empty. |
| Log | Every line any page logged this session (up to 5000), with a filter (Everything / Warnings and errors) and Clear. |
| Problems | This session's warnings and errors, and the tail of `~/.alphadex/crash.log` and `~/.alphadex/faults.log` (what the old Crash Log viewer showed), with Refresh. |

### 6.13 Settings (`settings.py`, form in `gui/dialogs/settings_drawer.py`)

A page now (the sidebar and Ctrl+,). No in-page title; one header action row above the
sections: the purpose in one muted line, the save state, **Revert** and **Save settings**
(primary). Then one glass card per group (`SECTION_GROUPS` in `settings_drawer.py`). The
controls and the save logic are `SettingsForm`, the same form the old dialog had, so
**every control persists under the same config key:**

| Section · card | Control | Config key |
|---|---|---|
| Metadata services · Lookup service | Service rows (`ServiceList`): AcoustID / MusicBrainz / Last.fm, each with a one-line note; Spotify and Gracenote shown disabled ("Not available yet") with the "not yet implemented, see ROADMAP.md" tooltip | `metadata_service` |
| Metadata services · Connection | API key (a pill password field labelled with the service), shown for AcoustID and Last.fm — **one field, a separate key per service** | `metadata_api_key` (AcoustID; empty = built-in key) / `lastfm_api_key` |
| | MusicBrainz app name / version / contact, shown for MusicBrainz | `musicbrainz_useragent` |
| | Test connection (refresh glyph) + a toned status | Real checks: AcoustID and Last.fm test the entered key; MusicBrainz runs a search. Runs off the GUI thread. |
| General · Default library | `PathField` (folder button, clear button; validated if you change it) | `library_root` |
| General · Duplicate thresholds | Near duplicate / Exact duplicate: spin, unit ("fingerprint distance, 0–1") and an info dot, all with the help tooltip | `near_duplicate_threshold` / `exact_duplicate_threshold` (range wide enough that saved values aren't clamped) |
| General · Motion | Reduce motion — applies on Save | `reduced_motion` |
| Appearance · Theme | Theme picker (§5.3): theme, accent, Reduce transparency — apply instantly, no Save needed | `theme`, `glass_accent`, `reduce_transparency` (via ThemeManager) |
| Appearance · Backdrop | Album art / Gradient / Plain as a segmented pill (`SegmentedChoice`, the page tabs' look) — applies on Save | `backdrop_mode` (and the legacy `bg_gradient_enabled`, kept in step: False = Plain) |

The service rows and the backdrop pill keep a `QComboBox`-like API (`currentText` /
`setCurrentText` / `findText`, and `findData` / `currentData` / `setCurrentIndex`), so the
names `_service_combo` and `_backdrop_combo` still work.

Unsaved edits survive leaving and coming back; with no edits the page re-reads the
config when shown (so a library chosen with Change Library appears). Revert shows the
saved values again.

There is no Settings control for the mixed-codec boost (edited on the Duplicates page).
`ALPHADEX_REDUCED_MOTION=1` also turns reduced motion on.

On save the panel emits `settings_saved`; the window refreshes the cached backdrop mode
(and rebuilds the backdrop) and the reduced-motion flag, calls every panel's `on_settings_saved()` (Duplicates and Similarity
re-read their thresholds there), and applies a changed default library through
`_apply_library()`.

### 6.14 Help (`help.py`)

| Element | Purpose |
|---|---|
| Documentation | One button per maintained doc: `docs/README.md`, overview, architecture, every `docs/features/` page, `gui_inventory.md` and `ROADMAP.md` (a test checks that each exists) |
| Keyboard Shortcuts | The app-wide shortcuts (Ctrl+O, Ctrl+, for the Settings page, Ctrl+L, Ctrl+1–9 for the first nine pages, Ctrl+W). The Player's own shortcuts are not listed here. |
| About AlphaDEX | Former name, config file location, supported formats, and a Report an Issue (GitHub) button |

---

## 7. Dialogs

### 7.0 Dialog chrome (every dialog)

Top-level dialogs stay **opaque** (a translucent top-level window is unreliable on
Windows), but read as part of the glass family:

* the window fill is the palette's Window role, the material's `backdrop_base`;
* `AlphaDEXStyle` paints a raised content surface (the theme's opaque card colour) down
  to the bottom button box, which sits on a footer band of the base fill under a
  hairline (`paint_dialog_chrome`, `dialog_footer_top` in `gui/themes/style.py`);
* pill buttons; the default button of a button box is the primary pill (unless the box
  has its own accent button); mnemonic markers are not drawn ("Yes", not "&Yes");
  button boxes show no icons;
* message boxes: painted 40 px icons (a tone-tinted disc: accent for information and
  questions, warning, danger for critical), body-size text close to the icon, the
  informative text muted;
* dialogs that keep Qt's default layout margins (message, input, progress and file
  dialogs) get 22 / 20 / 22 / 16 px margins; item views in dialogs are 12 px-radius wells;
* the non-native file dialog (landing, Change library) gets line icons for its tool
  buttons.

Covered by `tests/ui/test_ui_takeaway_settings_glass.py` (palette, roles and painted
regions in Liquid Glass and Midnight).

### 7.1 SettingsDrawer (`gui/dialogs/settings_drawer.py`)

The same `SettingsForm` in a modal dialog (Metadata services and General tabs, the same
cards; Save is the primary, default button on the footer band). **The app no longer opens
it** (Settings is a page, §6.13); it is kept as a thin wrapper.

### 7.2 ClusteringWizardDialog (`gui/dialogs/clustering_wizard_dialog.py`)

A 5-step wizard from Analyse's Advanced tab. It keeps every choice when you
go Back and Next.

| Step | Contents |
|---|---|
| 1. Feature Selection | tempo / mfcc / chroma / spectral / energy / onset_rate checkboxes with tooltips, plus Fast / Balanced / Complete presets. All six are computed when ticked. |
| 2. Normalization & Preprocessing | Standard / MinMax / Robust (used). Dimensionality reduction (None / PCA / t-SNE / UMAP) is **disabled: not available yet**. |
| 3. Algorithm Selection | K-Means (cluster count) or HDBSCAN (min cluster size, min samples); the choice is passed through correctly. |
| 4. Post-Processing | Remove small clusters / merge into "Miscellaneous": **disabled, not available yet**. |
| 5. Output Options | **Offer to save a playlist per cluster when the analysis finishes** (live, off by default; run-config key `create_playlists`, never saved to the settings file): a successful run opens the Save cluster playlists preview (§7.6). Generate quality report / open graph: **disabled**, with tooltips pointing at what to use instead (the report is offered when scores exist; "Open the map"). |

Refuses to proceed if every feature is unticked.

### 7.3 ClusterQualityReportDialog

Silhouette / Davies-Bouldin / Calinski-Harabasz with verdicts, a per-cluster breakdown
and suggestions. Opened from Analyse's results ("View Quality Report") and the Sound
Map's **Quality report** action.

### 7.4 ThemePickerDialog / AutoThemeDialog

See §5.3 and §5.4. Only AutoThemeDialog is still opened (from the picker's Configure…).

`CrashLogDialog` (`gui/dialogs/crash_log_dialog.py`): a title, both log paths as one
middle-elided muted line (full paths in the tooltip), the log tail in monospace on a
rounded well, Close on the footer band. `MediaTesterDialog` wraps the Verify panel the
same way.

### 7.5 Dialogs opened from the Duplicates page

Each is deleted when closed; closing one mid-run cancels its worker.

| Dialog | Purpose |
|---|---|
| `BucketingPocDialog` | Duplicate bucketing proof of concept. Asks before running without fpcalc and then says "Completed without fingerprints". |
| `ScanEngineDialog` | Ten duplicate-scan tuning fields. **Run is disabled** without numpy/librosa (the tooltip says what to install). Its output database has no reader in the app yet. |
| `FuzzyDupeDialog` | Proof of concept that now runs the real `near_duplicate_detector` over the Duplicate Finder's cached fingerprints; thresholds default to Settings; results list verdict, distance and the pair. **Still not implemented, and disabled with a tooltip:** the "Metadata Match Settings" group, "Include pairs without fingerprints", "Skip fingerprint check", "Include non-matching FP pairs in review", and "Send Matches to Duplicate Pair Review". |
| `PairReviewDialog` | Side-by-side review of pairs from the Duplicate Finder's plan (`Docs/duplicate_preview.json`), or else files with the same name in the **same folder** (reserved folders skipped). Prev / Next / Switch / Prefer MP3 / Jump to. **"Quarantine inferior…"** confirms with both file names, moves the loser to `Quarantine/` under a unique name (never deleting, never overwriting), points playlist entries at the kept file, and logs to `Docs/duplicate_pair_review_log.txt`. "No — skip" skips; "Tag" appends the pair to `Docs/duplicate_pair_tags.txt`. Shortcuts: Backspace = skip, Left/Right = navigate. Enter no longer triggers the action. Refuses while another library write runs. |

The Opus / M4A testers are embedded now (§6.10.2; `MediaTesterDialog` remains as a
wrapper the app does not open), and `CrashLogDialog` is no longer opened: Activity shows
the same crash-log tail (`gui/workspaces/activity.py` reuses its `crash_log_paths()` /
`_tail()`).

### 7.6 ClusterPlaylistsDialog (`gui/dialogs/cluster_playlists_dialog.py`)

The Save cluster playlists preview, opened by the Sound Map's page action (§6.8) after
the read-only summary job (`cluster_playlist_export.summarize_cluster_run`). Modal;
nothing is written from it. Spec: `docs/specs/cluster_playlists.md`.

| Element | Purpose |
|---|---|
| Header | Method and parameter, number of clusters and tracks, when the analysis ran, and the destination (`Playlists/Clusters/<folder>/`). On a downsampled map: "The map shows 5,000 of 20,000 tracks; the playlists include all of them." |
| Warnings | Shown only when there are any: the library was moved (paths mapped), tracks added since the analysis, "changed a lot; consider re-analysing", the map shows an older analysis, earlier sets kept in `Playlists/Clusters/`. |
| Order / Minimum tracks | *Closest to the centre first* (default) or *Library order*; minimum 2–500, default 5, counted in present tracks. |
| Table | One row per cluster (noise last): Save checkbox (pre-ticked: every cluster except those hidden in the map legend; locked, with the reason as tooltip, while too small or too long for Windows), colour swatch (an icon, not a style sheet), **editable name**, tracks present, missing (tooltip lists up to 20 paths), and the action: New, Replace or Skip · reason. "Unclustered (N) — not saved" has no checkbox. Two ticked rows with the same name (ignoring case) disable Save. |
| Left unchanged | Files already in the run folder that this save does not touch. |
| Buttons | Select all / Select none, Open folder (only when the run folder exists; `QDesktopServices`), **Save N playlists (K replace)** and Cancel (the default button). |

Every option, checkbox and committed name edit re-plans on the GUI thread with the pure
`plan_cluster_playlists`. After Save the flow asks once if anything would be replaced
("K existing files … will be replaced (backed up first …)") and then writes exactly
`dialog.plan()`.

---

## 8. Widgets and shell modules

| Module | Status |
|---|---|
| `gui/navigation.py` | The information architecture as data (Qt-free): pages, sections, groups, `resolve()` |
| `gui/pages.py` (`SectionedPage`, `SoundMapPage`) | Live — §3.2 |
| `widgets/top_bar.py` (`TopBar`) | Live — the page header, §2 |
| `widgets/backdrop.py` (`Backdrop`) | Live — §5.5 (album-art / gradient / plain backdrop) |
| `widgets/glass.py` (`GlassCard`, `GlassPanel`, `ShadowLayer`, `ElidedLabel`) | Live — §5.5 |
| `widgets/sidebar.py` (`Sidebar`) | Live — §3.1 |
| `widgets/log_drawer.py` (`LogDrawer`) | Live — §4 |
| `widgets/now_playing_bar.py` (`NowPlayingBar`) | Live — §1.1 |
| `widgets/gradient_bg.py` (`GradientWidget`) | Live — §5.5 (transparent scroll surface; painted card shadows) |
| `widgets/landing.py` (`MosaicLanding`) | Live — §0 |
| `widgets/pill.py` (`Pill`, `ExperimentalPill`) | Live — §3.2 |
| `widgets/path_field.py` (`PathField`) | Live — a chosen path as a middle-elided row on a well with a glyph button; `QLineEdit`-like text API (§6.10, §6.13) |
| `widgets/interactive_scatter_plot.py` (`InteractiveScatterPlot`) | Live — the Map's plot |
| `widgets/cluster_legend.py` (`ClusterLegendWidget`) | Live — Map legend |
| `widgets/track_details_panel.py` (`TrackDetailsPanel`) | Live — Map details |

Elsewhere under `gui/`: `playback_backend.py` (the Player's backend seam), `workers.py`
(JobRegistry), `themes/motion.py` (motion tokens and helpers), `themes/glass.py` (the
glass painter and role fonts), `themes/appearance.py` (cached backdrop setting),
`imaging.py` (worker-side image blur), `library_stats.py` (library counts), `workspaces/label_tone.py`
(theme tones and a reusable flash for status labels), `workspaces/launchers.py` (the
Duplicates dialog sections) and `compat.py` (Qt imports; QtWebEngine is imported lazily
on first use). `gui/audio_preview.py` is used only by the legacy Tk app, and
`gui/qt_launcher.py` is a placeholder "Qt Preview" window launched only from the legacy
app; neither is part of the Qt app.

---

## 9. Signals flow summary

```
TopBar.library_changed ─┐
Settings page (saved) ──┼─► AlphaDEXWindow._apply_library(path)   (rejects missing, no-op if unchanged)
Startup (after fade) ───┘      └─► ws.set_library_path() for every panel
                               └─► _StatsWorker (cancels the previous one) → TopBar.set_stats + Home.set_library_stats
                               └─► config.save_config()  (library_root)

SettingsWorkspace.settings_saved ─► refresh gradient + reduced-motion caches
                                  └─► ws.on_settings_saved() for every panel

TopBar.settings_requested / Ctrl+, ─► Settings page
TopBar.theme_requested ─────────────► Settings → Appearance (ThemePickerPanel → ThemeManager.apply)
                                         └─► style tokens + palette swapped, themed widgets re-polished
                                         └─► theme_changed(tokens) → self-coloured widgets refresh

Sidebar.nav_changed / Ctrl+1..9 ──► AlphaDEXWindow.select_page(page)   (keeps the page's section)
WorkspaceBase.navigate_requested ─► AlphaDEXWindow._on_nav_changed(key) (page/section, panel, section or page key)
                                       └─► page.show_section(); stack shows the page; "Page · Section" in the status bar
                                       └─► visible panel's on_activated() if it defines one
SectionedPage tab click ──────────► section_changed → status bar; panel's on_activated()

JobRegistry.jobs_changed ─► Activity job list + the sidebar's Activity badge

Sidebar.exit_requested ──► AlphaDEXWindow.close()  (closeEvent job protocol)

WorkspaceBase.log_message ──► LogDrawer.append() + Activity log history   (buffered until connected)
WorkspaceBase.status_changed ─► LogDrawer.set_status() + QStatusBar
WorkspaceBase.play_tracks_requested ─► AlphaDEXWindow.play_tracks
                              └─► PlayerWorkspace.load_tracks_and_play(paths, label)

PlayerWorkspace ⇄ NowPlayingBar   (now-playing / position / state / volume_changed one way;
                                    play-pause / next / prev / seek / volume / keyboard mode the other)

MosaicLanding.tile_clicked ──► Startup.play_dir ──► AlphaDEXWindow.play_directory()
                              └─► Player page, begins playback
```
