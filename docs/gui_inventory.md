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
880 px wide), with its shadow over the page. It is **the app's only transport**: the
Player page has none of its own. It appears when the first track loads (it fades in)
and stays on every page; with nothing loaded it hides, **except on the Player page**,
which pins it (idle: "Nothing playing", or "Playback unavailable" without VLC). Pages
keep 104 px of room under their content so it never covers anything.

| Element | Purpose |
|---|---|
| Cover | 52 px round cover; a glass disc with a music glyph when the track has no art |
| Title / artist · album | Current track (elided to fit, full text in the tooltip); idle text when nothing is loaded |
| Shuffle | Glyph toggle (accent + dot when on) |
| ⏮ / ▶ ⏸ / ⏭ | Previous (restarts after 3 s), play/pause (46 px white disc), next |
| Repeat | Glyph button cycling Off → Track ("1" badge) → All |
| Seek slider + times | Drag, wheel, keys and groove clicks all seek; 11 px tabular times |
| Volume | Speaker + slider (kept in sync with the Player without echo) |
| ⌨ toggle | Turns **keyboard mode** on and off (see §6.7) |

Without a playback backend every transport control is disabled and its tooltip gives the
reason. **Width tiers** keep it inside its width: *full* (as above), *compact* (the volume
slider moves into a popup behind the speaker button), *narrow* (the times move into the
seek bar's tooltip). At the 900 px minimum window it uses the narrow tier. There is no Stop
button: pause, Previous (restart) and the queue's Clear cover it.

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

Preview first; Execute applies **that** preview. Liquid Glass layout (R2): one hero
card, an Options card, a Progress-and-results card and the Run log. The page does not
repeat its title (the shell header shows it). The glass parts it uses live in
`gui/workspaces/organize_parts.py`.

| Element | Type | Purpose |
|---|---|---|
| Hero: "Organize this library" | Hero card (`headerCard`) | Title, one-line purpose, the step pills, the library, the mode and the main action |
| Step pills | `StepPills` | Preview → Review the report → Execute. **Real state**: 0 without a preview of this library, 1 once one exists, 2 while Execute is selected or running |
| Library | Path well (`ElidedLabel`, middle elide) | The current library; full path in the tooltip. "No library selected…" without one |
| Dry run: preview only, nothing is moved | Checkbox | Checked by default. Unchecking switches the Run button to Execute. |
| Run | Button (primary, play glyph) | "Run preview" while Dry run is checked; "Execute previewed plan" otherwise, enabled only while a preview of the current library exists |
| Cancel | Button (close glyph) | Cancels the running preview or execute (the worker has its own cancel event) |
| Open report | Button (external glyph) | Opens `Docs/MusicIndex.html` |
| Create playlists after execution | Checkbox (Options) | Checked: regenerate playlists after the moves. Unchecked: only update existing playlists' paths. |
| Advanced | Well in Options | **Cross-album scan (Phase C)** and **Flush fingerprint cache before scan**: disabled, each tooltip says why (no cross-album phase; no cache to flush). **Max workers**: 1–32, default **1** (how many files have their tags read at once during a preview; more helps on a network share or slow drive). Locked while a run is going. The row wraps on a narrow window. |
| Progress | Three thin pills | Phase A — Scanning, Phase B — Planning, Phase C — Applying changes (Execute only); each pill's tooltip describes its phase |
| Status line | Label | What is happening, or what to do next ("Preview ready. Open the report…") |
| Results | `StatRow` / `EmptyState` | Before a preview: "No preview yet". After one: **Previewed plan** tiles: to move · in place · untagged (go to Manual Review/) · non-audio · empty folders. After Execute, the same tiles relabelled: moved · in place · problems (red when any) · tidied · removed. A library change goes back to the empty state. Each tile's tooltip says it in full. |
| Run log | Mono text | The run's log lines; a caption lists the Docs/ outputs |

Execute shows a confirmation with the plan's counts (files moved or renamed, how many
go to Manual Review, leftovers moved to Trash/Docs, folders removed, playlist mode).
The backend refuses a plan whose library changed since the preview. A plan runs at
most once; afterwards the page needs a fresh preview. A missing library folder is
refused rather than created.

Worker: `IndexerWorker` computes `music_indexer_api.IndexerPlan` (preview) or calls
`apply_indexer_plan()` (execute; a `LIBRARY_WRITES` job).

### 6.3 Library Sync (`library_sync.py`)

Scan both folders → review matches and flag → build plan → preview → execute that
plan. **It fits the window** at 1300 × 860 and at the 900 × 600 minimum with no
horizontal scrolling (it used to need ~1090 px): the folder grid, the scan progress
and the review lists reflow, and the plan buttons wrap
(`tests/ui/test_ui_organize_glass.py`).

| Element | Type | Purpose |
|---|---|---|
| Hero: "Merge an incoming folder" | Hero card | Title, one-line purpose, step pills, the two folders, the scan |
| Step pills | `StepPills` | Choose folders → Scan and match → Build and execute. Real state: 0 until a scan of the current folders exists, 1 while reviewing it, 2 while a plan is built, fresh or running |
| Library / Incoming | Entry + folder glyph button | Validated before a scan: both set, both exist, not the same, not nested. Folders are never created. |
| Scan both folders / Cancel | Buttons (search / close glyphs) | `SyncScanWorker`; Cancel stops a scan, plan build or execute |
| State label | Label | Idle / Scanning… / Re-matching… / Cancelling… / Complete / Cancelled / Error |
| Library / Incoming progress | Two thin pills + status | Shown from the first scan on; side by side when wide, stacked when narrow |
| Matching card | Form | **Threshold** (0–1, validated, warns in the warning tone when unrelated tracks would match) · **Format overrides** (`.ext=value` per line; extensions normalised to `.ext`) · **Preset name** (disabled, "Not used yet: presets are not implemented.") |
| Recompute matches | Button (refresh glyph) | **Wired:** re-matches the last scan with the current thresholds, without re-reading or re-fingerprinting files |
| Save session | Button (save glyph) | Saves the existing/incoming paths and threshold (`sync_existing`, `sync_incoming`, `sync_threshold`); they are **restored** next time. Flags are not saved. |
| Review matches | Card | Before a scan: `EmptyState` "No matches yet". After one: the tiles, the two lists and Match details |
| Scan summary | `StatRow` | incoming · new · in library · upgrades (accent when any) · uncertain (low confidence; warning tone when any); tooltips say it in full |
| Export report | Button (download glyph) | Matches, flags and notes as HTML, JSON or CSV |
| Incoming tracks | Glass list (`QTreeWidget`) | Track, Status (shown in words, e.g. "Exact match"; the item keeps the engine name), Distance, **Flag** (Copy / Replace), **Note**; right-click: Copy, Replace, Clear flag, Add note… |
| Matched in your library | Glass list | **Populated:** each library track an incoming one matched — Track, Status ("Matched" / "Upgrade available"), Best matches |
| (the two lists) | `ReflowSplitter` | Side by side (with a drag grip) from ~880 px of card width, stacked below |
| Match details | Text | The selected match: paths, status, distance, confidence, quality scores, candidates, your flag and note, and its planned decision once a plan exists |
| Summary line | Label | What to do next, and how many earlier flags could not be carried over |
| Build plan | Button (primary, list glyph) | Builds the plan **from the reviewed matches and flags**, plus its HTML preview (`Docs/LibrarySyncPreview.html`) |
| Preview plan | Button (external glyph) | Opens that preview |
| Copy originals / Move originals | Toggle | Transfer mode; changing it requires rebuilding the plan |
| Write transfer playlist (.m3u8) | Checkbox | Passed to execute |
| Execute plan | Button (danger, play glyph) | Runs the plan verbatim (a `LIBRARY_WRITES` job) after its confirmation |
| Plan summary | `StatRow` | to add · to replace · skipped · need review; shown only while the plan is up to date |
| Plan status + plan bar | Label + thin pill | Build and execute progress, out-of-date reasons, results |
| Log + Export log… | Mono text + button | Humanised, batched log; lines from abandoned workers are dropped |

Item lists here are "glass" views: transparent on the card, 34 px rows, a rounded
hover/selection band from the theme's material, muted 12 px headers, no grid, no
inline stylesheet (`organize_parts.style_item_view`).

**The plan is tied to its inputs.** Changing a path, the transfer mode, a flag, or
re-scanning or switching library marks the plan out of date and disables Execute,
with the reason. The confirmation states library, incoming folder, copy vs move and
the counts (added, replaced with a backup, skipped, left for review). The result
names the run's own audit file and how many items were skipped because their files
changed since the plan was built. Flags survive a re-scan where they still apply; the
rest are listed in a notice.

### 6.4 File Cleanup (`tools.py`, `FileCleanupWorkspace`)

Moved out of Utilities onto its own page. One hero glass tile (`ToolTile`, a
`GlassCard`) titled "Clean up copy suffixes", whose status pill reports Ready /
Running… / Done / Error / Cancelled. Before the first preview the tile shows an
`EmptyState` ("Nothing previewed yet"); after it, a result area fades in (status line
and the rename log).

**Preview cleanup…** (primary, search glyph) plans in a worker, then opens a **preview
dialog** (§7.7) with the counts as stat tiles and the plan as a list (folder, current
name → new name, collisions flagged and skipped); on confirm it renames exactly the
previewed rows, writes `Docs/file_cleanup_<timestamp>.jsonl` and updates playlists.
**Undo last cleanup…** (undo glyph) previews the newest cleanup log in the same
dialog, confirms, restores the original names and points the changed playlists back.
Only true OS copy artifacts are stripped (` (1)`…` (99)`, ` - Copy`, Finder ` copy N`);
reserved folders are skipped; nothing is ever overwritten. Both are library writes.
**Cancel** stops the preview or the renames. A library change cancels a preview and
resets the tile; a confirmed cleanup or undo finishes.
### 6.5 Duplicates

Six sections (§3.2). The first two are the real workflow; Pair review and the three
prototypes open their dialogs (§7.5). Every section is drawn in the Liquid Glass card
language (one hero card, glass rows, stat tiles, glyph buttons; parts in
`gui/widgets/dupes_parts.py`) and fits the minimum window (900 × 600) without
horizontal scrolling. No section repeats the page or section title: the hero card's
title says what the card is for.

#### 6.5.1 Find duplicates (`duplicates.py`)

Scan → Preview → Execute. The page always works on the current library (there is no
separate path field).

| Card | Element | Type | Purpose |
|---|---|---|---|
| Hero "Scan your library" | Library | Glass row, folder glyph + middle-elided path (full path in the tooltip) | The current library |
| | Find duplicates | Button (primary, search glyph) | Fingerprints and groups. If fpcalc or FFmpeg is missing, asks once (with install links) whether to scan using cached fingerprints only. |
| | Exact / Near / Mixed-codec boost | `QDoubleSpinBox` ×3, caption above each | Start from the Settings keys (`exact_duplicate_threshold`, `near_duplicate_threshold`, `mixed_codec_threshold_boost`) and follow later Settings saves |
| | Scan · Review · Execute | Pill row (`dupStepRow`; labels `stepActive` / `stepInactive`) | Where the flow is; moves on after a scan and after an execution |
| | Status, Cancel, progress | Label; button (close glyph); thin bar | Cancel and the bar only show while a job runs. Cancel stops the scan, or asks the execution to stop at the next safe point. |
| "Duplicate groups" | Empty state | `EmptyState` | Before the first scan ("No scan yet"), without a library, while scanning, after a scan with no groups ("No duplicates found"), after a cancel or a failed scan |
| | Plan tiles | `TileGrid` of `StatTile`s | Groups · Duplicate files · Space reclaimable (size of the copies that will leave the library; not counting copies kept in place) · Flagged for review |
| | Outcome tiles | `TileGrid`, shown after an execution | Quarantined · Deleted · Left in place for safety · Not attempted · Errors, counted by `execution_counts` from the same tally as `summarize_execution` |
| | Reason line | Muted label | Before Execute: how many groups are flagged and that nothing has moved; after: the `summarize_execution` sentence |
| | Group rows | `QTreeWidget#dupGroupList` painted by a delegate | One glass row per group: **Keep** and the kept file, then each loser with its **disposition** (Quarantine / Delete / Keep in place), format and bitrate pills, a **Review** pill when flagged (tooltip: full paths and flags). Three loser lines at most, then "+ N more". |
| | Selected group + Losers | Glass row + `QComboBox` | **Wired:** Default (global choice) / Keep loser in place (retain) / Quarantine / Delete permanently, per group. The rows and the space tile follow it at once. |
| | Group details | Mono text | Full paths with each loser's disposition, and the review flags |
| "Apply the plan" | Quarantine duplicates (safe default) / Delete losers permanently | Radio buttons, plus a danger "Cannot be undone" pill | The global disposition |
| | Update playlists after execution | Checkbox | **Shown checked and disabled on purpose:** the executor always rewrites playlist entries that point at a moved loser, so it is not a choice. |
| | Preview | Button (external glyph) | Opens `Docs/duplicate_preview.html` |
| | Report | Button (file glyph) | Opens the newest execution report, including one from an earlier session |
| | Execute… | Button (danger, check glyph) | See below. Disabled once the plan has run, until the next scan. |
| "Log" | Log | Mono text | Per-file failures are capped at 20 lines |

**Execute** in order: refuses if another library write is running; if any group is
flagged for review, lists the flagged groups and why, and asks whether to **override**
(execute all groups exactly as previewed) or stop with nothing moved; writes the chosen
dispositions into the plan and re-exports the preview so the preview matches what will
run; confirms with counts (quarantined, **permanently deleted**, retained, tag and
artwork changes; a plan over 20,000 operations is called out); and, if anything will
be deleted, asks you to **type DELETE**, which opens the executor's double deletion
gate. Preflight refusals are reported with their reason.

The scan also writes `Docs/duplicate_preview.json`, which the Pair review section reads.

A library change cancels a running scan and ignores its result; a confirmed execution
is left to finish on the library it started on (its report goes to that library's
`Docs/`), like every other page's confirmed writes.

#### 6.5.2 Compare two files (`similarity.py`)

A two-file duplicate diagnostic that uses **the same pipeline and settings as Find
duplicates**.

| Card | Element | Type | Purpose |
|---|---|---|---|
| Hero "Pick two files" | File A / File B | Glass rows: letter badge, middle-elided path (tooltip: full path), Browse… (folder glyph) | The two audio files. Dropping an audio file on a row picks it too. |
| | Compare / Clear | Buttons (primary with search glyph; close glyph) | Clear also discards a late result from a running comparison |
| "Verdict" | Empty state | `EmptyState` | Until the first comparison |
| | Verdict | Pill | Exact duplicate (success) · Near duplicate (accent) · Not a duplicate (neutral) · Could not compare (danger); "Thresholds from Settings / Advanced Overrides" beside it |
| | Distances | `TileGrid` | Fingerprint distance (the verdict's) · **Overlap-only distance** · Effective threshold · Exact threshold |
| | Explanation | Muted text | Which threshold decided it, a clip-of-a-song hint, codecs and the mixed-codec boost |
| | Full report | Fold-out mono text | Files, fingerprint sources (the Find duplicates cache at `<library>/Docs/.duplicate_fingerprints.db`, fpcalc, or pyacoustid as a last resort), codecs, distances, thresholds, verdict |
| "Advanced overrides" | Checkbox | Off by default | Off: the fields fold away and one line names the Settings thresholds in use. On: trim silence, fingerprint offset / duration, silence threshold / min length, and exact / near / mixed-codec thresholds are used. An invalid field is named in a warning. |

Thresholds are re-read when Settings saves (`on_settings_saved`).

#### 6.5.3 Pair review (`launchers.py`, `PairReviewWorkspace`)

One card, "Decide pair by pair": what the tool does (pairs from the duplicate preview
in `Docs/`; tags, format and art side by side; a rejected file is moved to
`Quarantine/`, never deleted; decisions logged to `Docs/`) and **Open Pair Review…**
(external glyph), which opens `PairReviewDialog` (§7.5). Without a library it warns;
without a duplicate preview it says "No duplicate preview found — run Find duplicates
first".

#### 6.5.4 Fuzzy finder · Scan engine · Bucketing — **Experimental**

One card each, titled with what the prototype is for ("Near-duplicate candidates",
"A staged duplicate scan", "Duplicate buckets report") with the **Experimental** pill
beside the title, the note *"Experimental — not ready to test. A prototype of another
way to find duplicates: its results are not used by Find duplicates and have not been
checked."* in a well with a warning glyph, what the prototype does, and an **Open …**
button for its dialog (`FuzzyDupeDialog`, `ScanEngineDialog`, `BucketingPocDialog`;
§7.5). There is at most one live dialog per tool: closing one whose worker still runs
keeps it, and the next Open re-shows it instead of starting a second run.

### 6.6 Metadata

Four sections in the Liquid Glass look (redesign R2). None repeats the page or section
title: each view starts with one muted purpose line, then **one hero card** (radius 30,
title role) holding its main action, then a plan/review card. Item views are
transparent "glass rows" (`gui/workspaces/metadata_views.py`: rounded 34 px rows,
whole-row hover and selection from the material, one hairline between rows; a value
that would be written sits on an accent-tinted chip, current values are muted, tools and
"drop" values are pills; all painted from the theme at paint time, no per-item colours).
Counts are `StatRow`s; before there are results an `EmptyState` says what will appear.
Buttons carry painted glyphs (no emoji). Paths are elided in the middle with the full
path in the tooltip, status lines wrap, and every section fits the 900×600 minimum
window with no sideways scrolling (`tests/ui/test_ui_meta_glass.py`).

#### 6.6.1 Tag Fixer (`tag_fixer.py`)

Find tag fixes (lookups only) → tick field changes → Apply → confirm. There is no
service selector on this page; the sources come from Settings (see
features/tag_fixer.md).

| Element | Type | Purpose |
|---|---|---|
| **Tag lookup** (hero card) | Card | The scan's scope, sources and main action |
| Lookup sources line | Label (`_reason_lbl`) | Before a scan: where the sources are chosen. After: which lookup sources ran, and why nothing matched when nothing did |
| Find tag fixes | Button (primary, search glyph) | `TagFixWorker`; never writes |
| Cancel | Button (flat, close glyph) | Stops a scan or an apply at the next file |
| Scope: Whole library / One folder | Pill + elided path | What the scan looks at |
| Choose folder… / Whole library | Buttons (folder / refresh glyph) | Scan one folder inside the library instead of all of it. Folders outside the library and the reserved folders the Tag Fixer never touches are refused. A library change resets the scope. The database and change log stay the library's. |
| Lookup sources | Button (flat, external glyph) | Opens Settings → Metadata services |
| Progress | Status line + thin bar | A real percentage |
| **Review fixes** | Card | |
| Proposed changes / Files to fix / Ticked to write / No match | `StatRow` | Counts after a scan |
| Show: Title / Artist / Album / Genre | Checkboxes | Show or hide that field's rows; hidden rows are never written |
| Review grid | Glass tree | Tick, File, Field, Current (muted), Proposed (accent chip), Score. **One row per field change**; each tick selects that one field. As tall as its rows (up to 10, then it scrolls). Empty state before a scan; "Nothing to fix" / "Nothing could be matched" after one. |
| Tick all / Untick all | Buttons (flat, glyphs) | Shown rows only |
| Dry run | Checkbox | Checked by default; while checked, Apply only shows a message |
| Apply N changes | Button (success, check glyph) | Writes exactly the visible, ticked rows after a confirmation (a `LIBRARY_WRITES` job that also takes the shared tag-write guard) |
| **No match found (N)** | Card + glass list | Only when something was unmatched: each file with its reason |
| **Log** | Text (log role) | |

After an apply, the written rows drop out, the log is kept, and the change-log path
(`Docs/tag_changes_tagfixer_<timestamp>.jsonl`) is shown. There is no automatic
re-scan. A library change cancels a scan and clears the grid; a confirmed apply
finishes.

#### 6.6.2 Fill genres · Normalize genres (`genres.py`)

One panel serves both sections as two views: the Metadata page's tabs switch it (its own
tab bar only shows when the panel stands alone), with a short cross-fade (`motion.reveal`;
at once under reduced motion). The view stack is sized to the view on show. See
features/tag_fixer.md. Both read every genre value of a file, write real multi-value
lists, skip files changed since the preview, log every write to
`Docs/tag_changes_*.jsonl`, and share one tag-write guard with the Tag Fixer.

**Fill genres** (preview-first)

| Element | Type | Purpose |
|---|---|---|
| **Fill missing genres** (hero card) | Card | States the rules: top MusicBrainz community tags, existing genres kept, files with 2+ genres skipped |
| Look up genres | Button (primary, search glyph) | Lookups only; builds the plan, writes nothing |
| Cancel | Button (flat) | Stops the lookup or the apply |
| Progress + status | | |
| **Proposed additions** | Card | |
| Would get genres / Already 2+ genres / No usable match / Errors / can't write | `StatRow` | |
| Plan | Glass table | File, Current genres (muted), After (accent chip), MusicBrainz match. Empty state before the lookup. |
| Apply N additions | Button (danger, check glyph) | Enabled once proposals exist; confirms, then writes exactly the table (a `LIBRARY_WRITES` job) |
| **Log** | Text | |

**Normalize genres** (backed by `controllers/normalize_controller.py`)

| Element | Type | Purpose |
|---|---|---|
| **Build the mapping** (hero card) | Card | The inputs and the main action |
| Preview changes | Button (primary, list glyph) | Read-only job: the plan below, plus counts and files that can't be written |
| Scan library / Copy list / Copy prompt | Buttons (glyphs) | Every distinct raw genre string (shown as wrapping chips); the prompt for an external LLM |
| Load saved mapping / Save mapping | Buttons (glyphs) | Mapping file `Docs/.genre_mapping.json`. Load asks before replacing unsaved edits. |
| Mapping editor | Text (mono) | JSON, validated live. Unsaved text is kept per library across library switches. |
| Mapping table | Glass tree | Raw genre → Becomes, following the editor; `["invalid"]` and `null` show as a **Drop** pill. Side by side with the editor when there is room, below it when not. |
| **Preview** | Card | |
| Would change / Already canonical / No genre tag / Can't write | `StatRow` | |
| Plan | Glass table | File, Current genres (muted), After (accent chip; "All dropped" as a pill). Empty state before a preview. |
| Apply N changes / Cancel | Buttons (danger / flat) | Enabled only for a preview of the current mapping text; confirms; writes exactly the preview |

Preview is disabled while any apply runs. A library change cancels scans, previews and
lookups and drops their results.

#### 6.6.3 Undo changes (`tag_history.py`)

Rolls back one Tag Fixer or genre apply from its change log (`controllers/tag_revert.py`).

| Element | Type | Purpose |
|---|---|---|
| **Past applies** (hero card) | Glass rows | Every change log in `Docs/` that wrote something, newest first and preselected: When, Tool (a pill; an undo in the warning tone), Files, Tags, Status (an apply already undone is muted, its pill grey). Re-read on Refresh, on showing the tab, and after an undo. Empty state when nothing has been applied. |
| Refresh | Button (icon only, refresh glyph) | Re-reads the change logs |
| Preview undo | Button (primary, list glyph) | Read-only job: per file and tag, what it holds now and what it gets back. Writes nothing. |
| **What the undo would restore** | Card | |
| Tags to restore / Files / Left alone / Already restored | `StatRow` | |
| Preview | Glass tree | File, Tag, Now (muted), Restores to (accent chip). Files edited since the apply, missing or unreadable are muted rows with a **Left alone** pill and the reason. Empty state before a preview. |
| Undo these changes | Button (danger, undo glyph) | Enabled only with a preview that restores something. Confirms, then writes exactly the preview, re-checking each file first. Holds the tag-write guard and the library-write lock. Logged to `Docs/tag_changes_revert_*.jsonl`, so it can be undone too. |
| Cancel | Button (flat) | Stops the preview or the undo at the next file |

Selecting another apply drops the shown preview. A library change drops the preview; a confirmed undo finishes on its own library.

### 6.7 Player (`player.py`)

An in-app audio player behind a **playback backend seam** (`gui/playback_backend.py`). The
page is the library and the queue; **the Now Playing capsule (§1.1) is its transport**, and
the page pins it so it is always there. With no working backend (python-vlc missing, or
installed without libVLC) a quiet **"Playback unavailable"** notice (warning glyph and the
reason) sits under the purpose line, the capsule's transport is disabled, and nothing ever
pretends to play.

| Element | Type | Purpose |
|---|---|---|
| Purpose line | Muted text | "Double-click a track to play it, or build a queue. The bar at the bottom is the transport." |
| Library card | Glass card | The table on glass; the playing track's cover glows faintly through it (cross-fades between tracks) |
| Search | Glass pill | Filters Title / Artist / Album (the page header's search routes here); "No matches" when nothing does |
| Reload | Glyph button | Re-scan the library (reserved folders Playlists/Docs/Trash/Quarantine are skipped; Not Sorted and Manual Review are listed) |
| Open file | Glyph button | Open and play a single file |
| Library table | Sortable table | #, Title, Artist, Album, Time (default sort Artist / Album / Track); 34 px rows, muted header, the playing row tinted with the accent and marked ▶/⏸; narrow tables hide Album, then #. Right-click: Play now, Add to queue, Open folder, Save selection as M3U… |
| Status line · 30 s preview | Footer | Scan / play status; the audition toggle (seek to 0:30, advance after 15 s; pauses and resumes with playback) |
| Empty states | Text | No library selected / Reading your library… / No tracks found / No matches |
| Cover carousel | Display | Previous / current / next as glass album tiles; shown when the column has room for it and the queue |
| Hover popup | Tooltip-like card | After a 1.5 s dwell over a row: art and tags (loaded asynchronously) |
| Queue | Glass rows + glyph buttons | Add selected (+), Load M3U (folder), Save queue as M3U (save), Clear (trash); right-click: Play now, Remove from queue; "Queue is empty" state |
| Recently played | Collapsible glass rows | Chevron toggle; up to 30 entries |

Tracks without embedded art use the folder's cover image (`cover.jpg`, `folder.jpg`, …).
There are **no** seek-back / seek-forward buttons and no Stop button.

**Shortcuts** (active while the Player page is showing): `Space` play/pause, `N` next,
`P` previous, `Shift+Right` / `Shift+Left` seek ±30 s. There is no plain Left/Right
±5 s seek.

**Keyboard mode** (the ⌨ toggle on the Now Playing capsule) maps the plain arrow keys to
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
actions right of the tabs, visible in both sections, both glass pills with glyphs:

| Action | Purpose |
|---|---|
| Save cluster playlists… (save glyph) | Previews one playlist per cluster of the latest analysis (§7.6), then saves them to `Playlists/Clusters/<date time> <method>/`. The button belongs to a `ClusterPlaylistFlow` (`gui/cluster_playlist_jobs.py`) owned by the Map panel; the page places it. Enabled when the library has `Docs/cluster_members.json` (it does not need the map, or PyQtGraph); disabled with a tooltip without one ("Analyse your library first…") and while an analysis runs anywhere ("Wait for the analysis to finish."). While the read-only preview job runs it reads **Cancel preview**. The preview job is read-only (`ask_before_quit=False`); the save job takes the library-write lock. Both check the lock first ("Library Busy"). A result line ("Saved N cluster playlists to … (K replaced, backed up as .bak; E failed).") goes to the log and the Map's status line. |
| Quality report (info glyph) | The latest analysis' cluster quality report (§7.3); enabled once an analysis with scores finished in this session (`EnhancedClusteredWorkspace.result_changed` / `quality_report_available()`). |

While Save cluster playlists runs, Analyse's Run buttons wait for it. When the run's
"Offer to save a playlist per cluster" was ticked (settings card or wizard), a
successful run opens the preview (`analysis_succeeded` → `offer_after_run`); if the
Sound Map is not on screen then, the offer waits until it is shown. The Analyse results
card repeats **Save cluster playlists…**: it starts the same flow
(`save_playlists_requested` → `flow.start()`) and mirrors the page button's enabled
state and tooltip (the page watches the flow button's enabled/tooltip changes).

#### 6.8.1 Analyse (`clustered_enhanced.py`)

Formerly the Clustered page. One muted intro line, then three glass cards. **Analysis
only:** a run writes the feature cache, `Docs/cluster_members.json` (full membership),
`Docs/cluster_info.json` and `Docs/cluster_graph.html`; it does **not** write playlists
(Save cluster playlists… does, after its preview). The analysis job is named
`Clustering` (`gui.cluster_playlist_jobs.ANALYSIS_JOB_NAME`).

| Card | Contents |
|---|---|
| **Quick Start** (hero, `headerCard`) | The fixed settings in one line (8 K-Means clusters, listening for timbre, tempo and energy, standard scaling), **Run Quick Start** (primary, play glyph) and **Guided setup…** (the wizard, §7.2), which opens on the settings card's current choices and writes its answers back into it. |
| **Results** | Before the first run an `EmptyState` ("No analysis yet"). During a run: a title with **Cancel** (cooperative; finishes the files being analysed), the progress bar, status and live log. After: "Latest analysis" with the method and features, a `StatRow` (Clusters, Tracks, Unclustered when HDBSCAN left any, Not analysed when tracks failed, Silhouette · Good/Fair/Poor toned success/warning/danger), the hint pointing at **Save cluster playlists…** (or a lasso export from the map), and **Open the map** (primary), **Save cluster playlists…** (the page flow, see above) and **Quality report** (ghost; only with scores). A failure shows a warning glyph, "The analysis did not finish" and the error; a cancel "Analysis cancelled". Results fade in (`motion.reveal`). |
| **Your own settings** | *Listen for*: one `PillToggle` per feature block in `clustered_playlists.FEATURE_ORDER` (Timbre, Tempo, Harmony, Brightness, Energy, Rhythm density; six in a row when the content is ≥ 900 px wide, three per row otherwise). *Algorithm* (K-Means / HDBSCAN), *Scaling* (Standard / MinMax / Robust), *Clusters* (K-Means) or *Min cluster size* and *Min samples* (HDBSCAN). **Offer to save a playlist per cluster when the analysis finishes** (run-config key `create_playlists`, never saved to the settings file). A summary line of what Run will do, and **Run with these settings** (disabled while no feature is ticked). Starts from the Quick Start settings; a Quick Start run never changes it. |

Noise is not counted as a cluster and is excluded from the scores. Reserved folders
are not clustered. A library change cancels a run and resets the results (and
disables the Sound Map's Quality report). "Open the map" emits
`navigate_requested("graph")`, which switches to the Map section.

#### 6.8.2 Map (`graph.py`)

Formerly the Music Graph page. One muted intro line; the map is the hero card. Loading,
validation and export live in the Qt-free `cluster_graph_data.py`.

| Element | Type | Purpose |
|---|---|---|
| Pan / Box / Lasso | Segmented control (exclusive, glyphs move / marquee / lasso) | Drag behaviour; Ctrl or Shift adds to the selection. The chosen mode is a raised pill on a well track. |
| Fit view / Clear selection / Reload | Icon pills (fit / close / refresh glyphs, tooltips and accessible names) | Fit every point; clear the selection; re-read the cluster data |
| 3-D view | Glyph pill (external) | Opens (generating if needed) the browser 3-D page |
| Selection bar | Label + buttons, above the plot | "N tracks selected" with the map summary or the last result under it (middle-elided, full text in the tooltip); **CSV…**, **Export playlist…** (checks the library-write lock first; written through `playlist_generator`, so a replaced playlist gets a `.bak` beside it and a line in `Docs/playlist_writes.log`) and **Play selection** (primary) |
| Scatter plot | `InteractiveScatterPlot` | On a glass well with a hairline grid that pans and zooms with the points (no axes: the embedding has no units). One dot per track, a vivid core with a soft ring, coloured by cluster; noise grey. Hover rings the point and identifies it; **a click pins** the track in the details card. The selection is ringed in the accent and the rest dims; the lasso / box marquee is a dashed accent line. |
| Empty state | `EmptyState` on the same well | "No map yet" + **Go to Analyse** (no data), "The map could not be loaded" + the error (bad data), "Choose a library first" (no library). Without PyQtGraph the plot's own notice shows instead. |
| Cluster legend | `ClusterLegendWidget` in a glass card | Glass rows: the colour dot is the show/hide toggle (hiding drops the cluster's tracks from the selection; a hidden row's text dims), name, genres · tempo, count; clicking a row selects the cluster; Show all / Hide all |
| Track details | `TrackDetailsPanel` in a glass card | Empty state until a track is hovered or pinned; then the cover (rounded, or a music glyph), title, artist, album, Genre / Year / Cluster, the path (middle-elided) and **Play** / **Open folder** |

Layout: the legend and details cards are a 300 px column right of the map when the
content is at least 900 px wide; below the map, side by side, from 680 px; stacked in a
900 px window. The page never scrolls sideways.

The page **reloads itself when shown** if `Docs/cluster_info.json` changed; a failed
reload clears the plot and legend. The well, grid, selection ring and marquee follow
theme switches (the point sprites do not depend on the theme, so a switch never redraws
the points). Without a working PyQtGraph, the real import error is shown and the graph
controls stay disabled. Selections go to the Player via `play_tracks_requested`.

The browser page: orbit/zoom, hover tooltip, click-to-select with CSV / `.m3u` export,
legend visibility toggles, a spread slider with 1×/10×/30×/50×/100× presets, axes /
grid / orbit-ring toggles, auto-rotate, and Import JSON. It is self-contained (data
inlined at generation; a built-in sample set when opened with none).

### 6.9 Playlists (`playlists.py`)

Every kind is **preview-first**: a background job works out exactly which `.m3u`
files would be written and which existing ones replaced, a confirmation lists them,
and only then is exactly that plan written, with a timestamped `.bak` of anything it
replaces and a line in `Docs/playlist_writes.log`.

**Make playlists** (hero card): a segmented control picks the kind, a one-line
description explains it, its options sit in a well, and one primary button previews it
(its text names the kind). While a job runs the button is disabled and a busy bar and
**Cancel** (neutral "Cancelled" status) appear beside it.

| Kind | Options |
|---|---|
| Folder | Save to (blank = `Playlists/`; folder glyph to choose). One playlist per album folder; reserved folders skipped. |
| Tempo & energy | Tempo range (BPM) and Energy range: tracks outside them are left out. |
| Auto-DJ | Length (track count), Start from (seed track: must be a library track outside the reserved folders) → `Playlists/autodj.m3u` |
| Repair | Playlists to check: glass rows with add (+), remove (−) and clear glyphs (empty = every playlist in `Playlists/`), **Prefer Opus when a FLAC entry is missing** |

**Preview** card: an empty state before the first run; then the planned files as glass
rows (pills: *new*, *replace*, then *written* / *failed* after the write; track counts)
under a stat row (playlists · tracks · replaced, or for Repair: entries re-pointed · still
missing). Cancel, "Nothing to write" and errors show here too. **Details** opens the job's
log. Writes check the library-write lock before the confirmation. A library change cancels
planning (and clears the preview) but not a confirmed write.

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

Two glass tiles (`ToolTile`, a `GlassCard` with a painted shadow) from the old
Utilities page, under a one-line purpose. Each tile's status pill reports Ready /
Running… / Done / Error / Cancelled. Before the first export a tile shows an
`EmptyState` ("No list yet"); an export fades in the result area: a thin progress pill,
the status line, the **written file as a middle-elided path** (full path in the
tooltip) and, for Artist / Title, the log. Both jobs run in workers with Cancel, skip
Quarantine/, Trash/, Docs/ and Playlists/, and are cancelled by a library change, which
also resets the tiles (an export that finishes after a library change does not
re-enable Open buttons for the old library). Two tiles side by side from ~1000 px of
page width, stacked below.

| Tile | Contents |
|---|---|
| Artist / Title list | "Exclude FLAC files" and "Include per-album duplicate titles" switches; Export (primary, download glyph), Cancel, Open file, a round folder button ("Show the exported file in its folder"). Writes `Docs/artist_title_list.txt`. |
| Codec list | Per-extension chips (painted toggles: accent and a check when included), "Filenames only (no full paths)"; the same four buttons. Writes `Docs/codec_file_list.txt`. |

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

A 5-step wizard, opened by **Guided setup…** on Analyse. It opens on the settings card's
current choices (`set_config`) and its answers fill that card in; it keeps every choice
when you go Back and Next. A title, a step counter and a row of step pills (done steps
carry a check, the current one is raised) over one glass card per step; Cancel on the
left, Back and **Next / Finish** (primary) on the right.

| Step | Contents |
|---|---|
| 1. Feature Selection | One pill toggle per feature block in the engine's order (Timbre, Tempo, Harmony, Brightness, Energy, Rhythm density) with tooltips, plus Fast / Balanced / Complete presets. All six are computed when ticked. |
| 2. Normalization & Preprocessing | Standard / MinMax / Robust (used). Dimensionality reduction (None / PCA / t-SNE / UMAP) is **disabled: not available yet**. |
| 3. Algorithm Selection | K-Means (cluster count) or HDBSCAN (min cluster size, min samples); the choice is passed through correctly. |
| 4. Post-Processing | Remove small clusters / merge into "Miscellaneous": **disabled, not available yet**. |
| 5. Output Options | **Offer to save a playlist per cluster when the analysis finishes** (live, off by default; run-config key `create_playlists`, never saved to the settings file): a successful run opens the Save cluster playlists preview (§7.6). Generate quality report / open graph: **disabled**, with tooltips pointing at what to use instead (the report is offered when scores exist; "Open the map"). |

Refuses to proceed if every feature is unticked.

### 7.3 ClusterQualityReportDialog

Glass cards: **Overall** is a `StatRow` (Silhouette · good / fair / poor, Davies–Bouldin,
Calinski–Harabasz, and Unclustered when HDBSCAN left any), toned success / warning /
danger; the silhouette uses the same thresholds as the Analyse results and the
suggestions (good above 0.5, fair above 0.3). **Per cluster** lists glass rows (name,
genres · tempo range, size, its silhouette in the tone colour). **Suggestions** follow.
Opened from Analyse's results ("Quality report") and the Sound Map's **Quality report**
action.

### 7.4 ThemePickerDialog / AutoThemeDialog

See §5.3 and §5.4. Only AutoThemeDialog is still opened (from the picker's Configure…).

`CrashLogDialog` (`gui/dialogs/crash_log_dialog.py`): a title, both log paths as one
middle-elided muted line (full paths in the tooltip), the log tail in monospace on a
rounded well, Close on the footer band. `MediaTesterDialog` wraps the Verify panel the
same way.

### 7.5 Dialogs opened from the Duplicates page

Each is deleted when closed; closing one mid-run cancels its worker. Their contents use
the Duplicates area's glass parts: a title-role heading, glass section cards, captions
over fields, glyph buttons (no emoji) and theme-following text with no inline style
sheets. (The dialog window itself is the shell's.)

| Dialog | Purpose |
|---|---|
| `BucketingPocDialog` | Duplicate bucketing proof of concept: a folder card (path + Browse…) and Run. Asks before running without fpcalc and then says "Completed without fingerprints". |
| `ScanEngineDialog` | Paths card (library root, database) and ten duplicate-scan tuning fields in two columns. **Run is disabled** without numpy/librosa (the tooltip says what to install). Its output database has no reader in the app yet. |
| `FuzzyDupeDialog` | Proof of concept that runs the real `near_duplicate_detector` over the fingerprints Find duplicates has cached; "Fingerprint filter" card (thresholds default to Settings) and the results card (verdict, distance and the pair; the selected pair's paths below). **Still not implemented, and disabled with a tooltip:** the "Metadata match" card (marked "Not implemented"), "Include pairs without fingerprints", "Skip fingerprint check", "Include non-matching FP pairs in review", and "Send Matches to Duplicate Pair Review". |
| `PairReviewDialog` | Side-by-side review of pairs from the Find duplicates plan (`Docs/duplicate_preview.json`), or else files with the same name in the **same folder** (reserved folders skipped). Each side is a glass card: rounded cover tile, title, format pill, a winner badge ("Winner (kept)" / "Would be quarantined" / "Winner not set"), the main tags and the elided path. Prev / Next / Switch / Prefer MP3 / Jump to. **"Quarantine inferior…"** confirms with both file names, moves the loser to `Quarantine/` under a unique name (never deleting, never overwriting), points playlist entries at the kept file, and logs to `Docs/duplicate_pair_review_log.txt`. "No — skip" skips; "Tag" appends the pair to `Docs/duplicate_pair_tags.txt`. Shortcuts: Backspace = skip, Left/Right = navigate. Enter no longer triggers the action. Refuses while another library write runs. |

The Opus / M4A testers are embedded now (§6.10.2; `MediaTesterDialog` remains as a
wrapper the app does not open), and `CrashLogDialog` is no longer opened: Activity shows
the same crash-log tail (`gui/workspaces/activity.py` reuses its `crash_log_paths()` /
`_tail()`).

### 7.6 ClusterPlaylistsDialog (`gui/dialogs/cluster_playlists_dialog.py`)

The Save cluster playlists preview, opened by the Sound Map's page action (§6.8) or the
Analyse results card's button, after the read-only summary job
(`cluster_playlist_export.summarize_cluster_run`). Modal; nothing is written from it.
Spec: `docs/specs/cluster_playlists.md`.

| Element | Purpose |
|---|---|
| Header | Method and parameter, number of clusters and tracks, when the analysis ran, and the destination (`Playlists/Clusters/<folder>/`). On a downsampled map: "The map shows 5,000 of 20,000 tracks; the playlists include all of them." |
| Warnings | A well with a warning glyph, shown only when there are any: the library was moved (paths mapped), tracks added since the analysis, "changed a lot; consider re-analysing", the map shows an older analysis, earlier sets kept in `Playlists/Clusters/`. |
| Order / Minimum tracks | In the table's glass card. *Closest to the centre first* (default) or *Library order*; minimum 2–500, default 5, counted in present tracks. |
| Table | Transparent, 34 px rows, no grid lines. One row per cluster (noise last): Save checkbox (pre-ticked: every cluster except those hidden in the map legend; locked, with the reason as tooltip, while too small or too long for Windows), the cluster's map colour as a round dot (an icon, not a style sheet), **editable name**, tracks present, missing (tooltip lists up to 20 paths), and the action: New, Replace or Skip · reason. "Unclustered (N) — not saved" has no checkbox. Two ticked rows with the same name (ignoring case) disable Save. |
| Left unchanged | Files already in the run folder that this save does not touch. |
| Buttons | Select all / Select none / Open folder as ghost buttons with glyphs (Open folder only when the run folder exists; `QDesktopServices`), Cancel (the default button) and **Save N playlists (K replace)** (primary, save glyph). |

Every option, checkbox and committed name edit re-plans on the GUI thread with the pure
`plan_cluster_playlists`. After Save the flow asks once if anything would be replaced
("K existing files … will be replaced (backed up first …)") and then writes exactly
`dialog.plan()`.

### 7.7 FileCleanupPreviewDialog (`gui/dialogs/file_cleanup_preview_dialog.py`)

The preview-and-confirm step of Organize → File Cleanup and of its undo. An opaque
window (the shell styles its chrome) whose contents use the glass parts:

| Element | Purpose |
|---|---|
| Title | "Review the renames" / "Restore the original names" |
| Stat tiles | to rename (or to restore) · skipped (collisions; warning tone when any) · files checked · kept (title) — the last two only for a cleanup, "kept" only when some suffixes are part of a title tag |
| Summary line | What happens (renamed in place, collisions skipped, nothing overwritten, reserved folders untouched; for an undo, which log it undoes) |
| Plan list | On a card: Folder, Current name, New/Original name, Action; renames first, then the skipped rows; 34 px glass rows, long values middle-elided with the full text in the tooltip |
| Note | Where the log goes and how playlists are updated (and backed up) |
| Rename N file(s) / Restore N name(s) | The one primary button; disabled when there is nothing to rename. **Cancel is the default button**, so Enter never renames. |

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

PlayerWorkspace ⇄ NowPlayingBar   (now-playing / position / state / volume / shuffle / repeat /
                                    transport_changed / transport_pinned one way;
                                    play-pause / next / prev / seek / volume / shuffle / repeat /
                                    keyboard mode the other)

MosaicLanding.tile_clicked ──► Startup.play_dir ──► AlphaDEXWindow.play_directory()
                              └─► Player page, begins playback
```
