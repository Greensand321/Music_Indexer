# AlphaDEX GUI redesign: groundwork

_Started 2026-11-07. This folder holds the groundwork for redesigning the Qt app (`alpha_dex_gui.py`)._

The goal of the redesign is an app that is cleaner, looks far better, moves well, and is stable. Before anything
is redrawn, this phase did three things:

1. **Mapped the intent of every page** and how the pages fit together. The per-area reports are in
   [`pages/`](pages/), and the unified map is below.
2. **Built test harnesses** so the redesign has a safety net. See [`tests/ui/README.md`](../../tests/ui/README.md)
   and the Harness section below.
3. **Fixed the bugs** found on the way, crashes and data-loss first. See Bug status.

| Report | Area |
|---|---|
| [01_shell_navigation.md](pages/01_shell_navigation.md) | Startup, landing, main window, sidebar, top bar, log drawer, Settings, config |
| [02_design_system_motion.md](pages/02_design_system_motion.md) | Tokens, themes, style, typography, every animation, theme switching |
| [03_indexer_library_sync.md](pages/03_indexer_library_sync.md) | Indexer, Library Sync |
| [04_duplicates_similarity.md](pages/04_duplicates_similarity.md) | Duplicate Finder, Similarity, and the four diagnostic duplicate dialogs |
| [05_tag_fixer_genres.md](pages/05_tag_fixer_genres.md) | Tag Fixer, Genre Normalizer (Fill / Normalize) |
| [06_playlists_clustering_graph.md](pages/06_playlists_clustering_graph.md) | Playlists, Clustered, clustering wizard, quality report, Music Graph |
| [07_player.md](pages/07_player.md) | Player, Now Playing bar, keyboard mode |
| [08_tools_compression_help.md](pages/08_tools_compression_help.md) | Utilities (11 sub-tools), Compression, Help |

---

## 1. What the app is for: the user's jobs

AlphaDEX serves one person with one big music library. Every page is really in service of one of six jobs:

| Job | The user wants to… | Pages today |
|---|---|---|
| **A. Pick & understand my library** | point the app at a folder and see its state | Landing, Top bar (path, stats), Settings → Default library |
| **B. Organize it** | get a clean, consistent folder/file layout, and merge new music in without making a mess | Indexer, Library Sync, Utilities → File Cleanup, Utilities → Validator |
| **C. Clean it up** | remove duplicates and fix wrong or missing tags and genres | Duplicates, Similarity, Pair Review*, Fuzzy Finder*, Scan Engine*, Bucketing POC*, Tag Fixer, Genre Normalizer (Fill, Normalize) |
| **D. Listen & discover** | play music, audition tracks, find tracks that sound alike, make playlists | Player, Now Playing bar, Playlists, Clustered (+wizard, quality report), Music Graph |
| **E. Take it with me** | produce derived copies and lists | Compression (FLAC→Opus mirror), Utilities → Artist/Title export, Codec export |
| **F. Get help / trust the app** | know what happened, what went wrong, and how to use it | Activity Log drawer, status bar, Crash Log*, Help, Utilities → Opus/M4A testers* |

\* = reached only through the Utilities page ("Diagnostics") or a dialog.

### 1.1 Page-by-page intent

The status column shows what each page is in practice today: **core** (a real user workflow), **support**
(useful secondary page), **diagnostic** (developer tool or proof of concept), or **dead** (nothing reaches it).
The redesign column is a *proposal* (see the open decisions in §6).

| Page / surface | Intent (one line) | Status | Redesign proposal |
|---|---|---|---|
| Landing (`landing.py`) | A branded first impression that confirms or chooses the library; clicking a tile plays that album | core | Keep the moment, but make it short and interruptible; a missing library must offer "Choose another folder" |
| Main window shell | Permanent frame: nav, page stack, transport, log | core | Keep; add a global **Activity** area driven by the JobRegistry |
| Sidebar | Show the information architecture as task groups and switch pages | core | Regroup around the six jobs (§5) |
| Top bar | Active library, quick stats, Change Library / Theme / Settings | core | Becomes the library "header"; merge Theme into Settings |
| Log drawer | One place for every page's progress, warnings and errors | support | Becomes the Activity center: running jobs and history |
| Settings | Service credentials, default library, thresholds, appearance | support | Only show settings that something reads (4 of 8 controls read nothing today) |
| Indexer | Normalize folder structure and names, with a reviewable preview before anything moves | core | Shared **Plan review** component (with Library Sync and Duplicates) |
| Library Sync | Merge an incoming folder: add new, skip dupes, accept upgrades, with per-track overrides | core | The match table *is* the plan; bulk flags; show the matched existing track beside each incoming one |
| Duplicates | Fingerprint, group same-recording files, pick winners, preview, then quarantine losers | core | One page in four stages: Scan → Review → Preview → Execute |
| Similarity | Explain why two files do or don't match | support | Fold into Duplicates as "Explain match" |
| Pair Review | Side-by-side A/B decisions on duplicate pairs | diagnostic | Its compare UI moves into Duplicates → Review; drop its hard delete |
| Fuzzy Finder / Scan Engine / Bucketing POC | Prototypes of alternative duplicate strategies | diagnostic | Leave the user UI (Bucketing is already inside the real planner) |
| Tag Fixer | Look tracks up online; accept or reject corrections field by field | core | A **Metadata** page with one review grid, shared with genres |
| Genre Normalizer: Fill | Give genre-less tracks MusicBrainz community genres | core | A "genre-only lookup" mode of the Metadata page |
| Genre Normalizer: Normalize | Collapse genre spellings to a chosen vocabulary, preview-first | core | Keep; it is the model flow for the whole app |
| Playlists | Rule-based `.m3u` creation (folder / tempo-energy / Auto-DJ) and playlist repair | core | Keep separate; add preview, "send to Player", and a shared save sheet |
| Clustered (+ wizard, quality report) | Analyse audio, group similar tracks, judge the grouping | core | Merge with Graph into one **Sound Map** page; quality shown inline; wizard down to ~8 settings |
| Music Graph | Explore the clustered library as a map; lasso a region; play it or save it | core | The canvas of the Sound Map page |
| Player | Browse the library, play tracks, build a queue, audition, load/save M3U | core | Library and queue browser; the bar is the only transport; one "now playing + up next" model |
| Now Playing bar | Transport and now-playing info on every page; hosts Keyboard Mode | core | The single transport; keyboard mode scoped to the Player |
| Compression | A portable Opus copy of the library; originals untouched | support | Move under "Take it with me" with the exports; needs cancel, an errors-first result view, and reserved-folder skipping |
| Utilities | A grab-bag: exports, File Cleanup, Validator, Diagnostics launcher | support | Split up: exports → Export; File Cleanup → Organize (plan-based); Diagnostics → hidden developer menu |
| Help | Orientation: docs, shortcuts, about, issues | support | Link the maintained `docs/` suite; add per-page "?" links |
| Crash Log viewer | See the last crash | support | Activity center → "Problems" |
| Splash (`splash.py`), `clustered.py`, `interactive_3d_scatter.py`, `qt_launcher.py`, `audio_preview.py` | none in the Qt app (dead, or used only by the legacy Tk app) | dead | Delete once the owner agrees |

---

## 2. How the pages connect

```
                 ┌──────────── library path broadcast (set_library_path → all 13 pages) ────────────┐
                 │                                                                                   │
 Landing ──► Main window ◄── Top bar / Settings (3 separate ways to choose a library, 3 validations)│
   │ tile click                                                                                      │
   ▼                                                                                                 ▼
 Player ◄── play_tracks_requested ── Music Graph ◄── Docs/cluster_info.json (file, no signal) ── Clustered
   ▲                                                                                                 │
   └── Now Playing bar (relays to Player privates)                       navigate_requested("graph") ┘

 Indexer ──► moves files; regenerates Playlists/; runs update_playlists
 Library Sync ──► vendored copy of the indexer engine on the incoming folder; .m3u8 transfer list
 Duplicates ──► Quarantine/ (losers), Docs/duplicate_preview.html
     └─✗ Pair Review expects Docs/duplicate_preview.json (only the Tk app wrote it)
 Tag Fixer / Fill / Normalize ──► three different genre writers that undo each other's output
 File Cleanup / Pair Review / Playlists Repair ──► the same update_playlists backend (silently a no-op, TOOL-03)
```

**Shared buses with no owner:**

- **`Docs/`** is the shared output and cache bus (reports, logs, `.soundvault.db`, `.duplicate_fingerprints.db`,
  `cluster_info.json`, Library Sync backups).
- **The library folder itself.** Before this phase, several scans treated `Quarantine/`, `Trash/` and `Docs/` as
  music. The worst case: the Indexer moved quarantined duplicate losers back into the library, silently undoing
  the Duplicates workflow.
- **Config**, read and written with load-modify-save from many places (now atomic; see Bug status).

**Duplication the redesign should remove:**

| Concern | Copies today |
|---|---|
| Finding duplicates | 6 (Duplicates, Similarity, Pair Review, Fuzzy Finder, Scan Engine, Bucketing POC) |
| Fingerprint caches over the same files | 4 (`.duplicate_fingerprints.db`, `.soundvault.db`, `.duplicate_bucketing_poc_fps.db`, `duplicate_scan.db`) |
| Full library rescans at startup | 3 (top-bar stats, Player library, Indexer), and the Player's ran twice |
| Genre writers | 3, with incompatible formats (joined string vs. list) |
| Playlist repair | 3, all on one broken backend |
| Transport controls | 2 full sets (Player page and Now Playing bar) |
| Ways to choose a library | 3 (landing, top bar, Settings), each validating differently |

---

## 3. Systemic problems (what the ~205 bugs have in common)

1. **No job model.** Every confirmed crash in the shell came from worker threads. Workers shadowed
   `QThread.finished`, emitted it from inside `run()`, and the slot dropped the last reference to a still-running
   thread. The result was `QThread: Destroyed while thread is still running`, a hard abort, possibly mid
   file-move. Nothing stopped workers on quit, re-run or library change, and nothing prevented two pages from
   writing the same files at once.
   *Fixed at the core:* `gui/workers.py` JobRegistry + `WorkspaceBase._start_worker` + an ask/cancel/wait
   `closeEvent`. Every page was then moved onto it.
2. **What executes is not what was previewed.** This happened in the Indexer, Library Sync (copy flags became
   moves), Tag Fixer (unticked fields were written) and Genres (Apply wrote the *previous* library). The
   preview-first rule needs a single plan object that is persisted, shown, executed verbatim, and invalidated
   when its inputs change.
3. **Reserved folders were not respected** by the Indexer, Library Sync, Player, Tag Fixer, Genres, the mirror
   and the exports. There is now one definition, `utils/reserved_folders.py`.
4. **Destructive actions without the gates.**
   - Pair Review: one keystroke ran `os.remove`.
   - File Cleanup: renamed "Carbon Copy" → "Carbon" with no preview.
   - Playlist repair: rewrote every `.m3u` in place.
   - The "Delete permanently" option actually quarantined.
   - Genre tools silently dropped every genre after the first.
5. **About a third of the controls on several pages do nothing** (decorative steppers, thresholds, options the
   backend ignores). The redesign gets simpler just by not redrawing them.
6. **The design system exists but is bypassed.**
   - 237 hard-coded hex colours and 168 inline stylesheets outside `gui/themes/`.
   - 11 of 43 tokens unused.
   - An inverted type ramp.
   - Card shadows that never rendered.
   - Muted text below WCAG contrast in all 14 themes.
   - A theme switch that froze the UI for ~1.1 s (measured: ~70 dropped frames).
7. **Motion has no system.** There were ~20 hand-written animation sites with 7 easings and 15 durations. Some
   animated geometry that a layout also controls, some were unparented, and page switches were instant.
8. **The library is the central state but has no model.** The change broadcast had no veto, no same-path check
   and no busy check, and the saved library was applied twice at startup.

---

## 4. Design system & motion: the foundation the redesign needs

From [02_design_system_motion.md](pages/02_design_system_motion.md), refined by what the fixes established:

- **Tokens:** keep the colour token model (it is sound) and extend it with spacing, radius, type, elevation and
  **motion** groups. Pages must not carry colour literals: they ask for a role ("muted text", "danger surface"),
  and status colours become semantic states, not Midnight hex values.
- **Type:** one px-based ramp (display / title / heading / body / caption / mono) that widgets opt into, replacing
  today's three overlapping sources (app font, style font, QSS px sizes).
- **Elevation:** painted shadows, not `QGraphicsDropShadowEffect`. A widget has only one effect slot, which fades
  need, and blur re-renders on every child repaint.
- **Theme switching:** a fixed stylesheet that refers to palette roles, changing only palette and tokens on a
  switch. *Done in this phase:* ~1.1 s → ~60–80 ms (about 4 dropped frames), verified by a test that compares a
  live-switched window against one started in the target theme, widget by widget. The redesign's target is
  under one frame.
- **Motion:** `gui/themes/motion.py` (added in this phase) provides duration and easing tokens, a reduced-motion
  switch, and parented, re-entrancy-safe helpers. Rules for the redesign:
  - never animate geometry a layout owns (animate opacity or transforms, or a painter offset);
  - every animation has a parent and a single owner;
  - a fade always ends at opacity 1.0 or hidden;
  - page transitions get a short cross-fade or slide from the tokens;
  - nothing blocks the GUI thread for more than one frame during motion (measured by `FrameGapMeter`).

### Shared components the redesign should build once

| Component | Replaces |
|---|---|
| **Plan review** (table of planned actions with filters, per-row override, bulk actions, conflict flags) | Indexer HTML-only preview, Library Sync match table, Duplicates group list, Tag/Genre review grids |
| **Confirm sheet** (exactly what will change, the counts, where the log goes, a destructive-action variant) | Ad-hoc `QMessageBox.question` calls |
| **Job card** (progress, phase, cancel, result, open report), fed by the JobRegistry | Per-page progress bars and status chips |
| **Page header** (title, one-line intent, step indicator that really advances, page-level "?" link) | Section titles and decorative steppers |
| **Empty / error / missing-dependency states** (fpcalc, VLC, librosa, network, API key) | One log line per file, or silence |
| **Save playlist sheet** | Playlists, Graph and Player each saving `.m3u` their own way |

---

## 5. Proposed information architecture

A recommendation for the owner to decide on (§6), grouped by the jobs in §1:

```
LIBRARY      Home (library health: counts of untagged, duplicate candidates, Manual Review, last runs)
ORGANIZE     Indexer · Library Sync · File Cleanup
CLEAN UP     Duplicates (Scan → Review → Preview → Execute; "Explain match" built in)
             Metadata (Lookup · Fill genres · Normalize genres; one review grid)
LISTEN       Player · Sound Map (analyse + map + select → play/save) · Playlists
TAKE AWAY    Mirror (Opus) · Exports (artist/title, codec list)
             ─────────
             Activity (jobs + log + problems) · Settings (incl. appearance) · Help
```

That is 13 sidebar entries becoming about 11, each with one clear job, and no developer prototypes in the user's
path.

---

## 6. Open decisions for the owner

1. **IA:** adopt §5 (merge Clustered + Graph into Sound Map; fold Similarity and Pair Review into Duplicates; one
   Metadata page)?
2. **Permanent delete:** Duplicates now really deletes when asked, behind a confirm listing every deletion and a
   typed `DELETE` (the executor's existing double gate); quarantine is the default. Keep that in the redesign,
   or make quarantine the only disposition?
3. **Prototype dialogs** (Fuzzy Finder, Scan Engine, Bucketing POC): delete, or keep behind a developer menu?
4. **Dead files:** OK to delete `gui/widgets/splash.py`, `gui/workspaces/clustered.py`,
   `gui/widgets/interactive_3d_scatter.py`, the stray repo-root files (`player_106bd0e.py`, `player_diff.txt`,
   `player_full_diff.txt`, `diff_player*.txt`, `crash_test.txt`, `crash_report.txt`, `last_path.txt`, the
   committed `soundvault_crash.log*`), and the stray `=0.13.0` file (a pip-redirect accident)?
5. **Visual direction:** stay close to the current glassy dark look with better hierarchy, or a new direction?
   The design system work supports either.
6. **Startup:** keep the mosaic landing as a first-run and "change library" experience, but go straight to the
   main window when the saved library is valid?

---

## 7. Harness: how to use it during the redesign

Full guide: [`tests/ui/README.md`](../../tests/ui/README.md). The essentials:

- `pytest` runs the whole suite headless (GUI included). CI runs it on every push (`.github/workflows/tests.yml`),
  and the Linux job uploads a screenshot contact sheet as an artifact.
- **Before/after visuals:** `python scripts/gui_snapshots.py --themes all`
  renders every page in every theme to `snapshots/index.html`.
- **Guards that run on every GUI test:**
  - unplanned modal dialogs fail (they don't hang);
  - threads left running fail;
  - cross-thread, painting and bad-animation Qt warnings fail;
  - exceptions inside slots or paint fail;
  - config is isolated.
- **Ratchets** (`tests/test_gui_hygiene_ratchets.py`): hex colours, inline stylesheets, `font-size` literals,
  direct `QThread.start()` calls, unparented animations. Counts may only go down. Lower the ceiling whenever a
  redesign step removes some.
- **Qt API check:** every Qt name in `gui/` must exist in PySide6. This catches Qt5-era names and typos
  without running the code path.
- **Motion and responsiveness:** `MotionRecorder` (what animated, how long, which easing, orphans, stuck opacity)
  and `FrameGapMeter` (freeze length and dropped frames). Tighten the ceilings in
  `tests/ui/test_ui_responsiveness.py` as the redesign lands.
- **Real data:** `tests/library_factory.py` builds real tagged MP3/FLAC/M4A/OGG/Opus libraries with ffmpeg and
  mutagen, so flows can be tested end to end (preview → confirm → execute → assert the filesystem).

---

## 8. Bug status

205 bugs were recorded across the eight reports: 16 crash, 13 data-loss, 58 wrong-result, 26 stability, 63 UX,
29 cosmetic. The per-area status after the fix phase is in the table below; each report's bug table remains the
detailed record.

### Recorded bugs, by area

| Area | Recorded | Fixed | Partial / not fixed |
|---|---|---|---|
| Shell, startup & navigation (`SHELL-`) | 27 | 26 | SHELL-27: Ctrl+1..9 now follow the sidebar, but there are still no shortcuts beyond the 9th page and Ctrl+W still clears the log |
| Design system & motion (`DS-`) | 26 | 26 | none |
| Indexer & Library Sync (`LIB-`) | 24 | 24 | LIB-12 / LIB-17: controls the engine ignores are disabled with an explanatory tooltip, not implemented |
| Duplicates & Similarity (`DUP-`) | 21 | 21 | Proof-of-concept dialogs fixed so they don't crash or lie; folding them in is a redesign decision |
| Tag Fixer & Genres (`META-`) | 25 | 25 | META-23: every write is logged with before/after values; an in-app *revert* is not built yet |
| Playlists, clustering & graph (`PL-`) | 32 | 32 | PL-15: cluster playlists are honestly marked "not available" rather than built; PL-23: hard-coded colours remain in files that weren't otherwise rewritten |
| Player (`PLY-`) | 27 | 27 | none (the direct-play vs queue model is a redesign decision) |
| Tools, Compression & Help (`TOOL-`) | 23 | 23 | none |
| **Total** | **205** | **204** | 1 partial (SHELL-27), plus the noted limitations |

### Found and fixed beyond the recorded 205

The fixers, the integration work and three independent reviews (library data safety, file/tag/playlist writers,
GUI concurrency) turned up further real bugs. The serious ones:

- **Data loss:**
  - Duplicate quarantine let same-named losers overwrite each other (3 files became 1, and the run still reported
    success).
  - The duplicate planner would write `YEAR=Unknown` into FLAC files.
  - Playlist auto-update after a rename emptied non-UTF-8 playlists and scrambled the order of all others.
  - Tag Fixer dropped every artist/title/album value after the first.
  - In Library Sync Move mode, a failed Replace stranded the incoming file under a temp name.
  - Failed copies left truncated files under the real track name.
- **Wrong result:**
  - Library Sync wrote every path it created in lowercase on Windows.
  - Library Sync had no "library changed since plan" check.
  - The Indexer swept Opus files into `Trash/`, and re-running it added `(1)` to every file.
  - The Indexer rewrote artist tags it never previewed.
  - "Top 3 genres by popularity" was actually alphabetical.
  - Low-confidence matches contributed genres.
  - M4A cover art was never read, so the Player, landing and track details showed none.
  - Library Sync's fingerprint matching only matched byte-identical files (being ported to the vendored engine;
    see below).
- **Stability:**
  - The idle fingerprint-cache writer spun at 100 % CPU and starved the GUI (window build went from 0.4 s to 45 s).
  - A live theme switch left 71 widgets on the old palette (invisible sidebar headers on light themes).
  - Nothing stopped two pages from changing library files at once; there is now a library-write lock across
    Indexer, Library Sync, Duplicates, File Cleanup, the tag tools and playlist writes.
  - Closing could hang forever on a job that ignored cancel, and jobs started after closing began slipped past
    the confirmation.

### Hygiene ratchets (gui/)

| Metric | Before | After |
|---|---|---|
| Hard-coded hex colours outside tokens | 237 | 116 |
| Inline `setStyleSheet` calls | 168 | 94 |
| `font-size` literals outside the theme | 81 | 38 |
| Direct `QThread.start()` (bypassing the JobRegistry) | 28 | **0** (now zero tolerance) |
| Unparented animations | 7 | 2 (both in the Tk-only `qt_launcher.py`) |
| Lambdas connected to the theme singleton | 3 | 0 (zero tolerance) |

### Tests

185 tests at the start of this phase (9 failing; the result depended on file order). At the end of phase 2:
**904 passing**, including 374 headless GUI tests (backend verified in default, reversed and shuffled order).

