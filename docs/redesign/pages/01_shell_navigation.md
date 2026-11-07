# Application shell, startup & navigation — page intent map
_Scope: `alpha_dex_gui.py`, `gui/main_window.py`, `gui/widgets/landing.py`, `gui/widgets/splash.py`,
`gui/widgets/sidebar.py`, `gui/widgets/top_bar.py`, `gui/widgets/log_drawer.py`, `gui/widgets/gradient_bg.py`,
`gui/dialogs/settings_drawer.py`, `gui/dialogs/crash_log_dialog.py`, `gui/qt_launcher.py`, `gui/compat.py`,
`gui/icons.py`, `gui/fonts/`, `gui/__init__.py`, `gui/workspaces/base.py`, `config.py` (shell-related parts).
Date: 2026-11-07._

All numbers below were measured on this container (PySide6 6.12, `QT_QPA_PLATFORM=offscreen`, fresh `HOME`).
Probe scripts are in `/tmp/claude-0/-home-user-Music-Indexer/4300de3c-51f6-599a-948e-c4bafb432735/scratchpad/shell/`
(`t_startup.py` drives the real `alpha_dex_gui.main()` with timers; the other `t_*.py` files are focused probes).

## Summary
The shell is the frame every other feature lives in. It has five jobs: (1) open the app with a branded landing
that also asks "which library?"; (2) hold one library path for the whole app and send it to every page when it
changes; (3) let the user move between ~13 pages (sidebar, Ctrl+1..9, and pages asking to hand off to each other);
(4) give every page a shared log, status line and "now playing" strip; (5) hold global settings (metadata
credentials, default library, the gradient toggle) and theme switching. The intent holds up. The plumbing has
serious lifetime problems. The app aborts (SIGABRT) if you close it while any background thread is running. The
saved library is applied **twice** at startup, and the first time is before the user has chosen anything. Several
settings and the Crash Log viewer have no effect. The global "keyboard mode" grabs arrow keys everywhere.

---

## Pages

### Startup orchestrator (`alpha_dex_gui.py`, function `main`)
- **Intent:** get from process start to a usable main window with the right library loaded, behind a smooth
  branded transition.
- **When the user comes here / entry points:** `python alpha_dex_gui.py`. Nothing else calls `main()`.
- **Primary flow (the real timeline, measured):**
  1. `t≈0`: the QApplication is created, fonts registered (`gui.fonts.load_fonts`), app icon painted
     (`gui.icons.make_app_icon`), and `ThemeManager.load_persisted()` runs. That call also **writes** the config
     (`apply` → `_persist` → `save_config`).
  2. `config.load_config()["library_root"]` is read as `saved_lib`. **It is not validated** (no `is_dir`).
  3. `MosaicLanding(shared_geo, saved_lib)` is built. If `saved_lib` is set, its `_ArtScanner` QThread starts
     at once, inside the constructor.
  4. `QTimer.singleShot(0, _construct_main_window)` is queued, then `landing.show_animated()` sets opacity 0 and
     starts a 320 ms fade-in animation.
  5. `app.exec()`. **The first event processed is the 0 ms timer.** `import gui.main_window` (~280 ms: pyqtgraph
     ~95 ms, numpy, the player module ~18 ms) plus `AlphaDEXWindow()` (~350 ms fresh) block the loop. Measured
     event-loop block in the full flow: **0.32–0.53 s, with the landing at opacity 0 the whole time**. The fade-in
     only starts after that (sampled 0.14 → 0.27 → 0.39 … in 16 ms steps). The "next tick so the landing can
     begin its fade-in" design does not do what it says: the window is built *before* the first fade frame.
  6. If `saved_lib` is a real directory, `AlphaDEXWindow.__init__` → `_load_persisted_library()` →
     `_on_library_changed()` **already sends the library to all 13 pages here**. The Player starts its library
     scan thread, `_StatsWorker` starts, and the config is saved, all while the landing is still on screen and
     before the user clicks anything (`t≈560 ms` in the trace).
  7. Fade-in ends, then a 600 ms "logo moment" pause, then the tile fly-in starts (`t≈1.5 s`): 26 tiles,
     38 ms stagger, 500 ms each with OutBack easing, so it ends around `t≈2.5–3 s`.
  8. The user clicks **Initialize** (if a library is saved) or **Choose Library Folder** (a non-native,
     unparented `QFileDialog.exec()`). `_accept(path)` stops the fly-in, snaps the tiles into place, and
     starts the scatter: random 0–160 ms pause, 360 ms each, InBack easing.
  9. The scatter group finishes → `_fade_out()` emits `library_selected(path)` at the **start** of a 420 ms
     fade-out → `_landing_to_main`. That runs `window.play_directory(_play_dir)` if a tile was clicked, shows
     the main window at opacity 0, and fades it to 1 over 420 ms (cross-dissolve).
  10. **Deferred "scan":** `QTimer.singleShot(min(30 s − elapsed, 3 s), window.set_library(path))`. About 3 s
      after the click, the **whole library-change chain runs a second time**, for the same path (trace: 6.4 s).
      The Player rescans, a second `_StatsWorker` starts, Duplicates/Genres/Indexer reset their state, and the
      config is saved again. The comment's rule "scan 30 s after the landing appears" is not implemented as a
      timer. It only caps the delay, so after 30 s on the landing the delay becomes 0 and the scan runs during
      the cross-fade.
  11. The landing's `_done()` stops the art scanner. It **blocks the GUI thread for up to 2 s**
      (`wait(2000)`) and then calls `QThread.terminate()`. Measured: a 2,006 ms freeze in the middle of the
      main window's fade-in when the scanner was still walking directories.
- **How it can misbehave (all exercised in `t_startup.py` modes):**
  | Scenario | What happens |
  |---|---|
  | Fast click before the window exists | Can't happen in normal runs: construction blocks the loop before any input is processed. The 10 ms re-poll in `_landing_to_main` only matters if construction **raised**. Then it polls forever: the landing fades out, **no window ever appears, and the process stays alive with no visible window** (confirmed, SHELL-10). |
  | Double-click Initialize | `_accept` runs twice. No double transition was observed: Qt stops the first scatter's property animations when the second group animates the same `pos`. Tiles snap back to the grid mid-scatter (a small visual jump). `library_selected` fired once. |
  | Tile click | Tiles only get a `dirpath` when the art scanner found a cover, which needs a saved library. A click emits `tile_clicked` (stored in `_play_dir`), then `_accept(saved)`. A second tile click mid-scatter overwrites `_play_dir` (last one wins). The Player starts playing before the window fades in. 3 s later the deferred `set_library` replaces the Player's status text ("Now playing: Album2" → "Scanning library…"); the queue is kept. |
  | No saved library | The button reads "Choose Library Folder". Tiles stay as gradient placeholders and can't be clicked. Cancelling the dialog leaves you on the landing. Fine. |
  | Invalid saved library (folder deleted or drive unplugged) | `_load_persisted_library` skips it (it has an `is_dir` check), but the landing still says **Initialize** and offers no way to pick another folder. Initialize → `set_library(nonexistent)` sends it to every page, the stats show "–", and the bad path is **saved again** (SHELL-08). |
  | Closing the landing (Alt+F4; the frameless window has no close button and no Esc) | The landing is the only visible window, so the app quits at once. No confirmation, and the main window is thrown away. If the art scanner is still running: **SIGABRT, "QThread: Destroyed while thread 'ArtScanner' is still running"** (exit 134, confirmed, SHELL-02). |
- **Reads / writes:** reads config `library_root` and `theme`. Writes config 2× at startup (theme persist from
  the landing pass and again from the window) plus once per library change. `~/.soundvault_art_history.json` is
  written from the scanner thread.
- **Backend dependencies:** none directly. Everything goes through the landing and the window.
- **State owned:** the closure variables `window` and `_play_dir`, plus `startup_clock`.
- **Async model:** timers only. No error path at all.
- **Motion & visuals:** the cross-fade is a `QVariantAnimation` on `windowOpacity` (deleted on finish). Window
  opacity is a compositor feature, so on Linux setups without a compositor both fades are instant.
- **Redesign notes:** drop the double library application. Pick one moment (after Initialize) and one call. Build
  the main window *after* the landing's first frame, for example after the fade-in's `finished`, or build pages
  lazily (see the cost table below). Let the landing offer "Choose a different folder" when a library is saved,
  and validate the saved path (show "Library not found" instead of Initialize). Handle construction failure (show
  an error dialog and quit).
- **Testability seams:** `main()` uses `QApplication.instance()`, so a test can create the app, pre-patch
  `gui.main_window.AlphaDEXWindow` / `gui.widgets.landing.MosaicLanding.__init__` to capture instances, schedule
  `QTimer.singleShot` actions, and call `app.quit()`. `t_startup.py` is a working template. `main()` calls
  `app.exec()`, so the harness needs a "run main with a script of timed actions" helper.

### Mosaic landing (`gui/widgets/landing.py`, classes `MosaicLanding`, `_CTACard`, `_Tile`, `_ArtScanner`, `_ArtHistory`)
- **Intent:** a branded first impression that also asks for, or confirms, the music library, decorated with the
  user's own album art.
- **When the user comes here / entry points:** only at startup. It is never shown again (no "switch library"
  screen reuses it).
- **Primary flow:** see the orchestrator above. One smart button: **Initialize** (saved path → `reuse_clicked` →
  `_accept(saved)`) or **Choose Library Folder** (`open_clicked` → `QFileDialog` → `_accept`). Clicking a tile
  (only with cover art) = "open the library and play this folder". Nothing is modified, so no preview or confirm
  is needed.
- **Reads / writes:** reads cover sidecars (`cover.jpg`, `folder.jpg`, …) and embedded art via
  `utils.audio_metadata_reader.read_metadata` for **every** audio file in each scanned folder (up to depth 7)
  until 26 unique covers are found. Reads and writes `~/.soundvault_art_history.json` (up to 10,000 paths, written
  on the scanner thread). Paints from `config.load_config()["bg_gradient_enabled"]` on **every paint**
  (89 config file reads during 3 s of animation, measured, SHELL-14). Signals out: `library_selected(str)`,
  `tile_clicked(str)`, `finished()`.
- **Backend dependencies:** `utils.audio_metadata_reader.read_metadata` on 6 `ThreadPoolExecutor` workers
  (`ArtScan*`) owned by the `_ArtScanner` QThread. QImage baking happens on the scanner thread and the QPixmap
  conversion on the GUI thread (`_on_art_found`). That split is correct.
- **State owned:** tiles, the pending path, animation groups, the scanner. All of it is dropped when the landing
  hides. The landing widget itself is never deleted (the closure keeps it alive).
- **Async model:** `_ArtScanner` has **no parent and no `closeEvent` cleanup** (SHELL-02). Interruption is
  checked between futures (1 s wait timeout), but **`_ordered_dirs()` (a full recursive `os.scandir` walk) is not
  interruptible**. On a large library that walk alone can take longer than the 2 s `wait()`, which then leads to
  `terminate()` (SHELL-03). `executor.shutdown(wait=False, cancel_futures=True)` leaves the running pool threads
  reading files after the landing is gone. `_ART_SCAN_DEBUG = True` prints a summary line to stderr on every
  launch.
- **Motion & visuals:** window fade-in 320 ms (OutCubic) → 600 ms pause → staggered fly-in → scatter → fade-out
  420 ms. All animations are parented to the landing and kept referenced. Nothing leaks in practice, but
  `_fly_in_grp` and `_fly_out_grp` are never deleted. Hard-coded colours: CTA card fill `rgba(10,13,20,218)`,
  title `#ffffff`, button text `#ffffff`, tile gradients `_GRADS`, fallback `#0d1117`. The card is dark glass in
  every theme. Only the accent bar, tagline colour and button background follow the theme.
- **Decorative / dead controls:** the module docstring describes "Open Library" and "Continue" buttons that don't
  exist. It also says the landing is "shown after the splash screen", but no splash is shown.
  `docs/gui_inventory.md` §0 says the landing "validates the library path". It doesn't.
- **Overlaps:** library selection duplicates TopBar "Change Library" (Ctrl+O) and Settings → General →
  "Default library". That is three places to choose the library, with three different validation behaviours.
- **Redesign notes:** the "logo moment" and the mosaic are the strongest visual identity in the app, so keep
  them. But (a) when a library is saved, offer a secondary "Choose another folder…" link; (b) show the library
  path and a validity state on the card; (c) give the frameless window a quit affordance (Esc, or an × on the
  card); (d) make the scanner fully cancellable and parented, with no `terminate()`; (e) cap per-folder tag
  reads. Reading every file's embedded art is expensive on network drives.
- **Testability seams:** stub `_ArtScanner` (or `_ArtScanner.run`), because it does disk I/O and writes
  `_ArtHistory._PATH`. **`_ArtHistory._PATH` is a class attribute computed at import from `~`**. Setting `HOME`
  after import does not redirect it, so the harness must `monkeypatch.setattr(landing._ArtHistory, "_PATH", tmp)`.
  Patch `QFileDialog.exec` (instance method; the dialog is unparented). Useful handles: `landing._cta`
  (find its `QPushButton`), `landing._tiles[i]._dirpath`, `landing._fly_in_grp`, `landing._scanner`,
  `landing._pending`.

### Main window (`gui/main_window.py`, class `AlphaDEXWindow`)
- **Intent:** a permanent frame (top bar, sidebar, page stack, now-playing bar, log drawer, status bar) that owns
  the library path and routes navigation, logging and playback hand-offs between pages.
- **When the user comes here / entry points:** after the landing. It is the only main window.
- **Primary flow:** pick a page in the sidebar → `_on_nav_changed(key)` → `QStackedWidget.setCurrentWidget`.
  No page transition animation.
- **Navigation model:**
  - Sidebar groups (`sidebar.NAV_STRUCTURE`): ORGANIZE (Indexer, Library Sync) · CLEAN UP (Duplicates,
    Similarity, Tag Fixer, Genre Normalizer) · PLAYLISTS (Playlists, Clustered, Music Graph) · PLAYER (Player,
    Compression) · TOOLS (Utilities, Help), then Exit pinned at the bottom.
  - `_WORKSPACE_MAP` order is identical to the sidebar's visual order, so **Ctrl+1..9 → Indexer … Music Graph
    match the sidebar**. But the two lists are kept in sync by hand (`_setup_shortcuts` uses
    `_WORKSPACE_MAP`'s key order, the sidebar uses `NAV_STRUCTURE`). Player (10th), Compression, Utilities and
    Help have no shortcut.
  - Other shortcuts: Ctrl+O (Change Library, wired to the private `TopBar._on_change_library`), Ctrl+, (Settings),
    Ctrl+L (toggle log), **Ctrl+W = clear log** (most platforms use Ctrl+W for "close window"). The Player page
    adds Space / N / P / Shift+←/→ as window shortcuts on itself.
  - `WorkspaceBase.navigate_requested(key)` → `_on_nav_changed`. Only Clustered → "graph" uses it. It carries no
    payload and the target is not told to refresh. `GraphWorkspace` has no `showEvent`, so after a new clustering
    run "Open graph" shows the *old* data until the user presses Reload (traced).
  - `WorkspaceBase.play_tracks_requested(list, str)` → `play_tracks()` → switches to Player and calls
    `load_tracks_and_play`. Only the Music Graph emits it. `play_directory()` is called only by the landing's
    tile hand-off.
  - The status bar shows `key.replace("_"," ").title()`, which gives "Graph", "Tools", "Genres". These don't match
    the sidebar labels ("Music Graph", "Utilities", "Genre Normalizer").
- **All 13 pages are built eagerly** in `_init_workspaces` with `library_path=""`. Measured cost per class,
  built one at a time (`t_construct.py`); none does file I/O at construction (`t_io.py`):

  | key | class | construct (ms) |
  |---|---|---|
  | indexer | IndexerWorkspace | 7.0 |
  | library_sync | LibrarySyncWorkspace | 28.9 |
  | duplicates | DuplicatesWorkspace | 12.3 |
  | similarity | SimilarityWorkspace | 4.9 |
  | tag_fixer | TagFixerWorkspace | 8.4 |
  | genres | GenresWorkspace | 24.2 |
  | playlists | PlaylistsWorkspace | 8.0 |
  | clustered | EnhancedClusteredWorkspace | 5.2 |
  | graph | GraphWorkspace | 13.0 (import pulls in pyqtgraph ≈ 95–107 ms) |
  | player | PlayerWorkspace | 26.8 (also `import vlc` attempt; import 18 ms) |
  | compression | CompressionWorkspace | 3.2 |
  | tools | ToolsWorkspace | 53.6 |
  | help | HelpWorkspace | 7.3 |

  The sum is ≈ 200 ms of Python, but `AlphaDEXWindow()` fresh = **354 ms** (cProfile: ~116 ms inside
  `QStackedWidget.addWidget` and ~100 ms in `QBoxLayout.addWidget`, i.e. style polish through `AlphaDEXStyle.polish`,
  3,247 calls). `import gui.main_window` = **281 ms**. Total ≈ 0.63 s blocked on the GUI thread. In one process,
  each extra window gets slower (781 → 1,633 → 2,976 ms) because `ThemeManager.load_persisted()` in `__init__`
  re-applies the app-wide style/palette/QSS to **every live widget**. That matters for the test harness (see
  below).
- **Library-change propagation:** `TopBar.library_changed`, `set_library()`, `_load_persisted_library()` and
  `_on_settings_saved()` all end in `_on_library_changed(path)`. That sets `_library_path`, calls
  `top_bar.set_library` (often twice: `set_library()` and `_load_persisted_library()` call it before
  `_on_library_changed`, which calls it again), runs `ws.set_library_path(path)` on all 13 pages synchronously on
  the GUI thread, logs "Library set to:", starts `_StatsWorker`, and does a load-modify-save of `library_root`
  (errors silently swallowed). **There is no "same path" short-circuit, no `is_dir` check (except in
  `_load_persisted_library`), and no check for running jobs.** What each page does with it:
  Indexer clears its log and looks for `Docs/MusicIndex.html`. Library Sync fills the "existing" field.
  Duplicates clears its tables and **sets `_worker = None` even while the worker is running**: the Scan button is
  re-enabled while the old scan keeps going and its signals stay connected (confirmed with a stubbed slow worker,
  `t_dupe_reset.py`). Genres clears the mapping editor and loads a saved mapping. Graph synchronously loads
  `Docs/cluster_info.json` (56 ms for a 20k-track 3.8 MB payload). Player clears its table and starts a
  `_LibraryScanner` thread. If a previous scan is still running it waits up to 1 s on the GUI thread, then
  replaces the reference. Compression fills its source field. Measured full fan-out: 81 ms of GUI-thread time
  (20k-track cluster payload).
- **Stats worker:** `_StatsWorker(QThread)` runs `Path(lib).rglob("*")`, counts audio by extension, sums
  `stat().st_size`, and counts "artists" as `p.parent.parent.name`. Problems:
  - It **redefines `finished = Signal(int,float,int)`**, hiding `QThread.finished` from Python name lookup.
    Qt's own `finished()` still fires at the C++ level (reachable via `QThread.finished.__get__`), but no one can
    hook `deleteLater` with the usual idiom. Each worker is `setParent(self)` and never deleted: one leaked
    QThread per library change (2 after a normal startup, confirmed).
  - If the library changes twice quickly, both workers run and both stay connected, so **whichever finishes last
    wins**. Setting A (slow) then B gave B's path with **A's 5 tracks** (confirmed, SHELL-05).
  - Closing the window mid-scan → **SIGABRT** (confirmed, SHELL-01).
  - Reserved folders (`Quarantine/`, `Not Sorted/`, `Trash/` …) are counted. A file at the library root counts
    the *parent* of the library as an artist. A single broken symlink or unreadable file raises inside the loop
    and the whole result becomes `(0,0,0)`, shown as "–" (confirmed, SHELL-06).
- **Reads / writes:** config `library_root` (read, written on every change), theme (via ThemeManager). Signals
  in: from TopBar, Sidebar, NowPlayingBar and every workspace's 4 base signals. The window reaches into Player
  internals (`_on_play_pause`, `_vol_slider`, `_preview_next`, `_restart_current`).
- **Backend dependencies:** none. It only calls `config`.
- **State owned:** `_library_path` (lives for the session), `_workspaces`, `_stats_worker` (only the latest),
  `_theme_picker` (only the latest), `_keyboard_mode_active`.
- **Async model / close behaviour:** **no `closeEvent`, no `aboutToQuit` handler, no "a job is running, quit
  anyway?" prompt.** Sidebar *Exit* → `self.close()` → last window closed → the app quits → Python tears down
  QThread wrappers that are still running → `qFatal` → **SIGABRT (exit 134)**. Confirmed with `_StatsWorker`
  (`t_startup.py closemain`) and a Duplicates scan (`t_dupe_reset.py`). Grep of every workspace: the only
  cleanup hook is `EnhancedClusteredWorkspace.closeEvent`, and **it is never called** (child widgets don't get
  `closeEvent` when the main window closes, confirmed `t_close.py`). Workers that are never stopped on close:
  `IndexerWorker`, `SyncScan/Build/ExecuteWorker`, `DupeBuild/ExecuteWorker`, `TagFix/TagFixApplyWorker`,
  `GenreWorker`, genres `_Scan/_Plan/_ApplyWorker`, `PlaylistWorker`, both `ClusterWorker`s, `MirrorWorker`,
  `SimilarityWorker`, `FileCleanupWorker`, `ArtistTitleWorker`, Player `_LibraryScanner` and 4× `_ArtLoader`,
  plus dialog workers. Quitting during an *execute* step (Indexer move, Duplicates quarantine/delete, Library
  Sync copy/move, Tag write) kills the process mid-operation, which can leave a half-applied plan and an
  incomplete `Docs/` log (suspected; follows directly from the confirmed abort).
- **Global event filter / keyboard mode:** `app.installEventFilter(self)` in `__init__`, so a Python
  `eventFilter` runs for **every event of every object in the app** for the whole session. When keyboard mode is
  on (toggled from the NowPlayingBar pill), all four arrow keys are taken **app-wide, whatever has focus**: a
  QLineEdit on the Indexer page (cursor didn't move; `kb_lib_left` fired), the modal Settings dialog, file
  dialogs, spin boxes, tables (confirmed `t_kb.py`, SHELL-11). Keyboard mode isn't turned off when you leave the
  Player page. Also observed: when a Python exception is pending (see SHELL-12), PySide reports the error again
  from inside this `eventFilter` ("Error calling Python override of QMainWindow::eventFilter()").
- **Motion & visuals:** none in the window itself (no page transition). The landing → window cross-fade is in
  `alpha_dex_gui.py`.
- **Decorative / dead controls:** `Sidebar.set_badge` exists but is never called. `_on_theme_changed` duplicates
  `WorkspaceBase._on_theme_changed_base`, so `refresh_shadows()` runs **twice per page per theme switch**. Theme
  switch with the full window = **1,023 ms freeze** (measured).
- **Overlaps:** status is shown in three places: the QStatusBar, the log drawer's status chip, and the log itself.
  Nav changes overwrite the status bar, and so do workspace status updates.
- **Redesign notes:** build pages lazily on first visit (Graph/pyqtgraph and Player/VLC are the heavy ones), or
  at least after the landing is visible. Make the library a single observable model with validation, a
  same-path no-op and a "busy" guard, so pages can refuse or confirm a library switch while they run a job. Add
  one central job registry so the shell can (a) show global busy state, (b) confirm quit, and (c) cancel and wait
  on close. Keyboard mode should be a Player-scoped shortcut set, not an app-wide filter.
- **Testability seams:** `_WORKSPACE_MAP` (pages can be swapped for stubs: patch the dict before constructing).
  `w._workspaces[key]`, `w._stack.currentWidget()`, `w._sidebar._buttons[key]`,
  `w._top_bar._stat_widgets`, `w._log_drawer._text`, `w._stats_worker`. objectNames: `contentFrame`.
  Must stub: `_StatsWorker` (disk walk), the Player's `_LibraryScanner` and `vlc` import, `SettingsDrawer.exec`,
  `QFileDialog.exec`, `gui.themes.picker.open_theme_picker`. Every test must **wait for or stop every QThread
  before teardown**, or the process aborts.

### Sidebar (`gui/widgets/sidebar.py`, classes `Sidebar`, `NavContainer`, data `NAV_STRUCTURE`)
- **Intent:** show the app's information architecture as task groups and switch pages.
- **Entry points:** always visible. `activate(key)` is called by the window for programmatic navigation.
- **Primary flow:** click → `AnimatedNavButton.clicked_key` → `_on_nav_click` → `activate` (pill animation) →
  `nav_changed` → window `_on_nav_changed` → `activate` again. The second call restarts the pill animation from
  the same start point (redundant but harmless).
- **Reads / writes:** theme tokens. `bg_gradient_enabled` from config on **every paint**.
- **State owned:** `_active_key`, button map.
- **Async model:** none. `resizeEvent` → `singleShot(0, _relayout_buttons)` → `singleShot(0, _snap_pill)`
  (one-tick deferrals, fine).
- **Motion & visuals:** a sliding accent "pill" (`QVariantAnimation`, 320 ms OutBack, so it overshoots) painted
  by `NavContainer`. Buttons are stretched to fill the viewport height (32–68 px each), so on tall screens the
  items get oddly tall. Custom-painted gradient that matches `GradientWidget`. **Hard-coded colours:** the logo
  label is `#f8fafc`, which is **invisible on all 6 light themes** (screenshot confirmed, SHELL-18). The separator
  is `rgba(255,255,255,0.08)`. Fixed width **360 px** (28% of the default 1300 px window; at the 900 px minimum
  width the page gets 540 px).
- **Decorative / dead controls:** `NavItem.icon` emojis render as system emoji (inconsistent per OS).
  `set_badge` is unused.
- **Overlaps:** the "AlphaDEX" brand appears in both the sidebar and the top bar, stacked directly on top of each
  other.
- **Redesign notes:** groups are sensible, but "Compression" doesn't belong under PLAYER, and "Similarity" overlaps
  with Duplicates. Consider a collapsible, narrower rail. Drop one of the two brand marks.
- **Testability seams:** objectNames `sidebar`, `sidebarSectionLabel`. `Sidebar._buttons[key]` and
  `clicked_key`. `NAV_STRUCTURE` can be asserted against `_WORKSPACE_MAP` (a cheap test that would catch shortcut
  drift).

### Top bar (`gui/widgets/top_bar.py`, class `TopBar`)
- **Intent:** show which library is active plus quick stats, and offer Change Library, Theme and Settings.
- **Entry points:** always visible. Ctrl+O → `_on_change_library`.
- **Primary flow:** Change Library → non-native `QFileDialog.exec()` (unparented, centred by hand) →
  `library_changed(path)`. **No validation** that the folder is a library, and choosing the *same* folder
  re-runs the full fan-out, which wipes Duplicates/Genres/Indexer state.
- **Reads / writes:** theme tokens. Signals out: `library_changed`, `settings_requested`, `theme_requested`.
- **State owned:** `_library_path`.
- **Motion & visuals:** `AnimatedButton` hover animations. Path label colour is picked in `_apply_theme` based on
  whether a library exists, but `_apply_theme` only runs at construction and on theme change, so after the first
  `set_library` the path stays in the "no library" secondary colour (traced, cosmetic).
- **Decorative / dead controls:** none. `_clear_stats` is O(n²) via `.index()` (harmless).
- **Overlaps:** library selection, as noted above.
- **Redesign notes:** make the library chip clickable (recent libraries, "Reveal in file manager", validity
  state). Stats need a "scanning…" state and should exclude reserved folders.
- **Testability seams:** objectNames `topBar`, `appTitle`, `libraryPath`, `statNumber`, `statUnit`,
  `mutedLabel`. Patch `QtWidgets.QFileDialog.exec`/`selectedFiles`.

### Activity log drawer (`gui/widgets/log_drawer.py`, class `LogDrawer`)
- **Intent:** one place where every page's progress, warnings and errors pile up.
- **Entry points:** handle click, Ctrl+L, auto-expand on `warn`/`error`. Ctrl+W clears.
- **Primary flow:** `WorkspaceBase.log_message` → window `_on_log_message` → `append` (timestamped and coloured,
  500-line cap).
- **State owned:** text (session only, never saved), expanded flag.
- **Motion & visuals:** a 200 ms `maximumHeight` animation on the body (0 ↔ 220 px). **Ignores the theme:**
  handle `#1e293b`, body `LOG_BG #0d1117`, text and chip colours all come from hard-coded constants in
  `gui/theme.py`. In light themes it stays a black band (screenshot confirmed). The status chip colour comes from
  each caller as a hex string.
- **Decorative / dead controls:** none, but messages logged **during page construction are lost** because the
  signals are connected after `cls(...)` returns. For example, the Player's "libVLC not found" warning never
  reaches the log (confirmed, SHELL-20).
- **Overlaps:** most pages also have their own embedded log `QPlainTextEdit` (Indexer, Duplicates …), so output
  is duplicated.
- **Redesign notes:** decide between one global log and per-page logs. Add filter by page and level, copy, and
  "save to Docs/".
- **Testability seams:** objectNames `logDrawer`, `logText`. `_expanded`, `_status_chip`.

### Workspace template (`gui/workspaces/base.py`, class `WorkspaceBase`)
- **Intent:** the shared scaffolding of every page: scrolling content, themed background, card helpers, and the
  4 signals pages use to talk to the shell.
- **Contract:** signals `log_message(str,str)`, `status_changed(str,str)`, `navigate_requested(str)`,
  `play_tracks_requested(list,str)`. `set_library_path(path)` → `_on_library_changed(path)` hook.
  `refresh_shadows()`. Helpers `_make_card` (adds a `QGraphicsDropShadowEffect` per card),
  `_make_section_title`, `_make_subtitle`, `_make_card_title`, `_make_primary_button`, `_make_browse_row`.
- **Missing from the contract (and needed):** no `is_busy()` / `cancel_and_wait()` / `shutdown()` hook, no
  "page shown" hook (`on_activated`), and no veto on library change. The docstring lists a `_make_run_button`
  that doesn't exist and describes `status_changed(str)` with the wrong arity.
- **Motion & visuals:** each card gets its own drop-shadow effect. `refresh_shadows` swaps every card's effect
  and is called twice per theme switch. `_make_card` cards can't also hold an opacity effect (one
  QGraphicsEffect per widget), which matters for any future fade-in-per-card motion.
- **Testability seams:** objectNames `workspaceCard`, `sectionTitle`, `sectionSubtitle`, `cardTitle`,
  `primaryBtn`. `content_layout`.

### Gradient background (`gui/widgets/gradient_bg.py`, `GradientWidget`, `paint_window_gradient`, `_gradient_enabled`)
- **Intent:** a subtle accent glow so the window reads as one canvas.
- **Reads:** `_gradient_enabled()` calls `config.load_config()` (opens and parses the JSON file) on **every paint
  of every GradientWidget, the sidebar and the landing**: 3 file reads per main-window repaint, 89 during 3 s of
  landing animation (confirmed, SHELL-14). The setting changes take effect live (on the next repaint) because of
  this, which is the only upside.
- **Redesign notes:** cache the flag in the theme manager and re-read it on `settings_saved`.

### Settings dialog (`gui/dialogs/settings_drawer.py`, class `SettingsDrawer`)
- **Intent:** global preferences: metadata-service credentials, default library, duplicate thresholds,
  appearance.
- **Entry points:** TopBar Settings, Ctrl+,. It is a **modal QDialog**, not a drawer, and doesn't slide (the
  docstring says "slides in from the right").
- **Every setting, whether it's saved, and who reads it:**

  | Control | Config key | Saved? | Read by (Qt app) |
  |---|---|---|---|
  | Service combo | `metadata_service` | yes | **nobody in the Qt app.** Only the legacy Tk `metadata_config_frame.py`. `plugins/acoustid_plugin.py` hard-codes "AcoustID". Selecting Last.fm hides both credential frames, and its Test says "Unknown service". |
  | API Key | `metadata_api_key` | yes | `plugins/acoustid_plugin.py` (live) |
  | MB App/Version/Contact | `musicbrainz_useragent` | yes | `plugins/acoustid_plugin.py` (live) |
  | Test Connection | – | – | AcoustID: only GETs `https://api.acoustid.org/v2/`, **never checks the key**. MusicBrainz: a real search. The worker thread reads `QLineEdit.text()` off the GUI thread. The result comes back via `invokeMethod` to the `@Slot` (works). A dead `_done` closure is left in the code. |
  | Default library | `library_root` | yes | the window's `_on_settings_saved` → `_on_library_changed` **without `is_dir`**. A nonexistent path is sent to all pages and the Player starts "Scanning library…" (confirmed). |
  | Near-duplicate threshold | `near_duplicate_threshold` | yes | **not the Duplicates page** (its spinboxes are hard-coded 0.1/0.02 and never read config). Only `duplicate_bucketing_poc.py` (Tools → Bucketing POC). |
  | Exact-duplicate threshold | `exact_duplicate_threshold` | yes | same as above (range capped at 0.1 here vs 1.0 on the Duplicates page) |
  | Background gradient | `bg_gradient_enabled` | yes | `gradient_bg._gradient_enabled` (live, read on every paint) |
  | "Advanced" tab | – | – | **empty placeholder tab** |
- **Async model:** a daemon `threading.Thread` for Test. If the dialog closes first, nothing breaks, because the
  dialog is never deleted (see leak).
- **Motion & visuals:** header `#0f172a` / `#f8fafc` / `#94a3b8` and status colours are hard-coded. The **selected
  tab's label is never drawn**: `gui/themes/style.py:537` uses `QFont.Weight.SemiBold`, which doesn't exist
  (PySide6 has `DemiBold`). Every selected-tab paint raises AttributeError (confirmed, SHELL-12). This hits every
  QTabWidget in the app, not just Settings.
- **Decorative / dead controls:** Service combo (in effect), both thresholds (for the main Duplicates flow), the
  Advanced tab.
- **Overlaps:** default library vs TopBar vs landing. Thresholds vs the Duplicates page's own spinboxes.
- **Leak:** a new `SettingsDrawer(self)` is created per open and never deleted (3 opens → 3 live instances,
  confirmed). The same pattern applies to `ThemePickerDialog` (non-modal, one per Theme click, traced).
- **Redesign notes:** split into "Accounts & services" (per-service cards with real key validation and a
  disabled-with-reason state), "Library" (default and recents, validated), "Appearance" (theme picker merged in
  here, gradient), and "Advanced (fingerprint thresholds)", and make the Duplicates page *read* them as defaults.
- **Testability seams:** construct it directly. `_service_combo`, `_api_key_entry`, `_mb_app/_mb_ver/_mb_contact`,
  `_library_entry`, `_near_dup_spin`, `_exact_dup_spin`, `_bg_gradient_cb`, `_test_btn`, `_test_status`,
  `_on_save()`. Stub `requests.get`/`musicbrainzngs` and `QFileDialog.getExistingDirectory`. Patch
  `SettingsDrawer.exec` when testing through the window.

### Crash Log viewer (`gui/dialogs/crash_log_dialog.py`, class `CrashLogDialog`)
- **Intent:** let the user see the last crash.
- **Entry points:** Utilities → "View Crash Log…" (`gui/workspaces/tools.py:1391`).
- **Reality:** it looks for `~/soundvault_crash.log` and `./soundvault_crash.log`. **Nothing in the Qt app writes
  a crash log.** `crash_logger.install()` (default file `crash.log`) is only called from
  `music_indexer_api.py`'s `__main__`. So the dialog always says "No crash log found." (traced, SHELL-21). Line 29
  computes an unused `log_path`. The Qt app also installs no `sys.excepthook`, so slot exceptions only reach
  stderr (invisible under `pythonw` on Windows).
- **Redesign notes:** install a crash logger in `alpha_dex_gui.main()` writing to a known path, or remove the menu
  item.

### Splash screen (`gui/widgets/splash.py`, class `SplashScreen`): dead code
- Never imported (grep: only references are comments in `landing.py` and docs). The landing's logo pause
  replaced it. Safe to delete or archive.

### Qt preview launcher (`gui/qt_launcher.py`, class `QtPreviewWindow`): legacy-only
- Nothing in the Qt app imports it. Its only caller is the **legacy Tkinter** app (`main_gui.py:11102`, a
  `subprocess.Popen([sys.executable, "-m", "gui.qt_launcher"])` "Qt Preview" action). It is a self-contained
  mock-up with dummy actions. It is dead for the Qt app; delete it together with that Tk menu item, or leave it
  for the legacy app.

### Support modules (`gui/compat.py`, `gui/icons.py`, `gui/fonts/`, `gui/__init__.py`)
- `compat.py`: PySide6 first, PyQt6 fallback. **It eagerly imports `QtWebEngineWidgets` and `QtWebChannel`,
  which no module under `gui/` uses.** That makes QtWebEngine a hard startup dependency (an install without the
  WebEngine add-on fails at the first `from gui.compat import …`) and costs 31 ms here (more on Windows). The
  PyQt6 fallback is untested and would break on `QtCore.Q_ARG`, `Signal` shadowing and similar.
- `icons.py`: a painted multi-resolution icon (hard-coded indigo/violet, fine for a brand mark).
- `fonts/loader.py`: registers Inter (variable and static) and JetBrains Mono once. `TypeScale` constants.
  Fine.
- `gui/__init__.py`: a one-line docstring ("optional Qt tooling"), out of date.

### Config (shell-relevant parts of `config.py`)
- `CONFIG_PATH` is computed **at import** from `~`. `load_config()` returns a defaults dict on *any* exception,
  including a truncated or corrupt JSON file. `save_config()` writes in place (not atomic, no temp + rename).
  Because the shell saves on every startup (ThemeManager `_persist`, twice) and on every library change, **a
  corrupt config is silently replaced with defaults on the next launch, wiping the AcoustID key, theme, MB
  user agent and so on** (confirmed: "SECRET" key gone after one library change, SHELL-13). A crash or power loss
  during one of those routine saves is exactly how the file gets truncated.

---

## Cross-page relationships
- **Library:** landing / TopBar / Settings → `AlphaDEXWindow._on_library_changed` → all 13
  `WorkspaceBase.set_library_path`. No veto, no busy check, and no same-path no-op. At startup it fires twice
  (construction and Initialize+3 s). Pages react very differently: some only prefill a field, and others wipe
  results or start background scans (Player, Graph, Genres, Indexer). Duplicates drops its running worker's
  reference (see `04_duplicates_similarity.md`). Player replaces a still-running scanner after a 1 s GUI-thread
  wait (see `07_player.md`).
- **Navigation hand-offs:** Clustered → Graph via `navigate_requested("graph")` (no refresh, so stale data).
  Music Graph → Player via `play_tracks_requested`. Landing tile → Player via `play_directory`. No other page
  uses the base signals. Everything else ("open the HTML preview", "go fix tags") is either a browser launch or
  missing.
- **Player ↔ shell:** NowPlayingBar is wired straight to Player privates. Keyboard mode is a Player feature
  implemented as an app-wide event filter in the window.
- **Theme:** ThemeManager singleton → `theme_changed` → TopBar, GradientWidget (each page), WorkspaceBase
  (shadows) and the window (shadows again). Sidebar, LogDrawer, landing and Settings have hard-coded colours that
  don't follow it.
- **Config:** read and written from many places with load-modify-save (shell, ThemeManager, Settings, pages).
  All of it happens on the GUI thread, so there is no race today, but nothing is atomic.

## Bugs found
| ID | Severity | Location | Trigger | Effect | Confidence | Suggested fix |
|---|---|---|---|---|---|---|
| SHELL-01 | crash | `gui/main_window.py:376` (+ no `closeEvent` anywhere in the class) | Close the window or click Exit while any QThread runs (stats scan, or any page's worker) | Process aborts: "QThread: Destroyed while thread is still running", exit 134. Mid-execute quits can leave a half-applied file plan. | confirmed-by-running | Add `closeEvent`: ask the user if jobs are running, then cancel and `wait()` every registered worker. Give `WorkspaceBase` a `shutdown()` hook. |
| SHELL-02 | crash | `gui/widgets/landing.py:1073` | Alt+F4 / close the landing while `_ArtScanner` runs | SIGABRT "Destroyed while thread 'ArtScanner' is still running" | confirmed-by-running | `closeEvent` on `MosaicLanding` that interrupts and waits. Parent the scanner. |
| SHELL-03 | stability | `gui/widgets/landing.py:1045-1053`, `552-588` | Initialize while the scanner is still walking a big library | GUI thread blocked 2 s (measured 2,006 ms) during the main-window fade-in, then `QThread.terminate()` on a Python thread (can deadlock or corrupt the interpreter). The directory walk is not interruptible. | confirmed-by-running (freeze); traced (terminate risk) | Check `isInterruptionRequested()` inside `_walk`. Disconnect and let the thread finish on its own (`finished` → `deleteLater`). Never call `terminate`. |
| SHELL-04 | stability | `gui/main_window.py:365-373` + `alpha_dex_gui.py:135-138` | Any launch with a valid saved library | Library fan-out runs during the landing (Player scan, stats, config write) and again ~3 s after Initialize. Pages are reset a second time (Duplicates re-enables Scan while its worker runs) and the deferred-scan design is defeated. | confirmed-by-running | Remove `_load_persisted_library` from `__init__` (the landing owns the choice). Make `_on_library_changed` a no-op for an unchanged path. |
| SHELL-05 | wrong-result | `gui/main_window.py:376-387` | Change library A → B while A's stats scan is still running | Top bar shows B's path with A's track/GB/artist counts | confirmed-by-running | Tag results with the path or a generation counter and ignore stale ones. Cancel the old worker. |
| SHELL-06 | wrong-result | `gui/main_window.py:400-412` | Library containing Quarantine/Not Sorted/Trash, root-level files, or one broken symlink | Counts include reserved folders. "Artists" counts the library's parent and quarantine subfolders. One `stat()` error zeroes all stats ("–"). | confirmed-by-running | Skip reserved folders, try/except per file, use a better artist heuristic or a cache. |
| SHELL-07 | stability | `gui/main_window.py:391`, `380` | Every library change | `finished` redefined, so the QThread `finished` idiom can't be used. Worker is parented and never deleted (one leaked QThread per change). | confirmed-by-running | Rename to `stats_ready`. Connect `QThread.finished` → `deleteLater`. |
| SHELL-08 | UX | `alpha_dex_gui.py:79-83`, `gui/widgets/landing.py:266-273` | Saved library folder no longer exists | Landing shows "Initialize" with no warning and no way to choose another folder. The bad path is pushed to every page and saved again. | confirmed-by-running | Validate on the landing. Show a "not found" state plus "Choose folder". |
| SHELL-09 | UX | `gui/main_window.py:296-305`, `gui/dialogs/settings_drawer.py:213` | Type a nonexistent "Default library" in Settings and Save | Path immediately becomes the active library for all pages. Player starts "Scanning library…". | confirmed-by-running | Validate in the dialog. Treat "default library" separately from "switch now". |
| SHELL-10 | stability | `alpha_dex_gui.py:98-104`, `121-126` | Any exception in `AlphaDEXWindow()` (e.g. a page's `__init__` raises) | Landing fades out. No window is ever shown. The process keeps running invisibly, re-polling every 10 ms. | confirmed-by-running | try/except around construction, show `QMessageBox.critical`, then quit. Cap the poll. |
| SHELL-11 | UX | `gui/main_window.py:163-207` | Turn keyboard mode on, then type in any text field or dialog | Arrow keys hijacked app-wide (Indexer QLineEdit, modal Settings). Mode persists after leaving Player. A Python `eventFilter` sees every app event. | confirmed-by-running | Scope to the Player page (shortcuts with `WidgetWithChildrenShortcut`), or filter only when Player is current and focus isn't a text input. Remove the app-level filter. |
| SHELL-12 | cosmetic | `gui/themes/style.py:537` | Painting any selected tab (Settings and every QTabWidget) | `AttributeError: Weight has no attribute 'SemiBold'` on each paint. Selected tab label not drawn. Error cascades into later Python overrides. | confirmed-by-running | `QFont.Weight.DemiBold`. (Themes area: report there too.) |
| SHELL-13 | data-loss | `config.py:425-456` | Config file truncated or corrupt (e.g. crash during one of the routine startup saves) | `load_config` silently returns defaults. The next save (every startup, every library change) overwrites the file, wiping API key, theme, MB UA etc. | confirmed-by-running | Atomic save (temp + `os.replace`). On a decode error, back up the bad file and warn. Don't save on startup when nothing changed. |
| SHELL-14 | stability (perf) | `gui/widgets/gradient_bg.py:144-149` | Any repaint (landing animations, sidebar, every page) | Config JSON re-read and re-parsed per paint (89 reads in 3 s of landing animation, 3 per window repaint) | confirmed-by-running | Cache the flag. Invalidate on `settings_saved`. |
| SHELL-15 | UX (motion) | `alpha_dex_gui.py:98-104` | Every launch | Main-window import + build (~0.6 s) runs before the first landing frame. The window sits at opacity 0 for 0.3–0.5 s, which defeats the "construct on next tick" intent. | confirmed-by-running | Construct after the fade-in finishes, or lazily build pages. |
| SHELL-16 | UX | `gui/dialogs/settings_drawer.py:56-156` | Using Settings | Service combo, both duplicate thresholds and the "Advanced" tab have no effect in the Qt app. Last.fm shows no key fields. AcoustID "Test" never validates the key. | traced-in-code (+ Last.fm test confirmed) | Wire the thresholds as Duplicates defaults, remove or disable the service combo, real key test, drop the empty tab. |
| SHELL-17 | stability (leak) | `gui/main_window.py:289-293`; `gui/themes/picker.py:452-458` | Opening Settings or Theme repeatedly | A new dialog per open, parented to the window, never deleted (3 opens → 3 live SettingsDrawers). Multiple Theme pickers can be open at once. | confirmed-by-running (Settings); traced (picker) | `WA_DeleteOnClose`/`deleteLater`. Reuse a single picker instance. |
| SHELL-18 | cosmetic | `gui/widgets/sidebar.py:139-142`, `gui/widgets/log_drawer.py:36-55`, `gui/dialogs/settings_drawer.py:27-38`, `gui/widgets/landing.py:244-246,311` | Switch to any light theme | Sidebar "AlphaDEX" logo invisible (white on light). Log drawer stays a black band. Settings header stays navy. Landing card always dark. | confirmed-by-running (screenshot) | Use theme tokens and object names styled by the theme QSS. |
| SHELL-19 | stability | `gui/workspaces/clustered_enhanced.py:189` | Closing the main window during clustering | `closeEvent` cleanup on a child page never runs, so the worker is never stopped (feeds SHELL-01) | confirmed-by-running | Move cleanup into the shell-driven `shutdown()` hook. |
| SHELL-20 | UX | `gui/main_window.py:130-137` | Any page that logs in `__init__` (Player: "libVLC not found") | Message lost: signals are connected after construction | confirmed-by-running | Connect before building, or queue early messages in `WorkspaceBase`. |
| SHELL-21 | UX | `gui/dialogs/crash_log_dialog.py:28-43` | Utilities → View Crash Log | Always "No crash log found": the Qt app never installs a crash logger and the file name doesn't match `crash_logger`'s default | traced-in-code | Call `crash_logger.install(<known path>)` in `main()` and read that path. |
| SHELL-22 | stability (perf) | `gui/main_window.py:314-317`, `gui/workspaces/base.py:394-397` | Theme switch | ~1.0 s UI freeze (measured 1,023 ms). `refresh_shadows` runs twice per page. | confirmed-by-running (timing); traced (double call) | Remove the window-level duplicate. Consider not re-setting the app style for colour-only changes. |
| SHELL-23 | stability | `gui/dialogs/settings_drawer.py:230-243` | Test Connection with MusicBrainz selected | Worker thread calls `QLineEdit.text()` off the GUI thread | traced-in-code | Read the values on the GUI thread before starting the thread. |
| SHELL-24 | UX | `gui/compat.py:22-23` | Install without QtWebEngine add-ons | App can't start, although nothing in `gui/` uses WebEngine/WebChannel | traced-in-code | Import lazily, or drop them. |
| SHELL-25 | wrong-result | `gui/workspaces/graph.py:32-39` (no refresh on show) via `gui/main_window.py:250-256` | Clustered → "Open graph" after a new clustering run | Graph shows the previous data until the user clicks Reload | traced-in-code | Have `navigate_requested` carry intent, or add an `on_activated` hook that reloads if the file is newer. |
| SHELL-26 | cosmetic | `gui/widgets/landing.py:57` | Every launch with a saved library | `[ArtScan] …` debug line printed to stderr | confirmed-by-running | Default `False` / use `logging`. |
| SHELL-27 | UX | `gui/main_window.py:219-247` | Keyboard users | Ctrl+1..9 covers only the first nine pages (no Player/Compression/Utilities/Help). Ctrl+W clears the log instead of the platform's close convention. Shortcut and sidebar order are kept in sync by hand. | traced-in-code | Derive both from one nav model. Pick conventional bindings. |

### Redesign insights (shell)
1. **The library is the app's central state but has no model.** It has three entry points with three
   validation rules, a fan-out with no same-path no-op and no busy guard, and pages reacting inconsistently (some
   wipe results, some scan). A `LibraryModel` (path, validity, stats, `changing`/`changed` signals with veto)
   would fix SHELL-04/05/08/09 and make "switch library" a deliberate, confirmable action.
2. **The app needs a job model.** There is no shared notion of "a page is running something": no global busy
   indicator, no quit confirmation, no cleanup on close, and no navigate-away policy. This is the root of the
   only confirmed hard crashes in the shell (SHELL-01/02/19).
3. **Startup does too much, too early, twice.** Build the frame after the first landing frame. Build pages
   lazily (Graph and Player carry the heavy imports). Apply the library once, after Initialize.
4. **Settings promises more than it delivers.** Of 8 controls, 4 do nothing in the Qt app. Merge the theme
   picker in, make thresholds real defaults for the Duplicates page, and turn credentials into per-service cards
   with honest states.
5. **Theme coverage stops at the shell.** The parts every user sees all the time (sidebar logo, log drawer,
   landing card, settings header) are hard-coded dark. A redesign should treat these as first-class themed
   surfaces and drop per-paint config reads.

### Harness needs (shell)
1. **Isolated paths:** `config.CONFIG_PATH` and `landing._ArtHistory._PATH` are computed at import. Setting `HOME`
   isn't enough once the modules are imported, so monkeypatch both attributes to `tmp_path`.
2. **A thread-teardown fixture:** after every test, find all `QThread` instances (window children plus
   `_workspaces[*]._worker`-style attributes plus `landing._scanner`), interrupt or cancel them, and `wait()`.
   Otherwise pytest aborts with exit 134. Provide stubs for `_StatsWorker.run`, `_ArtScanner.run`, the Player's
   `_LibraryScanner` and `vlc`.
3. **Modal killers:** patch `QFileDialog.exec`/`selectedFiles`/`getExistingDirectory`, `SettingsDrawer.exec`,
   `QMessageBox.*`, and `open_theme_picker`. The landing and TopBar use the *instance* `QFileDialog.exec()`, not
   the static helpers.
4. **Window lifecycle:** each `AlphaDEXWindow()` re-applies the theme to every live widget (construction time
   grew 0.78 → 1.6 → 3.0 s across three windows in one process). The fixture must `deleteLater()` and flush
   `DeferredDelete` (`QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)`) between tests, or offer a
   session-scoped window with a reset helper. Allow swapping `_WORKSPACE_MAP` entries for lightweight stubs.
5. **A "drive main()" helper:** run `alpha_dex_gui.main()` with a scripted list of `(ms, action)` timers and an
   auto-quit, capturing the landing and window instances (pattern in `t_startup.py`). Plus an event-loop-gap
   sampler for motion regressions.
