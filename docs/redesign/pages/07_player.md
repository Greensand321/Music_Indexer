# Playback (Player workspace, Now Playing bar, audio preview) — page intent map
_Scope: `gui/workspaces/player.py` (PlayerWorkspace + `_LibraryScanner`, `_ArtLoader`, `_BgArtWidget`, `NowPlayingCovers`, `_TrackHoverPopup`), `gui/widgets/now_playing_bar.py` (NowPlayingBar), `gui/audio_preview.py` (VlcPreviewPlayer), the player wiring in `gui/main_window.py` (signal relays, keyboard mode + app-wide `eventFilter`, `play_directory` / `play_tracks`), the landing-tile hand-off in `alpha_dex_gui.py`, and the stray repo-root files `player_106bd0e.py`, `diff_player.txt`, `diff_player_prev.txt`, `player_diff.txt`, `player_full_diff.txt`. Date: 2026-11-07._

How this was checked: by reading the code, and by running offscreen probes (`QT_QPA_PLATFORM=offscreen`, `HOME` pointed at a fresh temp dir) against a generated 6-track library. The probes used (a) no VLC at all, which is how this environment is set up, (b) a fake `vlc` module that records calls, and (c) a `vlc` module that raises at import, the way real python-vlc does when libVLC is missing. Probe scripts are in `/tmp/claude-0/-home-user-Music-Indexer/4300de3c-51f6-599a-948e-c4bafb432735/scratchpad/player/`.

## Summary
This area lets you listen to your library without leaving AlphaDEX. The **Player workspace** is a full jukebox: a searchable table of the whole library, a play queue, recently played tracks, prev/current/next cover art, shuffle and repeat, M3U import/export, and a "30s Preview" audit mode that jumps to 0:30 and moves on after 15 s. The **Now Playing bar** is a slim transport strip that stays visible in every workspace once something plays. It also holds a **Keyboard Mode** toggle that turns the arrow keys into transport keys across the whole app. Tracks reach the Player from three places: its own library table, a landing-page album tile, and a Music Graph lasso selection.

The original intent (still visible in the 235-line first Qt version, `player_106bd0e.py`) was an *audition tool* "used for quick validation during duplicate review and similarity inspection". Today no Duplicates or Similarity screen can send tracks to the Player. The only playback backend is libVLC through python-vlc. With it missing the page still acts as if it is playing, and with python-vlc installed but libVLC missing the whole main window fails to build. `gui/audio_preview.py` is used only by the legacy Tkinter app.

## R2 (Liquid Glass, 2026-11-08): the bar is the only transport

The audit below describes the page before the redesign; every bug in its table is fixed (see
`docs/redesign/README.md`). The R2 restyle then changed the page's shape:

- **No transport on the page.** The seek bar, ⏮ ▶ ⏭ ■, shuffle, repeat and volume are gone from
  the Player page. The Now Playing capsule is the transport everywhere:
  - shuffle and repeat are glyph toggles beside prev/next on the capsule (repeat still cycles
    Off → Track → All; Track shows a small "1", an on-state also gets an accent dot);
  - **Stop was dropped.** Pause stops the sound, Previous after 3 s restarts the track, and Clear
    (queue card) unloads everything; modern players have no Stop, and the capsule has no room for
    a sixth button at the minimum width. `_on_stop()` still exists and is what the end of the
    queue, Clear and a failed track use;
  - the capsule mirrors what the player can do (`transport_changed(available, can_play, reason)`):
    without a backend its controls are disabled and the reason is in their tooltips.
- **The Player page pins the capsule** (`transport_pinned`): on that page it shows even before
  anything has played, idle ("Nothing playing" or "Playback unavailable"), so the page is never
  without a transport. Elsewhere it still appears only once something plays (R1's rule: an empty
  transport floating over another page would cover content and offer nothing to do).
- **Width tiers** keep the capsule inside its 880 px / window − 40 width: *full* (inline volume),
  *compact* (volume in a popup behind the speaker), *narrow* (the times move into the seek bar's
  tooltip, gaps tighten). Every control stays reachable in every tier; at the 900 px minimum
  window the capsule is ~620 px wide and uses the narrow tier.
- **Library card**: the table is a transparent item view on a glass card (34 px rows, muted 12 px
  header, tabular # and time, no grid). The playing track's row is tinted with the accent and its
  # cell shows a play (or pause) glyph. The card itself carries the blurred cover of the playing
  track (the old `_BgArtWidget`, which painted an opaque block, is now `_LibraryCard`). Search is a
  glass pill with the search glyph; Reload and Open file are glyph buttons; the status line and
  the "30 s preview" toggle sit in the card's footer. Narrow tables hide Album, then # (both stay
  searchable).
- **Side column**: the cover carousel is glass album tiles (radius 20, painted shadow, inset ring;
  a faint well with a music glyph when there is no art; neighbours only when the queue has them),
  shown only when the column has room for it *and* a usable queue. The queue and Recently played
  are glass rows; the queue card's header holds glyph buttons for add selected, load M3U, save
  M3U and clear.
- **Playback unavailable** is a quiet notice row under the page's purpose line (warning glyph,
  the reason), plus the idle capsule.
- Tracks without embedded art now use the folder's cover (`cover.jpg`, `folder.jpg`, …, the same
  names as the landing) in the capsule, the carousel and the card.
- New public API: `volume()`, `set_shuffle()`, `set_repeat()`, `cycle_repeat()`,
  `transport_state()`, `is_playing()`; signals `shuffle_changed`, `repeat_changed`,
  `transport_changed`, `transport_pinned`. The main window wires them in its existing
  Player ↔ capsule block.

## Pages

### Player workspace (`gui/workspaces/player.py`, class `PlayerWorkspace`)
- **Intent:** Browse the library and listen to it: play single tracks, build and play a queue, quickly audit tracks with 30-second previews, and save or load M3U playlists.
- **When the user comes here / entry points:**
  - Sidebar key `player`. It is the 10th entry in `_WORKSPACE_MAP`, so **it has no Ctrl+N shortcut** (only the first 9 get one).
  - `AlphaDEXWindow.play_tracks(paths, label)`: any workspace's `play_tracks_requested`. Today only the Music Graph emits it (`graph.py:289` "Play selection", `graph.py:294` play one).
  - `AlphaDEXWindow.play_directory(dirpath)`: a landing-page album tile (`alpha_dex_gui.py:143`), called *before* the deferred `window.set_library(p)`.
  - Both entry points switch to the Player and call `load_tracks_and_play`, which **replaces** the queue.
  - Workspace-scoped QShortcuts: `Space` play/pause, `N` / `P` next/prev, `Shift+Right` / `Shift+Left` ±30 s. These only fire while the Player is visible, and typing in the search box is not stolen (both confirmed).
- **Primary flow:**
  1. Library set (persisted at window construction, or via the top bar). `_on_library_changed` clears the table and starts `_LibraryScanner` right away, even if the user never opens the Player.
  2. Rows stream into the table in adaptive batches (50–500). The search box filters on Title/Artist/Album.
  3. Double-click a row, or right-click → "Play now": `_play_directly(path)`. This plays outside the queue (`_direct_row`) and leaves the queue alone.
  4. "+ Add Selected" or right-click "Add to queue" → `_add_paths_to_queue`. Double-click a queue row → `_play_current()`.
  5. Transport: Prev / Play-Pause / Next / Stop / Shuffle / Repeat (Off → Track → All), seek slider, volume slider. *(R2: all on the capsule; Stop dropped.)*
  6. When a track ends (`_tick_position` polls VLC state every 400 ms): a direct-played track hands over to the queue; otherwise the queue advances according to repeat and shuffle.
  7. M3U: "Load M3U" replaces the queue and plays. "Save Queue as M3U…" and right-click "Save selection as M3U…" write a `#EXTM3U` file of absolute paths.
  - Nothing here changes audio files, so the preview-first rule doesn't apply to playback. The only file writes are the M3U save (QFileDialog acts as the confirmation) and a `Playlists/` folder created *before* that dialog (PLY-19).
- **Reads / writes:**
  - Reads the library tree via `os.walk`, including the reserved folders `Quarantine/`, `Trash/`, `Not Sorted/` and `Playlists/` (PLY-22).
  - Reads tags and cover art via `utils.audio_metadata_reader` and mutagen.
  - Writes only `.m3u` files (default folder `<library>/Playlists`). No config keys: volume, repeat, shuffle, 30s-preview, recently played and the queue are all lost on restart.
  - Signals out: `now_playing_changed(str, str, str, object)`, `playback_state_changed(bool)`, `position_changed(int, int)`, and `log_message` from the base class.
  - Called by the main window: `_on_play_pause`, `play_next`, `play_prev`, `seek_to_ms`, `set_volume`, `set_keyboard_mode`, `kb_target`, `kb_lib_*`, `_preview_next`, `_restart_current`.
  - The main window also reaches into `player_ws._vol_slider` directly (`main_window.py:156-159`). *(Fixed earlier: it uses `volume_changed`; R2 removed the page slider.)*
- **Backend dependencies:**
  - `python-vlc`: `vlc.Instance()` and `media_player_new()`, all on the GUI thread, all calls synchronous.
  - `utils.audio_metadata_reader.read_metadata_from_mutagen` / `read_metadata`: on the scanner thread for the table, but **on the GUI thread** for every queued or handed-over track (`_load_row` in `_play_directly`, `_play_current`'s callers, `load_tracks_and_play`, `_add_paths_to_queue`) and for the hover popup (`show_for` reads tags *and* cover synchronously).
  - The Player doesn't use `playlist_generator` / `playlist_engine`.
- **State owned:**
  - `_queue` (list of row dicts), `_queue_index`, `_direct_row`, `_shuffle`, `_shuffle_history`, `_shuffle_next_idx`, `_repeat`.
  - `_recent_plays` (30, in memory), `_all_rows` (scan cache, cleared on library change), `_lib_kb_row`, `_kb_target` ("library" or "queue", set by a mouse press on either viewport), `_keyboard_mode`.
  - On library change the table is rescanned, but the queue, now-playing track and recently played list are kept, so the old library's tracks keep playing. `_lib_kb_row` goes stale (harmless).
- **Async model:**
  - `_LibraryScanner(QThread)` has no parent. It checks `isInterruptionRequested()` between files, and its `finally` *always* emits the last batch plus `scan_complete`.
  - A rescan calls `requestInterruption()` and `wait(1000)` **on the GUI thread**, then replaces the Python reference. The old thread's queued signals are not disconnected (PLY-11); if the wait times out, the thread object is destroyed while still running (PLY-12).
  - `_ArtLoader(QThread)`: up to 4 live at once (current, selected, prev, next). `_cancel_loader` calls `requestInterruption()`, which `run()` never checks, then `wait(2000)` on the GUI thread, so each track change can block the GUI for up to 3 × 2 s on slow storage.
  - The selected-row loader is replaced **without** waiting, so the app aborts (PLY-12).
  - There is no `closeEvent` or cleanup. Closing during a scan or art load aborts the process with `QThread: Destroyed while thread is still running` (exit code 134, PLY-13).
  - Timers: `_pos_timer` (400 ms, runs while "playing", whether or not the page is visible; with no backend it runs forever) and `_highlight_timer` (single-shot 15 s, **not paused by Pause**, PLY-06). Plus a 250 ms volume-apply single-shot and 300 ms preview-seek single-shots, none of which are cancellable.
  - There is no busy-guard on transport buttons; each click restarts playback synchronously.
- **Motion & visuals:**
  - `_BgArtWidget` cross-fades a blurred cover (3.5 s `QPropertyAnimation` on a custom `crossfade` property) behind the transparent library table. It paints hard-coded `QColor(13,17,23)` plus a 70 % `QColor(10,13,20)` vignette.
  - A new cover arriving mid-fade drops the half-faded image and the art "pops".
  - `NowPlayingCovers`: custom-painted prev/current/next covers with hard-coded placeholder colours.
  - `_TrackHoverPopup`: a parentless `Qt.Popup | Frameless` window with hard-coded dark QSS.
  - Inline styles that ignore the theme:
    - library-table QSS with dark `rgba(13,17,23,210)` items
    - separator `#30363d`
    - grey `#64748b` / `#94a3b8` labels
    - `#6366f1` scan-progress chunk and queue highlight
  - Under the light **Pearl** theme the library table is unreadable: dark palette text (`#111827`) on near-black rows (PLY-16, confirmed by screenshot).
- **Decorative / dead controls:**
  - `_on_table_single_click` starts an `_ArtLoader` on every single click, and its result handler `_on_sel_art_ready` is `pass` (player.py:1714-1717). It does no useful work and can crash the app (PLY-12).
  - The hover popup effectively never appears: the viewport has no mouse tracking, so hover `MouseMove` events never reach the filter (PLY-15).
  - `_TrackHoverPopup.schedule()` / its internal `_timer` are never used.
  - `PlayerWorkspace.keyPressEvent` (player.py:1924-1946) is unreachable, because the app-wide filter in `main_window.py` eats the arrow keys first.
  - `_scan_progress` value updates only run when `maximum() > 0`, which never happens because the range is always (0, 0).
  - `toggle_shuffle()` / `toggle_repeat()` have no callers.
  - The `_vlc_status_lbl` "VLC: checking…" text is replaced synchronously during construction, so nobody ever sees it.
- **Overlaps:**
  - M3U creation duplicates the Playlists workspace and the Graph's own M3U export.
  - "Open folder" (`_open_in_explorer`) duplicates Graph's `_on_reveal`.
  - The Player's transport duplicates the Now Playing bar (two seek sliders, two volume sliders, two play buttons on screen at once). *(R2: resolved, the bar is the only transport.)*
  - The library scan repeats `_StatsWorker`'s rglob in `main_window.py` and the Indexer's walk; nothing is shared or cached.
- **Redesign notes:**
  - **Direct play vs queue are two hidden modes.** Prev and Next only ever act on the queue, so after a library double-click Next jumps to some unrelated queue item, or does nothing. Pick one model ("Play now" = insert at queue head and play; "Up next") and show it.
  - **Two transports on one screen.** On the Player page the bar and the page's transport row both show seek, volume and play. Let the bar *be* the transport everywhere and have the page show the library and queue only.
  - "30s Preview" (checkbox), Keyboard Mode ↓ (one-shot preview) and the legacy audit intent are three versions of one feature: **audition mode**. Make it an explicit mode that works on a chosen list (library selection, a duplicate group, a graph lasso), with its own "next" source.
    - Neither the Qt nor the legacy Tkinter app contains a "most active segment" picker. Both use a fixed 30 s offset, despite commit 66d7bc3's backlog note.
  - Show backend status in the bar (no VLC means a disabled transport and an explanation), not only in a small grey label on the Player page.
  - The library table should come from a shared library index/model (a `QAbstractTableModel` with numeric sort roles) rather than a third private rescan with 5 `QTableWidgetItem`s per track.
  - Decide whether reserved folders (`Quarantine/`, `Trash/`) belong in a "library" view. They are dup losers and junk.
  - The original purpose, auditioning duplicates and similarity results, has no path today. Add `play_tracks_requested` emitters in Duplicates and Similarity, or an inline audition control there.
- **Testability seams:**
  - **VLC:** inject a fake `vlc` module (`sys.modules["vlc"]`) before constructing the workspace. The workspace only uses `Instance().media_new / media_player_new`, `MediaPlayer.set_media / play / pause / stop / is_playing / get_length / get_time / set_time / set_position / get_state / audio_set_volume` and `vlc.State.Ended`. A working 90-line fake is at `/tmp/claude-0/-home-user-Music-Indexer/4300de3c-51f6-599a-948e-c4bafb432735/scratchpad/player/fakevlc/vlc.py`.
  - **Better:** add a `PlaybackBackend` seam (constructor arg or factory) so tests needn't patch imports.
  - **QtMultimedia works offscreen here:** `QMediaPlayer` with the FFmpeg backend decodes and advances `position()` with "No audio device detected". It is a viable fallback backend and a realistic test backend.
  - **Modals:** `QFileDialog.getOpenFileName` (Browse…, Load M3U), `QFileDialog.getSaveFileName` (Save M3U), `QMessageBox.information/critical` (M3U), `QMenu.exec` (both context menus; drive the lambdas via `_play_directly`, `_add_paths_to_queue`, `_jump_queue`, `_remove_from_queue` instead).
  - **Threads:** monkeypatch `gui.workspaces.player._load_row` (the scanner and every queue add use it) and `utils.audio_metadata_reader.read_metadata` (art) to make them slow or instant. Wait for `ws._scan_finished` *and* `not ws._lib_scanner.isRunning()`.
    - At teardown, wait on `_lib_scanner`, `_art_loader`, `_prev_art_loader`, `_next_art_loader` and `_sel_art_loader`, or the test process aborts.
  - **Timing constants:** `player._HIGHLIGHT_DURATION_MS` / `_HIGHLIGHT_START_MS` are read at call time, so they can be monkeypatched to tens of ms.
  - **Handles** *(R2: `transportBar` and the page's transport attributes are gone; objectNames `playerLibraryCard`, `playerLibraryTable`, `playerSearch`, `playerQueueCard`, `playerQueueList`, `playerRecentList`, `playerNowPlayingCard`, `playerNotice`, `playerSide`; drive transport through the capsule, see `tests/ui/player_helpers.make_player_with_bar`)*: no objectNames except `transportBar` and `trackHoverPopup`. Tests have to use attributes: `_lib_table`, `_search`, `_queue_list`, `_recent_list`, `_play_btn`, `_prev_btn`, `_next_btn`, `_stop_btn`, `_shuffle_btn`, `_repeat_btn`, `_highlight_chk`, `_seek_slider`, `_vol_slider`, `_np_title`, `_np_artist`, `_status_lbl`, `_vlc_status_lbl`, `_covers`, `_bg_art`, `_hover_popup`, `_pos_timer`, `_highlight_timer`. The redesign should add objectNames.

### Now Playing bar (`gui/widgets/now_playing_bar.py`, class `NowPlayingBar`, objectName `nowPlayingBar`)
- **Intent:** Keep transport (prev/play/next, seek, volume) and "what's playing" in view from every workspace, and host the Keyboard Mode toggle.
- **When the user comes here / entry points:**
  - It lives in `AlphaDEXWindow`'s content frame between the workspace stack and the log drawer.
  - It is hidden until the first `update_now_playing`, then **never hidden again**: Stop and Clear queue leave the old title and art (PLY-27).
- **Primary flow:** Player emits now-playing/state/position → bar updates. The user presses transport → bar emits `play_pause_requested`, `next_requested`, `prev_requested`, `seek_requested(ms)` (on slider release only), `volume_changed(int)` (on `sliderMoved` / release only) or `keyboard_mode_toggled(bool)`. The main window relays these to the Player.
- **Reads / writes:**
  - Reads only theme tokens (`sidebar_bg`, `card_border`) in `paintEvent`. No config.
  - Slots: `update_now_playing(title, artist, album, pm)`, `set_playing(bool)`, `update_position(pos, dur)`, `set_volume(int)`.
- **Backend dependencies:** none. It is a pure view.
- **State owned:** `_is_playing`, `_duration_ms`, `_user_seeking` (stops position updates from fighting a drag), `_kb_anim`.
- **Async model:** none. Everything happens on the GUI thread through queued/direct signals.
- **Motion & visuals:**
  - `_animate_kb_press`: a 120 ms geometry "pulse" on a button that sits in a layout. Restarting it mid-pulse captures the shrunken geometry as the end value; a 1 px drift was confirmed after a double toggle (PLY-24).
  - `paintEvent` is theme-aware, but the inline QSS isn't: art placeholder `#1e2130`, grey labels `#64748b`, and the Keyboard Mode pill's hard-coded indigo `#6366f1/#818cf8`, which ignores the theme accent.
  - Transport glyphs are emoji/Unicode text (⏮ ▶ ⏸ ⏭ 🔊 ⌨).
- **Decorative / dead controls:**
  - Clicks on the seek-slider groove, the mouse wheel and arrow keys move the knob but emit nothing. Seek snaps back; volume visibly changes without changing loudness (PLY-09).
  - The play button's tooltip says "[Space]", but Space only works while the Player page is visible.
- **Overlaps:** a full duplicate of the Player page's transport row. *(R2: resolved.)*
- **Redesign notes:**
  - Make the bar the single transport. *(R2: done.)*
  - Add a "no backend" state. *(R2: done: disabled controls, the reason in tooltips, "Playback unavailable".)*
  - Show the keyboard-mode state somewhere other than a tiny pill.
  - Its tooltip only describes the *queue* meaning of the arrows, while the default target is the *library*, where ↓/→/← mean something else.
  - Stop/Clear should collapse or reset the bar.
- **Testability seams:** construct it standalone. Drive it with `update_now_playing` / `update_position` and read `_title_lbl`, `_artist_lbl`, `_art_lbl.pixmap()`, `_play_btn.text()`, `_seek_slider`, `_vol_slider`. `_kb_btn` has objectName `kbModeBtn`. Spy on the signals with `QSignalSpy` / `qtbot.waitSignal`.

### Keyboard Mode (cross-cutting: `NowPlayingBar._kb_btn` → `AlphaDEXWindow.set_keyboard_mode_global` / `eventFilter`, `main_window.py:163-215`)
- **Intent:** Hands-on-keyboard auditioning: ↓ previews the next track, → next, ← previous, ↑ restart.
- **Entry points:** the ⌨ pill in the Now Playing bar.
- **Primary flow:** toggle on → `_keyboard_mode_active = True` → **every** `KeyPress` in the application reaches `AlphaDEXWindow.eventFilter`. Arrow keys are consumed and sent to the library (`kb_lib_down/right/left/up`) or the queue (`_preview_next`, `play_next`, `play_prev`, `_restart_current`), depending on which list the user last *clicked*.
- **Reads / writes:** `PlayerWorkspace._kb_target`, `_lib_kb_row`.
- **Async model:** synchronous. Key auto-repeat fires one synchronous `_load_row` plus art-loader cycle per repeat.
- **Decorative / dead:** `PlayerWorkspace.keyPressEvent` duplicates this logic and is unreachable.
- **Redesign notes:**
  - The filter is installed on `QApplication` at construction and stays there, so it runs Python for every event in the app even when keyboard mode is off.
  - While keyboard mode is on, **arrow keys stop working everywhere**: line edits and spin boxes on other workspaces, dialogs (confirmed with a `QDialog`'s line edit), item views, combo popups. They silently change the track instead, even when you're in Tag Fixer (PLY-10).
  - Scope it to when the Player or bar has focus, or skip text inputs, popups and modal dialogs. Use a visible on-screen mode indicator.
- **Testability seams:** toggle `bar._kb_btn.setChecked(True)`, then `QTest.keyClick(<any widget>, Qt.Key_Left)` and check the fake VLC calls and the focused widget's state.

### `VlcPreviewPlayer` (`gui/audio_preview.py`)
- **Intent:** A clip-preview backend (start offset + duration, muted start then unmute) for the **legacy Tkinter app only** (`main_gui.py:71`). The Qt app never imports it.
- **Notes:**
  - It is better guarded than the Qt player: it catches `Exception` at import and init, and exposes `available` / `availability_error`.
  - Its own threads: a `threading.Timer` stop and an unmute poller thread.
  - VLC event callbacks (`_handle_end`) run on libVLC's thread and call `on_done` there. Fine for Tk via `after`, unsafe if it were ever reused with Qt widgets.
  - It lives in `gui/`, the Qt package, which is misleading. Move it next to `main_gui.py`, or reuse its import guard pattern in the Qt backend.
- **Testability seams:** has `PlaybackError`; with `VLC_AVAILABLE=False` it raises cleanly.

### Stray repo-root player files (not imported by anything)
Every one is a **UTF-16-LE, CRLF** file, the signature of a PowerShell `>` redirect, so `grep` and Python can't read them as-is (I decoded them with `iconv -f UTF-16`). Nothing in the repo imports or references them except `docs/reorganization_spec.md`, which already lists them for deletion. `pytest.ini`'s `testpaths = tests` means pytest never collects them. `player_106bd0e.py` isn't even importable, because Python doesn't accept UTF-16 source.

| File | What it is |
|---|---|
| `player_106bd0e.py` | Snapshot of `gui/workspaces/player.py` as of commit **106bd0e** ("rebuild GUI as PySide6 Qt Widgets…", the first Qt player): 235 lines, a single-file VLC preview with a file picker. It describes itself as "Used for quick validation during duplicate review and similarity inspection". It holds no fix absent from the current player. This clone is shallow and doesn't contain 106bd0e, so I couldn't byte-diff it; the content is clearly the initial version. Safe to delete. |
| `diff_player_prev.txt` | `git show 106bd0e` output (286 lines), the commit that created the Qt app. |
| `diff_player.txt`, `player_full_diff.txt` | `git show c09cc51` ("claude updates to player"), which grew player.py from 156 to 1194 lines. The second is the longer version of the same dump. |
| `player_diff.txt` | A working-tree diff `cccfee4..beecf8e` of player.py (themes-manager import, `_BgArtWidget` tweaks). Already superseded. |

## Cross-page relationships
- **Landing → Player:** `MosaicLanding.tile_clicked(dirpath)` → `_play_dir` → `window.play_directory(dirpath)` → `load_directory_and_play`.
  - This is **non-recursive** (`Path.iterdir`). A tile whose cover came from a folder image in a folder with no audio directly inside (e.g. an artist folder) silently plays nothing (PLY-26).
  - It runs before the deferred `set_library`, which then triggers the Player's second library scan (PLY-11).
- **Music Graph → Player:** `play_tracks_requested(paths, label)` → `AlphaDEXWindow.play_tracks` → navigate to `player` → `load_tracks_and_play`. This *replaces* the queue without asking and reads every track's tags on the GUI thread (1000 tiny local files took 0.28 s; real FLACs on HDD/NAS scale to seconds, PLY-23).
- **Player ⇄ Now Playing bar:** relayed in `AlphaDEXWindow._init_workspaces`.
  - The volume link uses `sliderMoved` / `sliderReleased` only, so it has no echo loop, but also misses wheel, keyboard and groove changes (PLY-09).
  - Seek: bar → `seek_to_ms` (absolute); page → `set_position` (fraction) on every `sliderMoved`. The page's own slider has no `_user_seeking` guard, so `_tick_position` moves it under the user's drag.
  - Neither slider loops back into the other.
- **Main window → Player:** `set_library_path` → `_on_library_changed` → immediate full rescan. It is called twice on a normal launch, once from `_load_persisted_library` and once from the landing's `set_library`, so the library is scanned twice.
- **Duplicates / Similarity → Player:** no link today, despite `load_tracks_and_play`'s docstring ("Called by other workspaces (duplicate finder, similarity)").
- **Player → Playlists:** an M3U saved from the Player lands in `<library>/Playlists`, which the Player's own scanner then indexes (it doesn't skip reserved folders).
- **docs/gui_inventory.md §6.10 has drifted:**
  - It lists "seek back / seek forward" transport buttons and "`Left`/`Right` seek ±5s" shortcuts. Neither exists; the buttons are Prev / Play / Next / Stop / Shuffle / Repeat and the only seek shortcuts are Shift+←/→ ±30 s.
  - It calls the file button "Open File"; the label is "Browse…".
  - It says the hover popup "appears after a 1.5-second dwell". In practice it doesn't (PLY-15).
  - Keyboard Mode isn't documented at all.

## Bugs found
| ID | Severity | Location | Trigger | Effect | Confidence | Suggested fix |
|---|---|---|---|---|---|---|
| PLY-01 | crash | `gui/workspaces/player.py:947` (+ `alpha_dex_gui.py:122-125`) | python-vlc installed (it's in `requirements.txt`) but libVLC missing. `import vlc` raises `NotImplementedError("Cannot find libvlc lib")`, and `vlc.Instance()` can also return `None` | `_init_vlc` catches only `ImportError`, so `PlayerWorkspace.__init__` raises and **`AlphaDEXWindow()` fails**. In `alpha_dex_gui.py`, `window` stays `None` and `_landing_to_main` re-schedules itself every 10 ms forever: the landing fades out and no main window ever appears | confirmed-by-running (window construction raises; the hang loop is traced-in-code) | Catch `Exception` and check `Instance()` for `None`. Use `audio_preview.py`'s guard pattern. Make the entry point show an error when construction fails |
| PLY-02 | UX | `player.py:1143-1185`, `1204-1266`, `1281` | No playback backend (this environment) and the user plays anything | The page and bar show ⏸ "playing", "Playing: …" is logged, the bar appears, and `_pos_timer` polls forever. Play/Pause does nothing (`_on_play_pause` returns early), so the button is stuck on ⏸ | confirmed-by-running | Disable transport when there's no backend, show the reason in the bar, and don't emit `playback_state_changed(True)` |
| PLY-03 | wrong-result | `player.py:1471-1473`, `1289-1293` | Fresh session: "+ Add Selected" (or "Add to queue"), then Play / Space / bar ▶ | `_add_paths_to_queue` sets `_queue_index = 0` without playing. `_on_play_pause` then sees `_queue_index >= 0` and calls `media_player.play()` with **no media** (fake VLC recorded `play(None)`). Nothing plays, the UI shows ⏸, and the title stays "—". The primary build-a-queue-then-Play flow is broken; after an earlier Stop it replays the *old* media instead | confirmed-by-running | Track "queue cursor" separately from "loaded media"; when nothing is loaded, Play should call `_play_current()` |
| PLY-04 | wrong-result | `player.py:1382-1386` with `1411` | Queue built but not started (index 0); a directly played track ends | `play_next` increments to index 1, so **queue item 0 is skipped** (confirmed: plays `02.mp3`). Same skip after any direct play mid-queue | confirmed-by-running | When resuming the queue after direct play, play `_queue_index` itself unless it was already played |
| PLY-05 | wrong-result | `player.py:1539-1544` | Right-click → "Remove from queue" on the **currently playing** item | The index is decremented, so the highlight, `_current_playing_row()`, art matching and neighbour covers point at a *different* track while VLC keeps playing the removed one. Shuffle history and `_shuffle_next_idx` aren't remapped (stale indices make Next/Prev silently do nothing once) | confirmed-by-running | Handle `qi == _queue_index` explicitly (stop, or keep playing as direct play); remap shuffle state |
| PLY-06 | wrong-result | `player.py:1283-1287` | "30s Preview" on (or a keyboard ↓ preview), user pauses | `_highlight_timer` keeps running; 15 s later `play_next()` **starts the next track while the user is paused** | confirmed-by-running | Pause and resume the highlight timer (`remainingTime()`) with playback; stop it on Stop |
| PLY-07 | wrong-result | `player.py:1879`, `1276-1277` | Keyboard Mode, library target, press ↓ ("preview next library track") | After 15 s `_highlight_advance` → `play_next()` advances the **queue**, not the library. With a queue it jumps to an unrelated queue item (confirmed `02.mp3` → `ArtistB/02.flac`); with an empty queue the preview just keeps playing | confirmed-by-running | Give the preview its own "next" source (library cursor vs queue) |
| PLY-08 | UX | `player.py:1376-1392` | A file VLC can't decode (state `Error`) | Only `Ended` is handled. The UI stays ⏸ "playing" forever, the timer polls forever, and there's no message or auto-skip | confirmed-by-running (fake VLC in `Error` state) | Handle `Error` (and `Stopped` while expecting play): log, mark the row, skip |
| PLY-09 | wrong-result | `player.py:854`, `919-920`; `now_playing_bar.py:111-112`, `134-137` | Mouse wheel, arrow / PageUp / PageDown keys, or a groove click on either volume slider or either seek slider | Only `sliderMoved` / `sliderReleased` are connected. The volume knob moves (80→56) but **no `audio_set_volume`** is sent and the other slider doesn't follow; seek knobs move then snap back with no seek | confirmed-by-running | Connect `valueChanged` / `actionTriggered`, using `QSignalBlocker` for programmatic sync instead of relying on `sliderMoved` |
| PLY-10 | UX | `gui/main_window.py:168`, `176-207` | Keyboard Mode on, then use arrow keys anywhere: another workspace's line edit or spin box, any dialog, item views, combo popups | Arrows are swallowed app-wide (confirmed: Indexer spin-box cursor didn't move; a `QDialog` line edit didn't move). With a library loaded they **change the playing track while you're on Indexer** (confirmed `set_media`). The filter also runs Python for every app event even when the mode is off | confirmed-by-running | Only intercept when focus is inside the Player/bar, and never for text inputs, popups or modal dialogs; install the filter only while the mode is active |
| PLY-11 | wrong-result | `player.py:976-997`, `131-134`; `main_window.py:370-372` + `alpha_dex_gui.py:136-139` | Rescan while a scan is running. Happens on **every normal launch** with a saved library (persisted load + landing `set_library` of the same path), or a Reload / library change mid-scan | The old scanner's queued `tracks_ready` / `scan_complete` are delivered after the table reset. **Duplicate rows** (400 files → 403 rows), and a premature "Loaded 33 tracks." with Reload re-enabled while the new scan is still running | confirmed-by-running | Disconnect old scanner signals, or tag batches with a generation id and drop stale ones; skip a rescan when the path is unchanged |
| PLY-12 | crash | `player.py:1085-1090` (also `1612-1626`, `976-978`) | Single-click two library rows within one art-read time (slow disk/NAS, big embedded art). Also any art load over 2 s or a scan stuck over 1 s, followed by a replacement | The parentless `QThread` reference is overwritten while running: `QThread: Destroyed while thread is still running`, **process abort (exit 134)**. The single-click loader's result is a no-op anyway | confirmed-by-running (1 s read delay, two clicks) | Delete the single-click loader. Keep loaders alive until `finished` (parent them, or a pool/QRunnable), check `isInterruptionRequested()`, and never `wait()` on the GUI thread |
| PLY-13 | crash | `player.py` (no `closeEvent`; scanner `993-998` has no parent) | Close the window while the library scan (or an art load) is running | Interpreter teardown destroys the running `QThread` and the **process aborts (exit 134)** on exit | confirmed-by-running | Add a shutdown hook: interrupt and wait the scanner and loaders, stop VLC, from `AlphaDEXWindow.closeEvent` |
| PLY-14 | wrong-result | `now_playing_bar.py:246-254`; `player.py:1646-1647` | Play a track with cover art, then one without | The bar **keeps the previous track's cover** under the new title (confirmed); the blurred background also keeps the old art | confirmed-by-running | Emit an explicit "no art" and clear the thumbnail to the ♪ placeholder; have `_BgArtWidget` fade to empty |
| PLY-15 | UX | `player.py:737`, `1812-1832`, `428` | Hover a library row for 1.5 s | The viewport has `mouseTracking=False`, so button-less `MouseMove` never reaches the object filter and **the hover popup never appears**. It only arms while a button is held (drag-select); when shown, it is a `Qt.Popup` that grabs mouse and keyboard mid-drag. `show_for` also reads tags and cover synchronously on the GUI thread | confirmed-by-running | `setMouseTracking(True)` on the viewport, use a `ToolTip`-type window (no grab), and reuse the row cache plus an async art loader |
| PLY-16 | cosmetic | `player.py:713-721`, `255`, `278`, `639-642`, `672` | Switch to a light theme (Pearl) | Library rows stay near-black (`#0e1117`) while the theme text turns dark (`#111827`): **the library is unreadable** (screenshot confirmed). Many other hard-coded dark colours | confirmed-by-running | Derive table, background art and vignette colours from theme tokens; repaint on `theme_changed` |
| PLY-17 | UX | `player.py:705`, `1034-1038` | Open the Player | Sorting is enabled with the default indicator (column "#", descending), and numbers and times are stored as **strings**. The library opens sorted by track number *descending across all albums* (screenshot: 3, 2, 1, …), with lexicographic order ("10" < "2"; "10:00" < "2:00") | confirmed-by-running | Default sort Artist/Album/Track ascending; set numeric `EditRole` data for # and Time |
| PLY-18 | UX | `player.py:1760-1764` | Load M3U on a non-UTF-8 (Latin-1/cp1252) playlist, common for `.m3u` | `UnicodeDecodeError` isn't caught (only `OSError`): an exception in the slot, no message, nothing loads | confirmed-by-running | Try utf-8-sig, then fall back to cp1252/latin-1 (`.m3u8` = utf-8); catch `ValueError` |
| PLY-19 | UX | `player.py:1732` | Click "Save Queue as M3U…" / "Save selection as M3U…" and cancel | `os.makedirs(<library>/Playlists)` runs **before** the dialog, so a folder is created in the library without confirmation (confirmed after cancel) | confirmed-by-running | Create the folder only after a destination is chosen (or let the dialog start from an existing directory) |
| PLY-20 | UX | `player.py:1419-1420` | Direct-play a track with an empty queue and press Prev (or ← in queue mode) | Returns immediately: can't restart the track and nothing happens (confirmed: no VLC calls) | confirmed-by-running | Apply the ">3 s → restart" rule before the empty-queue check |
| PLY-21 | UX | `player.py:1402-1403` | Repeat = Track, press Next | Next replays the same track; the user can't skip without turning repeat off (confirmed: `01.mp3` again) | confirmed-by-running | Repeat-Track should only affect auto-advance at end of track, not explicit Next |
| PLY-22 | wrong-result | `player.py:106-114` | Library contains `Quarantine/`, `Trash/`, `Not Sorted/`, `Playlists/` | The Player lists and plays quarantined duplicate losers and other reserved-folder content as normal library tracks (confirmed: `Quarantine/dupe.mp3` listed) | confirmed-by-running | Skip the reserved folders (as the Indexer and Duplicate Finder do), or show them as a separate, labelled source |
| PLY-23 | stability | `player.py:1798`, `1469`, `1140`, `491-499` | Graph lasso of thousands of tracks, large M3U, "Add Selected" on a big selection, hover popup | Each path gets a synchronous mutagen open on the GUI thread (1000 tiny local files = 0.28 s; real FLACs on HDD/NAS = seconds of frozen UI). Rows already scanned in `_all_rows` aren't reused | confirmed-by-running (timing) | Look up the scan cache by path; read the rest in a worker and fill rows lazily |
| PLY-24 | cosmetic | `now_playing_bar.py:194-207` | Toggle Keyboard Mode twice within 120 ms | A second geometry animation starts from the mid-pulse geometry and saves that as its end value; the button ends up offset (1 px drift confirmed) until the next relayout. Geometry animation fights the layout | confirmed-by-running | Animate a scale/opacity effect or `iconSize`, not `geometry`; stop the running animation before starting a new one |
| PLY-25 | UX | `player.py:1798-1805` (`load_tracks_and_play`) | Graph "Play selection", landing tile or Load M3U while a hand-built queue exists | The queue is **replaced** without warning or undo | traced-in-code | Offer "Play now / Add to queue / Replace queue", or keep the previous queue in history |
| PLY-26 | UX | `player.py:1780-1788` | Click a landing tile whose cover came from a folder with no audio directly inside it (e.g. an artist folder with `folder.jpg`) | `iterdir` is non-recursive, so there are no paths and it returns silently: the user lands on the Player with nothing playing and no message | suspected (depends on which folders the landing scanner picks as tiles) | Walk recursively (bounded), or log/status "No tracks in …" |
| PLY-27 | cosmetic | `player.py:1299-1310`, `1483-1490`; `now_playing_bar.py:256` | Press Stop or Clear queue | The page resets its title to "—", but the bar stays visible showing the old title and art (confirmed) | confirmed-by-running | Emit a "cleared" state; the bar resets or hides |
