# Utilities (Tools), Compression (Opus mirror), Help — page intent map
_Scope: `gui/workspaces/tools.py` (ToolsWorkspace, ToolTile, TileGrid, _PillToggle, _GlassBadge, _GlassResultChip, FileCleanupWorker, ArtistTitleWorker), `gui/workspaces/compression.py` (CompressionWorkspace, MirrorWorker), `gui/workspaces/help.py` (HelpWorkspace), `gui/dialogs/media_tester_dialog.py`, plus the dialogs the Tools page launches (`bucketing_poc_dialog.py`, `scan_engine_dialog.py`, `fuzzy_dupe_dialog.py`, `pair_review_dialog.py`, `crash_log_dialog.py`) and the backends they drive (`utils/opus_library_mirror.py`, `playlist_generator.update_playlists`, `validator.py`, `utils/audio_metadata_reader.py`, `utils/opus_metadata_reader.py`, `near_duplicate_detector.py`, `duplicate_scan_engine.py`, `duplicate_bucketing_poc.py`, `crash_logger.py`). Date: 2026-11-07._

Probes used to confirm behaviour are in
`/tmp/claude-0/-home-user-Music-Indexer/4300de3c-51f6-599a-948e-c4bafb432735/scratchpad/tools/`
(`probe_cleanup.py`, `probe_cleanup2.py`, `probe_mirror.py`, `probe_dest.py`, `probe_at.py`,
`probe_qthread.py`, `probe_close.py`, `probe_fuzzy.py`). All were run with `QT_QPA_PLATFORM=offscreen`
and a throwaway `HOME`.

## Summary
This area holds the app's catch-all screens. **Utilities** (sidebar key `tools`, page title "Export & Utilities") is a grid of five tiles: two read-only text exports (Artist/Title list, Codec list), one tool that renames files (File Cleanup), a launcher for seven diagnostic dialogs, and a layout validator. **Compression** (sidebar key `compression`, listed under the PLAYER group) builds a separate, portable copy of the library: FLAC is converted to Opus, everything else is copied, and the source is never changed. **Help** is a static page with doc links, a shortcut table and an About card. A user mostly comes to these pages for one-off jobs. The trouble is that "Utilities" mixes harmless exports with two of the most destructive actions in the app: File Cleanup renames files, and Pair Review permanently deletes them. Neither has the preview, Docs/ log or quarantine that CLAUDE.md requires. The page also hosts experimental duplicate tools that overlap the Duplicates workspace, and some of them are broken.

## Pages

### Utilities / Tools workspace (`gui/workspaces/tools.py`, class `ToolsWorkspace`)
- **Intent:** a single place for small library utilities: exports, renaming hygiene, diagnostics and layout validation.
- **When the user comes here / entry points:** sidebar item "Utilities" (key `tools`, TOOLS group). There is no Ctrl+N shortcut, because only the first 9 keys of `_WORKSPACE_MAP` get one (`main_window.py:241`). No other page navigates here. Diagnostics such as Pair Review tell the user to "Run Duplicate Finder first", but nothing links back.
- **Sub-tool inventory (every tool on the page):**

| # | Sub-tool | Where | Intent (one line) | Mutates files? | Preview + confirm? |
|---|---|---|---|---|---|
| 1 | Artist / Title Export | tile | Write `Docs/artist_title_list.txt`, a de-duplicated "Artist - Title" line per track (with optional album/track suffix for repeated titles) | Writes one report file only | n/a (read-only on audio) |
| 2 | Codec List Export | tile | Write `Docs/codec_file_list.txt`, file paths grouped by extension | Writes one report file only | n/a |
| 3 | **File Cleanup** | tile | Strip trailing " (1)", " copy" etc. from audio filenames, then patch playlists | **Renames audio files in place** | **Confirm only, no preview.** Generic Yes/No (`tools.py:1314`), no list of planned renames, no Docs/ log, no undo. **Preview-first violation.** |
| 4 | Library Validator | tile | Check that the library has `By Artist/` + `By Year/` (directly or under `Music/`) | No | n/a |
| 5 | M4A Tester… | diag button → `MediaTesterDialog(codec="m4a")` | Open one .m4a and show the tags and cover it parses | No | n/a |
| 6 | Opus Tester… | diag button → `MediaTesterDialog(codec="opus")` | Same for .opus (e.g. to check Compression output) | No | n/a |
| 7 | Bucketing POC… | diag → `BucketingPocDialog` | Run the duplicate-bucketing proof of concept, write an HTML report to `<folder>/Docs/` | Report only | n/a |
| 8 | Scan Engine… | diag → `ScanEngineDialog` | Run the staged duplicate scan engine with ten tunable thresholds into a SQLite DB (default `Docs/duplicate_scan.db`) | DB only | n/a |
| 9 | Fuzzy Finder… | diag → `FuzzyDupeDialog` | Metadata/fingerprint near-duplicate search with a results table | No | n/a (and broken, see TOOL-07) |
| 10 | **Pair Review…** | diag → `PairReviewDialog` | Step through duplicate pairs from the last Duplicate Finder preview and delete the inferior file, one pair at a time | **Permanently deletes audio (`os.remove`)**; appends to `Docs/duplicate_pair_tags.txt`; patches playlists | **The side-by-side panel is the only preview. One click deletes, with no confirm, no quarantine and no Docs/ action log** (TOOL-06) |
| 11 | View Crash Log… | diag → `CrashLogDialog` | Show the last 50 lines of the crash log | No | n/a (always empty, TOOL-11) |

- **Primary flow (per tile):**
  - Artist/Title: 1) set toggles → 2) Export → the drawer animates open, `ArtistTitleWorker` runs and reports progress → 3) "Open File" / folder buttons become enabled. No preview needed (read-only).
  - Codec: 1) toggle extension chips / "Filenames only" → 2) Export runs **synchronously on the GUI thread** → 3) Open File / folder.
  - File Cleanup: 1) Run Cleanup → 2) **confirm** (Yes/No text only; this is the only gate) → 3) worker renames everything in place and streams the log into the tile drawer → 4) `update_playlists(rename_map)`. **There is no preview step.**
  - Validator: 1) Run Validator → result text in the tile (synchronous).
  - Diagnostics: each button opens a modal dialog via `exec()`.
- **Reads / writes:** reads `self._library_path` (set by main window). It never touches config. It writes `<lib>/Docs/artist_title_list.txt` and `<lib>/Docs/codec_file_list.txt`, and renames audio files under `<lib>` (all subfolders, **including reserved `Quarantine/`, `Trash/`, `Not Sorted/`, `Playlists/`, `Docs/`**). It rewrites `.m3u` files under `<commonpath>/Playlists`. Signals out: `log_message` only (never `status_changed` or `navigate_requested`).
- **Backend dependencies:** `utils.audio_metadata_reader.read_tags` and `utils.opus_metadata_reader.read_opus_metadata` (ArtistTitleWorker thread), `playlist_generator.update_playlists` (FileCleanupWorker thread), `validator.validate_soundvault_structure` (GUI thread), `os.walk` (GUI thread for the codec export).
- **State owned:** worker refs (`_at_worker`, `_cleanup_worker`), "connected" flags for the Open-File buttons, and the last output path captured in lambdas. On library change nothing resets: the Open File buttons still point at the previous library's report, and the logs still show old output (`ToolsWorkspace` does not override `_on_library_changed`).
- **Async model:**
  - `ArtistTitleWorker(QThread, parent=self)` guards against a double click with `isRunning()`. The Export button is never disabled. There is no cancel.
  - `FileCleanupWorker(QThread)` has **no parent**. The Run button is disabled while running, which guards against a double click. There is no cancel. The only reference is dropped in the finished slot while the thread can still be alive, which **crashes the process** (TOOL-04, confirmed).
  - Navigating away is harmless, since workspaces are never destroyed (`main_window._init_workspaces` pre-creates all of them). **Closing the app mid-run aborts the process**, because `AlphaDEXWindow` has no `closeEvent` and nothing waits for or stops workers (same mechanism as TOOL-05).
  - The Codec export and the Validator block the GUI thread.
  - Both worker classes define a custom `finished` signal that shadows `QThread.finished`. It works, but it is the root of TOOL-04: the "finished" slot fires before the thread has actually finished.
- **Motion & visuals:**
  - `ToolTile` paints a "liquid glass" card with a hover glow that follows the mouse. This is driven by a **16 ms `QTimer` per tile that starts in `__init__` and never stops** (5 timers × 60 Hz from app start, visible or not; TOOL-18).
  - The drawer reveal is a `maximumHeight` `QPropertyAnimation` that ends by setting max to 16777215. It runs once only and never closes again.
  - `_PillToggle` animates its knob. `_GlassResultChip` does a fade-out/fade-in text swap. `_flash_label` animates the status-label colour.
  - Theme bypasses:
    - `ToolTile.paintEvent` reads the nonexistent `tokens.is_dark` (the token is `variant`), so light themes get the dark tint (TOOL-15).
    - `_GlassResultChip._COLOR` hard-codes slate/green/red.
    - Log boxes use `rgba(0,0,0,0.1)`.
    - `_flash_label` leaves a hard-coded final colour baked into the label stylesheet, so it is stale after a theme switch.
    - The codec chips do re-style on theme change (`_on_theme_changed_base`).
- **Decorative / dead controls:**
  - `ToolTile.status_chip` on every tile shows "Ready." forever. Nothing ever calls `show_success`/`show_error` (TOOL-17).
  - `ToolTile.set_running()` and `ToolTile.flash_status()` are never called.
  - `ToolsWorkspace._reveal_card()` is unused.
  - The Codec tile's `self._codec_prog` sits in a drawer that never opens, so its `setValue` calls are invisible.
  - Inside dialogs: FuzzyDupe "Send Matches to Duplicate Pair Review" is a stub message box (`fuzzy_dupe_dialog.py:239`), and its search params (`min_word_len`, `min_shared`) are stored but never passed to the backend.
- **Overlaps (explicit, for de-duplication):**
  - **Duplicate tooling is scattered across four Tools dialogs and two workspaces.** Fuzzy Finder, Pair Review, Bucketing POC and Scan Engine all overlap the **Duplicates** workspace (fingerprint grouping, preview plan, quarantine/delete) and the **Similarity Inspector** (explain why two files match). Pair Review even consumes Duplicates' own preview (`load_pairs` reads the Duplicate Finder snapshot/preview), and then deletes outside that workspace's quarantine-first executor.
  - **Playlist repair** exists in File Cleanup (automatic `update_playlists`) and Pair Review (`_update_playlists`), as well as the Playlists workspace's "Playlist Repair" tab (`playlists.py:266`).
  - **Opus Tester** overlaps verifying Compression's output. Both testers overlap Tag Fixer and Player track-details metadata display.
  - **Validator** duplicates the check `controllers/library_controller.py` / `import_controller.py` already run on library selection, and it encodes Indexer's layout. It belongs next to the Indexer or the library picker.
  - The **Artist/Title and Codec exports** are reports about the library. They sit naturally beside the Indexer's HTML preview and the library stats, or in a single "Reports" area alongside Library Sync's Export Report.
- **Redesign notes:**
  - Split the page by risk. Group read-only reports (exports, validator, crash log) separately from **actions that change files**.
  - File Cleanup should become plan → preview table → confirm → execute → Docs/ log, ideally reusing the Indexer's preview contract. It must skip reserved folders, and its suffix rules must be far stricter (TOOL-01).
  - Move Pair Review, Fuzzy Finder, Scan Engine and Bucketing POC into the Duplicates workspace as "advanced/experimental" tabs, or retire them. Pair Review must route deletes through `duplicate_consolidation_executor` (quarantine by default).
  - The always-"Ready." chip, the dead progress bar and the invisible `status_label` duplication should be removed or made real.
  - Modal `exec()` dialogs for long scans block the whole app, so prefer in-page panels with real cancel.
  - The glass tiles look good but cost 5 always-on timers. Drive hover with `QVariantAnimation` on enter/leave instead.
- **Testability seams:**
  - Patch `QtWidgets.QMessageBox.question/warning/information/critical`, `QFileDialog.*`, `subprocess.Popen` (Open File/Folder use `xdg-open`/`open`/`explorer`), and `webbrowser.open`.
  - Workers can be driven synchronously by calling `.run()` directly (as the probes do). For a real `start()`, wait until `isRunning()` is False *and* the instance has a parent, otherwise TOOL-04 aborts the test process.
  - Attributes to latch onto: `_exclude_flac_cb`, `_dupe_tracks_cb` (`_PillToggle`, use `setChecked`), `_codec_ext_cbs[ext]`, `_omit_paths_cb`, `_codec_status`, `_at_status`, `_at_log`, `_at_prog`, `_at_open`, `_at_folder`, `_cleanup_run_btn`, `_cleanup_log`, `_cleanup_status`, `_val_log`, plus tiles via `findChildren(ToolTile)`. The diag buttons have no objectName or text (text is in a child QLabel), so tests must call `_on_m4a_tester()` etc. directly. **Add objectNames.**
  - Patch the dialog classes in their modules (`gui.dialogs.*`), since the slots import them lazily.

### File Cleanup and Exports after the Liquid Glass restyle (R2, 2026-11-08)

The Utilities page described above was split by the information architecture into
**Organize → File Cleanup** (`FileCleanupWorkspace`) and **Take away → Exports**
(`ExportsWorkspace`), both still in `tools.py`. R2 restyled both without changing a
worker, a confirmation or a library-write lock. Current screens:
`docs/gui_inventory.md` §6.4, §6.11 and §7.7.

- **`ToolTile` is a `GlassCard`.** It paints the theme's card surface (opaque in
  classic themes and with Reduce transparency) and its shadow is painted by the page
  like any card's. The cursor-following hover glow and its timer are gone (the DS-09
  timer test now checks that hovering starts no timer or animation; the DS-08 tint
  test checks that a light theme gives a light tile). `_GlassBadge` (a text glyph in
  an accent square) and the unused text-scale API went with it.
- Each tile: a title, one muted line, a **status pill** (Ready / Running… / Done /
  Error / Cancelled, tinted by tone, with a check or warning glyph; no ⧖ ✓ ✗
  characters), an **EmptyState before the first run**, pill buttons with painted
  glyphs, and a result area that fades in (opacity) instead of animating its height,
  which a layout owns. The written file shows as a middle-elided path with the full
  path in its tooltip, instead of being pasted into the status line.
- File Cleanup is one **hero** tile ("Clean up copy suffixes"). Exports keeps two
  tiles under a one-line purpose; the codec chips are painted toggles (they used to
  restyle themselves with an inline stylesheet on every toggle and theme change).
- The preview dialog's contents (the window chrome is the shell's): a title, the
  counts as stat tiles, the plan as a glass list on a card, a muted note, and the
  confirm button as the one primary button. Cancel stays the default button, so
  Enter never renames.
- In-page titles that repeated the shell header are gone; the hygiene counts for
  this file dropped by four inline stylesheets and two `font-size` literals.

### Artist / Title Export (tile; `ArtistTitleWorker`)
- **Intent:** produce a plain-text "what's in my library" list (e.g. to share or diff against a wishlist).
- Reads tags on a worker thread. Missing title falls back to the filename stem, and missing artist becomes "Unknown Artist". Includes files in reserved folders such as `Quarantine/` (confirmed: a quarantined `loser.mp3` appears as "Unknown Artist - loser"; TOOL-14). Output path is fixed, so each run overwrites the previous file silently (acceptable for a report). Both toggles are read (`tools.py:1180-1181`) and functional.

### Codec List Export (tile)
- **Intent:** list every file of the chosen codecs, e.g. to find all lossy files for replacement.
- Synchronous `os.walk` on the GUI thread (`tools.py:1260`), so it freezes on large or networked libraries. It walks reserved folders. All chips and the toggle are read. It writes a file even when 0 files match.

### File Cleanup (tile; `FileCleanupWorker`)
- **Intent:** undo "Song (1).mp3" / "Song copy.mp3" artifacts left by OS file managers and keep playlists pointing at the right files.
- **Confirmed by running (`probe_cleanup.py`):** the regexes strip legitimate title text. "Carbon Copy.mp3" → "Carbon.mp3", "Xerocopy.mp3" → "Xero.mp3", "Live at Wembley (1986).mp3" → "Live at Wembley.mp3", "Symphony No. 5 (1808).flac" → "Symphony No. 5.flac". It also renamed `Quarantine/Loser (1).mp3`. No `Docs/` log is written; the only record is the tile's log box, which is lost on restart.
- **Confirmed (`probe_cleanup2.py`):** when all renames sit in one folder, `update_playlists` looks for `<that folder>/Playlists`, finds nothing and returns. The tool still logs "✓ Updated playlists", while the `.m3u` keeps pointing at `Song (1).mp3`.

### Library Validator (tile)
- **Intent:** tell the user whether the chosen folder is laid out the way AlphaDEX's Indexer lays it out.
- `validator.py` requires **both** `By Artist/` and `By Year/`. `music_indexer_api` only creates `By Year/` for "rare" artists (`music_indexer_api.py:668-672`), so an indexed library with no rare artists is reported invalid (TOOL-22, traced). The error text is generic and has no link to the Indexer.

### Diagnostics dialogs (launched from the Diagnostics tile)
- **MediaTesterDialog** (`gui/dialogs/media_tester_dialog.py`, class `MediaTesterDialog`). Intent: sanity-check that tags and cover art in an .m4a/.opus parse correctly. Flow: Browse (`QFileDialog.getOpenFileName`, starting at `Path.home()`), then `_load_file` runs synchronously on the GUI thread via `mutagen.File`. Results:
  - **Opus:** shows `Error:` for any Opus file missing one of ten keys, because `VCommentDict.get("\xa9…")` raises `ValueError`. Confirmed on a file produced by Compression (TOOL-08).
  - **Opus art:** never found, since the code looks for `covr`/`apic` and Opus uses `METADATA_BLOCK_PICTURE` (TOOL-09).
  - **M4A:** the fallback key mapping is wrong. Title/artist/date/track never show, and "Albumartist" shows the album (TOOL-10, confirmed).
  - **Stale art:** a parse failure (`f is None` / exception) leaves the previous file's art on screen.
  - **Theme:** the art box hard-codes `#e2e8f0` (bright box in dark themes).
  - **Seams:** call `_load_file(path)` directly, read `_meta_text`, `_art_lbl`.
- **BucketingPocDialog**: folder entry (defaults to library), Run, then a worker thread (unparented `_BucketingWorker`), then a modal "Open report?". No cancel. Closing the dialog leaves the worker running.
- **ScanEngineDialog**: library + DB path + ten numeric fields (validated only by `int()`/`float()`), Run, then a worker. No cancel. Guards against a double click.
- **FuzzyDupeDialog**: has a Cancel button (cooperative flag), but **every run fails immediately**: `find_near_duplicates(library_path, log_callback=…)` does not match the backend signature `(file_infos, ext_priority, threshold, …)` (TOOL-07, confirmed). The filter fields are decorative and "Send to Pair Review" is a stub.
- **PairReviewDialog**: see sub-tool #10 above (TOOL-06). Its `_update_playlists` uses the same commonpath-based `update_playlists`, so it is usually a no-op (TOOL-03).
- **CrashLogDialog**: looks for `~/soundvault_crash.log` and `./soundvault_crash.log`. The Qt app never calls `crash_logger.install()`, and the default path there is `crash.log`, so the dialog always says "No crash log found." (TOOL-11, confirmed). There is also a dead `log_path` variable at `crash_log_dialog.py:29`.
- **All dialogs:** created with `parent=self` and `exec()`'d, never `deleteLater()`'d, so each open leaks a QDialog (3 opens → 3 live `CrashLogDialog` children, confirmed). Re-opening Scan/Bucketing/Fuzzy while a previous dialog's worker still runs allows a concurrent second run.

### Compression (`gui/workspaces/compression.py`, class `CompressionWorkspace`)
- **Intent:** make a smaller, portable copy of the library (e.g. for a phone or DAP). FLAC becomes Opus at a chosen bitrate, all other files are copied unchanged, and originals are never touched.
- **When the user comes here / entry points:** sidebar "Compression" (key `compression`). It sits in the **PLAYER** group (`gui/widgets/sidebar.py:43`), which is an odd IA choice since it is a library-output tool. No shortcut, and nothing navigates here.
- **Primary flow:**
  1. Source is pre-filled from the library; pick a Destination (`QFileDialog.getExistingDirectory`).
  2. Set the bitrate (48–320, default 96) and the Overwrite option.
  3. **"Estimate Space Savings"** is the *preview*. It walks the source synchronously on the GUI thread, counts FLAC vs other files and bytes, and estimates Opus size as `flac_bytes × kbps / 900`. It enables Start.
  4. **Start Compression** first checks `shutil.which("ffmpeg")` (critical box if missing), then asks for an **explicit Yes/No confirmation** that lists source, destination and bitrate. The Overwrite state is not mentioned.
  5. `MirrorWorker` runs and the bar goes from indeterminate to determinate.
  6. On finish, it writes the HTML report to `<dest>/Docs/opus_library_mirror_reports/opus_library_mirror_report_<ts>.html` and enables "Open Report" (`webbrowser.open`).
- **What it produces (confirmed by running a 3-file library, `probe_mirror.py`):** `<dest>/Artist/Album/01 Song.opus` (tags and front cover copied into `METADATA_BLOCK_PICTURE`), `<dest>/Artist/Album/02 Other.mp3` (copied), **`<dest>/Quarantine/loser.mp3` (reserved folder copied, TOOL-14)**, plus the report. The result line read "Converted 1 FLAC → Opus, copied 2, skipped 0, errors 0".
- **Reads / writes:** never writes to the source except via TOOL-13: `_validate_paths` runs `makedirs(destination)` *before* checking that the destination is outside the source, so Estimate with a destination inside the library creates empty folders inside the library and only then refuses. Writes the whole mirror tree and the report. No config keys; bitrate, destination and overwrite are not persisted.
- **Backend dependencies:** `utils.opus_library_mirror.mirror_library` on the `MirrorWorker` thread. That fans out to a `ThreadPoolExecutor(max_workers=cpu_count)`, and each task runs `ffmpeg -c:a libopus -b:a Nk -n|-y` via `subprocess.run`, then mutagen copies the tags and picture. `write_mirror_report` runs on the GUI thread.
- **What if ffmpeg or the encoder is missing:**
  - No ffmpeg on PATH: blocked up front with a critical box.
  - ffmpeg present but without libopus: every FLAC fails individually ("FFmpeg error: …") and is counted in `errors` and the report.
  - mutagen missing: `_load_flac_metadata` returns None, so **every FLAC is refused** ("Mutagen not available."). A FLAC with unreadable tags is likewise not converted at all.
- **State owned:** `_worker` and `_report_path`. On library change only the Source text is updated. The previous estimate text and the Start button's enabled state stay as they were (stale preview for a different library). If a run is in progress, the disabled Source field's text still changes under the running job.
- **Async model:**
  - `MirrorWorker(QThread, parent=self)` emits progress from pool threads. Queued delivery was confirmed: all `_log` calls landed on the GUI thread.
  - Buttons and fields are disabled while running (double-click safe), **except the bitrate spin box**, which stays editable (confirmed).
  - **There is no cancel.** A large library ties up all CPU cores with ffmpeg until it finishes.
  - Navigating away is fine.
  - **Closing the window mid-run aborts the process** with `QThread: Destroyed while thread is still running` (exit 134, confirmed by `probe_close.py`). ffmpeg children keep running orphaned. Metadata is written by Python after ffmpeg exits, so an abort can leave untagged `.opus` files. A later run without "Overwrite" then skips them as "Destination already exists", so the gap becomes permanent.
- **Motion & visuals:** none animated. Hard-coded colours throughout: `#64748b` info/label text, `#e2e8f0` estimate text, `#4ade80` success, `#f87171` error, `#94a3b8` note. **`#e2e8f0` on the light themes' `#ffffff` card is near-invisible (≈1.2:1)** (TOOL-16).
- **Decorative / dead controls:** none fully dead. The "Estimate" preview is a counts-only summary with no per-file plan.
- **Overlaps:** Opus Tester (Tools) is the natural verifier of this page's output, but it is broken for Opus (TOOL-08/09). "Prefer Opus" in Playlists relates to mirrored Opus libraries.
- **Redesign notes:**
  - Move it out of PLAYER into a "Library outputs / Export" group with the text exports.
  - Add Cancel, which needs a cancel event in `mirror_library` and executor shutdown.
  - Exclude reserved folders, or at least `Quarantine/`, `Trash/` and `Docs/`, by default.
  - Make Estimate async and invalidate it when source, destination or bitrate change.
  - Show Overwrite in the confirm text.
  - Show conversion errors prominently. Today they are "info" lines buried under "normalized tag X" spam, and the 500-line LogDrawer cap scrolls them away (TOOL-20).
- **Testability seams:**
  - Patch `QMessageBox.question` (Yes), `QMessageBox.critical/warning`, `QFileDialog.getExistingDirectory`, `shutil.which` (to simulate ffmpeg missing), `webbrowser.open`.
  - Patch `utils.opus_library_mirror.mirror_library` or `convert_flac_to_opus` for speed. A real ffmpeg run on 2-second lavfi sine files takes under 1 s.
  - Widgets: `_source_edit`, `_dest_edit`, `_bitrate_spin`, `_overwrite_cb`, `_estimate_btn`, `_run_btn`, `_report_btn`, `_result_lbl`, `_prog_bar`, `_prog_counter`. Wait on `ws._worker is None`.
  - Never tear down the workspace while `_worker.isRunning()` (it aborts the test process).

### Help (`gui/workspaces/help.py`, class `HelpWorkspace`)
- **Intent:** orient the user. Where the docs are, which shortcuts exist, what this app is, and where to report issues.
- **Entry points:** sidebar "Help" (key `help`). No shortcut (F1 is not bound).
- **Primary flow:** click a doc button → `_open_doc` resolves `<repo>/docs/...` and either opens it with `webbrowser.open("file://…")` or shows a "Not Found" box.
- **Link check (confirmed, `probe_fuzzy.py`):**
  - `docs/project_documentation.html`: **exists**, but it is a Jan-2026 Tkinter-era document describing `main_gui.py`, so it is stale for the Qt app.
  - `docs/gui_inventory.md`: exists. It opens as a raw `.md` in the browser (it may download instead of rendering).
  - `docs/library_sync_redesign.md`: **missing**. The file now lives at `docs/archive/library_sync_redesign.md` and is superseded by `docs/features/library_sync.md`. Clicking it gives "File not found" (TOOL-12).
  - GitHub issues URL: opens.
  - The current docs entry points (`docs/README.md`, `docs/overview.md`, `docs/architecture.md`, `docs/features/*.md`) are **not linked at all**.
- **Shortcut table:** matches `main_window._setup_shortcuts` (Ctrl+O, Ctrl+,, Ctrl+L, Ctrl+W, Ctrl+1–9). It does not mention that only the first 9 sidebar items have shortcuts. `docs/gui_inventory.md` §6.13 claims the About card shows "Version, author". **There is no version or author in the code** (doc drift).
- **Reads / writes:** nothing. `file://` + raw path is wrong on Windows (backslashes); `Path.as_uri()` would be correct.
- **Async model:** none.
- **Motion & visuals:** none. The card headings are plain `QLabel`s rather than `_make_card_title` (inconsistent with other pages). The About text hard-codes `#64748b`.
- **Decorative / dead controls:** none, apart from the broken link.
- **Overlaps:** none functionally. A future in-app doc viewer could render `docs/*.md` instead of shelling out to a browser.
- **Redesign notes:** link the maintained docs suite; add version/build info from a single source; add an F1 shortcut; consider contextual "?" help per workspace that deep-links to `docs/features/<page>.md`.
- **Testability seams:** patch `webbrowser.open` and `QMessageBox.information`. The doc buttons are the only `QPushButton`s apart from "Report an Issue", so clicking each and asserting the opened URIs works (see probe).

## Cross-page relationships
- **Main window → all three:** `set_library_path(path)` on library change. Tools ignores it (no `_on_library_changed` override, so state goes stale). Compression copies it into Source only. Help ignores it. None of the three emits `status_changed` or `navigate_requested`. Everything goes to the LogDrawer via `log_message`.
- **Duplicates → Tools/Pair Review:** Pair Review reads the Duplicate Finder's preview/snapshot (`pair_review_dialog.load_pairs`). Afterwards it deletes files itself, bypassing Duplicates' quarantine-first executor. Tools tells the user to "Run Duplicate Finder first" but offers no navigation.
- **Tools/Fuzzy Finder → Pair Review:** intended hand-off; it is a stub.
- **Compression → Tools/Opus Tester:** the natural "verify the mirror" path. It is broken (TOOL-08/09).
- **Tools/File Cleanup & Pair Review → Playlists:** both rewrite `.m3u` files via `playlist_generator.update_playlists`, which is the same backend the Playlists workspace's Repair tab uses. All three inherit TOOL-03.
- **Indexer ↔ Validator:** the validator encodes the Indexer's output layout but disagrees with it (TOOL-22).
- **Help → docs:** links to one missing file and one stale file. It does not link the maintained suite.

## Bugs found
| ID | Severity | Location | Trigger | Effect | Confidence | Suggested fix |
|---|---|---|---|---|---|---|
| TOOL-01 | data-loss | `gui/workspaces/tools.py:40-41` | File Cleanup on a library with titles ending in "copy"/"Copy" or a parenthesised number | Renames legitimate files: "Carbon Copy"→"Carbon", "Xerocopy"→"Xero", "Live at Wembley (1986)"→"Live at Wembley", "Symphony No. 5 (1808)"→"Symphony No. 5" | confirmed-by-running | Require a separator before "copy" (`\s+-?\s*copy$` / `\s-\sCopy$`), restrict the numeric suffix to small counters `\s\((?:[1-9]|[1-9]\d)\)$`, and only rename when the de-suffixed sibling does *not* already exist… or better, only when a same-stem original exists |
| TOOL-02 | data-loss | `gui/workspaces/tools.py:1314`, `:45` | Run Cleanup → Yes | Renames in place with no preview of the planned renames, no `Docs/` log, no undo. Also walks reserved folders (renamed `Quarantine/Loser (1).mp3`). Violates CLAUDE.md preview-first and no-silent-data-loss rules | confirmed-by-running | Split into plan (dry run → table/HTML) → confirm → execute. Write `Docs/file_cleanup_<ts>.log` with old→new for undo. Skip reserved folders |
| TOOL-03 | wrong-result | `playlist_generator.py:231` (called from `tools.py:75`, `pair_review_dialog.py:622`) | Renames/deletes confined to one folder (typical: one album) | `update_playlists` derives the library root as `commonpath(changed paths)`, looks for `<album>/Playlists`, silently returns. File Cleanup still logs "✓ Updated playlists" and the `.m3u` keeps the old path | confirmed-by-running | Pass the library root explicitly (`update_playlists(changes, library_root=…)`) and report the number of playlists actually changed |
| TOOL-04 | crash | `gui/workspaces/tools.py:1330`, `:1342` (same pattern `scan_engine_dialog.py:220,226`, `fuzzy_dupe_dialog.py:236`, `bucketing_poc_dialog.py:113,121`) | Run File Cleanup repeatedly (aborted within 20 runs on an empty library) | Unparented `QThread`. Its only ref is dropped in a slot fed by a custom `finished` signal emitted *inside* `run()`, so Python GC destroys the still-running QThread → `QThread: Destroyed while thread is still running` → process abort (exit 134) | confirmed-by-running (tools); traced-in-code (dialogs) | Parent the worker (`parent=self`), connect the built-in `QThread.finished` to `deleteLater`, rename the custom signal (`done`), and only drop the ref after `QThread.finished` |
| TOOL-05 | crash | `gui/workspaces/compression.py:343` (also `ArtistTitleWorker`, no `closeEvent` in `gui/main_window.py`) | Close the main window while Compression (or AT export / cleanup) runs | Process aborts with "QThread: Destroyed while thread is still running". ffmpeg children are orphaned. `.opus` files whose tags are written after ffmpeg exit can be left untagged and are then skipped forever without "Overwrite" | confirmed-by-running | Add a cancel event to `mirror_library`. Add `AlphaDEXWindow.closeEvent` that asks running workspaces to stop and `wait()`s. Write to a temp name and rename after metadata is written |
| TOOL-06 | data-loss | `gui/dialogs/pair_review_dialog.py:533` | Tools → Pair Review → "Yes — delete inferior" | `os.remove` permanently deletes an audio file on one click: no confirm, no quarantine, no `Docs/` action log. Contradicts "default to quarantine, not delete" | traced-in-code | Route through `duplicate_consolidation_executor` (quarantine default), log each action to `Docs/`, and add an undo/"review queue then apply" step |
| TOOL-07 | wrong-result | `gui/dialogs/fuzzy_dupe_dialog.py:32` | Fuzzy Finder → Run | `find_near_duplicates(library_path, log_callback=…)` doesn't match `(file_infos, ext_priority, threshold, …)` → TypeError every run. The tool never returns results | confirmed-by-running | Build `file_infos` (fingerprints via cache) and pass thresholds from config, or retire the dialog in favour of the Duplicates workspace |
| TOOL-08 | wrong-result | `gui/dialogs/media_tester_dialog.py:86` | Opus Tester on an .opus missing any of the 10 probed keys (e.g. a Compression output) | `VCommentDict.get("\xa9…")` raises `ValueError` → dialog shows "Error: " and nothing else | confirmed-by-running | Only use the `\xa9` fallback for MP4. Use per-format key maps |
| TOOL-09 | wrong-result | `gui/dialogs/media_tester_dialog.py:97` | Opus Tester on a file with embedded art | Art lookup matches only `covr`/`apic`. Opus/FLAC use `METADATA_BLOCK_PICTURE` / `.pictures`, so it always shows "No album art found" | traced-in-code (masked by TOOL-08 in practice) | Decode `metadata_block_picture` (base64 → `mutagen.flac.Picture`) |
| TOOL-10 | wrong-result | `gui/dialogs/media_tester_dialog.py:86` | M4A Tester on a fully tagged .m4a | Fallback `\xa9`+first-3-letters gives `\xa9tit`/`\xa9art`/`\xa9dat`/`\xa9tra` (wrong), so title/artist/date/track are never shown, and "Albumartist" displays the **album** (`\xa9alb`) | confirmed-by-running | Explicit MP4 map: `\xa9nam`, `\xa9ART`, `\xa9alb`, `aART`, `trkn`, `disk`, `\xa9day`, `\xa9gen`, `cpil` |
| TOOL-11 | wrong-result | `gui/dialogs/crash_log_dialog.py:31-33` | View Crash Log… | Looks for `soundvault_crash.log` in `~` and cwd. The Qt app never installs `crash_logger`, whose default is `crash.log`. Always "No crash log found." | confirmed-by-running | Install `crash_logger` in `alpha_dex_gui.py` with a fixed path (e.g. under the config dir) and read that same constant here |
| TOOL-12 | UX | `gui/workspaces/help.py:31` | Help → "Library Sync Redesign Notes" | "File not found" box (file moved to `docs/archive/`). The "Project Documentation" link opens a stale Tkinter-era HTML. The maintained docs (`docs/README.md`, `overview.md`, `features/*`) are not linked | confirmed-by-running | Link `docs/README.md`, `overview.md`, `architecture.md`, `features/*.md`. Drop the archived link. Use `Path.as_uri()` |
| TOOL-13 | wrong-result | `gui/workspaces/compression.py:228` | Estimate (or Start) with a destination inside the source library | `os.makedirs(destination)` runs before the "inside source" check, so empty folders are created in the user's library, then the run is refused | confirmed-by-running | Do the containment/equality checks first; create the destination only after confirm |
| TOOL-14 | wrong-result | `utils/opus_library_mirror.py:51`; `gui/workspaces/tools.py:137`, `:1260` | Mirror / Artist-Title export / Codec export on a library with reserved folders | Mirror copies `Quarantine/` (duplicate losers), `Trash/`, `Docs/`, `Not Sorted/` into the portable copy. Exports list quarantined files as library tracks | confirmed-by-running | Shared "reserved folders" helper used by every walker. Exclude by default with an opt-in |
| TOOL-15 | cosmetic | `gui/workspaces/tools.py:555` | Switch to a light theme (Pearl, Azure…) | `getattr(t, "is_dark", True)`: tokens have `variant`, not `is_dark`, so tiles always get the dark-mode black tint | traced-in-code | `t.variant == "dark"` |
| TOOL-16 | cosmetic | `gui/workspaces/compression.py:302` (also `:120,164,171,180,368,402,429`; `help.py:76`; `media_tester_dialog.py:45`; `tools.py:265-269`) | Light theme → run Estimate | Estimate text is hard-coded `#e2e8f0` on a `#ffffff` card (≈1.2:1, unreadable). Other hard-coded slates/greens ignore the theme | traced-in-code | Use objectNames/properties styled by the theme (`statusLabel`, `[state="ok"]`) instead of inline hex |
| TOOL-17 | UX | `gui/workspaces/tools.py:443-444` | Any Tools run | Each tile's status chip says "Ready." forever. Nothing calls `show_success/show_error`, so the chip contradicts the real status (e.g. during a failure) | confirmed-by-running | Drive the chip from the run/finished/error slots, or remove it |
| TOOL-18 | stability | `gui/workspaces/tools.py:405-408` | App start (Tools is pre-created) | Five 16 ms `QTimer`s run for the app's lifetime even when the page is hidden (~300 wake-ups/s) | confirmed-by-running | Start the timer on `enterEvent`, stop it when the glow converges, and stop it in `hideEvent` |
| TOOL-19 | stability | `gui/workspaces/compression.py:268`; `gui/workspaces/tools.py:1260` | Estimate / Codec export on a large or NAS library | Full `os.walk` + `getsize` on the GUI thread freezes the window. Compression, Artist/Title and Cleanup have no cancel | traced-in-code | Move to workers with progress and cancel |
| TOOL-20 | UX | `utils/opus_library_mirror.py:252-306` (`_normalize_tags`) → `compression.py:351` | Any compression run | Logs "normalized tag X" for every tag of every file (9 log lines for one 3-tag file) at level "info", including real ffmpeg failures. The 500-line LogDrawer cap scrolls errors away | confirmed-by-running | Drop per-tag logs. Send failures at "error"/"warn" level and summarise in the result card |
| TOOL-21 | stability | `gui/workspaces/tools.py:1367-1415` | Open any diagnostic dialog repeatedly; close Scan/Bucketing/Fuzzy mid-run then reopen | Each `exec()`'d dialog is parented to the workspace and never deleted (3 opens → 3 live dialogs). A closed dialog's worker keeps running and a reopened dialog can start a second concurrent scan | confirmed-by-running (leak); traced-in-code (concurrent run) | `setAttribute(WA_DeleteOnClose)` or `deleteLater()` after `exec()`. Make closing cancel and wait, or keep one dialog instance |
| TOOL-22 | wrong-result | `validator.py:20-29` | Validate a library indexed by AlphaDEX that has no "rare" artists | Requires both `By Artist/` and `By Year/`, but the Indexer only creates `By Year/` for rare artists (`music_indexer_api.py:668-672`), so the result is a false "invalid" | traced-in-code | Accept `By Artist/` alone, or derive the rule from the Indexer's constants |
| TOOL-23 | UX | `gui/workspaces/compression.py:186`, `:356-361` | Change library after Estimate; change bitrate during a run | The estimate and Start-enabled state persist for the previous library. The bitrate spin stays enabled during a run (edits have no effect) | confirmed-by-running (spin) / traced-in-code (stale estimate) | Clear the estimate and disable Start on any source/destination/bitrate change. Disable the spin in `_set_running` |

## Redesign insights (top 5)
1. **Risk is mixed into a "utilities" grab-bag.** Two of the most destructive actions in the app sit next to harmless text exports: File Cleanup (regex rename, no preview) and Pair Review (permanent `os.remove`). The redesign should give every file-mutating tool the same plan → preview → confirm → execute → Docs/ log shell (shared with Indexer/Duplicates/Sync). Read-only reports should live elsewhere.
2. **Duplicate tooling is fragmented across six surfaces:** Duplicates workspace, Similarity Inspector, Fuzzy Finder, Pair Review, Scan Engine and Bucketing POC. Fold Pair Review (as a pairwise review mode) and Scan Engine (as advanced settings) into Duplicates, retire or fix Fuzzy Finder (broken), and drop or label Bucketing POC as experimental.
3. **Playlist repair is triplicated** (File Cleanup, Pair Review, Playlists → Repair), and all three share a backend that silently no-ops in the common case (TOOL-03). Make it one playlist service that takes the library root and reports what changed, and surface the result in one place.
4. **Compression is mis-filed and thin on feedback.** It belongs with "library outputs" (and the Artist/Title and Codec exports), not under PLAYER. It needs cancel, reserved-folder exclusion, a live estimate that invalidates on change, and an error-first results view. The Opus Tester, which should verify its output, is broken.
5. **Help points at the wrong docs.** It links a stale Tkinter-era HTML and a missing file, and never the maintained `docs/` suite. Contextual per-page help that deep-links to `docs/features/*.md` would fit the redesign. A version string is also missing (and `gui_inventory.md` claims one exists).

## Harness needs (top 5)
1. **A safe QThread fixture.** Run workers synchronously via `.run()`, or wait on the built-in `QThread.finished`. Never let a test drop the only ref to an unparented running worker or delete a workspace mid-run (TOOL-04/05 abort the whole pytest process with exit 134).
2. **Global modal and shell stubs:** `QMessageBox.{question,warning,information,critical}`, `QFileDialog.{getExistingDirectory,getOpenFileName,getSaveFileName}`, `QDialog.exec` (Tools opens seven dialogs via `exec()`), `webbrowser.open`, `subprocess.Popen` (Open File/Folder) and `shutil.which`.
3. **Synthetic media factory:** tiny tagged FLAC (with picture), MP3, M4A (with `covr`) and Opus generated via ffmpeg lavfi + mutagen, in `tmp_path`. Include a fixture library with reserved folders (`Quarantine/`, `Docs/`, `Playlists/` with an `.m3u` referencing a file in a single album folder) to catch TOOL-01/02/03/14.
4. **Theme switching helper** that applies a light theme (e.g. `pearl`) and asserts that no widget in these pages keeps a hard-coded stylesheet colour (TOOL-15/16). Also a timer audit (count active `QTimer`s on hidden pages, TOOL-18).
5. **objectNames on Tools controls.** The diagnostic buttons have no text or objectName (the label is a child `QLabel`), so tests must call private slots. The harness author should request objectNames for each tile's run button, the diag buttons and the Compression controls.
