# Library organization workflows (Indexer + Library Sync): page intent map
_Scope: `gui/workspaces/indexer.py`, `gui/workspaces/library_sync.py`, `gui/workspaces/base.py` (as it applies here), backend `music_indexer_api.py`, `indexer_control.py`, `dry_run_coordinator.py`, `library_sync.py`, `library_sync_review_state.py`, `library_sync_review_report.py`, `library_sync_types.py`, `validator.py`, `gui/main_window.py` (navigation, library change, shutdown), docs `docs/features/indexer.md` and `docs/features/library_sync.md`. Date: 2026-11-07._

_Method: I read all of the code above. I also drove both Qt workspaces offscreen (`QT_QPA_PLATFORM=offscreen`, `HOME` pointed at a fresh temp dir) against a synthetic library built with ffmpeg sine tones and mutagen tags. The library had messy folders, an untagged FLAC, a `Quarantine/` loser, a `Trash/` MP3 and two `cover.jpg` files; the incoming folder held one new track and two re-encoded duplicates. Library Sync needed two stubs. pydub is not installed in this sandbox, so I stubbed it like `tests/conftest.py` does. fpcalc is absent, so I replaced `fingerprint_generator.compute_fingerprints_parallel` with tag-derived fake fingerprints. Every execute run happened inside the temp dir. Probe scripts are in `/tmp/claude-0/-home-user-Music-Indexer/4300de3c-51f6-599a-948e-c4bafb432735/scratchpad/lib/`._

## Summary
This area is the "ORGANIZE" section of the sidebar (Ctrl+1, Ctrl+2). It contains the two workflows that physically reshape the user's library on disk.

- **Indexer** reorganizes a library in place. It renames files to `Artist_NN_Title.ext` and routes them into `By Artist/…` or `By Year/…` folders. It writes `Docs/MusicIndex.html` as the preview, then on request performs the moves, sweeps non-audio files into `Trash/` or `Docs/`, removes empty folders and regenerates playlists.
- **Library Sync** folds an incoming folder into an existing library. It fingerprints both sides, shows a match table where the user can override decisions per track, builds a copy/move plan with an HTML preview, then executes it with backups, an audit JSON and an executed-report HTML.

Both pages follow a "preview, then confirm, then execute" shape. The flaw is that what gets executed is not what was previewed: both backends recompute the plan, and Library Sync drops the user's flags at execution time. Both backends also ignore the reserved-folder contract (they scan `Quarantine/` and `Trash/`), and both pages can abort the whole process because of how they manage QThread lifetime.

## Pages

### Indexer (`gui/workspaces/indexer.py`, class `IndexerWorkspace`; worker `IndexerWorker`)
- **Intent:** Normalize the current library's folder structure and file names, with a reviewable HTML preview before anything moves.
- **When the user comes here / entry points:** Sidebar key `indexer` under ORGANIZE. It is the default page shown at startup (`main_window.py:128`) and is reachable with Ctrl+1. No other page navigates here, and the page emits no `navigate_requested`.
- **Primary flow:**
  1. The library path comes from the top bar or landing page (`set_library_path`). The page has no path field of its own.
  2. "Dry run (preview only)" is checked by default. Click **▶ Run Preview** to start `IndexerWorker(dry_run=True)`.
  3. **PREVIEW:** `api.run_full_indexer(dry_run_only=True)` calls `compute_moves_and_tag_index` and writes `Docs/indexer_log.txt`. `build_dry_run_html` then calls `compute_moves_and_tag_index` **a second time** and writes `Docs/MusicIndex.html`. A modal "Done" box appears and **📄 Open Report** opens the HTML in the external browser (`webbrowser.open`).
  4. Uncheck "Dry run" and click Run again. The button label still says "Run Preview" (LIB-16).
  5. **CONFIRM:** `QMessageBox.question` asks "This will move and rename files… Have you reviewed the preview report first?" (`indexer.py:271-279`). Nothing checks that a preview was ever run (LIB-24).
  6. Execute: `run_full_indexer(dry_run_only=False)` recomputes and rewrites the preview HTML. `apply_indexer_moves` then recomputes the plan a **third** time and executes that fresh plan (LIB-14). The execute step:
     - moves audio files;
     - sweeps non-audio files flat into `Trash/` or `Docs/` (LIB-01);
     - removes empty dirs;
     - optionally runs `generate_playlists`.
- **Reads / writes:**
  - Reads `<lib>/Music` if it exists, otherwise `<lib>`.
  - In preview, writes `Docs/MusicIndex.html` and `Docs/indexer_log.txt`, and also **creates** `Docs/`, `Not Sorted/` and `Manual Review/`. It creates the whole library path if it doesn't exist (LIB-15).
  - In execute, moves files, rewrites `Playlists/*.m3u` (one per old folder, e.g. `Playlists/Quarantine.m3u`), and calls `update_playlists`. It writes no execution log to `Docs/` (LIB-22).
  - Signals out: `log_message` (every backend line, duplicated into the LogDrawer) and `status_changed` with hard-coded hex colours.
  - No config keys are read or written.
- **Backend dependencies:** `music_indexer_api.run_full_indexer` → `compute_moves_and_tag_index` / `build_dry_run_html` / `apply_indexer_moves`. All of these run on the `IndexerWorker` QThread. Cancellation goes through the **process-global** `indexer_control.cancel_event`, which `duplicate_consolidation_executor` also uses as a fallback. `dry_run_coordinator.DryRunCoordinator` is vestigial: only section "A" is ever set.
- **State owned:** `_worker`, and `_report_path` for the life of the page. On library change (`_on_library_changed`, line 372) it resets the label, bars and log, and enables Open Report if `Docs/MusicIndex.html` already exists in the new library. It does **not** guard against a running worker: the run keeps going on the old path, then `_on_finished` looks up the report under the *new* path.
- **Async model:**
  - Runs on a `QThread` subclass with signals `progress(int,str)`, `log_line(str)` and `finished(bool,str)`. Note that `finished` **shadows** `QThread.finished`, so the built-in finished signal can't be used for cleanup.
  - Run is disabled while running, so a double-click is safe.
  - Cancel sets the global event, but the worker *clears* it at the top of `run()`, so an early cancel is lost (LIB-09). The second compute pass (`build_dry_run_html`) gets `progress_callback=None`, so it is not cancellable mid-pass.
  - `self._worker = None` in the slot can destroy a QThread that is still running and abort the process (LIB-07).
  - There is no close or navigate-away handling. Navigating away is harmless because pages are pre-created in a QStackedWidget, but quitting mid-run aborts (LIB-08).
  - Progress: Phase A always shows 0% because the scan passes `total=0`. Phase B fills during the metadata read. Phase C only fills during the *execute* move loop, even though it is labelled "Cross-album".
- **Motion & visuals:**
  - No animations.
  - Every card gets a `QGraphicsDropShadowEffect` (`base._make_card`). Shadows are re-created twice per theme change, once from `base._on_theme_changed_base` and once from `main_window._on_theme_changed`.
  - The status chip colours `#f59e0b`, `#22c55e` and `#ef4444` are hard-coded and ignore the theme.
  - Button labels use emoji glyphs.
  - The "workflow stepper" (`stepActive`/`stepInactive`) is static: step 1 is always highlighted and it never advances.
- **Decorative / dead controls:**
  - **Cross-album scan (Phase C)**, **Flush fingerprint cache** and **Max workers** are passed through but explicitly ignored. The `compute_moves_and_tag_index` docstring (`music_indexer_api.py:494`) says they are "retained for backward compatibility but are ignored".
  - The **Advanced** group's checkable state is never read. When unchecked it only disables its children, and `max_workers=4` is still passed.
  - The Phase C progress row is mislabelled.
  - The stepper is static.
- **Overlaps:**
  - Utilities → "Validate library" tile (`gui/workspaces/tools.py:1352`, `validator.validate_soundvault_structure`) checks the `By Artist/` + `By Year/` layout this page produces.
  - Playlist generation on execute overlaps the Playlists workspace.
  - Library Sync's plan builder runs the vendored copy of this same routing engine against the incoming folder.
- **Redesign notes:**
  - Preview and execute should be two distinct steps on one plan object: compute once, show it in-app, execute *that* plan. Replace the "Dry run" checkbox plus a single Run button with **Preview** and **Apply this preview** (disabled until a preview exists for the current library and is still fresh).
  - Show the preview in-app. `QtWebEngineWidgets` is already loaded by `gui/compat.py`, or a tree/table diff would work. The preview must also list the non-audio sweep, folder deletions and playlist rewrites, which it currently omits.
  - Remove or hide the three dead options.
  - Make progress honest (determinate scan phase) and drop the fake Phase C bar.
  - Surface the "0 audio files found" outcome instead of "Preview completed successfully".
- **Testability seams:**
  - Stub `QMessageBox.question/information/critical/warning` (the success path *always* shows a modal) and `webbrowser.open`.
  - Monkeypatch `music_indexer_api.run_full_indexer` for fast worker tests.
  - Reset `indexer_control.cancel_event` between tests.
  - Attributes to latch onto: `_dry_run_cb`, `_playlists_cb`, `_cross_album_cb`, `_flush_cache_cb`, `_max_workers_spin`, `_run_btn`, `_cancel_btn`, `_report_btn`, `_phase_bars` (list of (label, bar)), `_status_lbl`, `_log_area`, `_worker`.
  - No objectNames are unique to this page (it shares `headerCard`, `actionCard`, `workflowStepper`, `primaryBtn`).
  - Grab `ws._worker` *before* waiting, because the slot nulls it. Always `worker.wait()` in teardown or the process aborts (LIB-07/08).

### Library Sync (`gui/workspaces/library_sync.py`, class `LibrarySyncWorkspace`; workers `SyncScanWorker`, `SyncBuildWorker`, `SyncExecuteWorker`)
- **Intent:** Merge an incoming folder into the existing library: add what is new, skip duplicates, accept upgrades, with per-track human overrides and a preview before files move.
- **When the user comes here / entry points:** Sidebar key `library_sync` under ORGANIZE (Ctrl+2). The Existing Library field is overwritten with the global library path on every library change. No other page navigates here.
- **Primary flow:**
  1. Set **Existing Library** (pre-filled) and **Incoming Folder**. Both are free-text fields with Browse… buttons that open `QFileDialog.getExistingDirectory`. Optionally set the global threshold and per-format overrides.
  2. **🔍 Scan Both Libraries** → `SyncScanWorker` → `library_sync.compare_libraries(include_match_objects=True)`. This uses the vendored `library_sync_indexer_engine` and a fingerprint DB at `<existing>/Docs/.library_sync_fp.db`. The incoming table fills with Track / Status (NEW / EXACT_MATCH / COLLISION / LOW_CONFIDENCE) / Distance.
  3. Review step: right-click a row to open the `QMenu` with Copy / Replace / Clear flag / Add note (`QInputDialog.getMultiLineText`). Flags go into `ReviewStateStore`, and the Flag and Note columns update.
  4. **📋 Build Plan** → `SyncBuildWorker`:
     - `resolve_review_flags_to_paths(flags, match_results)` turns flags into `copy_only_paths` and `allowed_replacement_paths`;
     - `build_library_sync_preview` → `compute_library_sync_plan` **re-indexes the incoming folder** with the vendored engine, remaps destinations into `<existing>/Music` (or `<existing>`), and computes the items.
     - **PREVIEW:** writes `Docs/LibrarySyncPreview.html` and auto-opens it in the browser. **👁 Preview Plan** reopens it.
  5. **▶ Execute Plan**, then **CONFIRM** via `QMessageBox.question` ("This will transfer files… cannot be undone", lines 602-608). The dialog gives no counts and doesn't mention copy vs move.
  6. `SyncExecuteWorker` → `execute_library_sync_plan(plan)` → `execute_plan`. This **recomputes items from `plan.planned_moves()`** using `plan.copy_only` / `plan.allowed_replacements`, which were never filled in (LIB-03). It then copies or moves files, backs up replacements to `Docs/Backups/LibrarySync/`, and writes `Docs/LibrarySyncAudit.json`, `Docs/LibrarySyncExecuted.html` and optionally `Playlists/LibrarySync_Added_<stamp>.m3u8`.
  7. **📤 Export Report** (enabled after a scan) → `QFileDialog.getSaveFileName`. The format follows the extension: `.html` goes to `export_review_report_html`, `.csv` to `export_report(fmt="csv")`, anything else to JSON. The export is read-only.
- **Reads / writes:**
  - Config: **Save Session** writes `sync_existing`, `sync_incoming` and `sync_threshold` through `config.save_config`. Nothing ever reads them back.
  - Files: listed in the flow above. The scan and build also `mkdir -p <existing>/Docs`, which creates a typo'd existing path (LIB-15).
  - Signals out: `log_message` (only for page-level messages; worker lines go only to the page's own log area) and `status_changed` (hex colours).
- **Backend dependencies:**
  - `library_sync.compare_libraries`, `resolve_review_flags_to_paths`, `build_library_sync_preview`, `execute_library_sync_plan`;
  - `library_sync_review_state.ReviewStateStore` (built in `__init__` on the GUI thread);
  - `library_sync_review_report.export_report` / `export_review_report_html` (GUI thread, synchronous);
  - `config.FORMAT_PRIORITY` / `MIXED_CODEC_THRESHOLD_BOOST` / `DEFAULT_FP_THRESHOLDS`.
  - The scan, build and execute each run on their own QThread.
  - The Qt page does **not** depend on the Tkinter `library_sync_review.py`.
  - The vendored engine has its own `indexer_control`. Build cancellation works only because `compute_library_sync_plan` wraps the log/progress callbacks to raise the *root* `IndexCancelled`.
- **State owned:**
  - `_scan_result`, `_current_match_results` (MatchResult list), `_review_state_store` (flags, cleared on every scan without a warning, and `reconcile_best_matches` is never called) and `_build_worker` (kept alive because `_build_worker.plan` *is* the executable plan).
  - On library change only the Existing entry text changes. Match results, flags, the table and the built plan all survive, and **Execute stays enabled for a plan against the old library** (LIB-10).
- **Async model:**
  - Three QThread subclasses, each shadowing `finished` with a custom signature.
  - The scan disables Scan, Cancel-enables and Build-disables, but leaves **Execute enabled**. The build doesn't disable Scan, so a scan can start mid-build.
  - Cancel cancels whichever workers are running, but always writes "State: Cancelling…" to the *scan* state label.
  - The scan's `progress` signal is declared but never emitted or connected, so the side bars jump from 0 to 100. Execute has no progress at all.
  - `_worker = None` and `_exec_worker = None` in their slots can abort the process (LIB-07). Quitting mid-run aborts too (LIB-08).
  - Thousands of JSON log lines are emitted cross-thread during a scan (LIB-20).
- **Motion & visuals:**
  - No animations. Card shadows work as on the Indexer page.
  - The log area has an inline `setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")`. It only sets the font, but it is a per-widget stylesheet outside the theme.
  - The status chips use hard-coded hex colours, and a user cancel shows a red "Error" (LIB-21).
  - The stepper is static.
  - The context `QMenu()` is created without a parent.
- **Decorative / dead controls:**
  - **↺ Recompute Matches:** created disabled and never enabled. If it were enabled it would run a full rescan, not the "recompute without rescan" that `gui_inventory.md` claims.
  - **Preset name:** never read.
  - **💾 Save Session:** write-only.
  - **Existing Tracks** table: only ever `clear()`ed.
  - **Existing/Incoming progress bars:** fake.
  - **Match Inspector:** repeats the three visible columns and never shows the matched existing file, confidence, quality verdict or candidates.
  - **Replace** flag: has no effect on the plan (LIB-04). **Copy** flag: affects only the preview (LIB-03).
- **Overlaps:**
  - The plan builder *is* the Indexer engine (vendored copy) run on the incoming folder.
  - Matching overlaps the Duplicates and Similarity workspaces (same fingerprint and distance concepts, different engine and DB).
  - The transfer playlist overlaps Playlists.
  - The HTML review report draws three inert action buttons (documented).
- **Redesign notes:**
  - The core conflict: the page *presents* a match-and-review workflow, but the plan is built from an independent re-index of the incoming folder that ignores match status. Duplicates are therefore copied in, and Replace can't work. The redesign should make the reviewed match table *the* plan: one row per incoming track with its decided action (Add / Skip duplicate / Replace X / Copy-keep-original) and destination path, editable in place, with bulk selection.
  - Show the matched existing track side by side.
  - Fold the separate Build / Preview / Execute buttons into one plan view with a single "Apply N changes (copy|move)" confirm that states counts and mode.
  - Invalidate the plan when paths, mode, flags or library change.
  - Merge the threshold, override and preset controls into a validated "Matching strictness" panel.
  - Drop the empty Existing table, or turn it into the right half of a comparison.
  - Show the raw JSON log only in a debug drawer.
- **Testability seams:**
  - Stub `QMessageBox.*`, `QFileDialog.getExistingDirectory/getSaveFileName`, `QInputDialog.getMultiLineText`, `QMenu.exec` (the context menu is a modal `exec()`, so call `_flag_for_copy` / `_flag_for_replace` / `_flag_clear` / `_edit_note` directly) and `webbrowser.open`. Build and execute auto-open the browser.
  - Importing `library_sync` needs `pydub` and `acoustid`. Inject a pydub stub; note that `audio_norm.py:109` needs `AudioSegment` to be a mutable class, not `object`.
  - Fingerprinting needs fpcalc. Patch `library_sync.fingerprint_generator.compute_fingerprints_parallel` to return `(path, duration, fp)`.
  - Attributes: `_existing_entry`, `_incoming_entry`, `_threshold_entry`, `_overrides_text`, `_preset_entry`, `_scan_btn`, `_cancel_btn`, `_recompute_btn`, `_build_plan_btn`, `_preview_plan_btn`, `_transfer_toggle`, `_execute_plan_btn`, `_export_report_btn`, `_output_playlist_cb`, `_incoming_table` (row `UserRole` = `track_id`), `_existing_table`, `_review_state_store`, `_current_match_results`, `_build_worker.plan`, `_plan_status_lbl`, `_state_lbl`, `_match_summary_lbl`, `_log_area`.

## Cross-page relationships
- **Library path:** `MainWindow._on_library_changed` calls `set_library_path` on every pre-created workspace. The Indexer uses it directly. Library Sync copies it into the Existing field, and neither page resets in-flight or derived state (LIB-10, Indexer's report lookup on the new path).
- **Docs/ is a shared output bus:**
  - The Indexer writes `MusicIndex.html` and `indexer_log.txt`.
  - Library Sync writes `LibrarySyncPreview.html`, `LibrarySyncExecuted.html`, `LibrarySyncAudit.json`, `.library_sync_fp.db` and `Backups/LibrarySync/`.
  - Library Sync's existing-library scan walks `Docs/` (and `Quarantine/`, `Trash/`), so its own backups and the Duplicate Finder's quarantined losers become "existing" match candidates (LIB-06).
- **Duplicates → Indexer:** The Duplicate Finder quarantines losers into `Quarantine/`. The Indexer's next run re-routes them back into `By Year/…` and deletes the emptied `Quarantine/` folder (LIB-02). This silently undoes the Duplicates workflow.
- **Indexer ↔ Library Sync engines:** Library Sync's plan is the vendored Indexer engine run on the incoming folder, with output re-rooted into the library. The two engines already differ visibly: the root engine logs "1/4: Discovering audio files…", the vendored one "1/6". A routing fix in `music_indexer_api.py` does not reach Library Sync.
- **Playlists:** Indexer execution regenerates `Playlists/` and runs `update_playlists`. Library Sync writes a transfer `.m3u8` and runs `update_playlists` for replacements, which never happen in practice (LIB-04). The Playlists workspace consumes both.
- **Global cancel:** `indexer_control.cancel_event` is process-global. The Indexer sets it and clears it, and `duplicate_consolidation_executor.execute_consolidation_plan` falls back to it when no event is passed. The Qt Duplicates workspace does pass its own event.
- **No mutual exclusion:** Nothing stops an Indexer Execute and a Library Sync Execute from running against the same library at the same time.
- **Utilities → Validate library** (`tools.py`) is the only Qt consumer of `validator.py`. Neither page validates its input path.
- **Docs drift to fix after the code:**
  - `docs/features/indexer.md:38-40,173-178` says the Indexer skips Docs/Trash/Manual Review/Quarantine. It doesn't.
  - `indexer.md:122-131` promises a final execution log and Manual Review routing. Neither happens.
  - `docs/features/library_sync.md` promises "refusing to create duplicates", "your flags win", and warnings when flags can't be carried over. None of these hold.
  - `docs/gui_inventory.md:293-343` describes Phase C, Max workers, Preset, Recompute, Save Session and the Existing Tracks table as functional.

## Bugs found
| ID | Severity | Location | Trigger | Effect | Confidence | Suggested fix |
|---|---|---|---|---|---|---|
| LIB-01 | data-loss | `music_indexer_api.py:1190` | Indexer Execute on a library where two folders hold same-named non-audio files (`cover.jpg`, `folder.jpg`) | Leftovers are moved *flat* to `Trash/<name>` / `Docs/<name>` with `shutil.move`, which overwrites on POSIX. All but the last same-named file are silently destroyed. The sweep isn't in the preview HTML and isn't logged to Docs | confirmed-by-running (two `cover.jpg` became one `Trash/cover.jpg` = "jpeg-other") | Preserve relative paths under Trash/, de-duplicate names, list the sweep in the preview, log it to Docs |
| LIB-02 | data-loss | `music_indexer_api.py:529-536` (walk), `:1195` (skip_names) | Indexer Preview/Execute on a library with `Quarantine/` or `Trash/` (or `Manual Review/`, `Docs/`) audio | Only `Not Sorted` and `Playlists` are skipped. Quarantined duplicate losers and trashed tracks are planned and moved back into `By Year/…` (e.g. `Alpha_01_One (1).mp3`), then the emptied `Quarantine/` folder is deleted | confirmed-by-running | Prune all six reserved folders in every walk (scan, sweep, empty-dir passes). Add a regression test |
| LIB-03 | data-loss | `library_sync.py:902-921` (plan never stores `copy_only` / `allowed_replacements`) and `:1154-1167` (`execute_plan` recomputes items from them) | Move mode + right-click **Copy** on a track → Build Plan → Execute | The preview shows that track as `copy`, but execution recomputes with an empty `copy_only` and **moves** it out of the incoming folder ("3 moved, 0 copied"; audit says `move`). The executed plan is not the previewed plan | confirmed-by-running | Set `plan.copy_only` / `plan.allowed_replacements` / `plan.replacement_targets` in `compute_library_sync_plan`, or have `execute_plan` honour the stored `plan.items` |
| LIB-04 | wrong-result | `library_sync.py:836-840` vs `:790` | Right-click **Replace** on an EXACT_MATCH / upgrade row → Build Plan | `allowed_replacement_paths` holds the *existing match's* path, but items compare against the *computed destination* path. These are almost never equal (different name, folder or extension), so Replace has no effect even in the preview. `replacement_targets` is never filled, so the old file is never backed up or removed | confirmed-by-running (allowed = `Quarantine/dupe_loser.mp3`, item dst = `By Year/2001/Alpha_01_One.flac`, decision COPY) | Build replace items from the match (src → existing path or its slot) and fill `replacement_targets` |
| LIB-05 | wrong-result | `library_sync.py:845-921` (`compute_library_sync_plan`) | Any Build Plan after a scan that found matches | The plan ignores match results: it re-indexes the incoming folder and plans **every** file. EXACT_MATCH duplicates are copied into the library beside the existing copy (`other/b1.flac` + `By Year/2010/Beta_01_Solo.flac`). Contradicts "refusing to create duplicates" | confirmed-by-running | Filter or decide items from `MatchResult` status, quality verdict and flags (skip matched unless upgrade or flagged) |
| LIB-06 | wrong-result | `library_sync.py:248` (`_scan_folder` walk) | Scan when the existing library has `Quarantine/`, `Trash/` or `Docs/Backups/` audio | Reserved folders are scanned as "existing". Incoming tracks are matched to quarantined losers or backups (a1.flac matched `Quarantine/dupe_loser.mp3`) | confirmed-by-running | Prune reserved folders in the vendored scan (port deliberately per the two-engines rule) |
| LIB-07 | crash | `gui/workspaces/indexer.py:364`, `gui/workspaces/library_sync.py:742`, `:782` | Worker emits its custom `finished` and the slot runs before `run()` has returned. Masked on the Indexer success/error path by the modal QMessageBox shown first. Hit on Indexer **Cancel** and on **every Library Sync scan/execute completion** | `self._worker = None` drops the last reference to a still-running QThread: "QThread: Destroyed while thread '' is still running" and SIGABRT (exit 134) | confirmed-by-running (3/20 single runs abort; loops abort within 2-3 iterations, for both pages) | Rename the custom signal (it shadows `QThread.finished`), connect built-in `finished` → `deleteLater`, parent the worker, null the reference only after `QThread.finished` |
| LIB-08 | crash | `gui/main_window.py` (no `closeEvent`; `alpha_dex_gui.py:165`) | Quit or close the window while any Indexer or Library Sync worker runs | Interpreter teardown destroys the running QThread and the process aborts. During Execute, that kills the run mid-move with no audit trail | confirmed-by-running (`probe_close.py`, exit 134) | `closeEvent` asks to cancel, sets cancel, `wait()`s workers (or refuses to close during Execute) |
| LIB-09 | wrong-result | `gui/workspaces/indexer.py:45` | Click Cancel right after Run (before the thread reaches `cancel_event.clear()`, which includes the first-run import time) | The cancel is erased and the run completes, including a confirmed **Execute**. The UI shows "Cancelling…" then "completed successfully" | confirmed-by-running (3/3 immediate cancels ignored) | Clear the event in `_on_run` before `start()`, or give each worker its own event |
| LIB-10 | wrong-result | `gui/workspaces/library_sync.py:598-621`, `:832-833`, `:519-521`, `:424-431` | Build a plan, then (a) change library, (b) flip Copy/Move toggle, (c) edit paths or flags, or (d) start a re-scan | Execute stays enabled and runs the **old** plan: old library root, old transfer mode (toggle reads "Move Originals" while the plan copies, or the reverse), old flags | confirmed-by-running | Invalidate `plan` and disable Execute on any input change or scan start. Show the plan's mode and root in the confirm |
| LIB-11 | wrong-result | `gui/workspaces/library_sync.py:167-171` | Any Execute; Cancel during Execute; per-file failures | Reads `summary["executed_report_path"]` (backend key is `report_path`), so the executed report never opens. `summary["cancelled"]` and `summary["errors"]` are ignored, so cancelled or failed runs report "Execution complete — N moved" as success | confirmed-by-running (report not opened) / traced-in-code (cancel/errors) | Use `report_path`, branch on `cancelled`, surface error and skip counts |
| LIB-12 | UX | `music_indexer_api.py:494` (docstring), `gui/workspaces/indexer.py:147-170`, `:330` | Toggle Cross-album (Phase C), Flush cache or Max workers | No effect: the backend ignores all three. The Phase C bar actually tracks the execute move loop | traced-in-code | Remove the controls or implement them. Relabel the bars |
| LIB-13 | wrong-result | `music_indexer_api.py:607` | Untagged file (no artist or title) | `missing_core` can never be true because artist and title fall back to the filename. Manual Review routing is dead and untagged files are filed under `By Year/Unknown/` (docs say they go to Manual Review) | confirmed-by-running | Compute `missing_core` from the raw tags before the fallbacks |
| LIB-14 | UX | `music_indexer_api.py:1319`, `:1099`; `gui/workspaces/indexer.py:271` | Preview, then later Execute | The plan is computed twice per preview and three times per execute. Execute runs a *fresh* plan, not the reviewed one: if the library changed since the preview, unreviewed moves run. The log shows doubled lines, and the second pass has no progress or cancel checks | confirmed-by-running (duplicated log) / traced-in-code | Compute once, persist the plan, execute that plan, refuse it if stale |
| LIB-15 | UX | `gui/workspaces/indexer.py:57-58`; `gui/workspaces/library_sync.py:42-43`, `:95` | Library or existing path that doesn't exist (e.g. typed in Library Sync, or set via Settings) | The path tree is silently **created** (`Docs/`, `Not Sorted/`, `Manual Review/`) and the run reports success. In Library Sync everything shows as NEW and the plan copies all incoming files into the new folder | confirmed-by-running | Validate `is_dir()` (and incoming ≠ existing, not nested) before starting |
| LIB-16 | UX | `gui/workspaces/indexer.py:291` | Uncheck "Dry run" | The button still reads "▶ Run Preview" but executes (after the confirm). After an execute it keeps reading "Execute" until the next click | traced-in-code | Bind the label to `_dry_run_cb.toggled` |
| LIB-17 | UX | `gui/workspaces/library_sync.py:290-291`, `:547-558`, `:277`, `:381-387`, `:17`/`:529-531` | Look at the page | Decorative controls: Recompute is never enabled, Save Session is write-only (keys never loaded), Preset name is unread, the Existing Tracks table is never populated, and the scan side bars are fake (the `progress` signal is never emitted or connected). Execute has no progress | confirmed-by-running (recompute disabled, 0 existing rows, bars only 0→100) | Wire or remove each one |
| LIB-18 | UX | `gui/workspaces/library_sync.py:499-510` | Threshold "abc", or override line `flac=0.1` | Parse error is silently swallowed (`thresholds={}`, so defaults apply). Override keys without the leading dot never match an extension. No feedback | confirmed-by-running | Validate with inline errors, normalize keys to `.ext` |
| LIB-19 | wrong-result | `gui/workspaces/library_sync.py:653-679`; `library_sync_review_report.py:281` | Export Report as `.html` after flagging or noting tracks | The HTML report uses an empty `ReviewStateStore()`, so flags and notes are missing (JSON and CSV include them). Contradicts the button tooltip | confirmed-by-running | Pass the page's store into the HTML exporter |
| LIB-20 | UX | `library_sync.py:304-385` (JSON events) + `gui/workspaces/library_sync.py:38-40`, `:58-59` | Scan any library | The log area fills with raw JSON event lines (one or more per file: `cache_hit`, `fingerprint_progress`…), emitted cross-thread per line, which floods the GUI thread on big libraries. The summary message lists internal dict keys ("3 new_tracks … 3 match_objects") | confirmed-by-running | Throttle and humanize the log. Build the summary from `MatchStatus` counts |
| LIB-21 | cosmetic | `gui/workspaces/indexer.py:360`; `gui/workspaces/library_sync.py:740`, `:763` | User clicks Cancel | Status chip turns red "Error" | traced-in-code | Neutral "Cancelled" state |
| LIB-22 | UX | `music_indexer_api.py:1062-1263` | Indexer Execute | No record of the moves actually performed is written to `Docs/` (only the pre-move decision log). The run log lives only in the widget. This breaks CLAUDE.md's "log every action to Docs/" rule | confirmed-by-running (Docs held only `MusicIndex.html` and `indexer_log.txt`) | Write an executed-moves audit (JSON + HTML) like Library Sync's |
| LIB-23 | UX | `gui/workspaces/library_sync.py:512-514` | Re-scan after flagging | All flags are dropped with no warning. `ReviewStateStore.reconcile_best_matches` exists, and the docs promise warnings, but it is never called | traced-in-code | Reconcile and report, or confirm before discarding |
| LIB-24 | UX | `gui/workspaces/indexer.py:271-279` | Execute without ever running a preview | Allowed. The only gate is a "Have you reviewed…?" honour-system dialog | traced-in-code | Enable Apply only when a fresh preview exists for this library |

## Most important redesign insights
1. **"What you preview is what executes" is broken on both pages.** The fix is architectural, not cosmetic: one persisted plan object, rendered in-app, executed verbatim, and invalidated by any input change. The UI should be built around that plan view.
2. **Library Sync's review table and its plan are disconnected.** The plan comes from an independent re-index of the incoming folder, so match status, Replace flags and (at execution) Copy flags don't matter. A redesign should make the match table the plan (per-row action and destination, bulk edit, and the matched existing track visible side by side).
3. **The reserved-folder contract isn't enforced in the backends** (LIB-02, LIB-06). Until it is, any redesign that makes "Apply" easier makes the data loss more likely. Fix this before the UI work.
4. **About a third of the visible controls on these two pages do nothing.** These are Phase C, Flush, Max workers, Advanced, Recompute, Preset, Save Session, the Existing table, the fake bars and the static steppers. The redesign can drop most of them, which also makes the pages much simpler.
5. **Merge the two pages' shared "Plan" UI.** Both are "scan → plan tree with per-file source→destination → apply". A shared plan-review component (tree diff, counts, mode, filters, in-app HTML or native view) would serve the Indexer, Library Sync and probably the Duplicates executor too.

## Harness needs (for the pytest-qt builder)
1. A **worker-lifetime-safe wait helper**: capture `ws._worker` before waiting, wait on the custom `finished`, then `worker.wait()`. Teardown must join every QThread or the run aborts with SIGABRT (LIB-07/08). Consider a fixture that fails a test whenever "QThread: Destroyed while thread" appears on stderr.
2. **Modal suppression**: autouse patches for `QMessageBox.{question,information,warning,critical}` (with a configurable answer for `question`), `QFileDialog.getExistingDirectory/getSaveFileName`, `QInputDialog.getMultiLineText`, `QMenu.exec`, and `webbrowser.open`.
3. **Backend import and fingerprint shims for Library Sync**: a pydub stub with a *mutable* `AudioSegment` class (`audio_norm.py:109` assigns to it), and a patched `library_sync.fingerprint_generator.compute_fingerprints_parallel`. There is no fpcalc in CI.
4. **A tiny synthetic-library fixture** (ffmpeg sine tones + mutagen tags in `tmp_path`, `HOME=tmp_path`) that includes the reserved folders and same-named non-audio files. It should be able to assert on the filesystem tree after preview and after execute. Reset `indexer_control.cancel_event` between tests.
5. **Fast-path monkeypatches** for `music_indexer_api.run_full_indexer` and `library_sync.compare_libraries` / `build_library_sync_preview` / `execute_library_sync_plan`, so state-machine tests (button enablement, stale-plan invalidation, cancel, library change) run without I/O.
