# Duplicate detection & similarity — page intent map
_Scope: `gui/workspaces/duplicates.py`, `gui/workspaces/similarity.py`, `gui/dialogs/fuzzy_dupe_dialog.py`,
`gui/dialogs/pair_review_dialog.py`, `gui/dialogs/scan_engine_dialog.py`, `gui/dialogs/bucketing_poc_dialog.py`;
launchers in `gui/workspaces/tools.py` (Diagnostics tile); backend contracts in `duplicate_consolidation.py`,
`duplicate_consolidation_executor.py`, `duplicate_scan_engine.py`, `duplicate_bucketing_poc.py`,
`near_duplicate_detector.py`, `chromaprint_utils.py`, `fingerprint_cache.py`. Date: 2026-11-07._

> **Current state (2026-11-08, Liquid Glass R2).** This map was written on 2026-11-07, before the fix phase; the
> 21 `DUP-` bugs below are all fixed (see `docs/redesign/README.md`), and the area now sits on the Duplicates page
> as six sections: Find duplicates, Compare two files (formerly the Similarity page), Pair review, and the
> Experimental Fuzzy finder, Scan engine and Bucketing (formerly Tools → Diagnostics). What each section looks like
> now is in "Liquid Glass restyle (R2)" at the end, and control by control in `docs/gui_inventory.md` §6.5 and §7.5.

Probe scripts that back every "confirmed-by-running" claim below are in
`/tmp/claude-0/-home-user-Music-Indexer/4300de3c-51f6-599a-948e-c4bafb432735/scratchpad/dup/` (`probe_*.py`, with
ffmpeg-generated fixtures in `audio/` and `audio2/`). All ran offscreen with `HOME` set to a fresh temp dir and
QMessageBox / `webbrowser.open` patched.

## Summary
This area exists so the user can find copies of the same recording in their library, keep the best copy, and put the
rest in `Quarantine/`, after previewing and confirming the plan. It also explains individual verdicts ("why do these two
files match, or not?"). In the Qt app there is **one real user-facing flow, the Duplicates workspace** (sidebar
"Duplicates"). The Similarity workspace is a two-file diagnostic. The four dialogs under Tools → Diagnostics (Bucketing
POC, Scan Engine, Fuzzy Finder, Pair Review) are Tkinter-era proofs of concept that were ported. They overlap the main
flow, they don't feed it, and three of the four are broken outright.
**The main flow also doesn't work end to end today.** A fresh library never gets fingerprinted (DUP-01). When a plan
does exist, execution is blocked whenever any group carries a review flag, which is nearly always (DUP-04). The "Delete"
option actually quarantines (DUP-03). Changing the library or quitting while a run is in progress aborts the process
(DUP-02).

## Pages

### Duplicate Finder (`gui/workspaces/duplicates.py`, class `DuplicatesWorkspace`)
- **Intent:** fingerprint the current library, group same-recording files, pick a winner per group, preview the plan as
  HTML, then (after confirmation) quarantine the losers and apply the winner's metadata, artwork and playlist fixes.
- **When the user comes here / entry points:** sidebar key `duplicates` (group "CLEAN UP", label "Duplicates", icon 🔍).
  No other Qt page navigates here (`navigate_requested` is never emitted with `"duplicates"`). There is no keyboard
  shortcut. The legacy `gui/qt_launcher.py` "Duplicate Finder Launcher" tab is a static mock-up with hard-coded "Last
  scan: Today 2:14 PM • 36 groups found" text, and only the Tkinter app launches it. It is not part of this flow.
- **Primary flow:**
  1. Optionally adjust the three threshold spin boxes (exact 0.02 / near 0.1 / mixed-codec boost 0.03, all hard-coded,
     see DUP-15).
  2. **Scan Library** (`_on_scan`) starts `DupeBuildWorker`. Stage 1 walks the library, skipping reserved folders, and
     fingerprints each file through `Docs/.duplicate_fingerprints.db` plus `chromaprint_utils.fingerprint_fpcalc`
     (0–60 %). Stage 2 calls `duplicate_consolidation.build_consolidation_plan` (62–85 %). Stage 3 calls
     `export_consolidation_preview_html` and writes `Docs/duplicate_preview.html`.
  3. The groups table fills in (`#`, Winner, Losers, Review ⚑). Selecting a row shows a plain-text inspector listing the
     winner, each loser with its disposition, and the review flags.
  4. **Preview** (the preview step) opens `duplicate_preview.html` in the system browser. Nothing gates Execute on
     having opened it.
  5. Choose the "Quarantine duplicates (safe default)" or "Delete losers permanently" radio button.
  6. **Execute** shows a `QMessageBox.question` ("This will {quarantine|delete} duplicate losers…"). This is the
     **explicit confirmation**, and it is the only gate in the UI.
  7. `DupeExecuteWorker` calls `execute_consolidation_plan(plan, ExecutionConfig(...))`. The executor runs its own
     preflight checks (plan signature, `requires_review`, operation limit, library drift, deletion double-gate), then
     the 7-step order: playlists → artwork → metadata → losers → suffix cleanup. It writes
     `Docs/duplicate_execution_reports/consolidation_run_<ts>/reports/execution_report.{html,json}`, which opens
     automatically.
- **Reads / writes:**
  - **Writes:** `Docs/.duplicate_fingerprints.db` (fingerprint cache), `Docs/duplicate_preview.html`,
    `Docs/duplicate_execution_reports/...`, and `Quarantine/` (flattened).
  - **Stray write:** `<commonpath of tracks>/Docs/.duplicate_metadata_cache.json`, which can land *inside an album
    folder* (DUP-14).
  - **Never written:** `Docs/duplicate_preview.json`, the file Pair Review needs (DUP-09).
  - **Config:** reads none. The Settings drawer's `near_duplicate_threshold` / `exact_duplicate_threshold` are ignored.
  - **Signals out:** `log_message`, `status_changed` (hard-coded hex colours).
- **Backend dependencies:** `fingerprint_cache.ensure_fingerprint_cache / get_cached_fingerprint_metadata /
  store_fingerprint`, `chromaprint_utils.fingerprint_fpcalc`, `utils.audio_metadata_reader.read_tags`,
  `duplicate_consolidation.build_consolidation_plan / export_consolidation_preview_html` (all on `DupeBuildWorker`'s
  thread), and `duplicate_consolidation_executor.execute_consolidation_plan` (on `DupeExecuteWorker`'s thread; it
  spawns its own ThreadPoolExecutors).
- **State owned:**
  - `_groups` (the `GroupPlan` list), `_worker` (its `.plan` is the authoritative in-memory plan), `_exec_worker`,
    `_preview_html_path`.
  - All of it lives until the next scan or a library change. `_on_library_changed` clears the widgets and sets both
    worker references to `None`, which crashes if a worker is running (DUP-02). It does not reset `_groups`.
- **Async model:**
  - **Workers:** two `QThread` subclasses with no parent. Each redefines `finished` with its own signature, which
    shadows `QThread.finished`.
  - **Progress:** the scan reports coarse progress (no updates during Stage 2). Execute reports none: the bar is reset
    to 0 and jumps to 100 at the end.
  - **Cancel:** sets a `threading.Event`, honoured between files in Stage 1. It isn't passed to
    `build_consolidation_plan`, so Stage 2 can't be cancelled. For the executor it is honoured between operations.
  - **Double-click:** Scan and Execute disable themselves immediately, so double-clicking is safe.
  - **After Execute:** the Execute button is re-enabled on the now-stale plan. The executor's drift check then refuses
    with "Execution completed with errors." (DUP-13).
  - **Navigating away:** fine; the workspace stays alive in the stack.
  - **Library change or app quit mid-run:** the process aborts with "QThread: Destroyed while thread is still running"
    (confirmed). `AlphaDEXWindow` has no `closeEvent` and no worker shutdown.
- **Motion & visuals:**
  - No animations.
  - Cards get `card_shadow` effects from the base class, refreshed on theme change.
  - The workflow stepper ("1. Scan & fingerprint → 2. Review groups → 3. Execute plan") is static: `stepActive` is set
    on step 1 once and never moves.
  - Inline styles are monospace font only (`duplicates.py:442,458`), so they're theme-safe.
  - Emoji glyphs in button labels (🔍 📄 ▶ 📊 ✕).
- **Decorative / dead controls:**
  - Per-group **Disposition combo** (`duplicates.py:432-435`): only enabled/disabled in `_on_group_selected`. It has no
    change handler and is never read. Confirmed: setting it to "Retain" and executing still quarantined that group's
    loser. The plan's `loser_disposition` is always `"quarantine"` (`duplicate_consolidation.py:3119`) and the executor
    uses that value (`duplicate_consolidation_executor.py:1349`).
  - **"Delete losers permanently" radio** is effectively decorative and mislabelled (DUP-03): it maps to
    `retain_losers=True`, which the executor only consults when a disposition is invalid. Losers are quarantined, while
    the confirm text says "delete… cannot be undone".
  - **"Update playlists after execution" checkbox** (`duplicates.py:333`): never read. Playlist rewrites in
    `plan.playlist_rewrites` run regardless.
  - **Workflow stepper:** static, as described above.
  - **"Open Report":** opens the *preview*, not the execution report (DUP-12).
- **Overlaps:**
  - Fuzzy Finder, Scan Engine and Bucketing POC each re-implement "find duplicates in this library" with different
    engines.
  - Pair Review re-implements "act on a duplicate" with its own unconfirmed `os.remove`.
  - Library Sync uses the same fingerprint/distance machinery.
  - The legacy Tkinter app (`main_gui.py`) is the only thing that writes the JSON preview and populates
    `.duplicate_fingerprints.db`.
- **Redesign notes:**
  - **Review isn't really modelled.** The workspace promises "Review groups", but the executor refuses any plan with
    review flags and there is no override, no per-group accept, and no way to clear a flag. Review-flag reasons such as
    "No artwork available" and "Placeholder metadata" fire on ordinary files, so in practice Execute never moves
    anything. The redesign needs a real review stage:
    - per-group accept / skip / retain,
    - an "I've reviewed these" override that maps to `allow_review_required`,
    - a disposition per group that the executor actually reads (`GroupPlan.loser_disposition` already carries it),
    - an explicit, double-confirmed delete path that sets `allow_deletion` + `confirm_deletion`, or no delete path at
      all.
  - **Preview should be in-app and gating.** Show a summary table and diff in the app, keep the HTML as an export, and
    enable Execute only after the preview has been viewed for *this* plan signature.
  - **Surface preflight refusals.** The reason is written only into the report JSON. The UI says "Execution completed
    with errors."
  - **Explain decisions in place.** Fold the Similarity Inspector into the group inspector ("why is this a duplicate?"
    for any two members) instead of keeping it as a separate page.
  - **Read thresholds from config** so the Settings drawer and this page agree. Add a "reset to defaults" control.
  - **Check fpcalc before scanning.** Today a missing fpcalc produces one log line per file, then "No fingerprinted
    tracks found." (confirmed). Use a pre-flight banner with an install hint.
  - **Progress and cancel:** add progress for execution, and pass the cancel event into `build_consolidation_plan`.
- **Testability seams:**
  - **Patch:**
    - `QMessageBox.question/warning/information`
    - `webbrowser.open`
    - `chromaprint_utils.fingerprint_fpcalc` (patch the module attribute; `_gather_tracks_for_plan` imports the module
      inside the function)
  - **Fixtures:** small ffmpeg-generated FLAC/MP3 files work well (see `scratchpad/dup/audio`).
  - **Waiting:** wait on the worker's custom `finished(bool, str)` signal, or poll `_scan_btn.isEnabled()`.
  - **Plan control:** `ws._worker.plan` exposes the `ConsolidationPlan`, so tests can clear `review_flags` and call
    `refresh_plan_signature()` to drive Execute past the review gate.
  - **objectNames:** `headerCard`, `workflowStepper`, `actionCard`, `dangerBtn` (Execute), `primaryBtn` (Scan),
    `statusHint`. Every other control is reachable only through private attributes (`_scan_btn`, `_preview_btn`,
    `_execute_btn`, `_report_btn`, `_cancel_btn`, `_quarantine_rb`, `_delete_rb`, `_disposition_combo`,
    `_groups_table`, `_group_details`, `_log_area`, `_prog_bar`, `_prog_status`). Ask for stable objectNames in the
    redesign.
  - **Harness fixture:** something that drains and joins any running workspace QThreads at teardown. Without it, a
    failing test aborts the whole pytest process (DUP-02).

### Similarity Inspector (`gui/workspaces/similarity.py`, class `SimilarityWorkspace`)
- **Intent:** pick two audio files and see their fingerprint distance, the thresholds that apply, and the verdict, to
  understand why the Duplicate Finder did or didn't group them.
- **When the user comes here / entry points:** sidebar key `similarity` (label "Similarity", ⚖). Nothing links here from
  the Duplicates groups table.
- **Primary flow:** Browse… A, Browse… B (`QFileDialog.getOpenFileName`), optionally open "Advanced Overrides", then
  **Run Comparison**. `SimilarityWorker` gets one fingerprint per file and the text report appears. Read-only: nothing to
  preview or confirm.
- **Reads / writes:**
  - Tries to read a cache at `Path(a).parent.parent/"Docs"/".soundvault.db"`. That path is wrong in two ways: for an
    `Artist/Album/track` layout it points at `Artist/Docs`, and the Duplicate Finder's cache is
    `.duplicate_fingerprints.db`. The cache is therefore effectively never hit.
  - Falls back to `acoustid.fingerprint_file` (libchromaprint through pyacoustid, else fpcalc).
  - Writes nothing. Reads no config.
- **Backend dependencies:** `near_duplicate_detector.fingerprint_distance`, `fingerprint_cache.get_cached_fingerprint`,
  `acoustid.fingerprint_file`, all on the worker thread.
- **State owned:** the two path fields and the report text. Nothing reacts to a library change.
- **Async model:**
  - One unparented `QThread`. Run is disabled while it runs, so double-click is safe. No cancel, no progress.
  - `_on_finished` drops `self._worker` while `run()` may still be returning. I could not reproduce a crash in 300
    runs, so this is suspected only (DUP-19).
  - Clear doesn't stop a running worker; the late result overwrites the cleared view.
- **Motion & visuals:** no animations. Monospace inline style only.
- **Decorative / dead controls:**
  - "Trim leading/trailing silence", "Fingerprint offset (ms)", "Fingerprint duration (ms)", "Silence threshold (dB)"
    and "Silence min length (ms)" (`similarity.py:136-142`) are never read.
  - The "Advanced Overrides" group's checkbox does nothing either. Unchecked, it greys out the threshold fields, but
    their values are still used.
  - If any one threshold field has a typo, all three silently fall back to defaults (`similarity.py:208-209`).
- **Overlaps:** same job as the Duplicates group inspector and as `build_duplicate_pair_report` (the "Duplicate Pair
  Report" described in `docs/features/duplicate_finder.md`), which the Qt app doesn't expose at all. The verdict logic is
  re-implemented here and diverges from the backend: its own lossless set is `{.flac, .wav}`, versus
  `near_duplicate_detector.LOSSLESS_EXTS`.
- **Redesign notes:**
  - **Wrong verdicts today (DUP-05).** pyacoustid returns `bytes`, `fingerprint_distance` can only parse `str`, and any
    non-identical pair scores 1.0 "NOT A DUPLICATE". That includes a FLAC and its own MP3 transcode (confirmed).
  - **Reuse the backend.** Call `build_duplicate_pair_report` and render its step-by-step trace (metadata bucket,
    coarse keys, threshold applied).
  - **Fold it in.** Make this the "Explain" panel of the Duplicates inspector, keeping a standalone entry for arbitrary
    files.
- **Testability seams:** patch `QFileDialog.getOpenFileName`, `acoustid.fingerprint_file` and
  `QMessageBox.warning`. Set `_file_a` / `_file_b` text directly, click `_run_btn`, and wait for `_worker is None` or
  the `finished` signal. Assert on `_results_text.toPlainText()`.

### Fuzzy Duplicate Finder (`gui/dialogs/fuzzy_dupe_dialog.py`, class `FuzzyDupeDialog`)
- **Intent (as designed):** a metadata word-overlap search, optionally filtered by fingerprint, that produces candidate
  pairs and sends them to Pair Review.
- **When the user comes here / entry points:** Tools → Diagnostics tile → "Fuzzy Finder…"
  (`tools.py:1397`, `dlg.exec()`).
- **Primary flow:** Run Scan → `FuzzyDupeWorker` → `find_near_duplicates(library_path, log_callback=…)`. It **always
  fails immediately** with "find_near_duplicates() missing 2 required positional arguments: 'ext_priority' and
  'threshold'" (confirmed, DUP-07). Even with correct arguments it would read `result.groups` / `g.tracks` /
  `g.distance`, none of which exist (`NearDuplicateResult.review_groups`, `NearDuplicateGroup.winner/losers/
  max_distance`), so it would always report 0 pairs. Nothing to preview or confirm. It is read-only.
- **Reads / writes:** nothing. It would also need a `file_infos` dict, not a path.
- **Backend dependencies:** `near_duplicate_detector.find_near_duplicates`, called wrongly, on the worker thread.
- **State owned:** results table only.
- **Async model:** one unparented QThread. "Cancel" sets a flag that only suppresses log lines. Close (`reject`) during a
  run leaves the worker going. The dialog is parented to the Tools workspace and never deleted, so every open leaks one
  dialog (confirmed: 3 opens left 3 live children).
- **Motion & visuals:** inline `color: #64748b` on two labels (`:132,:155`), which is low contrast on dark themes.
- **Decorative / dead controls:**
  - "Search fields" checkboxes (not even stored in attributes).
  - "Ignore words in > N tracks".
  - The entire Fingerprint Filter group (exact/near/boost fields, "Include pairs without fingerprints", "Skip
    fingerprint check").
  - "Include non-matching FP pairs in review".
  - `min_word_len` / `min_shared` are packed into `params` but never used.
  - "Send Matches to Duplicate Pair Review" is a stub ("coming soon", `:240`).
  - The "Shared Words" column is always "—".
- **Overlaps:** the Duplicates workspace (fingerprint grouping) and Pair Review.
- **Redesign notes:** a proof of concept with no working path. **Remove it**, or rebuild it as a "metadata-similar but
  not fingerprint-matched" filter inside the Duplicates review list.
- **Testability seams:** construct with a library path, click `_run_btn`, wait for `_run_btn.isEnabled()`, and read
  `_prog_status.text()`. No modal dialogs on the failure path.

### Duplicate Pair Review (`gui/dialogs/pair_review_dialog.py`, class `PairReviewDialog`)
- **Intent (as designed):** step through duplicate pairs one at a time with cover art and tags side by side. Keep one,
  skip, swap the winner, prefer MP3 over M4A, or tag the pair for later.
- **When the user comes here / entry points:** Tools → Diagnostics → "Pair Review…" (`tools.py:1403`). The launcher
  calls `load_pairs()`, which reads `Docs/duplicate_pair_review.json` (snapshot) or `Docs/duplicate_preview.json`
  (plan). **Neither file is written by anything in the Qt app.** Only the Tkinter app writes them. A Qt-only user always
  gets "No duplicate preview found. Run Duplicate Finder first", and doing what it says doesn't help (confirmed,
  DUP-09).
- **Primary flow:** when one of those files does exist, the constructor raises `AttributeError: module
  'PySide6.QtWidgets' has no attribute 'QShortcut'` at `:355`, so the dialog **never opens** (confirmed, DUP-06). If that
  one-line import were fixed:
  1. The dialog shows pair N of M.
  2. **Yes**, the Return key, or **Enter pressed in the "Jump to" field** calls `os.remove()` on the "inferior" file
     immediately. There is no preview, no confirmation, no quarantine, and no `Docs/` log (confirmed with the shim,
     DUP-08).
  3. Pairs are built from the plan *plus* a library-wide filename match (`_collect_filename_pairs`: same stem minus a
     trailing " (n)", across **all** folders including `Quarantine/` and `Docs/`). Where no winner is set, "FLAC beats
     non-FLAC" decides. My probe deleted a different song from a different album that happened to share a file name.
- **Reads / writes:**
  - **Reads:** the two JSON files, tags and cover art (`utils.audio_metadata_reader.read_metadata`, on the **GUI
    thread**, per pair).
  - **Deletes:** audio files (`os.remove`).
  - **Appends:** `Docs/duplicate_pair_tags.txt`.
  - **Calls:** `playlist_generator.update_playlists`, whose errors are swallowed.
- **Backend dependencies:** `duplicate_consolidation.consolidation_plan_from_dict`, `utils.path_helpers.ensure_long_path`,
  `playlist_generator.update_playlists`. All synchronous on the GUI thread.
- **State owned:** `_pairs`, `_index`, and per-pair `manual_winner`. Never written back to the snapshot, so Switch / Prefer
  MP3 choices are lost on close.
- **Async model:** none. Metadata and cover reads block the UI on every navigation.
- **Motion & visuals:** hard-coded `#333` cover background and `#888` / `#22c55e` winner-badge colours (`:168,:220-226`),
  which ignore the theme.
- **Decorative / dead controls:** none beyond the fact that the whole dialog is unreachable.
- **Overlaps:** the review stage the Duplicates workspace is missing. This is the right *interaction idea* (side-by-side
  A/B with art and tags, keyboard-driven) attached to the wrong *action* (immediate hard delete).
- **Redesign notes:** **don't revive this as-is.**
  - Take its side-by-side compare UI into the Duplicates review stage, operating on `GroupPlan`s.
  - Decisions should edit the plan (winner override, per-group disposition) instead of deleting files.
  - Execution stays with the executor (quarantine plus report).
  - Remove the filename-only pairing, or restrict it to one folder and label it clearly.
  - Never bind Return to a destructive action.
- **Testability seams:** write a `Docs/duplicate_pair_review.json` fixture. Construction currently needs
  `QtWidgets.QShortcut = QtGui.QShortcut` patched in. Spy on `gui.dialogs.pair_review_dialog.os.remove`. Use
  `QTest.keyClick` for the shortcut tests.

### Duplicate Scan Engine (`gui/dialogs/scan_engine_dialog.py`, class `ScanEngineDialog`)
- **Intent:** run the experimental staged engine (audio headers → fingerprint LSH bands → chroma alignment
  verification) and store the results in SQLite.
- **When the user comes here / entry points:** Tools → Diagnostics → "Scan Engine…" (`tools.py:1385`).
- **Primary flow:** edit 10 numeric fields → Run Scan → `_ScanWorker` → `duplicate_scan_engine.run_duplicate_scan` →
  summary line in the log. Read-only on audio; it writes only `Docs/duplicate_scan.db`.
- **Reads / writes:**
  - Creates `Docs/` just by *opening* the dialog (`_default_db_path` calls `os.makedirs`).
  - Writes `Docs/duplicate_scan.db`, which **nothing in the app reads**: a dead end.
- **Backend dependencies:** `duplicate_scan_engine` (imports `librosa`, `numpy`, `chromaprint_utils`). The config
  dataclass is imported on the **GUI thread** inside `_run_scan`.
- **State owned:** field values only.
- **Async model:**
  - One unparented QThread. A re-entrancy guard (`isRunning`) prevents double runs. No cancel, no progress bar (log
    only).
  - Closing mid-run leaves it running, and the dialog leaks.
  - With `librosa` missing (it is listed in `requirements.txt` but absent here), clicking Run raises
    `ModuleNotFoundError` in the slot. Only `ValueError` is caught, so the user sees nothing (confirmed, DUP-10).
- **Motion & visuals:** none. Theme-safe (objectNames `logBox`, `primaryBtn`, `sectionSubtitle`).
- **Decorative / dead controls:** none in the form itself. The output is dead (no viewer).
- **Overlaps:** the Duplicates workspace, as a different engine for the same job.
- **Redesign notes:** a proof of concept. Remove it from the user UI, or keep it behind a developer flag. If its
  chroma-verification idea is valuable, add it as an optional "verify" stage in the main plan builder.
- **Testability seams:** patch `duplicate_scan_engine.run_duplicate_scan`. Check that `_status_lbl` is "Idle" and read
  `_log_box`.

### Duplicate Bucketing POC (`gui/dialogs/bucketing_poc_dialog.py`, class `BucketingPocDialog`)
- **Intent:** prototype of the metadata-bucket → artwork-merge → fingerprint grouping strategy. Writes an HTML report of
  the buckets and comparisons.
- **When the user comes here / entry points:** Tools → Diagnostics → "Bucketing POC…" (`tools.py:1379`).
- **Primary flow:** pick a folder → Run → `run_duplicate_bucketing_poc(folder)` → the "Open it now?" question → browser.
  Read-only on audio. Writes `Docs/duplicate_bucketing_poc_report.html` and `Docs/.duplicate_bucketing_poc_fps.db`.
- **Reads / writes:** as listed above. Also reads `load_config()` thresholds through the backend.
- **Backend dependencies:** `duplicate_bucketing_poc` (on the worker thread).
- **State owned:** none.
- **Async model:**
  - One unparented QThread. No cancel, no progress, no log callback, so all backend logging goes to stdout.
  - With fpcalc missing, the UI simply says "Completed", and the report has no fingerprints (confirmed).
  - Closing the dialog mid-run doesn't stop the worker. The "Open it now?" message box still pops up after the dialog
    is gone (confirmed, DUP-11).
  - The dialog leaks on every open.
- **Motion & visuals:** none. Theme-safe.
- **Decorative / dead controls:** none.
- **Overlaps:** the bucketing it prototypes already ships inside `build_consolidation_plan`
  (`_build_metadata_buckets`, `_merge_buckets_by_artwork`).
- **Redesign notes:** a proof of concept whose logic already lives in the production engine. **Remove it** from the UI.
  Expose its diagnostics through the main plan's preview (`GroupPlan.bucket_diagnostics` already exists).
- **Testability seams:** patch `duplicate_bucketing_poc.run_duplicate_bucketing_poc`, `QMessageBox.question` and
  `webbrowser.open`.

## Cross-page relationships
- **The real flow is Sidebar → Duplicates.** Its only outputs are files in `Docs/` and `Quarantine/`. No other page
  consumes its in-memory plan.
- **Duplicates → Pair Review is broken.** Pair Review expects `Docs/duplicate_preview.json` or
  `Docs/duplicate_pair_review.json`. The Qt Duplicates page writes only `duplicate_preview.html`. Only the legacy
  Tkinter app produces those JSON files, and the Qt dialog then crashes on construction anyway.
- **Fuzzy Finder → Pair Review is a stub button.**
- **Similarity is standalone.** It isn't linked from the group inspector and doesn't share the Duplicates cache: it looks
  for `.soundvault.db` in the wrong folder, while Duplicates uses `.duplicate_fingerprints.db`.
- **Settings drawer → Duplicates/Similarity: none.** The near/exact thresholds saved in Settings are read by neither
  page.
- **Fingerprint caches are fragmented.** `.duplicate_fingerprints.db` (Duplicates, legacy app), `.soundvault.db`
  (Tag Fixer, import, Library Sync engine), `.duplicate_bucketing_poc_fps.db` (Bucketing POC) and `duplicate_scan.db`
  (Scan Engine) all fingerprint the same files separately.
- **Library Sync** uses the same `fingerprint_distance`, so it inherits DUP-05/DUP-18. The vendored engine copy should
  be checked separately before a fix is assumed to reach it.
- **Playlists:** the executor rewrites playlists under `Playlists/` according to the plan, whatever the "Update
  playlists" checkbox says.
- **Recommended IA for the redesign:** one "Duplicates" destination with four stages:
  1. Scan (fpcalc pre-flight, thresholds from config).
  2. Review (group list + side-by-side A/B compare taken from Pair Review + "Explain match" taken from Similarity /
     `build_duplicate_pair_report`, with per-group keep / retain / quarantine and a winner override).
  3. Preview (in-app summary, HTML export).
  4. Execute (confirm sheet listing exactly what moves and what tags/artwork/playlists change; progress; cancel;
     report).

  Bucketing POC, Scan Engine and Fuzzy Finder leave the user UI.

## Bugs found
| ID | Severity | Location | Trigger | Effect | Confidence | Suggested fix |
|---|---|---|---|---|---|---|
| DUP-01 | wrong-result | `gui/workspaces/duplicates.py:58` | Scan any library whose files are not already in `Docs/.duplicate_fingerprints.db` (any fresh library in a Qt-only install), with fpcalc installed | `duration, fp = chromaprint_utils.fingerprint_fpcalc(path)`, but the function returns a single `str`. Every file logs "! Fingerprint failed: …: too many values to unpack (expected 2)", then "No fingerprinted tracks found." The main Duplicate Finder can never fingerprint a new file. | confirmed-by-running (fake `fpcalc` on PATH emitting real `-json` output) | `fp = chromaprint_utils.fingerprint_fpcalc(path)`. Get the duration elsewhere or pass `None` to `store_fingerprint`. Add a test that runs `_gather_tracks_for_plan` against the real function signature. |
| DUP-02 | crash | `gui/workspaces/duplicates.py:656-657` (also `gui/main_window.py`, no `closeEvent`) | Change library (top bar, Ctrl+O, Settings) or quit the app while a scan or execute is running | Worker references dropped → "QThread: Destroyed while thread is still running" → SIGABRT (exit 134). During Execute this kills the process part-way through moving files and tagging. | confirmed-by-running (library change and interpreter exit, both mid-scan) | On library change: cancel, then `wait()` (or keep the reference until `finished`). Give workers a parent or keep them in a registry. Add `AlphaDEXWindow.closeEvent` that cancels and joins every workspace worker, or blocks close while Execute runs. |
| DUP-03 | wrong-result | `gui/workspaces/duplicates.py:518,538` → `:231` | Select "Delete losers permanently", Execute, confirm | The confirm text says "This will delete duplicate losers… cannot be undone", but `quarantine=False` becomes `retain_losers=True`, which the executor ignores because the plan disposition is "quarantine". Losers are **quarantined**. Safe direction, but the UI misreports what happened, and a user can't predict the outcome. | confirmed-by-running (`probe_d`) | Drive the disposition through `GroupPlan.loser_disposition`. For delete, set `allow_deletion` + `confirm_deletion` behind a second, typed confirmation, or remove the delete option. |
| DUP-04 | wrong-result | `gui/workspaces/duplicates.py:226-242` | Execute any plan where a group has review flags. Flags such as "No artwork available to apply", "Placeholder metadata detected" and near-distance matches fire on ordinary files. | Executor preflight refuses ("Execution blocked: review-required groups present"). The UI only shows "Execution completed with errors.", and the reason is written only to the report JSON. The UI has no override, so in practice Execute does nothing. | confirmed-by-running (`probe_c`, both groups flagged on plain FLAC/MP3 pairs) | Add a real review step: per-group accept, plus an explicit "I reviewed these" that sets `allow_review_required=True` for the accepted groups. Log the preflight `RuntimeError` text into the workspace log. |
| DUP-05 | wrong-result | `gui/workspaces/similarity.py:37-38` + `near_duplicate_detector.py:56-73` | Compare any two non-byte-identical files in the Similarity Inspector, e.g. a FLAC and its own 64 kbps MP3 | `acoustid.fingerprint_file` returns `bytes`. `_parse_fp` calls `str.replace` / `bytes + str`, fails silently, so the distance is 1.0 and the verdict "NOT A DUPLICATE". | confirmed-by-running (`probe_i`: true raw-bit distance 0.0002, reported 1.000000) | Decode, and compare *raw* fingerprints (`chromaprint.decode_fingerprint` / `fpcalc -raw`), or call `build_duplicate_pair_report` so it matches the backend. |
| DUP-06 | crash | `gui/dialogs/pair_review_dialog.py:355-358` | Tools → Pair Review… when a snapshot or preview JSON exists | `QtWidgets.QShortcut` doesn't exist in Qt6 (it moved to `QtGui`). The constructor raises `AttributeError` and the dialog never opens; the exception goes to stderr only. | confirmed-by-running (`probe_n`) | Use `QtGui.QShortcut`, but only together with the DUP-08 fix. Otherwise this unmasks a data-loss path. |
| DUP-07 | wrong-result | `gui/dialogs/fuzzy_dupe_dialog.py:32-41` | Fuzzy Finder → Run Scan | `find_near_duplicates(path, log_callback=…)`: missing `ext_priority`/`threshold`, and the first argument must be a `file_infos` dict. Status shows the raw TypeError. Even with correct arguments it reads non-existent `result.groups`/`g.tracks`/`g.distance`, so it would always find 0 pairs. | confirmed-by-running (`probe_j`) | Remove the dialog (recommended) or rebuild it on `NearDuplicateResult.review_groups`. |
| DUP-08 | data-loss | `gui/dialogs/pair_review_dialog.py:355,484-500,34-57` | (Latent behind DUP-06; reproduced with a QShortcut shim.) Press Enter in the "Jump to" field, press Enter with focus on "No — skip", or click Yes | `os.remove()` on the "inferior" file straight away. No confirmation, no quarantine, no `Docs/` log. Return is bound to a dialog-wide QShortcut, so Enter in the jump box both jumps and deletes. Filename pairing spans the whole library, and "FLAC beats non-FLAC" picks the loser, so a *different song* from another album with the same file name was deleted. | confirmed-by-running (`probe_l`: Enter in "Jump to" deleted `Other/AlbumX/t440.mp3`, a different recording) | Never `os.remove` from this dialog. Record decisions into the plan and let the executor quarantine. Drop the Return shortcut. Limit filename pairing to the same folder, or drop it. |
| DUP-09 | wrong-result | `gui/dialogs/pair_review_dialog.py:16-17,144-149`; `gui/workspaces/tools.py:1407-1413` | Qt-only user runs Duplicates, then opens Pair Review | `load_pairs` needs `Docs/duplicate_preview.json` / `duplicate_pair_review.json`, which only `main_gui.py` writes. The user is told "Run Duplicate Finder first", which never helps. | confirmed-by-running (`probe_k`/`probe_n`) | Have the Duplicates worker write the plan JSON (`export_consolidation_preview`), or pass the in-memory plan, or merge Pair Review into the Duplicates review stage. |
| DUP-10 | UX | `gui/dialogs/scan_engine_dialog.py:180-200` | Scan Engine → Run Scan with `librosa` unavailable (in `requirements.txt`, absent here) | `ModuleNotFoundError` is raised in the GUI-thread slot (only `ValueError` is caught). Nothing happens on screen; the traceback goes to stderr. | confirmed-by-running (`probe_m`) | Catch `ImportError` and show it. Import inside the worker. Its output DB has no consumer, so consider removing the dialog. |
| DUP-11 | stability | `gui/dialogs/bucketing_poc_dialog.py:342-356` (also `fuzzy_dupe_dialog.py:236`, `scan_engine_dialog.py:220`) | Close a Diagnostics dialog while its worker runs | The worker keeps running (no cancel). Bucketing pops "Open it now?" after the dialog has closed. Workers are unparented, and the dialogs are parented to ToolsWorkspace and never deleted. | confirmed-by-running (`probe_o`) | `closeEvent`/`reject`: cancel and wait, or refuse to close while running. `setAttribute(WA_DeleteOnClose)` with a guarded worker lifetime. |
| DUP-12 | wrong-result | `gui/workspaces/duplicates.py:547-559` | Click "Open Report" after an Execute | It globs `Docs/duplicate_execution_reports/*.html`, but reports live in `…/consolidation_run_<ts>/reports/execution_report.html`, so it falls through and opens the **preview** instead. | confirmed-by-running (`probe_c`) | Store `report_path` from `_on_execute_finished` and open that, or `rglob("execution_report.html")` sorted by mtime. |
| DUP-13 | UX | `gui/workspaces/duplicates.py:600-601` | Execute finishes | Execute is re-enabled on the same, now-stale plan. A second click passes the confirm and is refused by the drift check ("Library changed since preview"), shown only as "completed with errors". | confirmed-by-running (`probe_d`) | Disable Execute after a run until the next scan, or invalidate the plan. |
| DUP-14 | cosmetic | `gui/workspaces/duplicates.py:85-97` → `duplicate_consolidation.py:241-252` | Scan a library whose audio sits under one subfolder (e.g. a single artist) | Track dicts carry no `discovery.scan_roots`, so the metadata cache goes to `<commonpath>/Docs/.duplicate_metadata_cache.json`, creating a stray `Docs/` **inside an album folder**. | confirmed-by-running (`Art/Alb/Docs/.duplicate_metadata_cache.json`) | Add `"discovery": {"scan_roots": [library_path]}` to each track dict in `_gather_tracks_for_plan`. |
| DUP-15 | wrong-result | `gui/workspaces/duplicates.py:304-322`, `similarity.py:143-145` | Change the near/exact thresholds in Settings | Neither page reads `config` (`near_duplicate_threshold`, `exact_duplicate_threshold`, `mixed_codec_threshold_boost`). The defaults are hard-coded and the Settings values have no effect here. | traced-in-code | Initialise the spin boxes from `config.load_config()`, and refresh them on `settings_saved`. |
| DUP-16 | UX | `gui/workspaces/duplicates.py:432-435,333,281-284`; `similarity.py:136-142`; `fuzzy_dupe_dialog.py:86-119,167` | Use the per-group Disposition combo, "Update playlists after execution", Similarity's trim/offset/duration/silence fields, or Fuzzy's filter fields | No effect: these controls aren't read. The stepper never advances. Probe: combo set to "Retain", loser still quarantined. | confirmed-by-running (combo) / traced-in-code (rest) | Wire them up or remove them (see the redesign notes). |
| DUP-17 | UX | `gui/workspaces/duplicates.py:56-66` | Scan with fpcalc absent | No pre-flight check. One "! Fingerprint failed: … Required tool 'fpcalc' not found" log line per file, then "No fingerprinted tracks found." with an error chip. Graceful (no crash), but noisy and not actionable. | confirmed-by-running (`probe_a`) | Check `shutil.which("fpcalc")` before starting the worker and show one banner with install guidance. |
| DUP-18 | wrong-result | `chromaprint_utils.py:185,220-221` + `near_duplicate_detector.py:94-109` (used by `duplicate_consolidation.py:2602,2937`) | Any lossy/lossless transcode pair in the real engine | `fpcalc -json` (no `-raw`) returns the *compressed* base64 fingerprint, and the distance is a Hamming distance over the compressed bytes. One changed bit shifts everything after it: a FLAC vs its MP3 scored **0.227** compressed vs **0.0002** on raw bits, well above near + boost (0.13). Cross-format duplicates are likely missed. Backend, but it decides whether this page is useful at all. | confirmed-by-running (synthetic 25 s signal through libchromaprint, same output format as fpcalc) | Compare raw integer fingerprints (`fpcalc -raw` or `chromaprint.decode_fingerprint`) and re-tune thresholds. Port the change deliberately to `library_sync_indexer_engine/`. |
| DUP-19 | stability | `gui/workspaces/similarity.py:231` (same pattern `fuzzy_dupe_dialog.py:236`, `bucketing_poc_dialog.py:356,364`, `scan_engine_dialog.py:220,226`) | Worker's custom `finished` is delivered to the GUI before `run()` returns | The slot drops the only reference to a QThread that may still be running → possible "Destroyed while running" abort. | suspected (0/300 reproductions) | Keep the reference until the built-in `QThread.finished` (rename the custom signals so they don't shadow it), then `deleteLater()`. |
| DUP-20 | stability | `gui/workspaces/tools.py:1379-1418` | Open any Diagnostics dialog repeatedly | `Dialog(..., self)` + `exec()` without `deleteLater`: every open leaves a live dialog (plus its worker) parented to ToolsWorkspace. | confirmed-by-running (3 opens → 3 live children each) | `dlg.setAttribute(WA_DeleteOnClose)` or `dlg.deleteLater()` after `exec()`, with the worker guarded. |
| DUP-21 | cosmetic | `gui/dialogs/pair_review_dialog.py:168,220-226`; `fuzzy_dupe_dialog.py:132,155`; `gui/workspaces/duplicates.py:482,596` | Switch to a light or dark theme | Hard-coded `#333`, `#888`, `#22c55e`, `#64748b` inline colours ignore the theme. Status-chip hex colours are hard-coded in workspaces, and a cancelled scan shows "Idle" in error red. | traced-in-code | Use theme tokens or objectNames. Pass a semantic level (ok/warn/error), not a hex value, in `status_changed`. |

_Doc drift noticed:_ `docs/gui_inventory.md` §6.2 lists a "Library path entry", "Start Scan" and a `DupeScanWorker`.
The workspace actually has no path entry (it uses the global library), the button is "Scan Library", and the workers
are `DupeBuildWorker` / `DupeExecuteWorker`. It also lists Preview/Open Report/Cancel only implicitly.

## Liquid Glass restyle (R2)
_2026-11-08. Shared parts (since R3 in `gui/widgets/`: tone pills and `ToneBadge` in `pill.py`, `GlyphLabel` in
`glyphs.py`, the path label in `glass.py`, `TileGrid`, `StepPills`, the hero title and dialog section cards in
`glass_parts.py`); the one area-specific style left is in the `[area:duplicates]` block of `gui/themes/manager.py`.
Tests: `tests/ui/test_ui_dupes_glass.py`._

- **Find duplicates.** A hero card holds the library (elided path), **Find duplicates**, the three thresholds and the
  real-state stepper as a compact pill row (Scan · Review · Execute); Cancel and the progress bar only appear while a
  job runs. The results card starts as an empty state, then shows the plan as stat tiles (groups, duplicate files,
  space reclaimable, flagged for review), a reason line and the groups as glass rows: the kept copy, each loser with
  its disposition, format and bitrate pills, a Review pill. The per-group disposition combo sits under the list and
  the rows and space tile follow it at once. After an execution a second tile row shows quarantined, deleted, left
  in place for safety, not attempted and errors; `execution_counts` counts them from the same tally
  `summarize_execution` uses, so its sentences did not change. "Apply the plan" keeps the quarantine default, the
  permanent-delete choice (marked "Cannot be undone"), the always-on playlist rewrite, Preview and Report (glyphs, no
  emoji) and Execute (danger) with all its confirmations.
- **Compare two files.** Two glass file rows (Browse or drop a file), the verdict as a toned pill, the four distances
  as tiles (fingerprint, **overlap-only**, effective threshold, exact threshold), a one-paragraph explanation, and
  the full text report folded away. Advanced overrides fold away while off and name the Settings values in use.
- **Pair review and the prototypes.** One card each, titled with what the tool is for (the section tab names it), with
  the Experimental pill beside the title, the "not ready to test" note in a well, the points as rows and the Open
  button. Their dialogs use glass section cards, captions above fields, glyph buttons; Pair review's sides are cards
  with rounded cover tiles and a toned winner badge.
- **Widths.** Every section fits 900 × 600 and 1300 × 860 with no horizontal scroll (tested with scan results on
  screen).
- **Still open (unchanged by the restyle):** the IA recommendation above (fold Compare and Pair review into a review
  stage of Find duplicates, retire the prototypes) is the owner's decision; the group list still shows at most three
  losers per row ("+ N more", all listed in the details when selected).
