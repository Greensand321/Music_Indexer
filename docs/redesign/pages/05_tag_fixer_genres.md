# Metadata repair — Tag Fixer and Genres — page intent map
_Scope: `gui/workspaces/tag_fixer.py`, `gui/workspaces/genres.py`, with the backend they drive: `tag_fixer.py`, `controllers/tagfix_controller.py`, `controllers/normalize_controller.py`, `update_genres.py`, `metadata_service.py`, `plugins/*.py`, `utils/audio_metadata_reader.py`, `config.py`, the metadata tab of `gui/dialogs/settings_drawer.py`, `docs/features/tag_fixer.md`. Date: 2026-11-07._

## Summary
This area is where the user fixes what is written inside the files: artist, title, album and genre tags. The Qt app splits it across two sidebar entries under CLEAN UP. **Tag Fixer** identifies each track online (AcoustID fingerprint, with MusicBrainz enrichment nested inside it) and proposes per-field corrections. **Genre Normalizer** has two tabs: one fills genres from MusicBrainz community tags, the other rewrites a messy genre vocabulary into a canonical one using a mapping the user builds with an outside LLM. Both pages are meant to be preview-first. Only the mapping tab actually works that way. The Fill tab writes straight to files with no preview and no confirmation. Tag Fixer's Apply button currently raises an exception and writes nothing. All three tools share one hidden flaw: the tag reader returns only the **first** genre value. So all three tools can silently delete extra genres from multi-genre files, and the preview never shows that this will happen.

## Status after the Liquid Glass restyle (R2, 2026-11-08)

The map below is the 2026-11-07 audit, kept as it was found; for which of its bugs are
fixed see the bug-status table in `docs/redesign/README.md`. The area now lives on one **Metadata** page with four
sections (Tag Fixer, Fill genres, Normalize genres, Undo changes); "Genre Normalizer" is
no longer a page or a title. In the Liquid Glass look (see `docs/gui_inventory.md` §6.6
for every control):

- **No repeated titles.** Each view has one muted purpose line, then one hero card with
  its main action, then a plan/review card. The "1 · 2 · 3 · 4" step numbers of the
  mapping view are gone (they were decoration; the cards' order says the same).
- **Glass item views** (`gui/workspaces/metadata_views.py`, Metadata only): rounded rows,
  whole-row hover/selection, the value that would be written on an accent-tinted chip,
  current values muted, tools and "drop" values as pills. Colours are painted from the
  theme at paint time; the panels set no per-item colours (the Undo page's warning and
  muted brushes used to go stale on a theme switch).
- **Counts are `StatRow`s; empty results are `EmptyState`s**; buttons carry painted glyphs.
- **Tag Fixer scope**: the hero card shows what a scan looks at and can narrow it to one
  folder inside the library (reserved folders and folders outside it are refused). The
  sources line links to Settings → Metadata services.
- **Normalize**: a live table beside the JSON editor shows the mapping; `["invalid"]` and
  `null` show as a **Drop** pill, and a file whose genres would all be dropped shows
  "All dropped" as a pill in the plan.
- **Fill ↔ Normalize** cross-fade when the page switches between them (one panel serves
  both, so the page's own section fade never ran), and the view stack is sized to the
  view on show (the Fill view used to sit above an empty scroll the height of the
  Normalize view).
- Every section fits the 900×600 minimum window without sideways scrolling.

Behaviour is unchanged: preview-first, Apply writes exactly what is ticked/previewed, every
write is confirmed, holds the tag-write guard and the library-write lock, and is logged.

## Pages

### Tag Fixer (`gui/workspaces/tag_fixer.py`, class `TagFixerWorkspace`; workers `TagFixWorker`, `TagFixApplyWorker`)
- **Intent:** Look each track up online and let the user accept or reject proposed corrections to artist, title, album and genre, field by field, before anything is written.
- **When the user comes here / entry points:** Sidebar key `tag_fixer` ("Tag Fixer", CLEAN UP section). Shortcut Ctrl+5, from `_WORKSPACE_MAP` order. No other page navigates here, and this page emits no `navigate_requested`.
- **Primary flow:**
  1. Optionally untick fields under "Fix fields" (Title/Artist/Album/Genre). These are read **only at apply time**; the preview shows every field either way.
  2. Click **Scan for Tag Issues**. `TagFixWorker` runs `prepare_library`, which creates `Docs/.soundvault.db`, then `gather_records(show_all=True)`. That queries every plugin for every file.
  3. **Preview:** the proposals tree gets one row per changed field (File / Field / Current / Proposed / Score). Every row is checked by default.
  4. Untick the Dry run checkbox (it is checked by default and just blocks Apply with an info box).
  5. Click **Apply Selected**. **Confirm:** `QMessageBox.question("Write tags to N file(s)? This cannot be undone.")`.
  6. `TagFixApplyWorker` → `apply_proposals` → `apply_tag_proposals` → `update_tags` (mutagen, easy mode). Then the page automatically rescans the whole library.
  - **Today, step 5 raises `TypeError: unhashable type: 'FileRecord'` (META-01) and nothing is written.**
- **Reads / writes:**
  - Reads: `config.load_config()["metadata_api_key"]`, falling back to the hard-coded `tag_fixer.ACOUSTID_API_KEY`; env var `LASTFM_API_KEY`; `Docs/.genre_mapping.json` (loaded by `prepare_library`, then discarded unused).
  - Writes: `Docs/.soundvault.db` table `files` (status plus old/new values; this file is shared with Library Sync's fingerprint cache) and the audio files' tags.
  - Signals out: `log_message`, `status_changed`. Signals in: `set_library_path` only.
- **Backend dependencies:**
  - `controllers.tagfix_controller.prepare_library` / `gather_records` / `apply_proposals`, on worker QThreads.
  - `tag_fixer.PLUGINS` = `[AcoustIDService, LastfmPlugin, TestPlugin]`. `AcoustIDService.check_connection()` makes one `requests.get` per scan, and each plugin's `identify()` makes network calls per file, all on the worker thread. **No network call runs on the GUI thread.**
  - The settings dialog imports `tag_fixer` on the GUI thread; that is cheap, about 0.1 s.
- **State owned:**
  - `_proposals`, `_all_records`, `_db_path`, `_item_to_record` (keyed by `id(QTreeWidgetItem)`). They live until the next scan.
  - **Nothing resets on library change.** There is no `_on_library_changed` override, so old proposals stay on screen and Apply stays enabled (META-09).
- **Async model:**
  - One scan worker and one apply worker, both parentless `QThread` subclasses. Both define their own `finished = Signal(bool, str)`, which shadows `QThread.finished`.
  - Scan button is disabled during a scan, so a double-click is safe.
  - Cancel only sets a flag that is checked before and after `gather_records`. The real loop never sees it (META-10).
  - Progress is emitted as a file index into a 0–100 bar (META-11).
  - Apply has no cancel.
  - Navigating away is harmless, because the workspace stays alive in the stack. Closing the app mid-run aborts the process (META-12).
- **Motion & visuals:**
  - No animations. Card shadows come from `WorkspaceBase`.
  - Inline styles: `_prog_status` and `_summary_lbl` use a hard-coded `color: #64748b`; the log uses a Consolas font.
  - Section headings are plain `QLabel("Options")`, `"Tag Proposals"`, `"Log"` rather than `_make_card_title`, so they don't match Genres.
  - Status colours are passed as hex literals.
  - Buttons carry emoji glyphs.
- **Decorative / dead controls:**
  - *Dry run* checkbox: a second gate that only shows an info box. The scan never writes, so "dry run" has no meaning at scan time.
  - *Fix fields* checkboxes: they don't filter the preview, and they override the per-row checkboxes at apply time (META-07).
  - *Per-row checkboxes*: they really select **files**, not fields.
  - The *Score* column sorts as text, and "?" means "no match".
  - The `.genre_mapping.json` load in `prepare_library` is dead.
- **Overlaps:** The Genre field overlaps both Genre Normalizer tabs, and each of the three writes genres in a different format (META-05). Its online lookup also overlaps the Fill tab: both are MusicBrainz community tags, through different code paths.
- **Redesign notes:**
  - Split "matched with changes" from "no match found". Unmatched files are listed today as blank "proposed" values (META-06).
  - Make the review grid truly per-field, or per-file with field columns. Drop the Dry-run checkbox, since the confirm dialog is the gate.
  - Show *why* nothing matched: no fpcalc, bad key, offline.
  - Show a real percentage and an ETA, and make Cancel work.
  - After apply, refresh only the rows that were applied instead of rescanning the whole library online, and keep the apply log.
  - Respect the reserved folders.
- **Testability seams:**
  - Patch `tag_fixer.PLUGINS` or the plugin classes' `identify`, and `AcoustIDService.check_connection` (it does a real HTTP GET).
  - Patch `QMessageBox.question` / `.information` / `.warning`.
  - Wait on `ws._worker is None` / `ws._apply_worker is None`.
  - Attributes to latch onto: `_scan_btn`, `_apply_btn` (objectName `successBtn`), `_cancel_btn`, `_select_all_btn`, `_deselect_btn`, `_dry_run_cb`, `_field_cbs[label]`, `_table` (QTreeWidget, column 0 is the check state), `_summary_lbl`, `_prog_bar`, `_prog_status`, `_log_area`.
  - `crash_watcher.record_event` and `watcher.traced` run for real; they are harmless.
  - `TestPlugin` lets a file named `dummy_*.flac` produce a deterministic match with no network.
  - No widget has an objectName except the buttons.

### Genre Normalizer — tab "Fill from MusicBrainz" (`gui/workspaces/genres.py`, class `GenresWorkspace`, worker `GenreWorker`)
- **Intent:** Give tracks that have no genre (or one) MusicBrainz's three most popular community tags.
- **When the user comes here / entry points:** Sidebar key `genres` ("Genre Normalizer"), Ctrl+6. It is the first, default tab.
- **Primary flow:**
  1. Untick *Dry run*.
  2. Click **Run Genre Update**.
  3. The worker walks the library and calls `update_genres.process_file` per file: tag search by artist and title, top 3 tags, then `update_genre_tag`, with a 1 s sleep after every file.
  - **There is no preview step and no confirmation dialog** (META-03).
  - With Dry run ticked, the worker only prints "[dry-run] would process: <name>" for every file. It does no lookup, so it can't preview what would be written.
- **Reads / writes:** Writes audio files' `genre` tags. Writes **no** Docs/ log; the CLI version wrote `Docs/genre_update_log.txt`, the GUI only writes to the on-screen log. Uses the global user agent `update_genres` sets on import: `"SoundVaultGenreUpdater", "youremail@example.com"`.
- **Backend dependencies:** `update_genres.process_file` / `update_genre_tag`, and `musicbrainzngs.search_recordings`, all on the worker thread.
- **State owned:** Log text and progress only.
- **Async model:**
  - Cancel is checked between files and works within about 1 s.
  - The run button is disabled while running.
  - Parentless QThread, so the app aborts if closed mid-run (META-12).
  - It can run at the same time as the mapping tab's Apply on the same files (META-20).
- **Motion & visuals:** None. Uses the `statusHint` objectName for the themed hint colour. The log has an inline font.
- **Decorative / dead controls:** The promise "files that already carry two or more genre tags are left untouched" is false (META-04). Dry run is honest but useless as a preview.
- **Overlaps:** Overlaps the Tag Fixer's Genre field (both pull MusicBrainz tags). It also overlaps the Tag Fixer conceptually as an online lookup, but it skips files without artist/title instead of fingerprinting them.
- **Redesign notes:** Make it a lookup → preview table → confirm → write flow, using the same review grid as the Tag Fixer. Better still, fold it into the Tag Fixer as "genre only" mode. Have it append or merge, not overwrite, and say which one it does. Include `.opus` (META-25).
- **Testability seams:**
  - Patch `update_genres.musicbrainzngs.search_recordings` and set `update_genres.RATE_LIMIT_SECONDS = 0`.
  - Latch onto `_dry_run_cb`, `_run_btn`, `_cancel_btn`, `_prog_bar`, `_prog_status`, `_log_area`.
  - Wait on `ws._worker is None`.

### Genre Normalizer — tab "Normalize with a mapping" (same class; workers `_ScanWorker`, `_PlanWorker`, `_ApplyWorker`)
- **Intent:** Collapse the library's accumulated genre spellings into a canonical vocabulary the user chooses, preview-first.
- **When the user comes here / entry points:** Second tab of `genres`. A saved `Docs/.genre_mapping.json` is loaded automatically into the editor on library set.
- **Primary flow:**
  1. **Scan library**: `scan_raw_genres` lists distinct raw genres, split on `; , /`. Then *Copy list*.
  2. *Copy prompt* (`PROMPT_TEMPLATE`). The user pastes the prompt and the list into an LLM, then pastes the returned JSON into the editor. The JSON is validated live. The user can *Save* or *Load* the mapping.
  3. **Preview changes** (enabled only for valid JSON and a library): `plan_genre_normalization` fills a File / Current / After table and the counts. **This is the preview.**
  4. **Apply N changes** (danger button). It is enabled only while the editor text is identical to the text the plan was computed from. **Confirm** with `QMessageBox.question`, default No. The mapping is saved, then `apply_genre_changes` writes exactly `plan.changes`. Cancel works per file. Afterwards the plan is invalidated.
- **Reads / writes:** Reads tags through `read_tags`. Writes `Docs/.genre_mapping.json` and the files' genre tags (via `update_genres.update_genre_tag`, so importing it sets the global MusicBrainz user agent as a side effect). No Docs/ audit log, and the before-values are not kept anywhere (META-23).
- **Backend dependencies:** `normalize_controller.scan_raw_genres` / `plan_genre_normalization` / `apply_genre_changes` on worker QThreads. `load_mapping` / `save_mapping` / `clean_mapping` run on the GUI thread; they are small JSON operations. Discovery uses `tagfix_controller.discover_files`, i.e. `tag_fixer.find_files`.
- **State owned:**
  - `_raw_genres`, `_plan`, `_plan_mapping_text`.
  - On library change: everything is cleared, **including any unsaved mapping text** (META-24). In-flight workers are not invalidated (META-08).
- **Async model:**
  - Scan and preview are guarded against re-entry with `if worker is not None: return`.
  - Apply has a cancel `threading.Event`.
  - Progress is wired with lambda slots. I checked that these run on the GUI thread (queued), so they are fine.
  - Preview can be re-enabled during an apply, because `_update_preview_gate` only checks `_plan_worker`.
  - Parentless QThreads, so the app aborts if closed mid-run.
- **Motion & visuals:** No animations. Hints use `statusHint`, card titles use `cardTitle`. Inline styling is limited to the mapping editor's and log's font. The status colours are hex literals.
- **Decorative / dead controls:** None. All controls are read.
- **Overlaps:** The Tag Fixer's Genre field and the Fill tab, which write different shapes (see Cross-page). The legacy Tkinter Playlist Creator had this as its "Genre Normalizer" sub-panel, next to a "Year Assistant" that the Qt app has not ported.
- **Redesign notes:**
  - This is the best-shaped page in the area and the pattern the other two should copy: numbered steps, a plan bound to its inputs, a confirm gate, a plan of exactly what will be written.
  - It still needs to read all genre values (META-02), map the parts of combined strings (META-14), and show files it can't write before the user commits (META-13).
  - Add a per-row include/exclude toggle, an undo log, and a mapping-coverage view ("12 raw genres still unmapped").
- **Testability seams:**
  - Every collaborator is injectable at the controller level (`files=`, `read_tags=`, `writer=`). The workers call the defaults, though, so GUI tests should patch `nc._default_discover` / `_default_read_tags` / `_default_writer` or use real tiny files.
  - Patch `QMessageBox.question`.
  - Latch onto `_scan_btn`, `_copy_raw_btn`, `_raw_list`, `_copy_prompt_btn`, `_load_map_btn`, `_save_map_btn`, `_map_edit`, `_map_status`, `_preview_btn`, `_preview_status`, `_changes_table`, `_apply_btn`, `_apply_cancel_btn`, `_apply_status`, `_norm_bar`.
  - Wait on `_scan_worker` / `_plan_worker` / `_apply_worker is None`.
  - `QApplication.clipboard()` works offscreen.
  - Verified offscreen: scan → paste mapping → preview fills the table → editing the mapping disables Apply (scripts in the scratchpad `meta/qt_genres.py`).

## Cross-page relationships
- **Every place in the Qt app that writes genres** (I checked with grep; nothing else writes tags):
  1. Tag Fixer Genre field (`tag_fixer.update_tags`): unions the *first* existing genre with the proposed ones and writes a **single joined string** `"Rock; Electronic"`.
  2. Fill from MusicBrainz (`update_genres.process_file`): **replaces** the genre with a multi-value list of the top 3.
  3. Normalize with a mapping (`update_genres.update_genre_tag`): replaces it with a multi-value list.
- **The three tools disagree with each other.** The Tag Fixer's joined string reappears as a "change" in the Normalizer's next preview, because the Normalizer splits on `;`. The Fill tab overwrites what the Normalizer canonicalised.
- **The legacy "genre tag-filling" from the Playlist Creator does not exist in Qt playlists.** It moved to this sidebar entry. The Year Assistant was not ported.
- **Settings (Ctrl+,) → Metadata Services** writes `metadata_service` and `metadata_api_key`. Only the key is read, and always as the *AcoustID* key. The service choice is never read (META-19).
- **The MusicBrainz user agent is global process state.** It gets set as a side effect of importing `update_genres` (Genres tab), or of clicking Settings → Test with MusicBrainz selected. The Tag Fixer's MusicBrainz enrichment only works after one of those has happened (META-15).
- **Shared files:**
  - `Docs/.soundvault.db` is shared with Library Sync, which is already documented.
  - `Docs/.genre_mapping.json` is written by Genres and loaded but ignored by the Tag Fixer.
- **Neither page tells the other, or the Indexer/Library Sync, that tags changed.** No signal goes out, and there is no shared "library changed on disk" event. Duplicates and Clustering caches built from tags go stale silently.
- **Neither page navigates anywhere.** A natural link would be "Tag Fixer → open unmatched files in Normalizer/Fill", or "Indexer Manual Review → Tag Fixer for the files missing metadata".
- **Docs drift found:**
  - CLAUDE.md still says `plugins/acoustid_plugin.py` imports tkinter. It doesn't any more (ROADMAP already records the fix), so the Qt page does **not** pull in Tkinter.
  - `docs/features/tag_fixer.md` claims the preview "shows you only the fields that would actually change", that genre writes are "additive, never remove a genre", and that Fill skips files with 2+ genres. My runs proved all three false.
  - `docs/gui_inventory.md` §6.5 calls the proposals widget a `QTableWidget`, names the button "Scan Library", calls Apply "primary", and omits Cancel.

## Bugs found
| ID | Severity | Location | Trigger | Effect | Confidence | Suggested fix |
|---|---|---|---|---|---|---|
| META-01 | crash | `gui/workspaces/tag_fixer.py:279` | Tag Fixer: scan, untick Dry run, click Apply Selected | `selected_set.add(rec)` raises `TypeError: unhashable type: 'FileRecord'`, because the dataclass has `eq=True` and so no `__hash__`. The slot dies before the confirm dialog. Apply can never write anything | confirmed-by-running | Collect selected records by `id()` or path into a dict, or make it `@dataclass(eq=False)`. Add a pytest-qt test for Apply |
| META-02 | data-loss | `utils/audio_metadata_reader.py:134,169,210` (`_first_value` on genre); consumed by `controllers/normalize_controller.py:244,298` | Normalize with a mapping on a file with multi-value genres, e.g. FLAC `["Rock","Pop"]` or ID3 `["hip hop","Jazz"]` | `read_tags` returns only the first genre. Scan never lists "Pop"/"Jazz". The preview shows `Rock → Classic Rock`, and apply writes `["Classic Rock"]`, **silently deleting "Pop"**. Verified on disk for the FLAC and MP3 files | confirmed-by-running | Give the genre tools a reader that returns the full genre list (a `read_genres(path) -> list[str]`), and use it in scan, plan, the Fill guard and the Tag Fixer |
| META-03 | data-loss | `gui/workspaces/genres.py:76-83` | Fill from MusicBrainz with Dry run unticked → Run | Writes genres to every matched file straight away: **no preview, no confirm dialog**. Dry run does no lookups, so a real preview doesn't exist. No Docs/ log either. A probe confirmed `QMessageBox` was never called and the files were rewritten | confirmed-by-running | Split it into lookup (dry) → table of proposed genres → confirm → apply exactly the plan, as the mapping tab does. Write an audit log to Docs/ |
| META-04 | data-loss | `update_genres.py:108,188` | Fill tab on a file that already has 2+ genres | `extract_genre_list` goes through `read_tags`, which returns at most one value, so `len(existing) >= 2` never fires. `["Rock","Pop"]` was overwritten with MusicBrainz tags, breaking the "left untouched" promise in the UI and the docs | confirmed-by-running | Read every genre value (as META-02). Decide and state whether files with one genre get appended to or replaced |
| META-05 | data-loss | `tag_fixer.py:140-151` | Tag Fixer apply with the Genre field ticked (latent until META-01 is fixed; confirmed by patching hashability in the probe) | `old = proposal.old_genres` is the first value only, so the union drops the other existing genres and writes one joined string `"Rock; TestGenre"`. Even **unmatched** files (no proposed genre) get stripped down to their first genre, because `merged != existing` | confirmed-by-running | Merge against the full on-disk list (`existing`). Write a multi-value list, not a joined string. Skip the genre write when `new_genres` is empty |
| META-06 | wrong-result | `gui/workspaces/tag_fixer.py:63,343-371`; `tag_fixer.py:275-285` | Any scan where lookups fail (offline, no fpcalc, bad key) | `status="unmatched"` records count as proposals. Every tagged field is shown as "Current X → Proposed ''", checked by default, and "30 change(s) across 13 file(s)" also counts files that have no rows. Matched records also show blank proposals for fields the service didn't return. The preview doesn't match what apply writes | confirmed-by-running | Filter out `unmatched` (show them as a separate "No match" list). Only emit rows where `new_* is not None` |
| META-07 | wrong-result | `gui/workspaces/tag_fixer.py:273-289` | Untick the Genre row but leave the Artist row of the same file checked | The row checkboxes pick *files*; the fields come from the global "Fix fields" boxes, so the unticked genre was still written (seen in the patched probe) | confirmed-by-running | Carry the per-row field selection into apply (a `{record: {fields}}` map), or show one row per file with per-field checkboxes |
| META-08 | wrong-result | `gui/workspaces/genres.py:391-397,543-575,581` | Click Preview, then change library before it finishes | `_on_library_changed` clears the plan, but the in-flight `_PlanWorker` result is adopted for the new library afterwards. Rows show `../qlibA/...`, Apply is enabled, and the mapping-text guard passes (both are `""`). Apply then wrote the **old** library's files | confirmed-by-running | Tag each worker with a generation/library token and drop stale results. Disable library switching, or cancel, while a plan or apply is running |
| META-09 | wrong-result | `gui/workspaces/tag_fixer.py:106-117` (no `_on_library_changed`) | Scan library A, switch to library B | A's proposals stay on screen with Apply enabled. Applying writes to A's files and A's DB while B is the active library | confirmed-by-running | Override `_on_library_changed` to clear the table and state and cancel the workers |
| META-10 | stability | `gui/workspaces/tag_fixer.py:46-59`; `controllers/tagfix_controller.py:42-64` | Click Cancel during a scan | The flag is only checked before and after `gather_records`. The scan ran all 150 files (about 49 s) after Cancel at 0.3 s, then reported "Cancelled." and threw the results away. With real network timeouts this can mean hours | confirmed-by-running | Pass a `cancel_event`/`should_cancel` into `build_file_records` and check it per file |
| META-11 | UX | `gui/workspaces/tag_fixer.py:55`; `tag_fixer.py:214,273` | Scan of more than 100 files | `progress_callback(idx)` sends a file *index* to a 0–100 bar. It reaches 100% at file 100 and stays there; I saw values 1…150 | confirmed-by-running | Emit `idx*100//total`, or `(idx, total)` as the Genres workers do |
| META-12 | crash | `gui/workspaces/tag_fixer.py:257`, `gui/workspaces/genres.py:454,488,536,608`; `gui/main_window.py` (no `closeEvent`) | Close the app or destroy the workspace while any worker is running | `QThread: Destroyed while thread '' is still running`, then SIGABRT (exit 134), reproduced for both pages. An abort during apply could leave a half-saved file (suspected) | confirmed-by-running | Parent workers to the workspace. Add a shutdown hook that cancels and `wait()`s, or refuses to close during an apply |
| META-13 | wrong-result | `tag_fixer.py:131-139` (assignments outside the `try`); `update_genres.py:133-136` | `.wav` with ID3-in-RIFF tags | Tag Fixer: `audio["artist"]=…` raises `TypeError: … not a Frame instance`, which escapes `update_tags` and **aborts the whole apply batch** at that file. Normalizer: preview lists the WAV and apply always "fails", so it reappears in every preview | confirmed-by-running | Use a format-aware writer for WAV/AIFF ID3. Wrap each per-file write in try/except. Have the preview flag "can't write this format" |
| META-14 | wrong-result | `controllers/normalize_controller.py:158-163` | Mapping built from the scan list (split parts) applied to a combined tag like `"hip hop/rap"` or `"hip hop; 90s"` | The lookup uses the whole unsplit string, misses, and falls back to splitting **without mapping the parts**. Result: `['hip hop','rap']`, and the `"90s"` marked as invalid survives. That contradicts the scan step's own reasoning ("so the mapping sees each part") | confirmed-by-running | Look up the whole string first, then map each split part |
| META-15 | wrong-result | `metadata_service.py:34-45`; `plugins/acoustid_plugin.py:31-40` | Tag Fixer scan in a fresh session, before the Genres tab or Settings → Test (MusicBrainz) has run | `musicbrainzngs.get_recording_by_id` raises `UsageError` (no user agent set), which is swallowed, so album and genres are never filled from the enrichment. Whether it works depends on the order of earlier user actions | confirmed-by-running (UsageError) / traced-in-code (order dependence) | Call `set_useragent` from config in `query_acoustid` (or when plugins load). Surface enrichment errors in the log |
| META-16 | wrong-result | `tag_fixer.py:107-116` (used by Tag Fixer and Normalizer); `gui/workspaces/genres.py:70-73` | Any scan, preview or fill | Files under `Quarantine/`, `Not Sorted/`, `Trash/`, `Playlists/`, `Manual Review/` are proposed and rewritten (seen: `Quarantine/q.flac`, `Not Sorted/ns.flac`). `Not Sorted` is the user's exclusion zone | confirmed-by-running | Prune the reserved folders in a shared discovery helper |
| META-17 | UX | `gui/workspaces/tag_fixer.py:410-411,246` | Apply succeeds (even partly) | `_on_apply_finished` calls `_on_scan()`, which clears `_log_area`. Per-file "Failed to save …" lines disappear (the patched probe showed the log no longer contained "Updated tags"). It also re-queries every file online | confirmed-by-running | Keep the apply log (or write it to Docs/). Refresh only the applied rows |
| META-18 | UX | `plugins/acoustid_plugin.py:36-40`; `tag_fixer.py:196-204` | No fpcalc, an empty or invalid API key, or offline | `identify` swallows every exception, so every file becomes "unmatched" with no reason given. The offline warning says lookups "will be skipped" but they still run per file. Clearing the key in Settings saves `""`, which beats the built-in default | confirmed-by-running (no fpcalc: silent) / traced-in-code (empty key) | Do a pre-flight check (fpcalc present, key valid, reachable) with a banner. Log or count the error reasons. Treat `""` as "use default" |
| META-19 | UX | `gui/dialogs/settings_drawer.py:206-207`; `plugins/lastfm.py:10`; `plugins/acoustid_plugin.py:33` | User picks "Last.fm" or "MusicBrainz" in Settings and enters a key | `metadata_service` is never read. The key is always sent as the AcoustID client key, and Last.fm only reads the `LASTFM_API_KEY` env var. The settings look functional but do nothing | traced-in-code | Store a key per service. Have the Tag Fixer show which sources are active, or remove the selector |
| META-20 | stability | `gui/workspaces/genres.py:420-422,601-611`; both pages | Run Fill and mapping-Apply (or Tag Fixer apply) at the same time; or edit the mapping during Apply | Concurrent writers can hit the same files. Editing the mapping during Apply re-enables Preview, so a plan can be built mid-write | traced-in-code | One shared "tag write in progress" lock across the metadata pages. Gate Preview on `_apply_worker` too |
| META-21 | UX | `controllers/tagfix_controller.py:72-87` | Apply where some writes fail (read-only, unreadable `fake.mp3`) | Every *selected* path is marked `applied` in `Docs/.soundvault.db`, whether or not the write succeeded | traced-in-code | Have `apply_tag_proposals` return the paths that succeeded and mark only those |
| META-22 | cosmetic | `gui/workspaces/tag_fixer.py:192,218,134,200,226` | Theme switch (light theme) | Hard-coded `#64748b` labels, and plain `QLabel` headings instead of `cardTitle`, ignore the theme and don't match Genres | traced-in-code | Use the `statusHint` / `cardTitle` objectNames |
| META-23 | UX | `controllers/normalize_controller.py:315-346`; `tag_fixer.py:308-331` | Any genre or tag apply | No before-value log or undo file in Docs/, so in-place tag rewrites can't be reversed (the Tag Fixer DB keeps only the first genre) | traced-in-code | Write `Docs/tag_changes_<ts>.json` (path, field, before, after) and offer "Revert last apply" |
| META-24 | UX | `gui/workspaces/genres.py:395`; `:516` | Change library, or click "Load saved mapping", with unsaved editor text | The pasted mapping is wiped without a prompt (after the switch the editor was `''`) | confirmed-by-running | Ask first, or keep a draft per library |
| META-25 | UX | `update_genres.py:66` vs `tag_fixer.py:55` | Fill tab on a library with `.opus` files | The Fill tab skips `.opus`, although the Tag Fixer and the Normalizer handle it | traced-in-code | Share one supported-extension set |

(Also already documented and not repeated as a bug: `plugins/test_plugin.py` ships live, so `dummy_*` files get a fake perfect match.)
