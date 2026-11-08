# Contradicting updates to existing docs and code

_A running list, started 2026-11-07, of changes that contradict what existing documentation, code comments,
or the original design said or did. The owner asked for every such change to be recorded here so any of them
can be walked back later. Each entry: **what it was → what it is now → why**, and where to undo it._

Status legend: **Done** (in the code now) · **Pending owner** (decision needed; nothing changed yet) ·
**In progress** (being built on this branch).

---

## A. Behaviour that changed

### A1. Indexer no longer rewrites artist-tag casing during Execute — **Pending owner**
- **Was** (commit `07ef7f5`, "Canonicalize artist casing for routing and metadata"): Execute rewrote the
  *artist tag* of every file it moved whenever that file's primary artist differed from the library's most
  common spelling **by letter case only** (e.g. `AURORA LINES feat. Bob` → `Aurora Lines feat. Bob`). That was an
  intentional normalisation step.
- **Is**: removed in `394d24c` (independent review: it was not in the preview, the confirm text says "nothing
  else is touched", it collapsed multi-value artist tags to one value, and it failed on files open in a player).
  Folder names still use the canonical spelling.
- **Undo / options**: see "Indexer tag step" in the session report. It can be restored as a *previewed*,
  multi-value-safe, logged step.

### A2. Library Sync default match threshold 0.3 → 0.2 — **Done**
- **Was**: `CLAUDE.md` / `config.py` default 0.3 (0.35 for some formats), on a distance computed from
  *compressed* fingerprint bytes.
- **Is**: 0.2 on the *decoded* Chromaprint bit-error-rate scale. A saved `sync_threshold` equal to the old
  default (0.3 / 0.35) is migrated once to 0.2, with a log line and a `sync_threshold_scale` marker. Values ≥ 0.4
  show a warning.
- **Why**: on the new scale, the same recording in any codec scores ≈ 0.01 and different recordings ≥ 0.43, so
  0.3 put unrelated tracks in "review" while the old metric missed real duplicates.
- **Undo**: `config.DEFAULT_FP_THRESHOLDS`, `library_sync_indexer_engine/indexer_engine/config.py`, and the
  migration in `gui/workspaces/library_sync.py`.

### A3. Tag Fixer: a proposal replaces every value of a multi-value field — **Done**
- **Was**: the Tag Fixer read only the first value of artist/title/album. Writing a proposal kept only the new
  value, so `["Artist A", "Artist B"]` silently lost "Artist B", and the preview showed "Artist A → New Artist".
- **Is**: all values are read. The preview shows `Artist A; Artist B → New Artist`, and the write replaces the
  whole field with exactly what the preview shows. The change log records the full before-list. A proposal equal
  to the joined current values counts as "no change".
- **Undo**: `tag_fixer.py` (`_read_text_field_lists`, `planned_changes`). The alternative semantics would be
  "replace only the first value, keep the rest".

### A4. Rewritten playlist entries keep their own path style — **Done**
- **Was**: rewritten entries took the OS separator and an absolute or relative form at the writer's choice.
- **Is**: an entry keeps its separator style (`/` or `\`) and its relative or absolute form, on every OS; across
  drives it falls back to absolute (`utils/path_helpers.playlist_entry`). Unchanged entries are left
  byte-for-byte.

### A5. Duplicates "Delete" really deletes — **Done** (owner confirmed: keep)
- **Was**: the "Delete losers permanently" option quarantined files while its confirm said deletion "cannot be
  undone".
- **Is**: it deletes, after a confirm that lists every deletion plus a typed `DELETE`. Quarantine is the default.

### A6. Quitting while jobs run — **Done**
- **Was** (undocumented): quitting mid-job aborted the process.
- **Is**: only jobs that may be writing files ask before quitting. Read-only scans are cancelled silently. After
  20 s a job that ignores cancel offers "Quit anyway".

### A7. Library-change behaviour — **Done**
- A confirmed write (Indexer, Library Sync, Duplicates, File Cleanup, tag tools, playlist writes) now runs to
  completion on its original library when the library is switched. Read-only work is cancelled. Duplicates used
  to cancel its execution part-way.

### A8. Album artwork files (`cover.jpg` etc.) still go to `Trash/` — **Done** (owner confirmed: intentional)
- No change. The independent reviewer had asked whether art should follow its album; the owner said no.

### A9. Library Sync stale plans: items are skipped one by one — **Done**
- The reviewer proposed refusing the whole run when the library changed since Build Plan. Instead, each stale
  item is skipped and reported, and the rest proceed (Library Sync items are independent).

### A10. Indexer "Max workers" is real now — **Done**
- **Was**: `docs/ROADMAP.md` ("Controls that are still switched off") and `CLAUDE.md` say the engine ignores
  Max workers. The control was disabled, with a default of 4.
- **Is**: `build_indexer_plan(..., max_workers=N)` reads tags on a bounded thread pool. Results come back in
  input order and cancellation is honoured. Each file's tags are now read **once** per scan, not twice.
- **Measured** on 3,000 tagged files:
  - local disk: 1 worker 0.69 s, 4 workers 2.34 s;
  - with 5 ms latency per file: 1 worker 16.7 s, 4 workers 4.4 s.
  So the control is now enabled with a default of **1**, and its tooltip says more only helps on network or slow
  drives. The old default of 4 would have made local scans about 3× slower. The control is locked while a run is
  going.
- **Contradicts**: `docs/ROADMAP.md` (now updated), `CLAUDE.md` (now updated), and the old "Advanced (not used by
  the current engine)" group title (now just "Advanced").
- **Undo**: `music_indexer_api._iter_tags` / `MAX_TAG_WORKERS`.

### A11. Indexer "Flush fingerprint cache" stays switched off, and its premise was wrong — **Done**
- The label promises a fingerprint cache, but the Indexer computes no fingerprints. The old flag deleted
  `Docs/.soundvault.db`, which is now the Duplicate Finder's cache.
- A tag cache that would give the flag a meaning was considered and **rejected**: taggers like Mp3tag can keep
  the old mtime, and tag edits often keep the file size, so the Indexer could file tracks by stale tags. The
  parameter is still accepted and ignored. The redesign should remove the control.

### A12. Library Sync HTML report: the three fake buttons are removed, not wired — **Done**
- **Was**: "Replace All" / "Replace with Better" rendered as buttons that did nothing (a ROADMAP known bug that
  said "wire or remove").
- **Is**: removed, and replaced by a note: "To replace, flag tracks in AlphaDEX → Library Sync and build a plan".
  The dead backing code `build_review_replacement_plan`, `_select_replacement_sources` and `ReviewAction`
  (no callers anywhere) was deleted. A static HTML file can't safely run file operations.

### A13. Tag writes can be undone; `None` now means "remove this tag" — **Done**
- **New**: `controllers/tag_revert.py` lists each Tag Fixer / Genre Normalizer apply from its `Docs/` change
  log, and restores exactly what was there before. That covers every value of a multi-value field, and removes
  a field that didn't exist before. Files edited since then are skipped as conflicts, and each revert is itself
  logged so it can be reverted.
- **In the app**: a new fourth tab, **Metadata → Undo changes**. It lists each apply, "Preview undo" writes
  nothing, and "Undo these changes" confirms, then writes exactly the preview. It holds the tag-write guard and
  the library-write lock.
- **Contradicts**: `utils/tag_change_log.py`'s "a future revert tool" note. `utils/tag_writer.py` now treats a
  `None` value as "remove the field" (no existing caller passed `None`).

### A14. Library Sync keeps decoded fingerprints on disk — **Done**
- Build Plan's index of a 20k-track library went from 31 s on every run to 4.3 s after the first run (with
  identical results). The cost is that Library Sync's cache file grows by about 80 MB per 20k tracks
  (164 MB in total at 20k).
- **Doc fix**: `docs/features/library_sync.md` and `docs/architecture.md` still said the vendored engine
  "compares the compressed fingerprint text" and listed that as a known bug. The decoded metric was already
  ported, so that text was stale and is corrected.

### A15. Short tracks are no longer padded with silence before fingerprinting — **Done**
- **Was**: `audio_norm.normalize_for_fp` padded any track shorter than the fingerprint window (120 s) with
  silence. The old `test_padding_short_track` required this.
- **Is**: the window is cut to the track, never padded.
- **Why**: measured with real Chromaprint, two *unrelated* 5 s clips scored 0.011–0.014 (under the 0.02
  exact-duplicate threshold). 20 s clips scored about 0.05 and 40 s clips about 0.1, both under Library Sync's
  0.2. So intros, skits and interludes could be matched to each other as duplicates.
- **Old cached fingerprints**: both caches now treat an old row (no `fp_scheme`) whose fingerprint ends in a
  run of padding silence as a cache miss, and recompute only those. That covers the root cache (A30) and
  Library Sync's vendored cache (A31). Library Sync's cache is the one that actually held padded prints,
  because only its generator fingerprinted through `normalize_for_fp`.

### A16. M4A track and disc numbers read as "3/12", not the integer 3 — **Done**
- `test_audio_metadata_reader.py` asserted an integer. Every other format already returned "N/T", and the M4A
  totals were being lost.

### A17. Mirror: playlists are rewritten, stale files detected, orphans offered for removal — **Done**
- **Was**: "all other formats copied unchanged" (page subtitle and info label). Existing destination files were
  skipped as "already exists" forever. There was no backend check that the folders don't overlap. The mirror
  never deleted anything.
- **Is**:
  - Copied playlists have their FLAC entries pointed at the converted `.opus`. The file is otherwise
    byte-exact.
  - A mirrored file older than its changed source is "stale". On the page, **Update mirror files whose
    source has changed** is checked by default, so stale files are redone. Unchecked, they are kept and
    reported.
  - Files left in the mirror after their source was deleted or renamed are listed as orphans. **Remove N
    orphaned file(s)…** deletes them only after a confirm, only inside the mirror, and logs every outcome.
  - The backend refuses the same folder, a destination inside the source, *and* a source inside the
    destination. The page previously allowed the last of these.
  - A run removes its own temporary files left by a killed earlier run.
  - A destination folder whose symlink or junction points outside the mirror, or into the source, is refused.
- **Undo**: `utils/opus_library_mirror.py`. The page default is `_update_cb` in
  `gui/workspaces/compression.py` (the backend default is off).

### A18. `update_genres.py` (command line) is preview-first — **Done**
- **Was**: its docstring said the CLI "writes as it goes". Each file was written as soon as its lookup
  returned.
- **Is**: it lists every planned change, asks (or takes `--yes`), then writes exactly that list and logs to
  `Docs/`. A file whose genres can't be read is never written.

### A19. File Cleanup renames artwork sidecars with their track — **Done**
- **Was**: only audio files were renamed, so `Song (1).jpg` was left behind when `Song (1).mp3` became
  `Song.mp3`.
- **Is**: a sidecar (`.jpg`/`.cover`/`.artwork`) is its own preview row. It is renamed only if its track is,
  and undone with it. (Album artwork going to `Trash/` during indexing is unchanged; see A8.)

### A20. Playlist engine edge cases — **Done**
- `autodj_playlist(n <= 0)` returns `[]` instead of `[seed]` (the page's minimum is 5).
- `bucket_by_tempo_energy` returns a new `failed` key. A NaN tempo or energy is a failed analysis instead of
  landing in "fast/high".
- One undecodable track no longer aborts an Auto-DJ run. A track listed twice is used once.
- **New on the Playlists page**: Auto-DJ keeps each track's analysis for the next run in the same library.
  A file whose size or mtime changed is analysed again.

### A21. Tag text that isn't UTF-8 is no longer guessed as UTF-16 — **Done**
- `b"Caf\xe9"` used to decode as CJK characters. It is now UTF-8, then UTF-16 only with a BOM, then cp1252,
  then Latin-1 (`utils/text_decoding.py`).

### A22. A failed genre write records why — **Done**
- `apply_genre_changes` now logs the writer's reason in the change log (for example a file open in another
  program), not just "write failed".

### A23. Duplicates: the lossless copy always wins — **Done**
- **Was**: readable cover art ranked first, so an MP3 with art beat the FLAC without it, and the FLAC was
  quarantined.
- **Is**: lossless first. Cover art only breaks ties within the same class. Between lossless copies,
  resolution decides, then `config.FORMAT_PRIORITY` (FLAC > WAV), so an uncompressed WAV no longer beats the
  same audio as FLAC on bitrate.
- **Docs** used to say equal-quality ties favour "more descriptive names". They are broken by path,
  deterministically, and the docs now say so.

### A24. Duplicates: the winner is no longer renamed by a blanket "(digits)" rule — **Done**
- **Was**: execution step 7 stripped any trailing "(digits)" from every winner, without a preview, so
  `Live (1986).flac` became `Live.flac`.
- **Is**: a winner only takes back the exact name of a same-group loser that really left (1–2 digits). It never
  overwrites, and the preview shows the rename.

### A25. Duplicates: playlists — **Done**
- Execution used to require every playlist entry to exist. One dead entry already in a playlist blocked the
  whole run. Now only the entries the rewrite changes must exist.
- Rewritten absolute entries stay absolute (they used to be made relative).
- `.m3u8` playlists are rewritten too (they were never touched).
- Non-UTF-8 playlists are read with fallback encodings and written back in their own encoding and line
  endings (they crashed the scan).

### A26. Duplicates: the metadata step writes only the previewed changes — **Done**
- **Was**: it wrote the whole planned tag dict, adding stray `TRACK=`/`YEAR=` fields.
- **Is**: only the previewed `metadata_changes`. A previewed year goes into `date`. Artwork and tag writes go
  to a copy that replaces the winner only on success. Covers are embedded each format's own way: the old ID3
  fallback made OGG/Opus/WAV/AIFF winners unreadable.

### A27. Duplicates: Cancel really stops — **Done**
- **Was**: the docs said "Cancel stops after the current operation", but every loser went to a thread pool up
  front, and the pool kept moving and deleting after a cancel or a failed move. None of those moves were
  reported.
- **Is**: at most the operations already running finish (≤ 16). Everything that happened is reported, and the
  rest are listed as "not attempted". A per-loser safety refusal doesn't stop other groups; a real I/O error
  stops the run.

### A28. Duplicates: new safety refusals — **Done**
- A loser is left in place (and reported) if:
  - its winner is missing;
  - the winner is the same file through a symlink or hard link;
  - the loser changed since the preview.
- Plans whose groups overlap are refused at preflight.
- Symlinks and hard links are no longer counted as duplicates of their target.
- Quarantine names are claimed with a placeholder file, so names differing only by case or Unicode form
  can't collide on Windows/macOS.
- A cross-drive move whose original can't be deleted removes its copy again and reports the loser as not
  moved.

### A29. Duplicates: short clips vs. full songs — **Done**
- A 30–60 s clip used to score 0.0 against its full song. Grouping now uses a length-aware
  `duplicate_fingerprint_distance`, so a clip is not called a duplicate of the song.
- `fingerprint_distance` itself is unchanged, because the vendored parity test relies on it.
- **Compare two files** now gives the Duplicate Finder's verdict, using the length-aware distance. Its report
  also shows the old overlap-only distance, labelled.
- The Duplicates page's result line separates **"left in place for safety"** (with the reasons) and **"not
  attempted"** from real errors. It used to call every refusal an error.

### A30. Duplicates: other changed behaviour — **Done**
- **Reserved folders**: the legacy scanners skipped "Not Sorted"/"Playlists" at any depth. Now all reserved
  folders are skipped, but only at the top level, so `Artist/Playlists/` is scanned.
- **Review flags**: missing year or artwork no longer raises "Placeholder metadata" / "unreadable" flags.
  Groups with no artwork are still flagged "No artwork candidates".
- **Grouping timeout** (120 s) now starts when grouping starts, not at the start of the build. Tag reading
  used to eat it, so first scans of big libraries found almost nothing. Limits hit are logged.
- **Fingerprint cache**:
  - a corrupt cache file is renamed `.corrupt-<time>` and rebuilt (it used to be fatal);
  - one bad row no longer kills the writer thread;
  - the schema gains `fp_scheme`;
  - old fingerprints that end in silence padding are recomputed; nothing else is re-fingerprinted.
- **Still open, for the owner**:
  - the executor writes an unpreviewed `<winner>.artwork` sidecar for every artwork transfer, and the readers
    depend on it;
  - the planner limits (20,000 candidates, 500,000 comparisons, 120 s) still truncate very large libraries.
    It is now logged and flagged.

### A31. Library Sync: a short clip no longer matches its full song; padded cache rows are redone — **Done**
- **Was**: Library Sync's matcher scored pairs with the overlap-only `fingerprint_distance`. A 30 s clip's
  print is a prefix of the full song's, so the pair scored 0.0 (exact match): an incoming full song was
  "already in the library" when only a clip was there, and a lossless clip of a lossy song was a "possible
  upgrade". `fingerprint_distance`'s root docstring said the vendored engine relied on the overlap-only
  behaviour.
- **Is**: `_match_tracks` scores with `library_sync.match_distance`, the vendored port of
  `duplicate_fingerprint_distance` (unmatched length counts as unrelated audio). A clip under about half of
  the song's printed length is NEW; around half is LOW_CONFIDENCE (review). `library_sync.fingerprint_distance`
  is still bound to the overlap-only vendored metric, which stays identical to the root's
  (`tests/test_vendored_fingerprint_distance.py` unchanged). Tests that fake the matcher's distance now
  patch `library_sync.match_distance` (`tests/test_library_sync_matching.py`,
  `tests/test_syncmatch_candidates.py`'s `distance_calls`): patching the old name would have silently
  stopped reaching the matcher.
- **Why**: the clip/song confusion could skip a full song or offer to replace it with a clip.
- **Also**: Library Sync's vendored cache got the root's `fp_scheme`/padding check (see A15) and its
  corrupt-file handling. The vendored generator's own reuse check is covered too: it used to hand back the
  cached padded print even when the scan had rejected it.
- **Undo**: point `match_distance` back at `idx_near_duplicate_detector.fingerprint_distance` in `library_sync.py`.
- **Still open, for the owner**:
  - an edit longer than 120 s (for example a 2:30 edit of a 5:00 song) still matches the full song, because both
    fingerprints cover the same first 120 s. The track lengths could tell them apart; that is a policy choice;
  - if re-fingerprinting a padded row fails, the track has no fingerprint and is planned as a COPY. It used to be
    matched by the padded print.

## B. Structure that changed (adopted IA) — **Done**

### B1. New page structure replaces the 13-page sidebar
- **Was**: ORGANIZE (Indexer, Library Sync) · CLEAN UP (Duplicates, Similarity, Tag Fixer, Genre Normalizer) ·
  PLAYLISTS (Playlists, Clustered, Music Graph) · PLAYER (Player, Compression) · TOOLS (Utilities, Help).
- **Is**: LIBRARY (Home) · ORGANIZE (Indexer, Library Sync, File Cleanup) · CLEAN UP (Duplicates — with Compare
  two files, Pair review, and the Experimental prototypes — and Metadata — Tag Fixer, Fill genres, Normalize
  genres) · LISTEN (Player, Sound Map, Playlists) · TAKE AWAY (Mirror, Exports) · Activity · Settings · Help.
- **Contradicts**:
  - `docs/gui_inventory.md` (being updated with the change).
  - `CLAUDE.md`'s "~14 user-facing workspaces" and the "Adding a Playlist Creator sub-panel" notes, which are
    Tk-era.
  - The page names used throughout `docs/features/*.md` ("Genre Normalizer", "Clustered", "Music Graph",
    "Compression", "Utilities"). *Updated 2026-11-08: the docs now use "Page → Section" names.*
- **Old keys still route**: `similarity`, `tag_fixer`, `genres`, `clustered`, `graph`, `compression` and `tools`
  open the right page and section. `main_window._workspaces[old_key]` still returns the panel for every old key
  except `tools`, which no longer has a panel.
- **Docs now updated**: `docs/gui_inventory.md` (rewritten) and `CLAUDE.md` (layout, plus a new "Adding a page or
  a section" pattern; the Tk "Tools menu" / "Playlist Creator" patterns are marked legacy).
- **Docs that used old names — now updated** (docs pass, 2026-11-08; places are written **Page → Section**):
  - `docs/architecture.md`: the sidebar table is now the groups / pages / sections table, with an "if you knew the
    older layout" map; "Settings dialog", "Utilities exports" and "Utilities → View Crash Log" now name the
    Settings page, the Exports page and Activity. **Updated.**
  - `docs/features/duplicate_finder.md`: "Utilities → Pair Review" is Duplicates → Pair review. **Updated.**
  - `docs/ROADMAP.md`: "Utilities → Diagnostics", "Crash Log viewer", "Settings dialog" and "Music Graph" now
    name Duplicates → Fuzzy finder, Activity, the Settings page and the Sound Map. **Updated.**
  - `docs/features/playlists_and_clustering.md`: "Music Graph room" and "sidebar's Clustered" are Sound Map →
    Map and Analyse; the mentions of the deleted 3-D widget are replaced by the browser page from
    `cluster_graph_3d.py`. **Updated.**
  - `docs/features/tag_fixer.md`: the Genre Normalizer sidebar entry is Metadata → Fill genres / Normalize
    genres. **Updated.**
  - `docs/README.md`. **Updated.**
  - the root `README.md`: "5 sections, 13 items" and tools.py as "Export & Utilities" are replaced by the new
    shell and pages. **Updated.**
  - `docs/redesign/README.md`: its analysis is kept and marked as the pre-IA state; "13 sidebar entries" is
    spelled out next to an "As built" structure, and the hygiene table shows today's ceilings. **Updated.**
  - Also checked: `docs/overview.md` and `docs/features/library_sync.md` (updated: "workspaces" became pages)
    and `docs/features/indexer.md` (no old names).
- **Panel headings** still use old names under the new pages ("Genre Normalizer", "Library Compression",
  "Similarity Inspector"). Only Analyse and Map were retitled. The backend copy in `cluster_graph_data.py`,
  `cluster_graph_3d.py` and `clustered_playlists.py` still says "Clustered Playlists" / "Music Graph".

### B2. Prototype duplicate tools live inside Duplicates with an "Experimental" pill
- Fuzzy Finder, Scan Engine and Bucketing POC moved from the Utilities → Diagnostics tile to Duplicates
  sections, each marked **Experimental** (not ready to test).
- Pair Review and the three prototypes are still **dialogs**, opened by an "Open …" button inside their
  section. Each has its own close button and worker lifecycle, so embedding them is redesign work.

### B3. Settings is a page; the theme picker is its Appearance section
- **Was**: a modal Settings dialog plus a separate Theme picker window.
- **Is**: Settings has three sections: Metadata services, General and Appearance. It saves under the same config
  keys.
- **Kept, but no longer opened by the app**: the thin wrappers `SettingsDrawer`, `ThemePickerDialog` /
  `open_theme_picker`, `MediaTesterDialog` and the `CrashLogDialog` class. Deleting them needs the owner's OK.

### B4. Utilities is gone
- Its pieces moved: File Cleanup → Organize, exports → Exports, Validator → Home, crash log → Activity, and the
  Opus/M4A testers → Mirror.

### B5. Metadata has a fourth section, "Undo changes" (see A13)
- This is not in the adopted IA proposal (§5 of `docs/redesign/README.md`), which listed three Metadata
  sections. It is added because undo is per-tool history and belongs next to the tools that write tags.

## C. Files deleted (owner approved) — **Done**
- Deleted: `gui/widgets/splash.py`, `gui/workspaces/clustered.py`, `gui/widgets/interactive_3d_scatter.py`,
  `mutagen_stub/`, `update_tools.py`, `update_lg.py`, `demo_tile_proposals.py`, `player_106bd0e.py`, the
  `player_*`/`diff_*` text dumps, `crash_test.txt`, the committed `soundvault_crash.log*`, and `=0.13.0`.
  `last_path.txt` and `crash_report.txt` are untracked and ignored.
- **Contradicts**:
  - `docs/reorganization_spec.md` (a deferred plan that still lists `mutagen_stub/`).
  - Older mentions of the 3-D widget in `docs/features/playlists_and_clustering.md`. *Updated 2026-11-08: the
    doc now describes the browser page from `cluster_graph_3d.py`, and names the deleted widget only in a
    historical note.*

## D. Features being finished — **Done**

### D1. Cluster playlists — **Done**
- **Was**: promised by the wizard and docs but never built; the current UI says "not available". "A run writes
  no playlists; export clusters by hand from the map" (docstrings, log line, Results hint, wizard tooltip,
  feature doc).
- **Is**: **Sound Map → Save cluster playlists…**, built from `docs/specs/cluster_playlists.md` (spec, then an
  independent audit, then the build).
  - It previews one `.m3u` per cluster: a descriptive name, the order inside the playlist, a minimum size,
    files that are missing, and new / replace / skip for each row. Then it writes exactly that into a per-run
    folder under `Playlists/Clusters/`. Replaced files are backed up and every write is logged.
  - An analysis now also writes `Docs/cluster_members.json`, the full membership, never downsampled.
- **Also changed**:
  - The wizard's "Create M3U playlists" option is live, as "Offer to save a playlist per cluster when the
    analysis finishes" (off by default).
  - The map's lasso **Export playlist…** used to write the file raw. It now checks the library-write lock,
    then uses the shared writer, which leaves a `.bak` and a log line.
  - `write_playlist_plan` always returns a `cancelled` key.
  - The Analyse Run buttons pause while a cluster-playlist preview or write runs.
- **One consequence**: maps analysed before this feature must be re-analysed once before they can be saved as
  playlists. The old map file is downsampled to 5,000 points, so it can't build complete playlists.
- **Owner's open questions** (spec §12): every one has a default in use. For example, earlier runs' playlist
  folders are kept, HDBSCAN's unclustered tracks never become a playlist, the order is closest to the centre
  first, and the minimum is 5 tracks.

## E. Process / tooling
- **E1.** Windows CI is a required job (it was informational).
- **E2.** `CLAUDE.md`'s "Library sync default" constant was updated to match A2.

## F. The look and the shell (redesign R1: Liquid Glass) — **Done**

The owner chose "Refined Liquid Glass" (docs/redesign). Everything below is undoable per item.

### F1. Default theme: Midnight → Liquid Glass, with a one-time migration
- **Was**: `DEFAULT_DARK = "midnight"`; a saved theme was always honoured.
- **Is**: `DEFAULT_THEME = DEFAULT_DARK = "glass"`. A saved theme other than Liquid Glass (Auto included) is
  switched to it **once**: the old key is kept as `theme_before_glass`, `theme_glass_migrated` stops it from
  happening again, and the main window logs a line. Choosing a classic theme afterwards sticks.
- **Undo**: `gui/themes/tokens.py` (`DEFAULT_THEME`, `DEFAULT_DARK`), `ThemeManager._migrate`.

### F2. "Background gradient" switch → Backdrop (Album art / Gradient / Plain)
- **Was**: a checkbox (`bg_gradient_enabled`) for two accent glows painted by every page and the sidebar.
- **Is**: `backdrop_mode`, painted once by the window's backdrop; Album art (a blurred wall of the library's
  covers, Liquid Glass only) is the default. `bg_gradient_enabled` is still read when `backdrop_mode` is absent
  (False = Plain) and is kept in step on save. `GradientWidget` no longer paints a background.
- **Undo**: `gui/themes/appearance.py`, `gui/widgets/backdrop.py`, `SettingsForm._build_appearance`.

### F3. Top bar → page header
- **Was**: a 64 px bar with the "AlphaDEX" brand, the path (last two segments), stats, and Change Library /
  Theme / Settings buttons.
- **Is**: no bar: the page title (34 px), a library chip (full path, elided), the stats, a search pill (Enter
  shows the Player filtered to the text) and Change library. The brand is in the sidebar; Theme and Settings
  buttons are gone (Settings: sidebar and Ctrl+,; theme: Settings → Appearance). The signals remain.

### F4. Sidebar: 360 px panel with icons and Exit → floating 250 px glass panel
- **Was**: emoji icons, an accent bar on the active item, an Exit item; item height stretched to fill.
- **Is**: text-only 40 px items (32 px minimum), a raised pill that slides between items, a "N running" badge;
  **no Exit item** — Ctrl+Q (new) and the window's close button quit through the same close protocol.

### F5. Status bar hidden
- **Was**: a visible status bar ("Page · Section", library, stopping messages).
- **Is**: still written (code and tests read `currentMessage()`), not shown; the header shows the page, the
  log strip's status chip shows status and the stopping message.

### F6. Now Playing: strip → floating capsule
- **Was**: a full-width strip above the log drawer.
- **Is**: a floating glass capsule at the bottom centre of the page area, with the same controls and signals,
  still hidden until something plays. Pages keep 104 px of room under their content; the Player page (which has
  its own transport) now reserves it too, so its own controls end 104 px higher.

### F7. Cards, controls and type
- Cards: opaque 12 px frames with accent strips on header/action cards → translucent glass (radius 26, hero 30),
  no accent strips, 8 / 6 px more padding, softer and larger painted shadows.
- Buttons, line edits, combo and spin boxes are pills; checkboxes 18 px; scroll bars thin on a transparent track;
  sliders a 5 px track with a white knob; tabs pills, page sections a segmented control; under glass a table
  selection is the translucent accent with white text.
- Type: Inter-first, point-based sizes → one px ramp (display 34 · title 20 · heading 16 · body 14 · label 13 ·
  caption 12 · micro 11 · mono) preferring Segoe UI Variable, then Segoe UI, SF Pro, the bundled Inter. The app
  font is 14 px (was 11 pt ≈ 14.7 px).

### F8. Landing
- **Was**: 110 px tiles in a centred grid around a reserved block; one button; no quit; ignored reduced motion.
- **Is**: a full-window 7 × 5 grid, a frosted glass card with the library's path and counts (a read-only count
  runs while the landing is up), **Choose another folder** (new link), a **quit** button (new), hover
  lift/ring/caption chip (album and artist read with the cover), and reduced motion skips the fly-in and the
  scatter. The covers found are handed to the main window's backdrop.

### F9. Transitions and housekeeping jobs
- Page and section switches were instant; they now cross-fade (opacity, `base` / `fast`; instant with reduced
  motion).
- Housekeeping jobs (backdrop build, landing frost) are `quiet`: kept alive and cancelled on quit like any job,
  but not listed on the Activity page or counted in its badge.

### F10. Reduce motion has a control
- **Was**: config key / environment variable only.
- **Is**: Settings → Appearance → Reduce motion (same key).

## G. Page contents in Liquid Glass (redesign R2) — **In progress**

Every page drops its in-page title. The shell header and the section tabs already name it. Each page keeps one
short muted line saying what it is for, and buttons use painted glyphs instead of emoji. Shared parts:
`gui/widgets/glass_parts.py` (stat tiles, empty states) and `gui/widgets/glyphs.py` (icons).

### G1. Duplicates
- **Find duplicates**:
  - The "1. Scan & fingerprint → 2. Review groups → 3. Execute plan" stepper → a pill row "Scan · Review ·
    Execute". It shows real state.
  - The groups table (#, Winner, Losers, Review ⚑) → one glass row per group. Each row shows the kept copy and
    each loser with its disposition, with format and bitrate pills. A row lists at most 3 losers plus "+ N more";
    the details box lists them all.
  - Results are stat tiles. The plan tiles are groups, files, space and flagged; after Execute, outcome tiles
    show quarantined, deleted, left in place for safety, not attempted, and errors. Every sentence
    `summarize_execution` produces is unchanged.
  - Buttons: "Scan Library" → **Find duplicates**, "Open Report" → **Report**, "Execute" → **Execute…**.
  - "Delete losers permanently (cannot be undone)" → "Delete losers permanently" plus a red "Cannot be undone"
    pill. The typed DELETE confirmation is unchanged.
  - Cancel and the progress bar show only while a job runs. They were always visible but disabled.
- **Compare two files**:
  - Editable path fields → elided path rows filled by Browse or drag-and-drop. **You can no longer type a path.**
  - The verdict is a large pill, followed by four distance tiles (including the overlap-only distance). The full
    text report folds behind "Full report".
  - The advanced overrides fold away when off, with a line naming the thresholds in use.
  - "⚖ Run Comparison" → **Compare**.
- **Pair review and the Experimental launchers**: one card each, titled by what the tool is for. The Experimental
  pill sits beside the title. Inside Pair review:
  - "Left Track" / "Right Track" → LEFT / RIGHT;
  - "◄ Prev" / "Next ►" → glyph buttons;
  - the ✅ winner mark → a toned "Kept" badge;
  - "Not winner (would be quarantined)" → a "Would be quarantined" badge.
- **Scan engine dialog**: the log uses the shared mono log style (`logBox` → `logText`).
- **Wording**: "Run Duplicate Finder first" → "Run Find duplicates first". The job name "Similarity" → "Compare
  two files".

### G2. Sound Map
- **Analyse: own settings**:
  - **Was**: a read-only text summary that only the wizard could change ("Please configure with the wizard
    first").
  - **Is**: an editable "Your own settings" card: feature pills, algorithm, scaling, K or the HDBSCAN sizes, and
    the playlist offer. The wizard pre-fills it and writes back to it.
  - The "Output" lines CP-43 required in that summary are gone. The results hint and the offer checkbox carry
    that wording now.
- **Analyse: save button**: Save cluster playlists… also appears on the results card. It starts the same flow as
  the page button and is enabled and disabled with it. CP-43 had kept it off the analysis page.
- **Quality report**:
  - **Was**: the silhouette verdict was normalised over −1…1, so 0.42 read "good" next to "Moderate silhouette
    score".
  - **Is**: good above 0.5 and fair above 0.3, the same as the Analyse tile and the report's own suggestions.
- **Map**:
  - The plot is transparent on a glass well with a faint hairline grid. It used to sit on `card_bg`, which PL-21
    asserted.
  - The selection outline is the accent colour (was `text_primary`), and the rest of the map dims.
  - The selection bar moved above the plot.
  - Labels:
    - "Rectangle" → **Box**;
    - "Open 3D view" → **3-D view**;
    - "Export CSV…" → **CSV…**;
    - "View Quality Report" → **Quality report**.
  - The "Track Details" title became an empty state until a track is hovered or pinned.
- **Legend**: the colour dot is the show/hide toggle (still exposed as `.checkbox`).
- **Wizard**: the thin progress bar became step pills.

### G3. Metadata
- **Tag Fixer scope**:
  - **Was**: "every supported audio file under the library" (`docs/features/tag_fixer.md`).
  - **Is**: the whole library or one folder inside it. Reserved folders and folders outside the library are
    refused. The database and the change log stay the library's.
- **Normalize genres**:
  - The "1 · 2 · 3 · 4" numbered steps, which the redesign page map once held up as a pattern, are removed.
  - A hero card holds the inputs and **Preview changes**; a plan card holds Apply.
  - Mappings to `["invalid"]` / `null` show as a red **Drop** pill. A file whose genres would all be dropped shows
    "All dropped".
- **Undo changes**:
  - The undo button is **dangerBtn** (was primaryBtn), because it overwrites tags.
  - The list column "Changes" → "Tags".
- **Tag Fixer labels**:
  - "Scan for Tag Issues" → **Find tag fixes**;
  - "Apply Selected" → **Apply N changes**;
  - "Select All / Deselect All" → **Tick all / Untick all**;
  - "Fix fields:" → **Show**.
  - `docs/features/tag_fixer.md` still uses the old labels.
- **Counts**: the Tag Fixer summary and Normalize's "N of M tagged files would change (…)" sentence → stat tiles.
  The status lines keep only the explanation.
- **Columns**: Metadata's tables stretch their columns and elide the middle of the text, so nothing scrolls
  sideways at 900 px. **Column widths can no longer be dragged.**
- **Row separators** (Metadata only): one continuous hairline per row, drawn by a delegate, instead of the style's
  per-cell line.
- **Genres, first load**: "Library changed — preview again" appears only when there was a plan or a lookup to drop.
- **Motion**: switching between Fill and Normalize cross-fades. Since R1 it had been an instant swap.

### G4. Organize (Indexer, Library Sync, File Cleanup, Exports)
- **Library Sync no longer scrolls sideways.**
  - One button row asked for about 1,090 px. The pickers are now a grid and the scan bars reflow.
  - The review lists sit in a splitter: side by side when wide, stacked below about 880 px. The plan buttons wrap.
  - Statuses read "Exact match" on screen; the engine value is unchanged underneath.
- **Summaries**: the Indexer's and Library Sync's prose summaries → stat tiles plus a line saying what to do
  next. The full sentences still go to the log and the Preview Ready box.
- **Indexer**:
  - The stepper tracks real state, so it is now a compact pill row.
  - Phase C, Flush and Max workers sit in an Advanced well.
  - Open report is a glyph button.
- **File Cleanup and Exports tiles**:
  - The cursor-following glow on a timer → static glass cards.
  - The result area opened by animating its height → it fades in. The motion rule is never to animate geometry a
    layout owns.
  - The tile glyph badge and the unused text-scale API are removed.
- **No emoji**: the Flag column and the right-click menu ("📋 Copy", "↻ Replace", "📝 Add note") use plain words
  with painted icons. The threshold warning lost its "⚠" prefix; the warning colour carries it.
- **Wording**:
  - "Scan Both Libraries" → "Scan both folders";
  - "Existing Library:" → "Library";
  - "Export Logs…" → "Export log…";
  - other labels changed only in casing.
  - The codec status line no longer contains the full output path, which moved to an elided path row.
- **File Cleanup preview dialog**:
  - The confirm button's object name `cleanupConfirmBtn` → `primaryBtn`.
  - Cancel is still the default button, so Enter never renames.
- **Docs still using old labels**:
  - `docs/library_sync_per_item_review_testing.md` (emoji labels);
  - `docs/features/library_sync.md` (old capitalisation);
  - `docs/architecture.md` ("Undo Last Cleanup").

### G5. Player and Playlists
- **The capsule is the only transport.** This follows the adopted IA: "the bar is the only transport".
  - **Was**: the Player page had its own seek bar, ⏮ ▶ ⏭ ■, shuffle, repeat and volume, so the screen showed two
    of each.
  - **Is**: shuffle and repeat are glyph toggles on the capsule. Repeat cycles Off → Track → All; Track shows a
    "1" badge.
- **Stop is dropped.** Pause silences, Previous after 3 s restarts, and Clear on the queue unloads everything. A
  sixth round button didn't fit the capsule at 900 px. The internal `_on_stop()` is kept.
- **When the capsule shows**: the Player page shows it idle ("Nothing playing" / "Playback unavailable") before
  anything has played. Other pages keep R1's rule: hidden until something plays. A page switch shows it at once,
  because the page itself fades in; loading a track still fades it in.
- **Capsule at narrow widths**: three tiers.
  - Below about 840 px the volume slider moves into a popup behind the speaker button.
  - Below about 724 px the times move into the seek bar's tooltip.
  - Every control stays reachable.
- **Cover art**: when a track has no embedded cover, the folder's `cover.jpg` (or the landing's other names) is
  used in the capsule, the carousel and the Player card. Before, only embedded art was used.
- **Player layout**:
  - The resizable splitter → fixed glass cards. **You can no longer drag the divider.**
  - The cover carousel hides when the column is too short.
  - No alternating row colours.
  - On narrow tables, Album and then # hide; search still matches them.
  - The playing row is tinted with the accent and shows a play or pause glyph.
- **Playlists**:
  - The "Playlist Generator" heading is gone.
  - The four "Generate" buttons → a kind switcher (Folder · Tempo & energy · Auto-DJ · Repair) with one Preview
    button.
  - The always-visible Log card → behind "Details".
  - The plan shows as glass rows with new / replace / written / failed pills.
- **Fixed (found through the new preview rows)**: tracks sitting directly in the library folder went into a playlist
  called `Playlists/..m3u`, because `os.path.relpath(root, root)` is ".". They now go into `root.m3u`, the
  fallback `_sanitize_name` always meant to use. This applies to both `plan_folder_playlists` and
  `generate_playlists`.
