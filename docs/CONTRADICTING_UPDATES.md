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
- **Old cached fingerprints**: the root cache marks rows with a new `fp_scheme`, and old rows whose fingerprint
  ends in silence padding are recomputed (only those; see A30). Library Sync's separate cache gets the same
  treatment (A31).

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
    "Compression", "Utilities").
- **Old keys still route**: `similarity`, `tag_fixer`, `genres`, `clustered`, `graph`, `compression` and `tools`
  open the right page and section. `main_window._workspaces[old_key]` still returns the panel for every old key
  except `tools`, which no longer has a panel.
- **Docs now updated**: `docs/gui_inventory.md` (rewritten) and `CLAUDE.md` (layout, plus a new "Adding a page or
  a section" pattern; the Tk "Tools menu" / "Playlist Creator" patterns are marked legacy).
- **Docs still using old names**, to fix in the docs pass after the redesign settles:
  - `docs/architecture.md` (the sidebar table, "Settings dialog", "Utilities exports", "Utilities → View Crash
    Log");
  - `docs/features/duplicate_finder.md` ("Utilities → Pair Review");
  - `docs/ROADMAP.md` ("Utilities → Diagnostics", "Crash Log viewer", "Settings dialog", "Music Graph");
  - `docs/features/playlists_and_clustering.md` ("Music Graph room", "sidebar's Clustered");
  - `docs/features/tag_fixer.md` (the Genre Normalizer sidebar entry);
  - `docs/README.md`;
  - the root `README.md` ("5 sections, 13 items", tools.py as "Export & Utilities");
  - `docs/redesign/README.md` (its "13 sidebar entries" wording and hygiene table).
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
  - Older mentions of the 3-D widget in `docs/features/playlists_and_clustering.md`.

## D. Features being finished — **In progress**

### D1. Cluster playlists
- **Was**: promised by the wizard and docs but never built; the current UI says "not available".
- **Is**: being built from `docs/specs/cluster_playlists.md` (audited first).
- **Contradicts** `docs/features/playlists_and_clustering.md` ("not available") and the clustering wizard's
  disabled output options.
- One consequence: maps analysed before the feature must be re-analysed before they can be saved as playlists.
  The old map file is downsampled to 5,000 points, so it can't be used to build complete playlists.

## E. Process / tooling
- **E1.** Windows CI is a required job (it was informational).
- **E2.** `CLAUDE.md`'s "Library sync default" constant was updated to match A2.
