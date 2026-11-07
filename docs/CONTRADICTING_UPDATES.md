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

### A10. Indexer "Max workers" is real now — **Done** (backend; the GUI control is wired with the new pages)
- **Was**: `docs/ROADMAP.md` ("Controls that are still switched off") and `CLAUDE.md` say the engine ignores
  Max workers. The control was disabled, with a default of 4.
- **Is**: `build_indexer_plan(..., max_workers=N)` reads tags on a bounded thread pool. Results come back in
  input order and cancellation is honoured. Each file's tags are now read **once** per scan, not twice.
- **Measured** on 3,000 tagged files:
  - local disk: 1 worker 0.69 s, 4 workers 2.34 s;
  - with 5 ms latency per file: 1 worker 16.7 s, 4 workers 4.4 s.
  So the default becomes **1**, and more workers are offered for network or slow drives. The old default of 4
  would have made local scans about 3× slower.
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

### A13. Tag writes can be undone; `None` now means "remove this tag" — **Done** (backend; GUI coming)
- **New**: `controllers/tag_revert.py` lists each Tag Fixer / Genre Normalizer apply from its `Docs/` change
  log, and restores exactly what was there before. That covers every value of a multi-value field, and removes
  a field that didn't exist before. Files edited since then are skipped as conflicts, and each revert is itself
  logged so it can be reverted.
- **Contradicts**: `utils/tag_change_log.py`'s "a future revert tool" note. `utils/tag_writer.py` now treats a
  `None` value as "remove the field" (no existing caller passed `None`).

### A14. Library Sync keeps decoded fingerprints on disk — **Done**
- Build Plan's index of a 20k-track library went from 31 s on every run to 4.3 s after the first run (with
  identical results). The cost is that Library Sync's cache file grows by about 80 MB per 20k tracks
  (164 MB in total at 20k).
- **Doc fix**: `docs/features/library_sync.md` and `docs/architecture.md` still said the vendored engine
  "compares the compressed fingerprint text" and listed that as a known bug. The decoded metric was already
  ported, so that text was stale and is corrected.

## B. Structure that changed (adopted IA) — **In progress**

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
  open the right page and section.

### B2. Prototype duplicate tools live inside Duplicates with an "Experimental" pill
- Fuzzy Finder, Scan Engine and Bucketing POC moved from the Utilities → Diagnostics tile to Duplicates
  sections, each marked **Experimental** (not ready to test).

### B3. Settings is a page; the theme picker is its Appearance section
- **Was**: a modal Settings dialog plus a separate Theme picker window.

### B4. Utilities is gone
- Its pieces moved: File Cleanup → Organize, exports → Exports, Validator → Home, crash log → Activity, and the
  Opus/M4A testers → Mirror.

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
