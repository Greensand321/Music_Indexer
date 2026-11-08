# AlphaDEX — Roadmap (What's Planned but Not Yet Built)

*This is the honest "what's left" document: the known bugs, the features that are
partially built or deliberately switched off, and what is planned. It is written in
plain language so anyone can understand the state of the project at a glance. The
**Known bugs** section below is the single source of truth for gaps; other documents
link here rather than keeping their own lists.*

*Last reviewed: 2026-11-07, after the fix phase that closed about 200 bugs across
every page of the Qt app. The per-area bug inventories (and which bug was fixed by
which change) live in `docs/redesign/`; this page lists only what is still open.*

---

## Quick status legend

- **Bug** — this doesn't do what the app says it does. Fixing it restores existing
  functionality rather than building something new.
- **Disabled, no backend** — the control is visible but switched off, with a tooltip
  saying so. Either build the backend or remove the control.
- **Backend ready, not connected** — the engine exists; only the wiring is missing.
- **Partially built** — some of it works; meaningful pieces remain.
- **Designed, not started** — there's a clear vision but no working code yet.

---

## Known bugs

Real defects in the current code, smallest first within each area.

### Legacy app only

- **"Reset Log" in the legacy Tag Fixer also wipes Library Sync's cache.** Both
  features store state in the library's `Docs/.soundvault.db`, and the legacy reset
  deletes the whole file without warning.

---

## Controls that are still switched off

These are honest now — each is disabled with a tooltip — but each is still a gap:
either build what it promises or remove it in the redesign.

- **Indexer: "Cross-album scan (Phase C)" and "Flush fingerprint cache before scan"**
  *(Disabled, no backend).* The engine has no Phase C and keeps no cache to flush (the
  Indexer computes no fingerprints). ("Max workers" is no longer on this list: it is
  enabled, defaults to 1, and only helps on slow or network storage.)
- **Library Sync: "Preset name"** *(Disabled, no backend).* Presets are not
  implemented.
- **Clustering wizard: dimensionality reduction (PCA / t-SNE / UMAP), the
  post-processing step (remove small clusters, merge into "Miscellaneous"), and the
  output options (create playlists, generate report, open graph)** *(Disabled, no
  backend).* Clusters are turned into playlists from the Music Graph instead.
- **Fuzzy Finder (Utilities → Diagnostics): the metadata word-matching group,
  "Include pairs without fingerprints", "Skip fingerprint check", "Include
  non-matching FP pairs in review", "Send Matches to Duplicate Pair Review"**
  *(Disabled, no backend).* The dialog is a proof of concept; whether to keep it at
  all is an open decision (see below).
- **Settings → Advanced tab** *(Empty).* There is also no Settings control for
  reduced motion (config key `reduced_motion` or `ALPHADEX_REDUCED_MOTION=1`) or for
  the mixed-codec threshold boost (editable only on the Duplicates page).

---

## Open decisions for the owner

The redesign groundwork (`docs/redesign/README.md`, section 6) lists decisions that
are not bugs but block some cleanup: the new page structure, whether to keep the
"delete permanently" option in Duplicates, whether to keep the three prototype
dialogs (Fuzzy Finder, Scan Engine, Bucketing POC), which dead files may be deleted,
the visual direction, and whether startup should skip the landing when the saved
library is valid.

---

## Recently completed

**The fix phase (2026-11-07).** About 200 bugs across every Qt page were fixed. The
headlines, so they don't get re-raised:

- **Settings keeps the AcoustID and Last.fm keys apart.** The single key field used to
  save every service's key as `metadata_api_key`, so a Last.fm key replaced the
  AcoustID key and the Last.fm source never read it. The field now shows the selected
  service's own key (`metadata_api_key` for AcoustID, `lastfm_api_key` for Last.fm,
  with the `LASTFM_API_KEY` environment variable still taking precedence).
- **Switching library lets a confirmed Duplicates execution finish**, like every other
  page's confirmed writes, instead of stopping it part-way.
- **The `test_plugin` fixture no longer runs in real Tag Fixer scans.**
- **Library Sync recognises the library's copies in any codec.** Its vendored engine
  now compares decoded Chromaprint fingerprints (ported from the Duplicate Finder), and
  candidates come from an index of decoded sub-fingerprints across all formats instead
  of "identical print, else the first 50 tracks of the same extension". On a synthetic
  20k-track library the old code called 169 of 170 duplicates "new"; now 0 of 370
  tracks are misclassified. Default threshold 0.2 (old saved defaults migrated once).

- **Quitting or re-running a job no longer crashes the app.** Every worker is held by
  a process-wide JobRegistry until its thread has really finished; closing asks only
  about jobs that may be writing, cancels cooperatively, and can't trap you behind a
  hung job. Only one job may change library files at a time.
- **Preview first, everywhere.** The Indexer, Library Sync, Duplicates, Tag Fixer,
  both Genre Normalizer tabs, the Playlist Generator and File Cleanup each compute a
  plan, show it, ask, and then execute *exactly that plan*; a plan whose inputs or
  files changed since the preview is refused or its stale items skipped. Genre fill
  used to write with no preview at all.
- **Controls that did nothing are wired:** the Duplicates per-group disposition (and
  a real permanent delete behind a typed DELETE), Library Sync's Recompute, Save
  Session restore and Matched Existing Tracks table, the Playlist Generator's
  tempo/energy ranges and "Prefer Opus", the Tag Fixer's per-field ticks, and every
  Settings control. The rest are disabled with a tooltip (see above).
- **No more silent data loss:** Pair Review quarantines with confirmation instead of
  deleting; quarantining same-named losers no longer overwrites files; File Cleanup
  has a preview and an undo; multi-value artist and genre tags are kept; playlist
  updates edit entries in place, in the file's own encoding, with a backup; Library
  Sync transfers are reversible, back up what they replace, and keep Windows letter
  case; settings saves are atomic and an unreadable settings file is backed up.
- **The Duplicate Finder compares decoded fingerprints**, so a FLAC and its MP3 now
  score below 0.01 instead of 0.08–0.46 (Library Sync excepted; see Known bugs).
- **Reserved folders** have one definition (`utils/reserved_folders.py`) used by
  every scan.
- **Themes switch in ~75 ms** instead of over a second, reach every widget, meet WCAG
  AA contrast, and cards have their (painted) shadows back. Animations go through one
  motion module with a reduced-motion switch.
- **The Player survives a missing libVLC** (the playback backend seam) and says
  "Playback unavailable" instead of pretending to play.
- **The crash log exists** (`~/.alphadex/crash.log`, `~/.alphadex/faults.log`) and the
  Crash Log viewer reads it. Help links point at the maintained docs.
- **The test suite is green and order-independent**, and there is a headless GUI
  harness (`tests/ui/`) with a synthetic music library.

**Earlier:**

- **Library Sync's Export Report button** is wired — a real "📤 Export Report" button
  in the Library Sync workspace saves HTML, JSON, or CSV reports.
- **Metadata settings honesty** — the Settings dialog now shows Spotify and Gracenote
  visibly but disabled, with a tooltip explaining they aren't implemented, instead of
  silently omitting them or listing them as if they worked.
- **The Duplicate Pair Report crash is fixed.** `build_duplicate_pair_report()` was
  calling `.items()` on the list returned by `_build_metadata_buckets()`, so it
  raised `AttributeError` on every single invocation. It now looks each track's
  bucket up by path and reads the bucket's metadata key, matching the
  `tuple[str, str] | None` type `DuplicatePairReport` already declared. The HTML
  pair report, which was unreachable as a result, now renders too.
- **The Cluster Quality Report crash is fixed.** The dialog called
  `_create_cluster_card()`; the real method is `_build_cluster_card()`. The
  per-cluster breakdown now renders against real results.
- **`plugins/` and `controllers/` no longer import Tkinter.** The legacy
  `MetadataServiceConfigFrame` moved out of `plugins/acoustid_plugin.py` into
  `metadata_config_frame.py` at the repo root (alongside `library_sync_review.py`,
  the project's other legacy Tkinter panel), and
  `controllers/library_index_controller.generate_index()` no longer pops its own
  `messagebox` — it returns the output path and the legacy GUI reports it. Both
  modules now import headlessly, which un-breaks
  `tests/test_musicbrainz_service.py` on machines without a system Tkinter.
- **The "Open Visual Graph" button navigates.** `WorkspaceBase` gained a
  `navigate_requested(key)` signal that the main window connects for every
  workspace; the Clustered workspace emits it instead of showing a message box
  telling you to click the sidebar yourself.
- **The 3-D browser graph was generating invalid JavaScript, and is fixed.**
  `_render_html()` substituted only the `/*__CLUSTER_DATA__*/` comment, leaving
  the template's trailing `null` in place and emitting
  `const CLUSTER_DATA = {...}null;` — a syntax error that stopped the whole page
  running. Substitution now consumes the `null`, `_render_html()` raises if the
  placeholder is missing rather than silently shipping sample data, and there are
  two new tests: one asserting the injected assignment is syntactically clean and
  one asserting the loud failure. (Every other test in that module is a substring
  check, which is exactly why this shipped unnoticed.)
- **The in-app 2-D interactive map is built and reachable.** The Music Graph
  workspace now embeds a live scatter plot alongside the cluster legend and a track
  details panel, instead of only launching a browser page. Pan / rectangle / lasso
  selection (Ctrl or Shift to add), hover to identify, click for full tags and cover
  art, click a legend row to select a whole cluster, and send a selection straight to
  the Player or export it as CSV or `.m3u`. The four widgets that had been sitting
  unused were substantially reworked rather than merely connected — see the report
  in the commit for what was wrong with each.
- **Cluster data is no longer misaligned when downsampled.** For libraries above
  5,000 tracks, `clustered_playlists` wrote subsetted `X_2d`/`X_3d` alongside
  *full-length* `labels` and `tracks`. That made `cluster_info.json` internally
  inconsistent: the 3-D generator rejected it outright ("Length mismatch"), so the
  browser graph simply never worked on a large library, and any index-based lookup
  would have mapped a point to the wrong song. Labels and tracks are now subset with
  the coordinates, and the original positions are recorded in `X_indices`.
- **The clustering wizard's feature checkboxes all work.** Harmonic content (chroma),
  brightness (spectral centroid), energy (RMS) and percussive density (onset rate)
  are now computed alongside timbre and tempo, on both the librosa and Essentia
  engines, and only the ticked ones are computed. The wizard's normalization choice
  (standard / min-max / robust) is honoured too instead of always using standard
  scaling. The feature cache is keyed by selection *and* engine — previously one
  `features.npy` served every run, which would have stacked mismatched vectors the
  moment a checkbox changed, and silently mixed librosa and Essentia values before
  that. The legacy MFCC+tempo/librosa combination keeps the original filename so
  existing caches still load.
- **Canonical genre mapping is in the modern app.** The Genre Normalizer workspace
  is now two tabs: the existing MusicBrainz gap-filler (with its two non-functional
  controls removed) and a new "Normalize with a mapping" tab — scan raw genres, build
  a raw → canonical mapping with an LLM's help, save/reload it, **preview** every file
  that would change, then apply exactly that plan. The backend
  (`controllers/normalize_controller.py`) gained `plan_genre_normalization` /
  `apply_genre_changes` with injectable discovery, reader and writer, so it is
  Qt-free and tested without mutagen or musicbrainzngs (17 tests). Two legacy
  behaviours were corrected in the port: the old "Apply" wrote with no preview, and
  it wrote the prompt's `["invalid"]` marker into files as a literal genre; both
  `null` and `["invalid"]` now mean "drop." The legacy Tkinter panel is untouched.
- **The 3-D graph template was upgraded to the `alphadex3` design.** Adds spread
  control (slider + 1×/10×/30×/50×/100× presets), axes / grid / orbit-ring
  toggles, a richer tooltip, a vignette, and JSON import, on top of the existing
  orbit / hover / select / CSV+M3U export. Cluster membership is now bucketed in a
  single pass instead of rescanning the full label array once per cluster, which
  had made legend and centroid construction O(clusters x points).
- **Duplicate Finder no longer plans "Unknown" over a real year.** The shared
  metadata reader turns a missing year into the placeholder `"Unknown"`, which the
  consolidation planner treated as a real value: an album-context tag source with no
  year produced a planned change like `year: 2019 → Unknown` on the winner, and the
  placeholder also blocked a real provided year from filling in. The placeholder now
  counts as "no value" when merging/planning, and the executor never writes it.
- **The Tag Fixer skips unreadable files again.** Moving its tag reads onto the
  shared reader (which swallows read errors) had silently dropped the "skip corrupt
  files" behaviour, so corrupt files showed up as phantom "unmatched" rows and were
  sent for metadata lookups. Read failures are now detected and skipped (and logged).
- **Fingerprint-cache robustness.** A cache DB file that exists without its table
  (e.g. created moments earlier by the background writer) is now an empty cache
  rather than a `no such table` error; the writer creates the schema before it is
  registered; and a fingerprint computed or stored moments ago is served from an
  in-process memo (validated by mtime/size) instead of being recomputed because the
  batched write hadn't reached disk yet.

---

## Settled decisions (recorded so they don't get re-raised)

- **The two indexer engines stay as they are.** Library Sync runs a vendored copy of
  the scanning/fingerprinting engine (`library_sync_indexer_engine/`) rather than the
  root-level modules the Indexer and Duplicate Finder use. This is a leftover from
  the project's two-generation history — an earlier program built around the legacy
  Tkinter interface, and the current iteration that became the modern Qt app — not a
  deliberate design decision. **The root-level engine is canonical for all new work;
  the vendored copy is left alone unless we're specifically making changes to it.**
  Unifying them is explicitly *not* planned; the only thing to remember is that a fix
  to a root module doesn't reach Library Sync automatically, so port it across
  deliberately if Library Sync specifically needs it (the fingerprint-distance fix
  above is exactly such a case). See **architecture.md** for the full picture.

---

## Library Sync

- **Decide the fate of the two Library Sync front-ends** *(Housekeeping).* A
  feature-flagged legacy Tkinter review panel (`library_sync_review.py`) and the
  modern Qt workspace both exist as separate, non-shared implementations of the same
  review-first workflow. If the Tkinter one is no longer meant to be used, retiring
  it removes a second place bugs can hide; if it's meant to stay as a fallback, that
  should be a stated decision rather than an accident of history.

- **Bulk flagging** *(Designed, not started).* Today you flag incoming tracks for
  copy/replace one at a time. A natural enhancement is to select many tracks at once
  and flag them in a single action (for example, "accept all the quality upgrades").

- **Persistent flag sessions** *(Designed, not started).* "Save Session" now remembers
  the two folders and the threshold, and flags survive a re-scan where they still
  apply, but the flags themselves still live only in memory and are gone after a
  restart. Saving them must be done carefully, so that saved flags don't end up
  pointing at matches that no longer exist (the re-scan carry-over logic is the model
  to follow).

- **Flag history and undo** *(Designed, not started).* The ability to step back through
  your flagging decisions.

- **Row highlighting** *(Designed, not started).* Coloring flagged rows in the review
  list (one color for "copy," another for "replace") so the state is visible at a
  glance, beyond the text in the Flag column.

## Duplicates

- **A per-group accept/skip review stage** *(Designed, not started).* Today a plan
  with review-flagged groups is either executed in full (an explicit override) or not
  at all. Accepting or skipping flagged groups one by one is the planned middle way.

- **Matching time-stretched or trimmed edits** *(Designed, not started).* Near-
  duplicate gating doesn't normalize duration or tempo before comparing fingerprints
  (the scan log says "duration/tempo normalization deferred"), so a sped-up radio edit
  may not match its album version.

## Playlists, Clustering & the Visual Music Graph

The foundation and the in-app map are built; the remaining work is the editing and
tuning layer on top of them.

- **Retire the dead `clustered.py` workspace** *(Housekeeping).* It is unreachable and
  now says so in its docstring; deleting it is one of the open decisions above.

- **Playlists straight from a clustering run** *(Designed, not started).* Today a run
  writes only the analysis, and you export a cluster as a playlist from the Music
  Graph. A previewed "one playlist per cluster" export would let the wizard's disabled
  "Create M3U playlists" option come back.

- **A choice of "map flattening" methods in the interface** *(Partially built).* The
  app already flattens many-dimensional sound data into a 2-D or 3-D picture. A planned
  enhancement lets you *choose and switch between* methods (the wizard's disabled
  PCA / t-SNE / UMAP choice) without re-running the whole clustering.

- **Live parameter tuning** *(Designed, not started).* Nudge a setting — "eight groups
  instead of six" — and watch the map re-form without recomputing every track's sound
  profile.

- **Advanced selection tools** *(Partially built).* Lasso and rectangle selection work
  in the in-app map, with Ctrl/Shift to add. Still to come: select-by-distance-from-a-
  point and filtering the map by metadata (artist, genre).

- **In-map cluster editing** *(Designed, not started).* Merging two clusters, moving a
  track between clusters, or splitting a selection into its own sub-cluster.

- **Suggestion engine** *(Designed, not started).* Plain-language advice when a
  grouping looks weak — "these groups overlap a lot; try making fewer of them."

- **One-click quality-report export** *(Designed, not started).* Saving the clustering
  quality report as a web page or PDF.

- **Genre and year playlists in the modern app** *(Backend ready, not connected).*
  `playlist_engine.export_genre_playlists` exists, but the Qt Playlist Generator only
  offers folder, tempo/energy, Auto-DJ and repair.

- **A genuine chronological/"year-gap" playlist tool doesn't exist** *(Designed, not
  started).* Earlier documentation described a "year-gap assistant." No such feature
  exists in either app; it would be new work. (Don't confuse it with the legacy app's
  "Year Assistant," which fills in *missing* year tags.)

## Metadata lookups (Tag Fixer & Genres)

- **MusicBrainz as an automatic fallback in the Tag Fixer** *(Partially built).* A
  MusicBrainz text search (`plugins/musicbrainz_plugin.py`) now joins the scan — but
  only when "MusicBrainz" is the service chosen in Settings. It doesn't yet step in by
  itself when AcoustID can't identify a file, and `MusicBrainzService` (the richer
  lookup) still doesn't subclass `MetadataPlugin`.

- **Spotify integration** *(Designed, not started).* Listed in Settings, honestly
  marked unavailable; the connection is an empty placeholder that returns nothing.
  It needs authenticated, partnership-style access, unlike the working providers.

- **Gracenote integration** *(Designed, not started).* Same situation as Spotify.

- **Discogs integration** *(Designed, not started).* Mentioned as a desirable future
  source; no work started.

> The providers that **do** work today are AcoustID, MusicBrainz, and Last.fm (Last.fm
> needs the `LASTFM_API_KEY` environment variable until the Settings bug above is
> fixed).

## Other / housekeeping

- **Root directory reorganization** *(Spec written, deliberately deferred).* ~60
  files sit loose in the repo root — crash logs, old diff dumps, one-off scripts,
  and ~35 real backend modules with no grouping. A two-phase plan (junk removal,
  then a domain-grouped `alphadex/` package) is written up in
  **[reorganization_spec.md](reorganization_spec.md)**, including which files are safe
  to delete and which are misfiled rather than junk. (Of the Tkinter-in-backend
  modules it found, `cluster_graph_panel.py` is a legacy Tk widget that still imports
  Tkinter at the top; `update_genres.py` now imports it only in its script `main()`.)
  Do not execute without being explicitly asked.

- **Small docstring mismatch** *(Housekeeping, low risk).*
  `chromaprint_utils.ensure_tool()` says it raises `RuntimeError` but raises
  `FingerprintError`. (The `config.load_config()` docstring mismatch noted here before
  is fixed: there is now one set of defaults, applied the same way on every path.)

- **Test-suite stubs** *(Mostly done).* The order dependence is fixed: stubs are scoped
  through `tests/_helpers/module_isolation.py` — `isolated_modules()` installs them for
  one test and restores every touched `sys.modules` entry afterwards, and
  `stub_if_unavailable()` (the only module-level use) stubs an optional dependency only
  when it isn't installed, so it can never shadow the real package. The suite gives the same result in
  default, reversed and shuffled order, and the `tests/test_cache.py` race (a real
  `cache_prewarmer` bug) is fixed. The fake `mutagen` is now **shared**:
  `make_mutagen_stub()` in the same helper is the only inline mutagen stub left, and
  the other tests that touch mutagen import the real package (it is in
  `requirements.txt`). That
  made the old **`mutagen_stub/` package dead**; it was deleted on 2026-11-07. `tests/conftest.py` still stubs `pydub` when it isn't installed, and gives
  every test an isolated home folder and config path.

- **A global activity indicator** *(Backend ready, not connected).* The JobRegistry
  emits `jobs_changed(count)` for exactly this; nothing displays it yet.

- **Tidal-dl sync** *(Designed, not started).* An old idea to pull music from Tidal;
  no interface or workflow exists. Dormant unless revived.

- **Small loose ends** *(Designed, not started).* `Sidebar.set_badge()` is fully built
  but never called; dead modules (`gui/widgets/splash.py`,
  `gui/widgets/interactive_3d_scatter.py`, `gui/workspaces/clustered.py`) await the
  delete decision above; jumping to the map automatically after a clustering run is
  still just an idea. (The Linux-only "View Playlists" button and the broken Help link
  are gone.)

---

## How to read this list if you're returning to the project

If you've been away and want the shortest path back to momentum:

1. **Fix the Known bugs first** — they are small and each one makes an existing
   feature tell the truth (the Last.fm key and the Library Sync fingerprint metric
   matter most).
2. **Decide the open questions** in `docs/redesign/README.md` §6; several cleanups
   (dead files, prototype dialogs, the disabled controls) wait on them.
3. **Library Sync bulk flagging and persistent sessions** remain the main friction in
   the app's strongest workflow.
4. **Treat the Visual Music Graph's deeper interactivity** — live re-tuning, in-map
   editing, the suggestion engine — as the big, deliberate project it always was.
5. **Everything under "Other / housekeeping" is safe to ignore** until you're already
   touching the relevant file.
