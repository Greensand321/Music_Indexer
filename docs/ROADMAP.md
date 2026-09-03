# AlphaDEX — Roadmap (What's Planned but Not Yet Built)

*This is the honest "what's left" document. Everything below is either partially built,
designed but not wired up, or planned for the future — plus a short list of actual
bugs found while re-verifying this document against the code. It is written in plain
language so anyone can understand the state of the project at a glance. Items are
grouped by area and, within each area, roughly ordered from "smallest / most ready" to
"largest / most speculative."*

*Last reviewed: 2026-09-03.*

---

## Quick status legend

- **Bug** — this used to work, or was believed to work, and currently doesn't. Fixing
  it restores existing functionality rather than building something new.
- **Backend ready, not connected** — the engine exists and works; only the visible
  button or final wiring is missing. These are the cheapest wins.
- **Partially built** — some of it works; meaningful pieces remain.
- **Designed, not started** — there's a clear vision but no working code yet.

## Recently completed

These items used to appear on this list and are done, so they've been removed from
the body below:

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
- **The 3-D graph template was upgraded to the `alphadex3` design.** Adds spread
  control (slider + 1×/10×/30×/50×/100× presets), axes / grid / orbit-ring
  toggles, a richer tooltip, a vignette, and JSON import, on top of the existing
  orbit / hover / select / CSV+M3U export. Cluster membership is now bucketed in a
  single pass instead of rescanning the full label array once per cluster, which
  had made legend and centroid construction O(clusters x points).

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
  deliberately if Library Sync specifically needs it. See **architecture.md** for the
  full picture.

---

## Library Sync

- **Wire or remove the "Replace All" / "Replace with Better" report actions**
  *(Backend ready, not connected — but only half-built).* The exported HTML review
  report renders three action buttons styled to look clickable; none of them do
  anything. The `ReviewAction` enum and `build_review_replacement_plan()` function
  that look like they were built to back these buttons exist in `library_sync.py`
  but have no caller anywhere in either app. Either wire the buttons to that
  function, or remove both the buttons and the dead backend function so the report
  doesn't imply a capability that isn't there.

- **Decide the fate of the two Library Sync front-ends** *(Housekeeping).* A
  feature-flagged legacy Tkinter review panel (`library_sync_review.py`) and the
  modern Qt workspace both exist as separate, non-shared implementations of the same
  review-first workflow. If the Tkinter one is no longer meant to be used, retiring
  it removes a second place bugs can hide; if it's meant to stay as a fallback, that
  should be a stated decision rather than an accident of history.

- **Bulk flagging** *(Designed, not started).* Today you flag incoming tracks for
  copy/replace one at a time. A natural enhancement is to select many tracks at once
  and flag them in a single action (for example, "accept all the quality upgrades").

- **Persistent flag sessions** *(Designed, not started).* Review flags currently live
  only in memory and are cleared when you re-scan or restart. A future option would let
  you save a review session and reload it later. (Note: this must be done carefully, so
  that saved flags don't end up pointing at matches that no longer exist — which is
  exactly why flags are session-only today.)

- **Flag history and undo** *(Designed, not started).* The ability to step back through
  your flagging decisions.

- **Row highlighting** *(Designed, not started).* Coloring flagged rows in the review
  list (one color for "copy," another for "replace") so the state is visible at a
  glance, beyond the small text badges used now.

## Playlists, Clustering & the Visual Music Graph

The foundation and the in-app map are now built; the remaining work is the richer
editing and tuning layer on top of them.

- **Retire the dead `clustered.py` workspace** *(Housekeeping).* An older, simpler
  clustering workspace still exists in the repository but is unreachable — the
  sidebar wires to `clustered_enhanced.py` instead. Leaving the old file in place
  risks a future contributor editing it and wondering why nothing changes.

- **A choice of "map flattening" methods in the interface** *(Partially built).* The
  app already flattens many-dimensional sound data into a 2-D or 3-D picture using
  established techniques. A planned enhancement lets you *choose and switch between*
  different flattening methods from the interface to see your collection from different
  angles, without re-running the whole clustering.

- **Live parameter tuning** *(Designed, not started).* The ability to nudge a setting —
  say, "make eight groups instead of six" — and watch the map re-form in near real time,
  without recomputing every track's sound profile from scratch.

- **Advanced selection tools** *(Partially built).* Lasso and rectangle selection
  now work in the in-app map, with Ctrl/Shift to add to a selection. Still to come:
  select-by-distance-from-a-point and filtering the map by metadata (artist, genre).

- **In-map cluster editing** *(Designed, not started).* Hands-on refinement of the
  groups after the fact: merging two clusters, moving a track from one cluster to
  another, or splitting a selection into its own sub-cluster.

- **Suggestion engine** *(Designed, not started).* Friendly, plain-language advice when
  a grouping looks weak — for example, "these groups overlap a lot; try making fewer of
  them."

- **One-click quality-report export** *(Designed, not started).* Saving the clustering
  quality report as a shareable document (web page or PDF). Now unblocked — the
  dialog it would export renders correctly again.

- **A genuine chronological/"year-gap" playlist tool doesn't exist** *(Designed, not
  started).* Earlier documentation described a "year-gap assistant" that helps build
  playlists telling a chronological story. No such feature exists in either app. If
  this is still wanted, it needs to be designed and built from scratch — it is not
  hiding somewhere unwired. (Don't confuse it with the legacy app's "Year Assistant,"
  which fills in *missing* year tags and has no relationship to playlist ordering.)

## Metadata lookups (Tag Fixer & Genres)

- **Bring MusicBrainz into the Tag Fixer's automatic scan as an independent source**
  *(Backend ready, not connected).* `MusicBrainzService` is a complete, working
  lookup implementation, but it doesn't inherit from the same plugin base class the
  Tag Fixer's automatic-discovery scan looks for, so it never contributes an
  independent identification — today it only ever appears nested inside an AcoustID
  match's enrichment step. Making it a first-class, independently-discoverable
  plugin would let the Tag Fixer fall back to MusicBrainz when AcoustID doesn't
  recognize a file.

- **Bring canonical genre-mapping into the modern app** *(Designed, not started, but
  a working reference implementation already exists in the legacy app).* The
  legacy Tkinter app has a real "map messy raw genres to a clean, chosen vocabulary"
  workflow (paste your raw genres into an LLM prompt, paste back a JSON mapping,
  apply it library-wide). The modern app's "Genre Normalizer" workspace does
  something much simpler — pick MusicBrainz's top 3 popular tags — and doesn't do
  any canonicalization at all. Porting the legacy mapping system forward (or
  designing a modern-app equivalent) is a real, user-visible gap; see
  **features/tag_fixer.md** for the full explanation of the naming collision.

- **Spotify integration** *(Designed, not started).* Spotify appears as an option in
  the settings (now honestly marked unavailable — see "Recently completed" above),
  and the supporting library is already installed, but the connection is an empty
  placeholder that returns nothing. Bringing it to life is future work — and
  is more involved than the others because it requires authenticated, partnership-style
  access rather than the open access the working providers enjoy.

- **Gracenote integration** *(Designed, not started).* Same situation as Spotify: listed
  as an option, now honestly marked unavailable, but currently an empty placeholder.

- **Discogs integration** *(Designed, not started).* Mentioned as a desirable future
  source; no work started.

> The providers that **do** work fully today are AcoustID, MusicBrainz, and Last.fm.
> The Settings screen is now honest about which of the five listed services are
> real — the remaining work here is building Spotify/Gracenote/Discogs themselves,
> not disclosure.

## Other / housekeeping

- **Fix known docstring/behavior mismatches** *(Housekeeping, low priority, low
  risk).* A handful of functions' docstrings no longer match what the code actually
  does — most notably `config.load_config()`, whose docstring claims it "returns an
  empty dict" on failure when it actually returns a specific ~25-key fallback
  dictionary that can drift out of sync with the normal-path defaults. Two separate
  `ensure_tool()` helpers (`fingerprint_generator.py`, `chromaprint_utils.py`) both
  claim in their docstring to raise `RuntimeError` but actually raise a custom
  `FingerprintError` — and the two modules each define their *own*, unrelated
  `FingerprintError` class with the same name, so `except FingerprintError` in code
  written against one module silently won't catch an error raised by the other.

- **Make the test suite's shared-dependency stubbing real** *(Housekeeping).*
  `CLAUDE.md` used to state that `tests/conftest.py` stubs `pydub`, `tkinter`, and
  "other heavy dependencies." In fact it only stubs `pydub` — there is no `tkinter`
  stub anywhere, and more than a dozen test files each build their own inline,
  near-duplicate fake `mutagen` module rather than sharing the dedicated
  (currently-unused) `mutagen_stub/` package. This also means the suite's pass/fail
  result can depend on execution order, since the ad-hoc stubs mutate global
  `sys.modules` state that bleeds between test files. `tests/test_cache.py` also has
  a real, order-independent race condition in its background cache-warming test.
  None of this blocks development, but a "the tests are green" claim currently
  depends on which files ran first.

- **Tidal-dl sync** *(Designed, not started).* An old idea to pull music from the Tidal
  service was referenced in early notes, but no interface or workflow for it exists.
  Consider this dormant unless revived.

- **Startup polish** *(Partially built).* A handful of low-priority refinements to make
  the app's opening sequence even smoother on very large libraries were identified and
  deferred. None of them block normal use; they are nice-to-haves.

- **Small loose ends** *(Designed, not started).* Minor conveniences noted during
  development, plus a few small dead ends worth cleaning up when someone's already in
  the area: `Sidebar.set_badge()` is fully built but never called by anything; the
  "View Playlists" folder-open button in the Clustered workspace only works on
  Linux (no Windows/macOS branch); a Help-screen documentation link points at an
  archived file that no longer exists at that path; automatically jumping to the
  visual map right after a clustering run finishes is still just an idea.

---

## How to read this list if you're returning to the project

If you've been away and want the shortest path back to momentum:

1. **Bring canonical genre mapping into the modern app.** It's the last piece of
   real functionality that exists only in the legacy Tkinter app, and a working
   reference implementation is already there to port from.
2. **Treat the Visual Music Graph's deeper interactivity — live re-tuning, in-map
   cluster editing, the suggestion engine — as the big, deliberate project it always
   was.** The map itself now exists to build on, which makes these additions rather
   than foundations.
3. **Everything under "Other / housekeeping" is safe to ignore indefinitely** — none
   of it blocks normal use of the app. Pick it up only when you're already touching
   the relevant file for another reason.

The bugs and gaps that used to head this list (Duplicate Pair Report, Cluster
Quality Report, the Tkinter imports in `plugins/` and `controllers/`, the broken 3-D
graph data injection, and the unreachable in-app map) are all fixed — see "Recently
completed" above.
