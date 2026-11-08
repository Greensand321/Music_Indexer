# AlphaDEX Documentation

A concept-driven guide to how AlphaDEX works — front-end and back-end — written in
plain language for anyone, not just programmers. Start at the top and follow the order
below.

## Known Gaps — What Still Needs Work

A quick-reference list of what's broken or not yet built, ordered by how urgent or
close-to-done each item is. The **Known bugs** section of **[ROADMAP.md](ROADMAP.md)**
is the single source of truth; it also lists the controls that are still switched off
and the lower-priority housekeeping items.

- **The legacy app's Tag Fixer "Reset Log" also wipes Library Sync's cache** *(Bug,
  legacy app only).* Both features keep state in the library's `Docs/.soundvault.db`,
  and the reset deletes the whole file without warning. This is the only open entry
  in ROADMAP.md's Known bugs. (The two bugs this list used to lead with — the Last.fm
  key saved in Settings being ignored, and Library Sync comparing compressed
  fingerprints — are fixed.)

- **Library Sync — bulk flagging & session persistence** *(Designed, not started).*
  Tracks can only be flagged one at a time; there is no multi-select batch flag. Review
  flags survive a re-scan where they still apply, but not a restart ("Save Session"
  remembers the folders and threshold, not the flags).

- **Metadata providers — Spotify & Gracenote remain unbuilt** *(Designed, not
  started; the Settings UI is already honest about this).* Both are visibly listed
  and clearly disabled in **Settings → Metadata services** rather than pretending to
  work, but the underlying lookup functions still just return nothing. Discogs is
  roadmap-only.

- **Controls that are still switched off** *(Disabled, no backend).* A few controls
  are visible but disabled with a tooltip saying why — for example the Indexer's
  "Cross-album scan (Phase C)", Library Sync's "Preset name", parts of the clustering
  wizard, and most of the options in **Duplicates → Fuzzy finder**. ROADMAP.md lists
  them all.

## Read in this order

1. **[overview.md](overview.md)** — What AlphaDEX is, who it's for, the core principles
   (preview-first, quarantine-not-delete), and the handful of concepts (fingerprints,
   lossless vs. lossy, reserved folders, the two apps) you need before anything else.
2. **[architecture.md](architecture.md)** — How the pieces fit together behind the
   scenes: the three layers, background workers, the preview contract, the cache, and
   how the app is laid out as a sidebar of pages and their sections. Places in the
   app are written **Page → Section** throughout these documents (for example
   **Metadata → Tag Fixer**).

## Feature deep-dives (read in any order)

- **[features/indexer.md](features/indexer.md)** — Reorganizing a messy library into
  clean Artist/Album/Track folders.
- **[features/duplicate_finder.md](features/duplicate_finder.md)** — Finding copies of
  the same song by sound, and safely clearing the extras.
- **[features/library_sync.md](features/library_sync.md)** — Merging an incoming folder
  into your library, with per-item review flags.
- **[features/tag_fixer.md](features/tag_fixer.md)** — Looking up correct tags online
  and proposing corrections, plus the honest story on genre normalization.
- **[features/playlists_and_clustering.md](features/playlists_and_clustering.md)** —
  Rule-based playlists, machine-learning clustering, and the Sound Map (Analyse · Map).

## Forward-looking

- **[ROADMAP.md](ROADMAP.md)** — What's planned but not yet built, with an honest
  status for each item.
- **[reorganization_spec.md](reorganization_spec.md)** — A deferred plan to clean up
  and reorganize the ~60 loose files sitting in the repo root. Not scheduled; do not
  execute until explicitly asked.

## Reference (kept from before)

- **[gui_inventory.md](gui_inventory.md)** — A detailed map of every screen and control
  in the modern app: the sidebar, every page and section, and every dialog.
- **[CONTRADICTING_UPDATES.md](CONTRADICTING_UPDATES.md)** — Every change that
  contradicts older documents, code comments or the original design (the page
  renames among them), recorded so the owner can walk any of them back.
- **[INTEGRATION_TESTING_GUIDE.md](INTEGRATION_TESTING_GUIDE.md)** — Test procedures for
  the clustering and graph features.
- **[library_sync_per_item_review_testing.md](library_sync_per_item_review_testing.md)**
  — Manual test scenarios for the Library Sync review flags.
- **[../tests/ui/README.md](../tests/ui/README.md)** — The headless GUI test harness:
  fixtures, the guards every GUI test gets, and the rules for writing background
  workers. (architecture.md has the short version.)

## Redesign groundwork

- **[redesign/README.md](redesign/README.md)** — The groundwork for the GUI redesign:
  the intent of every page, how the pages connect, the ~200-bug inventory from the fix
  phase and its status, the page structure that was proposed there and then adopted,
  and the decisions for the owner. Its analysis, and the per-area reports in
  `redesign/pages/`, describe the app *as it was before the fixes and the new page
  structure*, under the old page names (Similarity, Genre Normalizer, Clustered, Music
  Graph, Compression, Utilities); for how it behaves now, read the documents above.

## Historical material

- **archive/** — Superseded planning notes, audit snapshots, and implementation logs.
  Kept for history; not maintained. Nothing here should be treated as current.
