# AlphaDEX Documentation

A concept-driven guide to how AlphaDEX works — front-end and back-end — written in
plain language for anyone, not just programmers. Start at the top and follow the order
below.

## Known Gaps — What Still Needs Work

A quick-reference list of what is **not yet built**, ordered by how close each item is
to done. See **[ROADMAP.md](ROADMAP.md)** for the full details and effort estimates.

- **Library Sync — Export Report button** *(Backend ready, not connected — smallest
  outstanding win).* The report-generation code already exists and works
  (`export_report()` / `export_review_report_html()` in `library_sync_review_report.py`).
  Only the button to trigger it is missing.

- **Clustering — extra sound features** *(Partially built).* The wizard shows
  checkboxes for harmonic content, brightness, and percussive density, but the engine
  currently clusters on **timbre (MFCC) and tempo only**. Those checkboxes are UI
  scaffolding not yet wired into feature extraction.

- **Visual Music Graph — dimensionality-reduction selector** *(Partially built).* The
  app already computes PCA / t-SNE / UMAP reductions internally; what's missing is a
  control in the graph UI to choose and switch between them without re-running the whole
  clustering.

- **Visual Music Graph — interactive Phases 4–8** *(Designed, not started).* The 2D
  in-app and 3D browser visualizations are built, but the richer interaction layer is
  not: live parameter tuning, advanced selection tools (lasso, rectangle, metadata
  filter), in-map cluster editing (merge, re-assign, sub-cluster), a suggestion engine,
  and one-click quality-report export.

- **Metadata providers — Spotify & Gracenote** *(Designed, not started).* Both appear
  in Settings and `config.SUPPORTED_SERVICES`, and `spotipy` is installed, but the
  underlying functions in `metadata_service.py` return `{}`. Discogs is roadmap-only.
  Short-term fix: mark them as unavailable in the Settings UI so they stop looking like
  working options.

- **Library Sync — bulk flagging & session persistence** *(Designed, not started).*
  Tracks can only be flagged one at a time; there is no multi-select batch flag. Review
  flags also live in memory only and are cleared when you re-scan or restart.

## Read in this order

1. **[overview.md](overview.md)** — What AlphaDEX is, who it's for, the core principles
   (preview-first, quarantine-not-delete), and the handful of concepts (fingerprints,
   lossless vs. lossy, reserved folders, the two apps) you need before anything else.
2. **[architecture.md](architecture.md)** — How the pieces fit together behind the
   scenes: the three layers, background workers, the preview contract, the cache, and
   the workspaces.

## Feature deep-dives (read in any order)

- **[features/indexer.md](features/indexer.md)** — Reorganizing a messy library into
  clean Artist/Album/Track folders.
- **[features/duplicate_finder.md](features/duplicate_finder.md)** — Finding copies of
  the same song by sound, and safely clearing the extras.
- **[features/library_sync.md](features/library_sync.md)** — Merging an incoming folder
  into your library, with per-item review flags.
- **[features/playlists_and_clustering.md](features/playlists_and_clustering.md)** —
  Rule-based playlists, machine-learning clustering, and the Visual Music Graph.

## Forward-looking

- **[ROADMAP.md](ROADMAP.md)** — What's planned but not yet built, with an honest
  status for each item.

## Reference (kept from before)

- **[gui_inventory.md](gui_inventory.md)** — A detailed map of every screen and control
  in the modern app.
- **[INTEGRATION_TESTING_GUIDE.md](INTEGRATION_TESTING_GUIDE.md)** — Test procedures for
  the clustering and graph features.
- **[library_sync_per_item_review_testing.md](library_sync_per_item_review_testing.md)**
  — Manual test scenarios for the Library Sync review flags.

## Historical material

- **archive/** — Superseded planning notes, audit snapshots, and implementation logs.
  Kept for history; not maintained. Nothing here should be treated as current.
