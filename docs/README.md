# AlphaDEX Documentation

A concept-driven guide to how AlphaDEX works — front-end and back-end — written in
plain language for anyone, not just programmers. Start at the top and follow the order
below.

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
