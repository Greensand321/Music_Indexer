# AlphaDEX Documentation

A concept-driven guide to how AlphaDEX works — front-end and back-end — written in
plain language for anyone, not just programmers. Start at the top and follow the order
below.

## Known Gaps — What Still Needs Work

A quick-reference list of what's broken or not yet built, ordered by how urgent or
close-to-done each item is. See **[ROADMAP.md](ROADMAP.md)** for the full list,
including lower-priority housekeeping items.

- **Clustering — extra sound features** *(Partially built).* The wizard shows
  checkboxes for harmonic content, brightness, and percussive density, but the engine
  currently clusters on **timbre (MFCC) and tempo only**. Those checkboxes are UI
  scaffolding not yet wired into feature extraction.

- **Genre canonicalization only exists in the legacy app** *(Designed, not started
  in the modern app).* The Qt "Genre Normalizer" workspace just picks MusicBrainz's
  top 3 popular tags per track — it doesn't map messy genre variants to a clean,
  chosen vocabulary. That real mapping workflow only exists in the legacy Tkinter
  app today.

- **Library Sync — bulk flagging & session persistence** *(Designed, not started).*
  Tracks can only be flagged one at a time; there is no multi-select batch flag. Review
  flags also live in memory only and are cleared when you re-scan or restart.

- **Metadata providers — Spotify & Gracenote remain unbuilt** *(Designed, not
  started; the Settings UI is already honest about this).* Both are visibly listed
  and clearly disabled in Settings rather than pretending to work, but the
  underlying lookup functions still just return nothing. Discogs is roadmap-only.

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
- **[features/tag_fixer.md](features/tag_fixer.md)** — Looking up correct tags online
  and proposing corrections, plus the honest story on genre normalization.
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
