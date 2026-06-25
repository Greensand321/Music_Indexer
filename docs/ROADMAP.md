# AlphaDEX — Roadmap (What's Planned but Not Yet Built)

*This is the honest "what's left" document. Everything below is either partially built,
designed but not wired up, or planned for the future. It is written in plain language
so anyone can understand the state of the project at a glance. Items are grouped by
area and, within each area, roughly ordered from "smallest / most ready" to "largest /
most speculative."*

*Last reviewed: 2026-06-25.*

---

## Quick status legend

- **Backend ready, not connected** — the engine exists and works; only the visible
  button or final wiring is missing. These are the cheapest wins.
- **Partially built** — some of it works; meaningful pieces remain.
- **Designed, not started** — there's a clear vision but no working code yet.

---

## Library Sync

- **Export Report button** *(Backend ready, not connected).* Library Sync can already
  generate a complete, shareable report of a review session — every match, its status,
  confidence, quality verdict, and your flags and notes — as a web page or data file.
  The report machinery is built and tested. The only missing piece is a button in the
  interface to trigger it. This is the single most "almost done" item in the whole
  project.

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

This is the area with the largest gap between vision and current reality. The original
design imagined a deeply interactive analysis studio; today the foundation (feature
extraction, two clustering methods, quality scores, and a visual map) is built, and the
richer interactive layer is the outstanding work.

- **More sound features in clustering** *(Partially built).* The clustering wizard
  shows checkboxes for harmonic content, brightness, and percussive density, but the
  engine currently groups tracks using **timbre and tempo** only. Wiring the remaining
  features into the actual grouping would make the clusters richer and more musical.

- **A choice of "map flattening" methods in the interface** *(Partially built).* The
  app already flattens many-dimensional sound data into a 2-D or 3-D picture using
  established techniques. A planned enhancement lets you *choose and switch between*
  different flattening methods from the interface to see your collection from different
  angles, without re-running the whole clustering.

- **Live parameter tuning** *(Designed, not started).* The ability to nudge a setting —
  say, "make eight groups instead of six" — and watch the map re-form in near real time,
  without recomputing every track's sound profile from scratch.

- **Advanced selection tools** *(Designed, not started).* Richer ways to pick tracks off
  the visual map: free-form lasso, rectangle, distance-from-a-point, and filtering by
  metadata (artist, genre).

- **In-map cluster editing** *(Designed, not started).* Hands-on refinement of the
  groups after the fact: merging two clusters, moving a track from one cluster to
  another, or splitting a selection into its own sub-cluster.

- **Suggestion engine** *(Designed, not started).* Friendly, plain-language advice when
  a grouping looks weak — for example, "these groups overlap a lot; try making fewer of
  them."

- **One-click quality-report export** *(Designed, not started).* Saving the clustering
  quality report as a shareable document (web page or PDF).

## Metadata lookups (Tag Fixer & Genres)

- **Spotify integration** *(Designed, not started).* Spotify appears as an option in
  the settings, and the supporting library is already installed, but the connection is
  an empty placeholder that returns nothing. Bringing it to life is future work — and
  is more involved than the others because it requires authenticated, partnership-style
  access rather than the open access the working providers enjoy.

- **Gracenote integration** *(Designed, not started).* Same situation as Spotify: listed
  as an option, but currently an empty placeholder.

- **Discogs integration** *(Designed, not started).* Mentioned as a desirable future
  source; no work started.

> The providers that **do** work fully today are AcoustID, MusicBrainz, and Last.fm.
> A worthwhile near-term task, separate from building the missing providers, is simply
> to make the settings screen *honest* — clearly marking Spotify and Gracenote as
> unavailable so they don't appear to be working options.

## Other / housekeeping

- **Tidal-dl sync** *(Designed, not started).* An old idea to pull music from the Tidal
  service was referenced in early notes, but no interface or workflow for it exists.
  Consider this dormant unless revived.

- **Startup polish** *(Partially built).* A handful of low-priority refinements to make
  the app's opening sequence even smoother on very large libraries were identified and
  deferred. None of them block normal use; they are nice-to-haves.

- **Small loose ends** *(Designed, not started).* Minor conveniences noted during
  development, such as automatically jumping to the visual map right after a clustering
  run finishes.

---

## How to read this list if you're returning to the project

If you've been away and want the shortest path back to momentum:

1. **Start with the "Backend ready, not connected" items** — chiefly the Library Sync
   Export Report button. They deliver visible value for very little effort because the
   hard part is already done.
2. **Then make the metadata settings honest** — mark the non-working providers as
   unavailable so the app stops implying capabilities it doesn't have.
3. **Treat the Visual Music Graph interactivity as the big, deliberate project** — it's
   the area with the most outstanding work and the most upside, and it deserves to be
   scheduled as real feature work rather than squeezed in.
