# AlphaDEX — Architecture

*How the pieces fit together behind the scenes. This document is conceptual: it
explains the shape of the system and the reasoning behind it, with a handful of
plain-text diagrams for the parts that are easier to see than to describe. If you
have read **overview.md**, you already know the vocabulary used here.*

*Last verified against the code: 2026-07-22.*

---

## The big picture: three layers

It helps to picture AlphaDEX as a building with three floors. Work flows down from
the top floor and results travel back up.

```
┌──────────────────────────────────────────────────────────────────┐
│  TOP FLOOR — the interface (gui/workspaces/*.py)                  │
│  Buttons, tables, progress bars, previews. Shows things, collects  │
│  choices. Knows nothing about how to fingerprint a song.           │
└───────────────────────────────┬────────────────────────────────────┘
                                 │ user clicks a button
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│  MIDDLE FLOOR — the waiter (QThread workers / controllers/*.py)    │
│  Runs the real work off the GUI thread, reports progress back      │
│  via signals. Does not decide anything; carries orders and food.   │
└───────────────────────────────┬────────────────────────────────────┘
                                 │ calls into
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│  GROUND FLOOR — the engine (music_indexer_api.py, library_sync.py, │
│  duplicate_consolidation.py, clustered_playlists.py, tag_fixer.py, │
│  fingerprint_cache.py, …)                                          │
│  Pure Python. No Qt, no Tkinter. Testable and runnable headless.   │
└──────────────────────────────────────────────────────────────────┘
```

1. **The top floor — what you click (the interface).**
   This is everything you see: the buttons, the lists of tracks, the progress bars,
   the previews. Its only jobs are to *show* you things and to *collect* your
   choices. It deliberately knows nothing about how to actually fingerprint a song or
   move a file.

2. **The middle floor — the middleman (controllers and workers).**
   When you press a button, the interface hands the request to a middleman. The
   middleman's job is to run the actual work in the background, keep an eye on its
   progress, and carry the results back up to the interface when it's done. Think of
   it as the **waiter in a restaurant**: the waiter takes your order, walks it to the
   kitchen, and brings the food back — but the waiter does not cook.

3. **The ground floor — the engine (backend modules).**
   This is the kitchen. Here live the modules that do the genuine, heavy work:
   reading files, computing fingerprints, comparing audio, calculating where every
   track should go, and writing the final reports. These modules are pure machinery.

### Why keep the floors separate?

This separation is one of the most important design decisions in the whole project,
and it exists for two concrete reasons:

- **The interface must never contain the real logic.** If the rule for "which
  duplicate wins" were buried inside a button, you could never test it, reuse it, or
  trust it. By keeping all the real decisions on the ground floor, those decisions
  can be checked independently and reused by both versions of the app.
- **The engine must be able to run without any interface at all.** The ground-floor
  modules are written so they can be exercised and tested with no screen attached —
  the way you might bench-test a car engine before it ever goes into a car. This is
  what makes the automated test suite possible and what let the team build an
  entirely new interface (the modern app) on top of the *same* unchanged engine.

When you read the rule "business logic lives in the backend; the interface is
display-only," this three-floor picture is what it is protecting.

**One confirmed crack in the wall, worth knowing about:** `controllers/library_index_controller.py`
is a backend/controller module that directly imports and calls `tkinter.messagebox`
to show a result popup. It is the one place in the codebase that breaks the "no
Tkinter in backend modules" rule stated later in this document. It is a small,
self-contained violation (one tool, "Generate Library Index"), not a sign the rule
has broken down elsewhere — but a future contributor extending that file should
route the message back through a `log_callback` instead of adding more direct UI
calls.

## Why long jobs run "in the background"

Some of AlphaDEX's jobs take a long time — fingerprinting ten thousand songs can run
for many minutes. If the app tried to do that work the moment you clicked the button,
using the same attention it uses to draw the screen, the entire window would freeze.
Buttons wouldn't respond, the progress bar wouldn't move, and the app would look
crashed even though it was busy.

To avoid this, every long job is handed off to a **background worker** — a separate
line of work that runs alongside the interface instead of blocking it. The interface
stays awake and responsive; the worker grinds away on the slow task; and when the
worker has news (progress so far, or a finished result), it passes that news back to
the interface.

The crucial subtlety is *how* the news gets back. The screen can only be safely
updated by the part of the program that owns it — like a shared kitchen counter where
only one person is allowed to actually write on the notepad. A background worker
never reaches over and scribbles on the screen directly. Instead it **leaves a note
to be picked up** by the interface at a safe moment, and the interface updates itself
from that note. In the modern app this hand-off happens through a built-in
"signal" mechanism; in the legacy app it happens through a "schedule this for later"
call. The principle is identical: **workers report; only the interface touches the
screen.** This discipline is what keeps the app from corrupting its own display.

A concrete illustration from the Player workspace, preserved because it states the
rule precisely: album-art loading runs on a background `_ArtLoader` thread that
"emits a `QImage` (NOT a `QPixmap`) so the caller can convert to `QPixmap` on the
main/GUI thread. Creating `QPixmap` off the GUI thread is undefined behaviour in Qt
and silently produces null or broken pixmaps on Windows." (`gui/workspaces/player.py`)
— exactly the kind of platform-specific trap this discipline exists to avoid.

## The preview-first contract, structurally

**overview.md** introduced "look before you leap" as a promise to the user. Here is
how that promise is actually *built into the structure* rather than just being good
intentions.

```
   ┌───────────────┐        ┌──────────────┐        ┌───────────────┐
   │   DRY RUN      │  ──►   │   PREVIEW     │  ──►   │   EXECUTE      │
   │  compute the   │        │  human reads  │        │  carry out the │
   │  full plan;    │        │  an HTML      │        │  approved plan;│
   │  touch no file │        │  report       │        │  log every step│
   └───────────────┘        └──────────────┘        └───────────────┘
     duplicate_consolidation.py   you, in a browser    duplicate_consolidation_executor.py
     library_sync.compute_…       (Docs/*.html)         library_sync.execute_plan()
     music_indexer_api.build_…                          music_indexer_api.apply_indexer_moves()
```

Every major workflow is physically split into a **dry run** and an **execution**,
and these are separate operations, usually living in separate modules or files, that
produce different things:

- The **dry run** computes the entire plan — every move, every rename, every
  replacement, every deletion — and pours the result into a human-readable report
  (typically an HTML page). It is incapable of changing a single file, because the
  code that would touch the disk simply isn't run in this mode.
- The **execution** takes that approved plan and carries it out, logging every action
  as it goes.

Because these are genuinely different operations rather than one operation with a
checkbox, there is no accidental path by which a preview could quietly start moving
files. A returning developer should treat this split as sacred: features are expected
to preview first, and anything that mutates the library must be reachable only after
an explicit confirmation. A coordinator assembles the previews from each stage of a
job into one combined report, so even multi-step workflows show you the whole picture
before you commit.

The Duplicate Finder's executor states this contract explicitly in its own module
docstring, and it is worth quoting verbatim because it is the clearest statement of
the safety philosophy anywhere in the codebase:

> "The execution pipeline is intentionally conservative:
> - Playlists are backed up before edits and validated after rewrites.
> - A cancellation request stops processing after the current operation.
> - Errors stop subsequent steps while preserving any backups already written.
> - Every run produces machine-readable and human-readable reports."
> — `duplicate_consolidation_executor.py`

That executor carries out its work in seven strictly ordered steps — backup
playlists, rewrite playlists, *validate* the rewrite, apply artwork, apply metadata,
quarantine/delete losers, clean up redundant name suffixes — specifically so that
playlists are never left pointing at a file that has already been moved or deleted.
If a rewritten playlist still references a file that's about to be quarantined, the
executor refuses to continue ("Loser references remain after rewrite; aborting
cleanup") rather than risk a dangling reference.

Library Sync's engine states an equivalent promise for its own export path — a
smaller but related discipline of *computing* a thing fully before ever touching
disk:

> "The export routine deliberately avoids any move/replace logic and only writes the
> requested report file after the caller has previewed counts."
> — `library_sync_review_report.py`, `export_report()`

A companion rule travels with this one: **no silent data loss.** Anything that moves
or deletes a file writes a record of what it did into the `Docs` folder, and when a
decision is uncertain the app leans toward *quarantine* (set the file aside) rather
than *delete* (destroy it). The Duplicate Finder's executor enforces this with a
double confirmation gate — deletion only happens if the caller both sets
`allow_deletion=True` *and* explicitly passes `confirm_deletion=True` — and refuses
to execute at all if the plan being run no longer matches the plan that was
previewed (a "plan signature" mismatch), or if the library has visibly changed on
disk since the preview was captured.

**One thing the indexer deliberately does *not* do**, worth stating because it
clarifies the boundary between the Indexer and the Duplicate Finder: the indexer's
`find_duplicates()` function is a stub that always returns an empty list. Its own
docstring explains why:

> "Duplicate detection and keep/winner ranking are intentionally excluded from the
> indexer. The indexer treats every file independently and relies on deterministic
> routing plus collision handling (suffixing/buckets)."
> — `music_indexer_api.py`, `find_duplicates()`

In other words, the Indexer's job is filing, not judging — deciding "these two files
are the same song, keep the better one" is the Duplicate Finder's job alone. A GUI
hook that calls `find_duplicates()` still exists in the legacy app, but it can never
find anything; this is a real limitation and not just an unfinished stub someone
forgot to wire up.

## The fingerprint cache: paying a hard cost only once

Computing an audio fingerprint is expensive — it involves decoding the sound and
running it through analysis, taking real seconds per file. Doing that every single
time you scanned your library would be unbearable on a large collection.

So AlphaDEX keeps a **cache**: a small local record (stored as a single SQLite file
inside the `Docs` folder) that remembers the fingerprint it computed for each file.
The next time it encounters that same file, it reuses the remembered fingerprint
instead of recomputing it. The first scan of a big library is slow; the second is
dramatically faster, because almost everything is already remembered.

The clever part is knowing when a remembered fingerprint has gone stale. The cache
records, alongside each fingerprint, the file's **size and last-modified time**. If
either of those changes — meaning the file was edited or replaced — the cache
recognizes the entry as out of date and recomputes the fingerprint. This way the
speed-up never comes at the cost of using a wrong, outdated fingerprint. To keep the
cache from becoming a bottleneck itself, writes are batched and queued by a
dedicated background `FingerprintWriter` thread rather than performed one
painstaking entry at a time, and reads happen over a separate `PRAGMA query_only`
connection so a slow write never blocks a fast read.

### A documented architectural quirk: two indexer engines

This is worth calling out plainly, because it is easy to miss and it affects how
much you can trust "I fixed it in `music_indexer_api.py`" to mean "I fixed it
everywhere."

There are, today, **two separate copies of the indexing/fingerprinting engine** in
this repository:

```
     the "real" engine                the vendored engine
  ┌────────────────────────┐      ┌───────────────────────────────────┐
  │ music_indexer_api.py    │      │ library_sync_indexer_engine/       │
  │ fingerprint_cache.py     │      │   indexer_engine/                  │
  │ near_duplicate_detector.py│      │     music_indexer_api.py           │
  │ config.py                 │      │     fingerprint_cache.py           │
  │                            │      │     near_duplicate_detector.py     │
  │ used by: the Indexer,     │      │     config.py                      │
  │ Duplicate Finder, and the │      │                                     │
  │ legacy GUI directly       │      │ used by: library_sync.py only      │
  └────────────────────────┘      └───────────────────────────────────┘
```

`library_sync.py` does not import the root-level `music_indexer_api.py` or
`fingerprint_cache.py` at all. It imports its own vendored copy under
`library_sync_indexer_engine/indexer_engine/`, loaded dynamically by
`library_sync_indexer_engine/__init__.py`. The two copies started from the same
source but have since **drifted apart**: the vendored `fingerprint_cache.py` is 122
lines with no background writer thread, no WAL mode, and none of the extended
metadata columns (bitrate, codec, embedded tags, artwork) that the root
`fingerprint_cache.py` has grown to 735 lines to support. The vendored
`near_duplicate_detector.py` still uses an older, unindexed matching algorithm. The
vendored `music_indexer_api.py` is missing a few refinements (like richer
collision-decision logging) that the root copy has picked up since.

No comment anywhere in the codebase explains *why* this split exists — it isn't a
documented design decision, and searching the project's git history turns up no
earlier commit that explains it either. The one piece of evidence that someone
noticed and cared about the drift is a single test
(`tests/test_sanitize.py::test_engine_sanitize_matches_api_sanitize`) that loads both
copies of `sanitize()` and asserts they still agree — a guard against exactly one
function drifting, while the surrounding modules around it have already drifted
regardless.

**What this means in practice:** if you fix a bug or add a feature to the root
`fingerprint_cache.py` or `near_duplicate_detector.py`, **Library Sync will not see
it** unless you also apply the change to the vendored copy under
`library_sync_indexer_engine/indexer_engine/`. This is the single most important
"gotcha" for anyone extending the fingerprinting/caching layer, and it is tracked as
an open item in **ROADMAP.md**.

A smaller, related gotcha: Tag Fixer and the Library Sync review panel both store
their state in the same file, `Docs/.soundvault.db`, inside the library root. The
legacy GUI's "Reset Log" action for Tag Fixer deletes that entire file — which also
silently wipes Library Sync's fingerprint cache for that same library, with no
warning that the two features share a database file.

## One place for settings

Every preference in AlphaDEX — your library location, how strict the duplicate
matching should be, your online-service keys — is read from and written to **one
central file** (the `.soundvault_config.json` file introduced in the overview). No
part of the app reads or writes that file on its own; everything goes through a single
"load the settings" / "save the settings" pathway (`config.load_config()` /
`config.save_config()`).

The benefit is consistency. Because there is exactly one door in and one door out,
different parts of the app can never drift into disagreeing about what the current
settings are. If the duplicate-finder and the library-sync feature both need to know
your matching sensitivity, they are guaranteed to be reading the same value from the
same place. The file is deliberately a plain, human-readable text file, which makes it
easy to inspect or hand-edit when something looks wrong, and avoids the fragility of a
heavier database for what is really just a small bag of preferences.

One accuracy note for anyone reading `config.py` directly: its `load_config()`
docstring says it "returns an empty dict if the file does not exist or can't be
read," but that's not what the code actually does — on any read failure it returns a
specific ~25-key fallback dictionary of hard-coded defaults, kept as a hand-maintained
duplicate of the normal-path defaults a few lines above it. The two default sets can
drift out of sync with each other (some keys exist in one but not the other), so if a
setting seems to have "the wrong default," check whether you're looking at the
normal-load path or the exception-fallback path.

## Format priority: a single, auditable rule

When two files are the same song, which one should win? AlphaDEX answers this with a
fixed **format priority**: lossless formats outrank lossy ones, so a FLAC beats a WAV
beats an MP3 for the same recording. The reason is simply quality — the lossless file
preserves more of the original sound.

What matters architecturally is that this preference is written down as plain
settings in one spot, not scattered as guesswork across the code. That makes it easy
to *see* the rule, easy to *audit* whether the app is behaving according to it, and
easy to *change* if your priorities differ. Wherever the app must break a tie between
duplicates, it consults this one ranking. (The exact numeric thresholds around this —
what counts as "exact" versus "near," how much extra slack a mixed-codec comparison
gets — are plain constants in `config.py` with no comment explaining how the specific
values were chosen; treat them as tuned defaults you can adjust, not as figures with
a documented derivation.)

## The workspaces: the modern app's rooms

The modern app is organized as a **sidebar of workspaces** — each workspace is a
self-contained "room" dedicated to one job. You move between them from the sidebar,
and the currently selected room fills the main area. Behind a row of friendly names
sits a consistent pattern: every workspace is built on a shared template
(`gui/workspaces/base.py`) that gives it the same skeleton (a place for content, a
way to report progress to the log drawer at the bottom, and a status indicator at the
top).

The current rooms are, in plain terms (sidebar label first, since a few of the
group/room names differ from what you might expect from the feature name):

| Sidebar label | What you do there |
|---|---|
| **Indexer** | Scan and reorganize your library into clean Artist/Album/Track folders. |
| **Library Sync** | Merge an incoming folder into your existing library. |
| **Duplicates** | Find and clear out duplicate copies of songs. |
| **Similarity** | Compare two specific tracks and see *why* the app thinks they do or don't match — useful for understanding the duplicate detector. |
| **Tag Fixer** | Look up and correct the artist/album/title/genre information on your tracks. |
| **Genre Normalizer** | Batch-update genre labels (see the honest caveat in **features/tag_fixer.md** — this label means two different things depending on which app you're in). |
| **Playlists** | Build playlists by folder, tempo, energy, or automatic DJ flow. |
| **Clustered** | Group tracks by sound (K-Means / HDBSCAN) via a step-by-step wizard, and turn the groups into playlists. |
| **Music Graph** | Open a browser-based 3D visualization of your sonic clusters. |
| **Player** | A full-featured, built-in audio player — library table, queue, recently-played, shuffle/repeat, a persistent now-playing bar at the bottom of the whole app window. |
| **Compression** | Mirror your library into a space-saving Opus copy (FLAC → Opus transcoding), estimating savings before you run it. |
| **Utilities** | A tile grid of exports, diagnostics, and cleanup tools. |
| **Help** | Documentation, shortcuts, and project links. |

A persistent **Now Playing bar** sits between the workspace area and the log drawer,
shared across the whole app (not confined to the Player workspace) once a track is
loaded — it stays visible no matter which workspace you're in, with its own
transport controls that mirror the Player workspace's.

Two things worth knowing if you go looking for them: an **older "Clustered"
workspace** (`gui/workspaces/clustered.py`) still exists in the codebase but is
**dead code** — nothing imports it any more, and `main_window.py` wires the sidebar's
"Clustered" entry to the newer `clustered_enhanced.py` instead. And the four
interactive-graph widgets built for an in-app 2D cluster map
(`InteractiveScatterPlot`, `Interactive3DScatterPlot`, `ClusterLegendWidget`,
`TrackDetailsPanel`) are **fully built but never wired into any workspace** — the
"Music Graph" room you actually reach from the sidebar only launches a
browser-based 3D view, not an in-app one. See **ROADMAP.md** for what this means for
the feature's roadmap; see **features/playlists_and_clustering.md** for what
currently, actually works.

For the full screen-by-screen breakdown of every control in every workspace and
dialog — including which controls are fully wired versus decorative — see
**gui_inventory.md**.

## How the app wakes up

When you launch the modern app, it does not slam you straight into a busy work
screen. It opens with a calm **landing moment**: the window fades in, an animated
mosaic of album-art tiles builds itself from your library while you confirm which
library you want to work on (or pick a new one), and then the landing gracefully
cross-fades into the main window. A brief pause between the window's fade-in and the
tiles beginning to animate is the intentional "logo moment" — a comment in the
landing code notes this pause deliberately *replaces* an earlier, separate splash
screen that used to run before it; that splash-screen code (`gui/widgets/splash.py`)
still exists in the repository but is no longer invoked anywhere, having been
superseded by the mosaic landing itself.

There is a deliberate choice hiding in this choreography. The genuinely expensive
work — scanning and fingerprinting your library — is **postponed** until the opening
animations have finished, so that the introduction stays smooth instead of stuttering
behind a frozen progress bar. The app waits for whichever comes first: 30 seconds
after the landing first appears, or 3 seconds after you press the landing's call to
action ("Initialize" if you have a saved library, "Choose Library Folder" if you
don't). Clicking directly on one of the mosaic's album-art tiles skips straight past
that and opens the app already playing that album, in the Player workspace. The point
of all of it is that first impressions feel effortless, and the heavy lifting begins
only once you're settled in.

## A note on online help: what's real and what's a placeholder

Several features can consult **online music databases** to look up information about
your tracks. It is worth being honest about which of those connections actually work
today:

- **Fully working:** AcoustID (fingerprint-based identification), MusicBrainz (the
  open music encyclopedia), and Last.fm (genre and tag data). These three power the
  Tag Fixer and the Genres workspace — although, notably, AcoustID is the *only* one
  of the three that the Tag Fixer's automatic scan actually calls today; see
  **features/tag_fixer.md** for the detail on why MusicBrainz and Last.fm's real
  integration paths aren't both live at once in that particular workflow.
- **Listed but not yet built:** Spotify and Gracenote appear in the settings as
  options, but their connections are empty placeholders — selecting them currently
  returns nothing. The Settings screen already marks this honestly: both appear
  visibly in the service dropdown but are disabled with a tooltip explaining they
  aren't implemented yet, rather than being silently hidden. `config.py` names this
  set explicitly as `UNAVAILABLE_SERVICES`, with the comment "The Settings UI shows
  these as disabled so users aren't misled."

This distinction is carried through into **ROADMAP.md**, which tracks what is planned
but not finished.

## What the LLM-related references in older docs actually mean

`CLAUDE.md` and the root `README.md` both mention a `bindings/` folder, a
`third_party/` folder, and a `plugins/assistant_plugin.py` file, describing an
optional local-LLM assistant feature. Worth stating plainly for anyone hunting for
this feature: **none of the three exist**, and a full search of the git history
confirms they never have. There is no dormant or experimental LLM integration
sitting in the codebase — only these dangling references in the docs (now corrected)
and a stray mention of `bindings/build/` in `.gitignore`. If you want to build this
feature, you are starting from nothing, not resuming unfinished work.

---

*Next: dive into any feature in the **features/** folder, or see **ROADMAP.md** for
what's still ahead.*
