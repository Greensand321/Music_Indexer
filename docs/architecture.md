# AlphaDEX — Architecture

*How the pieces fit together behind the scenes. This document is conceptual: it
explains the shape of the system and the reasoning behind it, with a handful of
plain-text diagrams for the parts that are easier to see than to describe. If you
have read **overview.md**, you already know the vocabulary used here.*

*Last verified against the code: 2026-11-07. Page names and navigation updated for
the adopted page structure on 2026-11-08.*

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

**This rule is currently held everywhere**, but it has been broken before and the
two repairs are worth knowing as precedent. `controllers/library_index_controller.py`
used to import `tkinter.messagebox` and pop its own "here's where I saved it"
dialog; it now simply returns the output path and lets the calling UI report it.
`plugins/acoustid_plugin.py` used to carry a Tkinter configuration `Frame`
alongside its lookup services, which made the whole plugin module unimportable on
any machine without Tkinter — that widget now lives in `metadata_config_frame.py`
at the repository root, next to `library_sync_review.py`, the project's other
legacy Tkinter panel. Both fixes followed the same shape: **the backend computes
and returns; the UI layer decides how to show it.** Reach for that shape rather
than adding a dialog call to a backend module.

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

A concrete illustration from the Player page, preserved because it states the
rule precisely: album-art loading runs on a background `_ArtLoader` thread that
"emits a `QImage` (NOT a `QPixmap`) so the caller can convert to `QPixmap` on the
main/GUI thread. Creating `QPixmap` off the GUI thread is undefined behaviour in Qt
and silently produces null or broken pixmaps on Windows." (`gui/workspaces/player.py`)
— exactly the kind of platform-specific trap this discipline exists to avoid.

### The job registry: starting, cancelling and outliving a worker

Background workers in the modern app are `QThread`s, and Qt is unforgiving about
them: if the last reference to a thread disappears while it is still running, Qt
aborts the whole process ("QThread: Destroyed while thread is still running") —
possibly halfway through moving files. That used to happen on almost every page,
because workers announced "I'm finished" from *inside* their own run, and the page
dropped its reference before the thread had actually stopped.

The fix is one process-wide **JobRegistry** (`gui/workers.py`). Think of it as a
cloakroom: every worker is checked in when it starts and only handed back once the
thread has *really* finished (Qt's own C++ `finished()` signal, which fires after the
run returns and which no Python subclass can shadow). After that, a page can forget
its worker at any moment without risk. The rules every page follows:

- **Start every thread through the registry**: `WorkspaceBase._start_worker(worker,
  "Name")`, or `gui.workers.start_worker(...)` outside a workspace. Never call
  `QThread.start()` directly (a test ratchet counts these). The registry also detaches
  a worker from any widget parent, because a widget would destroy its child thread
  mid-run.
- **`ask_before_quit`** marks whether a job may be writing files. Read-only work —
  scans, previews, lookups, album-art and tag loads, statistics, estimates — passes
  `False` and is cancelled silently when you quit. Anything that writes keeps the
  default (`True`), so quitting mid-write asks first.
- **`LIBRARY_WRITES`** is a mutual-exclusion group for jobs that move, rename, delete
  or rewrite files inside the library: Indexer execute, Library Sync execute,
  Duplicates execute, File Cleanup and its undo, Tag Fixer apply, both genre applies,
  and playlist writes. A page starts such a job with `_start_worker(...,
  library_write=True)`, and *before* showing its confirmation it calls
  `_library_writes_blocked("What I'm about to do")`. That check looks at the group and
  at the shared tag-write guard (`controllers/tag_write_guard.py`), and if anything is
  already writing it tells you which job and refuses — so you are never asked to
  confirm something that can't run. Membership ends when the thread finishes, so
  there is no manual "release" to forget on an error path. (Pair Review's quarantine
  checks the same group.)
- **Cancellation is cooperative.** `cancel_thread()` calls the worker's own `cancel()`
  (most set a `threading.Event` the backend polls) and `requestInterruption()`. Results
  that arrive from a worker the page has since abandoned — after a library change, say
  — are recognised (by sender or generation number) and ignored.

**Closing the window** (`AlphaDEXWindow.closeEvent`) follows a fixed protocol:

1. No jobs running: each panel's `shutdown()` hook runs (stop timers, release the
   playback backend), and the window closes.
2. Jobs running: if any of them has `ask_before_quit`, a "Background tasks running"
   question lists them; No keeps the window open. Otherwise (or on Yes) the window
   enters a *stopping* state — its UI is disabled so nothing new can start — cancels
   every job, and retries the close every 100 ms, cancelling again each time (which
   also catches a job chained from another).
3. A job that ignores cancellation (a hung network drive, a long native clustering
   stage) must not trap you: after `CLOSE_PATIENCE_S` (20 s) you are asked once whether
   to quit anyway. Yes exits immediately via `os._exit`, because normal interpreter
   shutdown would destroy the still-running threads and abort.

`main()` in `alpha_dex_gui.py` also cancels and waits for any stragglers (such as the
landing's art scanner) after the event loop ends.

**Garbage collection happens on the GUI thread only** (`gui/gc_guard.py`). Python's
cycle collector normally runs in whichever thread happens to allocate the object that
crosses its threshold — often a worker. A collection frees every unreachable cycle,
including windows, dialogs and widgets that live on the GUI thread, so their C++
destructors ran in the worker. For a running animation that is a crash: it unregisters
from the *worker's* animation timer (Qt keeps one per thread) and the GUI thread's timer
segfaults on its next frame. The guard switches automatic collection off at start-up
and collects from a GUI-thread timer instead, generation by generation with Python's
own thresholds; the GUI test harness installs it for the whole session too. Objects
freed by reference counting are unaffected.

Two smaller hooks complete the lifecycle. **`on_settings_saved()`** is called on every
panel after the Settings page saves; panels that cache settings (the duplicate
thresholds used by Duplicates → Find duplicates and Duplicates → Compare two files)
re-read them there, rather than reading the config file every time they are shown —
the design system forbids config I/O on navigation or paint. And the registry's
`jobs_changed(int)` signal drives the **Activity** page's list of running jobs (each
with a Cancel button) and the "N running" badge on the sidebar's Activity entry.
Housekeeping jobs, such as building the album-art backdrop, are marked *quiet*: they
are kept alive and cancelled on quit like any other job, but not listed or counted.

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
     duplicate_consolidation       you, in a browser    duplicate_consolidation_executor
       .build_consolidation_plan   (Docs/*.html) or       .execute_consolidation_plan()
     library_sync                  a table in the app   library_sync.execute_plan()
       .compute_library_sync_plan
     music_indexer_api                                  music_indexer_api.apply_indexer_plan()
       .build_indexer_plan
```

Every major workflow is physically split into a **dry run** and an **execution**,
and these are separate operations, usually living in separate modules or files, that
produce different things:

- The **dry run** computes the entire plan — every move, every rename, every
  replacement, every deletion — and pours the result into a human-readable report
  (typically an HTML page). It is incapable of changing a single file, because the
  code that would touch the disk simply isn't run in this mode.
- The **execution** takes that approved plan and carries it out, logging every action
  as it goes. It runs **the plan object that was previewed**, not a recomputation:
  the Indexer's `IndexerPlan`, Library Sync's plan (built from the matches and flags you
  reviewed) and the Duplicate Finder's plan are each executed verbatim, and each
  refuses a plan whose library changed since the preview, or skips the items whose
  files changed. The same shape now holds for the smaller writers: the Tag Fixer and
  both genre tools write exactly the previewed rows, File Cleanup and its undo rename
  exactly the rows of their preview dialog, and the Playlist Generator writes exactly
  the files its confirmation listed. In the GUI, an Execute or Apply button is only
  enabled while a preview of the current inputs exists.

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

### What each run leaves behind in `Docs/`

"Log every action" is concrete. These are the records the workflows write inside the
library's `Docs/` folder (timestamps make each run's files unique):

| Workflow | Preview | Record of what actually happened |
|---|---|---|
| Indexer | `MusicIndex.html`, `indexer_log.txt` | `indexer_executed_<date>_<time>.json` and `.txt` — every move, sweep and skip, also when cancelled |
| Library Sync | `LibrarySyncPreview.html` | `LibrarySyncAudit_<run>.json` and `LibrarySyncExecuted_<run>.html` per run (`LibrarySyncAudit.json` / `LibrarySyncExecuted.html` are copies of the latest); replaced files are backed up under `Backups/LibrarySync/<run>/` |
| Duplicate Finder | `duplicate_preview.html` and `.json` | `duplicate_execution_reports/` (playlist backups included) |
| Pair review (Duplicates) | — | `duplicate_pair_review_log.txt` (quarantines), `duplicate_pair_tags.txt` |
| Tag Fixer, Genre fill, Genre normalize | the on-screen table | `tag_changes_<tool>_<timestamp>.jsonl` — the before and after values of every write, failure and skip (`utils/tag_change_log.py`). Metadata → Undo changes previews and writes a revert from one of these logs, logged as `tag_changes_revert_<timestamp>.jsonl` (`controllers/tag_revert.py`) |
| File Cleanup | the preview dialog | `file_cleanup_<timestamp>.jsonl`; "Undo Last Cleanup" reads it back and appends each restore to it |
| Playlist writes | the confirmation list | `playlist_writes.log`; the generator leaves a timestamped `.bak` beside a playlist it replaces, and `update_playlists` (run after renames and moves) backs each playlist up to `playlist_backups/` before editing only the changed entries |
| Clustering (Sound Map → Analyse) | — (an analysis changes no library file) | `cluster_members.json` (the run's full membership, never downsampled), `cluster_info.json` (the map, downsampled above 5,000 tracks), `cluster_graph.html`; both JSON files carry the same `run` block (`run_id`, method, parameters, features, library root) |
| Save cluster playlists (Sound Map) | the Save cluster playlists dialog | one `.m3u` per cluster in a per-run folder, `Playlists/Clusters/<date time> <method>/` (earlier runs' folders are never touched); `playlist_writes.log` and a `.bak` beside each replaced file, as for every playlist write |
| Opus mirror | the estimate; for orphan removal, the list of orphaned files | `<destination>/Docs/opus_library_mirror_reports/` — the run report (stale files, orphaned files with no source any more, leftover temporary files removed) and, when orphans are removed after review, a log of every deletion, refusal and failure (`remove_mirror_orphans`) |

Execution never overwrites an existing file: a destination that has appeared since
the preview is skipped and reported, and plans snapshot what they will touch so that
a file changed since the preview is skipped rather than acted on blind.

**Cluster arrays are index-aligned.** In both `cluster_info.json` and
`cluster_members.json`, element *i* of every per-track array (`X_2d`, `X_3d`,
`labels`, `tracks` in the map; `tracks`, `labels`, `centroid_distance`,
`tempo_bpm`, `energy` in the members file) describes the same track, and they are
consumed positionally. Never subset one array without the others. The members file
is never downsampled; the map records which members it kept (`X_indices`), and the
cluster playlist export refuses a map and members file that are not the same run
(`cluster_graph_data.check_members_match_payload`).

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
connection so a slow write never blocks a fast read. Because a queued write may not
have reached disk yet, the cache also keeps a small in-process memo of fingerprints
it has just computed, stored or read (checked against the same size/mtime), so
asking again a moment later doesn't recompute; anything that needs the rows on disk
(the background cache pre-warmer does this before it finishes) calls
`flush_fingerprint_writes()`.

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
source but have since **drifted apart**: the vendored `fingerprint_cache.py` is about 360
lines with no background writer thread, no WAL mode, and none of the extended
metadata columns (bitrate, codec, embedded tags, artwork) that the root
`fingerprint_cache.py` has grown to about 1,000 lines to support; it does keep a
`decoded` column so Library Sync decodes each fingerprint once, not on every run. The
decoded-Chromaprint distance was **ported** to the vendored `near_duplicate_detector.py`,
so both engines now score matches on the same scale. So were the duplicate-finder
audit's cache fixes, adapted to the simpler cache: the `fp_scheme` column and
`has_silence_padding` check (old rows holding a silence-padded short-track print are a
cache miss — and this is the cache that really held them, because only Library Sync's
generator fingerprinted through `audio_norm.normalize_for_fp` with a window), and the
corrupt-file handling (renamed to `.corrupt-<time>`, rebuilt). Library Sync reads and
writes that cache in three places — `library_sync._scan_folder`, the vendored
generator's own reuse check, and `fingerprint_cache.get_fingerprint` — and all three
use the same helpers (`row_is_current`, `fingerprint_trusted`, `store_fingerprint_row`,
which also clears the row's `decoded` blob). There is no writer thread to kill, so the
root's "one unbindable row stops every later write" bug never applied; a row that
cannot be stored is now skipped instead of failing the scan. The length-aware
distance (`duplicate_fingerprint_distance`: the part of the longer print with no
counterpart counts as unrelated audio, so a 30 s clip is not its full song) was ported
too, and it is what Library Sync's matcher scores pairs with (`library_sync.match_distance`);
the vendored `fingerprint_distance` itself still compares the overlap only, exactly
like the root one. The
vendored `music_indexer_api.py` is missing a few refinements (like richer
collision-decision logging) that the root copy has picked up since.

**Why this exists:** the split is a leftover from the project's two-generation
history, not a design decision made on purpose. AlphaDEX was effectively built
twice — an earlier program (built around the now-legacy Tkinter interface, among
other dated choices) and the current iteration that grew into the modern Qt app.
The vendored `library_sync_indexer_engine/` copy is a snapshot carried over from
that first generation; the root-level modules are the ones that kept evolving.
Nothing in the code says this, which is why it reads as mysterious when you first
trip over it — hence this note.

**The working rule that follows from that history:**

- **The root-level engine is the canonical one.** New work, fixes, and features
  belong in `music_indexer_api.py`, `fingerprint_cache.py`, and
  `near_duplicate_detector.py`.
- **Leave the vendored copy alone** unless you are deliberately making a change to
  it. It is not a target for opportunistic refactoring, and there is no active plan
  to unify the two — treating it as "debt to pay down" would be churn for its own
  sake.
- **Know the one real consequence:** a fix to a root-level module does **not**
  reach Library Sync automatically. If you fix something in the root engine and you
  specifically need Library Sync to get that fix too, you have to port it across
  deliberately. That's the whole gotcha; it doesn't require any larger cleanup.

One small piece of evidence that someone previously noticed the drift: a single
test (`tests/test_sanitize.py::test_engine_sanitize_matches_api_sanitize`) loads
both copies of `sanitize()` and asserts they still agree — a guard on exactly one
function, while the modules around it have drifted freely. Keep that test passing
if you touch `sanitize()`; don't read it as an aspiration to keep everything else
in lockstep.

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

Two safety properties protect that file. **Saves are atomic**: `save_config()` writes
a temporary file and swaps it into place with `os.replace`, so a crash or a full disk
can never leave a half-written settings file. And **an unreadable file is never
silently lost**: `load_config()` falls back to the defaults (one set, applied the same
way whether the file is missing or corrupt) and remembers the problem
(`last_load_error()`); before anything overwrites it, `save_config()` copies the
unreadable file to `<name>.corrupt-<timestamp>` (`last_backup_path()`), and the main
window says so in the log. Before this, a truncated file was replaced by defaults on
the next routine save, wiping the AcoustID key, theme and MusicBrainz user agent.

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

## The pages: how the modern app is laid out

The modern app is organized as a **sidebar of pages**, grouped by the job you came to
do. Choosing a page fills the main area with it. Some pages hold more than one
**section**, shown as tabs across the top of the page — Duplicates, for example, has
"Find duplicates" and "Compare two files" — and a page remembers the section you
looked at last. Behind each section sits a **panel**, and every panel is built on a
shared template (`gui/workspaces/base.py`) that gives it the same skeleton (a place
for content, a way to report progress to the log drawer at the bottom, a status
indicator, and the background-job helpers described above). One panel can serve
several sections of a page: the genre panel serves both "Fill genres" and "Normalize
genres", and the page's tabs tell it which to show.

The whole structure — the groups, the pages in order, their sections, and which panel
provides each section — is written down as plain data in one file,
`gui/navigation.py`. The sidebar, the Ctrl+1…9 shortcuts (the first nine pages, in
sidebar order) and the "Page · Section" labels all read from it, so they cannot drift
apart. These documents write a place in the app the same way, as **Page → Section**:
for example **Duplicates → Compare two files**, or **Sound Map → Map**.

| Group | Page | Sections | What you do there |
|---|---|---|---|
| Library | **Home** | — | See the library at a glance: its size, its health (whether it has the Indexer's layout, and how many tracks wait in `Manual Review/`, `Quarantine/` and `Not Sorted/`), the latest run of each workflow, and shortcuts to the common jobs. Read-only. |
| Organize | **Indexer** | — | Scan and reorganize your library into clean Artist/Album/Track folders. |
| | **Library Sync** | — | Merge an incoming folder into your existing library. |
| | **File Cleanup** | — | Remove copy suffixes such as " (1)" and " - Copy" from file names, with a preview first and an undo afterwards. |
| Clean up | **Duplicates** | Find duplicates · Compare two files · Pair review · Fuzzy finder, Scan engine and Bucketing (each marked *Experimental*) | Find and clear out duplicate copies of songs; compare two specific files and see *why* the app thinks they do or don't match; review duplicate pairs side by side. The three Experimental sections are prototypes of other ways to find duplicates — they are not part of the real workflow. |
| | **Metadata** | Tag Fixer · Fill genres · Normalize genres · Undo changes | Look up and correct the artist/title/album/genre information on your tracks; fill missing genres from MusicBrainz; collapse messy genre spellings with a mapping you build; roll back an earlier tag change. Every write is previewed first (see **features/tag_fixer.md**). |
| Listen | **Player** | — | A built-in audio player — library table, queue, recently played, shuffle/repeat. |
| | **Sound Map** | Analyse · Map | Group tracks by how they sound (K-Means / HDBSCAN), with a Quick Start or a step-by-step wizard; then explore the groups on an in-app 2-D map (select, inspect, play, export a playlist) or open the browser-based 3-D view. The analysis writes no playlists itself; **Save cluster playlists…**, at the top of the page in both sections, previews and saves one per cluster. |
| | **Playlists** | — | Build playlists by folder, tempo, energy, or automatic DJ flow, and repair playlists whose entries no longer point at a file. |
| Take away | **Mirror** | Build the mirror · Verify the mirror | Make a space-saving Opus copy of your library (FLAC → Opus), estimating the savings before you run it; then open a converted file and check that its tags and cover art came through. The originals are never touched. |
| | **Exports** | — | Write an artist/title list or a per-codec file list of the library into `Docs/`. |
| *(foot of the sidebar)* | **Activity** | — | The jobs running right now (each with Cancel), every log line of the session, and the latest problems, including the crash log. |
| | **Settings** | Metadata services · General · Appearance | Online-service keys, the default library and the duplicate thresholds, and the theme, backdrop and motion. |
| | **Help** | — | Documentation, shortcuts, and project links. |

**If you knew the older layout.** Until 2026-11-07 the sidebar was a flat list of 13
entries. Similarity is now **Duplicates → Compare two files**; Tag Fixer and the
Genre Normalizer became **Metadata**'s sections; Clustered and Music Graph became
**Sound Map → Analyse** and **Sound Map → Map**; Compression is **Mirror → Build the
mirror**. **Utilities is gone**: File Cleanup became its own page under Organize, the
exports went to **Exports**, the library Validator became Home's "Library health"
card, the crash-log viewer became part of **Activity**, the Opus and M4A testers sit
in **Mirror → Verify the mirror**, and Pair Review and the three duplicate prototypes
became Duplicates sections. Settings is a page rather than a dialog, and the theme
picker is its Appearance section. The old navigation keys still route — asking for
`"graph"` lands on Sound Map → Map, and `"tools"` on Exports — so nothing that
navigates by an old name breaks. `docs/CONTRADICTING_UPDATES.md` §B records every
part of this change.

Two leftovers of the old names are worth knowing about. Pair review and the three
Experimental sections still open their tool in a separate window (an "Open …"
button in the section). And a few panels still carry their old heading inside the
new page: Duplicates → Compare two files is headed "Similarity Inspector", the two
genre sections "Genre Normalizer", and Mirror → Build the mirror "Library
Compression".

A persistent **Now Playing capsule** floats at the bottom centre of the page area,
shared across the whole app (not confined to the Player page) once a track is
loaded — it stays visible no matter which page you're on, with its own transport
controls that mirror the Player's. Pages keep room under their content so it never
covers anything.

The interactive cluster map is made of three widgets (`InteractiveScatterPlot`,
`ClusterLegendWidget`, `TrackDetailsPanel`), and all three are live in Sound Map →
Map. An in-app 3-D plot widget was built once but never used, and has been deleted;
the 3-D view you reach — the Map's **Open 3D view** button — is a standalone browser
page that `cluster_graph_3d.py` generates. See **features/playlists_and_clustering.md**
for what currently works.

For the full screen-by-screen breakdown of every control on every page and in every
dialog — including which controls are fully wired and which are switched off — see
**gui_inventory.md**.

## How the app wakes up

When you launch the modern app, it does not slam you straight into a busy work
screen. It opens with a calm **landing moment**: the window fades in, an animated
mosaic of album-art tiles builds itself from your library while you confirm which
library you want to work on (or pick a new one), and then the landing cross-fades
into the main window. The brief pause between the fade-in and the tiles beginning to
animate is the "logo moment"; it replaced an earlier splash screen, whose code has
since been deleted.

The choreography is arranged so that nothing heavy happens while you are watching it:

1. The landing appears first. The main window — a big build of every panel — is
   only constructed once the landing has **fully faded in**, during the logo pause, so
   the first frames are never blocked (it is built on demand if you click very fast).
2. You confirm the saved library ("Initialize"), choose another (the "Choose another
   folder" link, or the "Choose Library Folder" button when no library is saved; a saved
   library that no longer exists is shown as "not found"), or click an album-art tile,
   which also opens the Player already playing that album.
3. The landing fades out while the main window fades in over the same duration.
4. Only **after the cross-fade** is the chosen library applied — exactly once. That is
   when every page learns the library, the Player scans it and the library statistics
   in the page header start, so that work never stutters the transition. The main
   window opens on **Home**.

The sequence is factored into `Startup` / `build_startup()` in `alpha_dex_gui.py`, so
the test harness can drive it without running the event loop. If the main window
cannot be built at all, the app shows an error, writes the traceback to the crash log,
and exits with status 1 instead of hanging. `multiprocessing.freeze_support()` runs
first thing, so a frozen (PyInstaller) build's clustering pool doesn't spawn extra
copies of the app.

## A note on online help: what's real and what's a placeholder

Several features can consult **online music databases** to look up information about
your tracks. It is worth being honest about which of those connections actually work
today:

- **Fully working:** AcoustID (fingerprint-based identification), MusicBrainz (the
  open music encyclopedia), and Last.fm (genre and tag data). In the Tag Fixer's scan,
  AcoustID runs whenever fpcalc and the network allow; a MusicBrainz text search joins
  when MusicBrainz is the service chosen in Settings; and Last.fm joins only when a
  Last.fm API key is set — the one saved in **Settings → Metadata services**, or the
  `LASTFM_API_KEY` environment variable, which wins when both are set. Each scan
  begins by checking every source and says which ones ran and why the others didn't.
  See **features/tag_fixer.md**.
- **Listed but not yet built:** Spotify and Gracenote appear in the settings as
  options, but their connections are empty placeholders — selecting them currently
  returns nothing. **Settings → Metadata services** already marks this honestly: both
  appear visibly in the service dropdown but are disabled with a tooltip explaining they
  aren't implemented yet, rather than being silently hidden. `config.py` names this
  set explicitly as `UNAVAILABLE_SERVICES`, with the comment "The Settings UI shows
  these as disabled so users aren't misled."

This distinction is carried through into **ROADMAP.md**, which tracks what is planned
but not finished.

## The reserved folders, in code

**overview.md** lists the six reserved top-level folders (`Not Sorted/`, `Playlists/`,
`Manual Review/`, `Docs/`, `Trash/`, `Quarantine/`). Until recently each workflow kept
its own partial list of them — or none — so duplicate losers sitting in `Quarantine/`
came back as library tracks, statistics counted `Docs/` backups, and so on. There is
now **one definition**, `utils/reserved_folders.py`, and every scan that treats files
as "the library" goes through it.

- `RESERVED_FOLDERS` is all six. `NON_LIBRARY_FOLDERS` is the four whose contents are
  not your music at all (`Playlists`, `Docs`, `Trash`, `Quarantine`); views that let
  you browse or play everything — the Player, the Exports page — skip only
  these, so tracks waiting in `Manual Review/` or `Not Sorted/` still show up there.
- `walk_library()` and `iter_library_files()` are `os.walk` wrappers that never descend
  into a reserved folder; `prune_reserved_dirs()` does the same inside a hand-written
  walk; `is_reserved_path()` / `reserved_component()` answer "is this file inside one?".
- Only **top-level** folders are reserved (`Artist/Album/Docs/` is ordinary content),
  and matching ignores letter case, as Windows and macOS file systems do.

Who uses which set: the Indexer, Duplicate Finder, Library Sync (both sides), File
Cleanup, Pair review, clustering, playlist generation and the library statistics skip
all six. The Tag Fixer skips all of them **except `Manual Review/`**, because those are
exactly the tracks missing metadata. The Opus mirror leaves out `Quarantine/`,
`Trash/` and `Docs/`.

## Themes and motion

The modern app has 15 themes — **Liquid Glass** (the default, in four accents) and
14 classic ones — plus an Auto mode, and it animates a fair amount (nav hover, the
sidebar's sliding pill, the section tabs' thumb, page cross-fades, the log drawer,
status chips, the landing). A few pieces keep this fast and well-behaved.

**Glass is just tokens** (`gui/themes/tokens.py`, `gui/themes/glass.py`). Every theme
carries a *material*: per-elevation surfaces (sidebar, card, raised, capsule, well) with
a fill, border, top highlight and shadow, plus the backdrop. Liquid Glass's fills are
translucent; the classic themes get an opaque material derived from their colours, so
both render through the same components (`paint_surface`, `GlassCard`, the style).
On the web, glass blurs what is behind it at paint time; Qt has no cheap live blur, so
the **backdrop** (`gui/widgets/backdrop.py`: a wall of the library's covers) is composed
and blurred *once*, off the GUI thread with `QImage` and numpy, cached per library and
size bucket, and simply blitted on paint. A translucent fill over an already-blurred,
static backdrop reads as glass. Reduce transparency flattens every surface to opaque.

**A theme switch is a palette swap, not a restyle** (`gui/themes/manager.py`). Any
change to an application stylesheet makes Qt re-polish every widget in the window —
about 0.7 s for the main window no matter what the rules say — which is why switching
theme used to freeze the app for over a second. Now the stylesheet is **static**: it is
installed once, and its colours are `palette(...)` references. A switch takes the fast
path: the one `AlphaDEXStyle` instance swaps its tokens in place, the app palette is
replaced, and a single top-down pass re-polishes only the widgets the stylesheet
colours (plus any widget with its own inline stylesheet, and its descendants, whose
palette Qt froze at polish time); every other widget is told to inherit the new
palette. Then `theme_changed` is emitted so widgets that colour themselves refresh.
A switch now costs roughly 60–80 ms. `tests/ui/test_ui_theme_switch_equivalence.py`
builds a window, switches it live, builds a second window directly in the target
theme, and checks that every widget's resolved palette matches.

**Shadows are painted, not attached** (`gui/themes/glass.py`, `gui/widgets/gradient_bg.py`).
Cards get their soft shadow painted by the transparent `GradientWidget` surface beneath
them (the sidebar's by the backdrop, the Now Playing capsule's by a mouse-transparent
`ShadowLayer` above the page), from a cached 9-slice pixmap per colour, blur and radius,
clipped so it never paints under the surface itself. A `QGraphicsDropShadowEffect` would re-render and
blur the whole card every time anything inside it repainted (a hover frame, a progress
tick, a blinking caret), and would occupy the card's only graphics-effect slot, which
fades need. (The old effect also never showed: it was passed as a temporary and
dropped.) The backdrop setting is read once and cached, so no paint event reads the
config file.

**Every animation goes through the motion module** (`gui/themes/motion.py`). It defines
the duration tokens (`micro` 120 ms, `fast` 150, `base` 220, `slow` 320, `page` 420,
`ambient` 3500, `flash` 1600, plus a per-item `stagger`) and four easings (`standard`,
`exit`, `emphasis`, `ambient`), and helpers that avoid the four bugs hand-rolled
animations kept producing: `animate_value` / `animate_property` build animations
**parented** to their owner (so they aren't garbage-collected mid-flight), keep **one
animation per owner and key** (a new request replaces the old one instead of two
fighting over a property), and refuse to animate geometry a layout owns (which
re-laid out the whole window every frame); `fade()` always ends at full opacity or
hidden, never half-transparent. Page and section switches, and results that appear,
use `reveal()`, a *snapshot* fade: the page is rendered once and that picture fades in
over the backdrop before the live page takes over. (An opacity effect alone re-renders
the whole page on every frame, since Qt caches effect sources only for graphics items.)
**Reduced motion** (**Settings → Appearance → Reduce
motion**, saved as the config key `reduced_motion`, or the `ALPHADEX_REDUCED_MOTION=1`
environment variable) makes every token 0 ms and jumps straight to the end state,
running the same callbacks, so the code paths are identical either way.

## Playback: the backend seam

The Player never talks to VLC directly. It talks to a `PlaybackBackend`
(`gui/playback_backend.py`), a small, Qt-free interface — load, play, pause, stop,
state, length, position, seek, volume, release — whose query methods never raise.
`create_backend()` returns a `VlcBackend` when python-vlc *and* the libVLC library can
be loaded, and otherwise a `NullBackend` that carries a human-readable reason.

This seam exists because of a crash: python-vlc is in `requirements.txt`, so on a
machine without the libVLC library `import vlc` raises `NotImplementedError` or
`OSError` rather than `ImportError`, and the Player — and with it the whole main
window — failed to build. Now the Player shows "Playback unavailable" with the reason,
disables its transport controls, and never shows a fake "playing" state. Tests pass a
fake backend to `PlayerWorkspace(backend=...)` or patch `create_backend`.

## When something crashes: the crash log

`main()` installs `crash_logger.install_for_qt()` before anything else. From then on,
uncaught Python exceptions — on the GUI thread, inside Qt slots (PySide routes those
through `sys.excepthook`) and on worker threads (`threading.excepthook`) — are appended
to **`~/.alphadex/crash.log`** (rotating, about 1 MB), and native crashes (segfaults,
aborts) are written by `faulthandler` to **`~/.alphadex/faults.log`**. The hooks chain
to the previous ones, so tracebacks still reach the console. The **Activity** page's
Problems card shows the tail of exactly those two files, next to the session's own
warnings and errors. (The old crash-log viewer, which used to sit under Utilities,
looked for a file the Qt app never wrote, so it always said "No crash log found.")

## Testing the GUI: the harness

Besides the backend suite, the repository has a **headless GUI harness** under
`tests/ui/` (pytest-qt, `QT_QPA_PLATFORM=offscreen`). Every page can be built and
driven against a real synthetic library of tagged MP3 / FLAC / M4A / OGG / Opus files
(`tests/library_factory.py`), and the failure modes that made the app feel unstable
are test failures: an unplanned modal dialog fails instead of hanging, a test that
leaves a `QThread` running fails, cross-thread and painting warnings fail, exceptions
inside slots fail, and every test gets an isolated home folder so it can never touch
your real `~/.soundvault_config.json`. Static "ratchets" count hex colours, inline
stylesheets, font-size literals, direct `QThread.start()` calls and unparented
animations, and only let those counts go down. `scripts/gui_snapshots.py` renders every
page in every theme as a contact sheet.

The full guide — fixtures, markers, and the rules for writing workers the harness can
live with — is **[tests/ui/README.md](../tests/ui/README.md)**.

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
