# AlphaDEX — Architecture

*How the pieces fit together behind the scenes. This document is conceptual: it
explains the shape of the system and the reasoning behind it, without code. If you
have read **overview.md**, you already know the vocabulary used here.*

---

## The big picture: three layers

It helps to picture AlphaDEX as a building with three floors. Work flows down from
the top floor and results travel back up.

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

## The preview-first contract, structurally

**overview.md** introduced "look before you leap" as a promise to the user. Here is
how that promise is actually *built into the structure* rather than just being good
intentions.

Every major workflow is physically split into a **dry run** and an **execution**,
and these are separate operations that produce different things:

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

A companion rule travels with this one: **no silent data loss.** Anything that moves
or deletes a file writes a record of what it did into the `Docs` folder, and when a
decision is uncertain the app leans toward *quarantine* (set the file aside) rather
than *delete* (destroy it).

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
cache from becoming a bottleneck itself, writes are batched and queued in the
background rather than performed one painstaking entry at a time.

## One place for settings

Every preference in AlphaDEX — your library location, how strict the duplicate
matching should be, your online-service keys — is read from and written to **one
central file** (the `.soundvault_config.json` file introduced in the overview). No
part of the app reads or writes that file on its own; everything goes through a single
"load the settings" / "save the settings" pathway.

The benefit is consistency. Because there is exactly one door in and one door out,
different parts of the app can never drift into disagreeing about what the current
settings are. If the duplicate-finder and the library-sync feature both need to know
your matching sensitivity, they are guaranteed to be reading the same value from the
same place. The file is deliberately a plain, human-readable text file, which makes it
easy to inspect or hand-edit when something looks wrong, and avoids the fragility of a
heavier database for what is really just a small bag of preferences.

## Format priority: a single, auditable rule

When two files are the same song, which one should win? AlphaDEX answers this with a
fixed **format priority**: lossless formats outrank lossy ones, so a FLAC beats a WAV
beats an MP3 for the same recording. The reason is simply quality — the lossless file
preserves more of the original sound.

What matters architecturally is that this preference is written down as plain
settings in one spot, not scattered as guesswork across the code. That makes it easy
to *see* the rule, easy to *audit* whether the app is behaving according to it, and
easy to *change* if your priorities differ. Wherever the app must break a tie between
duplicates, it consults this one ranking.

## The workspaces: the modern app's rooms

The modern app is organized as a **sidebar of workspaces** — each workspace is a
self-contained "room" dedicated to one job. You move between them from the sidebar,
and the currently selected room fills the main area. Behind a row of friendly names
sits a consistent pattern: every workspace is built on a shared template that gives it
the same skeleton (a place for content, a way to report progress to the log drawer at
the bottom, and a status indicator at the top).

The current rooms are, in plain terms:

| Workspace | What you do there |
|---|---|
| **Indexer** | Scan and reorganize your library into clean Artist/Album/Track folders. |
| **Library Sync** | Merge an incoming folder into your existing library. |
| **Duplicates** | Find and clear out duplicate copies of songs. |
| **Similarity** | Compare two specific tracks and see *why* the app thinks they do or don't match — useful for understanding the duplicate detector. |
| **Tag Fixer** | Look up and correct the artist/album/title information on your tracks. |
| **Genres** | Batch-update genre labels from online music databases. |
| **Playlists** | Build playlists by folder, tempo, energy, or automatic DJ flow. |
| **Clustered** | The original clustering room: group tracks by sound and turn the groups into playlists. |
| **Clustered Enhanced** | The newer clustering room, with a step-by-step wizard and richer results. |
| **Graph** | Explore your collection as an interactive visual map of sonic clusters. |
| **Player** | A built-in audio player with a library browser and queue. |
| **Compression** | Make a space-saving copy of a library, converting lossless files to a compact format while preserving their information. |
| **Tools** | A catch-all for exports, diagnostics, and utilities. |
| **Help** | Documentation, shortcuts, and project links. |

(Two additional internal modules in the same folder are scaffolding — a shared
template and a registry — rather than user-facing rooms.)

## How the app wakes up

When you launch the modern app, it does not slam you straight into a busy work
screen. It opens with a calm **landing moment**: the window fades in, the brand
appears for a beat, and a mosaic of album-art tiles animates into place while you
confirm which library you want to work on. Then the landing gracefully cross-fades
into the main window.

There is a deliberate choice hiding in this choreography. The genuinely expensive
work — scanning and fingerprinting your library — is **postponed** until the opening
animations have finished, so that the introduction stays smooth instead of stuttering
behind a frozen progress bar. The app waits for whichever comes first: a short delay
after you press "Initialize," or a longer safety timeout if you linger. The point is
that first impressions feel effortless, and the heavy lifting begins only once you're
settled in.

## A note on online help: what's real and what's a placeholder

Several features can consult **online music databases** to look up information about
your tracks. It is worth being honest about which of those connections actually work
today:

- **Fully working:** AcoustID (fingerprint-based identification), MusicBrainz (the
  open music encyclopedia), and Last.fm (genre and tag data). These three power the
  Tag Fixer and the Genres workspace.
- **Listed but not yet built:** Spotify and Gracenote appear in the settings as
  options, but their connections are empty placeholders — selecting them currently
  returns nothing. They are reserved for future work, partly because they require
  paid or partnership-based access rather than the open access the others enjoy.

This distinction is carried through into **ROADMAP.md**, which tracks what is planned
but not finished.

---

*Next: dive into any feature in the **features/** folder, or see **ROADMAP.md** for
what's still ahead.*
