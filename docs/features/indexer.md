# Feature — The Indexer

*The Indexer is the librarian that turns a chaotic pile of music files into a clean,
predictable, browsable collection. This document explains, in plain language, what it
does, the reasoning behind its choices, and why it is careful never to surprise you.*

---

## The problem it solves

Picture a music folder that has grown wild over a decade. Some songs sit in neat
`Artist/Album` folders; others are dumped loose in a `Downloads` directory with names
like `track01.mp3` or `Beyoncé - Halo (1).flac`. Capitalization is inconsistent.
There are stray "copy of" duplicates. Some files have rich, correct tags embedded in
them; others have almost nothing. Finding anything is a chore, and any music player
pointed at this folder inherits the mess.

The Indexer's job is to walk through that chaos and produce a single, consistent
structure: every track filed under its artist and album, named in a uniform way, with
the oddities swept into clearly-labeled side folders for you to deal with.

## The core metaphor: a postal sorting office

The cleanest way to picture the Indexer is as a **postal sorting office**. Every file
is a letter. The office reads the address on each letter (the track's embedded
information — artist, album, title, track number), decides which pigeonhole it belongs
in, and stacks it there. Letters with no readable address don't get guessed at and
flung somewhere random; they go to a "return to sender" desk for a human to handle.

Crucially, the sorting office first produces a complete **plan on paper** of where
every letter *would* go. Nothing actually moves until you read that plan and approve
it.

## How it works, stage by stage

### 1. Walking the library
The Indexer recursively explores your library folder, gathering every supported audio
file (FLAC, M4A, AAC, MP3, WAV, OGG). As it walks, it **steps around the reserved
folders** — `Not Sorted`, `Playlists`, `Manual Review`, `Docs`, `Trash`, and
`Quarantine` — because those are spaces you or the app have deliberately set aside,
and reorganizing them would defeat their purpose.

### 2. Reading the address
For each file it reads the embedded tags: artist, title, album, year, genre, track
number, and disc number. If the essential tags are missing, it will try to make sense
of the file name as a fallback — but if it still can't determine at least an artist
and a title, the file is destined for the "return to sender" desk (see *Manual
Review* below).

### 3. Taking a census of artists
Before deciding where anything goes, the Indexer does something clever: it takes a
**census** of your whole collection to count how many tracks each artist has. This
matters because it lets the Indexer treat prolific artists differently from one-off
appearances. An artist who shows up many times across your library is treated as a
"known" artist worthy of their own folder; an artist who appears only once or twice is
treated as a rarity. (The dividing line is currently set at ten tracks.) This census
also builds a consistent spelling for each artist, so that "deadmau5," "Deadmau5," and
"DEADMAU5" all converge on a single canonical name rather than fragmenting into three
folders.

### 4. Deciding where each track goes
With the census in hand, the Indexer routes each track using a layered set of rules.
In plain terms:

- **Rare artists** (below the ten-track line) are filed by **year** rather than by
  artist, on the logic that a one-off track is easier to find chronologically than
  buried in a sea of single-artist folders.
- **Known artists** get their own `By Artist/{Artist}` home.
- When a track credits **several artists** (a collaboration, a "featuring," a remix),
  the Indexer picks the most prominent one — the collaborator who appears most often
  across your collection — as the folder owner, so the track lands where you're most
  likely to look for it.
- Within an artist's space, the Indexer distinguishes **real albums** from **singles
  and loose tracks**. A genuine album (more than a few tracks sharing an album name)
  gets its own album folder; a lone track, or one whose "album" is really just the
  song title, is gathered into a `Singles` folder so it doesn't create a clutter of
  one-song album folders.
- **Multi-disc albums** are split into `Disc 1`, `Disc 2`, and so on, but only when
  the Indexer actually sees more than one disc number for that album — it won't invent
  disc folders where they aren't warranted.
- **Remix collections** get special handling so that an album explicitly built of
  remixes is filed under the remixer in a sensible way rather than scattered.

The thread running through all of these rules is the same: **file each track where a
human would intuitively go looking for it**, and avoid creating noise (like hundreds
of single-song folders) in the process.

### 5. Cleaning up the names
As it assigns destinations, the Indexer **normalizes** file and folder names — the
unglamorous but essential work of making names consistent and safe. In practice this
means:

- **Removing characters the file system forbids or dislikes** — the slashes, colons,
  question marks, and quotation marks that can break folders on some systems are
  stripped out, while harmless characters like apostrophes are left alone.
- **Standardizing odd text** — accented and unusual characters are converted to a
  consistent form so that names sort and match predictably.
- **Falling back gracefully** — if cleaning a name somehow leaves nothing behind, the
  Indexer substitutes a safe placeholder like "Unknown" rather than producing a
  blank, broken name.
- **Catching garbled artist fields** — occasionally a tag is mangled into something
  like a name repeated twice ("DROELOEDROELOE"). The Indexer recognizes these
  patterns and keeps the original file name instead of building a folder around the
  garbage.

Files end up named in a uniform `Artist_TrackNumber_Title` shape, so that within any
folder the tracks line up in a tidy, readable order.

### 6. Handling collisions
Two different files can occasionally claim the exact same destination — for instance,
two versions of the same track number and title. The Indexer detects these
**collisions** and gently distinguishes them by adding a numeric suffix, so nothing
silently overwrites anything else. If a later pass finds that a collision no longer
exists, it tidies the now-unnecessary suffix back off.

### 7. The preview, then the move
Finally — and only as a plan, not an action — the Indexer writes an **HTML preview**:
a web page you can open in any browser showing the proposed folder tree, every file's
new name, and notes explaining notable decisions (why this counted as a real album,
why that track collided, and so on). At this moment **nothing on your disk has
changed.** You read the plan; if you approve it, a separate execution step actually
performs the moves, sweeps leftover non-music files into `Docs` or `Trash`, removes
the now-empty folders, updates your playlists to point at the new locations, and
writes a final log of everything it did.

## The Manual Review desk

Some files simply don't carry enough information to be filed responsibly — they're
missing an artist or a title, and even the file name offers no clue. Rather than guess
and risk burying a track under the wrong artist forever, the Indexer routes these to
the **`Manual Review`** folder and leaves them untouched, with a note in the log.

This is a deliberate philosophy: **it is better to set a file aside than to misfile
it.** A track parked in Manual Review is easy to find and fix; a track silently filed
under "Unknown Artist / Unknown Album" can be lost for years. Once you've corrected
its tags, you simply feed it back through and it gets filed properly.

## What the Indexer deliberately does not do

It's worth being explicit about a boundary that's easy to assume away: **the Indexer
does not detect or remove duplicates.** It treats every file independently and files
it on its own merits — it does not compare two files to each other to decide which
one is "better." The code itself states this as a deliberate design choice, not an
oversight:

> "Duplicate detection and keep/winner ranking are intentionally excluded from the
> indexer. The indexer treats every file independently and relies on deterministic
> routing plus collision handling (suffixing/buckets)."

So if two copies of the same song land in the same album folder, the Indexer will
happily file both of them — side by side, disambiguated with a numeric suffix if
their names collide — rather than trying to pick a winner. That job belongs entirely
to the **Duplicate Finder** (next document). If you run the Indexer on a library full
of duplicates and expect them to be thinned out, they won't be; run the Duplicate
Finder afterward for that.

*(One more thing worth knowing if you also use Library Sync: that feature runs its
own, separately-maintained copy of this same routing/fingerprinting logic rather than
calling this module directly. See **architecture.md**'s note on the "two indexer
engines" if you're changing indexer behavior and wondering why Library Sync doesn't
seem to have picked up your fix.)*

## Why a web-page preview?

You might wonder why the plan is an HTML page rather than just a list inside the app.
The choice is intentional. A web page is **rich, portable, and permanent**: you can
open it in your browser, scroll through a large plan comfortably, search within it,
print it, or keep it as a record of what the library looked like before a big
reorganization. And because generating that page changes nothing on disk, it doubles
as the firewall between "thinking about it" and "doing it." The preview *is* the
safety mechanism, not just a convenience.

## The reserved folders, from the Indexer's point of view

The Indexer is the feature that most directly enforces the reserved-folder system:

- It **skips** `Not Sorted`, `Playlists`, `Manual Review`, `Docs`, `Trash`, and
  `Quarantine` when scanning, so they are never reorganized out from under you.
- It **populates** `Manual Review` with under-described tracks, `Docs` with its
  reports and logs, and `Trash` with stray non-music files it sweeps out of the way.

Understanding these folders is most of what you need to predict the Indexer's
behavior.

---

*Related: the **Duplicate Finder** (next) handles the case where the Indexer's tidy
library still contains two copies of the same song.*
