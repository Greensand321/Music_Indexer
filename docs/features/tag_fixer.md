# Feature — The Tag Fixer

*The Tag Fixer looks up the correct artist, title, album, and genre information for
your tracks from online music databases and proposes corrections for you to review
before anything is written. This document explains how it actually works today,
including a few places where the current behavior is narrower than you might expect
from the feature's name.*

---

## What it solves

Ripped CDs, old downloads, and years of manual retagging leave a library full of
small metadata sins: an artist name spelled two different ways, a title missing its
featured-artist credit, an album field that's blank, a genre that was never filled in
at all. The Tag Fixer's job is to look each track up against an online reference and
propose the correction — never to just silently rewrite your files.

## How a scan works

For every supported audio file under the folder you point it at, the Tag Fixer:

1. **Reads the file's current tags.** This becomes the "before" side of any proposed
   change.
2. **Asks every available lookup service to identify the track**, and keeps whichever
   answer came back with the highest confidence score.
3. **Only turns that answer into a proposal if the winning score clears a
   threshold** (0.75 out of 1.0 today). Below that, the track is left alone — the app
   would rather show you nothing than a guess it isn't confident about.
4. **Compares the proposed values to what's already on the file** and shows you only
   the fields that would actually change.

The result is a table: one row per proposed field change, with a checkbox so you
choose exactly which corrections to accept. Nothing is written until you explicitly
apply your selection.

### An honest note on the "score tiers"

If you read the code, you'll find two named thresholds:  a 0.90 "apply
automatically" tier and a 0.75 "prompt the user" tier. In the current app, only the
0.75 tier is real — every match that clears it becomes a reviewable proposal, full
stop. There is no code path anywhere that writes a tag automatically, no matter how
confident the match. If you were expecting a "just trust the 90%+ matches" mode, it
doesn't exist yet; every accepted change today goes through your review.

## Which online services actually answer

The Tag Fixer can, in principle, ask five different services. In practice, one
matters for the automatic scan, and the split is worth knowing:

- **AcoustID** is the one service that's wired into the automatic scan today. It
  fingerprints the audio itself (not just the existing tags) and, when it finds a
  MusicBrainz recording match, enriches the answer with that recording's album and
  genre tags.
- **MusicBrainz** and **Last.fm** are both fully functional services elsewhere in the
  app (MusicBrainz search, and Last.fm's popularity-based genre tagging), but neither
  is currently part of the Tag Fixer's automatic plugin scan — MusicBrainz's lookup
  class isn't built on the same plugin base the scan discovers, so it never
  contributes an independent identification; it only ever shows up *nested inside* an
  AcoustID match's enrichment step.
- **Spotify** and **Gracenote** are honestly labeled as unavailable — see
  **architecture.md**'s note on this. They contribute nothing today.

This means a practical mental model of "what the Tag Fixer actually asks" is: *"does
AcoustID recognize this recording, and if so, what does MusicBrainz say about it?"* —
narrower than the presence of five services in the settings list might suggest.

## Genres are additive, never a replacement

This is worth calling out specifically because it's easy to assume "fixing" a genre
means replacing it. It doesn't. When the Tag Fixer writes a genre change, it
**merges** the newly proposed genres into whatever the file already has — it unions
the two lists, de-duplicates, and writes the combined set back. It never removes a
genre that was already there, even a wrong one.

The practical effect: running the Tag Fixer repeatedly on the same file can only ever
*grow* its genre list, never correct a bad entry that's already present. If a track
is mistagged as "Pop" and you'd like it purely as "Electronic," the Tag Fixer alone
won't get you there — you'd need to clear the tag by hand first, or use the separate
Genre Normalizer tooling (see below).

## The "dry run" checkbox — what it actually guards

The Qt workspace has a "Dry run (propose only, do not write tags)" checkbox, checked
by default. It's worth understanding exactly what it does, because it's simpler than
it sounds: it's a **pure interface gate**. If it's checked, clicking "Apply Selected"
just shows a message telling you to uncheck it — the backend functions that actually
write tags have no dry-run mode of their own; there is no separate code path that
"pretends" to write. The checkbox doesn't need to be more than that, because the
review table itself *is* the preview — nothing is looked up-and-applied in one step
regardless of the checkbox's state; the write only happens when you explicitly click
Apply with real fields selected and confirm a "this cannot be undone" prompt.

## CLI vs. GUI: a real behavioral difference

The Tag Fixer also has a command-line entry point (`python tag_fixer.py <folder>
[--interactive]`), and it's worth knowing it behaves differently from the Qt
workspace in two ways:

- **The CLI only ever writes artist and title**, even if a scan proposed album or
  genre changes too — those two fields are hard-coded into the CLI's apply call. Only
  the GUI paths (Qt and legacy Tkinter) can write album and genre corrections.
- **The CLI skips files it already marked "applied" on a previous run** (and skips
  anything that looks like a remix, by filename or title). The Qt workspace always
  re-scans and re-queries every file, every time, regardless of prior status — so a
  fresh Qt scan re-hits AcoustID for files it already fixed last time.

If you're scripting a batch tag-fix and expect it to also normalize genres, use the
GUI; the CLI is narrower by design.

## What gets remembered between scans

Each library gets a small SQLite database (`Docs/.soundvault.db`) that records, per
file, its current proposal status (new / unmatched / no-difference / applied /
skipped). This is what lets the CLI's "skip already-applied files" behavior work.

One thing this database does **not** do, despite having a table that implies it
might: there is no cache of raw fingerprints or API responses. Every Qt scan
re-queries AcoustID for every file; nothing about network lookups is memoized. If
you're scanning a large library repeatedly, expect the same lookup cost each time.

*(A gotcha to know about if you also use Library Sync: that feature's fingerprint
cache lives in the same `Docs/.soundvault.db` file inside your library. The legacy
app's "Reset Log" button for the Tag Fixer deletes this file outright — which also
silently clears Library Sync's cache for that library. There's no warning that the
two features share a database.)*

## A quirk worth knowing: a test fixture ships live

The plugin system that powers the automatic scan discovers its lookup sources by
scanning the `plugins/` folder for anything that looks like a plugin — including
`plugins/test_plugin.py`, a fixture built for the automated test suite that returns a
fake, perfect-confidence match for any file whose name starts with `dummy_`. It is
not excluded from a normal run. In the extremely unlikely event you have a real file
named `dummy_something.mp3`, it will get a fabricated tag proposal from this test
fixture rather than a real lookup. Harmless in practice, but worth knowing if you
ever see a suspiciously perfect, suspiciously generic match.

---

## A related, differently-named feature: the Genre Normalizer

The sidebar has a workspace called **Genre Normalizer**, and it's important to
understand that **this label means two different things depending on which app
you're in** — worth being very clear about, since the same words describe two
unrelated pieces of code.

- **In the modern (Qt) app**, "Genre Normalizer" is a thin wrapper around a batch
  MusicBrainz genre updater: for each file with fewer than two existing genre tags,
  it looks up the recording on MusicBrainz, takes the three most popular community
  tags, and writes them in. There is no canonical mapping step — "normalize" here
  just means "pick MusicBrainz's most popular raw tags," not "map messy variants to a
  clean, consistent vocabulary." Its "Overwrite existing genre tags" checkbox and its
  MusicBrainz/Last.fm/Both source selector are both currently decorative — neither
  has any effect on what the tool actually does (it always uses MusicBrainz, and it
  always skips files that already have 2+ genres, regardless of the checkbox).
- **In the legacy (Tkinter) app only**, "Genre Normalizer" is a genuinely different,
  more ambitious feature: it scans your library's raw genre tags, hands you a
  ready-to-paste prompt for an external LLM (ChatGPT, Claude, or similar) asking it to
  group your messy raw genres into a small set of canonical categories, and lets you
  paste the LLM's JSON answer back in to build a genre-mapping table that then gets
  applied library-wide. This is the actual canonical-genre-mapping system the name
  implies — but it does not exist in the modern app at all.

If your goal is genuinely consolidating "Hip Hop," "hip-hop," and "Rap" into one
clean label, that capability currently only exists in the legacy Tkinter app.
Bringing the canonical-mapping workflow into the modern app is a real, trackable gap
— see **ROADMAP.md**.

*(The same legacy app also has a "Year Assistant" that fills in missing `year` tags
via the same copy-paste-to-an-LLM pattern, with a genuinely honored dry-run mode and
its own audit log. It is unrelated to playlist ordering — see
**features/playlists_and_clustering.md** for the correction of an earlier, inaccurate
description of a "year-gap" playlist feature that doesn't actually exist.)*

---

*Related: **Playlists & Clustering** covers what happens once your tags are trustworthy —
turning a clean library into playlists, automatically grouped by how the music
actually sounds.*
