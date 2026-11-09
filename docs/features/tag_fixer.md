# Feature — The Tag Fixer

*The Tag Fixer looks up the correct artist, title, album, and genre information for
your tracks from online music databases and proposes corrections for you to review
before anything is written. This document explains how it actually works today,
including a few places where the current behavior is narrower than you might expect
from the feature's name.*

---

**Where to find it.** In the app the Tag Fixer is **Metadata → Tag Fixer**. The same
page holds the two genre tools, **Metadata → Fill genres** and **Metadata → Normalize
genres** (covered at the end of this document), and **Metadata → Undo changes**, which
rolls back an earlier apply from any of the three.

## What it solves

Ripped CDs, old downloads, and years of manual retagging leave a library full of
small metadata sins: an artist name spelled two different ways, a title missing its
featured-artist credit, an album field that's blank, a genre that was never filled in
at all. The Tag Fixer's job is to look each track up against an online reference and
propose the correction — never to just silently rewrite your files.

## How a scan works

Before looking anything up, the scan checks each lookup source once — is
fpcalc (Chromaprint) installed, is api.acoustid.org reachable, is a key set — and
prints a "Lookup sources: …" line. A source that can't run is **skipped, with the
reason shown**, instead of every file silently coming back "unmatched".

Then, for every supported audio file in the chosen scope (the whole library, or one
folder inside it, picked in the Tag lookup card; either way the reserved top-level
folders `Not Sorted/`, `Quarantine/`, `Trash/`, `Docs/` and `Playlists/` are never
entered; `Manual Review/` *is* scanned, because its tracks are exactly the ones
missing metadata), the Tag Fixer:

1. **Reads the file's current tags** — every value of every field it can write, not
   just the first (a track tagged with two artists shows both). This becomes the
   "before" side of any proposed change.
2. **Asks every available lookup source to identify the track**, and keeps whichever
   answer came back with the highest confidence score.
3. **Only turns that answer into a proposal if the winning score clears a
   threshold** (0.75 out of 1.0 today). Below that, the track is left alone — the app
   would rather show you nothing than a guess it isn't confident about. Genres are
   gathered only from answers that clear the same threshold, so a weak match can't
   slip its genres in beside a confident one.
4. **Compares the proposed values to what's already on the file** and shows you only
   the fields that would actually change. A field the service didn't return is never
   proposed as blank. An artist, title or album proposal **replaces every value** of
   that field with the one proposed value, and the row says so: a file tagged
   "Artist A" and "Artist B" shows "Artist A; Artist B → New Artist", and the change
   log records both old values.

The result is a table: one row per proposed *field* change, built from the same
function the writer uses, so a row shows exactly what applying it writes. Each row's
checkbox selects that field change — untick a file's Genre row and keep its Artist
row, and only the artist is written. The "Show" checkboxes hide or show a
field's rows; hidden rows are never written. Files nothing could identify are not
shown as blank proposals: they are listed under "No match found", each with its
reason (a lookup error, "no lookup source available", a low-confidence match…).
Cancel stops a scan before the next file.

Nothing is written until you explicitly apply your selection and confirm. Apply
writes exactly the ticked rows; a file whose tags changed on disk since the scan is
skipped (never overwritten blind); one bad file (read-only, an odd format) is
reported and the rest of the batch continues; only files whose write succeeded are
marked "applied". Every write — and every failure or skip — is recorded with its
before and after values in `Docs/tag_changes_tagfixer_<timestamp>.jsonl`. After
applying, the written rows drop out of the table and the log is kept; there is no
automatic re-scan. Only one tag-writing job runs at a time across the Tag Fixer and
both genre sections.

**Undoing an apply.** Those logs are enough to put the old values back, and
`controllers/tag_revert.py` does it the preview-first way: pick a log (newest first),
see per file and field what would be restored, then write exactly that. A file whose
tags changed since the logged write is skipped and reported, never overwritten; a
file that has been moved or deleted is reported as missing. Every value of a
multi-value field comes back, a field that did not exist before is removed again, and
WAV files with ID3 tags work too. The revert is itself logged
(`Docs/tag_changes_revert_<timestamp>.jsonl`), so it can be reverted. In the app this is
**Metadata → Undo changes**: pick an apply, **Preview undo** (writes nothing), then
**Undo these changes**.

### An honest note on the "score tiers"

If you read the code, you'll find two named thresholds:  a 0.90 "apply
automatically" tier and a 0.75 "prompt the user" tier. In the current app, only the
0.75 tier is real — every match that clears it becomes a reviewable proposal, full
stop. There is no code path anywhere that writes a tag automatically, no matter how
confident the match. If you were expecting a "just trust the 90%+ matches" mode, it
doesn't exist yet; every accepted change today goes through your review.

## Which online services actually answer

The Tag Fixer can, in principle, ask five different services. Which ones a scan
uses depends on what is installed and on **Settings → Metadata services**:

- **AcoustID** always runs when it can. It fingerprints the audio itself (not just
  the existing tags) and, when it finds a MusicBrainz recording match, enriches the
  answer with that recording's album and genre tags. It needs fpcalc (Chromaprint)
  and the network; the key is the one in Settings, or the built-in key when that
  field is empty. If the enrichment step fails, the scan says so.
- **MusicBrainz search** joins the scan when the Service in Settings → Metadata
  services is set to "MusicBrainz": each track is also looked up by its existing artist and title,
  which mostly contributes album and genre proposals. MusicBrainz requests always
  use the app name / version / contact from Settings (blank fields fall back to
  "AlphaDEX 1.0"), so enrichment no longer depends on which page you opened first.
- **Last.fm** joins only when a Last.fm API key is set: the one saved in Settings →
  Metadata services, or the `LASTFM_API_KEY` environment variable, which wins when
  both are set. Settings shows one key field for whichever service you pick, but each
  service keeps its own key (`metadata_api_key` for AcoustID, `lastfm_api_key` for
  Last.fm), so saving a Last.fm key no longer replaces your AcoustID key; "Test
  Connection" tests the key you type. Picking "Last.fm" with no key set makes the scan
  say where the key goes. (The field used to be shared, and the scan read the Last.fm
  key from the environment variable only. Both are fixed.)
- **Spotify** and **Gracenote** are honestly labeled as unavailable — see
  **architecture.md**'s note on this. They contribute nothing today.

## Genres are additive, never a replacement

This is worth calling out specifically because it's easy to assume "fixing" a genre
means replacing it. It doesn't. When the Tag Fixer writes a genre change, it keeps
**every** genre the file already has, in order, and adds each proposed genre it
lacks (ignoring letter case) after them — written as a real multi-value tag, not a
single "Rock; Pop" string. It never removes a genre that was already there, even a
wrong one, and it doesn't touch the genre at all when the lookup found none.

The practical effect: running the Tag Fixer repeatedly on the same file can only ever
*grow* its genre list, never correct a bad entry that's already present. If a track
is mistagged as "Pop" and you'd like it purely as "Electronic," the Tag Fixer alone
won't get you there — you'd need to clear the tag by hand first, or use
**Metadata → Normalize genres** (see below).

## The "dry run" checkbox — what it actually guards

The Tag Fixer section has a "Dry run (propose only, do not write tags)" checkbox, checked
by default. It's worth understanding exactly what it does, because it's simpler than
it sounds: it's a **pure interface gate**. If it's checked, clicking "Apply changes"
just shows a message telling you to uncheck it — the backend functions that actually
write tags have no dry-run mode of their own; there is no separate code path that
"pretends" to write. The checkbox doesn't need to be more than that, because the
review table itself *is* the preview — nothing is looked up-and-applied in one step
regardless of the checkbox's state; the write only happens when you explicitly click
Apply with ticked rows and confirm a prompt that says how many field changes will be
written to how many files.

## CLI vs. GUI: a real behavioral difference

The Tag Fixer also has a command-line entry point (`python tag_fixer.py <folder>
[--interactive]`), and it's worth knowing it behaves differently from the Qt app's
Tag Fixer in two ways:

- **The CLI only ever writes artist and title**, even if a scan proposed album or
  genre changes too — those two fields are hard-coded into the CLI's apply call. Only
  the GUI paths (Qt and legacy Tkinter) can write album and genre corrections.
- **The CLI skips files it already marked "applied" on a previous run** (and skips
  anything that looks like a remix, by filename or title). The Qt app always
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

## Where the lookup sources come from

The automatic scan loads its lookup sources from a fixed list of plugin modules in
`plugins/` (AcoustID, Last.fm, MusicBrainz, and the shared service base class). It
used to discover them by scanning the folder for anything that looked like a plugin,
which also picked up `plugins/test_plugin.py` — a fixture for the automated test
suite that returns a fake, perfect-confidence match for any file whose name starts
with `dummy_` — and which could not find the plugins at all in a frozen (packaged)
build. With the fixed list, the test fixture never runs in a real scan.

---

## The genre tools: Metadata → Fill genres and Normalize genres

The Metadata page has two genre sections. They are one panel underneath (its
sidebar entry used to be called "Genre Normalizer"), and two sections because they answer two genuinely different questions:

- **Fill genres** (the panel calls it "Fill from MusicBrainz") — *"some of my tracks
  have no genre at all."* For each
  file with fewer than two existing genre values (every value is counted, not just
  the first), it looks the recording up on MusicBrainz by artist and title and takes
  the three most popular community tags. It is preview-first: **Look up genres**
  only builds a table of proposed additions (current → after, and the MusicBrainz
  match); nothing is written until you click Apply and confirm, and then exactly
  that table is written. A genre the file already has is kept, with the new tags
  added after it. Files that already carry two or more genres are skipped, so this
  fills gaps rather than second-guessing choices you've already made. Opus files are
  included, like everywhere else. The command-line version
  (`python update_genres.py [folder] [--yes]`) follows the same rules and is now
  preview-first too: it lists every planned change and asks before writing (it used
  to write each file as soon as its lookup returned), writes exactly that list, and
  logs to `Docs/genre_update_log.txt` and `Docs/tag_changes_genre_fill_cli_*.jsonl`.

- **Normalize genres** ("Normalize with a mapping") — *"my genres are a mess of
  spellings."* This is the
  canonical-mapping workflow: collapse "Hip Hop," "hip-hop," "hiphoprap," and "Rap"
  into whatever clean vocabulary *you* choose. It runs in four numbered steps on the
  screen, and the shape is deliberately preview-first:

  1. **Scan** your library to collect every distinct raw genre string (combined
     entries like "Hip-Hop/Rap" are split, so the mapping sees each part).
  2. **Build the mapping.** Copy the provided prompt, paste it and the raw list into
     an LLM of your choice, and paste back the JSON it returns — keys are raw genres,
     values are the canonical name(s). A value of `["invalid"]` or `null` means "this
     isn't a genre; drop it." Unmapped genres pass through unchanged, so a partial
     mapping never makes things worse. Mappings can be saved to and reloaded from
     `Docs/.genre_mapping.json`, so you build it once and reuse it.
  3. **Preview.** The app reads every genre value of every file and shows a table of
     exactly which files would change and how (current genres → new genres), plus
     how many were already canonical and how many had no genre to normalize. A
     combined value such as "hip hop/rap" is mapped as a whole if the mapping has
     it, otherwise part by part. Files that would change but can't be written
     (read-only, a format that can't hold tags) are listed up front and left out.
     Nothing is written.
  4. **Apply.** After a confirmation, it writes *exactly* what the preview showed —
     the plan, not a fresh decision. Editing the mapping after previewing disables
     Apply until you preview again, the same rule Library Sync applies to its plan;
     switching library throws the preview away. A file whose genres changed on disk
     since the preview is skipped.

  Both sections log every write with its previous value to
  `Docs/tag_changes_genre_fill_*.jsonl` / `Docs/tag_changes_genre_normalize_*.jsonl`,
  so either kind of apply can be rolled back from **Metadata → Undo changes**, the
  same way as a Tag Fixer apply.
  An unsaved mapping in the editor is kept per library when you switch libraries,
  and "Load saved mapping" asks before replacing unsaved edits.

  This workflow previously existed only in the legacy Tkinter app, where "Apply"
  read, decided and wrote in a single pass with no preview, and where the `["invalid"]`
  marker the prompt asks for was written into files as a literal genre named
  "invalid." The port fixed both: it's preview-first now, and `["invalid"]` drops the
  entry as the prompt promises.

*(The same legacy app also has a "Year Assistant" that fills in missing `year` tags
via the same copy-paste-to-an-LLM pattern, with a genuinely honored dry-run mode and
its own audit log. It is unrelated to playlist ordering — see
**features/playlists_and_clustering.md** for the correction of an earlier, inaccurate
description of a "year-gap" playlist feature that doesn't actually exist.)*

---

*Related: **Playlists & Clustering** covers what happens once your tags are trustworthy —
turning a clean library into playlists, automatically grouped by how the music
actually sounds.*
