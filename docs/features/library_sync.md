# Feature — Library Sync

*Library Sync merges a new batch of music into your existing collection — adding what's
genuinely new, upgrading what's better, and refusing to create duplicates — all under
your review. This document explains the workflow and the thinking behind it in plain
language.*

---

## The problem it solves

You keep a carefully tended **master library** on your main machine. Then a new batch
of music arrives: a folder of downloads, a friend's USB drive, the contents of an old
laptop. You want to fold the good parts of that **incoming folder** into your master
library, but doing it by hand is a minefield. Which of these tracks are genuinely new?
Which are duplicates you already own? Which are *better* versions of songs you have —
a lossless copy of something you only had as an MP3 — that you'd actually like to
upgrade to? Drag everything in and you get a mess of duplicates; cherry-pick by hand
and you'll miss things.

Library Sync automates the judgment while leaving the final decisions to you. It
compares the two collections track by track and proposes, for each incoming file, one
of a few sensible outcomes: **add it** (it's new), **skip it** (you already have it as
good or better), or **flag it as an upgrade** (it's a higher-quality version of
something you own).

## The workflow as an assembly line

Library Sync is best understood as an **assembly line**, where each station refines the
work of the previous one and hands the result down the line. Nothing is welded shut
until the very end.

1. **Choose the two folders.** You point Library Sync at your existing library and at
   the incoming folder, and set how strict the matching should be. Both folders must
   exist and must not be the same folder or inside one another — Library Sync refuses
   rather than creating a mistyped path.
2. **Scan and fingerprint.** Both collections are read and fingerprinted (reusing the
   cache wherever possible, so repeat runs are fast). Each track becomes a tidy record
   carrying its fingerprint and its quality details. The reserved folders
   (`Quarantine/`, `Trash/`, `Docs/` — including Library Sync's own backups —
   `Manual Review/`, `Playlists/`, `Not Sorted/`) are not scanned, so a quarantined
   duplicate loser is never mistaken for a track you own.
3. **Match.** Every incoming track is compared against your library to find its closest
   counterpart, and each pairing is labeled — brand new, exact duplicate, likely
   duplicate, or borderline.
4. **Review.** You inspect the matches and, where you disagree with the automatic
   call, **override** it (this is the feature's standout capability — see below).
5. **Build the plan.** Your decisions, plus the automatic ones, are turned into a
   concrete list of copies and replacements.
6. **Preview.** That plan is rendered as a readable report showing exactly what will
   happen — before anything happens.
7. **Execute and report.** On your approval, the files are moved or copied, every
   existing file that gets replaced is backed up first, every action is logged, and a
   final human-readable summary is produced.

The first three stations gather and analyze; the middle station hands you the wheel;
the last three commit, but only after you've seen the whole plan.

This assembly-line shape is not the feature's original shape — it's the result of a
deliberate redesign. Two older functions still exist in the code
(`copy_new_tracks`/`replace_tracks`) purely as guard rails, and calling either one
raises an error on purpose:

> "Deprecated: blocked per review-first redesign." — raises `RuntimeError("File
> operations are disabled in the Library Sync review tool.")`

In other words, Library Sync used to have direct copy/replace entry points that
skipped the plan-preview-execute pipeline, and they were intentionally disabled when
the feature moved to the review-first design described above. If you ever see a
reference to those two function names, know that they're deliberately dead ends, not
bugs.

## How the matching works

At the heart of stations 2 and 3 is the same **fingerprint-and-distance** idea used by
the Duplicate Finder: each track is reduced to a numeric summary of its sound, and two
tracks are compared to yield a distance between **0** (identical) and **1** (unrelated).

For each incoming track, Library Sync finds the closest match in your library and
classifies the relationship:

- **New** — nothing in your library sounds close enough; this is a fresh track.
- **Exact match** — your library already contains this exact recording.
- **Collision / likely duplicate** — close enough to be confidently the same song.
- **Low confidence** — sitting right on the borderline, worth a human glance.

The distance is the share of differing bits between the two *decoded* Chromaprint
fingerprints: the same recording in any codec scores about 0.01, a copy whose start is
shifted by a fraction of a second 0.07–0.14, and different recordings 0.43 or more. The
default threshold is **0.2** for every format (a lossless-vs-lossy pair gets +0.03). The
strictness is adjustable, but from 0.4 upward unrelated tracks start to match, and the
page warns you. Borderline cases are deliberately surfaced as "low confidence" rather
than silently decided — that band reaches 25 % above the threshold but never 0.4 — and
a **match confidence** score (a friendly 0-to-100% reading derived from how far inside
or outside the threshold a pairing falls) shows not just *whether* two tracks matched
but *how sure* the app is.

**Finding candidates.** Library Sync indexes the library's decoded fingerprints, so
each incoming track is compared with every library track that shares part of its
fingerprint, whatever its format or position in the library. When that finds no match,
it is also compared with every track of about the same length (±3 s or 2 %) after a
quick check of the first ~16 seconds. Fingerprints too short or too uniform to identify
a recording (under about 4 seconds, silence) never match anything. Building the index
needs every library fingerprint decoded — about 30 seconds for 20,000 tracks. The
decoded form is saved in Library Sync's fingerprint cache next to the fingerprint, so
only the first scan (and files that changed since: a different mtime or size) pays
that; later scans read it back. Recompute matches reuses the index outright. A threshold saved before this change that equals the old
default (0.3 / 0.35, which meant something else on the old scale) is replaced by 0.2
once, with a line in the log; any other saved value is kept.

**A clip is not the song.** A fingerprint covers the first two minutes of a track,
so a 30-second clip's fingerprint is the *beginning* of the full song's. Comparing only
the part the two have in common scored the pair 0 — an exact match — so an incoming
full song was "already in the library" when the library held only a clip of it, and a
lossless clip of a lossy library song was offered as an upgrade (one Replace flag away
from swapping the song for half a minute of it). The part of the longer fingerprint
that has no counterpart now counts as unrelated audio. A clip shorter than about half
of the song's fingerprinted length is **new**; around half it is **low confidence**
(sent to review, never decided automatically); the same recording in another codec,
or a copy a few seconds shorter or shifted, scores as before. The candidate search
still finds the song for its clip — they share their sub-fingerprints — so the pair
is compared; it is the distance that now tells them apart. Two tracks longer than two
minutes are compared on their first two minutes either way, so a long edit of a long
song still matches it.

**Short tracks are fingerprinted as they are.** The fingerprint covers the first
two minutes of a track (the fingerprint duration setting). A track shorter than that
used to be padded with silence up to the full window, so the prints of two
*different* short tracks agreed on all that silence: unrelated 5-second clips scored
about 0.01 (an "exact match"), 20-second clips about 0.05 and 40-second clips about
0.1 — so interludes, intros and short songs of similar length were matched to each
other. Nothing is padded now (`audio_norm.normalize_for_fp`), and Library Sync's
cache no longer reuses the padded fingerprints it already holds: a cached fingerprint
from before the fix whose end is a run of padding silence (about 5 seconds of one
repeated value) is a cache miss, so that track is fingerprinted once more on the next
scan. Every other cached fingerprint stays valid, so only short tracks are redone —
the stored duration could not tell them apart (it equals the window either way), the
fingerprint itself can. Fingerprints written now are marked as made without padding
and are always trusted, so a short track that really ends in silence is redone once,
not on every scan. The decoded copy kept next to a fingerprint (see *Finding
candidates*) is cleared whenever the fingerprint is rewritten.

**A damaged cache is rebuilt, not fatal.** If Library Sync's fingerprint cache file is
unreadable ("file is not a database"), it is renamed to `….corrupt-<date>` — not
deleted — and a fresh cache is started; that scan simply fingerprints everything. A
single unreadable row is a cache miss, and a fingerprint the cache cannot store is
still used for the scan.

Alongside the match, Library Sync also weighs **quality**: it compares the incoming
file's format and bitrate against the existing one and labels the pairing as a
"Potential Upgrade" (the incoming copy is better) or "Keep Existing" (yours is as good
or better). This is what lets it spot the lossless-upgrade case automatically.

### When the matching is wrong

No automatic matcher is perfect. It can occasionally pair two different songs that
happen to sound alike (a **false positive**), or fail to recognize a genuine duplicate
whose fingerprint drifted (a **false negative**). Library Sync's answer to both is the
same: **it never executes on the automatic decision alone.** The review station exists
precisely so that a human can catch and correct these cases before anything is
committed.

## The power feature: per-item review flags

This is the capability that sets Library Sync apart, so it's worth understanding well.

After matching, you're presented with the list of incoming tracks and the app's
proposed decision for each. Wherever you disagree, you **right-click the track** and
override the automatic call. Your options are:

- **Copy** — "I want this incoming track added, regardless of what the algorithm
  decided." This forces the track into your library even if the matcher thought it was
  a duplicate.
- **Replace** — "Use this incoming track *in place of* the existing one I already
  have." This authorizes overwriting your current copy with the incoming version — the
  way you'd accept an upgrade the app was too cautious to make on its own.
- **Clear** — remove any flag you've set, returning the track to the automatic
  decision.
- **Add Note** — attach a short free-text reminder to a track ("the remaster, keep
  this one"), purely for your own reference.

The important guarantee is that **your flags win.** When the plan is built, a track you
flagged "Copy" is guaranteed to be copied in, and a track you flagged "Replace" is
guaranteed to replace its counterpart — overriding whatever the automatic quality
comparison would have chosen. The algorithm advises; you decide.

Concretely, the plan is built *from the reviewed match table*:

| Match / flag | What the plan does |
|---|---|
| New track | Added to the library (copied or moved, per the Copy/Move toggle) |
| Exact match or likely duplicate | Skipped — no duplicate is copied in |
| "Potential Upgrade" (no flag) | Left for review: flag **Replace** to take it |
| Low confidence (no flag) | Left for review: flag **Copy** or **Replace** to decide |
| Flag **Copy** | Added regardless, always as a copy (the incoming original stays); a taken name gets a `(1)` suffix |
| Flag **Replace** | Takes the matched track's place (same folder and name, the incoming file's extension); the old file is backed up first and moved into the backups area |

The preview states the transfer mode, the counts, and the reason for every decision,
and **Execute runs exactly that plan.** If you change the paths, the Copy/Move toggle or
any flag after building it — or re-scan, or switch library — the plan is marked out of
date and Execute is disabled until you build it again.

A few practical notes:

- **Flags are remembered only for the current session.** They live in memory while you
  work, not in a saved file. This is a deliberate simplicity choice (see below).
  **Save session** remembers the two folders and the threshold (restored the next time
  you open the page), not the flags.
- **Re-scanning keeps only the flags that still make sense.** If you change the
  strictness and recompute the matches (the **Recompute matches** button re-matches the
  last scan without re-reading any file), the whole landscape of matches can shift — a
  track you flagged for "Replace" might now match something different, or nothing at
  all. Copy flags and notes are kept for tracks that are still there; a Replace flag is
  kept only if the track still matches the *same* library track (it is never silently
  re-pointed at a different file). Everything that could not be carried over is listed
  in a notice and in the log.
- **Flagging is currently one track at a time.** There is not yet a "select many and
  flag them all at once" shortcut; that bulk capability is on the roadmap.

### Why flags aren't saved permanently

You might expect your flags to persist across restarts. They deliberately don't, and
the reasoning is sound: flagging is part of a single, self-contained review session
that runs from scan to plan in one sitting. Saving flags would invite a subtler
problem — stale flags pointing at matches that no longer exist after the library or the
settings change. By keeping flags in memory and tied to the current set of match
results, the app stays simple and avoids quietly acting on out-of-date instructions.
Persisting flags across sessions is a recognized future enhancement, listed in the
roadmap, but the current trade-off favors safety and simplicity.

## A note on the two interfaces

Library Sync actually has **two separate front-end implementations** that both drive
the same backend engine described above: the modern Qt app's **Library Sync** page
(in the sidebar's Organize group; `gui/workspaces/library_sync.py`, described
throughout this document) and an older, feature-flagged Tkinter panel
(`library_sync_review.py`) that still exists in the legacy app. They are structurally
independent pieces of UI code, not a shared component — if you're debugging a
Library Sync issue, first confirm which of the two interfaces you're actually
looking at, since a fix to one won't be visible in the other.

*(Also worth knowing: the underlying engine Library Sync calls is a separately
maintained copy of the Indexer's scanning/fingerprinting logic, not the same module
the Indexer and Duplicate Finder use. See **architecture.md**'s "two indexer
engines" note for what that means in practice — most importantly, that the
fingerprint cache Library Sync uses has a different schema than the one the rest of
the app uses, and the two can drift apart. Fingerprint *matching* no longer drifts: the vendored copy decodes
Chromaprint fingerprints and compares raw sub-fingerprints, the same metric as the
Duplicate Finder (same recording in any codec ≈ 0.01, different recordings ≥ 0.43), and
Library Sync's cache keeps the decoded form so later runs skip the decode.)*

## Committing the changes, safely

When you approve the plan, Library Sync carries it out with the same caution as the
rest of the app:

- New tracks are **copied or moved** into the correct place in your library. A new
  track whose routed name is already taken by a *different* song is added as
  `name (1).ext` (the preview says so); only a byte-identical file there counts as
  "already in the library". Folder and file names keep their letter case on Windows,
  and an incoming `Music/` folder is not nested as `Music/Music/`.
- Every transfer is **reversible**. A file is written under a temporary name next to
  its destination and renamed into place only once complete, so a full disk or a
  crash never leaves a half-copied track under a real name. A move to another drive
  copies first and deletes the original only after everything else worked. If any
  step fails — for example a Replace while the library file is open in a player on
  Windows — the step is undone (the incoming file goes back where it was, the library
  file is untouched) and the audit says exactly what happened, including any file
  that could not be cleaned up.
- Any existing file that is being **replaced is backed up first** — the outgoing copy
  is preserved in a backups area inside `Docs` (`Docs/Backups/LibrarySync/<date_time>/`,
  one folder per run, so an earlier backup is never overwritten) before the new one
  takes its place — so an "upgrade" can always be undone. "Replaced" is only counted
  when a file really was displaced.
- **If the library changed after Build plan**, the items it affects are not done: a
  Replace whose library file was moved, edited or deleted, or a track whose incoming
  file changed, is skipped and reported ("scan again"), never turned into a plain
  copy. No other existing file is ever overwritten: if a destination appeared after
  the plan was built, that item is skipped and reported.
- Every run is recorded in its own **audit log** (`Docs/LibrarySyncAudit_<date_time>.json`,
  naming its backup folder) and **report** (`Docs/LibrarySyncExecuted_<date_time>.html`,
  a readable web page of what was added, replaced, skipped and what failed).
  `Docs/LibrarySyncAudit.json` and `Docs/LibrarySyncExecuted.html` are always a copy
  of the latest run.
- Until you give that approval, the execution code does not run; the preview is purely
  a description of intent.

This is the same preview-first, no-silent-loss, leave-a-recoverable-copy philosophy
that governs the Indexer and the Duplicate Finder — applied to the act of merging two
libraries.

## The export report

Library Sync can produce a standalone, shareable **report of the review** — a summary
of every match, its status, the distance and confidence, the quality verdict, and any
flags or notes you added, available as a web page, a JSON file, or a CSV file. An
**Export report** button on the Library Sync page's Review matches card lets you save
one as soon as a scan has finished; it opens a save dialog defaulting into the
library's `Docs/` folder and writes whichever format you choose based on the file
extension you pick.

The export routine states its own scope limit directly in its docstring, and it's
worth repeating because it's a good example of the project's "compute vs. write"
discipline: "The export routine deliberately avoids any move/replace logic and only
writes the requested report file after the caller has previewed counts." Exporting a
report can never itself move or replace a file — it's a read-only summary, full stop.

*(The HTML report used to render three "Replace All," "Replace with Better" and
"Cancel" buttons that did nothing. They are gone: a static file cannot safely drive
file operations. In their place the report says "To replace, flag tracks in AlphaDEX
→ Library Sync and build a plan" — replacing always goes through the normal
flag → Build plan → preview → execute flow.)*

---

*Related: **Playlists & Clustering** (next) covers the analytical side of AlphaDEX —
turning your now-tidy library into playlists, automatically grouped by how the music
actually sounds.*
