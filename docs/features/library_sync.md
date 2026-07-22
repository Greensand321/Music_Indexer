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
   the incoming folder, and set how strict the matching should be.
2. **Scan and fingerprint.** Both collections are read and fingerprinted (reusing the
   cache wherever possible, so repeat runs are fast). Each track becomes a tidy record
   carrying its fingerprint and its quality details.
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

Two refinements make this trustworthy. First, the strictness is **adjustable**, with a
default sensitivity that is a little more forgiving than the Duplicate Finder's because
cross-collection matches naturally vary more; lossy formats like MP3 are given a touch
more leeway than lossless ones, since compression blurs their fingerprints. Second,
borderline cases are deliberately surfaced as "low confidence" rather than silently
decided, because fingerprint distances are noisiest exactly at the boundary — and a
**match confidence** score (a friendly 0-to-100% reading derived from how far inside or
outside the threshold a pairing falls) is shown so you can see not just *whether* two
tracks matched but *how sure* the app is.

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

A few practical notes:

- **Flags are remembered only for the current session.** They live in memory while you
  work, not in a saved file. This is a deliberate simplicity choice (see below).
- **Re-scanning clears your flags.** If you change the strictness and recompute the
  matches, the whole landscape of matches can shift — a track you flagged for
  "Replace" might now match something different, or nothing at all. Rather than carry
  forward flags that may no longer make sense, the app clears them and lets you start
  fresh against the new results, warning you where a previous flag could not be
  carried over.
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
the same backend engine described above: the modern Qt workspace
(`gui/workspaces/library_sync.py`, described throughout this document, reached from
the sidebar) and an older, feature-flagged Tkinter panel
(`library_sync_review.py`) that still exists in the legacy app. They are structurally
independent pieces of UI code, not a shared component — if you're debugging a
Library Sync issue, first confirm which of the two interfaces you're actually
looking at, since a fix to one won't be visible in the other.

*(Also worth knowing: the underlying engine Library Sync calls is a separately
maintained copy of the Indexer's scanning/fingerprinting logic, not the same module
the Indexer and Duplicate Finder use. See **architecture.md**'s "two indexer
engines" note for what that means in practice — most importantly, that the
fingerprint cache Library Sync uses has a different schema than the one the rest of
the app uses, and the two can drift apart.)*

## Committing the changes, safely

When you approve the plan, Library Sync carries it out with the same caution as the
rest of the app:

- New tracks are **copied or moved** into the correct place in your library.
- Any existing file that is being **replaced is backed up first** — the outgoing copy
  is preserved in a backups area inside `Docs` before the new one takes its place — so
  an "upgrade" can always be undone.
- Every action is recorded in a detailed **audit log**, and a final **report** (a
  readable web page) summarizes what was added, what was replaced, what was skipped,
  and anything that failed.
- Until you give that approval, the execution code does not run; the preview is purely
  a description of intent.

This is the same preview-first, no-silent-loss, leave-a-recoverable-copy philosophy
that governs the Indexer and the Duplicate Finder — applied to the act of merging two
libraries.

## The Export Report

Library Sync can produce a standalone, shareable **report of the review** — a summary
of every match, its status, the distance and confidence, the quality verdict, and any
flags or notes you added, available as a web page, a JSON file, or a CSV file. An
**Export Report** button in the Qt workspace's Plan & Execution panel lets you save
one as soon as a scan has finished; it opens a save dialog defaulting into the
library's `Docs/` folder and writes whichever format you choose based on the file
extension you pick.

The export routine states its own scope limit directly in its docstring, and it's
worth repeating because it's a good example of the project's "compute vs. write"
discipline: "The export routine deliberately avoids any move/replace logic and only
writes the requested report file after the caller has previewed counts." Exporting a
report can never itself move or replace a file — it's a read-only summary, full stop.

*(One inert detail worth knowing if you ever open the HTML report format directly:
it renders three action buttons — "Replace All," "Replace with Better," "Cancel" —
styled to look clickable. They currently do nothing; no script wires them to
anything, and the backend function they'd need to call exists but isn't connected to
any live control in either app. Treat the HTML report as a read-only summary, not an
in-report control surface.)*

---

*Related: **Playlists & Clustering** (next) covers the analytical side of AlphaDEX —
turning your now-tidy library into playlists, automatically grouped by how the music
actually sounds.*
