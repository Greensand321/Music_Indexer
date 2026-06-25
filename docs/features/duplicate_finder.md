# Feature — The Duplicate Finder

*The Duplicate Finder identifies copies of the same song scattered through your
library — even when their names, tags, and formats differ — and helps you keep the
best copy while safely setting the rest aside. This document explains the ideas
behind it without code or jargon.*

---

## What "duplicate" really means here

The naive idea of a duplicate is "two files with the same name." That idea is almost
useless for music, because the same song routinely lives under a dozen different
names: `Halo.mp3`, `03 - Halo.flac`, `Beyonce - Halo (Album Version).m4a`. They look
like three unrelated files to your computer, but they are the same recording.

The Duplicate Finder ignores names entirely and asks a deeper question: **do these two
files actually sound the same?** Its definition of a duplicate is *content-based* —
two files are duplicates if the audio inside them is the same (or nearly the same)
recording, regardless of what they're called, what tags they carry, or what format
they're stored in. The analogy is comparing two photographs by looking at the picture,
not by reading the file names printed on the back.

## How it listens: fingerprints

To compare sound rather than names, the Duplicate Finder reduces each file to a
**fingerprint** — the short numeric summary of its actual audio introduced in the
overview. Two fingerprints can be compared to produce a single **distance number**:

- **0** means the two fingerprints are identical — the same audio.
- **1** means they are completely different — nothing in common.
- Values in between measure *how* different. A small number means "almost the same";
  a large number means "barely related."

This is the engine's entire notion of similarity. Everything else — what counts as a
duplicate, which copy wins — is built on top of this one distance number.

## How similar is "the same"? Thresholds

A distance of exactly zero is rare in the real world, because re-encoding, ripping,
and editing introduce tiny variations. So the Duplicate Finder uses **thresholds** —
cut-off lines that say "closer than this counts as a match." It distinguishes two
kinds of match:

- An **exact duplicate** is a very tight match (distance at or below roughly **0.02**)
  — for all practical purposes the identical recording, just a different file.
- A **near duplicate** is a looser match (distance at or below roughly **0.10**) — the
  same song but in a meaningfully different form: a different rip, a remaster, a
  slightly different edit, or simply a different encoding.

You can tune these lines. Tightening them makes the finder stricter (fewer matches,
fewer false alarms, but it may miss genuine duplicates that drifted apart). Loosening
them makes it more aggressive (it catches more, at the risk of pairing songs that
merely sound alike). The defaults are chosen to be confident without being reckless.

### The mixed-codec allowance

There is one subtle but important wrinkle. A lossless file and a lossy file of the
*same* recording will **never** have a distance of zero, because the lossy version
literally threw away some of the sound when it was compressed. Comparing a FLAC to an
MP3 of the same song always shows a little extra distance that has nothing to do with
them being different songs — it's just the compression talking.

To stop this from causing the finder to miss obvious duplicates, it grants a small
**allowance** (about **0.03** of extra slack) whenever it compares a lossless file
against a lossy one. In effect it says: "these two are in different formats, so I
expect a bit more distance between them — I won't hold that against them." This keeps
cross-format duplicates from slipping through the cracks.

## Which copy wins? Format priority

Once a group of duplicates is identified, the finder must pick a **winner** — the copy
to keep — and treat the rest as **losers** to be set aside. It decides primarily on
**quality**, using the same lossless-beats-lossy logic that runs through the whole
app:

- A **FLAC** (lossless) is preferred over a **WAV** is preferred over an **MP3**, and
  so on down the line. The higher-quality copy is the one worth keeping.
- When two copies are of equal quality (say, two FLACs of the same track), the finder
  falls back to a sensible tie-breaker, favoring the file whose name carries more
  descriptive information.

The reasoning is simple and consistent: when you have the same song twice, there is no
reason to keep the worse-sounding version, so the best copy survives and the
redundant ones step aside.

## Quarantine, not delete

Here is the safety promise, applied to duplicates. When the finder decides a copy is a
redundant loser, its default action is **not to delete it** but to **move it into the
`Quarantine` folder** — a holding area, like a customs impound lot. The file is out of
your main library but not destroyed. If the finder ever gets a call wrong, your file is
still right there to recover.

Permanent deletion is possible, but it is an explicit, opt-in choice you have to make
deliberately — never the default, and never silent. And whichever path you choose,
every action is **logged**: the finder writes detailed records into the `Docs` folder
describing exactly which files were grouped, which won, which lost, and where each one
went. Nothing happens without a paper trail.

## Plan first, then act

Like every major workflow in AlphaDEX, the Duplicate Finder is split into a **dry run**
and an **execution**:

- The **dry run** scans the library, computes fingerprints, groups the duplicates,
  picks winners and losers, and assembles a complete plan — all without touching a
  single file. You get a readable report of every group and every proposed action.
- The **execution**, only after your approval, carries the plan out: it updates your
  playlists so they point at the surviving copy instead of the removed one, carries
  over useful extras like embedded album art from a loser to the winner, moves the
  losers into quarantine (or deletes them, if you explicitly asked), and writes a
  final report of everything it did.

This split is why you never have to trust the finder blindly. You always see the full
plan before any file moves.

## Why the second scan is so much faster: the cache

The first time you scan a large library, the Duplicate Finder has to compute a
fingerprint for every file, which takes real time. To avoid paying that cost over and
over, it **remembers** every fingerprint it computes in a small local cache (described
in **architecture.md**).

On later scans it reuses those remembered fingerprints, recomputing only for files
that are **new or have changed** since last time. It detects change by watching each
file's size and last-modified date — if those shift, the old fingerprint is treated as
stale and refreshed. The practical result is dramatic: a first scan of ten thousand
tracks might take many minutes, while a follow-up scan of the same library finishes in
seconds, because almost nothing needs to be re-analyzed.

## Tuning, in plain terms

If you find the finder too cautious or too eager, the levers available to you are:

| If you want to… | …then |
|---|---|
| Catch only truly identical files | Tighten the exact-duplicate threshold. |
| Also catch remasters and alternate rips | Loosen the near-duplicate threshold. |
| Be more forgiving across formats (FLAC vs MP3) | Increase the mixed-codec allowance. |
| Reduce false alarms | Tighten thresholds; review groups before executing. |

In every case, the safest habit is the one the app encourages anyway: **read the dry-
run plan before you approve it.** Your own eyes are the final threshold.

---

*Related: **Library Sync** (next) uses the same fingerprint-and-distance machinery,
but for a different purpose — merging a new folder into your library rather than
cleaning duplicates within it.*
