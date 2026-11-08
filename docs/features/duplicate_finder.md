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
- About **0.5** means they are unrelated: roughly half of the fingerprint's bits differ,
  which is what two different recordings look like.
- Values in between measure *how* different. A small number means "almost the same";
  a large number means "barely related."
- **1** is reserved for "nothing to compare": a missing fingerprint, or the empty one
  fpcalc returns for clips too short to fingerprint (about 3 seconds or less). Two
  empty fingerprints never count as a match.

Technically, the distance is the **bit error rate** between the two *decoded*
Chromaprint fingerprints (the best of a few small time alignments, so a slightly
different start does not matter). fpcalc and the cache store fingerprints in
Chromaprint's compressed text form; the finder decodes them before comparing. (It
used to compare the compressed text directly, which made a FLAC and its own MP3 look
0.1–0.45 apart — so most cross-format duplicates were missed.)

Measured on the app's own pipeline: the same recording in any codec or bitrate
(MP3 64–320 kbps, AAC, Vorbis, Opus) lands at **0.000–0.008**; the same recording
starting 0.3–0.6 s later lands at **0.03–0.08**; different recordings sit at **0.43+**.
`tests/test_dupes_fingerprint_distance.py` pins these ranges.

**Length counts, too.** fpcalc fingerprints the first two minutes, so a
30-second clip's fingerprint is simply the beginning of the full song's. The
Duplicate Finder therefore scores the part of the longer fingerprint that has no
counterpart (beyond the small alignment window) as unrelated audio: a clip is
*not* a duplicate of the song it was cut from, a copy missing its last tenth is
only a near duplicate (flagged for review), and copies a few fractions of a second
apart still score 0. (Before, a clip scored 0.0 against its song and the two were
grouped as exact duplicates — with the clip kept if it was the FLAC. The Similarity
page and Library Sync still compare only the overlapping part.)

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
*same* recording rarely have a distance of exactly zero, because the lossy version
literally threw away some of the sound when it was compressed. Comparing a FLAC to an
MP3 of the same song usually shows a little extra distance that has nothing to do with
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

- **Lossless always beats lossy.** A FLAC/WAV/AIFF/APE copy is kept over any MP3,
  AAC, Vorbis or Opus copy, whatever else differs.
- **Between lossless copies**, higher resolution (sample rate, then bit depth, then
  channels) wins; at equal resolution the format priority from the app's
  configuration decides — **FLAC over WAV** (FLAC carries tags and artwork properly).
  The bitrate is not used here: for lossless audio it only measures compression, so
  an uncompressed WAV used to "out-bitrate" the same audio as FLAC and win.
- **Between lossy copies**, the higher bitrate wins, then sample rate and the rest.
- **Readable cover art** only breaks a tie between copies of the same class (two
  FLACs, two MP3s). It used to come first, so an MP3 with a cover beat the FLAC
  without one and the lossless master was the copy set aside. OGG and Opus covers
  (Vorbis `METADATA_BLOCK_PICTURE` comments, and the older `COVERART`) count as
  readable: the shared reader used to miss them, so those files always looked
  coverless — even right after the executor had embedded one.
- Anything still tied is decided by the file path, deterministically — the same
  library always gives the same winner, whatever order the files were found in.

The reasoning is simple and consistent: when you have the same song twice, there is no
reason to keep the worse-sounding version, so the best copy survives and the
redundant ones step aside.

**A file and a link to it are not duplicates.** A symlink (or hard link) to a library
file fingerprints identically to it. The finder recognises paths that lead to the same
underlying file and considers only one of them, so it can never "keep" the link and
quarantine the real file behind it (the scan log names any alias it ignored).

## Quarantine, not delete

Here is the safety promise, applied to duplicates. When the finder decides a copy is a
redundant loser, its default action is **not to delete it** but to **move it into the
`Quarantine` folder** — a holding area, like a customs impound lot. The file is out of
your main library but not destroyed. If the finder ever gets a call wrong, your file is
still right there to recover.

Permanent deletion is possible, but it is an explicit, opt-in choice you have to make
deliberately — never the default, and never silent. In the Qt app you choose "Delete
losers permanently" (or "Delete permanently" for one group in the group inspector);
Execute then lists how many files will be deleted and asks you to **type DELETE** as a
second confirmation before the executor's deletion gate is opened. Each group can also
be set to "Keep loser in place (retain)". And whichever path you choose,
every action is **logged**: the finder writes detailed records into the `Docs` folder
describing exactly which files were grouped, which won, which lost, and where each one
went. Nothing happens without a paper trail.

Quarantined files keep their own name (`Quarantine/Song.mp3`); a name that is already
taken gets a copy number (`Song (1).mp3`). Names are compared the way Windows and
macOS compare them — `Song.mp3`/`song.mp3` and the two Unicode spellings of `Café`
count as the same name — so one loser can never replace another, or a file quarantined
by an earlier run. When `Quarantine/` is on another drive, the loser is copied there
first and only then removed; if the original cannot be removed (it is open in a player,
read-only, …) the copy is removed again and the loser simply stays in the library,
reported as not moved.

The one-pair-at-a-time tool, **Utilities → Pair Review**, follows the same rule. It
used to delete the inferior file with a single click; its action is now
"Quarantine inferior…", which names both files, asks, moves the loser into
`Quarantine/` under a name that never overwrites anything, points playlist entries at
the kept file, and logs the move to `Docs/duplicate_pair_review_log.txt`. It reviews
the pairs from the Duplicate Finder's last scan (`Docs/duplicate_preview.json`), or
else files with the same name in the same folder.

## Plan first, then act

Like every major workflow in AlphaDEX, the Duplicate Finder is split into a **dry run**
and an **execution**:

- The **dry run** scans the library, computes fingerprints, groups the duplicates,
  picks winners and losers, and assembles a complete plan — all without touching a
  single file. You get a readable report of every group and every proposed action.
- The **execution**, only after your approval, carries the plan out in a strict,
  intentionally conservative order — described next — and writes a final report of
  everything it did.

This split is why you never have to trust the finder blindly. You always see the full
plan before any file moves.

### The execution order, and why it's fixed

The executor's own design statement is worth quoting directly, because it explains
exactly what "intentionally conservative" means in practice:

> "Playlists are backed up before edits and validated after rewrites. A cancellation
> request stops processing after the current operation. Errors stop subsequent steps
> while preserving any backups already written. Every run produces machine-readable
> and human-readable reports."

Concretely, execution always happens in the same seven steps, in the same order:

```
1. Back up any playlist that will change
2. Rewrite those playlists to point at the surviving copy
3. Validate the rewrite (every entry it changed must point at an existing file)
4. Apply artwork transfers
5. Apply metadata normalization
6. Quarantine or delete the losers
7. Give a winner back the name its same-named loser just vacated
```

The ordering is not arbitrary: playlists are fixed and *checked* **before** any loser
is touched, specifically so a mid-run failure can never leave a playlist pointing at
a file that's already gone. If a playlist rewrite still references a file about to be
quarantined, the executor refuses to continue rather than risk a dangling reference.

Some details of each step, because they are what keep your files safe:

- **Playlists** (`.m3u` and `.m3u8` under `Playlists/`) are rewritten in their own
  encoding and line endings — a Windows-made cp1252 `.m3u` keeps every other entry
  byte for byte — and an absolute entry stays absolute. Entries that already pointed
  at missing files before the run are left as they were and do not stop it; an entry
  the playlist's encoding cannot spell stops the run before any loser moves.
- **Artwork and tag changes on the winner** (steps 4–5) are written to a copy beside
  it, and the copy replaces the winner only when the write succeeded, so a crash or a
  full disk can never leave a half-written winner. Cover art is embedded the way each
  format expects (FLAC pictures, Vorbis/Opus picture comments, ID3 in MP3/WAV/AIFF,
  MP4 atoms); formats without a known way are left untouched. Only the tag changes the
  preview listed are written, and no tag is ever removed.
- **Before each loser is touched** (step 6) the executor checks, at that moment, that
  its winner still exists, that the winner is not the very same file reached through a
  link, and that the loser is unchanged since the preview. If any check fails, that
  loser is left in place and reported; the other groups carry on.
- **Cancel and errors stop at a safe point.** Nothing new starts after you cancel or
  after a file operation fails; the few operations already running finish, and every
  operation that happened is in the report (with its destination), while the ones that
  never started are listed as "not attempted". The summary counts match what is on
  disk.
- **Step 7** only drops a copy number — `Song (1).flac` becomes `Song.flac` — when
  `Song.flac` was a loser of the same group that really left in this run; the preview
  shows the rename ("Then renamed to"). It never overwrites anything. (It used to strip
  any trailing number in brackets from every winner, unpreviewed — `Live (1986).flac`
  became `Live.flac`.)
- Plans whose groups overlap (one file a loser in two groups, or a loser that is
  another group's winner) are refused before anything moves.

Groups the planner is unsure about carry **review flags** (missing or unreadable
artwork, placeholder metadata, a near rather than exact match, …). The executor refuses
a plan with review flags unless the caller explicitly overrides. The Qt Duplicates
page shows which groups are flagged and why, and lets you either override ("execute
all groups exactly as previewed") or stop with nothing moved; refusals are reported
with their reason. A per-group accept/skip review stage is planned (see
**ROADMAP.md**). If you change a disposition after scanning, the preview
(`Docs/duplicate_preview.html`, plus `Docs/duplicate_preview.json` for Tools → Pair
Review) is rewritten before you confirm, so what executes is what the preview shows.

Deletion carries an extra, deliberate double gate — it only happens if the caller
both allows deletion in general *and* explicitly confirms this specific run, and the
executor separately checks that the plan being executed still matches the plan you
previewed (rejecting a run if the library visibly changed underneath it since the
preview was generated). This is the same "no silent data loss" discipline described
in **architecture.md**, applied concretely here.

### A few specific trade-offs worth knowing about

A handful of policy choices only show up as log messages inside the app, but they're
worth stating plainly since they explain behavior you might otherwise find puzzling:

- **Missing artwork never blocks a match.** If one copy has no embedded cover art,
  that alone doesn't stop it from being grouped and consolidated with its duplicates
  — a track is "kept in its artwork variant, but missing art does not block
  consolidation."
- **Different artwork on musically-identical tracks is treated as intentional, not
  as noise.** If two files are audio-identical but carry different cover art, the
  finder keeps them as **separate consolidation groups** rather than merging them and
  picking one image — the reasoning given is to preserve what might be a deliberate
  variant (a single vs. an album cut with different sleeve art) rather than erase it.
- **Artwork never migrates across different releases.** If a winner and loser belong
  to different multi-track releases (two different albums, say), the finder
  deliberately skips copying artwork between them "to avoid cross-album
  contamination" — it won't paste one album's cover onto tracks from another.

### A known, honest limitation: duration/tempo-shifted edits

The near-duplicate matching that groups album tracks does not currently normalize for
differences in duration or tempo before comparing fingerprints. The scan's own log
summary states this directly: gating decisions happen with "duration/tempo
normalization deferred." In practice, this means a significantly time-stretched or
trimmed edit of a track — a radio edit built by speeding up the original, for
instance — may not reliably match its full-length counterpart as a near-duplicate.
This is a known, named gap rather than an unnoticed bug; see **ROADMAP.md**.

### The Duplicate Pair Report

Alongside the main scan, there's a focused diagnostic that inspects **one specific
pair** of files and shows you, step by step, exactly why they were or weren't
matched: which metadata bucket each landed in, whether they shared any coarse
fingerprint keys, the raw distance between them, and which threshold applied. It's
the tool to reach for when the finder's verdict on a particular pair surprises you.

*(Historical note: this tool used to crash on every invocation — it looked up each
track's metadata bucket using a dictionary operation against what is actually a
list, so it raised an error before producing anything. That's fixed; both the
on-screen report and its HTML export work now.)*

## Before the first scan: fpcalc and FFmpeg

Fingerprints are computed by **fpcalc** (Chromaprint) after **FFmpeg** decodes the
audio. If either is missing from `PATH`, the Duplicates page says so once, with
install links, and offers to scan using only fingerprints already in the cache (files
never fingerprinted are skipped and counted in one log line).

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

Fingerprints of short tracks made by older versions with **silence padding** (tracks
shorter than the fingerprint window used to be padded to its length, so different
short tracks matched each other) are not reused: a cached fingerprint from before the
fix that ends in a run of padding silence is recomputed once; every other cached
fingerprint stays valid, so the rest of the library is not re-analysed. (The Duplicates
page's own fpcalc fingerprints were never padded — the padding came from the
fingerprint generator Library Sync uses — but the check costs microseconds.)

The cache is only ever a shortcut. If its file is damaged ("file is not a database"),
it is renamed to `….corrupt-<date>` — not deleted — and a fresh cache is started; a
row the cache cannot read is simply a cache miss. (A damaged cache used to make every
scan fail.)

Every scanner of the library — this one, the legacy finder and the two experimental
engines — skips the reserved top-level folders (`Quarantine/`, `Trash/`, `Docs/`,
`Playlists/`, `Not Sorted/`, `Manual Review/`), so quarantined losers never come back
as "duplicates" of the copies they were consolidated into.

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

*Related: **Library Sync** (next) uses fingerprint-and-distance machinery for a
different purpose — merging a new folder into your library rather than cleaning
duplicates within it. Note that it uses the vendored engine copy under
`library_sync_indexer_engine/`, which still compares compressed fingerprints; the
decoding described above has not been ported there.*
