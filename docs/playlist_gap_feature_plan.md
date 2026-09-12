# Feature Plan — Playlist Gap (“What am I missing?”)

*Status: **concept plan**, not a spec. This document exists to agree on the problem,
the workflow, and the tools before anyone writes a technical breakdown. Nothing here
is committed implementation detail; the follow-on spec turns the agreed parts of this
into precise behaviour.*

*Written: 2026-09-12. Target app: the **Qt** app (`alpha_dex_gui.py`). Not the legacy
Tkinter app.*

---

## Revision note — what a real export actually looks like

*Added 2026-09-12, after inspecting an actual TuneMyMusic export of a YouTube Music
"Liked videos" playlist. This section supersedes the optimistic assumptions in §7 and §9
below; those sections are patched to match, and the changes are called out inline.*

The export has the right *columns* and almost none of the *data*:

| Column | Reality |
|---|---|
| `Track name` | **The entire YouTube video title, as one free-text blob.** |
| `Artist name` | **Empty.** |
| `Album` | **Empty.** |
| `ISRC` | **Empty.** |
| `Playlist name` | `Liked videos` |
| `Type` | `Favorite` |
| *(duration)* | **No column at all.** |

Eight real rows, and what each one costs us:

| Raw `Track name` | What it actually is | Difficulty |
|---|---|---|
| *(blank)* | A deleted or private video | Must be **reported**, never silently dropped |
| `Self Aware x Babydoll` | A **mashup**. No artist anywhere. `x` means "mashed with", not "featuring" | Severe — and see the parser bug below |
| `FIFTY FIFTY - Cupid (Twin Version)` | `Artist - Title (Version)`; "Twin Version" is a distinct official recording | Easy parse, Class B modifier |
| `French 79 · New Constellations - Colors Collide` | Two different delimiters (`·` and `-`); unclear which marks the artist boundary | Hard |
| `(Triple Vibe) HOME - Resonance but it's beats 3,3` | Uploader prefix, then artist, then title, then a **fan-edit descriptor** | Severe |
| `Wavebeatmaker - Resonance` | A *different* "Resonance" than the row above | Easy parse — but proves title-only matching is dangerous |
| `The Sways - Someday We Will Dream About Today` | `Artist - Title` | Easy |
| `Kate Bush - Running Up That Hill (A Deal With God)` | The parenthetical is **part of the canonical title**, not a modifier | Easy parse — and a trap for naive paren-stripping |

### What this invalidates

- **Rung 0 (ISRC) is dead** for this source. The column exists and is empty.
- **Rung 1 (exact artist + title) is dead.** There is no artist field to compare.
- **Duration corroboration (§9) is dead** for this source. No duration column — which
  removes the signal the plan was relying on to shrink the confirm bucket.
- **Artist-blocked matching is dead**, and that was what made the ladder fast. Without an
  artist there is nothing to block on, so rung 4 collapses into rung 5's full sweep.

So the plan's floor is much lower than assumed: **one free-text string per row.** That
demands a pipeline stage the plan didn't have — see "Stage 2.5" below.

### Two verified bugs this surfaced

1. **`extract_primary_and_collabs()` would actively mis-parse mashups.** Its separator
   list is `[" feat.", " ft.", " & ", " x ", ", ", ";"]` — so `Self Aware x Babydoll`
   parses as primary artist "Self Aware" featuring "Babydoll". It has a second trap too:
   a fallback that splits on a lowercase→uppercase boundary, which is aggressive on
   video titles. Do not reuse it unmodified on display strings.
2. **The metadata reader cannot see a YouTube identity even when the file carries one.**
   `TAG_KEYS` in `utils/audio_metadata_reader.py` is
   `artist, albumartist, title, album, date, year, track, tracknumber, disc, discnumber,
   genre, compilation` — no `comment`, no `purl`, no `website`. `yt-dlp --embed-metadata`
   writes the source URL into exactly those tags. Three added keys would unlock the
   identity strategy below.

### The reframe: this is a backfill problem, not a permanent matching problem

The plan treated fuzzy matching as the feature's permanent core. It shouldn't be. Every
good source (below) returns a **YouTube video ID** — a stable, exact identity key. So
split the work in two:

- **Backfill (hard, one-time).** Existing files have to be matched the hard way — string
  parsing, fuzzy comparison, human confirmation. This is genuinely difficult and will
  never be perfect.
- **Going forward (easy, permanent).** Capture the video ID at download time, and every
  future comparison is an exact key lookup. **The problem stops growing.**

Framed that way, the fuzzy work is a bounded cleanup with an end date, not the feature's
steady state. The single highest-leverage thing the spec can do is make sure that from
day one, anything newly downloaded is never ambiguous again.

### The second pass — verify what actually arrived

*Added after the mockup round. This was missing from the first draft and it is not a nicety;
it closes the loop that causes the original problem.*

Exporting the missing list is **not** the end. The downloads then have to be checked, because
**downloaders pick the wrong recording surprisingly often** — and the way that fails is nasty:

> You ask for `FIFTY FIFTY - Cupid (Twin Version)`. The downloader hands you `Cupid` — the plain
> version, which you already own. Now you have **a duplicate of Cupid**, you **still don't have
> the Twin Version**, and the ledger has marked the row satisfied so **it will never ask again.**

That is a false MATCHED arriving through the back door, and it is exactly the silent-loss error
class §3 rules out. Worse, it *corrupts the ledger* — the one mechanism the whole feature relies
on to make later runs cheap. Without a verification pass, the ledger's accuracy decays every
month, and the feature slowly becomes the problem it was built to solve.

So the pipeline has a second pass:

```
  pass 1  →  find & sort  →  the download list  →  [ the user downloads ]
                                                            ↓
  pass 2  →  verify what arrived  →  correct the ledger  →  re-list what's still missing
```

**What it checks.** Point it at the folder the downloads landed in. For each new file, answer one
question: *is this the recording that was asked for?* Two mechanisms, cheapest first:

1. **Identity check.** If the file kept its source `[videoId]` (filename or a `comment`/`purl`
   tag), compare it directly to the video ID on the wanted row. Exact, instant, no audio decoded.
   For anything downloaded with `yt-dlp` this settles most of the folder.
2. **Fingerprint check.** For the rest, fingerprint the file and compare it against both the
   wanted row's expectation *and* the existing library. Duration corroborates: the Twin Version
   is 41 s from the plain version, so the mismatch is unmistakable.

**Five outcomes, each with a different consequence:**

| Outcome | What it means | What the app does |
|---|---|---|
| ✅ **Correct** | Fingerprint/ID matches what was asked for | Mark the ledger row **satisfied**. Done. |
| ⚠️ **Wrong version** | A different recording than requested — often one already owned | **Two actions, and both matter:** quarantine the duplicate, *and* put the wanted row **back on the missing list**. Never mark it satisfied. |
| ⚠️ **Already owned** | A straight duplicate of an existing file | Hand to the Duplicate Finder; leave the wanted row satisfied only if it was genuinely the right track. |
| ❌ **Didn't arrive** | Exported, nothing showed up | Row stays **missing** and stays visible; don't let it quietly age out. |
| ❓ **Unrecognized** | Fingerprint matches nothing known | Almost certainly a genuine new track. Accept and index it. |

**It is mostly composition, not new engine work.** The pieces already exist and are already
tested:

- `near_duplicate_detector.fingerprint_distance(fp1, fp2)` — a public pairwise fingerprint
  comparison, which is precisely the primitive this needs.
- `fingerprint_generator.compute_fingerprint_for_file()` — fingerprints a single file.
- `fingerprint_cache` — the library side is already fingerprinted and cached, so only the newly
  downloaded files need work.
- `duplicate_consolidation` / its executor — for quarantining whatever the pass finds is a
  duplicate, using the project's existing preview-then-execute contract.
- Library Sync — for actually merging the verified-correct files into the library.

**This supersedes §14's Option D**, which treated the post-download check as workflow guidance
("point the user at Library Sync") rather than a step. That was wrong. It is a first-class stage
with its own screen, because the *wrong-version* outcome needs a decision the user can only make
here, and because nothing else writes the correction back into the ledger.

Two notes for the spec:

- **Only the wrong-version outcome is interesting.** The other four are counts. Design the screen
  so the handful of wrong-version rows are unmissable and the rest collapse into a summary.
- **Verification is re-runnable and incremental.** Downloads trickle in over days; the pass should
  be safe to run repeatedly against the same folder, skipping files it has already judged.

### Saved sources — templates for one-press re-runs

Also missing from the first draft, and cheap to build.

A **saved source** (template / preset) stores everything a run needs:

- source type and URL (`https://music.youtube.com/playlist?list=…`), plus a friendly name,
- the library folder and its inclusion flags,
- the duration tolerances and fuzzy floor in force,
- a pointer to that source's ledger.

With those saved, the routine monthly use collapses to: open the workspace → the saved sources are
listed with *last run* and *new since then* → press ▶ on one → read the download list. One press,
no re-typing a URL, no re-choosing folders, no re-answering settled questions.

Why it matters more than it sounds: the whole value proposition is that **run two is cheaper than
run one**. The ledger makes the *thinking* cheaper; templates make the *setup* cheaper. Without
them, every run starts with a URL paste and a folder pick, which is exactly the friction that
stops a tool from being used monthly.

Worth specifying alongside it:

- **Run all** — compare every saved source in one go, with a combined download list. If four
  playlists share 30 wanted tracks, you want to download each one once.
- **Per-source ledgers, one shared library snapshot.** The snapshot is expensive and identical
  across sources; build it once per session, not once per playlist.
- Templates are plain config, so they belong in `config.load_config()` / `save_config()` per the
  project's config rule — not a bespoke file.

### Stage 2.5 — parse the display string into candidate artist/title splits

A new pipeline stage between "import" and "match", and now the highest-risk component.

The governing principle: **don't commit to a parse.** Getting `French 79 · New
Constellations - Colors Collide` right up front is guesswork. Instead, generate *several
candidate* `(artist, title)` splits per row — split at each delimiter, both orderings,
with and without a leading uploader prefix, with and without trailing descriptors — and
**probe the library with all of them.** The library is the oracle: if exactly one
candidate hits, that's both the match and the confirmation that the parse was right. If
several hit, it's a confirm row. If none hit, it's missing.

This inverts the usual order (parse, then look up) into (enumerate, then let lookups
choose) and is far more robust on strings this messy. It also composes with the ledger —
a confirmed parse is remembered, so the same blob never gets re-parsed.

Practical notes for the spec:

- Delimiters seen in real data: ` - `, ` – `, ` — `, ` · `, ` | `, ` _ `, `: `. Treat
  `x` as a delimiter **only** as a mashup marker, never as an artist separator.
- Uploader/label prefixes in leading parens or brackets — `(Triple Vibe)`, `[NCS
  Release]`, `[Free Download]` — should be lifted off as their own token, not folded into
  the artist.
- **Parenthetical handling is position- and content-dependent.** `(A Deal With God)` is
  canonical title; `(Twin Version)` is a Class B modifier; `(Triple Vibe)` is an uploader
  tag. "Strip all parens" is wrong three different ways. Classify by content against the
  lexicon, not by bracket type.
- **Fan-edit descriptors are a new Class B family, and they are common in this library:**
  `but it's …`, `sped up`, `slowed + reverb`, `nightcore`, `bass boosted`, `8D`, `loop`,
  `mashup`, `x` (as mashup), `AMV`, `edit`. This moves Class B work from "2% of a rock
  library" to first-priority.

### The filename inversion

For a YouTube-sourced library, **match against library filenames first and tags second** —
the reverse of what §14's Option G proposed.

Reasoning: files downloaded from YouTube typically keep the video title as the filename
(yt-dlp's default template is `%(title)s [%(id)s].%(ext)s`), while their *tags* are
usually empty or junk — which is exactly why such files end up in `Manual Review/`. So
comparing the CSV's `Track name` to the library **filename** is close to exact string
matching, not fuzzy semantics, and it sidesteps the parsing problem entirely for any file
that still carries its original name.

Better still: if the filename retained yt-dlp's `[videoId]` suffix, or the file carries
the source URL in a `comment`/`purl` tag, that is **rung 0 restored** — exact identity,
no fuzz at all.

> **Do this before writing the spec:** look at the actual files already in the library.
> Are they named like video titles? Do any carry a `[dQw4w9WgXcQ]`-style suffix, or a
> YouTube URL in a comment tag? If yes, most of this feature's difficulty evaporates and
> the spec should be built around exact identity matching with fuzzy work as the
> exception. If no, the fuzzy path is the main path. **This one check changes the
> architecture**, and it costs five minutes.

### Better sources — ranked

The user's instinct is correct and important: **transferring the playlist to another
service to get a cleaner CSV is the worst option, not the best one.** Services like
TuneMyMusic produce clean artist/album fields by *matching your videos against a
catalogue*, and anything that doesn't match — mashups, fan edits, unofficial uploads,
obscure tracks, which is a large share of this library — is silently dropped. A dropped
row never reaches the wanted list, is never reported missing, and is never downloaded.
That is precisely the **silent-loss error class §3 rules out**, except worse: it happens
before the app ever sees the data, so no amount of careful matching downstream can
recover it.

The better sources are not websites. They read YouTube directly:

1. **`ytmusicapi`** *(recommended ceiling).* A Python library, so it fits the app
   natively. `get_playlist()` returns per-track `videoId`, `title`, `artists[]` (names
   **and** channel IDs), `album{name, id}`, `duration`, `isExplicit`, and
   `isAvailable` — which is exactly the structured data the CSV is missing, straight from
   YouTube Music with no transfer and no drops. `isAvailable` even handles the blank
   deleted-video row properly. Cost: one-time browser-header authentication.
2. **`yt-dlp --dump-json`** *(recommended floor, and probably already installed).* No
   authentication at all. `--flat-playlist` is fast but sparse (id, title, uploader,
   often duration); full extraction is slower but can surface real `track` / `artist` /
   `album` fields for videos that carry music metadata. Since yt-dlp is likely already in
   the download toolchain, this is the cheapest real upgrade — and it's the same tool that
   can be configured to preserve the video ID on download, which is the permanent fix.
3. **Google Takeout** *(no-code option, clunky).* `music-library-songs.csv` carries Video
   ID, Song Title, Album Title, and Artist Name; the per-playlist CSVs are essentially
   video IDs. Useful as a one-time backfill of identities, awkward as a routine step.
4. **TuneMyMusic CSV** *(what exists today).* Keep it supported — it is the universal
   on-ramp and the only option that needs no setup. Just stop treating it as the accuracy
   baseline; it is the accuracy **floor**.
5. **Transfer to Spotify, then export.** **Do not.** See above.

### The architectural consequence

**The pluggable "wanted-list source" moves from §14 Option B's Phase 5 wishlist into
Phase 1 core.** It is no longer an extensibility nicety — it is the thing that decides
the feature's accuracy ceiling.

Concretely: a source declares which fields it can provide (`display_string`, `title`,
`artist`, `album`, `duration`, `isrc`, `video_id`, `availability`), and the match ladder
**skips the rungs its inputs can't support** rather than assuming a schema. Same triage
UI, same ledger, same reports, regardless of source.

Design for the floor — one free-text string — so the feature works today with the CSV
already in hand. Let better sources *upgrade* accuracy rather than being required for it.
Then adding `ytmusicapi` later is a strict improvement to every existing playlist, not a
rewrite.

### Honest expectation setting

A meaningful share of this particular library — the mashups, the fan edits, the
`but it's beats 3,3` remixes — will **never** match confidently from a title string
alone, and shouldn't. Those rows belong in NEEDS CONFIRMATION, and the confirm bucket on
a first run will be larger than the "few percent" §1 hoped for. That is the correct
outcome, not a failure: per §3, an unmatched mashup shown to the user costs a glance,
while a wrongly-matched one costs the song. The bulk pattern-triage in §11 and the ledger
in §10 are what keep that bucket from being re-litigated every month — which makes them
more important now, not less.

---

## 1. The problem

The library is thousands of tracks deep, and it grows by **re-downloading playlists**.
The cycle looks like this:

1. Export / download a playlist's worth of music.
2. Merge it into the library.
3. Two or three months later the playlist has grown by a handful of songs.
4. Download the playlist again — because there's no way to tell which handful is new.
5. Discover, after the fact, that most of what just arrived is a duplicate of what
   was already there.

Step 4 is the whole problem. **The information needed to avoid it — "which of these
tracks do I not already own?" — exists, but nothing in the app can answer it.**

Note carefully what the problem is *not*:

- It is **not** "find duplicates in my library." That's the Duplicate Finder, and it
  runs *after* the damage — after bandwidth, time, and a messy merge.
- It is **not** "reconcile two folders of audio." That's Library Sync, and it needs
  audio files on both sides. Here, one side is a **text list**: there is no audio to
  compare, because the whole point is that the files haven't been downloaded yet.

The problem is **acquisition triage**: given a list of songs I *want*, and a library of
songs I *have*, produce the short list of songs I should go download. The user
downloads them by hand; the app's job is to make that list short and trustworthy.

## 2. What "done" looks like

The feature works if, in practice:

- Re-running a playlist that was already fully processed yields a MISSING list of
  roughly **zero** — no re-downloading what's already owned.
- Running a playlist that grew by 12 songs yields a MISSING list of roughly **those 12**.
- The ambiguous middle bucket is small enough to clear in **one sitting** — a few
  percent of rows, not a third of them.
- The second and third runs of the same playlist are **cheaper than the first**,
  because the app remembers the judgement calls already made.

That last point is the difference between a tool that gets used and a demo that gets
run once. See §10.

## 3. The one asymmetry that drives every design decision

The two ways this feature can be wrong are **not** equally bad.

| Mistake | What happens | How bad |
|---|---|---|
| **False MISSING** — says "you don't have it" when you do | A duplicate gets downloaded | Annoying. Wastes a download. **Recoverable** — the Duplicate Finder already exists to clean it up, and it's *visible*: the file lands and you see it. |
| **False MATCHED** — says "you already have it" when you don't | The song is never downloaded, and **never mentioned again** | Serious. A song silently vanishes from the wishlist. **Unrecoverable** without redoing the whole comparison, because nothing ever draws attention to it. |

A false MATCHED is a *silent* failure, and the library already has a policy for silent
failures: don't have them. (`CLAUDE.md`: *"No silent data loss"*, *"When in doubt,
default to quarantine, not delete."*)

**So: when the system is unsure, it must never resolve toward MATCHED.** Uncertainty
goes to NEEDS CONFIRMATION. If a row must be auto-resolved without asking, it resolves
toward MISSING. This single rule decides most of the threshold questions the spec will
otherwise agonise over.

It also explains why a three-bucket design is right and a confidence score alone is
wrong: a score forces one threshold to serve two errors with wildly different costs.
Two thresholds and a middle bucket let each error be priced separately.

## 4. Where this fits in AlphaDEX

**A new Navigator workspace**, alongside Indexer / Library Sync / Duplicates. Working
name: **Playlist Gap**. (Alternatives: "Wishlist", "Acquisition List", "What's
Missing". Pick one before the spec — the name appears in the nav rail, the report
titles, and the config keys.)

It is a **read-only, preview-only** workspace. It never copies, moves, deletes, or
retags anything. That makes it the easiest feature in the app to trust, and it sits
perfectly inside the project's strongest architectural rule (preview-first, never
destructive by default) without needing an execution stage at all. The "execution"
step is the user opening a downloader.

### Why not just add a tab to Library Sync?

Considered, and it has real merit: Library Sync already owns the vocabulary
(incoming vs. existing, match statuses, per-item review flags, an export report) and
users would find the two workflows next to each other.

**Recommendation: separate workspace anyway**, for three reasons:

1. Library Sync's matching engine is **fingerprint-first** — it compares audio to
   audio. This feature has no audio on the incoming side. Sharing a screen would imply
   a shared engine that can't exist, and would set wrong expectations about accuracy.
2. Library Sync's workspace is already dense (two track tables, a match inspector, a
   plan panel, an execution log). Adding a fundamentally different input mode makes it
   worse.
3. This is a **recurring, standalone habit** ("what should I download this month?"),
   not a sub-mode of merging a folder. It deserves its own front door.

What *should* be shared is the **review vocabulary and the report plumbing**, not the
screen. See §5.

## 5. What already exists that this can stand on

This is less new machinery than it first appears.

**Reusable as-is or near-as-is:**

- **The library-side index already exists.** The fingerprint cache
  (`fingerprint_cache.py`) is a SQLite table keyed by path that already stores
  `tags_json`, `duration`, `ext`, `bitrate`, **and** dedicated
  `normalized_artist` / `normalized_title` / `normalized_album` columns. That is
  exactly the lookup table this feature needs, already built and already warmed by
  normal use. ⚠️ **But see the caveat below.**
- **The three-bucket review pattern.** `library_sync_review_state.py`'s
  `ReviewStateStore` is a small, clean store of per-item user decisions that survives
  re-sorting and re-filtering of the underlying results, and knows how to re-bind
  decisions when the underlying best match changes. The gap triage needs the same
  thing with different verbs (`accept as owned` / `send to missing` / `note`).
- **The report exporter.** `library_sync_review_report.py` already writes HTML, JSON,
  and CSV from a set of match results, with a summary block. The shopping list is the
  same shape of output.
- **Artist-credit splitting.** `music_indexer_api.py` already splits multi-artist
  strings on `feat.`, `ft.`, `&`, `x`, `,`, `;` and picks a canonical primary artist.
  That is half of the modifier problem (§8) already solved and battle-tested against
  this library's real tags.
- **Remix/version awareness.** `near_duplicate_detector.py` carries
  `EXCLUSION_KEYWORDS = ['remix', 'remastered', 'edit', 'version']` — a seed for the
  modifier lexicon.
- **A working fuzzy-name matcher as prior art.** `playlist_generator._pick_best_match`
  does token-overlap plus `difflib.SequenceMatcher`, with a candidate shortlist built
  from a token index, and a "too weak, return nothing" floor. The approach transfers
  directly; the difference is it matches *filenames*, and this feature should match
  *tags* (with filenames as a fallback).
- **No new dependency required.** `difflib` is stdlib and is already this project's
  similarity tool. (`rapidfuzz` would be faster and better at this — see §14, Option F.)

**Two real prerequisites, not assumptions:**

1. ⚠️ **The normalized columns are only populated by the legacy app.** `main_gui.py`
   computes `normalized_artist/title/album` and passes them to `store_fingerprint`.
   The Qt Duplicates workspace calls `store_fingerprint(...)` **without** them — so in
   the active app those columns are `NULL`. The feature therefore needs either its own
   library-index build step, or a fix so the Qt path fills them in. **This is a
   prerequisite, and the spec must state which route it takes.**
2. ⚠️ **Reserved folders must be treated as "owned".** The Indexer and Duplicate
   Finder deliberately *skip* `Not Sorted/`, `Quarantine/`, `Manual Review/`, `Trash/`,
   `Playlists/`, `Docs/`. For *this* feature, several of those are emphatically
   **things you already have**:
   - `Not Sorted/` — downloaded but unfiled. **Counts as owned.** Skipping it would
     cause exactly the false-MISSING re-download the feature is meant to prevent.
   - `Quarantine/` — a duplicate loser awaiting review. **Counts as owned.**
   - `Manual Review/` — missing required metadata. **Counts as owned**, but it can't
     be matched on tags, so it needs the filename fallback.
   - `Trash/` (non-audio leftovers) and `Docs/` / `Playlists/` — genuinely skip.

   Reusing an existing scan's folder-exclusion list unmodified would be a quiet,
   damaging bug. The gap scan needs its **own** inclusion policy.

## 6. The pipeline, conceptually

Six stages. Each is a place the user can stop, look, and go back.

```
  ┌─────────────┐   ┌──────────────┐   ┌───────────┐   ┌────────┐   ┌────────┐   ┌──────────┐
  │ 1  IMPORT   │ → │ 2  LIBRARY   │ → │ 3 CANON-  │ → │ 4 MATCH│ → │ 5      │ → │ 6 SHOP   │
  │ the wanted  │   │    SNAPSHOT  │   │  ICALISE  │   │  LADDER│   │ TRIAGE │   │  LIST    │
  │ list (CSV)  │   │  what I own  │   │ both sides│   │        │   │ review │   │  + LEDGER│
  └─────────────┘   └──────────────┘   └───────────┘   └────────┘   └────────┘   └──────────┘
```

*(Revised: there are now **eight** stages — a saved-source step in front of stage 1, and a
verification pass after stage 6. See the revision note's "second pass" and "saved sources".)*

1. **Import the wanted list.** A CSV from TuneMyMusic (and anything CSV-shaped).
   Because column names vary by source service, the importer shows a **column-mapping
   preview** — "this column looks like Title, this one Artist, this one Album,
   Duration, ISRC" — with the user able to correct it, and the mapping remembered per
   source. Support importing several playlists and keeping them as named lists, since
   the real habit is "my four playlists", not "one file once".
2. **Snapshot what I own.** Build (or refresh from the cache) a tag-level index of the
   library under this feature's own inclusion policy (§5). Show a plain **freshness
   indicator** — "library snapshot: 8,412 tracks, read 4 minutes ago" — with a refresh
   button. This is the single biggest correctness lever in the feature and it should be
   impossible to misread.
3. **Canonicalise both sides.** Reduce each row and each file to a comparable form:
   a *core title*, a *primary artist*, a *set of modifiers*, plus corroborating fields.
   This is the interesting part — §8 and §9.
4. **Run the match ladder.** Cheap, certain checks first; expensive, fuzzy checks only
   for what's left. §7.
5. **Triage review.** Three buckets, with the *evidence* for every verdict visible, and
   the ability to override any row — in bulk. §11, §12.
6. **Produce the shopping list, and write the ledger.** The list is what the user acts
   on; the ledger is what makes next month cheap. §10, §13.

## 7. The match ladder

Rather than one similarity score, a cascade. Each rung is cheaper and more certain than
the one below it, and a row leaves the ladder as soon as a rung resolves it.

**Which rungs are even available depends on the source** (see the revision note). A
source declares its fields; the ladder skips the rungs its inputs can't support. Against
today's TuneMyMusic CSV, rungs 0, 1 and 4 are unavailable and everything funnels into
3/5/6 via the Stage 2.5 candidate splits — which is exactly why the better sources are
worth the setup.

| Rung | Comparison | Typical verdict |
|---|---|---|
| 0 | **Identity signal** — a YouTube video ID, an ISRC, or a stored source track ID, matches | MATCHED (definitive) |
| 0b | **Filename identity** — library filename carries the source `[videoId]`, or a `comment`/`purl` tag carries the source URL | MATCHED (definitive) |
| 1 | Exact normalized `artist` + `title` | MATCHED |
| 2 | Exact **core title** + **primary artist**, modifier sets agree | MATCHED |
| 3 | Core title + primary artist agree, **modifier sets differ** | → modifier adjudication (§8) |
| 4 | Fuzzy core title *within the same primary artist* | MATCHED / CONFIRM by score + corroboration |
| 5 | Fuzzy core title **across all artists** (catches artist-credit differences, remixer-credited-as-artist, "The" prefixes) | CONFIRM at best |
| 5b | Fuzzy **whole display string** against library **filenames** — the primary path for a YouTube-sourced library, since the filename usually *is* the video title | MATCHED / CONFIRM |
| 6 | Nothing plausible found | MISSING |

Two things matter about the shape:

- **Blocking by artist (rung 4 before rung 5) is what makes this fast.** Thousands of
  library tracks × hundreds of CSV rows is a lot of comparisons if done naively;
  restricting fuzzy work to the handful of files by the same artist collapses it. The
  cross-artist sweep (rung 5) only runs for rows that survived everything else — a
  small set.
- **Rung 0 is where the feature is won or lost.** ✅ *Checked — the TuneMyMusic export's
  ISRC column is present and empty.* But every better source returns a **YouTube video
  ID**, which is a stronger identity key than ISRC for this library: it identifies the
  exact upload, mashups and fan edits included, which an ISRC cannot. So rung 0 isn't
  dead — the identity just has to come from YouTube rather than from the music industry.
  Getting video IDs onto both sides is the highest-leverage work in the feature; see the
  revision note's backfill / going-forward split.

- **Blocking needs a fallback when there is no artist.** With a source that provides no
  artist field, rung 4 has nothing to block on and the search space stops collapsing.
  Cheap substitutes worth specifying: block on rare *title tokens* (an inverted token
  index, which `playlist_generator` already builds), or on the uploader/channel name
  where the source provides one.

## 8. The heart of it: what modifiers mean

This is the part the user correctly identified as "unique." A title is rarely just a
title:

```
  Song Title (feat. Guest) - 2011 Remastered Radio Edit
  └── core ──┘ └── modifiers ───────────────────────────┘
```

The proposal: **stop treating modifiers as noise to strip, and treat them as evidence
to classify.** Strip them off the core title for comparison purposes, but keep them as
a set, and adjudicate the *difference between the two sets*.

Modifiers fall into three classes:

**Class A — cosmetic. Same recording; the difference is just how it was labelled.**
`feat. X` / `ft.` / `with X`, `(Explicit)`, `(Clean)`, `(Album Version)`,
`(Bonus Track)`, `(Official Audio)` / `(Official Video)`, `(Remastered 2011)`,
trailing year stamps, `(Single)`. These almost always reflect whether the row came
from a single release or an album — exactly as the user described. **Difference here
should not block a match.**

**Class B — substantive. A genuinely different recording that the user would want
separately.** `(X Remix)`, `(Live at …)`, `(Acoustic)`, `(Instrumental)`,
`(Extended Mix)`, `(Club Mix)`, `(Demo)`, `(Cover)`, `(Karaoke)`, `(Sped Up)`,
`(Slowed + Reverb)`, `(Edit)` by a named third party. **Difference here should block a
match** — owning the original is not owning the remix.

**Class C — genuinely ambiguous, and library-dependent.** `(Original Mix)` vs. a bare
title, `(Radio Edit)` vs. bare, `(Mix)`, `(Version)`, `(Deluxe)`, `(Remaster)` when
you own an unlabelled file that may or may not already be that remaster. **These are
what the CONFIRM bucket is for.** They should never be guessed.

Three design consequences:

1. **This is a data table, not a pile of regexes.** A **modifier lexicon** —
   pattern → class → notes — is inspectable, testable, user-editable, and extendable
   without touching matching logic. Users' libraries differ; genre matters enormously
   here (an EDM library is 40% Class B, a rock library is 2%). Hard-coding this is the
   main way this feature gets abandoned as "it doesn't understand my music."
2. **Direction matters, and this is the non-obvious bit.** The same modifier difference
   means different things depending on which side has it:
   - CSV says `Song (feat. B)`, library has `Song` → **likely MATCHED.** Your file just
     doesn't carry the guest credit. (Class A, and the common case.)
   - CSV says `Song`, library has **only** `Song (Chris Lake Remix)` → **this is
     MISSING, not MATCHED.** You own a remix; you do not own the original, and the
     original is what's on the list. A naive "core titles match" rule gets this exactly
     backwards, and gets it backwards *in the dangerous direction* (§3).
   - CSV says `Song (Chris Lake Remix)`, library has only `Song` → **MISSING**, same
     logic mirrored.
3. **Class C is a queue, not a dead end.** Every Class C adjudication the user makes is
   information: "in *my* library, `(Original Mix)` and a bare title are the same
   thing." That belongs in the ledger (§10) and, ideally, gets promoted into the
   lexicon so it's never asked again.

## 9. Corroborating signals — how to shrink the CONFIRM bucket

Title and artist alone will leave an uncomfortably large ambiguous bucket. Other fields
can resolve many of those rows *without* asking the user.

- **Duration is the strongest and cheapest — where it exists.** The library side always
  has it (the fingerprint cache stores `duration`). ⚠️ The wanted side is the problem:
  the TuneMyMusic export inspected for the revision note has **no duration column at
  all**, so this signal is simply absent for today's source. Both recommended upgrades
  (`ytmusicapi`, `yt-dlp`) restore it — which is most of the argument for adopting one.
  Until then the confirm bucket runs larger, and §11's bulk triage carries the load
  instead.
  A remix or extended mix is typically 30–120 seconds off the original; the same
  recording tagged two different ways is within a second or two. So:
  matching core titles **+ duration within a couple of seconds** → confidently MATCHED
  even across a modifier difference. Matching core titles **+ duration off by more
  than ~15 seconds** → confidently a different version, so MISSING. **This one signal
  plausibly converts the majority of Class C rows into confident verdicts**, and it
  should be treated as a first-class part of the design, not an optimisation.
- **Album** — disambiguates the single-vs-album case directly, and helps with
  compilations and re-releases.
- **Year** — weak on its own (re-releases, remasters), useful as a tiebreaker.
- **ISRC / source track ID** — definitive where present (§7, rung 0).
- **Track number** — near-useless here; ignore.

The principle: **a verdict should be explainable in one sentence.** "Same core title,
same artist, duration within 1s, only difference is a `feat.` credit → owned." If a
verdict can't be stated that way, it belongs in CONFIRM. The evidence line is also
exactly what the review screen should display (§12), so building the explanation isn't
extra work — it *is* the output.

## 10. The bit most plans would miss: the decision ledger

Re-read the original problem. It isn't "compare a list once." It's **"this happens
again every few months."** A tool that answers perfectly but forgets everything has
solved a third of the problem.

So: **persist the outcome of each comparison as a per-playlist ledger** — a small
record, stored alongside the library (`Docs/`) or in config, holding for each wanted
track:

- what it was (artist / title / modifiers / identity signals),
- the verdict, and whether that verdict was **automatic or user-confirmed**,
- any user decision: *confirmed owned*, *confirmed missing*, *don't want this one*
  (an explicit ignore — the interlude, the skit, the track you hate), *already
  downloaded on <date>*,
- a note.

What this buys, directly against the stated problem:

- **Re-running a playlist next month surfaces only the genuinely new rows.** Everything
  previously settled stays settled, and the screen can lead with "**14 new tracks since
  you last checked this playlist**" — which is the answer the user actually wants.
- **Confirmation work is done once.** The user never re-adjudicates the same
  `(Original Mix)` question.
- **"Don't want" is respected permanently** instead of resurfacing forever.
- **A download checklist across sessions.** Exported-but-not-yet-downloaded rows stay
  visible as outstanding, so a half-finished download session isn't lost.
- **The playlist's own churn becomes visible** — tracks *removed* from the playlist
  upstream can be reported too, which is interesting on its own.

This is what turns the feature from a report into a workflow, and it's the single
highest-value idea in this plan. **It should be in Phase 1's data model even if its UI
lands later** — retrofitting identity onto a ledger afterwards is painful, and the
"what is this row, stably, across runs?" question is a design decision, not a detail.

## 11. Scale: triage by pattern, not by row

With thousands of tracks, even a 5% CONFIRM bucket is 100+ rows. Clicking through them
one at a time is the feature failing at the last step — and note that this exact
limitation is already recorded against Library Sync's per-item flags ("one-at-a-time
flagging; no bulk flag yet").

**Don't repeat it.** Group the CONFIRM bucket **by reason**, and let the user resolve a
whole class at once:

> **42 rows** differ only by a `feat.` credit — *the library file lacks the credit.*
> → **[ Accept all as owned ]** · [ Review individually ]
>
> **17 rows** where the library has a remix but not the version on the list.
> → **[ Send all to missing ]** · [ Review individually ]
>
> **9 rows** where core titles match but durations differ by more than 15s.
> → [ Accept all as owned ] · **[ Send all to missing ]** · [ Review individually ]
>
> **6 rows** with a weak fuzzy title match and no corroborating signal.
> → [ Review individually ] *(no bulk option — these genuinely need eyes)*

This reframes the user's job from "adjudicate 74 songs" to "make four decisions, then
look at six songs." It also composes with the ledger: a class decision can be recorded
as a **rule** ("always treat a `feat.`-only difference as owned"), so the class shrinks
to zero on future runs.

## 12. The workflow, screen by screen

A single workspace, stepping left-to-right the way the Indexer and Library Sync
workspaces already do.

**1 · Source**
- Import CSV (button + drag-drop). Multiple files allowed.
- Column-mapping preview with a few sample rows, editable, remembered.
- Saved playlists list: name, source service, row count, *last compared*, and *new
  since last compare*.
- Row-count and parse-warning summary ("1,204 rows; 3 skipped, no title").

**2 · Library snapshot**
- Library path (inherited from the workspace base, like every other workspace).
- Snapshot freshness + track count + **which folders were included** — stated
  explicitly, because §5's inclusion policy is surprising and the user must be able to
  see that `Not Sorted/` was counted.
- Refresh button. Runs in a worker thread with progress, per the project's threading
  model.

**3 · Compare**
- One primary button. Progress + the log drawer the app already has.
- Result header: the three counts, large and unmissable.
  `✅ MATCHED 1,031  ·  ⚠️ NEEDS CONFIRMATION 68  ·  ❌ MISSING 105`

**4 · Triage**
- Three sections/tabs, one per bucket. **MISSING and NEEDS CONFIRMATION lead**;
  MATCHED is collapsed by default (it's the boring, reassuring pile).
- CONFIRM opens on the **pattern groups** from §11, with the bulk actions up front.
- Every row shows: wanted artist/title, the best library candidate (if any), and the
  **one-sentence reason** for the verdict.
- Selecting a row opens an evidence panel: side-by-side wanted vs. candidate, field by
  field (core title, primary artist, modifiers, duration, album, year), with the
  differences highlighted and the other candidates listed. This is the same job
  Library Sync's "Match Inspector" does, and should look and feel like it.
- Any row can be moved to any bucket, individually or by selection. Decisions persist
  to the ledger.
- Optional and cheap: a **play** button on the candidate, since the app has a player —
  the fastest way to settle "is this the remix or not?" is to hear it.

**5 · Output**
- Export the shopping list (§13).
- "Mark exported rows as pending download," which is what makes the next run's
  outstanding-items view work.

## 13. Outputs — the thing the user actually uses

The MISSING list is the deliverable, and its format should suit *how it gets used*: as
input to manual downloading.

- **Plain text, one `Artist - Title` per line.** Unglamorous and the most useful thing
  here — it pastes into a downloader, a search box, or a notes app. Should be the
  default.
- **CSV**, carrying everything (album, duration, year, source playlist, reason).
- **HTML report into `Docs/`**, matching the app's existing report convention so it
  sits with the indexer and sync reports.
- **Grouped by artist / album, optionally.** A worthwhile workflow touch: if 6 of the
  missing tracks are from one album, it's often easier and better-quality to grab the
  album once than the 6 singles separately. Surfacing that grouping changes what the
  user does.
- **Clipboard copy** of the whole list. Sounds trivial; it's the action taken most.

## 14. Alternative approaches and additions considered

Presented with a recommendation each, since several have real merit.

**Option A — CSV import (still the core, but now understood as the floor).** Universal:
TuneMyMusic bridges Spotify, Apple Music, YouTube, Deezer, Tidal. No authentication, no
API keys, no rate limits, no partnership access. Works offline. Matches the habit the user
already has. ⚠️ **Revised:** for YouTube Music it delivers only a free-text video title —
no artist, no album, no duration, no ISRC (see the revision note). It stays the required
Phase 1 source because it needs no setup, but it is the **accuracy floor**, not the
baseline, and the engine must not be built around its schema.

**Option A′ — read YouTube directly (`ytmusicapi` / `yt-dlp`). NEW, and now the
recommended upgrade path.** `ytmusicapi.get_playlist()` returns `videoId`, `title`,
`artists[]`, `album`, `duration` and `isAvailable` per track — the structured data the CSV
lacks, with no transfer step and therefore no dropped rows. `yt-dlp --dump-json` needs no
authentication at all and yields id, title, uploader and duration. Either restores rungs 0
and 4 and the duration signal. **Recommendation: not required for Phase 1, but the
importer must be pluggable from Phase 1 so this drops in as a strict improvement to every
already-imported playlist.**

**Option B — Connect directly to the playlist service (Spotify API).** `spotipy` is
already in `requirements.txt`. Upside is real: one-click refresh instead of a manual
export; richer, cleaner data (canonical titles, ISRC, exact duration, album, stable
track IDs) which would substantially shrink the CONFIRM bucket; and the service's own
change tracking could tell you what's new without any comparison at all.
Downside: authentication setup, one service only, and this repo's own roadmap records
that Spotify access here *"requires authenticated, partnership-style access rather than
the open access the working providers enjoy."*
**Recommendation: don't build it now, but design the importer as a pluggable
"wanted-list source" so a Spotify source drops in later without touching the matcher
or the triage UI.** The matching engine should never know where the list came from.

**Option C — Canonicalise both sides through MusicBrainz.** The "proper" fix: resolve
every wanted row and every library file to a MusicBrainz *recording*, then compare
identities instead of strings. This genuinely solves the remix/variant problem rather
than heuristically approximating it.
Against it: slow and rate-limited across thousands of tracks; needs network; the
library side realistically needs AcoustID fingerprinting to resolve reliably; and in
this codebase MusicBrainz isn't even an independently-discoverable lookup source today
(it only appears nested inside an AcoustID match — a known, recorded gap).
**Recommendation: not core, and *weaker than first assessed*.** The revision note's
sample shows a library heavy on mashups, fan edits and unofficial uploads — material that
largely **isn't in MusicBrainz at all**, so canonicalisation would fail on exactly the
rows that are hardest. Keep it only as an opt-in "deep resolve" right-click for stubborn
mainstream-catalogue CONFIRM rows, never as a pipeline stage.

**Option D — Close the loop with the existing audio tooling.** After downloading the
missing tracks, the files *do* exist — so the existing Duplicate Finder and Library
Sync can verify the result with real fingerprints, catching anything this feature
wrongly called MISSING before it pollutes the library.
**Recommendation — SUPERSEDED. This is now a first-class pipeline stage, not guidance.**
See "The second pass" in the revision note. Pointing the user at Library Sync is not
enough, because the interesting outcome — *the downloader fetched a different recording
than you asked for* — needs a decision made here, and needs writing back into the
ledger. Left as guidance, the ledger silently decays: rows marked satisfied by a
wrong-version download are never asked about again.

**Option E — Learn from confirmations.** Every Class C adjudication is a labelled
example. Promoting repeated decisions into lexicon rules ("for this library,
`(Original Mix)` ≡ bare title") makes the CONFIRM bucket shrink with use.
**Recommendation: yes, but as simple remembered rules, not machine learning.** Phase 4.
Explicitly *not* recommended: embeddings / trained models for title matching — the
problem is small, the rules are legible, and a model that can't explain a verdict is
disqualified by §9's one-sentence rule.

**Option F — `rapidfuzz` instead of `difflib`.** Materially faster and better at exactly
this kind of short-string matching, with token-set ratios that handle reordered artist
credits well. Cost: a new dependency, in a project that currently does fuzzy matching
with stdlib.
**Recommendation: design against an interchangeable scorer; start on `difflib` (proven
here, zero cost) and switch if the artist-blocked search proves too slow in practice.**
Decide with a measurement, not up front.

**Option G — Match on filenames instead of tags.** The library's filenames are
normalised and structured by the Indexer, so they're unusually reliable here, and
`playlist_generator` already proves the approach works.
**Recommendation — REVISED to the primary path for a YouTube-sourced library.** Tags
are richer in principle (duration, album, year), but files pulled from YouTube usually
keep the video title as their filename while carrying empty or junk tags — which is why
they land in `Manual Review/` in the first place. Comparing the wanted list's raw
`Track name` against library **filenames** is therefore close to exact string matching
rather than fuzzy semantics, and it bypasses Stage 2.5's parsing problem entirely for any
file that kept its original name. Match filenames first, tags second, and treat a
retained `[videoId]` filename suffix as a rung-0 identity.

**Option H — Also flag "owned but not on any list."** The inverse report: library
tracks that appear in no imported playlist. Cheap to produce once both sides are
indexed, and interesting for pruning.
**Recommendation: out of scope.** It answers a different question, and mixing it in
dilutes a workspace whose value is having exactly one clear answer.

## 15. Risks and edge cases to handle explicitly

Most of these are text-normalisation traps, and every one of them produces a **false
MISSING** — annoying but recoverable (§3). They're listed so the spec can decide each
one deliberately rather than discovering them in the field.

- **CSV column drift.** TuneMyMusic's headers differ by source service and change over
  time. Hence the editable mapping step — never hard-code column positions or names.
- **Unicode and punctuation.** Curly vs. straight apostrophes (`Don't` / `Don’t`),
  accents (`Beyoncé`), em-dashes, `&` vs. `and`, full-width characters, emoji in
  titles. The project has **no** transliteration dependency today, so decide: fold
  accents (needs a dependency or a hand-rolled table) or compare accent-sensitively
  and accept some CONFIRM rows.
- **Leading articles.** `The Beatles` vs. `Beatles`; `A Day in the Life`.
- **Artist order in collaborations.** `Artist A, Artist B` vs. `Artist B feat. A` vs.
  `Artist A & B`. Partly solved by the existing primary-artist selection, but the CSV
  and the tags may disagree about who's primary — which is why rung 5's cross-artist
  sweep exists.
- **Remixer credited as the artist.** The library file may be tagged with the *remixer*
  as artist (and the Indexer deliberately does this in some cases), while the CSV
  credits the original artist. Artist-blocked matching will miss these entirely; only
  the cross-artist rung catches them.
- **"Various Artists" and DJ mixes.** Compilation tags are unreliable; continuous mixes
  have one file for many "tracks."
- **Classical, and long descriptive titles.** `Symphony No. 5 in C minor, Op. 67: I.
  Allegro con brio` — performer matters more than composer, movement names collide
  across recordings, and fuzzy title matching does badly. Worth an explicit "this
  feature is weak on classical" note rather than a silent poor experience.
- **Untagged or badly tagged library files.** These *are* owned but can't be matched on
  tags → false MISSING. Handled by the filename fallback (Option G) and by including
  `Manual Review/`.
- **Reserved folders.** Covered in §5 — the single most consequential correctness
  decision in the feature.
- **Duplicate rows within one CSV**, and the same track appearing across several
  imported playlists. De-duplicate the wanted list itself, and remember which playlists
  a wanted track came from.
- **Multiple library copies of the same song** (different codecs, a `Quarantine/`
  loser). MATCHED is MATCHED — but the evidence panel should show all copies, and this
  is a natural hand-off to the Duplicate Finder.
- **Library changes mid-session.** The snapshot's freshness indicator must be honest;
  a stale snapshot is a silent source of both error types.

## 16. Suggested phasing

Each phase ends with something usable, and Phase 1 alone already beats the status quo.

**Phase 0 — the five-minute check that decides the architecture.** Look at the files
already in the library: are they named like video titles, does any keep a `[videoId]`
suffix, does any carry a YouTube URL in a `comment`/`purl` tag? Add `comment`, `purl` and
`website` to `TAG_KEYS` so the answer is even visible. If identities are recoverable, most
of the rest is exact matching; if not, the fuzzy path is the main path.
→ *Cheapest, highest-leverage step in the plan.*

**Phase 1 — the walking skeleton.** Pluggable source interface (field declaration, not a
fixed schema) with the CSV as its first implementation; column mapping; library snapshot
with the corrected inclusion policy **and filename indexing**; Stage 2.5 candidate-split
parsing; rungs 0/0b, 2, 5b; two buckets plus an "unsure" pile; plain-text + CSV export.
**Establish the ledger's data model here, keyed on a stable identity, even if nothing
reads it yet.**
→ *Already answers "what's new in this playlist?" for most rows.*

**Phase 1.5 — stop the problem growing.** Capture and persist a source identity (video
ID) for every newly acquired track, so future comparisons are exact lookups rather than
fuzzy guesses. Small, and it bounds all the fuzzy work above to a one-time backfill.

**Phase 2 — the modifier lexicon and the third bucket.** Class A/B/C lexicon as data;
directional adjudication; duration corroboration; the real three-bucket triage screen
with per-row evidence and overrides; HTML report.
→ *This is where the feature becomes trustworthy.*

**Phase 3 — the ledger, live.** Persisted decisions; "new since last compare";
"don't want"; pending-download tracking; playlist-churn reporting.
→ *This is where the feature becomes a habit rather than a one-off.*

**Phase 4 — scale and learning.** Pattern-grouped bulk triage; promoting repeated
confirmations into lexicon rules; user-editable lexicon UI; grouped-by-album output.
→ *This is where a 68-row CONFIRM bucket becomes four clicks.*

**Phase 2.5 — the second pass.** Verify a download folder: identity check first, fingerprint
fallback, the five outcomes, and the wrong-version correction written back into the ledger.
Slots here rather than later because without it the ledger's accuracy decays from the first run
onward — and it is mostly wiring existing tested modules together.
→ *This is what stops the feature from recreating the problem it solves.*

**Phase 3.5 — saved sources.** Templates for source + URL + settings, listed with *last run* and
*new since then*, each with a one-press run. Plus "run all" with a combined, de-duplicated list.
→ *Turns a five-minute setup into one click, which is what makes it a monthly habit.*

**Phase 5 — optional extensions.** Pluggable Spotify source (Option B); MusicBrainz
deep-resolve escalation (Option C); post-download verification hand-off (Option D);
scorer swap if measurement calls for it (Option F).

## 17. Open questions to settle before the spec

1. ✅ **ANSWERED — and badly.** The TuneMyMusic export of a YouTube Music playlist has an
   ISRC column that is empty, no duration column at all, and empty artist and album; the
   entire payload is a free-text video title. See the revision note at the top, which
   supersedes several assumptions below.

2. **Look at the library files on disk** — the new highest-value question. Are they named
   like video titles? Does any carry a `[videoId]` filename suffix, or a YouTube URL in a
   `comment`/`purl` tag? This decides whether the feature is built around exact identity
   matching or around fuzzy string work.

3. **Is `yt-dlp` already in the download toolchain**, and can its output template and
   `--embed-metadata` be turned on going forward? If yes, the "stop the problem growing"
   half of the reframe is nearly free.
2. **Feature name**, since it lands in the nav rail and in config keys.
3. **The normalized-columns prerequisite (§5.1):** fix the Qt cache writer so the
   existing columns get populated, or give this feature its own index? Fixing the
   writer helps other features too; a private index is self-contained and can't
   regress anything.
4. **Duration tolerances.** Proposed starting points: ≤3s → same recording; >15s →
   different; in between → CONFIRM. Needs a sanity check against the real library.
5. **How much CONFIRM is acceptable** on a first run before the feature feels like
   homework? This sets the thresholds.
6. **Genre reality check.** How much of the library is electronic/remix-heavy? That
   determines how much of the total effort belongs in §8 versus everything else.
7. **Where the ledger lives** — `Docs/` alongside the reports (travels with the
   library, survives a reinstall) or the config file (travels with the app)? `Docs/`
   looks right, since it belongs to the library.

## 18. Back to the problem

Every part of this plan should be checkable against one sentence:

> *"I have many new songs mixed in with old songs that need to be found, sorted, then
> downloaded by me."*

- **Found** → the import + match ladder (§6, §7).
- **Sorted** → the three buckets, biased so that uncertainty never hides a song (§3, §8).
- **Downloaded by me** → a short, trustworthy, pasteable list (§13) — and the ledger
  that means next month's list is short too (§10).

If a proposed addition doesn't serve one of those four, it belongs in a different
feature.
