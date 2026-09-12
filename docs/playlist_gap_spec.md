# Technical Spec — Playlist Gap

*Implementation spec for the feature described in `docs/playlist_gap_feature_plan.md`.
The plan argues **why**; this document says **what to build**. Where they disagree, this
document wins for mechanics and the plan wins for intent.*

*Target: the **Qt** app (`alpha_dex_gui.py`). Written 2026-09-12.*

---

## 0. Scope

**In scope.** A new Qt workspace (`playlist_gap`) and a Qt-free backend that:
reads a list of wanted tracks from a pluggable source; snapshots the library; parses
free-text display strings into candidate artist/title splits; matches wanted rows against
the library through a cascading ladder; sorts the result into three buckets; records every
decision in a durable ledger; exports a download list; and verifies a folder of downloads
against what was actually asked for.

**Out of scope for v1.** Spotify/Apple sources; MusicBrainz deep-resolve; the inverse
"owned but on no list" report; any automatic downloading. Also out of scope: changing the
Duplicate Finder or Library Sync beyond calling them.

**Non-negotiable constraints** (from `CLAUDE.md`, restated because they shape every module
below):

- Backend modules must be importable without Qt or Tkinter. Business logic never imports
  `PySide6`.
- GUI state is mutated only on the main thread. Long work runs in `QtCore.QThread`
  subclasses that emit signals, matching the `SyncScanWorker` / `SyncBuildWorker` pattern in
  `gui/workspaces/library_sync.py`.
- Preview before mutation. The only stage that writes to the library is the Verify pass's
  quarantine action, and it goes through `duplicate_consolidation`'s existing
  plan-then-execute contract.
- All user settings read/write through `config.load_config()` / `config.save_config()`.
- Heavy collaborators (source readers, tag readers, fingerprinters, clocks) are injectable,
  following `controllers/normalize_controller.py`. Defaults import lazily inside the
  functions that need them.

---

## 1. Module manifest

### New backend modules (Qt-free, repo root unless noted)

| Module | Responsibility |
|---|---|
| `playlist_gap_types.py` | Dataclasses and enums shared by everything below. No logic, no imports beyond stdlib — exists to prevent the circular-import problem `library_sync_types.py` was created to solve. |
| `playlist_gap_sources.py` | The `WantedListSource` protocol, `SourceCapabilities`, and the CSV / ytmusicapi / yt-dlp implementations. |
| `playlist_gap_parse.py` | Stage 2.5. Display-string → candidate `(artist, title, modifiers)` splits. |
| `playlist_gap_lexicon.py` | Modifier lexicon: load, merge user overrides, classify a modifier, adjudicate two modifier sets. |
| `playlist_gap_snapshot.py` | Library snapshot: inclusion policy, cache reads, the three lookup indexes. |
| `playlist_gap_match.py` | The match ladder, the scorer protocol, the duration gate, reason codes. |
| `playlist_gap_ledger.py` | SQLite ledger: schema, migrations, read/write API. |
| `playlist_gap_verify.py` | Second pass: identity check, fingerprint check, the five outcomes. |
| `playlist_gap_report.py` | Download-list and run-report rendering (txt / csv / html / m3u). |
| `controllers/playlist_gap_controller.py` | Thin orchestration: run a source end-to-end, wire progress callbacks. No Qt. |

### New Qt modules

| Module | Responsibility |
|---|---|
| `gui/workspaces/playlist_gap.py` | The workspace: tile strip, seven panes, five `QThread` workers. |
| `gui/widgets/gap_tile_strip.py` | The tile strip widget (overview + navigation). |
| `gui/widgets/gap_evidence.py` | The wanted-vs-candidate evidence comparison, incl. cover art. |

### Existing files touched

| File | Change | Why |
|---|---|---|
| `gui/main_window.py` | Add `"playlist_gap": PlaylistGapWorkspace` to `_WORKSPACE_MAP`. | Register the workspace. |
| `utils/audio_metadata_reader.py` | Add `comment`, `purl`, `website` to `TAG_KEYS`. | **Prerequisite.** Without these the app cannot see a YouTube URL that `yt-dlp --embed-metadata` already wrote. Three entries; benefits other features too. |
| ~~`gui/workspaces/duplicates.py`~~ → `fingerprint_cache.py` | **Implemented differently — see note.** `store_fingerprint()` now *derives* `normalized_artist` / `normalized_title` / `normalized_album` from the `tags` argument when the caller did not pass them explicitly. | **Prerequisite.** Those columns exist but only the legacy Tkinter writer filled them, so they were `NULL` in the Qt app. Fixing the writer rather than one call site means every current and future caller that passes tags gets them, instead of each having to remember. Explicit values still win, so the legacy writer is unaffected and no call site changed. |
| `config.py` | `cfg.setdefault("playlist_gap", {...})` in `load_config()`. | Saved sources and thresholds. |
| `requirements.txt` | Add `ytmusicapi` and `yt-dlp` as **optional**, commented like `essentia`. | Neither may be a hard dependency — the CSV source must work without them. |

### New tests

`tests/test_playlist_gap_parse.py`, `_lexicon.py`, `_match.py`, `_ledger.py`,
`_snapshot.py`, `_sources.py`, `_verify.py`, `_report.py`. One per backend module, following
the repo's `test_<module>.py` convention, using `monkeypatch` and `tmp_path`, no custom base
classes.

---

## 2. Data model — `playlist_gap_types.py`

```python
class Verdict(str, Enum):
    OWNED   = "owned"      # the library has this recording
    UNSURE  = "unsure"     # needs a human decision
    MISSING = "missing"    # go download this
    IGNORED = "ignored"    # user said "never want this"
    UNAVAILABLE = "unavailable"   # source says the row is gone (deleted video)


class Rung(str, Enum):
    IDENTITY       = "0"    # video_id / isrc / source track id
    FILE_IDENTITY  = "0b"   # library filename [videoId] or comment/purl tag
    EXACT_TAG      = "1"    # normalized artist + title
    CORE_ARTIST    = "2"    # core title + primary artist, modifier sets agree
    MODIFIER       = "3"    # as above but modifier sets differ -> adjudication
    FUZZY_IN_ARTIST= "4"    # fuzzy core title, blocked by primary artist
    FUZZY_GLOBAL   = "5"    # fuzzy core title across artists
    FUZZY_FILENAME = "5b"   # fuzzy whole display string vs library filenames
    NONE           = "6"    # nothing plausible


class ModifierClass(str, Enum):
    COSMETIC     = "cosmetic"      # same recording, labelled differently
    SUBSTANTIVE  = "substantive"   # a genuinely different recording
    AMBIGUOUS    = "ambiguous"     # library-dependent; never guessed
    TITLE_PART   = "title_part"    # not a modifier at all — part of the real title


@dataclass(frozen=True)
class SourceCapabilities:
    """What fields a source can actually provide. The ladder skips rungs it can't support."""
    display_string: bool = True     # always true; the floor
    title: bool = False
    artist: bool = False
    album: bool = False
    duration: bool = False
    isrc: bool = False
    video_id: bool = False
    availability: bool = False
    artwork_url: bool = False


@dataclass
class WantedTrack:
    row_id: str                     # stable identity — see §8.2
    display: str                    # the raw string, always present
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration: int | None = None     # seconds
    isrc: str | None = None
    video_id: str | None = None
    artwork_url: str | None = None
    available: bool = True
    source_id: str = ""
    playlists: tuple[str, ...] = ()  # a row can come from several saved sources
    position: int | None = None


@dataclass
class LibraryTrack:
    path: str
    ext: str
    duration: int | None
    bitrate: int | None
    fingerprint: str | None
    tags: Dict[str, object]
    norm_artist: str | None
    norm_title: str | None
    norm_album: str | None
    video_id: str | None            # from filename [id] or comment/purl tag
    filename_norm: str              # normalized basename, for rung 5b


@dataclass(frozen=True)
class Modifier:
    raw: str
    cls: ModifierClass
    key: str                        # normalized, for set comparison
    actor: str | None = None        # e.g. the remixer name, when captured


@dataclass(frozen=True)
class SplitCandidate:
    artist: str | None
    title: str
    modifiers: tuple[Modifier, ...]
    prior: float                    # 0..1, how likely this split is a priori
    origin: str                     # "provided" | "delimiter:-" | "reversed" | "whole"


@dataclass
class Candidate:
    track: LibraryTrack
    score: float                    # 0..1
    rung: Rung
    duration_delta: int | None      # seconds, abs
    modifier_delta: tuple[Modifier, ...]
    split: SplitCandidate | None    # which parse produced the hit


@dataclass
class GapResult:
    wanted: WantedTrack
    verdict: Verdict
    rung: Rung
    candidates: list[Candidate]     # ranked, best first; may be empty
    reason_code: str                # see §7.2 — drives bulk triage grouping
    reason_text: str                # the one-sentence explanation
    auto: bool = True               # False once a human decided it
```

**Rule:** `candidates` is a *ranked list*, never a single best match. The mockup's
"possible counterparts" column and every bulk action depend on the full list being carried
through, and the lowest-text-scoring candidate is sometimes the only certain one (an
unfiled file whose name still holds the video ID).

---

## 3. Sources — `playlist_gap_sources.py`

```python
class WantedListSource(Protocol):
    key: str                        # "csv" | "ytmusic" | "ytdlp"
    def capabilities(self) -> SourceCapabilities: ...
    def fetch(
        self,
        spec: SourceSpec,
        *,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> list[WantedTrack]: ...
```

`SourceSpec` is the saved-source record (§9): `kind`, `url_or_path`, `name`, plus a
free-form `options` dict.

### 3.1 `CsvSource`

- Sniffs the dialect with `csv.Sniffer`, reads with `csv.DictReader`.
- **Never hard-codes column names.** Produces a `ColumnMapping` proposal by scoring each
  header against synonym sets (`{"track name", "title", "song", "song title"}` etc.), which
  the UI shows and the user can correct. The corrected mapping is stored on the saved source.
- Capabilities are computed **per file, after mapping** — a column that exists but is empty
  in every row reports `False`. This is the mechanism that made the TuneMyMusic export's
  `ISRC` column honest.
  > Implementation: sample the first 200 mapped rows; a field is available if ≥ 5% non-empty.
- Blank `display` rows become `available=False`, `Verdict.UNAVAILABLE`. **Never silently
  dropped** — they are counted and listed.

### 3.2 `YtMusicSource` (optional dependency)

- `from ytmusicapi import YTMusic` inside the function, guarded by `ImportError` → the UI
  offers CSV instead with a clear message.
- `get_playlist(playlist_id, limit=None)` for a playlist; `get_library_songs()` for the
  library; `LM` for Liked.
- Field mapping: `videoId → video_id`, `title → title`, `artists[0].name → artist`
  (all names joined for display), `album.name → album`, `duration` (`"M:SS"`) parsed to
  seconds, `isAvailable → available`, `thumbnails[-1].url → artwork_url`.
- Capabilities: everything except `isrc`.
- Auth is `ytmusicapi`'s browser-header file; path stored per saved source, never in the
  ledger.

### 3.3 `YtDlpSource` (optional dependency)

- Subprocess: `yt-dlp --dump-json --flat-playlist <url>`, one JSON object per line.
- Field mapping: `id → video_id`, `title → display`, `uploader/channel → artist` *hint only*
  (it is a channel name, not reliably an artist — it feeds the parser's prior, never rung 1),
  `duration → duration`.
- Capabilities: `video_id`, `duration`, `display_string`. **Not** `title`/`artist` — flat
  mode gives a video title, so the parser must run.
- `options["deep"] = True` drops `--flat-playlist` for richer `track`/`artist`/`album`
  fields at the cost of one request per video; off by default, offered for small playlists.

**Adding a source later** means implementing the protocol and registering it in
`SOURCE_REGISTRY`. Nothing in `playlist_gap_match` or the workspace changes.

---

## 4. Library snapshot — `playlist_gap_snapshot.py`

```python
def build_snapshot(
    library_root: str,
    *,
    include: FolderPolicy = DEFAULT_INCLUDE,
    cache_db: str | None = None,
    read_tags: Callable[[str], dict] = ...,     # injectable
    progress: Callable[[int, int, str], None] | None = None,
) -> LibrarySnapshot: ...
```

### 4.1 Inclusion policy — the correctness lever

```python
DEFAULT_INCLUDE = FolderPolicy(
    include_reserved={"Not Sorted", "Quarantine", "Manual Review"},
    exclude_reserved={"Trash", "Docs", "Playlists"},
)
```

This **deliberately differs** from the Indexer's and Duplicate Finder's skip lists.
`Not Sorted/` holds downloaded-but-unfiled music, `Quarantine/` holds duplicate losers, and
`Manual Review/` holds badly-tagged files — all three are *things the user already has*, and
excluding them produces exactly the false-`MISSING` re-download this feature exists to
prevent.

> **Do not reuse `music_indexer_api`'s or `simple_duplicate_finder`'s folder skip constants
> here.** Define this policy locally and surface the resulting counts in the UI so the user
> can see `Not Sorted` was counted. A regression test asserts a file under `Not Sorted/`
> matches.

### 4.2 Data sources, cheapest first

1. `fingerprint_cache.get_cached_fingerprint_metadata(path, db_path)` → `(fp, metadata)`.
   Supplies `tags_json`, `duration`, `ext`, `bitrate`, and the `normalized_*` columns for
   free on any file the Duplicate Finder has already seen.
2. For cache misses, `utils.audio_metadata_reader.read_tags(path)`. **No fingerprinting
   during snapshot** — pass 1 never needs audio. Fingerprints are read from cache if present
   and left `None` otherwise; only the Verify pass computes them.
3. `video_id` extraction, in order: an `[11-char]` suffix in the filename
   (`re.compile(r"\[([A-Za-z0-9_-]{11})\]")`), then a YouTube URL in the `purl`, `comment`,
   or `website` tag. **Requires the `TAG_KEYS` prerequisite.**

### 4.3 Indexes built once

```python
@dataclass
class LibrarySnapshot:
    tracks: list[LibraryTrack]
    by_video_id: dict[str, list[int]]
    by_artist: dict[str, list[int]]      # normalized primary artist -> row indexes
    by_title_token: dict[str, set[int]]  # rare-token inverted index
    by_filename_token: dict[str, set[int]]
    read_at: float
    counts: dict[str, int]               # per included folder, for the UI
```

The token indexes are what keep the ladder fast when the source gives no artist. Follow
`playlist_generator._build_audio_index`'s shape — it already proves the approach. Drop
tokens appearing in > 5% of tracks (they select nothing useful).

**Freshness.** `read_at` is surfaced verbatim in the UI. A snapshot is never silently
reused across a library change; the workspace re-reads on library path change and offers a
manual refresh.

---

## 5. Parsing — `playlist_gap_parse.py`

```python
def candidate_splits(
    display: str,
    *,
    lexicon: Lexicon,
    provided_artist: str | None = None,
    provided_title: str | None = None,
    channel_hint: str | None = None,
    max_candidates: int = 8,
) -> list[SplitCandidate]: ...
```

**Governing rule: never commit to a parse.** Emit several candidates and let the library
decide. One hit is both the match and proof the parse was right; several hits is an
`UNSURE`; no hits is `MISSING`.

### 5.1 Algorithm

1. **Unicode fold.** NFKC; curly → straight quotes; en/em dash → hyphen; collapse whitespace.
   *Accent folding is a separate, configurable step* — default **on**, implemented with a
   hand-rolled table so no new dependency is needed (`unicodedata.normalize("NFKD")` then
   drop combining marks).
2. **If the source provided `title` and `artist`**, emit that as a candidate with
   `prior=1.0`, `origin="provided"`. Still continue — a provided artist can be a channel name.
3. **Lift bracketed prefixes.** A leading `(...)` or `[...]` before any delimiter becomes an
   `uploader_prefix` token, not part of the artist: `(Triple Vibe) HOME - Resonance…`.
4. **Extract trailing/inline modifiers** via the lexicon (§6), leaving the core string.
   A parenthetical classified `TITLE_PART` stays in the core.
5. **Split on delimiters**, in priority order:
   `" - "`, `" – "`, `" — "`, `" · "`, `" | "`, `" _ "`, `": "`.
   For each occurrence emit **both orderings** — `(left=artist, right=title)` with
   `prior=0.8`, and the reverse with `prior=0.35`. Multiple occurrences of the same
   delimiter emit one candidate per split point.
6. **Mashup guard.** `" x "` / `" X "` / `" vs "` is **never** an artist/collab separator
   here. When it is the only delimiter, emit a single `artist=None` candidate over the whole
   string, flagged `mashup`.
   > This is why `music_indexer_api.extract_primary_and_collabs()` **must not** be reused on
   > display strings: its separator list is `[" feat.", " ft.", " & ", " x ", ", ", ";"]`, so
   > `Self Aware x Babydoll` parses as *Self Aware* featuring *Babydoll*. It also falls back
   > to splitting on a lowercase→uppercase boundary, which mangles video titles. Use it only
   > on real `artist` tag values, never on `display`.
7. **Whole-string fallback** with `artist=None`, `prior=0.15`, so rung 5b always has
   something to work with.
8. Deduplicate by `(norm(artist), norm(title), modifier_keys)`; sort by `prior` desc;
   truncate to `max_candidates`.

### 5.2 Prior adjustments

- `+0.1` if the left side matches a known library artist exactly.
- `+0.05` if `channel_hint` normalizes to the left side (an `Artist - Topic` channel).
- `-0.2` if either side is empty after stripping, or is < 2 characters.

Priors only order the probes; they never decide a verdict on their own.

---

## 6. Modifier lexicon — `playlist_gap_lexicon.py`

### 6.1 Format

Shipped default at `data/playlist_gap_lexicon.json`, merged with a user file at
`<library>/Docs/playlist_gap_lexicon.json` (user entries win on the same `id`).

```json
{
  "version": 1,
  "rules": [
    {"id": "feat",        "pattern": "\\b(feat\\.?|ft\\.?|featuring|with)\\s+(?P<actor>.+)$",
     "class": "cosmetic",    "scope": "inline"},
    {"id": "remaster",    "pattern": "\\b(\\d{4}\\s+)?remaster(ed)?\\b",  "class": "cosmetic"},
    {"id": "explicit",    "pattern": "\\b(explicit|clean|album version|bonus track)\\b",
     "class": "cosmetic"},
    {"id": "official",    "pattern": "\\bofficial\\s+(audio|video|music video|lyric video)\\b",
     "class": "cosmetic"},

    {"id": "remix",       "pattern": "\\b(?P<actor>.+?)\\s+remix\\b",     "class": "substantive"},
    {"id": "live",        "pattern": "\\blive(\\s+at\\s+(?P<actor>.+))?\\b", "class": "substantive"},
    {"id": "acoustic",    "pattern": "\\b(acoustic|instrumental|karaoke|demo|cover)\\b",
     "class": "substantive"},
    {"id": "extended",    "pattern": "\\b(extended|club)\\s+(mix|version)\\b", "class": "substantive"},
    {"id": "fanedit",     "pattern": "\\b(but it'?s\\s+.+|sped\\s*up|slowed(\\s*\\+\\s*reverb)?|nightcore|bass\\s*boost(ed)?|8d\\s*audio|mashup)\\b",
     "class": "substantive"},
    {"id": "twin",        "pattern": "\\btwin\\s+version\\b",             "class": "substantive"},

    {"id": "originalmix", "pattern": "\\boriginal\\s+mix\\b",             "class": "ambiguous"},
    {"id": "radioedit",   "pattern": "\\bradio\\s+edit\\b",              "class": "ambiguous"},
    {"id": "bareversion", "pattern": "\\b(version|mix|edit|deluxe)\\b",   "class": "ambiguous"}
  ],
  "title_parts": ["a deal with god", "what's going on", "part 1", "pt. 2"]
}
```

**The fan-edit family is first priority, not an afterthought.** This library is heavy on
`but it's beats 3,3`-style derivatives; the plan's "2% of a rock library" framing does not
apply here.

`title_parts` is an explicit allow-list of parentheticals that are part of a canonical title
(`Running Up That Hill (A Deal With God)`). Matching is case-insensitive on the
parenthetical's inner text. It is user-editable because it can never be complete.

### 6.2 Adjudication

```python
def adjudicate(
    wanted: Sequence[Modifier],
    candidate: Sequence[Modifier],
) -> tuple[Verdict, str]:   # (suggested verdict, reason_code)
```

Compares the two sets **directionally**:

| Situation | Result | Reason code |
|---|---|---|
| Sets equal | pass through to the next signal | `mod_equal` |
| Difference is only `COSMETIC` | `OWNED` | `mod_cosmetic_only` |
| Wanted has a `SUBSTANTIVE` the candidate lacks | `MISSING` | `mod_wanted_substantive` |
| Candidate has a `SUBSTANTIVE` the wanted lacks | `MISSING` | `mod_candidate_substantive` |
| Any `AMBIGUOUS` in the difference | `UNSURE` | `mod_ambiguous` |
| Both sides have different `SUBSTANTIVE` actors | `MISSING` | `mod_different_actor` |

The third and fourth rows are the directional asymmetry the plan calls out: **owning a remix
is not owning the original, and vice versa.** Both must be tested explicitly.

---

## 7. Matching — `playlist_gap_match.py`

```python
def match_all(
    wanted: Sequence[WantedTrack],
    snapshot: LibrarySnapshot,
    *,
    caps: SourceCapabilities,
    lexicon: Lexicon,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    scorer: Scorer = DifflibScorer(),
    ledger: Ledger | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[GapResult]: ...
```

### 7.1 The ladder

Each rung is a predicate over `(WantedTrack, SplitCandidate, LibrarySnapshot)`. A row leaves
the ladder as soon as one resolves it. **Rungs whose inputs the source cannot provide are
skipped** — that is the entire point of `SourceCapabilities`.

| Rung | Requires | Predicate | Resolves to |
|---|---|---|---|
| 0 | `video_id` or `isrc` | `snapshot.by_video_id[w.video_id]` non-empty | `OWNED`, definitive |
| 0b | — | any library track whose extracted `video_id == w.video_id` | `OWNED`, definitive |
| 1 | `artist` + `title` | `norm(artist), norm(title)` exact hit | `OWNED` |
| 2 | — | core title + primary artist exact, modifier sets equal | `OWNED` |
| 3 | — | core title + artist exact, modifier sets differ | `adjudicate()` (§6.2) |
| 4 | artist known | fuzzy core title within `by_artist[artist]`, `score ≥ fuzzy_floor` | duration gate |
| 5 | — | fuzzy core title over `by_title_token` shortlist | `UNSURE` at best |
| 5b | — | fuzzy `display` over `by_filename_token` shortlist | duration gate |
| 6 | — | nothing above fired | `MISSING` |

**Blocking.** Rung 4 restricts comparisons to one artist's tracks. When the source gives no
artist (rung 4 unavailable), rung 5 builds its shortlist from the *rarest* two or three
tokens of the core title rather than scanning all 8,412 tracks. Budget: full run over 1,204
wanted × 8,412 library in **under 5 seconds**, excluding I/O, on a cold snapshot.

**Every split candidate is probed** at rungs 2–5b, in `prior` order. If exactly one split
produces a hit, that split is recorded on the `Candidate` and the verdict stands. If two or
more different splits hit different tracks, the verdict is forced to `UNSURE` with reason
`split_ambiguous` — the parser could not be disambiguated by the library.

### 7.2 The duration gate

```python
@dataclass(frozen=True)
class Thresholds:
    same_seconds: int = 3         # <= this: same recording
    differ_seconds: int = 15      # >  this: a different recording
    fuzzy_floor: float = 0.82     # below this a fuzzy hit is not a candidate
    fuzzy_confident: float = 0.94 # at/above this, fuzzy alone can say OWNED
```

Applied whenever both durations are known:

- `delta ≤ same_seconds` → confirm `OWNED` even across a cosmetic modifier difference.
- `delta > differ_seconds` → force `MISSING`, **even if titles match exactly.** This is what
  catches `Cupid` vs `Cupid (Twin Version)` at 41 s.
- otherwise → `UNSURE`, reason `duration_gap`.

**When the source has no duration** (today's CSV), the gate is skipped and its would-be rows
land in `UNSURE`. This is expected and must not be papered over: the UI says which signals
were unavailable.

### 7.3 Verdict precedence

Uncertainty **never** resolves toward `OWNED`. Concretely, when signals disagree:

```
UNAVAILABLE > IGNORED (ledger) > MISSING > UNSURE > OWNED
```

…except that a rung 0 / 0b identity match outranks everything below `IGNORED`. A test
asserts that no combination of fuzzy score and modifier class can produce `OWNED` when the
duration gate says `MISSING`.

### 7.4 Reason codes

`reason_code` is a stable machine string; `reason_text` is the one-sentence human
explanation. The codes **are** the bulk-triage grouping keys, so they must be coarse enough
to group usefully:

`id_match`, `exact_match`, `mod_cosmetic_only`, `mod_wanted_substantive`,
`mod_candidate_substantive`, `mod_ambiguous`, `mod_different_actor`, `duration_gap`,
`fuzzy_confident`, `fuzzy_weak`, `split_ambiguous`, `mashup_unparsed`, `no_candidate`,
`source_unavailable`.

### 7.5 Scorer

```python
class Scorer(Protocol):
    def ratio(self, a: str, b: str) -> float: ...
```

`DifflibScorer` wraps `difflib.SequenceMatcher` (stdlib, already this project's similarity
tool). `RapidFuzzScorer` is a drop-in for later; the swap is a constructor argument and a
benchmark decision, not a rewrite.

---

## 8. The ledger — `playlist_gap_ledger.py`

SQLite at `<library_root>/Docs/playlist_gap.sqlite3`. SQLite rather than JSON because the
project already uses it (`fingerprint_cache`), the row count is in the thousands per source,
and partial writes during a long triage session must not corrupt prior decisions.

### 8.1 Schema

```sql
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    location    TEXT,
    last_run_at REAL,
    UNIQUE(kind, location)
);

CREATE TABLE IF NOT EXISTS wanted (
    row_id      TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    display     TEXT NOT NULL,
    artist      TEXT, title TEXT, album TEXT,
    duration    INTEGER, isrc TEXT, video_id TEXT,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    removed_at  REAL,                      -- set when it leaves the playlist upstream
    PRIMARY KEY (row_id, source_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    row_id      TEXT PRIMARY KEY,
    verdict     TEXT NOT NULL,             -- Verdict value
    decided_by  TEXT NOT NULL,             -- "auto" | "user" | "verify"
    reason_code TEXT,
    matched_path TEXT,
    note        TEXT,
    decided_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending (
    row_id       TEXT PRIMARY KEY,
    exported_at  REAL NOT NULL,
    verified_at  REAL,
    outcome      TEXT,                     -- VerifyOutcome value
    arrived_path TEXT
);

CREATE TABLE IF NOT EXISTS learned_rules (
    rule_id   TEXT PRIMARY KEY,
    kind      TEXT NOT NULL,               -- "modifier" | "split"
    payload   TEXT NOT NULL,               -- JSON
    hits      INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    source_id  TEXT NOT NULL,
    started_at REAL, finished_at REAL,
    counts     TEXT                        -- JSON snapshot of bucket counts
);
```

Schema upgrades follow `fingerprint_cache._initialize_db`'s idiom: `CREATE TABLE IF NOT
EXISTS` plus `PRAGMA table_info` checks and `ALTER TABLE ADD COLUMN`. Never a destructive
migration — the ledger is the one piece of user work the feature cannot regenerate.

### 8.2 `row_id` — the identity decision

```python
def row_id_for(w: WantedTrack) -> str:
    if w.video_id: return f"yt:{w.video_id}"
    if w.isrc:     return f"isrc:{w.isrc}"
    return "s:" + hashlib.sha1(
        f"{w.source_id}\x00{normalize_for_identity(w.display)}".encode("utf-8")
    ).hexdigest()[:20]
```

`normalize_for_identity` is **deliberately more aggressive** than match normalization (fold
case, strip all punctuation and bracketed segments, collapse whitespace) so that a cosmetic
upstream title edit does not orphan a decision.

**Known limitation, to be stated in the UI, not hidden:** a fallback `s:` row whose display
string is edited substantially upstream produces a new `row_id` and will be re-asked once.
This is the argument for using a source that supplies `video_id`. When a row later gains a
`video_id`, the ledger migrates the `s:` row's decision to the `yt:` row and marks the old
one superseded (`decisions.row_id` update inside one transaction).

### 8.3 API

```python
class Ledger:
    def __init__(self, db_path: str) -> None: ...
    def upsert_source(self, spec: SourceSpec) -> str: ...
    def sync_wanted(self, source_id: str, rows: Sequence[WantedTrack]) -> WantedDiff: ...
    def decision_for(self, row_id: str) -> Decision | None: ...
    def record(self, row_id: str, verdict: Verdict, *, by: str, ...) -> None: ...
    def record_many(self, updates: Sequence[DecisionUpdate]) -> None: ...   # bulk triage
    def mark_exported(self, row_ids: Sequence[str]) -> None: ...
    def mark_verified(self, row_id: str, outcome: VerifyOutcome, path: str | None) -> None: ...
    def reopen(self, row_id: str, reason: str) -> None: ...                 # verify's fix
    def outstanding(self, source_id: str | None = None) -> list[str]: ...
```

`sync_wanted` returns a `WantedDiff` with `added`, `unchanged`, `removed_upstream` — this is
what powers **"14 new since you last checked."**

**Precedence during matching:** a stored `user` decision always overrides a fresh `auto`
verdict; an `IGNORED` row never reappears. A `verify`-written decision overrides `auto` but
not `user`.

---

## 9. Saved sources — config schema

Stored under `config.load_config()["playlist_gap"]`:

```json
{
  "sources": [
    {
      "source_id": "3f2a9c1e",
      "name": "Liked videos",
      "kind": "ytmusic",
      "location": "https://music.youtube.com/playlist?list=LM",
      "options": {"auth_file": "~/.ytmusic_headers.json"},
      "column_mapping": null,
      "library_root": "D:/Music/SoundVault",
      "folder_policy": {"include": ["Not Sorted", "Quarantine", "Manual Review"]},
      "thresholds": {"same_seconds": 3, "differ_seconds": 15, "fuzzy_floor": 0.82},
      "last_run_at": 1757700000.0
    }
  ],
  "active_source_id": "3f2a9c1e",
  "lexicon_overrides": "Docs/playlist_gap_lexicon.json"
}
```

Sources live in **config** (travels with the app); the ledger lives in the **library's
`Docs/`** (travels with the library). They are joined by `source_id`, which is also written
into the ledger's `sources` table — so if the config is lost, the ledger can still list its
sources by stored name and the user can re-point them.

**Run all.** `run_all(specs)` builds **one shared `LibrarySnapshot`** (it is the expensive
part and identical across sources), runs each source against it, then merges the results by
`row_id` so a track wanted by three playlists appears **once** in the download list, with
its `playlists` tuple populated.

---

## 10. Second pass — `playlist_gap_verify.py`

```python
class VerifyOutcome(str, Enum):
    CORRECT       = "correct"
    WRONG_VERSION = "wrong_version"
    ALREADY_OWNED = "already_owned"
    NOT_ARRIVED   = "not_arrived"
    UNRECOGNIZED  = "unrecognized"


def verify_downloads(
    folder: str,
    ledger: Ledger,
    snapshot: LibrarySnapshot,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    fingerprint: Callable[..., tuple] = ...,   # injectable; defaults to
                                               # fingerprint_generator.compute_fingerprint_for_file
    progress: Callable[[int, int, str], None] | None = None,
) -> VerifyReport: ...
```

### 10.1 Why this stage exists

Downloaders fetch the wrong recording often. Ask for `Cupid (Twin Version)`, receive plain
`Cupid` — now there is a duplicate, the Twin Version is still missing, **and the ledger has
marked the row satisfied so it will never be asked about again.** That is a false `OWNED`
entering through the back door, and it corrupts the one mechanism that makes later runs
cheap. Verification is what keeps the ledger honest.

### 10.2 Algorithm

For each audio file in `folder` not already recorded in `pending` with a `verified_at`:

1. **Identity check (no audio decoded).** Extract `video_id` from filename or
   `comment`/`purl`/`website` tag. If it equals the `video_id` of an outstanding `pending`
   row → `CORRECT`. This settles most of a `yt-dlp` folder.
2. **Fingerprint check.** `compute_fingerprint_for_file((path, cache_db, fp_settings))` →
   `(path, duration, fp, error)`. Then, using
   `near_duplicate_detector.fingerprint_distance(fp, other_fp)`:
   - against the **library** snapshot's cached fingerprints — a hit under
     `config.EXACT_DUPLICATE_THRESHOLD` means this file duplicates something already owned;
   - against the **expected** wanted row — compare duration to the wanted row's duration
     using the §7.2 gate.
3. **Classify:**

| Condition | Outcome | Ledger effect |
|---|---|---|
| Identity matches, or fingerprint matches expectation and `delta ≤ same_seconds` | `CORRECT` | `mark_verified`; decision stands `OWNED` |
| Duration/fingerprint indicates a different recording than requested | `WRONG_VERSION` | **`reopen(row_id)` → back to `MISSING`**, and if it also duplicates a library file, stage a quarantine |
| Duplicates a library file and no outstanding row expected it | `ALREADY_OWNED` | stage a quarantine; leave decisions alone |
| An exported row has no arriving file | `NOT_ARRIVED` | stays `MISSING`, stays visible |
| Fingerprint matches nothing known | `UNRECOGNIZED` | accept; offer to index |

4. **Re-runnable and incremental.** Files with a `verified_at` are skipped, so the pass can
   be run repeatedly as downloads trickle in over days.

### 10.3 Actions — and the preview rule

The only library-mutating action is quarantining a duplicate. It **must** go through
`duplicate_consolidation` + `duplicate_consolidation_executor`, which already implement the
project's plan-then-execute contract and the `Docs/` action log. `playlist_gap_verify`
builds the plan and hands it over; it never moves a file itself.

`WRONG_VERSION`'s primary action performs **two** things in one transaction —
quarantine the duplicate *and* `reopen()` the wanted row. Doing only the first is the bug
this stage exists to prevent.

---

## 11. Qt workspace — `gui/workspaces/playlist_gap.py`

### 11.1 Structure

`PlaylistGapWorkspace(WorkspaceBase)` — inherits `log_message`, `status_changed`,
`navigate_requested`, `play_tracks_requested`.

```
titlebar          active-source picker (▾) · ▶ Run
GapTileStrip      7 tiles (Setup · Read · Compare · Triage · Export list · Verify · Summary)
QStackedWidget    one pane per tile
```

Tiles carry live state and act as the navigation — clicking one switches the stack; the
strip owns no business logic, it renders a `RunState` dataclass. The `you download` divider
between Export and Verify marks the pass-1/pass-2 boundary.

### 11.2 Workers

Each is a `QtCore.QThread` subclass at module top, mirroring `library_sync.py`:

| Worker | Signals | Wraps |
|---|---|---|
| `GapFetchWorker` | `progress(int,str)`, `log_line(str)`, `finished(bool,str,object)` | `source.fetch()` |
| `GapSnapshotWorker` | same | `build_snapshot()` |
| `GapMatchWorker` | same | `match_all()` |
| `GapVerifyWorker` | same | `verify_downloads()` |
| `GapExportWorker` | `finished(bool,str,str)` | `playlist_gap_report` writers |

All accept a `should_cancel` callable backed by a `threading.Event`; the workspace sets it
on Cancel and on `closeEvent`. No worker touches a widget — results travel in the `finished`
payload and the main thread renders them.

### 11.3 Triage pane

Three columns: filter chips + row queue (left), evidence (centre), ranked counterparts
(right). The evidence panel shows cover art on both sides — the source's `artwork_url` for
the wanted side, `read_metadata(path, include_cover=True)` for the library side.

Keyboard: `J`/`K` move, `A` owned, `M` missing, `X` never want, `U` undo, `1`–`9` select a
counterpart, `Space` play both (via `play_tracks_requested`).

**Bulk triage** groups the unsure bucket by `reason_code` (§7.4) and offers a single action
per group. Each bulk action is one `ledger.record_many()` call, and offers "remember this as
a rule" which writes a `learned_rules` row.

---

## 12. Reports — `playlist_gap_report.py`

| Format | Contents |
|---|---|
| `.txt` | `Artist - Title`, one per line. The default; it is what gets pasted into a downloader. |
| `.csv` | every field plus `reason_code`, `source name`, `row_id`. |
| `.html` | run report written to `<library>/Docs/`, matching the existing report convention. |
| `.m3u` | optional, for the owned bucket only. |

Grouping modes: flat, by artist, by album. The by-album mode annotates
`(4 of 12 tracks missing)` so whole-album grabs become visible.

---

## 13. Test plan

Following repo convention — `monkeypatch`, `tmp_path`, no base classes, inline stubs for
`mutagen` where a module needs it.

**`test_playlist_gap_parse.py`** — each of the eight real rows from the user's export is a
named test:
`Kate Bush - Running Up That Hill (A Deal With God)` (paren is `TITLE_PART`, survives into
the core); `FIFTY FIFTY - Cupid (Twin Version)` (paren is `SUBSTANTIVE`);
`(Triple Vibe) HOME - Resonance but it's beats 3,3` (prefix lifted, fan-edit tail
classified); `French 79 · New Constellations - Colors Collide` (≥ 2 candidates emitted);
`Self Aware x Babydoll` (**asserts `artist is None`** and that no candidate splits on `x`);
a blank row (`UNAVAILABLE`, not dropped); plus unicode/apostrophe and `The `-prefix cases.

**`test_playlist_gap_lexicon.py`** — the §6.2 adjudication table, row by row. Explicitly:
wanted-has-remix→`MISSING`, candidate-has-remix→`MISSING` (the directional pair),
cosmetic-only→`OWNED`, ambiguous→`UNSURE`.

**`test_playlist_gap_match.py`** — one test per rung; capability-gating (a source without
`duration` skips the gate and lands in `UNSURE`); the precedence invariant of §7.3 as a
property-style test; `split_ambiguous` when two splits hit different tracks; performance
smoke test asserting 1,200 × 8,000 completes under the budget with a fake scorer.

**`test_playlist_gap_snapshot.py`** — **a file under `Not Sorted/` matches** (the
regression that protects the inclusion policy); `Trash/` is excluded; `video_id` extracted
from both a `[id]` filename and a `purl` tag; cache hit avoids calling `read_tags`.

**`test_playlist_gap_ledger.py`** — `sync_wanted` diff (added/removed); user decision
survives a re-run and overrides `auto`; `IGNORED` never resurfaces; `reopen()` flips a
satisfied row back to `MISSING`; `s:`→`yt:` migration preserves the decision; schema upgrade
from a v1 file adds columns without loss.

**`test_playlist_gap_verify.py`** — the five outcomes, with `WRONG_VERSION` asserted to do
**both** things (stage the quarantine **and** reopen the row); re-running skips already
verified files; identity path never calls the fingerprint stub.

**`test_playlist_gap_sources.py`** — CSV column mapping proposal; **an ISRC column that
exists but is empty reports `isrc=False` in capabilities**; blank rows preserved; ytmusicapi
and yt-dlp sources tested against recorded fixture payloads, never the network.

---

## 14. Delivery phases

Each phase ends green and usable.

**Phase 0 — prerequisites (half a day).** `TAG_KEYS` gains `comment`/`purl`/`website`; the
Qt Duplicates workspace passes `normalized_*` to `store_fingerprint()`. Two small fixes that
unblock rungs 0b and 2, both useful independently.
*Acceptance:* a file with a YouTube URL in its comment tag reports that URL through
`read_tags`; a Qt duplicate scan populates the normalized columns.
**Status: done** — `tests/test_playlist_gap_phase0.py`, 23 tests. Note `.opus` has a
*separate* reader (`utils/opus_metadata_reader.py`) with its own `TAG_KEYS`, and since opus
is yt-dlp's default YouTube container it needed the same change; the spec's original manifest
missed it.

**Phase 1 — walking skeleton.** `playlist_gap_types`, `_sources` (CSV only), `_parse`,
`_snapshot`, `_match` (rungs 0/0b/2/5b/6), `_ledger` (schema + `sync_wanted` + `record`),
`_report` (txt/csv). Controller. Workspace with Setup / Triage(list only) / Export panes.
*Acceptance:* import the real TuneMyMusic CSV, produce a missing list, re-run and see zero
new rows.

**Phase 2 — the lexicon and the third bucket.** `_lexicon`, rung 3 adjudication, duration
gate, full evidence panel with ranked counterparts and cover art, HTML report.
*Acceptance:* `Cupid (Twin Version)` lands in `MISSING` with reason `duration_gap`, not
`OWNED`.

**Phase 2.5 — the second pass.** `_verify`, the Verify pane, the quarantine hand-off.
*Acceptance:* a folder containing plain `Cupid` when `Twin Version` was wanted yields
`WRONG_VERSION`, stages a quarantine, and the wanted row returns to `MISSING`.

**Phase 3 — ledger, live.** New-since-last-run, never-want, pending tracking, removed-
upstream reporting, Summary pane.
*Acceptance:* second run of the same playlist asks about only genuinely new rows.

**Phase 3.5 — saved sources.** Config schema, the sources list with per-row run, run-all
with a shared snapshot and merged list.
*Acceptance:* one press re-runs a stored source end to end.

**Phase 4 — scale and learning.** Bulk triage by `reason_code`, `learned_rules` promotion,
lexicon editor, grouped-by-album export.
*Acceptance:* a 68-row unsure bucket clears in four bulk actions plus six individual rows.

**Phase 5 — optional.** `YtMusicSource` / `YtDlpSource` wired in the UI (the backend lands
in Phase 1 behind the protocol), `RapidFuzzScorer` swap if measured to be needed.

---

## 15. Decisions still open

1. **Accent folding default.** Proposed **on**, hand-rolled table, no new dependency.
   Needs a check against the real library for artists whose names are only distinguished by
   an accent.
2. **Duration tolerances** (3 s / 15 s) are starting values and need calibrating against the
   real library before Phase 2 ships.
3. **Whether Compare stays a distinct pane.** It is a sub-5-second operation with no input;
   it may read better as a results panel on Read & split. Cosmetic, decide during Phase 2.
4. **Cover-art caching.** Fetching `artwork_url` per row is a network call per wanted track.
   Proposed: lazy-load only for the row selected in triage, memoised per run. Confirm that
   is acceptable UX before Phase 2.
5. **`ytmusicapi` auth storage.** Its browser-header file path goes in the saved source; the
   file itself stays wherever the user put it and is never copied into the repo, the config,
   or the ledger.
