# AlphaDEX — Authoritative UI Design Spec

**Status:** binding. **Scope:** every page under `gui/workspaces/` in the PySide6 app
(`alpha_dex_gui.py` → `gui/main_window.py`). `main_gui.py` (Tkinter) is out of scope entirely.

This document is written for ~12 agents redesigning ONE page each, in parallel, without talking to
each other. Where two readings are possible, this spec picks one. **If you find yourself choosing,
you have found a bug in this spec — pick the option that matches the majority of already-shipped
pages and note it in your PR body; do not invent a third.**

Page → archetype assignment (this table is the first thing to obey):

| key (`_WORKSPACE_MAP`) | module | archetype |
|---|---|---|
| `indexer` | `gui/workspaces/indexer.py` | **pipeline** (reference page) |
| `duplicates` | `gui/workspaces/duplicates.py` | **pipeline** |
| `library_sync` | `gui/workspaces/library_sync.py` | **pipeline** |
| `compression` | `gui/workspaces/compression.py` | **pipeline** |
| `tag_fixer` | `gui/workspaces/tag_fixer.py` | **pipeline** |
| `genres` | `gui/workspaces/genres.py` | **pipeline** (smallest diff — build first) |
| `playlists` | `gui/workspaces/playlists.py` | **generator** |
| `clustered` | `gui/workspaces/clustered_enhanced.py` (`EnhancedClusteredWorkspace`) | **generator** |
| `similarity` | `gui/workspaces/similarity.py` | **canvas + inspector** |
| `graph` | `gui/workspaces/graph.py` | **canvas + inspector** |
| `player` | `gui/workspaces/player.py` | **player** (full-bleed, the one sanctioned exception) |
| `tools` | `gui/workspaces/tools.py` | **launcher** |
| `help` | `gui/workspaces/help.py` | **reference** |

---

## 1. PRINCIPLES

Six, each decidable by inspection of one page.

1. **One page, one spine.** A page states where the user is with exactly one progress idiom
   (`StageRail` in the header) and exactly one place to act (`RunBar` at the bottom); if a second
   stepper, second progress bar, or second primary button exists, the page fails.
2. **The preview is the promise.** No control that writes to the library may be *rendered* before a
   dry-run plan exists in memory, and every control that is rendered must feed the dict handed to
   the worker — a decorative control is a lie and must be wired or deleted, never restyled.
3. **Red means your bytes change.** Colour encodes consequence, not emphasis: `dangerBtn` iff the
   action deletes/moves/overwrites/rewrites a file the user already has; `primaryBtn` for reads,
   previews and brand-new files; `successBtn` is never an action.
4. **The theme owns every pixel of colour and every point of type.** A page reads
   `self.tokens` and sets object names; it never writes a colour literal and never writes a
   `font-size`, `setPointSize`, or `setFont`.
5. **540 px is the design width.** Every region wraps, stacks, or overflows into a menu at a
   ~540 px content width; nothing clips, nothing forces horizontal scrolling, and
   `scripts/page_smoke.py` proves it.
6. **The view stays a view.** Layout changes never move logic out of the backend, never bypass the
   existing QThread workers, and never rename a signal, slot, or public attribute another module
   already calls.

---

## 2. COMPONENT VOCABULARY

### 2A. ALREADY EXISTS — use it, do not re-create it

Everything below is live **today**. Reaching past it to hand-roll a stylesheet is a review failure.

#### `WorkspaceBase` helpers (`gui/workspaces/base.py` — **read-only for you**)

```python
self.content_layout            # QVBoxLayout inside a QScrollArea (the scrolling column)
self._inner                    # GradientWidget holding content_layout (page_smoke measures this)
self.layout()                  # the OUTER QVBoxLayout: [QScrollArea]. RunBar mounts HERE.
self.tokens                    # live ThemeTokens — read at paint time, never cache
self.on_theme_changed(tokens)  # override hook; called on construction and every theme change
self.refresh_shadows()         # re-applies card shadows (base calls it for you)

self._make_card()                    -> QFrame  (objectName "workspaceCard", shadowed, tracked)
self._make_section_title(text)       -> QLabel  (objectName "sectionTitle")
self._make_subtitle(text)            -> QLabel  (objectName "sectionSubtitle", wordWrap)
self._make_card_title(text)          -> QLabel  (objectName "cardTitle")
self._make_primary_button(text)      -> QPushButton (objectName "primaryBtn", minHeight 34)
self._make_browse_row(label, ph)     -> (QLineEdit, QPushButton "Browse…")
self._log(msg, level="info")         # → log_message signal → the GLOBAL LogDrawer

# Signals you must keep wired, unchanged:
log_message(str, str) · status_changed(str, str) · navigate_requested(str) · play_tracks_requested(list, str)
```

#### Object names (painted by `gui/themes/style.py` + the residual QSS in `gui/themes/manager.py`)

| objectName | What it is | Do NOT use it for |
|---|---|---|
| `workspaceCard` | the only card shell | nesting a card inside a card |
| `headerCard` | page header band (accent left border) | anything below the header |
| `actionCard` | legacy accented action strip | **deprecated — `RunBar` replaces it** |
| `workflowStepper` | frame around the stage rail | a container for random controls |
| `stepActive` / `stepInactive` / `stepArrow` | rail chip inks | body text |
| `sectionTitle` | page title, 18px/700 | card headings |
| `sectionSubtitle` | the page's one-sentence subtitle, 12px | multi-paragraph prose |
| `cardTitle` | card heading, 13px/600 | page title |
| `phaseLabel` / `phaseDesc` | phase name / phase detail | buttons |
| `statusHint` | status and secondary inks, 12px | anything clickable |
| `notesHint` | inline hints, 11px | status |
| `logText` | monospace log body | non-log text |
| `primaryBtn` `dangerBtn` `successBtn` `secondaryBtn` `ghostBtn` `segmentBtn`/`toggleBtn` | see §4.5 | see §4.5 |

`secondaryBtn`, `ghostBtn`, `segmentBtn`/`toggleBtn` landed this session and are usable now
(`gui/themes/style.py:240-246`, `:220-228`). `segmentBtn`/`toggleBtn` require `setCheckable(True)`
— the painter fills with `accent` + `text_inverse` only when checked.

#### Geometry scales (`gui/themes/effects.py`)

```python
from gui.themes.effects import R, S, card_shadow
R.button 8 · R.input 7 · R.card 12 · R.tab 6 · R.checkbox 4 · R.nav_item 8
S.xs 4 · S.sm 8 · S.md 12 · S.lg 16 · S.xl 20 · S.xxl 24
S.card_padding 16 · S.card_gap 20 · S.row_gap 12 · S.inline_gap 8 · S.page_margin 24
```

#### Type scale (`gui/fonts/loader.py`)

`TypeScale` (pt) is authoritative **inside the theme engine only**. Page code never imports it.
See §5.

#### The global log drawer

`gui/main_window.py:115` builds one `LogDrawer`; `:133` connects **every** workspace's
`log_message`. **Delete every per-page log card** (`library_sync.py:475`, `genres.py:232`,
`tag_fixer.py:223`, and siblings). Emit `self._log(...)` instead.

---

### 2B. NEW — must be built first, exactly as specified

All new widgets live in `gui/widgets/` as **new files**. They are pure view code: no backend
imports, no business logic, colours from `self._tokens` (passed in or read via
`get_manager().current`) at paint time.

**Parallel-safety rule:** if the file already exists, **use it and do not change its API**. If it
does not, create it with exactly the signatures below. Never fork a variant
(`StageCard2`, `MyStageRail`); never widen an API to suit one page.

Ownership: components 1–7 are needed by the pipeline and generator pages; 8–11 by
launcher/player/canvas. Build in the order listed.

---

#### 1. `PageHeader` — `gui/widgets/page_header.py`

**Kills the three header patterns.** Every one of the 13 pages starts with this and nothing else.

```python
class PageHeader(QtWidgets.QFrame):
    def __init__(self, title: str, subtitle: str, *, flush: bool = False,
                 parent: QtWidgets.QWidget | None = None) -> None: ...
    def set_subtitle(self, text: str) -> None: ...
    def set_rail(self, rail: QtWidgets.QWidget) -> None:   # StageRail, optional
    def set_status(self, pill: QtWidgets.QWidget) -> None: # StatusPill, right-aligned
```

* `flush=False` (default, 12 pages): `objectName("headerCard")`, mounted as the first item of
  `content_layout`.
* `flush=True` (Player only): square, shadowless, `content_bg` fill, 1 px `card_border` bottom
  hairline, **left inset exactly `S.page_margin`** so the title's left edge is pixel-identical to
  the card pages.
* Internals in both variants: `sectionTitle`, then `sectionSubtitle` (**mandatory, exactly one
  sentence, never empty**), then optional rail row, with the status pill right-aligned on the
  title row.
* **Never** use for: a second header, a card heading, or a place to park buttons.

```python
self._header = PageHeader("Tag Fixer", "Look up canonical tags, review every change, then write.")
self._header.set_rail(self._rail)
self._header.set_status(self._pill)
self.content_layout.addWidget(self._header)
```

---

#### 2. `StageRail` — `gui/widgets/stage_rail.py`

The app's **only** progress-position idiom. Replaces the hardcoded steppers
(`indexer.py:110-120`, `library_sync.py:212`), Genres' numbered card headings, and the Clustering
Wizard's "Step 1 … 1 of 5" counter with its off-palette green bar.

```python
class StageState(enum.Enum):
    PENDING = 0; CURRENT = 1; DONE = 2; BLOCKED = 3; ERROR = 4

class StageRail(QtWidgets.QFrame):          # objectName "workflowStepper"
    stage_clicked = Signal(int)             # emitted ONLY for DONE/CURRENT stages
    def set_stages(self, names: list[str]) -> None: ...
    def set_state(self, index: int, state: StageState) -> None: ...
    def set_compact(self, on: bool) -> None: ...   # auto-switches below 700px in resizeEvent
```

* Chip inks: `CURRENT` → `stepActive`; `PENDING` → `stepInactive`; separators → `stepArrow`;
  `DONE` / `BLOCKED` / `ERROR` marks are **painted inside this widget** with `QPainter` using
  `self.tokens.success` / `.text_muted` / `.danger`, or rendered as themed `QLabel` text.
  **No new QSS object names, and no new primitives in `gui/themes/style.py`** — that file paints
  all 14 themes and is the highest-blast-radius file in the app.
* Compact mode (<700 px content width): one line `Step 2 of 4 · Preview` above a 4-segment track.
* Not in the tab order (`setFocusPolicy(NoFocus)`). Unreached chips are inert.
* **Never** use for: tab navigation, a filter bar, or a rail whose stages the user can skip.

---

#### 3. `StageCard` — `gui/widgets/stage_card.py`

One component covers both "numbered pipeline stage" and "collapsible section / advanced
disclosure". There is no separate `CollapsibleStrip` and no separate `DisclosureSection`.

```python
class CardState(enum.Enum):
    LOCKED = 0; OPEN = 1; RUNNING = 2; COLLAPSED = 3

class StageCard(QtWidgets.QFrame):          # objectName "workspaceCard"
    edit_requested = Signal()               # the COLLAPSED state's ghost "Edit"
    toggled = Signal(bool)                  # collapsible (number=None) use
    def __init__(self, title: str, *, number: int | None = None,
                 collapsible: bool = False, parent=None) -> None: ...
    def set_body(self, w: QtWidgets.QWidget) -> None: ...
    def set_footer(self, widgets: list[QtWidgets.QWidget]) -> None: ...
    def set_state(self, state: CardState) -> None: ...
    def set_summary(self, text: str) -> None: ...   # the COLLAPSED one-liner
    def set_locked_reason(self, text: str) -> None: # the LOCKED one-liner
```

* `number=2` renders the title as `"2 · Preview"` — **MIDDOT U+00B7, spaces either side**, never
  `2.` and never `2)`. Genres already ships this glyph; it is the house style.
* `LOCKED`: muted title, body is one line of prerequisite text, **zero buttons rendered** (not
  disabled buttons — a disabled button invites clicking).
* `RUNNING`: **`setReadOnly(True)` on the inputs and `setEnabled(False)` on the buttons only.**
  Never `setEnabled(False)` on the body container: `effects.build_palette` lerps the Disabled group
  55 % toward the background, so a disabled form greys out exactly the settings the user needs to
  read while the job runs.
* `COLLAPSED`: a one-line summary plus a `ghostBtn` "Edit". This *is* the strip.
* Internal layout: `setContentsMargins(*[S.card_padding]*4)`, `setSpacing(S.row_gap)`.
* Footer row: `setSpacing(S.inline_gap)`; if the row's `sizeHint().width()` exceeds the card width,
  stack the buttons full-width, primary on top.
* **Never** use for: a page header, a tab, or wrapping another `StageCard`.

---

#### 4. `PhaseProgressList` — `gui/widgets/phase_progress.py`

Extracted from `indexer.py:210-231`. The **only** progress surface on a page. Deletes the duplicate
bars at `library_sync.py:325` and on Genres.

```python
class PhaseProgressList(QtWidgets.QWidget):
    def set_phases(self, phases: list[tuple[str, str]]) -> None:   # (name, one-line description)
    def set_progress(self, index: int, pct: int, msg: str = "") -> None: ...
    def mark_done(self, index: int, summary: str = "") -> None: ...
    def reset(self) -> None: ...
```

* Row = `phaseLabel` name, `phaseDesc` detail, `QProgressBar` **beneath** the labels (label-above-bar
  survives 540 px). Progress colour comes from `progress_fg`; **no green bar on a blue app.**
* Instantiated at most twice per page: preview (stage 2) and execution (stage 4).
* Phase names per page — **use these strings verbatim**:
  * Indexer: `Scan` / `Plan` / `Cross-album`  ← this is where "Phase A/B/C" goes
  * Duplicates: `Fingerprint` / `Group`
  * Library Sync: `Scan incoming` / `Scan existing` / `Match`
  * Compression: `Probe` / `Transcode`
  * Tag Fixer: `Fingerprint` / `Look up`
  * Genres: `Collect` / `Diff`
* **Never** use for: an indeterminate spinner with no phases (use `RunBar`'s own bar), or for
  per-file logging (that is `self._log`).

---

#### 5. `CountStrip` — `gui/widgets/count_strip.py`

Merges the pipeline's "ChangeSummaryBar" and the generator's "ResultSummaryCard": one wrapping row
of labelled counts, used for both "what the plan will do" and "what the run produced".

```python
class CountStrip(QtWidgets.QWidget):
    def set_counts(self, items: list[tuple[str, int, bool]]) -> None:
        """(label, value, emphasise) — emphasise=True inks the value with tokens.danger
        and is reserved for irreversible quantities (deleted, overwritten)."""
    def clear(self) -> None: ...
```

* Wrapping flow layout **by construction** — it must reflow, never elide, at 540 px.
* Values in `text_primary`, labels in `statusHint`. Emphasised values in `danger`.
* On cancel/failure: **keep the partial counts and add "Cancelled after N of M"**. Never clear.
* **Never** use for: a table replacement, or for a single status sentence (that is `StatusPill`).

---

#### 6. `CommitGate` — `gui/widgets/commit_gate.py`

Encodes preview-first in a widget. **It contains no button** — the commit button lives in `RunBar`
(§7) so "exactly one action slot per page" is structural rather than aspirational.

```python
class CommitGate(QtWidgets.QWidget):
    armed_changed = Signal(bool)
    def set_consequences(self, lines: list[str]) -> None:   # 2–4 plain-language effects
    def set_unrecoverable(self, on: bool) -> None: ...      # see the modal rule below
    def is_armed(self) -> bool: ...
    def reset(self) -> None: ...                            # unchecks, clears
```

* Consequence lines are **generated from the actual plan**, never hardcoded prose:
  `"12 files move to Quarantine/ — recoverable from there"`, `"0 files are deleted"`,
  `"Every action is logged to Docs/"`.
* Arm checkbox: `"I have reviewed this plan"`. It sits **outside the plan table's tab chain**
  (`setTabOrder` explicitly) so a plan cannot be armed by blind tabbing, and it
  **resets to unchecked whenever the plan, the selection, or any input changes**.
* `set_unrecoverable(True)` (a hard delete, not a move to `Quarantine/`) makes the page additionally
  raise one `QMessageBox` whose text states the exact count and whose **default button is Cancel**.
  Quarantine-only and additive plans get **no modal at all** — that is what keeps the modal
  meaningful.
* **Never** use for: confirming a preview, confirming an export, or as a generic checkbox.

---

#### 7. `RunBar` — `gui/widgets/run_bar.py`

The page's **one** action bar. Sticky, non-scrolling, mounted **outside** the scroll area:

```python
self.layout().addWidget(self._run_bar)   # self.layout() is the outer QVBoxLayout from _setup_scroll
```

This needs **no edit to `base.py`**.

```python
class RunBar(QtWidgets.QFrame):
    primary_clicked = Signal()
    cancel_clicked  = Signal()
    def set_action(self, label: str, variant: str) -> None:
        """variant ∈ {"primaryBtn", "dangerBtn"} — chosen by the §4.4 consequence rule."""
    def set_blocked(self, reason: str | None) -> None:
        """reason=None enables the button. A string disables it AND prints the reason
        beside it. A blocked primary is NEVER clickable-then-scolded."""
    def set_running(self, on: bool) -> None: ...        # swaps in Cancel (secondaryBtn) + bar
    def set_progress(self, pct: int, text: str = "") -> None: ...
    def set_secondary(self, widgets: list[QtWidgets.QWidget]) -> None:  # ghostBtn only, left side
```

* Exactly one primary slot. Label is **verb + count** when a count is known
  (`"Quarantine 12 duplicates"`, `"Write tags to 38 files"`), otherwise verb only
  (`"Run preview"`).
* Cancel is **`secondaryBtn`, never `dangerBtn`** — stopping work is not destruction.
* `set_blocked` deletes the post-click scolding dialogs: `playlists.py:334`'s
  `QMessageBox.warning("No Library")` and Clustered's "Please configure with the wizard first."
* Styling: `card_bg` fill, 1 px `card_border` top hairline, **no radius**, margins
  `(S.lg, S.sm, S.lg, S.sm)`, spacing `S.inline_gap`. Applied in the widget's own
  `on_theme_changed` from tokens — no literals.
* **Never** use for: navigation, filters, more than one non-ghost button, or on pages that run no
  job (Help, Tools, Player have no `RunBar`).

---

#### 8. `StatusPill` — `gui/widgets/status_pill.py`

Promoted from `tools.py::_GlassResultChip`, **tokenised on promotion**: `#94a3b8` → `text_muted`,
`#22c55e` → `success`, `#ef4444` → `danger`, resolved at paint time.

```python
class StatusPill(QtWidgets.QWidget):
    def show_neutral(self, text: str) -> None: ...   # "Ready." / "No library selected"
    def show_success(self, text: str) -> None: ...   # "Executed · 38 moved, 0 errors"
    def show_error(self, text: str) -> None: ...
```

Keeps the 160 ms out / 260 ms in crossfade. One per page, right-aligned in `PageHeader`; one per
tile on the launcher page. **Never** as a button, and never more than one per header.

---

#### 9. `IconBadge` — `gui/widgets/icon_badge.py`

Promoted from `tools.py::_GlassBadge`. Accent-tinted rounded badge holding one icon, for card and
tile iconography. **Delete the literal `QColor(255,255,255,α)` specular/rim gradients on promotion**
— they become a milk film on light themes. Fill from `accent` at low alpha computed from tokens.

#### 10. `TileGrid` — `gui/widgets/tile_grid.py`

Promoted from `tools.py`. Width-driven reflow; `add_tile(w, span_full=False)`; full-width tiles span
all columns and start a new row; gaps `S.card_gap`. Launcher only (today).
**Fix on promotion:** the per-tile 16 ms `QTimer` must start on `enterEvent` and stop on
`leaveEvent` — today five tiles repaint the page at 60 fps forever.

#### 11. `OverflowActionButton` — `gui/widgets/overflow_button.py`

```python
class OverflowActionButton(QtWidgets.QPushButton):   # objectName "ghostBtn"
    def __init__(self, actions: list[QtGui.QAction], threshold_px: int, parent=None) -> None: ...
```

Shows `⋯` (tooltip "More actions") and swallows its actions into a `QMenu` when the owning row is
narrower than `threshold_px`. Required by the Player toolbar and any canvas control strip.

---

## 3. PAGE ANATOMY PER ARCHETYPE

Region order is **fixed**. A region marked *required* must exist even when empty (it renders its
empty state, §4.9).

### 3.1 Pipeline — `indexer`, `duplicates`, `library_sync`, `compression`, `tag_fixer`, `genres`

Six of thirteen pages; this is the app's default shape. **One scrolling column** inside
`content_layout` plus one sticky `RunBar`. No page-level `QSplitter` spanning regions, no tabs, no
side columns.

| # | Region | Required? | Contents |
|---|---|---|---|
| 1 | `PageHeader` | required | title, one-sentence subtitle, `StageRail(["1 · Configure","2 · Preview","3 · Review","4 · Execute"])`, `StatusPill` |
| 2 | `StageCard(number=1, "Configure")` | required | a one-line "what this does" preamble, then **every input the page has** |
| 3 | `StageCard(number=2, "Preview")` | required | the page's only preview `PhaseProgressList`; footer when done = ghost "Open HTML report" + ghost "Re-run preview" |
| 4 | `StageCard(number=3, "Review")` | required | `CountStrip`, then the page's own plan view, then selection controls. **No footer buttons — nothing here writes.** |
| 5 | `StageCard(number=4, "Execute")` | required | `CommitGate` only, plus the execution `PhaseProgressList` while writing |
| 6 | `RunBar` | required | the page's single action slot (see §4.4 for its variant) |

Rules:

* **Rail wording is byte-identical on all six pages:** `1 · Configure`, `2 · Preview`,
  `3 · Review`, `4 · Execute`. Do not localise, abbreviate, or re-verb them.
* **Exactly one action bar, directly below the stage column, never split.** Library Sync's split
  action rows (`:290-293` and `:416-438`) collapse into it; Build Plan / Preview Plan / Recompute
  Matches become one "Run preview"; Export Report and Save Session become `ghostBtn`s in
  `RunBar.set_secondary(...)`.
* Stage 1's action label is `"Run preview"` on all six pages. Stage 4's is verb + count.
* Exactly one stage is `OPEN` at a time. Completing stage N collapses it and opens N+1. Reached
  stages reopen on rail click or the collapsed card's ghost "Edit".
* **Editing stage 1 after a preview invalidates 2–4**: plan cleared, rail stages reset to
  `PENDING`, `CommitGate.reset()`. Warn inline inside stage 1 (`notesHint`:
  "Editing clears the current plan"). **Never a modal for this.**
* Commit enablement is a conjunction re-evaluated on every relevant signal: plan non-empty AND ≥1
  item selected AND armed AND no worker running AND inputs unchanged since the preview. A failing
  condition goes to `RunBar.set_blocked("<the specific reason>")`.
* One job per page: while a worker runs, every control outside the `RUNNING` card is disabled and
  Cancel is the only live control.
* Keyboard: `Ctrl+Return` = the open stage's action, `F5` = re-run preview, `Esc` = cancel (same
  rules as the Cancel button), `Alt+1..4` = jump to a reached stage. Opening a stage focuses its
  first control. **No shortcut ever fires a `dangerBtn` action.**
* Cancel during preview is immediate. Cancel during execution asks once and states that files
  already written stay written and are listed in the report.
* Library Sync keeps its incoming/existing `QSplitter` and Match Inspector **intact inside stage 3**
  — it is the best review surface in the app. Below 700 px the splitter flips to vertical and the
  Match Inspector becomes a third stacked section. Nine cards become four.

### 3.2 Generator — `playlists`, `clustered`

Re-run by a returning user; the expensive stage *is* the result. **Tabs are deleted from both
files; `QTabWidget` must not appear in either.**

| # | Region | Required? | Contents |
|---|---|---|---|
| 1 | `PageHeader` | required | title, subtitle, `StatusPill`. **No `StageRail`** — these are not pipelines |
| 2 | Recipe chooser | required | exclusive `QButtonGroup` of checkable `segmentBtn`s in a wrapping row, inside a `StageCard(collapsible=False)` titled "What to build" |
| 3 | Essentials form | required | at most 4 rows — the things that change between runs (e.g. Clustered: "How many groups?") |
| 4 | `StageCard(collapsible=True, "Advanced")` | optional | everything else. Its header carries a summary suffix of current values: `"8 groups · K-Means · standard normalisation"`. Collapsed by default |
| 5 | Output region | required | `CountStrip` + the result list/table. Revealed on first run; **never cleared on cancel or failure** |
| 6 | `RunBar` | required | one action slot, `set_blocked` reasons, always-visible progress |

Rules:

* **`StageRail` appears on exactly one recipe: Repair**, the only one that rewrites existing user
  `.m3u` files — a two-stage rail `["1 · Check", "2 · Apply"]`. The rail's *presence* is the
  overwrite signal. Do not paint a rail on additive generators; that would be a lie.
* After a run, the configure cards collapse to their one-line summaries so the output owns the
  viewport; ghost "Edit" reopens them.
* Clustered must show the feature-cache preflight fact in the header's `StatusPill`:
  `"features cached for this selection — about 20 seconds"` vs
  `"no cache — expect a full extraction pass"`. `feature_cache_name` is keyed by selection and
  engine, so this is cheap and truthful.
* Changing the recipe or editing config after a Repair check discards the preview, resets the rail
  to stage 1, and says so inline.
* The decorative controls (Playlists' tempo/energy range fields, "Prefer Opus") are **wired or
  deleted** in this pass. A rendered control whose value never reaches the worker dict fails review.

### 3.3 Canvas + inspector — `similarity`, `graph`

| # | Region | Required? | Contents |
|---|---|---|---|
| 1 | `PageHeader` | required | title, subtitle, `StatusPill` |
| 2 | Control strip | required | ONE row: `segmentBtn`s for modes, `ghostBtn`s for utilities, `OverflowActionButton` below ~760 px. Not a card |
| 3 | `QSplitter`: canvas \| inspector | required | canvas stretch, `minimumWidth 260`; inspector 240–340, `setChildrenCollapsible(False)` |
| 4 | `RunBar` | only if the page runs a job | otherwise omitted entirely |
| 5 | Status strip | required | one `statusHint` line, flush to the bottom |

Rules: below ~560 px content width, hide the inspector and surface it as a checkable `segmentBtn`
in the control strip rather than squeezing the canvas below 260 px. The canvas keeps its own
internal pan/zoom; it never propagates horizontal scroll to the page.

### 3.4 Player — `player` (the one full-bleed page)

Keep the `_setup_scroll()` override with `contentsMargins(0,0,0,0)` and spacing 0. **Do not
card-ify the Player, and no other page may copy this.**

| # | Region | Required? |
|---|---|---|
| 1 | `PageHeader(..., flush=True)` | required — replaces the ad-hoc 50 px toolbar title |
| 2 | Toolbar: search (stretch 2) + scan progress + `ghostBtn` utilities + `segmentBtn` modes | required |
| 3 | `QSplitter`: library table \| now-playing + queue rail | required |
| 4 | `transportBar`: seek row above control row, pinned, never scrolls, never reorders | required |
| 5 | Status strip: one `statusHint` line flush to the bottom | required |

Rules:

* Delete `title_lbl.setFixedWidth(68)` — it clips "Player" under wider theme fonts. Replace the
  fixed 50 px toolbar height with a `minimumHeight`.
* Insets from `S`: toolbar `(S.lg, S.sm, S.lg, S.sm)`; right rail `(S.md,)*4` spacing `S.sm`;
  transport `(S.lg, S.sm, S.lg, S.xs)`; splitter handle gap `S.xs`; status `S.xs`/`S.lg`.
* Below ~760 px content width, collapse Reload / Browse… / Load M3U into one
  `OverflowActionButton`; keep search (`minimumWidth 140`) and the 30 s Preview toggle inline.
* Below ~560 px, hide the right rail behind a checkable `segmentBtn` "Queue"; `_lib_table`
  `minimumWidth 260`.
* Variants: play/pause = `primaryBtn` (the page's one primary); prev/next/stop = `ghostBtn`;
  shuffle, repeat and 30 s Preview = checkable `segmentBtn` (the 30 s Preview `QCheckBox` becomes a
  `segmentBtn` — one shape for one idea); every other utility = `ghostBtn`. **No `dangerBtn` and no
  `successBtn` on this page** — nothing here writes to the library.
* `transportBar` already carries that objectName but the QSS layer has no rule for it. Because
  `gui/themes/**` is off-limits, style it in the page's `on_theme_changed` from tokens
  (`card_bg` fill, top hairline `card_border`, no radius) — **colour only, never a font-size.**
* **External contract is frozen:** `now_playing_changed`, `playback_state_changed`,
  `position_changed`, `_on_play_pause`, `play_next`, `play_prev`, `set_volume`, `seek_to_ms`,
  `load_tracks_and_play`, every `kb_*`, and `_vol_slider`. Keep `sliderMoved`/`sliderReleased` on
  the volume slider. If a layout change appears to require renaming a contracted member, the
  layout change is wrong.
* Keep: the `_BgArtWidget` translucent-table treatment, the 1.5 s dwell hover art popup, the
  splitter's `setChildrenCollapsible(False)` and 720/280 defaults, and the region order.

### 3.5 Launcher — `tools`

| # | Region | Required? |
|---|---|---|
| 1 | `PageHeader` ("Export & Utilities" + the existing Docs/ sentence) | required |
| 2 | `TileGrid` of tiles | required |
| — | **no page-level `RunBar`** | each tile owns its own action, `StatusPill`, and collapsible log `StageCard` |

Rules: promote `_GlassResultChip` → `StatusPill`, `_GlassBadge` → `IconBadge`, the grid → `TileGrid`
(§2B.8–10). **Delete** the diagnostics buttons' label-above-icon inversion (a `QVBoxLayout` stuffed
in a `QPushButton` with `WA_TransparentForMouseEvents`) and the private `_make_chip` stylesheet —
`segmentBtn` already paints that. Fix the three colour violations named in §6. The iOS pill toggle
survives **only** for a persistent on/off mode with an immediate effect; any other boolean is a
`QCheckBox` or a `segmentBtn`.

### 3.6 Reference — `help`

| # | Region | Required? |
|---|---|---|
| 1 | `PageHeader` | required — Help currently has **no subtitle**; add one sentence |
| 2 | Content | required — `QTabWidget` permitted **here and nowhere else** (§4.12); strip emoji from the tab labels |
| — | no `RunBar`, no rail, no `StatusPill` | |

---

## 4. RULES THAT SETTLE TODAY'S INCONSISTENCIES

Each rule states the **REJECTED** alternative so it cannot be relitigated.

### 4.1 Page header
**RULE:** every page's first widget is `PageHeader` (§2B.1) with a mandatory one-sentence subtitle;
`flush=True` only on Player.
**REJECTED:** the bare `sectionTitle` on the background (8 pages today); Player's inline app
toolbar title; Help having no subtitle; any page inventing a fourth header.

### 4.2 The one progress/stage idiom
**RULE:** position in a multi-step job is shown **only** by `StageRail` in the header; work inside a
step is shown **only** by `PhaseProgressList` inside the running `StageCard`; overall run progress
is shown **only** by `RunBar`'s bar. Maximum per page: one rail, two `PhaseProgressList` instances,
one `RunBar` bar.
**REJECTED:** numbered card headings as the progress signal (Genres); the wizard's "Step 1 … 1 of 5"
counter and its green bar; a second progress card (`library_sync.py:325`); any page-level stepper on
a generator page.

### 4.3 Phase A/B/C
**RULE:** "Phase A/B/C" ceases to exist as user-visible wording. Indexer's three phases become the
`PhaseProgressList` rows `Scan` / `Plan` / `Cross-album` **inside stage 2**, i.e. sub-steps of
Preview, never a rival numbering.
**REJECTED:** keeping "Phase A" text anywhere in a label; renaming the rail to A/B/C; mapping A/B/C
onto the rail's 1/2/3.

### 4.4 Commit vs destructive colour — the Duplicates-red vs Tag-Fixer-green settlement
**RULE**, app-wide, decided by consequence and nothing else:

| Variant | Used when | Examples |
|---|---|---|
| `dangerBtn` | the action **deletes, moves, overwrites, or rewrites bytes the user already has** | Duplicates Execute (quarantine/delete), Library Sync Execute, Indexer Execute (renames/moves), Genres Apply changes, **Tag Fixer Apply Selected**, Compression transcode-in-place, Playlist Repair, Auto-DJ overwriting `Playlists/autodj.m3u` |
| `primaryBtn` | the action **reads, previews, or creates only new files** | Run preview, Generate playlists, Export report, Recompute matches |
| `successBtn` | **never an action.** Completed-state ink only | the pill/summary after a finished run |
| `secondaryBtn` | Cancel, Close, Back | Cancel during a run |
| `ghostBtn` | optional utilities | Open HTML report, Save session, Edit |

**Tag Fixer `:162` changes from `successBtn` to `dangerBtn`** — it writes into the user's existing
files. Duplicates stays red. This is not a style preference; it is the rule.
**REJECTED:** green-means-commit; red-means-emphasis; red for Cancel; colouring by role
("commit is always red") regardless of whether files are overwritten — that would paint Repair and a
folder scan identically.

### 4.5 Button hierarchy — exactly when each variant is used
**RULE:**
* **At most one `primaryBtn`-or-`dangerBtn` per page, and it lives in `RunBar`.** Pages without a
  `RunBar` (Player) may have one `primaryBtn` for their single defining control (play/pause).
* `secondaryBtn` — Cancel / Close / Back / "Choose different folder". Card-coloured with a border.
* `ghostBtn` — chromeless utilities: Browse…, Open report, Export, Edit, overflow `⋯`.
* `segmentBtn`/`toggleBtn` — **and only these** — for any checkable mode, view switch, or exclusive
  choice; always `setCheckable(True)`, exclusive sets in a `QButtonGroup`.
* Variant is set by `setObjectName` only. `setStyleSheet` on a button is a review failure.

**REJECTED:** four `_make_primary_button` calls on one page (Clustered today); an unstyled
`QPushButton` left as "the leftover"; expressing the same mode twice (a `QCheckBox` next to a
fill-when-on button on one strip); hand-rolled chip stylesheets.

### 4.6 Emoji in label strings
**RULE:** no emoji in any label, button, tab, or card title string. Delete them:
`"📄 Preview"`, `"▶ Execute"` (`duplicates.py:361/365`), `"✓ Apply Selected"`
(`tag_fixer.py:161`), `"▶ Start Compression"` (`compression.py:142`), `"✎ Apply changes"`
(`genres.py:352`), `"📋 Build Plan"` / `"👁 Preview Plan"` / `"▶ Execute Plan"` /
`"📤 Export Report"` / `"💾 Save Session"` / `"↺ Recompute Matches"` (`library_sync.py:290-438`),
`"⟳  Reload"`, `"💾  Save Queue as M3U…"`, and every `"✕ Cancel"`. Labels become words.

**The complete permitted-glyph whitelist** (text-presentation characters that take the theme's
button ink, not the emoji font's palette):
`⏮ ⏭ ▶ ■ ↻ ⤨` (Player transport only) · `⋯` (overflow) · `▾ ▸` (disclosure chevrons) ·
`·` (U+00B7, stage numbering). `✕` only on an icon-only close button with **no** accompanying text.
`tokens.icon` (the theme swatch emoji) is data, not a label — leave it.
**REJECTED:** "emoji make it friendly"; emoji tab labels on Clustered; swapping the transport glyphs
for emoji or for icon files.

### 4.7 Icons and icon-vs-label order
**RULE:** icon **left**, label **right**, always, with `S.xs` between them — via
`QPushButton.setIcon()` / `setIconSize()`, or `IconBadge` for card and tile iconography. An
icon-only button must set both `toolTip` and `accessibleName`.
**REJECTED:** label-above-icon (delete `tools.py`'s diagnostics inversion); a `QVBoxLayout` inside a
`QPushButton`; `WA_TransparentForMouseEvents` child hacks; emoji as an icon.

### 4.8 Form-row label alignment
**RULE:** every input group is a `QFormLayout` with:

```python
form = QtWidgets.QFormLayout()
form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
form.setHorizontalSpacing(S.inline_gap)
form.setVerticalSpacing(S.row_gap)
```

Label text is **sentence case with no trailing colon** ("Library folder", not "Library Folder:").
**REJECTED:** hand-built `QGridLayout` label columns; left-aligned labels on some pages and right on
others; trailing colons; Title Case.

### 4.9 Empty state
**RULE:** one `statusHint` line in the region where the content will appear, phrased
`"Nothing here yet — <the one thing to do>."` (e.g. `"No plan yet — run a preview."`), plus at most
one `ghostBtn`. Tables and trees get this via a placeholder label in a `QStackedWidget` swapped on
`rowCount()`. A locked `StageCard` uses `set_locked_reason(...)` instead.
**REJECTED:** an empty card with a title and nothing else; illustrations; disabled buttons standing
in for an explanation; a `QMessageBox` telling the user the list is empty.

### 4.10 Where status text lives
**RULE:** exactly two places. (1) `StatusPill` in `PageHeader` — the page's live state: the blocking
prerequisite when idle, elapsed + live counter while running, the outcome when finished
(`"Executed · 38 moved, 12 quarantined, 0 errors"`). (2) `RunBar.set_blocked(reason)` — why the
action is unavailable. Per-line detail goes to the **global** `LogDrawer` via `self._log(...)`.
**Critically: the rail stops at 3 when the user previewed but never executed — a page never claims
work it did not do.**
**REJECTED:** per-card status labels; a per-page log card (delete `library_sync.py:475`,
`genres.py:232`, `tag_fixer.py:223`); a second status line at the bottom of a pipeline page;
`QMessageBox` as a status channel.

### 4.11 Card titling
**RULE:** every card has exactly **one** `cardTitle`, sentence case, no trailing colon, no emoji.
Pipeline stage cards are titled `"N · Verb"` with MIDDOT U+00B7 (`"1 · Configure"`). Non-stage cards
take a noun phrase (`"What to build"`, `"Advanced"`). **No untitled cards** — retitle or absorb
Library Sync's two untitled cards. **No card inside a card.**
**REJECTED:** `"1. Configure"`, `"1) Configure"`, `"Step 1: Configure"`; a title plus a redundant
subtitle inside the same card; `actionCard`/`headerCard` used as generic containers.

### 4.12 Tabs vs segmented control
**RULE:** `QTabWidget` is permitted **only** on `help`. Everywhere else, a set of alternatives is an
exclusive row of checkable `segmentBtn`s, and progressive detail is a
`StageCard(collapsible=True)`.
**REJECTED:** tabs on Clustered and Playlists — and specifically the
`clustered_enhanced.py:444` `self._tab_widget.setCurrentIndex(2)` that yanks the user out of the tab
they were configuring; tabs as a stand-in for stages; a tab bar plus a rail on one page.

### 4.13 Decorative controls
**RULE:** a control may be rendered only if its value reaches the dict handed to the worker. In this
pass: Duplicates' per-group disposition combo, Playlists' tempo/energy range fields, and "Prefer
Opus" are **wired or removed**.
**REJECTED:** restyling a dead control; leaving it with a "not yet implemented" tooltip; deferring it
to a later pass.

---

## 5. TYPE AND SPACING

### 5.1 px vs pt — resolved

**AUTHORITATIVE MECHANISM: the QSS layer keyed by `objectName`** (the residual QSS block in
`gui/themes/manager.py`, applied per theme). A widget-level stylesheet beats `polish()`, which is
exactly why the pt `TypeScale` is currently dead — and `gui/themes/**` is off-limits to page agents.

Therefore: **page code sets NO font size, weight, or family. Ever.** You pick a role by setting an
object name. `TypeScale` (pt) remains authoritative *inside* the theme engine; page code never
imports it.

| Role | objectName | Rendered (from the QSS layer today) |
|---|---|---|
| Page title | `sectionTitle` | 18 px / 700 |
| Page subtitle | `sectionSubtitle` | 12 px |
| Card / stage title | `cardTitle` | 13 px / 600 |
| Phase name | `phaseLabel` | 12 px / 600 |
| Phase detail | `phaseDesc` | 11 px |
| Status, secondary, counts label | `statusHint` | 12 px |
| Inline hint / caveat | `notesHint` | 11 px |
| Rail chips | `stepActive` / `stepInactive` / `stepArrow` | 12 px |
| Log / monospace | `logText` | theme-owned mono |
| Body, inputs, buttons | *(no objectName)* | inherited app font |

The 68 hardcoded `font-size` rules across 10 values die here. Concretely, delete the sizes at
`compression.py:120/171/180`, `tag_fixer.py:192/218`, Player's `_apply_theme()` 11 px labels,
`_np_title`'s `13px; font-weight:600` (→ `cardTitle`), `vol_lbl`'s 12 px (→ `statusHint`), and every
`font-family: 'Consolas'; font-size: 12px` log rule (→ `logText`). Player's
`_np_artist`, `_q_count_lbl`, `_pos_lbl`, `_dur_lbl`, `_vlc_status_lbl`, `_status_lbl` all become
`statusHint`.

A `setStyleSheet` call in page code is permitted **only** for properties QSS-by-objectName cannot
reach (table grid/alternating rows, progress chunk, separators, the `transportBar` fill) and **only**
with values interpolated from `self.tokens`. It must never contain `font-size`, `font-family`,
`font-weight`, or a colour literal.

### 5.2 Spacing — one scale, named roles

```python
from gui.themes.effects import R, S

# Page column (every card page)
self.content_layout.setContentsMargins(S.page_margin, S.xl, S.page_margin, S.xl)   # 24, 20, 24, 20
self.content_layout.setSpacing(S.card_gap)                                          # 20

# Inside every card
lay.setContentsMargins(S.card_padding, S.card_padding, S.card_padding, S.card_padding)  # 16
lay.setSpacing(S.row_gap)                                                               # 12

# Any horizontal row of controls
row.setSpacing(S.inline_gap)   # 8
# Icon ↔ label, chip internals
tight.setSpacing(S.xs)         # 4
```

| Use | Value |
|---|---|
| Page margin (left/right) | `S.page_margin` = 24 |
| Page margin (top/bottom) | `S.xl` = 20 |
| Inter-card gap | `S.card_gap` = 20 (base defaults to 16 — **set it explicitly**) |
| Card padding, all four sides | `S.card_padding` = 16 |
| Intra-card row spacing | `S.row_gap` = 12 |
| Button/control row spacing | `S.inline_gap` = 8 |
| Icon–label, chip internals, splitter handle | `S.xs` = 4 |

Radii come from `R` (`R.card` 12, `R.button` 8, `R.input` 7). **No new literal spacing or radius
numbers** — one exception class only: responsive breakpoints (`~560`, `~620`, `~700`, `~760` px),
which are thresholds, not spacing. Minimum control height stays 34 (`_make_primary_button`).

---

## 6. HARD PROHIBITIONS — self-audit before you open a PR

Answer every line "no" (or "yes" where marked ✓).

* [ ] **No hardcoded colours.** `grep -nE '#[0-9a-fA-F]{3,8}|rgba?\(|QColor\(' <your file>` returns
      only matches whose values come from `self.tokens`. Specifically fix, if your page is `tools`:
      `_GlassResultChip._COLOR` (`#94a3b8`/`#22c55e`/`#ef4444`), the
      `QColor(255,255,255,α)` washes in `ToolTile.paintEvent` / `_GlassBadge.paintEvent`, and both
      log boxes' `background: rgba(0,0,0,0.1)`.
* [ ] **No `font-size` / `font-family` / `font-weight` literals, no `setPointSize`, no `setFont`.**
      `grep -nE 'font-size|font-family|font-weight|setPointSize|setFont\(' <your file>` is empty.
* [ ] **No emoji in any label string.** Only the §4.6 whitelist survives.
* [ ] **No edits to `gui/themes/**` or `gui/workspaces/base.py`.** No new object names, no new QSS
      rules, no new `QProxyStyle` primitives. `git diff --name-only` must not list them.
* [ ] **No logic moved out of the backend.** No file I/O, no fingerprinting, no plan building in the
      view; every state transition is driven by the `progress` / `log_line` / `finished` signals the
      existing QThread workers already emit.
* [ ] **No change to the page's external contract.** Every signal, slot, public method and public
      attribute another module or a test references keeps its exact name (Player: the full frozen
      list in §3.4).
* [ ] **No widget touched off the main thread.** Workers emit; slots mutate.
* [ ] **No horizontal overflow at ~540 px.** `self._inner.minimumSizeHint().width() <= 900`
      (page_smoke asserts it). No `setFixedWidth` on anything that holds text — delete Player's
      `title_lbl.setFixedWidth(68)`.
* [ ] **No second spine.** One `StageRail`, one `RunBar`, one `primaryBtn`-or-`dangerBtn`, ≤2
      `PhaseProgressList`, zero per-page log cards, zero `QTabWidget` (except `help`).
* [ ] **No write action rendered before a plan exists**, and no `setEnabled(False)` on a form
      container (use `setReadOnly`).
* [ ] **No `QMessageBox`** except the single unrecoverable-delete confirmation of §2B.6.
* [ ] ✓ **Every rendered control's value reaches the worker dict.**
* [ ] ✓ **Shared widget APIs used verbatim**; if a `gui/widgets/` file already existed, you did not
      change its signatures.

---

## 7. DEFINITION OF DONE — one page redesign

1. Archetype confirmed against the page→archetype table at the top of this document; region order matches §3 exactly, in order, nothing
   extra, nothing missing.
2. `PageHeader` is the first widget; the subtitle is one sentence and non-empty.
3. Every shared component your archetype requires is imported from `gui/widgets/` (created verbatim
   per §2B if absent, untouched if present). No local fork of a shared widget.
4. Every per-page log card deleted; all line-level output goes through `self._log(...)`.
5. Exactly one action bar (`RunBar`) mounted via `self.layout().addWidget(...)`, with one action
   slot, `set_blocked` reasons instead of post-click warnings, Cancel as `secondaryBtn`.
6. Commit colour matches §4.4 (Tag Fixer is now `dangerBtn`); `successBtn` appears in no action.
7. All emoji stripped except the §4.6 whitelist; all icons are icon-left/label-right.
8. All `font-size`/colour literals removed; every text role carries its §5.1 object name; every
   margin and spacing comes from `S`, every radius from `R`.
9. Forms use the §4.8 `QFormLayout` block; empty states use the §4.9 sentence pattern; all status
   text lives only in `StatusPill` and `RunBar.set_blocked`.
10. Decorative controls on your page are wired or deleted (§4.13).
11. Preview-first verified by inspection: with no plan in memory, **zero** write-capable controls are
    rendered; editing configuration invalidates the plan, resets the rail, and unarms `CommitGate`.
12. Interaction rules pass a manual pass: stage gating, invalidation, one-job-per-page, the keyboard
    map, Cancel semantics during preview vs execution.
13. Responsive pass at 540 px: nothing clipped, nothing horizontally scrolling, rail in compact mode,
    footers stacked, splitters flipped/hidden per §3.
14. External contract unchanged: `git diff` shows no renamed signal, slot, public method, or public
    attribute.
15. §6 self-audit checklist fully answered, including the `grep` commands.
16. `pytest` passes (no new failures attributable to your page).
17. The smoke test exits 0 — run it and paste the output:

```
xvfb-run -a python3 scripts/page_smoke.py <key>
```

`<key>` is your page's key from the page→archetype table at the top of this document (`indexer`, `duplicates`, `library_sync`,
`compression`, `tag_fixer`, `genres`, `playlists`, `clustered`, `similarity`, `graph`, `player`,
`tools`, `help`). It must print `OK — no exceptions, no overflow` and exit 0. It builds the page,
cycles themes (`--themes midnight,pearl` by default; run
`--themes` with more for a wider sweep), walks every `QTabWidget` index, grabs a pixmap, and fails
on any exception raised inside a paint/event override **and** on
`minimumSizeHint().width() > 900`.

**Build order across the twelve agents** (relevant when a shared widget is missing): `genres` first
(already 1/2/3/4 and already red — smallest diff, proves the vocabulary), then `indexer` as the
reference page, then `duplicates` / `compression` / `tag_fixer`, then `library_sync` last of the
pipeline set (nine cards → four is the largest refactor). Generator, canvas, player, launcher and
reference pages proceed in parallel.
