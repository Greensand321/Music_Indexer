# Design system, theming & motion (cross-cutting): page intent map
_Scope: `gui/themes/` (tokens.py, style.py, manager.py, picker.py, animations.py, effects.py, `__init__.py`), `gui/widgets/gradient_bg.py`, `gui/fonts/loader.py`, `gui/icons.py`, `gui/theme.py` (shim), plus a sweep of all of `gui/` for how the design system is actually used. `gui/qt_launcher.py` is the legacy Tk-launched preview window. It is left out of the counts unless stated. Date: 2026-11-07._

Evidence: probe scripts and screenshots are in
`/tmp/claude-0/-home-user-Music-Indexer/4300de3c-51f6-599a-948e-c4bafb432735/scratchpad/design_system/`
(`probe_themes.py`, `probe_effects.py`, `probe_motion.py`, `probe_landing.py`, `chip_probe.py`, `shot_*.png`). Everything was run with `QT_QPA_PLATFORM=offscreen` on PySide6 6.12, with `HOME` set to a fresh temp dir.

## Summary
The design system is the thing the user never opens as a page but sees on every screen. It provides 14 colour themes (8 dark, 6 light) chosen from a non-modal swatch picker, plus an "Auto" day/night pair. It also provides the Inter / JetBrains Mono typefaces, a QPainter-based widget style (`AlphaDEXStyle`), and a small set of hover and slide animations.

The architecture is a good one: a frozen token dataclass, one manager singleton, and a custom `QProxyStyle`. In practice, though, many widgets bypass it:
- 121 inline `setStyleSheet` calls.
- 194 hard-coded hex colours outside `tokens.py`, most of them a leftover Tailwind-slate palette that belongs to no theme.
- Two competing typography systems (pt in `TypeScale`, px in QSS).
- Motion written ad hoc per widget: about 20 animation sites using 7 easings and 15 distinct durations, and nothing in `animations.py` beyond the hover mixin.

Several "designed" effects never actually show up:
- The card drop shadows are silently garbage-collected (0 of 42 cards have one).
- The workflow steppers never advance.
- The `log_*`, `sidebar_hdr`, `sidebar_text` and `swatch_*` tokens are unused.

There are also real stability problems:
- The landing page can hang forever.
- Every repaint of the sidebar or workspace background re-reads the config JSON from disk.
- Each theme switch blocks the UI for about 1 second.

---

## Pages

### ThemeManager (`gui/themes/manager.py`, class `ThemeManager`, singleton `get_manager()`)
- **Intent:** apply a named theme (or Auto) to the whole running app, remember the choice, and tell widgets about it.
- **When the user comes here / entry points:** indirectly, from the Theme picker. `alpha_dex_gui.main()` and `AlphaDEXWindow.__init__` both call `load_persisted()`, so the theme is applied twice at startup.
- **Primary flow:** `apply(key)` → `_apply_tokens` → `app.setStyle(AlphaDEXStyle(tokens))` → `app.setPalette(build_palette)` → `app.setStyleSheet(_residual_qss)` → `app.setFont(Inter 11pt)` → emit `theme_changed(tokens)` → `_persist()` writes `cfg["theme"]`. There is no preview or confirmation step; changes apply instantly, by design.
- **Reads / writes:** config keys `theme`, `auto_dark_theme`, `auto_light_theme`, read and written through `config.load_config/save_config` (compliant). Signal out: `theme_changed(ThemeTokens)`.
- **Backend dependencies:** `config` only, on the main thread.
- **State owned:** `_current`, `_auto_mode`, `_auto_dark`, `_auto_light`, kept for the life of the process. Being a singleton, it outlives every window, which matters for tests.
- **Async model:** synchronous, roughly 1 s per switch (measured: `setStyleSheet` 776 ms, `setStyle` 197 ms, palette 1.5 ms, signal fan-out 10 ms, with 953 widgets). Auto mode listens to `styleHints().colorSchemeChanged`. It has no timer for the time-of-day fallback, and that fallback is unreachable anyway on Qt ≥ 6.5 (DS-10).
- **Motion & visuals:** none. The residual QSS styles tooltips, text edits, item views, headers, menus, cards (`#workspaceCard/#headerCard/#actionCard`), the stepper and the workspace typography roles.
- **Decorative / dead controls:** none, but several tokens are never consumed (see the token inventory).
- **Overlaps:** `gui/theme.py` is a shim that still exports Midnight-only `LOG_*` constants, and `log_drawer.py` uses those instead of the tokens.
- **Redesign notes:** keep the singleton and the signal. Stop re-setting the application stylesheet on every switch: make the QSS static and refer to colours with `palette(...)` roles or dynamic properties, so a switch costs only `setPalette` plus a style token swap.
- **Testability seams:** `ThemeManager._instance = None` resets the singleton between tests. `apply()` persists to config, so point `HOME` at a temp dir. Lambdas connected to `theme_changed` survive widget deletion (DS-07), so a harness that builds many windows must reset the singleton or check for disconnects.

### AlphaDEXStyle (`gui/themes/style.py`, class `AlphaDEXStyle(QProxyStyle("fusion"))`)
- **Intent:** paint every standard control (buttons, inputs, checkboxes, radios, tabs, progress bars, scrollbars, sliders, combos and spin boxes) from tokens with rounded, anti-aliased shapes.
- **Entry points:** installed app-wide by the manager.
- **Primary flow:** `drawPrimitive`, `drawControl` and `drawComplexControl` read `self._t`. Button variants are selected by objectName (`primaryBtn`, `dangerBtn`, `successBtn`). Hover progress comes from `widget._hover_p` (HoverMixin) or, failing that, `State_MouseOver`. `polish()` turns on `WA_Hover` and assigns fonts by objectName role (`appTitle`, `sectionTitle`, `sectionSubtitle`/`mutedLabel`, `sidebarSectionLabel`, `navButton`, `libraryPath`/`libStats`, `statNumber`, `statUnit`, `logText`). Every other button, combo, spin box and line edit gets Inter 11pt.
- **Reads / writes:** tokens, plus `gui.fonts.loader.TypeScale`.
- **State owned:** a reference to the tokens. A new style instance is created on every theme switch.
- **Motion & visuals:**
  - `_hover_t` lerps colours as `HoverMixin._hover_p` animates.
  - The indeterminate progress bar is drawn as a solid fill. A gradient is built but never used (`style.py:466-473`), so there is no pulse even though the comment says "pulsing".
  - `CE_ProgressBarLabel` is suppressed, so progress text is never shown.
- **Decorative / dead:** the `libStats` role has no widget. The `QStyleSheetStyle` wrapper (present because the app has a stylesheet) handles any widget that has an inline QSS rule, which bypasses this style entirely for those widgets.
- **Redesign notes:** keep this. It is the strongest part of the system. Move role fonts out of `polish()` (DS-11) and into a type-ramp helper that widgets opt into. Replace objectName button variants with a dynamic property (`variant="primary"`), so names stay free for tests.
- **Testability seams:** pure painting, so it can be checked by grabbing a single widget per theme for snapshot tests.

### Theme picker (`gui/themes/picker.py`, classes `ThemePickerDialog`, `AutoThemeDialog`, `_SwatchCard`)
- **Intent:** let the user preview and choose a theme visually, or set up the Auto day/night pair.
- **Entry points:** the Top bar's "Theme" button → `AlphaDEXWindow._on_theme_requested` → `open_theme_picker(self)`. The dialog is non-modal and held in `self._theme_picker`. There is no keyboard shortcut.
- **Primary flow:** click a swatch → `get_manager().apply(key)`, which applies and persists immediately with no confirmation (fine for cosmetic settings). "Configure…" opens the modal `AutoThemeDialog` (`exec()`), which emits `pair_selected` → `configure_auto` → `_apply_auto`. "Select" on the Auto card switches to auto mode.
- **Reads / writes:** manager state, and config through the manager.
- **State owned:** `_current_key`, rebuilt each time the dialog opens. Repeated opens create new dialogs; the old one is closed but not deleted (`close()` without `WA_DeleteOnClose`).
- **Async model:** synchronous. Each swatch click freezes the UI for about 1 s while the switch runs.
- **Motion & visuals:** custom-painted swatch cards with no hover or selection animation. Fonts are hard-coded (`setPointSize` 8, 9, 14, 20).
- **Decorative / dead:** the `swatch_bg`, `swatch_accent` and `icon` tokens (emoji) are never drawn. The swatch paints its own mock instead.
- **Overlaps:** a Settings drawer exists separately. The theme is not in Settings, so it is split across two entry points.
- **Redesign notes:** the swatch name and checkmark use the current theme's colours on top of the swatch's own colours, which makes dark-theme names unreadable while a light theme is active (DS-15, visible in `shot_picker.png`). The Auto description is cut off at 680 px ("day/n…"). The picker would fit better as a section inside Settings, with a live preview that does not stall.
- **Testability seams:** `AutoThemeDialog` is opened with `exec()`, so patch it out. Swatches are found with `findChildren(_SwatchCard)` and `.tokens.key`. `theme_applied(str)` is a usable signal.

### Hover/animated controls (`gui/themes/animations.py`: `HoverMixin`, `AnimatedButton`, `AnimatedNavButton`, `AnimatedTabButton`)
- **Intent:** give controls a 150 ms colour ease on hover, and give sidebar items an active-bar and badge treatment.
- **Entry points / consumers:**
  - `AnimatedButton`: the 3 Top-bar buttons and the Tools footer buttons (`tools.py:593-616, 989`).
  - `AnimatedNavButton`: the 14 sidebar items plus Exit.
  - `AnimatedTabButton`: **unused**.
  - Every other `QPushButton` in the app (hundreds) only gets the style's instant `State_MouseOver` flip, so hover motion is inconsistent from one button to the next.
- **Primary flow:** `enterEvent`/`leaveEvent` → `QVariantAnimation` from the current `_hover_p` to 0 or 1 (OutCubic, parented to self, reused, stopped before each restart). This part is clean.
- **Motion & visuals:** `AnimatedNavButton.paintEvent` paints its own hover tint, a 4×20 px accent bar and the badge, and reads `get_manager().current` on every paint.
- **Bugs:** the `active` setter forces `_hover_p = 1.0` and never animates back when the item is deactivated (DS-05).
- **Redesign notes:** this is the only shared motion primitive. Extend it into a motion module (see the recommendations) instead of hand-rolling a `QVariantAnimation` in each widget.

### Gradient background (`gui/widgets/gradient_bg.py`: `GradientWidget`, `paint_window_gradient`, `_gradient_enabled`)
- **Intent:** a faint two-corner accent glow behind workspace content, the sidebar and the landing page, to give each theme some atmosphere.
- **Entry points / consumers:**
  - `GradientWidget` is the scroll-inner widget of every `WorkspaceBase` (`base.py:54`) and of the Player (`player.py:556`).
  - `paint_window_gradient` is used by `Sidebar.paintEvent` and `MosaicLanding.paintEvent`.
- **Reads / writes:** config key `bg_gradient_enabled`, read **on every paint** (DS-03). It is not exposed in the Settings UI as far as this sweep found; it can only be changed by editing the JSON by hand.
- **Motion & visuals:** two full-rect radial fills per paint. `GradientWidget` anchors its glow to its *own* rect, which is the whole scrollable content height. Sidebar and landing anchor to *window* coordinates. The docstring promises "one seamless canvas", but the workspace glow scrolls with the content and is sized by content height, so the two never line up.
- **Redesign notes:** paint the glow once per resize into a cached pixmap, read the flag once, and anchor it to the viewport rather than the content.
- **Testability seams:** monkeypatch `gui.widgets.gradient_bg._gradient_enabled`.

### Fonts (`gui/fonts/loader.py`: `load_fonts`, `TypeScale`, `UI_FAMILY`, `MONO_FAMILY`)
- **Intent:** bundle Inter (variable plus 4 static weights) and JetBrains Mono (Regular, Medium), and define a pt-based type ramp.
- **Entry points:** `alpha_dex_gui.main()` before any widget is built. All 7 files load (`probe_themes.py`).
- **Redesign notes:** `TypeScale` is only partly honoured (see the typography inventory). The `LS_*` letter-spacing constants are unused; style.py hard-codes 98% and 108%. `TAB` and `INPUT` are unused.

### App icon (`gui/icons.py`: `make_app_icon`)
- **Intent:** a programmatic equaliser-bars icon at 7 sizes. It uses 4 hard-coded hexes (`#1e2235`, `#0d1117`, `#a78bfa`, `#6366f1`) that are brand colours rather than theme colours, which is acceptable. Keep it, and move the 4 colours into a `brand` token group.

### SplashScreen (`gui/widgets/splash.py`). **Dead code**
- Nothing imports it; `landing.py` says it "replaces the old SplashScreen". It holds a looping spinner animation (`loopCount -1`), two fill phases and a fade, plus its own 8-colour palette dict. Delete it during the redesign, or keep it only as motion reference.

---

## 1. Token inventory

**Themes (14):** dark: `midnight` (default), `obsidian`, `graphite`, `navy`, `twilight`, `aurora`, `forest`, `ember`. Light: `pearl` (default light), `azure`, `blossom`, `meadow`, `lavender`, `sunset`. Every light theme uses a *dark* log area (`log_bg` is dark in all 14).

**Tokens (43 string fields):** references found by grepping `\.<token>` outside `tokens.py`:

| Group | Token | Used by | Status |
|---|---|---|---|
| identity | key, name, variant | manager, picker, sidebar, gradient_bg | used |
| identity | icon | — | **unused** (emoji never shown) |
| sidebar | sidebar_bg | sidebar, landing, now_playing_bar, picker | used |
| sidebar | sidebar_hover | style, animations, manager QSS | used (doubles as the generic "hover" colour) |
| sidebar | sidebar_active | sidebar pill only | used |
| sidebar | sidebar_accent | picker swatch dot only | effectively unused in the app |
| sidebar | sidebar_text, sidebar_hdr | — | **unused** (section headers render in `text_primary` through the palette) |
| content | content_bg, card_bg, card_border | style, manager, gradient_bg, many | used |
| content | card_shadow | effects.card_shadow | used, but the effect never survives (DS-02) |
| text | text_primary / secondary / muted / inverse | style, manager, top_bar, tools, animations | used |
| interactive | accent, accent_hover | widely | used |
| interactive | accent_pressed | style only | used |
| interactive | success, danger, danger_hover | style, tools | used |
| interactive | warning | — | **unused** (workspaces pass `"#f59e0b"` literals instead) |
| inputs | input_bg / border / focus | style, manager | used |
| progress | progress_bg, progress_fg | style | used |
| tabs | tab_bg / active_bg / text / active_text | style | used |
| log | log_bg, log_text | manager QSS (`#logText`), tooltips, palette | used |
| log | log_handle, log_ok, log_warn, log_err | — | **unused** (LogDrawer uses the Midnight constants in `gui/theme.py`) |
| swatch | swatch_bg, swatch_accent | — | **unused** |

That makes 11 of 43 tokens unused, and 2 more (`sidebar_accent`, `card_shadow`) with no visible effect.

**Missing tokens** the app needs, as shown by the literals it uses instead:
- status colours for the log-drawer chip (`warning`, `success`, `danger` exist but are bypassed);
- a "surface-2 / elevated" colour (the player uses `#1c2128` and `#0d1117`);
- an overlay or scrim colour (`rgba(10,13,20,…)` in player and landing);
- a brand pair (`#6366f1`/`#a78bfa`, which appears in icons, the landing page, the Player progress bar, the kb button and the queue highlight);
- a categorical chart palette (landing `_GRADS` has 18 pairs; cluster colours live in the graph widgets).
- There are no radius, spacing, elevation or motion tokens. Radii live in `effects.R`, which only `style.py` uses; inline QSS uses 11 different radius values.

**Hard-coded colours outside tokens.py:** 194 hex literals (plus 45 in legacy `qt_launcher.py`) and 13 `rgba()` literals.

| File | hex | File | hex |
|---|---|---|---|
| widgets/landing.py | 42 (36 are the tile palette) | dialogs/settings_drawer.py | 7 |
| workspaces/player.py | 16 | workspaces/genres.py | 7 (status) |
| widgets/now_playing_bar.py | 15 | workspaces/duplicates.py, log_drawer.py, pair_review_dialog.py | 6 each |
| dialogs/cluster_quality_report_dialog.py | 10 | gui/theme.py (shim) | 5 |
| workspaces/library_sync.py | 9 (status) | clustered.py, icons.py | 4 each |
| tag_fixer.py, compression.py, splash.py, clustering_wizard_dialog.py | 8 each | tools, similarity, playlists, indexer, clustered_enhanced | 3 each |
| style.py | 2 (#fff/#000 lerp targets, fine) | fuzzy_dupe, media_tester | 2 each |
| help, top_bar, sidebar, interactive_3d_scatter | 1 each | | |

About 40 of these literals are the `status_changed.emit(text, "#f59e0b" / "#22c55e" / "#ef4444")` calls in 8 workspaces. These are the token values of the *Midnight* theme passed as strings, so the log-drawer status chip never changes with the theme. The most common colour that belongs to no theme is `#64748b` (Tailwind slate-500), used by 15+ labels.

**`setStyleSheet(` calls per file:** 121 in the active app (plus 1 in manager and 47 in `qt_launcher.py`).

| File | n | File | n |
|---|---|---|---|
| workspaces/player.py | 16 | widgets/top_bar.py | 4 (theme-aware, re-applied on change) |
| dialogs/cluster_quality_report_dialog.py | 14 | widgets/landing.py | 4 |
| workspaces/tools.py | 11 (several theme-aware) | dialogs/pair_review_dialog.py | 4 |
| dialogs/clustering_wizard_dialog.py | 9 | tag_fixer.py, fuzzy_dupe_dialog.py | 3 each |
| workspaces/compression.py | 8 | genres, duplicates, clustered, cluster_legend, media_tester | 2 each |
| widgets/now_playing_bar.py | 8 | similarity, playlists, library_sync, help, track_details_panel, interactive_3d_scatter, crash_log_dialog | 1 each |
| widgets/log_drawer.py | 6 | | |
| widgets/sidebar.py, dialogs/settings_drawer.py | 5 each | | |
| clustered_enhanced.py | 4 | | |

Only `top_bar.py` and parts of `tools.py` re-apply their inline styles on `theme_changed`. All the others freeze the colours they were built with.

**Measured stale inline styles:** in `probe_themes.py`, each widget's inline QSS hexes were checked against the active theme's token values. This found 25 to 31 widgets per theme (25 even on Midnight) with colours from no theme at all. Breakdown on Pearl:
- 15 labels with `#64748b`;
- 3 labels and the Player table header with `#94a3b8`;
- the sidebar logo `#f8fafc`;
- the log-drawer handle `#1e293b`, body `#0d1117`, and Clear button `#334155`/`#64748b`;
- the kb-mode button `#30363d`/`#64748b`/`#818cf8`;
- the track hover popup `#1c2128`/`#30363d`;
- the now-playing art `#1e2130`/`#475569`;
- the Player separator `#30363d`;
- the Player art and title labels `#0d1117`/`#e6edf3`.

Theme cycling raised no exceptions and printed no Qt warnings.

**objectNames, QSS vs widgets:**
- QSS selectors in `manager.py`: `workspaceCard, headerCard, actionCard, workflowStepper, stepActive, stepInactive, stepArrow, sectionTitle, sectionSubtitle, cardTitle, phaseLabel, phaseDesc, statusHint, notesHint, logText`. All 15 are set by some widget, so there are **no dead QSS selectors**.
- Style-dispatch names in `style.py`: `primaryBtn` (9 widgets), `dangerBtn` (3), `successBtn` (1), `appTitle`, `sectionTitle`, `sectionSubtitle`, `mutedLabel` (7), `sidebarSectionLabel`, `navButton`, `libraryPath`, **`libStats` (dead: no widget has it)**, `statNumber`, `statUnit`, `logText`.
- objectNames set but **styled by nothing** (neither QSS nor style): `contentFrame`, `topBar`, `sidebar`, `logDrawer`, `nowPlayingBar`, `transportBar`, `tabButton`, `liquidGlassTile` (styled by its own inline QSS), `logBox` (3), `statusLabel`, `sectionHeader` (picker), `autoCard`, `trackHoverPopup` and `kbModeBtn` (both inline-only). These are good test hooks, but they are not design hooks.
- `mutedLabel` gets a font from `polish()` but **no colour rule**, so "muted" labels render in `text_primary` (for example the picker subtitle and the top-bar dots).

## 2. Typography & spacing inventory

**Fonts loaded:** `InterVariable.ttf`, Inter Regular, Medium, SemiBold and Bold, and JetBrains Mono Regular and Medium. All 7 register successfully.

**Three sources of size, which override each other:**
1. `app.setFont(Inter, TypeScale.BODY=11pt)`. This is the default for every label without a role, about 14.7 px at 96 dpi.
2. `AlphaDEXStyle._polish_font`, in pt: appTitle 16pt, sectionTitle 14pt, nav 13pt, sidebar headers 10pt, subtitle/muted 10pt, stats 9pt, log 10pt mono, and **every** button, combo, spin box and line edit at 11pt. It runs on first show and again on every theme switch, and it overrides `setFont()` (DS-11).
3. The residual QSS in px, which wins over (2): sectionTitle 18px, subtitle 12px, cardTitle 13px, phaseLabel 12px, phaseDesc 11px, statusHint 12px, notesHint 11px, step labels 12px, header sections 12px, status bar 12px, text edits 10pt, tooltips 9pt.

Inline QSS adds more: 108 `font-size` declarations (`12px`×39, `11px`×23, `10px`×9, `14px`×6, `13px`×6, `18px`×3, `16px`×3, `15px`×2, `22px`, `9px`, `10pt`×2, `9pt`). The code also has 10 explicit pt sizes (picker 8/9/14/20, tools 10, landing 42/15, splash 52/11).

**Effective sizes measured in the running app** (`QFontInfo` px): sectionTitle 18, cardTitle **13**, sectionSubtitle 12, statusHint 12, body (unstyled labels and checkboxes) about **15**, navButton **17**, appTitle 21, sidebar header 13, log 13.

The hierarchy is inverted: card titles (13 px, semibold) are *smaller* than the body text under them (about 15 px), and sidebar items (17 px) are nearly as large as the page heading (18 px). `TypeScale.SECTION_HDR` (14pt ≈ 18.7 px) is silently replaced by QSS 18 px. The brand appears twice on screen: top bar "AlphaDEX" at 21 px and sidebar "AlphaDEX" at 22 px inline (visible in every screenshot).

**Spacing:** 35 distinct `setContentsMargins` tuples. The most common are (16,16,16,16)×44 and (0,0,0,0)×30, followed by (8,8,8,8), (16,12,16,12), (12,12,12,12), (20,20,20,20) and (20,16,16,16)×3. Odd values include (44,52,44,52), (18,…) variants and (24,24,24,20).

`setSpacing` values: 8×28, 10×24, 12×19, 0×15, 6×13, 4×12, 16×7, 1×3, 2×2, and 3, 14 and 18 once each. Mostly a 2/4-px grid, with 10 and 6 as frequent off-grid steps. Workspace padding is 24/20 (`base.py:56`), card padding is 16, Tools tiles use 18, and the picker uses 24/20.

**Radii:** `effects.R` defines button 8, input 7, card 12, tab 6, checkbox 4, radio 8 and scroll 4. Inline QSS uses 8×11, 12×9, 10×8, 6×8, 4×5, 3×2, 11×2, 7, 2 and 999. The painted widgets use more again: nav 8, NavContainer pill 8, ToolTile 16, CTA card 20, landing tiles 10, swatch 10.

**Fixed heights:** buttons are at least 34 (style), the top bar is 64, nav items stretch between 32 and 68 (`sidebar._relayout_buttons`), the now-playing bar is 60, the log handle 30, progress bars 6 (×12) and the sidebar width a fixed 360.

**Contrast** (WCAG, computed from tokens):
- `text_primary`/`card_bg` ≥ 9.7 in every theme (good). `text_secondary` ≥ 4.8 in every theme (ok).
- **`text_muted`/`card_bg` is 1.5 to 2.75 in all 14 themes**, yet it carries functional text: `statusHint` (17 labels), `phaseDesc`, `notesHint`, `stepInactive`, disabled labels.
- `text_inverse`/`accent` (primary button labels) is below 4.5 in obsidian 2.72, meadow 2.54, sunset 2.80, blossom 3.53, azure 3.68, lavender 4.23 and pearl 4.47.
- `tab_text`/`tab_bg` is below 3 in 11 of 14 themes.
- Inline outliers: compression's result label `#e2e8f0` on a light card is 1.23; the sidebar logo `#f8fafc` on Pearl's sidebar is 1.07 (invisible, see `shot_pearl_*.png`); the wizard's `#666` on a Midnight card is 2.82.

## 3. Motion inventory

`gui/themes/animations.py` offers only `HoverMixin` (150 ms OutCubic, parented, reused, stops before restart) and three button subclasses. Every other animation is hand-written where it is used. There are no shared duration or easing constants, no reduced-motion switch, and no page-transition system: `QStackedWidget.setCurrentWidget` switches instantly.

| # | file:line | What it animates | Duration / easing | Lifetime & cleanup | Issues |
|---|---|---|---|---|---|
| M1 | themes/animations.py:20 | `_hover_p` (0↔1) colour lerp, AnimatedButton/NavButton | 150 / OutCubic | parented, reused, stop-before-restart | nav `active` setter leaves `_hover_p=1` (DS-05). Each frame `update()`s a *transparent* button, so the parent sidebar repaints and reads config twice (DS-03) |
| M2 | widgets/sidebar.py:67 | NavContainer pill y | 320 / **OutBack** (overshoot) | parented, reused | The accent bar on the button jumps instantly while the pill slides, so they are briefly out of sync (`shot_*_tools.png`). 42 config reads per slide |
| M3 | widgets/log_drawer.py:78 | `_log_body.maximumHeight` 0↔220 | 200 / InOutCubic | unparented but held in `self._anim`, reused | animates a widget **inside a layout**, so the whole content stack re-lays out every frame. The natural height is 192, so about 20% of each run has no visible effect (DS-18) |
| M4 | widgets/now_playing_bar.py:196 | kb button `geometry` keyframe pulse | 120 / OutCubic | `finished→deleteLater`, but the previous animation is never stopped | geometry animation on a widget the layout owns; a rapid double toggle leaves it offset by (+2,+1) px (DS-17) |
| M5 | workspaces/tools.py:302-325 | `_GlassResultChip` opacity out→swap→in | 160 + 260 / OutCubic | `DeleteWhenStopped` | the `self._anim` reference dangles, so a second transition raises RuntimeError (DS-06) |
| M6 | workspaces/tools.py:405 | ToolTile hover glow, using a **16 ms QTimer** loop with exponential smoothing (k=0.15) | continuous | timer never stops: 5 tiles produce about 315 wakeups/s even while Tools is hidden (DS-09) | full-rect radial gradient and clip path per paint |
| M7 | workspaces/tools.py:640 | ToolTile drawer `maximumHeight` 0→sizeHint | 250 / OutCubic | parented, `DeleteWhenStopped`, resets max to QWIDGETSIZE_MAX | opens once only and never closes (no collapse animation) |
| M8 | workspaces/tools.py:653, 1143 | label colour, applied by **`setStyleSheet` every frame** | 1600 / OutCubic | `DeleteWhenStopped`, not stopped on re-trigger | about 100 restyles per flash, and overlapping flashes compete (DS-20). `flash_status` targets a hidden label |
| M9 | workspaces/tools.py:780 | `_PillToggle` knob x | 150 / OutCubic | parented, reused | fine (lambda theme connection, DS-07) |
| M10 | workspaces/tools.py:1135 | `_reveal_card` maximumHeight | 220 / OutCubic | parented, `DeleteWhenStopped` | fine |
| M11 | workspaces/player.py:200 | `_BgArtWidget.crossfade` (Qt Property) | **3500** / InOutSine | unparented, held, reused | `paintEvent` runs `pm.scaled()` on full-size pixmaps **twice per frame**. The table viewport is translucent, so every table scroll or hover also repaints this widget. Background is a hard-coded dark `QColor(13,17,23)` plus a vignette. A new art arriving mid-fade swaps instantly (`_set_cf`) |
| M12 | widgets/landing.py:922 | landing window opacity 0→1 | 320 / OutCubic | parented, held | then `singleShot(600, _fly_in)` is **never cancelled** (DS-01) |
| M13 | widgets/landing.py:941 | 32 tiles `pos` fly-in, Parallel(Sequential(Pause(i·38), Prop)) | 500 / **OutBack**, stagger 38 (total ≈ 1.68 s) | group parented to the landing | the landing itself is only hidden, never deleted, so 32 tiles and their pixmaps stay alive for the whole process |
| M14 | widgets/landing.py:1011 | 32 tiles scatter | Pause rand(0–160) + 360 / **InBack** | parented | when fly-in starts on the same `(tile,pos)`, Qt stops the scatter, `finished` never fires, and the app hangs (DS-01). A second `_accept` snaps tiles back mid-flight (DS-23) |
| M15 | widgets/landing.py:1035 | landing opacity 1→0 | 420 / InCubic | parented, held | cross-fade only; on platforms without window opacity (Wayland, offscreen: "This plugin does not support setting window opacity") the windows just cut |
| M16 | alpha_dex_gui.py:148 | main window opacity 0→1 | 420 / OutCubic | parented to the window, `finished→deleteLater` | ends at 1.0. Clean |
| M17 | widgets/splash.py:113-146 | spinner (looping), fills, fade | various | — | **dead code** |
| T1 | workspaces/player.py:600 | `_pos_timer` 400 ms | — | started and stopped with playback | unparented QTimer (`QTimer()`) but held. OK |
| T2 | workspaces/player.py:471, 732 | hover-popup 1500 ms single-shot | — | parented | popup uses hard-coded dark styling |
| T3 | sidebar.py:197, 275, 298; alpha_dex_gui.py:104, 125, 135 | `singleShot` 0/10/3000 ms layout and startup deferrals | — | — | `_landing_to_main` re-polls every 10 ms with no cap |

**Graphics effects:** the only `setGraphicsEffect` use is `card_shadow` (`base.py:71,79`). Because PySide6 6.12 does not keep a Python-created effect alive when it is passed as a temporary, **0 of 42 cards have a shadow** (`probe_effects.py`; a minimal repro is in the bug table). There are no `QGraphicsOpacityEffect`s, so there is no effect stacking today. Fixing DS-02 naively would introduce two jank sources:
1. A `QGraphicsDropShadowEffect` (blur 18) on a card re-renders the whole card off-screen whenever any child repaints. That includes M1 hover frames, every progress tick, and text-cursor blinks.
2. Any future fade on a card would collide with its shadow effect, because a widget can only have one effect.

**Other jank sources found:**
- **Config I/O inside paint:** `_gradient_enabled()` → `load_config()` (open, `json.load`, fill defaults) on every `GradientWidget`, `Sidebar` (twice) and `MosaicLanding` paint. Measured 5 reads per full window repaint, 42 per pill slide, 22 per nav hover and about 1 per progress-bar update (DS-03).
- **Theme switch stall** of about 1 s (DS-04).
- **Per-frame `setStyleSheet`** (M8).
- **Heavy `paintEvent`s:** player background scaling (M11); ToolTile radial gradient at 60 fps while the pointer is moving; workspace GradientWidget filling two radial gradients across the full content height, which can be thousands of px.
- **Timers running while hidden:** M6.
- **Layout-owned geometry animations:** M3, M4, M7 and M10 animate `maximumHeight` or `geometry` of widgets inside layouts, so every frame relays out the parent. M3 does this for the entire workspace stack.

**Opacity at rest:** every opacity animation found ends at 1.0 or ends hidden (landing). The chip `_opacity` returns to 1.0. Nothing is left translucent, except the frozen landing in the DS-01 state, where it stays at full opacity and keeps covering the screen.

## 4. Theme switching end to end

`ThemePickerDialog._apply_theme(key)` → `ThemeManager.apply` → `_apply_tokens`:
1. `app.setStyle(new AlphaDEXStyle)`, which re-polishes every widget and re-runs `_polish_font`, overwriting `setFont` calls (DS-11).
2. `app.setPalette`.
3. `app.setStyleSheet(residual)`, about 0.78 s.
4. `app.setFont`.
5. Emit `theme_changed`, then `_persist` writes the config.

Listeners on `theme_changed`:
- `AlphaDEXWindow._on_theme_changed` → `ws.refresh_shadows()` for each workspace.
- `WorkspaceBase._on_theme_changed_base` → `refresh_shadows()` **again**, so shadows are rebuilt twice per switch (DS-21), and neither set survives (DS-02).
- `GradientWidget._on_theme_changed` → `update()`.
- `TopBar._on_theme_changed` → re-applies its inline styles (the only widget that does this properly).
- `tools.py` → `ToolTile._on_theme_changed` (separator colour), `ToolsWorkspace._on_theme_changed_base` (codec chips re-styled), and `_GlassBadge`, `_GlassResultChip` and `_PillToggle` through lambdas (DS-07).

Widgets that read `get_manager().current` at paint time update for free: Sidebar, NavContainer, AnimatedNavButton, NowPlayingBar background, ToolTile, swatches, landing. Widgets that read colours once at construction stay stale: the log drawer, the NowPlayingBar labels and kb button, the Player (table, popup, background art, labels), Compression, Tag Fixer, Clustered and the clustering wizard, the cluster-quality and pair-review dialogs, Settings, the sidebar logo and separator, the landing CTA, and the graph background (`interactive_scatter_plot.py:145`, set from the palette once, with a white highlight pen).

**Offscreen run** (`probe_themes.py`): an `AlphaDEXWindow` cycled through all 14 themes produced 0 exceptions and 0 Qt warnings. Each switch took 773–1110 ms. The stale-inline scan found 25–31 widgets per theme.

**What the screenshots show:**
- `shot_midnight_indexer.png`: coherent dark UI. The steppers read "1. Preview (dry run) → 2 → 3", with step 1 highlighted permanently. No card shadows. The "Activity Log" handle is slate.
- `shot_pearl_indexer.png`: light cards look good, but the **sidebar "AlphaDEX" logo is invisible**, the **log-drawer handle stays dark navy**, and the **Player and Utilities nav rows keep a hover tint** from earlier programmatic navigation (DS-05).
- `shot_pearl_player.png`: the **Player library table is a black slab** on the light theme, and the cover placeholders are dark grey.
- `shot_pearl_tools.png`: tiles are **muddy grey** because of the black tint meant for dark themes (DS-08). The sidebar pill is caught mid-slide at "Duplicates" while the accent bar is already on "Utilities".
- `shot_picker.png`: dark-theme swatch names are illegible under Pearl (DS-15), and the Auto description is cut off.

## 5. Recommendations for the redesign's design system

**Keep:**
- The frozen `ThemeTokens` dataclass and registry.
- The `ThemeManager` singleton and `theme_changed` signal.
- `AlphaDEXStyle` painting standard controls.
- `HoverMixin`'s pattern: one reusable, parented animation; stop before restart; animate from the current value.
- Inter + JetBrains Mono.
- The programmatic icon.

**Replace or fix:**
1. **Split tokens** into `color` (keep), `space` (4-pt scale: 0, 4, 8, 12, 16, 24, 32), `radius` (sm 4, md 8, lg 12, xl 16), `type` (one px ramp, for example caption 11, body 13, label 12, title 15, heading 20, display 28, with weights), `elevation` (0 to 2) and `motion`. Add the missing colour roles: `surface_2`, `overlay`, `brand_a/b`, `status_*`, `chart_*`, and `text_muted` re-tuned to at least 3:1 (4.5:1 if it carries status text).
2. **One typography system.** Delete either `TypeScale` pt or the QSS px rules. Never set fonts in `polish()`; use an opt-in role property instead (`role="title"`).
3. **Make theme switches cheap.** Use a static QSS that refers to `palette(...)` and dynamic properties, or no QSS at all, plus `setPalette` and a token swap on the existing style followed by `update()`. Target under 100 ms.
4. **No inline colours.** A widget that needs a custom colour paints it from tokens at paint time, or subscribes via a bound method. Workspaces emit a *semantic* status (`"running"`, `"ok"`, `"error"`), not a hex string.
5. **Elevation by painting, not by QGraphicsEffect.** Render a cached 9-slice shadow pixmap per (radius, elevation, theme) in the card's own `paintEvent`. Keep the one-effect-per-widget slot free for opacity transitions.
6. **Motion module** (`gui/themes/motion.py`) providing `fade(widget)`, `slide(widget)`, `expand(widget)` and `hover()`. Each is parented, reuses or replaces one animation per (widget, property), handles re-entrancy, honours a global reduced-motion flag (duration × 0), and never animates `geometry` of a layout-managed widget. Use transform-like painting offsets or a `QStackedLayout` overlay snapshot for page transitions instead.
7. **Background glow:** cache per resize and theme, anchor to the viewport, read the flag once.

**Proposed motion tokens** (based on what already exists and feels right):

| Token | ms | Easing | Today's users |
|---|---|---|---|
| `motion.instant` | 0 | — | reduced-motion |
| `motion.micro` | 120 | OutCubic | kb pulse, press feedback |
| `motion.fast` | 150 | OutCubic | hover (M1), toggles (M9) |
| `motion.base` | 220 | OutCubic in / InCubic out | drawers and reveals (M3, M7, M10), chip (M5: 160 + 260 → base out + base in) |
| `motion.slow` | 320 | OutCubic (default) / `emphasis` OutBack(1.2) | nav pill (M2), window fade-in (M12) |
| `motion.page` | 420 | OutCubic in / InCubic out | landing↔main cross-fade (M15, M16) |
| `motion.stagger` | 38 per item, capped at 600 total | — | landing tiles (M13) |
| `motion.ambient` | 3500 | InOutSine | album-art cross-fade (M11) |
| `motion.flash` | 1600 | OutCubic | colour flash (M8), painted rather than restyled |

Easings collapse to 4: `standard` = OutCubic, `exit` = InCubic, `emphasis` = OutBack, `ambient` = InOutSine. Drop InBack (the scatter "pull-back" reads as a hitch) and InOutCubic.

**Minimum harness checks to keep it honest:**
1. **Static, ratcheted:** no `#rrggbb`/`rgba(` literals in `gui/**` outside `tokens.py` (allow-list `icons.py` until brand tokens exist). Current baseline 194, it must only go down. The same ratchet applies to `setStyleSheet(` (baseline 121) and to `font-size:` outside the theme module.
2. **Theme sweep:** apply all 14 themes on a full `AlphaDEXWindow`. Expect no exceptions or `qWarning`s, zero widgets with inline hexes outside the active token set (baseline 25–31), and each switch under a time budget (baseline about 1 s; target 150 ms).
3. **Token contrast test:** primary/secondary text at least 4.5 on `card_bg` and `content_bg`, muted at least 3.0, `text_inverse` on `accent` at least 4.5, for every theme.
4. **Animation hygiene, at runtime:** after exercising flows and pumping for 1 s, no `QAbstractAnimation` in the app is Running unless it is declared ambient. Every animation has a parent or `DeleteWhenStopped`. Repeating a trigger N times does not grow the animation count.
5. **Idle check:** after 1 s idle, no active `QTimer` with interval under 50 ms anywhere (catches M6), and all widgets have `windowOpacity()==1.0`, any `QGraphicsOpacityEffect.opacity()==1.0`, and `_hover_p==0` for buttons not under the mouse (catches DS-05).
6. **No I/O in paint:** patch `config.load_config` to raise while `QEvent.Paint` is being handled (or count calls during `w.repaint()`). Expect zero.
7. **Connection hygiene:** create and destroy every workspace, dialog and widget, then switch the theme. Expect no `RuntimeError: Internal C++ object … already deleted`.
8. **Landing robustness:** call `_accept()` at random delays between 0 and 2000 ms, and twice in a row. `library_selected` must be emitted exactly once, within 2 s.
9. **Effects present:** if shadows remain an effect, assert `card.graphicsEffect() is not None` after construction and after a theme switch. If they become painted, assert none exist.

## Cross-page relationships
- Tokens → `ThemeManager` → (`AlphaDEXStyle` + palette + residual QSS) feeds every page. `theme_changed` fans out to the main window, `WorkspaceBase` (all 13 workspaces), TopBar, `GradientWidget` and Tools sub-widgets.
- `WorkspaceBase` is where the design system meets the pages: `_make_card` (shadow), `_make_section_title`, `_make_subtitle`, `_make_card_title`, `_make_primary_button` and the GradientWidget scroll surface. Workspaces that bypass these helpers (Player overrides `_setup_scroll`; Tools builds glass tiles) are the ones that diverge visually.
- Workspaces → `status_changed(text, hex)` → `LogDrawer.set_status` and the status bar. The colour vocabulary is fixed to Midnight's values outside the design system.
- Sidebar ↔ main window: `nav_changed`, `activate(key)` (programmatic: Ctrl+1..9, `navigate_requested`, `play_tracks`, `play_directory`). This programmatic path is what triggers DS-05.
- Landing (`MosaicLanding`) → `library_selected` → `alpha_dex_gui._landing_to_main` → main-window fade-in and deferred `set_library`. DS-01 cuts this hand-off.
- `bg_gradient_enabled` (config) is read by the GradientWidget, sidebar and landing. There is no UI for it.

## Bugs found
| ID | Severity | Location | Trigger | Effect | Confidence | Suggested fix |
|---|---|---|---|---|---|---|
| DS-01 | stability | `gui/widgets/landing.py:929` (+ `:986`, `:1011`) | Click "Initialize" about 0.4–0.92 s after the landing appears (during the logo pause) | The pending `singleShot(600, _fly_in)` fires during the scatter. Qt stops the scatter's `pos` animations (same target and property), the group never emits `finished`, `_fade_out` never runs and `library_selected` is never emitted. The frameless landing stays on screen forever and the main window never appears. Confirmed for accepts at 650, 750 and 850 ms (`probe_landing.py`). | confirmed-by-running | Hold the fly-in delay as a member `QTimer` and stop it, plus `_fly_in_grp`, in `_accept`. Add an `_exiting` guard to make `_accept` idempotent. Start the fade from `stateChanged` or a safety timeout, not only from `finished`. |
| DS-02 | wrong-result | `gui/workspaces/base.py:71`, `:79`; `gui/themes/effects.py:28` | Any card creation or theme switch | `setGraphicsEffect(card_shadow(...))` passes a temporary, and PySide6 6.12 does not keep it, so **0 of 42 cards ever get a shadow**. Minimal repro: `f.setGraphicsEffect(QGraphicsDropShadowEffect()); f.graphicsEffect() is None`. The `card_shadow` token and `refresh_shadows` are dead. | confirmed-by-running | Give the effect a parent (`QGraphicsDropShadowEffect(card)`), or better, paint the shadow (see recommendation 5) to avoid the blur cost on every child repaint. |
| DS-03 | stability | `gui/widgets/gradient_bg.py:21`, `:120`; `gui/widgets/sidebar.py:238`, `:250`; `gui/widgets/landing.py:1102` | Any repaint: hover, pill slide, progress tick, scrolling | `_gradient_enabled()` calls `config.load_config()` (opens the file and parses JSON) inside `paintEvent`. Measured 5 reads per full repaint, 42 per nav-pill slide, 22 per nav hover and about 1 per progress update. This causes jank on slow or anti-virus-scanned disks and may log warnings if the config is mid-write. | confirmed-by-running | Read the flag once into a module cache and invalidate it on `settings_saved`. Never do I/O in paint. |
| DS-04 | UX | `gui/themes/manager.py:274-276` | Click any swatch in the Theme picker | The UI freezes for about 0.8–1.1 s per switch (`setStyleSheet` 776 ms + `setStyle` 197 ms for 953 widgets) | confirmed-by-running | Use a static QSS with `palette()` references (or no app QSS), and swap tokens on one style instance followed by `update()` instead of `setStyle` and `setStyleSheet` on every switch. |
| DS-05 | UX | `gui/themes/animations.py:116-119` | Programmatic navigation: Ctrl+1..9, `navigate_requested`, Graph→Player `play_tracks`, landing tile → Player | The `active=True` setter forces `_hover_p=1.0`. On deactivation it is never animated back, so the previous item keeps a hover tint and bright text with no mouse over it. Probe: `_hover_p 1.0, underMouse False`; visible in `shot_pearl_indexer.png`. | confirmed-by-running | When the item is deactivated, call `_start_hover(self.underMouse())`. When it is activated, don't touch `_hover_p`. |
| DS-06 | crash | `gui/workspaces/tools.py:303` | Second status transition on a `_GlassResultChip` (latent: only one `show_neutral` per chip today) | `self._anim` points at an animation already deleted by `DeleteWhenStopped`, so `.stop()` raises `RuntimeError: Internal C++ object (QVariantAnimation) already deleted` and the chip text never updates | confirmed-by-running | Don't use `DeleteWhenStopped` for a stored reference. Reuse one parented animation, or clear `self._anim` on `finished`. |
| DS-07 | stability | `gui/workspaces/tools.py:214`, `:280`, `:786` | Delete a `_GlassBadge`, `_GlassResultChip` or `_PillToggle` (or a test tears down a window) and then switch theme | Lambdas connected to the process-wide `ThemeManager` keep calling `self.update()` on dead objects: `RuntimeError: libshiboken: Internal C++ object (_PillToggle) already deleted` on every later theme change | confirmed-by-running | Connect bound methods (PySide disconnects them automatically), or disconnect in `destroyed`. |
| DS-08 | cosmetic | `gui/workspaces/tools.py:555` | Any light theme on the Utilities page | `getattr(t, "is_dark", True)`: `ThemeTokens` has no `is_dark`, so light themes get the black tint meant for dark themes and tiles look muddy grey (`shot_pearl_tools.png`) | confirmed-by-running | Use `t.variant == "dark"`. |
| DS-09 | stability | `gui/workspaces/tools.py:405-408` | App running (any page) | Each of the 5 ToolTiles runs a 16 ms QTimer forever, about 315 wakeups/s while Tools is hidden, which wastes battery and competes with other animations | confirmed-by-running | Start the timer on `enterEvent` and stop it once the glow has settled, or replace it with a `QVariantAnimation`. Stop it in `hideEvent`. |
| DS-10 | wrong-result | `gui/themes/manager.py:309-312` | Choose "Auto" on a platform that reports `ColorScheme.Unknown` (many Linux desktops) | On Qt ≥ 6.5 the method returns `scheme == Light`, which is always False, so Auto is always dark and the documented 07:00–20:00 fallback never runs. There is also no timer to switch while the app is running. Probe at 18:00 local: `is_daytime()` returned False. | confirmed-by-running | Fall back to the clock when `scheme == Unknown`. Add a 15-minute re-check timer while in Auto. |
| DS-11 | UX | `gui/themes/style.py:830-837` | Any code that calls `setFont` on a button, combo, spin box or line edit (and any named role label) | `polish()` overwrites the font with Inter 11pt at first show **and on every theme switch**. Probe: `setFont(20pt)` became 11 after show, 20 after a re-set, then 11 again after a theme switch. | confirmed-by-running | Stop assigning fonts in `polish()`. Use an opt-in role property, or apply only when the widget's font is unresolved (`font().resolveMask()==0`). |
| DS-12 | cosmetic | `gui/widgets/sidebar.py:140`, `:184` | Any light theme | The sidebar logo is hard-coded `#f8fafc` (contrast 1.07 on Pearl, effectively invisible) and the separator is `rgba(255,255,255,0.08)` | confirmed-by-running | Paint them from `text_primary` and `card_border`, or drop the duplicate logo; the top bar already shows the brand. |
| DS-13 | cosmetic | `gui/workspaces/player.py:434-463`, `:714-722`, `:251-280` | Any light theme on the Player | The library table items, header, hover popup and background-art widget are hard-coded dark, leaving a black slab in a light UI (`shot_pearl_player.png`) | confirmed-by-running | Use token roles (`surface_2`, `overlay`) and repaint on `theme_changed`. |
| DS-14 | cosmetic | `gui/workspaces/compression.py:302` | Light theme, Compression result shown | Result label `#e2e8f0` on a white card has contrast 1.23, so the result text is unreadable | traced-in-code | Use `text_primary` and the `success`/`danger` tokens. |
| DS-15 | cosmetic | `gui/themes/picker.py:115` | Open the Theme picker while a light theme is active | Swatch names are drawn in the *current* theme's `text_primary` over each swatch's own (dark) colours, so dark-theme names are illegible (`shot_picker.png`) | confirmed-by-running | Draw the label below the mock in the dialog's colours, or pick the label colour from the swatch's own `text_primary`. |
| DS-16 | cosmetic | `gui/themes/picker.py:366` | Auto mode active during the day | The "Night:" label shows `mgr.current.name`, which is the light theme at that moment | traced-in-code | Always show `THEMES[mgr.auto_dark()].name`. |
| DS-17 | cosmetic | `gui/widgets/now_playing_bar.py:196-207` | Toggle Keyboard Mode twice within 120 ms | The geometry pulse runs on a layout-owned widget and the earlier animation is not stopped, leaving the button offset (952,16) → (954,17) until the next relayout | confirmed-by-running | Animate a painted scale value instead of `geometry`, and stop the previous animation. |
| DS-18 | cosmetic | `gui/widgets/log_drawer.py:65`, `:78`, `:113-121` | Toggle the Activity Log (Ctrl+L, or auto-expand on warn/error) | `maximumHeight` animates to 220 but the body's natural height is 192, so the last ~20% of expand and the first ~15% of collapse show no movement and the ease-out is cut off. Every frame relays out the whole workspace stack. Drawer colours are fixed to Midnight (`gui/theme.py`), so the `log_*` tokens are unused. | confirmed-by-running | Animate to `sizeHint()` or a fixed body height. Animate an overlay or painted offset instead of a layout constraint. Use the `log_*` tokens. |
| DS-19 | UX | `gui/themes/tokens.py` (all themes) | Reading any `statusHint`, `phaseDesc`, `notesHint` or inactive-step text, or primary-button labels | `text_muted` on `card_bg` is 1.5–2.75:1 in all 14 themes. `text_inverse` on `accent` is below 4.5 in 7 themes (obsidian 2.72, meadow 2.54, sunset 2.80, …). | confirmed-by-running (computed) | Re-tune the muted token to at least 3:1 (4.5:1 for functional text) and pick `text_inverse` per accent. Add a contrast unit test. |
| DS-20 | cosmetic | `gui/workspaces/tools.py:1143-1153`, `:653-663` | Any Tools export or cleanup finishes | A colour flash calls `setStyleSheet` on every frame for 1600 ms (about 100 restyles). Overlapping flashes on the same label are not stopped and fight each other. `flash_status` targets a hidden label. | traced-in-code | Paint the colour (or use the palette with `update()`), and keep one animation per label. |
| DS-21 | cosmetic | `gui/main_window.py:313-317` + `gui/workspaces/base.py:127-129` | Every theme switch | `refresh_shadows()` runs twice for each workspace (two effect allocations per card per switch); harmless today only because DS-02 discards them | traced-in-code | Keep a single listener (the base class one). |
| DS-22 | UX | `gui/workspaces/indexer.py:116`, `gui/workspaces/library_sync.py:218`, `gui/workspaces/duplicates.py:283` | Run any step of Indexer, Library Sync or Duplicates | The workflow stepper is decorative: labels are created once with step 1 active and never re-styled (no stored references, no `polish`), so the stepper never reflects progress | traced-in-code | Make the stepper a component with `set_step(i)` that toggles a dynamic property and re-polishes. |
| DS-23 | cosmetic | `gui/widgets/landing.py:1000` | Click Initialize again (or a tile) while tiles are scattering | `_accept` snaps every tile back to its grid slot mid-flight (probe: (297,212) → (235,135)) before restarting the scatter | confirmed-by-running | Make `_accept` idempotent (see DS-01). |
| DS-24 | cosmetic | `gui/themes/manager.py:138-168` vs `gui/fonts/loader.py:98-134` | Every page | Typography is inverted: card titles 13 px and subtitles 12 px sit below about 15 px body text, nav items are 17 px against an 18 px page heading, and `TypeScale.SECTION_HDR` is silently overridden by the QSS px value | confirmed-by-running | Adopt one px type ramp (recommendation 2). |
| DS-25 | cosmetic | `gui/widgets/interactive_scatter_plot.py:145`, `:230` | Switch theme after the Graph has been built, or use a light theme | The plot background is taken from the palette once at construction and never updated; the selection highlight pen is hard-coded white, which is invisible on light backgrounds | traced-in-code | Subscribe to `theme_changed` and take the highlight colour from `accent`. |
| DS-26 | cosmetic | `gui/widgets/landing.py:57` | Every startup with a saved library | `_ART_SCAN_DEBUG = True` prints per-file art-scan diagnostics to stderr in a shipping build | traced-in-code | Default it to False, or route it through `logging.debug`. |
