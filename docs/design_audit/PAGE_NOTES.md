# Per-page findings from the design pass

Specific, line-level findings produced during design convergence and kept out of
DESIGN_SPEC.md so the spec stays about rules rather than individual bugs.
Each implementation agent should read the section for its own page.

Items marked **VERIFIED** were independently re-checked against the source.

## Player (gui/workspaces/player.py)

### Must fix

- Do not reintroduce the WorkspaceBase scroll wrapper. Keep the _setup_scroll() override with contentsMargins(0,0,0,0) and spacing 0. The Player is the app's one sanctioned full-bleed page; every other page keeps the scroll wrapper.
- Replace the ad-hoc 50 px toolbar title with the shared full-bleed header region above the toolbar. Delete title_lbl.setFixedWidth(68) — a fixed width clips the title under wider theme fonts. The header's left inset MUST equal S.page_margin (24) so the 'Player' title's left edge is pixel-identical to the sectionTitle on the twelve card pages; this alignment is the single strongest signal that it is the same app.
- The header band is flush: no border-radius, no card_shadow, background from tokens.content_bg, separated from the toolbar by a 1 px tokens.card_border hairline. It must reuse the header's typography objectNames (sectionTitle + sectionSubtitle) verbatim — same type, different container.
- Every inset in the page comes from S. Toolbar contentsMargins → (S.lg, S.sm, S.lg, S.sm); right rail → (S.md, S.md, S.md, S.md) with spacing S.sm; transport → (S.lg, S.sm, S.lg, S.xs); splitter handle gap → S.xs. No new literal pixel values for spacing.
- Toolbar must not force horizontal scrolling at ~540 px content width. Below a ~760 px content width, collapse Reload / Browse… / Load M3U into a single ghostBtn overflow button ('⋯', tooltip 'More actions') opening a QMenu; keep only the search field and the 30s Preview toggle inline. Search field minimumWidth 140.
- Retire the fixed 50 px toolbar height in favour of a minimumHeight so the row grows with the theme's font rather than clipping.
- Button variants, applied by objectName only — never by stylesheet: play/pause = primaryBtn (the page's ONE primary, already correct); prev/next/stop = ghostBtn; shuffle, repeat and the 30s Preview control = segmentBtn (all three are checkable modes, and segmentBtn already fills with accent + inverse ink when checked, which is exactly the on-state these need); Reload / Browse… / Load M3U / Save Queue as M3U… / + Add Selected / Clear = ghostBtn; the Recently Played disclosure = ghostBtn. No dangerBtn and no successBtn anywhere on this page — nothing here writes to the library.
- Convert the 30s Preview QCheckBox and the shuffle QPushButton to one shape: both are persistent modes, so both become checkable segmentBtn. A checkbox and a fill-when-on button currently express the same idea two ways on one strip.
- Strip emoji from label strings: '⟳  Reload' → 'Reload', '💾  Save Queue as M3U…' → 'Save Queue as M3U…', the '🔊' volume QLabel → a themed speaker icon or nothing (the slider's tooltip already says 'Volume'), '▶' on the Recently Played disclosure → a chevron glyph. EXCEPTION, and it is deliberate: the transport glyphs ⏮ ⏭ ■ ⤨ ↻ ▶ stay. They are universal media-control symbols, not decoration, and the existing code comment correctly explains that text-presentation glyphs are used precisely so they take the theme's button ink instead of the emoji font's own palette. Do not 'fix' them into emoji, and do not swap them for icon files.
- Delete every hardcoded font-size from this page. _apply_theme() currently bakes 'font-size: 11px' into five labels (muted/second strings, _status_lbl) and _np_title hardcodes 'font-size: 13px; font-weight: 600', and vol_lbl hardcodes 12px. Replace with objectNames the QSS layer owns: _np_title → cardTitle; _np_artist, _q_count_lbl, _pos_lbl, _dur_lbl → statusHint; _vlc_status_lbl and _status_lbl → statusHint. _apply_theme() then sets colour only where QSS cannot reach (the table, the progress chunk, the separator) and never a font size.
- The bottom status line uses the same statusHint component and the same wording register as every other page. It keeps its padding from S (padding: S.xs S.lg S.xs) rather than the current literal '2px 18px 4px'.
- Splitter sizing must survive the 540 px floor: right rail keeps minimumWidth 240 / maximumWidth 340; give _lib_table a minimumWidth of 260; setChildrenCollapsible(False) stays. When content width drops below ~560 px, hide the right rail and surface it as a segmentBtn 'Queue' toggle in the toolbar rather than letting the table squeeze below 260.
- _apply_theme() must keep driving the table, header sections, progress chunk and separator from tokens — that migration landed this session and is correct. Keep using _rgba(...) against token values; never introduce a literal colour.
- Nothing in this pass may touch playback logic, VLC handling, the QThread art/scan loaders, or any of the contracted signals/methods/_vol_slider. If a layout change appears to require renaming a contracted member, the layout change is wrong.
- The transport bar already carries objectName 'transportBar' but the QSS layer has no rule for it. Add one in gui/themes/manager.py (tokens.card_bg fill, top hairline tokens.card_border, no radius) instead of styling it from the workspace.

### Must NOT change (already correct)

- The _setup_scroll() override and the full-bleed, non-scrolling page shape. Do not card-ify the Player.
- The transport glyphs ⏮ ⏭ ■ ⤨ ↻ ▶ as text-presentation characters, and the reasoning comment above _shuffle_btn explaining why they are not emoji.
- The region order itself: toolbar → content → transport → status is correct for a player and must not be rearranged.
- The library-table / right-rail QSplitter, its setChildrenCollapsible(False), and the 720/280 default sizes.
- The _BgArtWidget translucent-table treatment — semi-transparent rows over background art is a genuine media-view affordance and themes correctly through _apply_theme.
- The 1.5 s dwell hover art popup and its event-filter plumbing.
- The entire external contract: now_playing_changed, playback_state_changed, position_changed, _on_play_pause, play_next, play_prev, set_volume, seek_to_ms, load_tracks_and_play, all kb_* methods, and the _vol_slider attribute.
- sliderMoved/sliderReleased (rather than valueChanged) on the volume slider — the comment correctly notes this prevents a feedback loop when set_volume() repositions it programmatically.

## Utilities (gui/workspaces/tools.py)

### Must fix

- PROMOTE _GlassResultChip → gui/widgets/status_pill.py as StatusPill, and tokenise it in the same change: _COLOR's #94a3b8 → tokens.text_muted, #22c55e → the theme's success token, #ef4444 → the theme's danger token, resolved at paint time (it already repaints on theme_changed, so the plumbing exists). Keep the show_neutral/show_success/show_error API and the 160 ms out / 260 ms in crossfade. This is the highest-value promotion on the page: it gives every workflow a single 'Ready.' → 'Done' → 'Error' vocabulary.
- PROMOTE _GlassBadge → gui/widgets/icon_badge.py as IconBadge, and fix its contrast while promoting. It paints ~125 total alpha of accent over an unknown background, then draws the glyph in tokens.text_inverse — on light themes that is inverse ink on a pale wash. Resolve the ink against the composited fill (or simply use tokens.accent ink on the tint) and replace the literal white specular/rim gradients with token-derived values. Align its corner radius to a fraction of R.card rather than the magic 0.28.
- PROMOTE TileGrid → gui/widgets/tile_grid.py as ResponsiveTileGrid, unchanged in behaviour but with _TILE_MIN_WIDTH as a constructor parameter (the hardcoded 500 is right for tool tiles, wrong for smaller cards). Its resize-driven reflow with full-width spanning is the only responsive container in the app and other pages need it.
- PROMOTE _PillToggle → gui/widgets/toggle_switch.py as ToggleSwitch, and publish the scoping rule with it so it does not metastasise: a ToggleSwitch means 'a persistent mode or option that takes effect immediately'; a QCheckBox means 'an item selected inside a set, or an input to a plan that has not run yet'. Utilities' options (Exclude FLAC, Include duplicate titles, Filenames only) are modes and keep switches. Do NOT retro-fit the other twelve pages' checkboxes — a preview/plan input must stay a checkbox.
- RETIRE _make_chip() and _refresh_chip() entirely. The codec chips are checkable pills that fill with accent when on — which is exactly what segmentBtn now paints in gui/themes/style.py. Re-express them as checkable QPushButtons with objectName 'segmentBtn'. This deletes two hand-rolled stylesheets, two literal 'font-size: 13px' rules, the _codec_chips dict, and the private _on_theme_changed_base refresh loop, and makes chips look identical to toggles everywhere else in the app.
- FIX the diagnostics buttons — this is the change that most stops Utilities being a private dialect. Every other button in the app is icon-then-label; _make_diag_button() puts the label above the icon. Rebuild each as a plain QPushButton with setIcon() + setText() (Qt's native icon-left layout), objectName 'ghostBtn', minimumHeight 34, and delete the whole inner QVBoxLayout / QLabel / WA_TransparentForMouseEvents / setFixedHeight(72) construction along with its 'font-size: 10px' override. Lay them out in the ResponsiveTileGrid or a wrapping QHBoxLayout instead of the fixed 4-column grid with spacer padding.
- ADD a #logBox rule to gui/themes/manager.py (tokens.input_bg fill, R.input radius, MONO_STACK, tokens.text_secondary ink) and delete both inline 'background: rgba(0,0,0,0.1); border-radius: 6px' stylesheets — ToolTile.log_box and _build_validator_tile's _val_log. A black wash is invisible on dark themes and a grey smear on light ones. Every page's log surface then inherits one look.
- REPLACE the hex-taking status APIs with semantic ones: ToolTile.flash_status(from_hex) and ToolsWorkspace._flash_label(label, from_hex) both require the caller to supply a literal colour, which is how hardcoded colours get in. Change the signatures to take a state ('success' | 'error' | 'neutral') and resolve against tokens internally. Also drop the 'font-size: 12px' that _flash_label re-applies on every animation frame.
- FIX the idle repaint cost: ToolTile's constructor starts a 16 ms QTimer that never stops, so all five tiles animate their glass wash at 60 fps permanently. Start the timer in enterEvent and stop it in leaveEvent once _hover_t has settled back to 0.
- Tokenise ToolTile.paintEvent: the bg_alpha / t_alpha washes built from QColor(255,255,255, α) must derive from tokens.card_bg and tokens.card_border so light themes get a card, not a fog. Bring its radius from the magic 16.0 to R.card (12) so tool tiles have the same corner as workspaceCard.
- Take every inset from S. ToolTile's header (18,16,18,12), options (18,0,18,12), footer (18,8,18,16) and drawer (18,10,18,16) become S.lg horizontally with S.md / S.sm vertically; TileGrid's setSpacing(16) becomes S.card_gap.
- Keep the tile's header ▸ options ▸ footer ▸ drawer contract and its public API (add_option, add_option_layout, set_run_button, add_secondary_button, add_icon_button, finish_footer, hide_footer, open_drawer, set_running). Existing tool slots bind to tile.status_label / log_box / progress_bar by attribute; do not rename them.
- Footer button variants: set_run_button keeps primaryBtn; add_secondary_button must set objectName 'secondaryBtn' (it currently returns an unstyled default button, which is why it reads as a leftover next to the accent primary); add_icon_button becomes ghostBtn at a square 34×34. No tool on this page writes destructively, so no dangerBtn here.
- Adopt the shared page header for 'Export & Utilities' — including its subtitle, which this page already writes well and which should be preserved verbatim as the model for other pages.
- Do not move any worker logic. FileCleanupWorker and ArtistTitleWorker stay QThreads reporting through signals; the promotion is purely a widget move.

### Must NOT change (already correct)

- The ToolTile archetype itself — header ▸ options ▸ footer ▸ animated log drawer — and its whole public API. It is the best card in the app and other workflow pages should converge toward it, not away.
- The open_drawer() 0→natural-height animation and the separator that only appears once the drawer opens. Progress and logs staying hidden until a run starts is correct.
- TileGrid's width-driven column count and full-width-tile-starts-a-new-row behaviour, including the 500 px min width that correctly yields one column at the ~540 px content floor.
- The status pill's crossfade timing (160 ms out / 260 ms in) and its Idle/Done/Error glyph prefixes.
- The subtitle text explaining that all exports land in Docs/ inside the library folder — concrete, non-obvious, and the right register for every page's subtitle.
- The per-tile 'Open File' + 'Open folder' affordance pair after a successful export, and keeping them disabled until output exists.
- Utilities' habit of reading tokens at paint time rather than caching them. That is the pattern the other twelve pages need to copy.

## Help (gui/workspaces/help.py)

### Must fix

- Add the missing subtitle — this is the only page in the app without one, and its absence is why the page reads as unfinished. Use _make_subtitle() with concrete text in the same register as Utilities', e.g. 'Open the local documentation, review keyboard shortcuts, and report problems.' Do not ship a generic one-liner.
- Replace the three full-width QPushButtons in the Documentation card with LinkRow instances: icon at the left, label left-aligned, a trailing external-link chevron, ghostBtn painting (chromeless until hover), full row width, minimumHeight 34. A full-width button with centred text is the shape this page must stop using.
- Strip the '📄' emoji from the link labels. The icon belongs to the LinkRow component (a themed glyph or Qt standard icon), never baked into the label string.
- Delete about_text.setStyleSheet('color: #64748b; font-size: 12px;'). Give the label objectName 'sectionSubtitle' (or statusHint for the denser config/formats lines) and let the QSS layer colour and size it. This is one of the app's clearest hardcoded-colour violations and it lives in the shortest file.
- Source the shortcut table from a single registry (see componentsRequired) instead of the hardcoded five-tuple list. The table must render every binding the app actually installs, grouped by scope: Global (Ctrl+O, Ctrl+comma, Ctrl+L, Ctrl+1–9, Ctrl+W) and Player (Space, P, N, and the kb_* navigation keys installed by player.py's _install_shortcuts). Today the Player's keys are entirely absent from Help, so the page is factually wrong as shipped.
- Keep the QTableWidget for shortcuts — a two-column key/action grid is the right form and it is already correctly configured (no edit triggers, no selection, no vertical header, stretch last section). Replace only the magic setFixedHeight(len*30+28) with a height computed from the table's own row height and header height so it does not clip under themes with larger fonts, and wrap it in an overflow-x container per the responsive rule.
- Give the shortcut key column a monospace treatment from the existing MONO_STACK (via a #shortcutKey QSS rule or the table's font role) so keys read as keys. Do not hardcode a font family or a px size at the call site.
- 'Report an Issue (GitHub)' becomes the page's single accented control with objectName 'secondaryBtn'. It must not be primaryBtn: nothing on this page commits anything, and a blue primary here would compete with the real primaries on the twelve workflow pages. It is not full-width — size it to content and left-align it inside the About card.
- Do not invent a primary action, a stepper, a progress idiom, or a status line for this page. It has no run, no preview and no commit. Absence of a primary action is the correct answer for a reference page, and the header plus the first card is a sufficient entry point.
- Take all card insets from S: the three cards' literal (16,16,16,16) become S.card_padding and the literal setSpacing(8) becomes S.row_gap. Card headings use _make_card_title() rather than a bare QtWidgets.QLabel — 'Documentation', 'Keyboard Shortcuts' and 'About AlphaDEX' are currently unstyled plain labels.
- Keep the graceful-degradation path in _open_doc(): checking p.exists() and showing an information box rather than opening a dead file:// URL is correct behaviour and must survive the rewrite.
- The page stays inside the WorkspaceBase scroll wrapper with _make_card() cards. It is the simplest possible instance of the standard archetype and should be the reference example of it.

### Must NOT change (already correct)

- The content set and its order: docs links, keyboard shortcuts, About, Report an Issue. Nothing needs adding or removing.
- Having no primary action. Do not manufacture one.
- The QTableWidget for shortcuts, with NoEditTriggers, NoSelection, hidden vertical header, and stretchLastSection.
- The three-card structure on the standard scroll wrapper — this page is the cleanest example of the default archetype in the app.
- _open_doc()'s exists() check and its 'Not Found' information dialog.
- The About content itself — naming the config path (~/.soundvault_config.json) and the supported extensions is genuinely useful for a single-user local tool.

## Verification notes

- **VERIFIED — `tools.py` ToolTile timer.** `ToolTile.__init__` starts a 16 ms `QTimer`
  unconditionally (`gui/workspaces/tools.py:405-408`) and never stops it; `enterEvent`/
  `leaveEvent` only toggle `_is_hovered`. With 7 tiles that is ~420 timer callbacks/sec for
  the life of the process, even while the page is hidden.
  **Severity corrected:** `_tick` DOES early-return before `self.update()` when nothing is
  animating (`tools.py:529-531`), so this is *not* 60 fps repainting as first reported. The
  cost is the wakeups themselves, which stop the CPU idling. Fix by starting the timer in
  `enterEvent` and stopping it in `leaveEvent` once `_hover_t` has settled — worthwhile, but
  an efficiency fix, not a rendering bug.

- **VERIFIED — `help.py` shortcut table drift.** The table is a hardcoded 5-tuple literal.
  `gui/main_window.py` installs Ctrl+O, Ctrl+comma, Ctrl+L, Ctrl+W and Ctrl+1..9, while
  `player.py` installs its own transport and `kb_*` navigation bindings that the table never
  mentions. The page under-reports the app's real bindings today.
