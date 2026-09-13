# Workspace API contracts — must survive any redesign

Anything listed here is referenced from OUTSIDE the page's own file.
A redesign may move or restyle widgets, but these names must keep working.

## clustered  (`ClusteredWorkspace`)
- No external method contract beyond WorkspaceBase.

## clustered_enhanced  (`EnhancedClusteredWorkspace`)
- **Methods that must survive:**
  - `closeEvent(event)` — public

## compression  (`CompressionWorkspace`)
- No external method contract beyond WorkspaceBase.

## duplicates  (`DuplicatesWorkspace`)
- No external method contract beyond WorkspaceBase.

## genres  (`GenresWorkspace`)
- No external method contract beyond WorkspaceBase.

## graph  (`GraphWorkspace`)
- **Methods that must survive:**
  - `reload_data()` — public

## help  (`HelpWorkspace`)
- No external method contract beyond WorkspaceBase.

## indexer  (`IndexerWorkspace`)
- No external method contract beyond WorkspaceBase.

## library_sync  (`LibrarySyncWorkspace`)
- No external method contract beyond WorkspaceBase.

## player  (`PlayerWorkspace`)
- **Signals:** `now_playing_changed`, `playback_state_changed`, `position_changed`
- **Methods that must survive:**
  - `_on_play_pause()` — used-externally
  - `set_volume(value)` — public
  - `seek_to_ms(ms)` — public
  - `play_next()` — public
  - `play_prev()` — public
  - `load_directory_and_play(dirpath)` — public
  - `load_tracks_and_play(paths, label)` — public
  - `toggle_shuffle()` — public
  - `eventFilter(obj, event)` — public
  - `set_keyboard_mode(active)` — public
  - `kb_target()` — public
  - `kb_lib_down()` — public
  - `kb_lib_right()` — public
  - `kb_lib_left()` — public
  - `kb_lib_up()` — public
  - `keyPressEvent(event)` — public
  - `_preview_next()` — used-externally
  - `_restart_current()` — used-externally
  - `toggle_repeat()` — public

## playlists  (`PlaylistsWorkspace`)
- No external method contract beyond WorkspaceBase.

## similarity  (`SimilarityWorkspace`)
- No external method contract beyond WorkspaceBase.

## tag_fixer  (`TagFixerWorkspace`)
- No external method contract beyond WorkspaceBase.

## tools  (`ToolsWorkspace`)
- No external method contract beyond WorkspaceBase.


## Externally-accessed ATTRIBUTES (break silently if renamed)

- `PlayerWorkspace._vol_slider` — `gui/main_window.py:156-158` connects
  `.sliderMoved` and `.sliderReleased` and reads `.value()`. Must remain a
  QSlider attribute with that exact name.

## Main-window wiring that must keep working (gui/main_window.py:140-170)

PlayerWorkspace <-> NowPlayingBar is wired by the main window:
  player -> bar: now_playing_changed, playback_state_changed, position_changed
  bar -> player: _on_play_pause, play_next, play_prev, _vol_slider
Any Player redesign MUST keep these names and signal signatures intact.
