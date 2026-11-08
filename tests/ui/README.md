# AlphaDEX GUI test harness

Headless pytest-qt harness for the PySide6 app (`alpha_dex_gui.py`). It is the
safety net for the GUI redesign: every page is exercised against a real
(synthetic) music library, in every theme, with the failure modes that made
the app feel unstable turned into test failures.

```bash
pip install -r requirements-dev.txt      # PySide6, pytest-qt, pytest-timeout, ...
pytest tests/ui                          # GUI suite only (~30 s)
pytest                                   # everything
pytest tests/ui/test_ui_responsiveness.py -s   # print freeze measurements
python scripts/gui_snapshots.py --themes all   # PNG contact sheet of every page
```

No display is needed (`QT_QPA_PLATFORM=offscreen` is set by the conftests).
Without PySide6 or pytest-qt installed, `tests/ui` is skipped, not failed.

## What every GUI test gets automatically

| Guard | What it catches | Where |
|---|---|---|
| **Isolated HOME** | Tests can never read or clobber the real `~/.soundvault_config.json` (applies to the *whole* suite, not just GUI tests). | `tests/conftest.py::_isolated_home` |
| **DialogGuard** (`dialogs`) | A modal dialog nobody planned for fails the test instead of hanging the run. Decision dialogs (`question`, file pickers, `QDialog.exec`, popup menus) need a queued answer; info/warning/critical boxes are auto-acknowledged and recorded. | `ui_harness.DialogGuard` |
| **ThreadRecorder** (`threads`) | Fails a test that leaves a `QThread` running (the cause of `QThread: Destroyed while thread is still running` aborts). | `ui_harness.ThreadRecorder` |
| **MotionRecorder** (`motion`) | Records every animation (target, property, duration, easing, parent). Compresses durations to ~1 ms by default so animated flows reach their end state instantly. | `ui_harness.MotionRecorder` |
| **Qt-log policy** | Fails on Qt warnings that mean real bugs: cross-thread widget/timer use, illegal painting, animating a missing property, bad `connect`. | `ui_harness.QtWarningPolicy` |
| **pytest-qt exception capture** | Any exception raised inside a slot, paint event or other Qt virtual fails the test. | pytest-qt |
| **Theme reset** | The `ThemeManager` singleton is reset between tests. | `conftest._reset_theme_manager` |
| **Timeout** | A test hung inside a Qt event loop is killed after 300 s (`timeout_method = thread`). | `pytest.ini` |

## Fixtures

```python
def test_example(main_window, library, dialogs, threads, motion, qtbot):
    main_window.set_library(str(library.root))   # real tagged files on disk
    assert threads.wait_all(10_000) == []        # let background scans finish
    dialogs.answer_yes()                          # queue the confirm dialog's answer
    ...
```

* `main_window`: a shown `AlphaDEXWindow` with no library selected, its housekeeping (the backdrop build,
  a quiet job) settled. It opens on Home.
  `main_window._workspaces[panel_key]` is a panel (the old workspace keys still work),
  `main_window._pages[page_key]` a page; `current_page()` / `current_panel()` say what is
  on screen. Navigate with `_on_nav_changed("page/section")` (or any key from
  `gui/navigation.py`), or `select_page(page_key)` as the sidebar does.
* `make_workspace(key, library_path=...)`: one panel on its own (keys from `gui.main_window._WORKSPACE_MAP`).
* `library` / `synthetic_library`: a fresh copy of the standard synthetic library from `tests/library_factory.py`:
  an MP3 album, a FLAC album, a cross-codec duplicate, M4A/OGG/Opus tracks, an untagged file, an artist-and-title-only file,
  non-audio leftovers and an `.m3u`. Build custom libraries with `library_factory.build_library(root, [TrackSpec(...)])`.
* `dialogs`: queue answers with `answer_yes()`, `answer_no()`, `choose_directory(p)`, `choose_open_file(p)`,
  `choose_save_file(p)`, `accept_file_dialog(p)` (instance `QFileDialog().exec()`), `on_exec(fn_or_code)`,
  `on_menu("Action text" | None | fn)`. Inspect with `dialogs.calls` / `dialogs.calls_of(kind)`.
* `threads.wait_all(ms)` returns the names of threads still running (empty list when all are done).
* `motion.settle(ms)`, `motion.records`, `motion.orphans()`, plus `ui_harness.stuck_opacity(widget)`.
* `ui_harness.FrameGapMeter`: measures how long the GUI thread is blocked (UI freezes and dropped frames).

Markers: `@pytest.mark.motion(time_scale=1.0)` (real-time animations),
`@pytest.mark.allow_qt_warnings("regex")`, `@pytest.mark.allow_running_threads`.

## Writing worker code the harness (and users) can live with

Start every `QThread` through `WorkspaceBase._start_worker(worker, "Name")` (or
`gui.workers.start_worker`). The `JobRegistry` keeps the thread alive until
the real C++ `QThread::finished()` arrives, so dropping `self._worker` in a
slot, or re-running a job, can't destroy a running thread. The main window's
`closeEvent` asks before quitting with jobs running, cancels them
cooperatively (`worker.cancel()` and `requestInterruption()`), and waits. Give
long-running workers a `cancel()` method.

## Static ratchets (no Qt needed)

* `tests/test_gui_hygiene_ratchets.py` counts hex colours outside tokens, inline `setStyleSheet`
  calls, `font-size` literals, direct `QThread.start()` calls and unparented animations,
  and fails if any count goes above its ceiling. Ceilings can only come down. Lambdas
  connected to `theme_changed` have a zero-tolerance ceiling.
* `tests/ui/test_ui_static_qt_api.py` resolves every `QtCore.*`/`QtGui.*`/`QtWidgets.*` name the GUI
  references against the installed PySide6. Misspelled or Qt5-era names (like
  `QFont.Weight.SemiBold` or `QtWidgets.QShortcut`) fail here without the code path
  having to run.
