"""Reusable pieces of the AlphaDEX Qt test harness.

Everything here is plain Python + Qt so it can be used from pytest fixtures
(see ``tests/ui/conftest.py``) *and* from standalone scripts such as
``scripts/gui_snapshots.py``.

Components
----------
``DialogGuard``
    Replaces every blocking Qt dialog entry point (``QMessageBox`` statics,
    ``QFileDialog`` statics, ``QInputDialog`` statics, ``QDialog.exec``,
    ``QMenu.exec``). A test *queues* the answer it wants; a decision dialog
    nobody queued an answer for raises ``UnexpectedDialog`` instead of hanging
    the run forever. Purely informational boxes (information / warning /
    critical) are acknowledged automatically and recorded so tests can assert
    on them.

``ThreadRecorder``
    Records every ``QThread`` started while it is installed, so a test can
    wait for workers and the harness can fail tests that leave threads
    running (the source of "QThread: Destroyed while thread is still running"
    aborts, which kill the whole interpreter).

``MotionRecorder``
    Records every ``QAbstractAnimation`` started (target, property, duration,
    easing, parent), can optionally compress durations so animated flows
    finish instantly in tests, and offers helpers to settle animations and to
    check the end state (no stuck opacity effects, no orphan animations).

``QtWarningPolicy``
    Patterns for Qt log messages that indicate real thread-safety or painting
    bugs. The conftest fails a test when one of them is logged.
"""
from __future__ import annotations

import collections
import re
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401  (QtGui used in annotations)


# ══════════════════════════════════════════════════════════════════════════════
# Dialog guard
# ══════════════════════════════════════════════════════════════════════════════

class UnexpectedDialog(AssertionError):
    """A blocking dialog opened that the test did not plan for."""


@dataclass
class DialogCall:
    kind: str                 # e.g. "question", "getExistingDirectory", "exec"
    title: str = ""
    text: str = ""
    widget: Any = None        # the dialog/menu instance for exec-style calls
    result: Any = None


_INFO_KINDS = ("information", "warning", "critical", "about", "aboutQt")
_STATIC_KINDS = {
    QtWidgets.QMessageBox: ("question", *_INFO_KINDS),
    QtWidgets.QFileDialog: (
        "getExistingDirectory", "getOpenFileName", "getOpenFileNames",
        "getSaveFileName", "getExistingDirectoryUrl", "getOpenFileUrl",
        "getOpenFileUrls", "getSaveFileUrl",
    ),
    QtWidgets.QInputDialog: ("getText", "getItem", "getInt", "getDouble", "getMultiLineText"),
    QtWidgets.QColorDialog: ("getColor",),
    QtWidgets.QFontDialog: ("getFont",),
}


class DialogGuard:
    """Intercepts blocking dialogs. Install with ``install(monkeypatch)``.

    Queue answers *before* triggering the code that opens the dialog::

        dialogs.answer("question", QtWidgets.QMessageBox.StandardButton.Yes)
        dialogs.choose_directory(tmp_path)
        dialogs.on_exec(lambda dlg: QtWidgets.QDialog.DialogCode.Accepted)
        dialogs.on_menu("Copy")          # or a callable(menu) -> QAction, or None

    After the action, inspect ``dialogs.calls`` / ``dialogs.calls_of(kind)``.
    """

    def __init__(self) -> None:
        self.calls: list[DialogCall] = []
        self._answers: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.strict_info = False  # True → information/warning/critical also need an answer

    # ── queueing API ───────────────────────────────────────────────────────
    def answer(self, kind: str, value: Any) -> "DialogGuard":
        self._answers[kind].append(value)
        return self

    def answer_yes(self) -> "DialogGuard":
        return self.answer("question", QtWidgets.QMessageBox.StandardButton.Yes)

    def answer_no(self) -> "DialogGuard":
        return self.answer("question", QtWidgets.QMessageBox.StandardButton.No)

    def choose_directory(self, path: Any) -> "DialogGuard":
        return self.answer("getExistingDirectory", str(path))

    def cancel_directory(self) -> "DialogGuard":
        return self.answer("getExistingDirectory", "")

    def choose_open_file(self, path: Any, selected_filter: str = "") -> "DialogGuard":
        return self.answer("getOpenFileName", (str(path), selected_filter))

    def choose_open_files(self, paths: Iterable[Any], selected_filter: str = "") -> "DialogGuard":
        return self.answer("getOpenFileNames", ([str(p) for p in paths], selected_filter))

    def choose_save_file(self, path: Any, selected_filter: str = "") -> "DialogGuard":
        return self.answer("getSaveFileName", (str(path), selected_filter))

    def on_exec(self, handler: Callable[[QtWidgets.QDialog], Any] | int) -> "DialogGuard":
        """Answer the next ``dialog.exec()``: an int result code, or a callable
        that receives the (already constructed) dialog and returns the code."""
        return self.answer("exec", handler)

    def accept_file_dialog(self, path: Any) -> "DialogGuard":
        """Answer the next *instance* ``QFileDialog(...).exec()`` by selecting
        ``path`` and accepting (the landing and top bar use this style)."""
        def handler(dlg):
            if isinstance(dlg, QtWidgets.QFileDialog):
                if dlg.fileMode() == QtWidgets.QFileDialog.FileMode.Directory:
                    dlg.setDirectory(str(path))
                dlg.selectFile(str(path))
            return QtWidgets.QDialog.DialogCode.Accepted
        return self.on_exec(handler)

    def on_menu(self, handler: Callable[[QtWidgets.QMenu], Any] | str | None) -> "DialogGuard":
        """Answer the next ``menu.exec()``: an action text to pick, ``None`` to
        dismiss, or a callable receiving the menu and returning a QAction/None."""
        return self.answer("menu", handler)

    def calls_of(self, kind: str) -> list[DialogCall]:
        return [c for c in self.calls if c.kind == kind]

    def pending(self) -> dict[str, int]:
        """Answers that were queued but never consumed (usually a test bug)."""
        return {k: len(v) for k, v in self._answers.items() if v}

    # ── installation ───────────────────────────────────────────────────────
    def install(self, monkeypatch) -> "DialogGuard":
        for cls, kinds in _STATIC_KINDS.items():
            for kind in kinds:
                if hasattr(cls, kind):
                    monkeypatch.setattr(cls, kind, staticmethod(self._make_static(kind)))
        guard = self

        def dialog_exec(dlg, *args, **kwargs):
            return guard._handle_exec(dlg)

        monkeypatch.setattr(QtWidgets.QDialog, "exec", dialog_exec)
        if hasattr(QtWidgets.QDialog, "exec_"):
            monkeypatch.setattr(QtWidgets.QDialog, "exec_", dialog_exec)

        # QMenu.exec cannot be monkeypatched: Shiboken binds the overloaded
        # static/instance ``exec`` natively, ignoring the class attribute. So
        # menus are intercepted when they are *shown*: the guard then picks
        # the queued action (or dismisses the menu) from inside exec's nested
        # event loop, and exec returns exactly what it would for a real user.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            self._menu_filter = _MenuShowFilter(self)
            app.installEventFilter(self._menu_filter)
        return self

    def uninstall(self) -> None:
        app = QtWidgets.QApplication.instance()
        flt = getattr(self, "_menu_filter", None)
        if app is not None and flt is not None:
            app.removeEventFilter(flt)
            self._menu_filter = None

    # ── handlers ───────────────────────────────────────────────────────────
    def _make_static(self, kind: str) -> Callable[..., Any]:
        def handler(*args, **kwargs):
            title, text = _title_text(args)
            call = DialogCall(kind=kind, title=title, text=text)
            self.calls.append(call)
            queue = self._answers.get(kind)
            if queue:
                call.result = queue.popleft()
                return call.result
            if kind in _INFO_KINDS and not self.strict_info:
                call.result = QtWidgets.QMessageBox.StandardButton.Ok
                return call.result
            raise UnexpectedDialog(
                f"Unexpected blocking dialog {kind}(title={title!r}, text={text[:200]!r}). "
                f"Queue an answer with dialogs.answer({kind!r}, ...) before triggering it."
            )
        return handler

    def _handle_exec(self, dlg: QtWidgets.QDialog) -> Any:
        title = dlg.windowTitle() if isinstance(dlg, QtWidgets.QWidget) else ""
        text = dlg.text() if isinstance(dlg, QtWidgets.QMessageBox) else ""
        call = DialogCall(kind="exec", title=title, text=text, widget=dlg)
        self.calls.append(call)
        queue = self._answers.get("exec")
        if not queue:
            raise UnexpectedDialog(
                f"Unexpected modal {type(dlg).__name__}.exec() (title={title!r}). "
                "Queue a result with dialogs.on_exec(...) before triggering it."
            )
        handler = queue.popleft()
        result = handler(dlg) if callable(handler) else handler
        if isinstance(result, QtWidgets.QDialog.DialogCode):
            result = result.value
        call.result = result
        if isinstance(dlg, QtWidgets.QDialog) and isinstance(result, int):
            dlg.setResult(result)
        return result

    def _handle_menu(self, menu: QtWidgets.QMenu) -> None:
        """Called (via a zero-delay timer) once a popup QMenu is on screen."""
        try:
            if not menu.isVisible():
                return
        except RuntimeError:
            return
        call = DialogCall(kind="menu", title=menu.title(), widget=menu,
                          text=" | ".join(a.text() for a in menu.actions()))
        self.calls.append(call)
        queue = self._answers.get("menu")
        if not queue:
            menu.close()
            raise UnexpectedDialog(
                f"Unexpected popup menu with actions [{call.text}]. "
                "Queue a choice with dialogs.on_menu(...) before triggering it."
            )
        handler = queue.popleft()
        if handler is None:
            action = None
        elif callable(handler):
            action = handler(menu)
        else:
            action = find_action(menu, str(handler))
            if action is None:
                menu.close()
                raise UnexpectedDialog(f"Menu has no action {handler!r}; actions: [{call.text}]")
        call.result = action
        if action is None:
            menu.close()
            return
        # Activate exactly the way a keyboard user would, so QMenu.exec()
        # returns the action and its ``triggered`` signal fires.
        from PySide6 import QtTest
        menu.setActiveAction(action)
        QtTest.QTest.keyClick(menu, QtCore.Qt.Key.Key_Return)
        if menu.isVisible():  # e.g. action lives in a submenu
            action.trigger()
            menu.close()


class _MenuShowFilter(QtCore.QObject):
    def __init__(self, guard: DialogGuard) -> None:
        super().__init__()
        self._guard = guard

    def eventFilter(self, obj, event):  # noqa: N802
        if (
            event.type() == QtCore.QEvent.Type.Show
            and isinstance(obj, QtWidgets.QMenu)
            and obj.isWindow()
        ):
            menu = obj
            QtCore.QTimer.singleShot(0, lambda: self._guard._handle_menu(menu))
        return False


def find_action(menu: QtWidgets.QMenu, text: str) -> "QtGui.QAction | None":
    """Depth-first search for an action whose text (sans '&') matches."""
    wanted = text.replace("&", "").strip().lower()
    for action in menu.actions():
        label = action.text().replace("&", "").strip().lower()
        if label == wanted or label.startswith(wanted):
            return action
        sub = action.menu()
        if sub is not None:
            found = find_action(sub, text)
            if found is not None:
                return found
    return None


def _title_text(args: tuple) -> tuple[str, str]:
    """Best-effort (title, text) from static dialog args (parent, title, text...)."""
    strs = [a for a in args if isinstance(a, str)]
    title = strs[0] if strs else ""
    text = strs[1] if len(strs) > 1 else ""
    return title, text


# ══════════════════════════════════════════════════════════════════════════════
# Thread recorder
# ══════════════════════════════════════════════════════════════════════════════

class ThreadRecorder:
    """Track every QThread started while installed."""

    def __init__(self) -> None:
        self._threads: list[tuple[weakref.ref, str]] = []

    def install(self, monkeypatch) -> "ThreadRecorder":
        original = QtCore.QThread.start
        recorder = self

        def start(thread, *args, **kwargs):
            recorder._threads.append((weakref.ref(thread), type(thread).__name__))
            return original(thread, *args, **kwargs)

        monkeypatch.setattr(QtCore.QThread, "start", start)
        return self

    @property
    def started(self) -> list[str]:
        return [name for _, name in self._threads]

    def alive(self) -> list[QtCore.QThread]:
        out = []
        for ref, _name in self._threads:
            t = ref()
            if t is None:
                continue
            try:
                if t.isRunning():
                    out.append(t)
            except RuntimeError:  # C++ object already deleted
                continue
        return out

    def wait_all(self, timeout_ms: int = 5000) -> list[str]:
        """Wait for recorded threads (processing events meanwhile, so queued
        signals from the workers are delivered). Returns names still running."""
        deadline = time.monotonic() + timeout_ms / 1000
        app = QtWidgets.QApplication.instance()
        while time.monotonic() < deadline:
            running = self.alive()
            if not running:
                # Every recorded thread has finished, so whatever it emitted
                # is already queued: deliver it, and stop only if that started
                # no new thread. (Checking alive() right *after* a
                # processEvents raced with a thread finishing in between: its
                # queued signal -- e.g. "art ready", whose slot starts the
                # neighbour-art loaders -- ran in the final processEvents
                # below, after the wait had already returned.)
                started = len(self._threads)
                if app is not None:
                    app.processEvents()
                if len(self._threads) == started:
                    # Done. Return without another processEvents: a timer
                    # firing in it (e.g. the backdrop's 140 ms rebuild
                    # debounce after set_library) would start a thread and
                    # report it as "still running" although nobody waited.
                    return []
                continue
            if app is not None:
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            for t in running:
                t.wait(10)
        if app is not None:
            app.processEvents()
        return [type(t).__name__ for t in self.alive()]


# ══════════════════════════════════════════════════════════════════════════════
# Motion recorder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnimationRecord:
    kind: str
    target: str
    prop: str
    duration: int
    easing: str
    parented: bool
    ref: weakref.ref = field(repr=False)

    @property
    def alive(self) -> QtCore.QAbstractAnimation | None:
        anim = self.ref()
        if anim is None:
            return None
        try:
            anim.state()
        except RuntimeError:
            return None
        return anim


class MotionRecorder:
    """Record (and optionally compress) every animation the app starts.

    ``time_scale`` multiplies every ``QVariantAnimation`` duration at start;
    0 means "finish on the next tick" (durations become 1 ms), which lets
    animated flows run to their end state quickly in tests without changing
    the code under test.
    """

    def __init__(self, time_scale: float = 1.0) -> None:
        self.time_scale = time_scale
        self.records: list[AnimationRecord] = []

    def install(self, monkeypatch) -> "MotionRecorder":
        original = QtCore.QAbstractAnimation.start
        recorder = self

        def start(anim, *args, **kwargs):
            recorder._record(anim)
            return original(anim, *args, **kwargs)

        monkeypatch.setattr(QtCore.QAbstractAnimation, "start", start)
        return self

    def _scale(self, anim: QtCore.QAbstractAnimation) -> None:
        if self.time_scale == 1.0:
            return
        if isinstance(anim, QtCore.QVariantAnimation):
            anim.setDuration(max(1, int(anim.duration() * self.time_scale)))
        elif isinstance(anim, QtCore.QPauseAnimation):
            anim.setDuration(max(1, int(anim.duration() * self.time_scale)))
        elif isinstance(anim, QtCore.QAnimationGroup):
            for i in range(anim.animationCount()):
                self._scale(anim.animationAt(i))

    def _record(self, anim: QtCore.QAbstractAnimation) -> None:
        self._scale(anim)
        target, prop, easing = "", "", ""
        if isinstance(anim, QtCore.QPropertyAnimation):
            obj = anim.targetObject()
            target = type(obj).__name__ if obj is not None else "<none>"
            name = obj.objectName() if obj is not None else ""
            if name:
                target += f"#{name}"
            prop = bytes(anim.propertyName()).decode(errors="replace")
        if isinstance(anim, QtCore.QVariantAnimation):
            easing = anim.easingCurve().type().name
        self.records.append(AnimationRecord(
            kind=type(anim).__name__, target=target, prop=prop,
            duration=anim.totalDuration(), easing=easing,
            parented=anim.parent() is not None, ref=weakref.ref(anim),
        ))

    # ── queries ────────────────────────────────────────────────────────────
    def running(self) -> list[QtCore.QAbstractAnimation]:
        out = []
        for rec in self.records:
            anim = rec.alive
            if anim is not None and anim.state() == QtCore.QAbstractAnimation.State.Running:
                out.append(anim)
        return out

    def settle(self, timeout_ms: int = 5000) -> list[QtCore.QAbstractAnimation]:
        """Process events until no recorded animation is running (or timeout).
        Infinite-loop animations (loopCount == -1) are ignored. Returns any
        finite animations still running at the deadline."""
        deadline = time.monotonic() + timeout_ms / 1000
        app = QtWidgets.QApplication.instance()
        while True:
            if app is not None:
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            finite = [a for a in self.running() if a.loopCount() != -1]
            if not finite or time.monotonic() >= deadline:
                return finite
            time.sleep(0.005)

    def orphans(self) -> list[AnimationRecord]:
        """Animations started without a QObject parent.

        In PySide an unparented animation is kept alive only by its Python
        reference; if that reference is dropped (a local variable), the
        animation is garbage-collected mid-flight and the widget is left
        half-way (e.g. stuck at 40 % opacity)."""
        return [r for r in self.records if not r.parented]


def stuck_opacity(root: QtWidgets.QWidget, tolerance: float = 0.01) -> list[str]:
    """Visible widgets under ``root`` whose opacity effect is left below 1.0."""
    stuck = []
    widgets = [root, *root.findChildren(QtWidgets.QWidget)]
    for w in widgets:
        if not w.isVisible():
            continue
        eff = w.graphicsEffect()
        if isinstance(eff, QtWidgets.QGraphicsOpacityEffect) and eff.opacity() < 1 - tolerance:
            stuck.append(f"{type(w).__name__}#{w.objectName()} opacity={eff.opacity():.2f}")
    if root.isWindow() and root.windowOpacity() < 1 - tolerance:
        stuck.append(f"{type(root).__name__} windowOpacity={root.windowOpacity():.2f}")
    return stuck


# ══════════════════════════════════════════════════════════════════════════════
# Qt warning policy
# ══════════════════════════════════════════════════════════════════════════════

class QtWarningPolicy:
    """Qt log messages that indicate genuine bugs.

    These are the messages Qt prints when code touches widgets/timers/objects
    from the wrong thread, paints illegally, or animates something that does
    not exist — every one of them is either a crash waiting to happen or a
    silent no-op the author did not intend.
    """

    FATAL_PATTERNS: tuple[str, ...] = (
        r"Cannot create children for a parent that is in a different thread",
        r"Cannot set parent, new parent is in a different thread",
        r"Timers cannot be started from another thread",
        r"Timers cannot be stopped from another thread",
        r"Timers can only be used with threads started with QThread",
        r"Cannot send events to objects owned by a different thread",
        r"It is not safe to use pixmaps outside the GUI thread",
        r"QPixmap: Must construct a QGuiApplication before a QPixmap",
        r"QWidget::repaint: Recursive repaint detected",
        r"A paint device can only be painted by one painter at a time",
        r"QPainter::begin: Paint device returned engine == 0",
        r"you're trying to animate a non-existing property",
        r"you're trying to animate the property .* of an object that",
        r"QPropertyAnimation::updateState .* starting an animation without",
        r"QObject::connect: No such (signal|slot)",
        r"QThread: Destroyed while thread is still running",
        r"QObject::killTimer: Timers cannot be stopped from another thread",
    )

    def __init__(self, extra: Iterable[str] = ()) -> None:
        self._regex = re.compile("|".join([*self.FATAL_PATTERNS, *extra]))

    def violations(self, messages: Iterable[str]) -> list[str]:
        return [m for m in messages if self._regex.search(m)]


# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════

def process_events(ms: int = 0) -> None:
    """Spin the event loop for ``ms`` milliseconds (0 = one pass)."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    deadline = time.monotonic() + ms / 1000
    while True:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if time.monotonic() >= deadline:
            break
        time.sleep(0.002)


def visible_texts(root: QtWidgets.QWidget) -> list[str]:
    """Text of every visible label/button under ``root`` (for assertions)."""
    out = []
    for w in root.findChildren(QtWidgets.QWidget):
        if not w.isVisibleTo(root):
            continue
        text = getattr(w, "text", None)
        if callable(text) and isinstance(w, (QtWidgets.QLabel, QtWidgets.QAbstractButton)):
            value = text()
            if value:
                out.append(value)
    return out


def find_button(root: QtWidgets.QWidget, text: str) -> QtWidgets.QAbstractButton:
    """The unique button under ``root`` whose text (sans '&', case-insensitive) contains ``text``."""
    wanted = text.replace("&", "").lower()
    matches = [
        b for b in root.findChildren(QtWidgets.QAbstractButton)
        if wanted in b.text().replace("&", "").lower()
    ]
    if len(matches) != 1:
        labels = sorted({b.text() for b in root.findChildren(QtWidgets.QAbstractButton) if b.text()})
        raise LookupError(f"{len(matches)} buttons match {text!r}; buttons: {labels}")
    return matches[0]


# ══════════════════════════════════════════════════════════════════════════════
# Responsiveness (jank) meter
# ══════════════════════════════════════════════════════════════════════════════

class FrameGapMeter:
    """Measure how long the GUI thread stays blocked.

    A fast repeating timer records when it actually gets to run; the gap
    between consecutive ticks is how long the event loop could not paint or
    react. At 60 fps a frame is 16.7 ms, so any gap above ~17 ms during an
    animation is a dropped frame, and gaps of hundreds of ms are visible
    freezes. Use around an action::

        with FrameGapMeter() as meter:
            get_manager().apply("pearl")
            process_events(50)
        assert meter.max_gap_ms < 250

    Synchronous work inside the ``with`` block counts as one long gap — which
    is exactly the freeze a user would see.
    """

    FRAME_MS = 1000 / 60

    def __init__(self, interval_ms: int = 4) -> None:
        self._timer = QtCore.QTimer()
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._ticks: list[float] = []

    def _tick(self) -> None:
        self._ticks.append(time.perf_counter())

    def __enter__(self) -> "FrameGapMeter":
        self._ticks = [time.perf_counter()]
        self._timer.start()
        return self

    def __exit__(self, *exc) -> None:
        process_events(0)
        self._ticks.append(time.perf_counter())
        self._timer.stop()

    @property
    def gaps_ms(self) -> list[float]:
        return [(b - a) * 1000 for a, b in zip(self._ticks, self._ticks[1:])]

    @property
    def max_gap_ms(self) -> float:
        return max(self.gaps_ms, default=0.0)

    @property
    def dropped_frames(self) -> int:
        """Frames that would have been missed at 60 fps."""
        return sum(int(g // self.FRAME_MS) for g in self.gaps_ms if g > self.FRAME_MS)

    def summary(self) -> str:
        gaps = sorted(self.gaps_ms)
        p95 = gaps[int(len(gaps) * 0.95)] if gaps else 0.0
        return f"max {self.max_gap_ms:.0f} ms, p95 {p95:.0f} ms, ~{self.dropped_frames} dropped frames"
