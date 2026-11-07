"""Motion tokens and safe animation helpers for the AlphaDEX Qt app.

Every animation in the app should get its duration and easing from here, and
should be built with one of the helpers below rather than a hand-rolled
``QPropertyAnimation``. The helpers exist because the hand-rolled versions
kept producing the same four bugs: animations garbage-collected mid-flight
(no parent), two animations fighting over one property (no re-entrancy
guard), widgets left half-transparent (fade never reaching its end), and
whole-window relayouts every frame (animating geometry a layout owns).

Tokens
------
Durations, in ms (derived from what the app already used; see
``docs/redesign/pages/02_design_system_motion.md`` §3 and §5):

==========  =====  ==========================================================
token        ms    used for
==========  =====  ==========================================================
instant        0   reduced motion; "jump to the end"
micro        120   press / pulse feedback
fast         150   hover colour eases, toggles
base         220   drawers, reveals, chip swaps
slow         320   nav pill slide, window fade-in
page         420   landing <-> main window cross-fade
stagger       38   per-item delay in staggered entrances (``STAGGER_MAX`` caps
                   the total at 600 ms)
ambient     3500   album-art cross-fade and other background drift
flash       1600   colour flash on a status label
==========  =====  ==========================================================

Easings (four, by role): ``standard`` (OutCubic, things arriving or
settling), ``exit`` (InCubic, things leaving), ``emphasis`` (OutBack with a
small 1.2 overshoot, for the nav pill), ``ambient`` (InOutSine).

Reduced motion
--------------
``reduced_motion()`` is true when the environment variable
``ALPHADEX_REDUCED_MOTION`` is ``1``/``true``/``yes``/``on``, or when the
config key ``reduced_motion`` is true (read once through
``config.load_config()`` and cached; call ``refresh_reduced_motion()`` after
the settings change). ``set_reduced_motion(True/False/None)`` overrides both
(``None`` clears the override). With reduced motion every token resolves to
0 ms and the helpers jump straight to the end state, running the same
completion callbacks, so code paths are identical either way.

Helpers
-------
``duration(token)`` -> int ms (0 under reduced motion).
``easing(name)`` -> ``QEasingCurve``.

``animate_value(owner, key, start, end, on_value, *, duration="base",
easing="standard", on_finished=None)``
    Drive a plain Python value (a paint offset, a hover progress) with a
    ``QVariantAnimation`` parented to ``owner``. ``on_value(v)`` is called
    for every frame and always with ``end`` last. One animation per
    ``(owner, key)``: calling again stops the running one (its
    ``on_finished`` is dropped) and reuses the object, so repeated triggers
    never grow the animation count.

``animate_property(target, prop, end, *, start=None, duration="base",
easing="standard", on_finished=None)``
    Same contract for a Qt property of ``target`` (a ``QPropertyAnimation``
    parented to ``target``). ``start`` defaults to the current value. Refuses
    (``ValueError``) ``geometry``/``pos``/``size``/``minimumHeight``/
    ``maximumHeight``/... of a widget that sits in a layout: animate a paint
    offset with ``animate_value`` instead, or move a child of a widget that
    has no layout.

``fade(widget, visible, *, duration="base", on_finished=None)``
    Opacity fade through a ``QGraphicsOpacityEffect`` parented to the
    widget. The end state is always clean: fading in ends at opacity 1.0
    with the effect disabled (no off-screen rendering at rest); fading out
    ends with the widget hidden and the opacity reset to 1.0. A widget
    already carrying a different graphics effect is never clobbered: it is
    just shown or hidden.

``stop(owner, key)`` stops a helper animation without running callbacks;
``animation_for(owner, key)`` returns it (or ``None``) for tests.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from gui.compat import QtCore, QtWidgets

# ── Tokens ────────────────────────────────────────────────────────────────────

INSTANT = 0
MICRO = 120
FAST = 150
BASE = 220
SLOW = 320
PAGE = 420
STAGGER = 38
STAGGER_MAX = 600
AMBIENT = 3500
FLASH = 1600

DURATIONS: dict[str, int] = {
    "instant": INSTANT,
    "micro": MICRO,
    "fast": FAST,
    "base": BASE,
    "slow": SLOW,
    "page": PAGE,
    "stagger": STAGGER,
    "ambient": AMBIENT,
    "flash": FLASH,
}

_EasingType = QtCore.QEasingCurve.Type
EASINGS: dict[str, Any] = {
    "standard": _EasingType.OutCubic,
    "exit": _EasingType.InCubic,
    "emphasis": _EasingType.OutBack,
    "ambient": _EasingType.InOutSine,
}
EMPHASIS_OVERSHOOT = 1.2

ENV_VAR = "ALPHADEX_REDUCED_MOTION"
CONFIG_KEY = "reduced_motion"

_TRUTHY = {"1", "true", "yes", "on"}

# ── Reduced motion ────────────────────────────────────────────────────────────

_override: bool | None = None
_config_cache: bool | None = None


def _config_flag() -> bool:
    global _config_cache
    if _config_cache is None:
        try:
            from config import load_config
            _config_cache = bool(load_config().get(CONFIG_KEY, False))
        except Exception:
            _config_cache = False
    return _config_cache


def reduced_motion() -> bool:
    """True when animations should jump straight to their end state."""
    if _override is not None:
        return _override
    if os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY:
        return True
    return _config_flag()


def set_reduced_motion(value: bool | None) -> None:
    """Force reduced motion on/off for this process (``None`` clears)."""
    global _override
    _override = value


def refresh_reduced_motion() -> None:
    """Forget the cached config flag (call after settings are saved)."""
    global _config_cache
    _config_cache = None


def duration(token: str | int) -> int:
    """Resolve a duration token (or a raw ms value) honouring reduced motion."""
    if reduced_motion():
        return 0
    if isinstance(token, str):
        return DURATIONS[token]
    return max(0, int(token))


def easing(name: str = "standard") -> QtCore.QEasingCurve:
    curve = QtCore.QEasingCurve(EASINGS[name])
    if name == "emphasis":
        curve.setOvershoot(EMPHASIS_OVERSHOOT)
    return curve


def stagger_delay(index: int, count: int) -> int:
    """Delay for item ``index`` of ``count`` so the total stays <= STAGGER_MAX."""
    if reduced_motion() or count <= 1:
        return 0
    step = min(STAGGER, STAGGER_MAX // max(1, count - 1))
    return index * step


# ── Helper animation objects ─────────────────────────────────────────────────

_PREFIX = "adx-motion:"

# Properties a layout recomputes: animating them on a laid-out widget fights
# the layout (the widget ends up offset) or relayouts the window every frame.
_LAYOUT_OWNED = {
    "geometry", "pos", "size", "x", "y", "width", "height",
    "minimumHeight", "maximumHeight", "minimumWidth", "maximumWidth",
    "minimumSize", "maximumSize",
}


class _Run:
    """Per-run completion bookkeeping shared by the helper animations."""

    def _init_run(self) -> None:
        self._run_id = 0
        self._callbacks: list[Callable[[], None]] = []
        self.finished.connect(self._on_done)  # type: ignore[attr-defined]

    def _begin(self, on_finished: Callable[[], None] | None) -> None:
        self._run_id += 1
        self._callbacks = [on_finished] if on_finished else []

    def _on_done(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            cb()


class _ValueAnimation(_Run, QtCore.QVariantAnimation):
    def __init__(self, owner: QtCore.QObject, key: str) -> None:
        super().__init__(owner)
        self.setObjectName(_PREFIX + key)
        self._on_value: Callable[[Any], None] | None = None
        self.valueChanged.connect(self._emit_value)
        self._init_run()

    def _emit_value(self, value: Any) -> None:
        if self._on_value is not None:
            self._on_value(value)


class _PropertyAnimation(_Run, QtCore.QPropertyAnimation):
    def __init__(self, target: QtCore.QObject, prop: bytes) -> None:
        super().__init__(target, prop, target)
        self.setObjectName(_PREFIX + "prop:" + prop.decode())
        self._init_run()


def _find(owner: QtCore.QObject, name: str):
    for child in owner.children():
        if isinstance(child, (_ValueAnimation, _PropertyAnimation)) and child.objectName() == name:
            return child
    return None


def animation_for(owner: QtCore.QObject, key: str):
    """The helper animation for ``(owner, key)``; ``key`` is the
    ``animate_value`` key, or ``"prop:<name>"`` for ``animate_property``."""
    return _find(owner, _PREFIX + key)


def stop(owner: QtCore.QObject, key: str) -> None:
    """Stop a helper animation without running its completion callback."""
    anim = animation_for(owner, key)
    if anim is not None:
        anim.stop()
        anim._callbacks = []


def _configure(anim, start: Any, end: Any, dur: int, ease: str) -> None:
    anim.setDuration(dur)
    anim.setEasingCurve(easing(ease))
    anim.setStartValue(start)
    anim.setEndValue(end)


def animate_value(
    owner: QtCore.QObject,
    key: str,
    start: Any,
    end: Any,
    on_value: Callable[[Any], None],
    *,
    duration: str | int = "base",  # noqa: A002 - token name mirrors the API
    easing: str = "standard",  # noqa: A002
    on_finished: Callable[[], None] | None = None,
) -> QtCore.QVariantAnimation:
    """Animate a Python-side value; see the module docstring."""
    anim = _find(owner, _PREFIX + key)
    if anim is None:
        anim = _ValueAnimation(owner, key)
    anim.stop()
    # Detach while configuring: setting start/end on a stopped animation
    # emits valueChanged, which would deliver a stray value.
    anim._on_value = None
    anim._begin(on_finished)
    dur = _resolve(duration)
    _configure(anim, start, end, dur, easing)
    anim._on_value = on_value
    if dur == 0:
        on_value(end)
        anim._on_done()
    else:
        anim.start()
    return anim


def animate_property(
    target: QtCore.QObject,
    prop: bytes | str,
    end: Any,
    *,
    start: Any = None,
    duration: str | int = "base",  # noqa: A002
    easing: str = "standard",  # noqa: A002
    on_finished: Callable[[], None] | None = None,
) -> QtCore.QPropertyAnimation:
    """Animate a Qt property of ``target``; see the module docstring."""
    name = prop.decode() if isinstance(prop, bytes) else prop
    if (
        name in _LAYOUT_OWNED
        and isinstance(target, QtWidgets.QWidget)
        and _in_layout(target)
    ):
        raise ValueError(
            f"refusing to animate {name!r} of a {type(target).__name__} that a layout owns; "
            "animate a paint offset (animate_value) instead"
        )
    anim = _find(target, _PREFIX + "prop:" + name)
    if anim is None:
        anim = _PropertyAnimation(target, name.encode())
    anim.stop()
    anim._begin(on_finished)
    if start is None:
        start = target.property(name)
    dur = _resolve(duration)
    _configure(anim, start, end, dur, easing)
    if dur == 0:
        target.setProperty(name, end)
        anim._on_done()
    else:
        anim.start()
    return anim


def _resolve(token: str | int) -> int:
    return globals()["duration"](token)


def _in_layout(widget: QtWidgets.QWidget) -> bool:
    parent = widget.parentWidget()
    if parent is None or parent.layout() is None:
        return False
    return parent.layout().indexOf(widget) >= 0 or _layout_contains(parent.layout(), widget)


def _layout_contains(layout: QtWidgets.QLayout, widget: QtWidgets.QWidget) -> bool:
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        if item.widget() is widget:
            return True
        sub = item.layout()
        if sub is not None and _layout_contains(sub, widget):
            return True
    return False


# ── Fade ──────────────────────────────────────────────────────────────────────

def fade(
    widget: QtWidgets.QWidget,
    visible: bool,
    *,
    duration: str | int = "base",  # noqa: A002
    on_finished: Callable[[], None] | None = None,
) -> QtCore.QPropertyAnimation | None:
    """Fade ``widget`` in (``visible=True``) or out-and-hide; see module doc."""
    effect = widget.graphicsEffect()
    if effect is not None and not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
        # One effect slot per widget: never clobber someone else's effect.
        widget.setVisible(visible)
        if on_finished:
            on_finished()
        return None
    if effect is None:
        effect = QtWidgets.QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0 if visible and not widget.isVisible() else 1.0)
        widget.setGraphicsEffect(effect)

    if visible:
        if not widget.isVisible():
            effect.setOpacity(0.0)
        effect.setEnabled(True)
        widget.show()

        def _done() -> None:
            effect.setOpacity(1.0)
            effect.setEnabled(False)   # nothing renders off-screen at rest
            if on_finished:
                on_finished()

        return animate_property(effect, b"opacity", 1.0, duration=duration,
                                easing="standard", on_finished=_done)

    if not widget.isVisible():
        effect.setOpacity(1.0)
        effect.setEnabled(False)
        if on_finished:
            on_finished()
        return None
    effect.setEnabled(True)

    def _hidden() -> None:
        widget.hide()
        effect.setOpacity(1.0)
        effect.setEnabled(False)
        if on_finished:
            on_finished()

    return animate_property(effect, b"opacity", 0.0, duration=duration,
                            easing="exit", on_finished=_hidden)
