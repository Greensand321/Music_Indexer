"""Playback backends for the Qt Player — the seam between the UI and libVLC.

Why this exists
---------------
The Player used to import ``vlc`` inside ``PlayerWorkspace.__init__`` and
catch only ``ImportError``. python-vlc is in ``requirements.txt``, so on a
machine without the libVLC *library* ``import vlc`` raises
``NotImplementedError`` (or ``OSError``) instead, and the whole main window
failed to build. With no VLC at all the page still pretended to play.

Now ``PlayerWorkspace`` talks only to a :class:`PlaybackBackend`:

* :class:`VlcBackend` wraps python-vlc. Any failure while importing or
  initialising libVLC raises :class:`BackendUnavailable`.
* :class:`NullBackend` is what you get when no backend can be loaded. It is
  ``available == False`` and carries a human-readable ``unavailable_reason``
  the UI shows instead of a fake "playing" state.
* :func:`create_backend` picks the first backend that works. Tests inject a
  fake either by passing ``backend=`` to ``PlayerWorkspace`` or by
  monkeypatching ``gui.playback_backend.create_backend``.

This module is Qt-free and has no side effects at import time.
"""
from __future__ import annotations

import enum
from typing import Any, Callable


class PlaybackState(enum.Enum):
    IDLE = "idle"          # nothing loaded
    OPENING = "opening"    # loading / buffering
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ENDED = "ended"        # reached the end of the track
    ERROR = "error"        # the track could not be opened or decoded


ACTIVE_STATES = frozenset({PlaybackState.OPENING, PlaybackState.PLAYING, PlaybackState.PAUSED})


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot be created or is asked to play while unavailable."""


class PlaybackBackend:
    """The operations the Player needs. Times are milliseconds; volume 0-100.

    Implementations must never raise from the query methods (``state``,
    ``length_ms``, ``time_ms``); return the "unknown" value instead.
    """

    name: str = "none"
    available: bool = False
    unavailable_reason: str = ""

    def load(self, path: str) -> None:
        raise BackendUnavailable(self.unavailable_reason or "No playback backend")

    def play(self) -> bool:
        """Start (or resume) playback of the loaded track. False on failure."""
        return False

    def pause(self) -> None:
        """Pause (not toggle)."""

    def stop(self) -> None:
        """Stop playback; the track stays loaded."""

    def state(self) -> PlaybackState:
        return PlaybackState.IDLE

    def is_playing(self) -> bool:
        return self.state() is PlaybackState.PLAYING

    def length_ms(self) -> int:
        return -1

    def time_ms(self) -> int:
        return -1

    def set_time_ms(self, ms: int) -> None:
        """Seek to ``ms``."""

    def set_volume(self, value: int) -> None:
        """Set the volume, 0-100."""

    def release(self) -> None:
        """Free native resources. The backend is unusable afterwards."""


class NullBackend(PlaybackBackend):
    """No playback available. Every command is a no-op; ``load`` raises."""

    name = "none"
    available = False

    def __init__(self, reason: str = "No playback backend is available.") -> None:
        self.unavailable_reason = reason


class VlcBackend(PlaybackBackend):
    """libVLC through python-vlc.

    ``vlc_module`` lets tests pass a fake module; by default ``vlc`` is
    imported here, and *any* exception (``ImportError``, the
    ``NotImplementedError``/``OSError`` python-vlc raises when the libVLC
    library is missing, ...) becomes :class:`BackendUnavailable`.
    """

    name = "VLC"
    available = True

    def __init__(self, vlc_module: Any = None) -> None:
        if vlc_module is None:
            try:
                import vlc as vlc_module  # type: ignore
            except ImportError as exc:
                raise BackendUnavailable(
                    "python-vlc is not installed. Install VLC and python-vlc for in-app playback."
                ) from exc
            except Exception as exc:  # noqa: BLE001 - libVLC missing raises odd types
                raise BackendUnavailable(
                    f"VLC could not be loaded ({type(exc).__name__}: {exc}). "
                    "Install the VLC media player for in-app playback."
                ) from exc
        self._vlc = vlc_module
        try:
            instance = vlc_module.Instance("--quiet")
            player = instance.media_player_new() if instance is not None else None
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"libVLC failed to start ({type(exc).__name__}: {exc}).") from exc
        if instance is None or player is None:
            raise BackendUnavailable("libVLC failed to start (no VLC instance).")
        self._instance = instance
        self._player = player
        self._volume: int | None = None
        states = getattr(vlc_module, "State", None)
        self._state_map: dict[Any, PlaybackState] = {}
        for vlc_name, ours in (
            ("NothingSpecial", PlaybackState.IDLE),
            ("Opening", PlaybackState.OPENING),
            ("Buffering", PlaybackState.OPENING),
            ("Playing", PlaybackState.PLAYING),
            ("Paused", PlaybackState.PAUSED),
            ("Stopped", PlaybackState.STOPPED),
            ("Ended", PlaybackState.ENDED),
            ("Error", PlaybackState.ERROR),
        ):
            member = getattr(states, vlc_name, None) if states is not None else None
            if member is not None:
                self._state_map[member] = ours

    def load(self, path: str) -> None:
        media = self._instance.media_new(path)
        if media is None:
            raise BackendUnavailable(f"VLC could not open {path}")
        self._player.set_media(media)

    def play(self) -> bool:
        result = self._player.play()
        ok = result is None or result == 0
        if ok and self._volume is not None:
            self._safe(self._player.audio_set_volume, self._volume)
        return ok

    def pause(self) -> None:
        set_pause = getattr(self._player, "set_pause", None)
        if callable(set_pause):
            set_pause(1)
        elif self.state() is PlaybackState.PLAYING:
            self._player.pause()  # toggles; only call it while playing

    def stop(self) -> None:
        self._player.stop()

    def state(self) -> PlaybackState:
        try:
            raw = self._player.get_state()
        except Exception:  # noqa: BLE001
            return PlaybackState.ERROR
        return self._state_map.get(raw, PlaybackState.IDLE)

    def is_playing(self) -> bool:
        try:
            return bool(self._player.is_playing())
        except Exception:  # noqa: BLE001
            return False

    def length_ms(self) -> int:
        return self._safe(self._player.get_length, default=-1)

    def time_ms(self) -> int:
        return self._safe(self._player.get_time, default=-1)

    def set_time_ms(self, ms: int) -> None:
        self._player.set_time(max(0, int(ms)))

    def set_volume(self, value: int) -> None:
        self._volume = max(0, min(100, int(value)))
        self._safe(self._player.audio_set_volume, self._volume)

    def release(self) -> None:
        for obj in (self._player, self._instance):
            rel = getattr(obj, "release", None)
            if callable(rel):
                self._safe(rel)

    @staticmethod
    def _safe(fn: Callable, *args: Any, default: Any = None) -> Any:
        try:
            out = fn(*args)
        except Exception:  # noqa: BLE001
            return default
        return default if out is None and default is not None else out


def create_backend() -> PlaybackBackend:
    """Return the first playback backend that can be created.

    Never raises: when nothing works the result is a :class:`NullBackend`
    whose ``unavailable_reason`` says why.
    """
    try:
        return VlcBackend()
    except BackendUnavailable as exc:
        return NullBackend(str(exc))
    except Exception as exc:  # noqa: BLE001 - never break window construction
        return NullBackend(f"Playback unavailable ({type(exc).__name__}: {exc}).")
