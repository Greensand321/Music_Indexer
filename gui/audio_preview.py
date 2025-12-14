import logging
import threading
from typing import Optional


try:
    import vlc

    VLC_AVAILABLE = True
    VLC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - import guard
    VLC_AVAILABLE = False
    VLC_IMPORT_ERROR = exc


class PlaybackError(Exception):
    pass


class VlcPreviewPlayer:
    def __init__(self, on_done=None) -> None:
        self._logger = logging.getLogger(__name__)
        self._on_done = on_done
        self._play_lock = threading.RLock()
        self._instance: Optional["vlc.Instance"] = None
        self._player: Optional["vlc.MediaPlayer"] = None
        self._event_mgr = None
        self._is_playing = False
        self._init_error: Exception | None = VLC_IMPORT_ERROR
        self._setup_player()

    def _setup_player(self) -> None:
        if self._init_error:
            self._logger.error("python-vlc not available: %s", self._init_error)
            return

        try:
            self._instance = vlc.Instance()
            self._player = self._instance.media_player_new()
            self._event_mgr = self._player.event_manager()
            self._event_mgr.event_attach(
                vlc.EventType.MediaPlayerEndReached, self._handle_end
            )
            self._event_mgr.event_attach(
                vlc.EventType.MediaPlayerStopped, self._handle_end
            )
            self._event_mgr.event_attach(
                vlc.EventType.MediaPlayerEncounteredError, self._handle_error
            )
            self._logger.debug("VLC preview backend initialized")
        except Exception as exc:
            self._init_error = exc
            self._logger.exception("Failed to initialize VLC preview backend")

    @property
    def available(self) -> bool:
        return self._init_error is None

    @property
    def availability_error(self) -> str | None:
        if self._init_error:
            return str(self._init_error)
        return None

    def stop_preview(self) -> None:
        with self._play_lock:
            if not self._player:
                return
            try:
                self._logger.debug("Stopping VLC preview playback")
                self._player.stop()
            except Exception:
                self._logger.exception("Error while stopping VLC preview")
            self._is_playing = False

    def play_clip(self, path: str, start_ms: int = 30000, duration_ms: int = 15000) -> None:
        with self._play_lock:
            if not self.available or not self._player or not self._instance:
                raise PlaybackError(self.availability_error or "VLC backend unavailable")
            if self._is_playing:
                self._logger.debug("Stopping existing preview before starting new one")
                self.stop_preview()
            self._is_playing = True

            media = self._instance.media_new(path)
            media.add_option(":no-video")
            media.add_option(f":start-time={start_ms / 1000}")
            media.add_option(f":run-time={duration_ms / 1000}")

            self._logger.info(
                "Starting VLC preview: path=%s start_ms=%s duration_ms=%s",
                path,
                start_ms,
                duration_ms,
            )
            self._player.set_media(media)
            try:
                self._player.play()
            except Exception as exc:
                self._is_playing = False
                self._logger.exception("Failed to start VLC playback")
                raise PlaybackError(str(exc)) from exc

    def _handle_end(self, event=None) -> None:  # pragma: no cover - event driven
        with self._play_lock:
            self._is_playing = False
        self._logger.debug("VLC preview finished")
        if self._on_done:
            try:
                self._on_done()
            except Exception:
                self._logger.exception("Error in preview completion callback")

    def _handle_error(self, event=None) -> None:  # pragma: no cover - event driven
        with self._play_lock:
            self._is_playing = False
        self._logger.error("VLC reported an error during preview playback")
        if self._on_done:
            try:
                self._on_done()
            except Exception:
                self._logger.exception("Error in preview error callback")
