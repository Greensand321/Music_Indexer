import threading
import time
from tkinter import messagebox

from pydub import AudioSegment
import simpleaudio as sa


class LoopingPreviewPlayer:
    """Plays a looping preview clip from an audio file."""

    def __init__(self, start_ms: int = 30000, duration_ms: int = 30000) -> None:
        self.start_ms = start_ms
        self.duration_ms = duration_ms
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._play_obj: sa.PlayObject | None = None

    def play(self, path: str) -> None:
        """Start looping preview for ``path``."""
        with self._lock:
            self.stop()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, args=(path,), daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop any current playback."""
        self._stop_event.set()
        if self._play_obj and self._play_obj.is_playing():
            self._play_obj.stop()
        if self._thread:
            self._thread.join(timeout=0.1)
            self._thread = None

    def _loop(self, path: str) -> None:
        try:
            audio = AudioSegment.from_file(path)
            clip = audio[self.start_ms : self.start_ms + self.duration_ms]
        except Exception as exc:
            messagebox.showerror("Playback Error", str(exc))
            return
        while not self._stop_event.is_set():
            self._play_obj = sa.play_buffer(
                clip.raw_data,
                num_channels=clip.channels,
                bytes_per_sample=clip.sample_width,
                sample_rate=clip.frame_rate,
            )
            while self._play_obj.is_playing() and not self._stop_event.is_set():
                time.sleep(0.1)
            if self._stop_event.is_set():
                self._play_obj.stop()
                break
            time.sleep(0.5)


_preview_player = LoopingPreviewPlayer()


def play_preview(path: str) -> None:
    """Play or switch to a looping preview for ``path``."""
    _preview_player.play(path)


def stop_preview() -> None:
    """Stop any currently playing preview."""
    _preview_player.stop()
