from pydub import AudioSegment
import simpleaudio as sa
from tkinter import messagebox
import threading
import time

_play_obj = None
_play_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None


def stop_preview() -> None:
    """Stop any currently playing preview."""
    global _play_obj, _play_thread, _stop_event
    if _stop_event is not None:
        _stop_event.set()
    if _play_obj and _play_obj.is_playing():
        _play_obj.stop()
    _play_thread = None
    _stop_event = None
    _play_obj = None


def _loop_play(clip: AudioSegment, stop_evt: threading.Event) -> None:
    global _play_obj
    while not stop_evt.is_set():
        _play_obj = sa.play_buffer(
            clip.raw_data,
            num_channels=clip.channels,
            bytes_per_sample=clip.sample_width,
            sample_rate=clip.frame_rate,
        )
        while _play_obj.is_playing() and not stop_evt.is_set():
            time.sleep(0.1)
        if stop_evt.is_set():
            _play_obj.stop()
            break
        time.sleep(0.25)


def play_preview(path: str, start_ms: int = 30000, duration_ms: int = 30000) -> None:
    """Play a looping preview of ``path``.

    The snippet plays for ``duration_ms`` starting at ``start_ms`` then pauses
    briefly before looping until :func:`stop_preview` is called.
    """
    global _play_thread, _stop_event

    stop_preview()
    try:
        audio = AudioSegment.from_file(path)
        clip = audio[start_ms : start_ms + duration_ms]
    except Exception as exc:
        messagebox.showerror("Playback Error", str(exc))
        return

    _stop_event = threading.Event()
    _play_thread = threading.Thread(
        target=_loop_play, args=(clip, _stop_event), daemon=True
    )
    _play_thread.start()
