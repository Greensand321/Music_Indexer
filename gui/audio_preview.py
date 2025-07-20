from pydub import AudioSegment
import simpleaudio as sa
from tkinter import messagebox

_play_obj = None

def play_preview(path: str, start_ms: int = 30000, duration_ms: int = 15000) -> None:
    """Play a short preview of the audio file at ``path``.

    Parameters
    ----------
    path : str
        File path of the audio clip.
    start_ms : int, optional
        Starting point in milliseconds, by default 30000.
    duration_ms : int, optional
        Duration of the preview in milliseconds, by default 15000.
    """
    global _play_obj
    try:
        audio = AudioSegment.from_file(path)
        clip = audio[start_ms : start_ms + duration_ms]
        if _play_obj and _play_obj.is_playing():
            _play_obj.stop()
        _play_obj = sa.play_buffer(
            clip.raw_data,
            num_channels=clip.channels,
            bytes_per_sample=clip.sample_width,
            sample_rate=clip.frame_rate,
        )
    except Exception as exc:
        messagebox.showerror("Playback Error", str(exc))
