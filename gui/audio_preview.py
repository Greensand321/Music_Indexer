from pydub import AudioSegment
import simpleaudio as sa
import threading
import subprocess
import shutil


class PlaybackError(Exception):
    """Raised when preview playback fails."""


class PreviewPlayer:
    """Play short audio previews with thread-safe state."""

    def __init__(self) -> None:
        self._play_obj = None
        self._ffplay_proc = None
        self._play_lock = threading.Lock()

    def play_preview(
        self, path: str, start_ms: int = 30000, duration_ms: int = 15000
    ) -> None:
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
        self.stop_preview()
        try:
            audio = AudioSegment.from_file(path)
            clip = audio[start_ms : start_ms + duration_ms]
            with self._play_lock:
                self._play_obj = sa.play_buffer(
                    clip.raw_data,
                    num_channels=clip.channels,
                    bytes_per_sample=clip.sample_width,
                    sample_rate=clip.frame_rate,
                )
            return
        except Exception as exc:
            sa_error = exc

        ffplay = shutil.which("ffplay")
        if not ffplay:
            raise PlaybackError(str(sa_error)) from sa_error
        try:
            cmd = [
                ffplay,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-ss",
                str(start_ms / 1000),
                "-t",
                str(duration_ms / 1000),
                path,
            ]
            self._ffplay_proc = subprocess.Popen(cmd)
        except Exception as exc:
            raise PlaybackError(str(exc)) from exc

    def stop_preview(self) -> None:
        """Stop any currently playing preview."""
        with self._play_lock:
            if self._play_obj and self._play_obj.is_playing():
                self._play_obj.stop()
            self._play_obj = None
        if self._ffplay_proc and self._ffplay_proc.poll() is None:
            self._ffplay_proc.terminate()
            try:
                self._ffplay_proc.wait(timeout=1)
            except Exception:
                self._ffplay_proc.kill()
        self._ffplay_proc = None
