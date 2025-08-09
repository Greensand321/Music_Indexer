from pydub import AudioSegment
import simpleaudio as sa
import threading
import subprocess
import shutil


class PlaybackError(Exception):
    """Raised when preview playback fails."""


class PreviewPlayer:
    """Play short audio previews with improved thread safety."""

    def __init__(self) -> None:
        self._play_obj = None
        self._ffplay_proc = None
        # Re-entrant lock so stop_preview can be called while already held
        self._play_lock = threading.RLock()
        self._is_playing = False

    def _monitor_playback(self, play_obj=None, ffplay_proc=None) -> None:
        """Wait for playback to finish then clean up.

        Parameters
        ----------
        play_obj:
            The ``simpleaudio`` play object to monitor.
        ffplay_proc:
            The ``ffplay`` subprocess to monitor.
        """

        if play_obj is not None:
            play_obj.wait_done()
        elif ffplay_proc is not None:
            ffplay_proc.wait()

        with self._play_lock:
            # Ensure this monitor corresponds to the current playback
            if self._play_obj is play_obj and self._ffplay_proc is ffplay_proc:
                self.stop_preview()

    def play_preview(
        self, path: str, start_ms: int = 30000, duration_ms: int = 15000
    ) -> None:
        """Play a short preview of the audio file at ``path``."""

        with self._play_lock:
            if self._is_playing:
                self.stop_preview()
            self._is_playing = True

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
                play_obj = self._play_obj
            threading.Thread(
                target=self._monitor_playback, args=(play_obj, None), daemon=True
            ).start()
            return
        except Exception as sa_error:
            # Fallback to ffplay if simpleaudio fails
            ffplay = shutil.which("ffplay")
            if not ffplay:
                with self._play_lock:
                    self._is_playing = False
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
                with self._play_lock:
                    self._ffplay_proc = subprocess.Popen(cmd, start_new_session=True)
                    proc = self._ffplay_proc
                threading.Thread(
                    target=self._monitor_playback, args=(None, proc), daemon=True
                ).start()
            except Exception as exc:
                with self._play_lock:
                    self._is_playing = False
                raise PlaybackError(str(exc)) from exc

    def stop_preview(self) -> None:
        """Stop any currently playing preview."""
        with self._play_lock:
            if self._play_obj and self._play_obj.is_playing():
                self._play_obj.stop()
            self._play_obj = None

            if self._ffplay_proc and self._ffplay_proc.poll() is None:
                try:
                    self._ffplay_proc.terminate()
                    self._ffplay_proc.wait(timeout=1)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        self._ffplay_proc.kill()
                        self._ffplay_proc.wait(timeout=1)
                    except Exception:
                        pass
                self._ffplay_proc = None

            self._is_playing = False
