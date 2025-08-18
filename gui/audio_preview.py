import simpleaudio as sa
import threading
import subprocess
import shutil


class PlaybackError(Exception):
    pass


class PreviewPlayer:
    def __init__(self, on_done=None) -> None:
        self._play_obj = None
        self._ffplay_proc = None
        self._play_lock = threading.RLock()
        self._is_playing = False
        self._on_done = on_done

    def stop_preview(self) -> None:
        with self._play_lock:
            if self._play_obj is not None:
                try:
                    self._play_obj.stop()
                except Exception:
                    pass
                self._play_obj = None
            if self._ffplay_proc is not None:
                try:
                    self._ffplay_proc.terminate()
                except Exception:
                    pass
                self._ffplay_proc = None
            self._is_playing = False

    def _monitor_playback(self, play_obj, proc) -> None:
        try:
            if play_obj is not None:
                try:
                    play_obj.wait_done()
                except Exception:
                    pass
            elif proc is not None:
                try:
                    proc.wait()
                except Exception:
                    pass
        finally:
            with self._play_lock:
                self._is_playing = False
                self._play_obj = None
                self._ffplay_proc = None
            if self._on_done:
                try:
                    self._on_done()
                except Exception:
                    pass

    def play_preview(
        self, path: str, start_ms: int = 30000, duration_ms: int = 15000
    ) -> None:
        with self._play_lock:
            if self._is_playing:
                self.stop_preview()
            self._is_playing = True

        try:
            from pydub import AudioSegment

            clip = AudioSegment.from_file(path)[start_ms : start_ms + duration_ms]
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
        except Exception as sa_error:
            ffplay = shutil.which("ffplay")
            if not ffplay:
                with self._play_lock:
                    self._is_playing = False
                raise PlaybackError(
                    f"simpleaudio failed and ffplay not found: {sa_error}"
                ) from sa_error
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
                raise PlaybackError(f"ffplay failed: {exc}") from exc
