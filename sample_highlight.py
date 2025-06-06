# sample_highlight.py

try:
    # pydub requires the built-in audioop (or pyaudioop) module. On Python 3.13,
    # audioop may be missing, so this may throw ImportError or ModuleNotFoundError.
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYDUB_AVAILABLE = False


def play_file_highlight(path, duration_ms=5000):
    """
    Play a short highlight from `path`. Returns the start time (in seconds)
    where the snippet began. Raises RuntimeError if pydub/audioop isn't available.
    """
    if not PYDUB_AVAILABLE:
        raise RuntimeError(
            "pydub (and/or audioop) is not available. "
            "To enable audio highlighting, run:\n"
            "  pip install pydub\n"
            "and ensure ffmpeg is on your PATH (or use Python 3.11/3.10)."
        )

    # Load the audio into a pydub AudioSegment
    audio = AudioSegment.from_file(path)

    # Compute midpoint and extract a snippet of `duration_ms` length
    midpoint_ms = len(audio) // 2
    start_ms = max(0, midpoint_ms - duration_ms // 2)
    snippet = audio[start_ms : start_ms + duration_ms]

    # Play the snippet (calls ffmpeg under the hood)
    play(snippet)

    return start_ms / 1000.0
