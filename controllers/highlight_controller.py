try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYDUB_AVAILABLE = False


def play_snippet(file_path: str, duration: float = 5.0) -> float:
    """Play a snippet around the midpoint of ``file_path`` and return the start time in seconds."""
    if not PYDUB_AVAILABLE:
        raise RuntimeError(
            "pydub (and/or audioop) is not available.\n"
            "To enable audio highlighting, run:\n  pip install pydub\n"
            "and ensure ffmpeg is on your PATH."
        )
    audio = AudioSegment.from_file(file_path)
    duration_ms = int(duration * 1000)
    midpoint_ms = len(audio) // 2
    start_ms = max(0, midpoint_ms - duration_ms // 2)
    snippet = audio[start_ms : start_ms + duration_ms]
    play(snippet)
    return start_ms / 1000.0
