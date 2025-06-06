"""
sample_highlight.py

Play the loudest 30-second segment of an audio file.

Usage:
    python sample_highlight.py path/to/song.mp3

Requires ``pydub`` and ``simpleaudio`` packages.
"""

import sys
from pydub import AudioSegment
import simpleaudio as sa

WINDOW_MS = 30_000  # 30 seconds
STEP_MS = 1_000     # slide window every second


def find_loudest_segment(audio: AudioSegment, window_ms: int = WINDOW_MS, step_ms: int = STEP_MS) -> int:
    """Return the start time (ms) of the loudest window of the given audio."""
    if len(audio) <= window_ms:
        return 0

    best_start = 0
    best_rms = -1
    for start in range(0, len(audio) - window_ms + 1, step_ms):
        seg = audio[start:start + window_ms]
        if seg.rms > best_rms:
            best_rms = seg.rms
            best_start = start
    return best_start


def play_segment(segment: AudioSegment) -> None:
    """Play the given audio segment and wait until it finishes."""
    playback = sa.play_buffer(
        segment.raw_data,
        num_channels=segment.channels,
        bytes_per_sample=segment.sample_width,
        sample_rate=segment.frame_rate,
    )
    playback.wait_done()


def play_file_highlight(path: str) -> float:
    """Play the loudest 30-second highlight of the given audio file.

    Returns the start time of the highlight in seconds."""
    try:
        audio = AudioSegment.from_file(path)
    except Exception as exc:
        raise RuntimeError(f"Could not open '{path}': {exc}") from exc

    start_ms = find_loudest_segment(audio)
    highlight = audio[start_ms:start_ms + WINDOW_MS]
    play_segment(highlight)
    return start_ms / 1000


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python sample_highlight.py <audio_file>")
        return
    path = sys.argv[1]

    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        print(f"Could not open '{path}': {e}")
        return

    start_ms = find_loudest_segment(audio)
    highlight = audio[start_ms:start_ms + WINDOW_MS]
    print(f"Playing highlight starting at {start_ms / 1000:.2f} s")
    play_segment(highlight)


if __name__ == "__main__":
    main()
