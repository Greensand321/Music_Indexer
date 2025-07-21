from __future__ import annotations

import io
from typing import Callable

from pydub import AudioSegment, silence


def _with_thresh(func, *args, **kwargs):
    """Call ``func`` with the correct silence threshold argument."""
    try:
        return func(*args, silence_thresh=SILENCE_THRESH, **kwargs)
    except TypeError:
        return func(*args, silence_threshold=SILENCE_THRESH, **kwargs)


SILENCE_THRESH = -50  # dBFS used for silence detection


def normalize_for_fp(
    path: str,
    fingerprint_offset_ms: int = 0,
    fingerprint_duration_ms: int = 120_000,
    allow_mismatched_edits: bool = True,
    log_callback: Callable[[str], None] | None = None,
) -> io.BytesIO:
    """Return normalized audio segment for fingerprinting as a BytesIO buffer."""
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(44100).set_channels(2)

    lead = _with_thresh(silence.detect_leading_silence, audio)
    trim_lead = 0
    if lead > 100:
        trim_lead = min(lead - 100, 500)
    trail = 0
    end_sil = _with_thresh(silence.detect_silence, audio, min_silence_len=50)
    if end_sil:
        last_start, last_end = end_sil[-1]
        if last_end >= len(audio):
            trail = len(audio) - last_start
    trim_trail = 0
    if trail > 100:
        trim_trail = min(trail - 100, 500)

    if trim_lead or trim_trail:
        audio = audio[trim_lead: len(audio) - trim_trail]

    trimmed_len = len(audio)
    if abs(trimmed_len - fingerprint_duration_ms) > 5000:
        if log_callback:
            log_callback(
                f"WARNING: trimmed duration {trimmed_len}ms differs from target {fingerprint_duration_ms}ms"
            )
        if not allow_mismatched_edits:
            raise ValueError("Mismatched edit length")

    start = fingerprint_offset_ms
    segment = audio[start:start + fingerprint_duration_ms]
    if len(segment) < fingerprint_duration_ms:
        segment += AudioSegment.silent(duration=fingerprint_duration_ms - len(segment))

    buf = io.BytesIO()
    segment.export(buf, format="wav")
    buf.seek(0)
    return buf
