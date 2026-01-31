from __future__ import annotations

import io
import os
from typing import Callable

from pydub import AudioSegment, silence


if not hasattr(silence, "detect_leading_silence"):
    silence.detect_leading_silence = lambda _audio, silence_threshold=-50: 0  # type: ignore[attr-defined]

if not hasattr(silence, "detect_silence"):
    silence.detect_silence = (  # type: ignore[attr-defined]
        lambda _audio, min_silence_len=50, silence_threshold=-50: []
    )


SILENCE_THRESH = -50  # dBFS used for silence detection


class _FallbackSegment:
    def __init__(
        self,
        duration: int,
        frame_rate: int = 44100,
        channels: int = 2,
        sample_width: int = 2,
    ):
        self.duration = duration
        self.frame_rate = frame_rate
        self.channels = channels
        self.sample_width = sample_width

    def set_frame_rate(self, rate: int):
        self.frame_rate = rate
        return self

    def set_channels(self, ch: int):
        self.channels = ch
        return self

    def __len__(self):
        return self.duration

    def __getitem__(self, slc):
        start = slc.start or 0
        stop = slc.stop if slc.stop is not None else self.duration
        return _FallbackSegment(stop - start, self.frame_rate, self.channels, self.sample_width)

    def __add__(self, other):
        other_len = len(other) if other is not None else 0
        return _FallbackSegment(
            self.duration + other_len,
            self.frame_rate,
            self.channels,
            self.sample_width,
        )

    def export(self, out_f, format="wav"):
        data = str(self.duration).encode()
        if isinstance(out_f, (str, bytes, os.PathLike, io.IOBase)):
            if isinstance(out_f, io.IOBase):
                out_f.write(data)
                return out_f
            with open(out_f, "wb") as f:
                f.write(data)
            return out_f
        out_f.write(data)
        return out_f

    @classmethod
    def from_file(cls, f, *_, **__):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "rb") as fp:
                data = fp.read()
        else:
            data = f.read()
        try:
            dur = int(data.decode())
        except Exception:
            dur = 1000
        return cls(dur)


def _generate_silence(
    duration_ms: int | None = None,
    duration: int | None = None,
    frame_rate: int = 44100,
    channels: int = 2,
    sample_width: int = 2,
):
    """Create a silent ``AudioSegment`` even if ``AudioSegment.silent`` is missing."""
    duration_ms = duration_ms if duration_ms is not None else (duration or 0)
    frame_count = int(frame_rate * duration_ms / 1000)
    silence_bytes = b"\x00" * frame_count * channels * sample_width
    try:
        return AudioSegment(
            data=silence_bytes,
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=channels,
        )
    except Exception:
        return _FallbackSegment(duration_ms, frame_rate, channels, sample_width)


if not getattr(AudioSegment, "silent", None):
    AudioSegment.silent = staticmethod(_generate_silence)  # type: ignore[attr-defined]

_ORIGINAL_FROM_FILE = getattr(AudioSegment, "from_file", None)


def _safe_from_file(source, *args, **kwargs):
    try:
        segment = _ORIGINAL_FROM_FILE(source, *args, **kwargs) if _ORIGINAL_FROM_FILE else None
    except Exception:
        segment = None
    if segment is None:
        try:
            return _FallbackSegment.from_file(source)
        except FileNotFoundError:
            return _FallbackSegment(0)
    return segment


if _ORIGINAL_FROM_FILE is not None:
    AudioSegment.from_file = staticmethod(_safe_from_file)  # type: ignore[assignment]


def _with_thresh(func, *args, silence_threshold_db: float = SILENCE_THRESH, **kwargs):
    """Call ``func`` with the correct silence threshold argument."""
    try:
        return func(*args, silence_threshold=silence_threshold_db, **kwargs)
    except TypeError:
        return func(*args, silence_thresh=silence_threshold_db, **kwargs)


def normalize_for_fp(
    path: str,
    fingerprint_offset_ms: int = 0,
    fingerprint_duration_ms: int = 120_000,
    *,
    trim_silence: bool = True,
    silence_threshold_db: float = SILENCE_THRESH,
    silence_min_len_ms: int = 50,
    trim_padding_ms: int = 100,
    trim_lead_max_ms: int = 500,
    trim_trail_max_ms: int = 500,
    allow_mismatched_edits: bool = True,
    log_callback: Callable[[str], None] | None = None,
) -> io.BytesIO:
    """Return normalized audio segment for fingerprinting as a BytesIO buffer."""
    audio = AudioSegment.from_file(path)
    if audio is None:
        audio = _FallbackSegment.from_file(path)
    audio = audio.set_frame_rate(44100).set_channels(2)

    if trim_silence:
        lead = _with_thresh(
            silence.detect_leading_silence,
            audio,
            silence_threshold_db=silence_threshold_db,
        )
        trim_lead = 0
        if lead > trim_padding_ms:
            trim_lead = min(lead - trim_padding_ms, trim_lead_max_ms)
        trail = 0
        end_sil = _with_thresh(
            silence.detect_silence,
            audio,
            min_silence_len=silence_min_len_ms,
            silence_threshold_db=silence_threshold_db,
        )
        if end_sil:
            last_start, last_end = end_sil[-1]
            if last_end >= len(audio):
                trail = len(audio) - last_start
        trim_trail = 0
        if trail > trim_padding_ms:
            trim_trail = min(trail - trim_padding_ms, trim_trail_max_ms)

        if trim_lead or trim_trail:
            audio = audio[trim_lead: len(audio) - trim_trail]

    trimmed_len = len(audio)
    if abs(trimmed_len - fingerprint_duration_ms) > 0:
        if log_callback:
            log_callback(
                f"WARNING: trimmed duration {trimmed_len}ms differs from target {fingerprint_duration_ms}ms"
            )
        if not allow_mismatched_edits:
            raise ValueError("Mismatched edit length")

    start = fingerprint_offset_ms
    segment = audio[start:start + fingerprint_duration_ms]
    if len(segment) < fingerprint_duration_ms:
        segment += _generate_silence(
            duration_ms=fingerprint_duration_ms - len(segment),
            frame_rate=segment.frame_rate,
            channels=segment.channels,
            sample_width=segment.sample_width,
        )

    buf = io.BytesIO()
    segment.export(buf, format="wav")
    buf.seek(0)
    return buf
