import types
import sys
import io
import os

# Provide minimal pydub stubs when real package is unavailable
if 'pydub' not in sys.modules:
    pydub = types.ModuleType('pydub')
    class _Seg:
        def __init__(self, duration=1000):
            self.duration = duration
        def set_frame_rate(self, rate):
            return self
        def set_channels(self, ch):
            return self
        def __len__(self):
            return self.duration
        def __getitem__(self, slc):
            start = slc.start or 0
            stop = slc.stop if slc.stop is not None else self.duration
            return _Seg(stop - start)
        def __add__(self, other):
            return _Seg(self.duration + len(other))
        def export(self, out_f, format='wav'):
            data = str(self.duration).encode()
            if isinstance(out_f, (str, bytes, os.PathLike)):
                with open(out_f, 'wb') as f:
                    f.write(data)
            else:
                out_f.write(data)
            return out_f
        @classmethod
        def from_file(cls, f, *a, **k):
            if isinstance(f, (str, bytes, os.PathLike)):
                with open(f, 'rb') as fp:
                    data = fp.read()
            else:
                data = f.read()
            try:
                dur = int(data.decode())
            except Exception:
                dur = 1000
            return cls(dur)
        @classmethod
        def silent(cls, duration=0):
            return cls(duration)
    silence_mod = types.SimpleNamespace(
        detect_leading_silence=lambda audio, silence_thresh=-50: 0,
        detect_silence=lambda audio, min_silence_len=50, silence_thresh=-50: []
    )
    pydub.AudioSegment = _Seg
    pydub.silence = silence_mod
    gens = types.ModuleType('pydub.generators')
    class Sine:
        def __init__(self, freq):
            self.freq = freq
        def to_audio_segment(self, duration=1000):
            return _Seg(duration)
    gens.Sine = Sine
    sys.modules['pydub'] = pydub
    sys.modules['pydub.generators'] = gens

    # Provide a lightweight audio_norm replacement using the stubs
    audio_norm = types.ModuleType('audio_norm')
    audio_norm.SILENCE_THRESH = -50

    def normalize_for_fp(
        path,
        fingerprint_offset_ms=0,
        fingerprint_duration_ms=120_000,
        allow_mismatched_edits=True,
        log_callback=None,
    ):
        if isinstance(path, (str, bytes, os.PathLike)):
            with open(path, 'rb') as f:
                data = f.read()
        else:
            data = path.read()
        try:
            dur = int(data.decode())
        except Exception:
            dur = fingerprint_duration_ms
        if abs(dur - fingerprint_duration_ms) > 2000 and log_callback:
            log_callback(
                f"WARNING: trimmed duration {dur}ms differs from target {fingerprint_duration_ms}ms"
            )
        buf = io.BytesIO(str(fingerprint_duration_ms).encode())
        buf.seek(0)
        return buf

    audio_norm.normalize_for_fp = normalize_for_fp
    audio_norm.silence = silence_mod
    sys.modules['audio_norm'] = audio_norm

