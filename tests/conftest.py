import types
import sys
import io
import os

# Qt must never try to open a real display during tests (CI has none, and a
# developer's desktop should not flash windows). Set before anything imports Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

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
        detect_leading_silence=lambda audio, silence_threshold=-50: 0,
        detect_silence=lambda audio, min_silence_len=50, silence_threshold=-50: []
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



# ── Isolation: never touch the real user profile ─────────────────────────────
#
# ``config.CONFIG_PATH`` is computed from HOME at import time, the landing page
# keeps an art-history file in HOME, and Library Sync writes logs under HOME.
# Every test gets a fresh, empty HOME so a test run can never read or clobber
# the developer's real ~/.soundvault_config.json (and so tests can't silently
# depend on whatever happens to be configured on the machine running them).

@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows expanduser()
    cfg_path = str(home / ".soundvault_config.json")
    try:
        import config
    except Exception:  # pragma: no cover - config import must never break the suite
        config = None
    if config is not None:
        monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    # Modules that copied CONFIG_PATH by value at import time.
    for mod_name in ("plugins.acoustid_plugin",):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "CONFIG_PATH"):
            monkeypatch.setattr(mod, "CONFIG_PATH", cfg_path)
    yield home


# ── Synthetic music library ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def _synthetic_library_template(tmp_path_factory):
    from library_factory import build_standard_library

    root = tmp_path_factory.mktemp("synthetic_template") / "Music"
    return build_standard_library(root)


@pytest.fixture
def synthetic_library(_synthetic_library_template, tmp_path):
    """A fresh copy of the standard synthetic library (see tests/library_factory.py).

    Real, tagged audio files: an MP3 album, a FLAC album, a cross-codec
    duplicate, M4A/OGG/Opus tracks, an untagged file, non-audio leftovers and
    a playlist. Safe to mutate — every test gets its own copy.
    """
    import shutil
    from library_factory import SyntheticLibrary

    template = _synthetic_library_template
    root = tmp_path / "Music"
    shutil.copytree(template.root, root)
    return SyntheticLibrary(
        root=root,
        tracks={root / p.relative_to(template.root): spec for p, spec in template.tracks.items()},
        extras=[root / p.relative_to(template.root) for p in template.extras],
    )
