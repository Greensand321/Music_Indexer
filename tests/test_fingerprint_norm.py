import io
import types
import sys
import importlib
from pydub import AudioSegment
from pydub.generators import Sine


def setup_acoustid(monkeypatch):
    acoustid_stub = types.ModuleType('acoustid')
    def fake_fp(fileobj=None, path=None):
        if fileobj is None:
            fileobj = open(path, 'rb')
        data = fileobj.read()
        return 0, str(hash(data))
    acoustid_stub.fingerprint_file = fake_fp
    monkeypatch.setitem(sys.modules, 'acoustid', acoustid_stub)


class DummyLogger:
    def __init__(self):
        self.msgs = []

    def __call__(self, msg):
        self.msgs.append(msg)


def test_trim_leading_silence(tmp_path, monkeypatch):
    setup_acoustid(monkeypatch)
    import audio_norm
    tone = Sine(440).to_audio_segment(duration=1700)
    audio = AudioSegment.silent(duration=300) + tone
    path = tmp_path / "s.wav"
    audio.export(path, format="wav")
    buf = audio_norm.normalize_for_fp(str(path), fingerprint_duration_ms=2000)
    seg = AudioSegment.from_file(buf)
    lead = audio_norm.silence.detect_leading_silence(seg, silence_threshold=audio_norm.SILENCE_THRESH)
    assert lead <= 100


def test_format_independence(tmp_path, monkeypatch):
    setup_acoustid(monkeypatch)
    import audio_norm
    tone = Sine(440).to_audio_segment(duration=1000)
    a1 = tone.set_frame_rate(48000).set_channels(1)
    a2 = tone.set_frame_rate(44100).set_channels(2)
    p1 = tmp_path / "a1.wav"
    p2 = tmp_path / "a2.wav"
    a1.export(p1, format="wav")
    a2.export(p2, format="wav")
    buf1 = audio_norm.normalize_for_fp(str(p1), fingerprint_duration_ms=1000)
    buf2 = audio_norm.normalize_for_fp(str(p2), fingerprint_duration_ms=1000)
    import acoustid
    _, fp1 = acoustid.fingerprint_file(fileobj=buf1)
    _, fp2 = acoustid.fingerprint_file(fileobj=buf2)
    assert fp1 == fp2


def test_radio_edit_alignment(tmp_path, monkeypatch):
    setup_acoustid(monkeypatch)
    import audio_norm
    tone = Sine(440).to_audio_segment(duration=8000)
    long = tone
    short = tone[:5000]
    p1 = tmp_path / "long.wav"
    p2 = tmp_path / "short.wav"
    long.export(p1, format="wav")
    short.export(p2, format="wav")
    log = DummyLogger()
    audio_norm.normalize_for_fp(str(p1), fingerprint_duration_ms=5000, log_callback=log)
    audio_norm.normalize_for_fp(str(p2), fingerprint_duration_ms=5000, log_callback=log)
    assert any("WARNING" in m for m in log.msgs)


def test_padding_short_track(tmp_path, monkeypatch):
    setup_acoustid(monkeypatch)
    import audio_norm
    tone = Sine(440).to_audio_segment(duration=500)
    p = tmp_path / "s.wav"
    tone.export(p, format="wav")
    buf = audio_norm.normalize_for_fp(str(p), fingerprint_duration_ms=1000)
    seg = AudioSegment.from_file(buf)
    assert len(seg) == 1000

