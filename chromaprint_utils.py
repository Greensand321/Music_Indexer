import subprocess
import tempfile
import os
import json
import shutil

from utils.path_helpers import strip_long_path_prefix

class FingerprintError(Exception):
    """Raised when fingerprint computation fails."""


def ensure_tool(name: str) -> None:
    """Raise RuntimeError if external tool is missing."""
    if shutil.which(name) is None:
        raise FingerprintError(f"Required tool '{name}' not found")


def trim_silence(
    input_path: str,
    *,
    threshold_db: float = -50.0,
    min_silence_duration: float = 0.5,
    sample_rate: int = 44100,
    channels: int = 1,
) -> str:
    """Use FFmpeg to trim leading and trailing silence."""
    ensure_tool("ffmpeg")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    output_path = tmp.name
    ffmpeg_filter = (
        f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_duration={min_silence_duration},"
        "areverse,"
        f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_duration={min_silence_duration},"
        "areverse"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-af",
        ffmpeg_filter,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        output_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        os.remove(output_path)
        msg = proc.stderr.decode(errors="ignore").strip()
        raise FingerprintError(f"FFmpeg error: {msg}")
    return output_path


def fingerprint_fpcalc(
    path: str,
    *,
    trim: bool = True,
    threshold_db: float = -50.0,
    min_silence_duration: float = 0.5,
) -> str | None:
    """Return fingerprint string computed via fpcalc."""
    ensure_tool("fpcalc")
    tmp = None
    to_process = path
    try:
        if trim:
            tmp = trim_silence(
                path,
                threshold_db=threshold_db,
                min_silence_duration=min_silence_duration,
            )
            to_process = tmp
        safe_path = strip_long_path_prefix(to_process)
        cmd = ["fpcalc", "-json", safe_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise FingerprintError(proc.stderr.strip())
        data = json.loads(proc.stdout)
        fp = data.get("fingerprint")
        if not fp:
            return None
        return " ".join(fp.split(","))
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
