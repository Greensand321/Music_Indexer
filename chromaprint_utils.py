import subprocess
import tempfile
import os
import json
import shutil
import time
import logging

from utils.path_helpers import strip_ext_prefix

verbose: bool = True
_logger = logging.getLogger(__name__)


def _dlog(label: str, msg: str) -> None:
    if not verbose:
        return
    _logger.debug(f"{time.strftime('%H:%M:%S')} [{label}] {msg}")

# Backwards compatibility -------------------------------------------------
strip_long_path_prefix = strip_ext_prefix

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
    start_sec: float = 0.0,
    duration_sec: float = 120.0,
    threshold_db: float = -50.0,
    min_silence_duration: float = 0.5,
) -> str | None:
    """Return fingerprint string computed via fpcalc."""
    ensure_tool("fpcalc")
    ensure_tool("ffmpeg")
    tmp1 = None
    tmp2 = None
    to_process = path
    try:
        if trim:
            tmp1 = trim_silence(
                path,
                threshold_db=threshold_db,
                min_silence_duration=min_silence_duration,
            )
        else:
            tmp1 = path

        tmp2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp2.close()
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-t",
            str(duration_sec),
            "-i",
            tmp1,
            "-ar",
            str(44100),
            "-ac",
            str(1),
            tmp2.name,
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        to_process = tmp2.name
        safe_path = strip_ext_prefix(to_process)
        cmd = ["fpcalc", "-json", safe_path]
        _dlog("FPCLI", f"cmd={cmd}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _dlog("FPCLI", f"stdout={proc.stdout.strip()}")
        _dlog("FPCLI", f"stderr={proc.stderr.strip()}")
        if proc.returncode != 0:
            raise FingerprintError(proc.stderr.strip())
        data = json.loads(proc.stdout)
        fp = data.get("fingerprint")
        if not fp:
            return None
        fp_str = " ".join(fp.split(","))
        _dlog("FP", f"value={fp_str} prefix={fp_str[:16]}")
        return fp_str
    finally:
        for t in (tmp1 if trim else None, tmp2.name if tmp2 else None):
            if t and os.path.exists(t) and t != path:
                try:
                    os.remove(t)
                except OSError:
                    pass
