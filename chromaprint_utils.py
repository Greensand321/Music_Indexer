import subprocess
import tempfile
import os
import json
import shutil
import time
import logging

from config import FP_SUBPROCESS_TIMEOUT_SEC, load_config
from utils.path_helpers import ensure_long_path, strip_ext_prefix

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


def _tail_stderr(stderr_text: str, max_lines: int = 10) -> str:
    if not stderr_text:
        return ""
    lines = stderr_text.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines).strip()


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
    if not os.path.exists(input_path):
        raise FingerprintError(f"file missing: {input_path}")
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
        ensure_long_path(input_path),
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
        msg = _tail_stderr(proc.stderr.decode(errors="ignore"))
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
    if not os.path.exists(path):
        raise FingerprintError(f"file missing: {path}")
    ensure_tool("fpcalc")
    ensure_tool("ffmpeg")
    tmp1 = None
    tmp2 = None
    to_process = path
    try:
        cfg = load_config()
        timeout_sec = float(cfg.get("fingerprint_subprocess_timeout_sec", FP_SUBPROCESS_TIMEOUT_SEC))
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
            ensure_long_path(tmp1),
            "-ar",
            str(44100),
            "-ac",
            str(1),
            tmp2.name,
        ]
        try:
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise FingerprintError("Fingerprint timeout") from exc
        except subprocess.CalledProcessError as exc:
            msg = _tail_stderr((exc.stderr or b"").decode(errors="ignore"))
            raise FingerprintError(f"FFmpeg error: {msg}") from exc
        to_process = tmp2.name
        safe_path = strip_ext_prefix(to_process)
        cmd = ["fpcalc", "-json", safe_path]
        _dlog("FPCLI", f"cmd={cmd}")
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise FingerprintError("Fingerprint timeout") from exc
        _dlog("FPCLI", f"stdout={proc.stdout.strip()}")
        _dlog("FPCLI", f"stderr={_tail_stderr(proc.stderr)}")
        if proc.returncode != 0:
            raise FingerprintError(_tail_stderr(proc.stderr))
        data = json.loads(proc.stdout)
        fp = data.get("fingerprint")
        if not fp:
            return None
        fp_str = " ".join(fp.split(","))
        _dlog("FP", f"prefix={fp_str[:16]} len={len(fp_str)}")
        return fp_str
    finally:
        for t in (tmp1 if trim else None, tmp2.name if tmp2 else None):
            if t and os.path.exists(t) and t != path:
                try:
                    os.remove(t)
                except OSError:
                    pass
