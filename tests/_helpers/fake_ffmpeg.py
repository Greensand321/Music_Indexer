"""A fake ``ffmpeg`` that works on Windows as well as POSIX.

A shebang script named ``ffmpeg`` on PATH only works on POSIX: on Windows
``CreateProcess`` finds ``ffmpeg.exe`` and nothing else, so the real ffmpeg
(or none) ran instead. Here the fake is a Python script, and while installed
any ``subprocess.Popen`` of ``ffmpeg`` runs ``[sys.executable, script, *args]``
— a real child process, so terminating it is tested for real. Shims on PATH
(``ffmpeg`` with a shebang, ``ffmpeg.bat``) only make ``shutil.which("ffmpeg")``
succeed; they are never executed.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


def install_fake_ffmpeg(bin_dir: Path, monkeypatch, body: str) -> Path:
    """Install a fake ffmpeg whose Python source is ``body``; returns the script.

    ``body`` sees ``sys.argv`` exactly as ffmpeg would (``sys.argv[-1]`` is the
    output path).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "fake_ffmpeg.py"
    script.write_text(body, encoding="utf-8")

    shim = bin_dir / "ffmpeg"
    shim.write_text(f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(script)!r}, run_name='__main__')\n",
                    encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    (bin_dir / "ffmpeg.bat").write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    real_popen = subprocess.Popen

    class _FakeFfmpegPopen(real_popen):  # type: ignore[misc, valid-type]
        def __init__(self, args, *a, **kw):
            if (
                isinstance(args, (list, tuple))
                and args
                and os.path.basename(str(args[0])).lower() in ("ffmpeg", "ffmpeg.exe")
            ):
                args = [sys.executable, str(script), *map(str, args[1:])]
            super().__init__(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", _FakeFfmpegPopen)
    return script


def pid_alive(pid: int) -> bool:
    """Whether process ``pid`` is still running (a zombie counts as gone)."""
    if os.name == "nt":
        # os.kill(pid, 0) would *terminate* the process on Windows.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); check its state on Linux.
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        return "zombie" not in status.read_text().lower()
    return True
