"""Example script to run ``fpcalc`` on a list of audio files.

Invoke as ``python fingerprint_examples.py <files>``. Glob patterns are
allowed. On Windows, paths with a ``\\?\\`` prefix are handled correctly.
"""

import argparse
import glob
import subprocess
from pathlib import Path
from typing import Iterable


def strip_ext_prefix(p: str) -> str:
    """Remove Windows extended-length path prefix."""
    return p[4:] if p.startswith(r"\\?\\") else p


def expand_files(inputs: Iterable[str]) -> list[str]:
    """Return file paths from ``inputs`` with basic glob expansion."""

    paths: list[str] = []
    for raw in inputs:
        matches = glob.glob(raw)
        if matches:
            paths.extend(matches)
        else:
            paths.append(raw)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run fpcalc on files")
    parser.add_argument("files", nargs="+", help="Audio files or glob patterns")
    args = parser.parse_args(argv)

    files = expand_files(args.files)

    for path_str in files:
        safe_path = strip_ext_prefix(path_str)
        if not Path(safe_path).exists():
            print(f"Missing: {safe_path}")
            continue
        cmd = ["fpcalc", "-json", safe_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            print("❌ fpcalc not found on PATH.")
            break
        except subprocess.CalledProcessError as e:
            print(f"❌ fpcalc failed for {safe_path!r}:")
            print(e.stderr.strip())
            continue
        print(f"✅ {Path(safe_path).name}:")
        print(proc.stdout.strip())
        print("-" * 40)


if __name__ == "__main__":
    main()
