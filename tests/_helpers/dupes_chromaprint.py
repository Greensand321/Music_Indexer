"""Chromaprint fixtures for the duplicate-detection tests.

* ``compress_chromaprint`` — pure-Python port of chromaprint's
  FingerprintCompressor: raw 32-bit sub-fingerprints → the compressed,
  URL-safe base64 string that ``fpcalc -json`` and pyacoustid emit.
* ``fake_raw_print`` / ``flip_bits`` — deterministic synthetic raw prints:
  the same seed is "the same recording", a few flipped bits is "a lossy copy".
* ``real_fingerprints`` — genuine fingerprints of synthetic audio
  (``tests/data/dupes_real_fingerprints.json``).
* ``write_fake_fpcalc`` — an executable stand-in for ``fpcalc`` that prints
  ``-json`` output, for tests that must run the real ``chromaprint_utils``.
"""
from __future__ import annotations

import base64
import json
import os
import random
import stat
import sys
from pathlib import Path
from typing import Iterable, Sequence

DATA = Path(__file__).resolve().parents[1] / "data" / "dupes_real_fingerprints.json"


def _pack(values: Iterable[int], width: int) -> bytes:
    acc = nbits = 0
    out = bytearray()
    for value in values:
        acc |= (value & ((1 << width) - 1)) << nbits
        nbits += width
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def compress_chromaprint(items: Sequence[int], algorithm: int = 1) -> str:
    normal: list[int] = []
    exceptional: list[int] = []
    prev = 0
    for idx, item in enumerate(items):
        item &= 0xFFFFFFFF
        x = item if idx == 0 else item ^ prev
        prev = item
        bit, last_bit = 1, 0
        while x:
            if x & 1:
                value = bit - last_bit
                if value >= 7:
                    normal.append(7)
                    exceptional.append(value - 7)
                else:
                    normal.append(value)
                last_bit = bit
            x >>= 1
            bit += 1
        normal.append(0)
    n = len(items)
    header = bytes([algorithm & 0xFF, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF])
    payload = header + _pack(normal, 3) + _pack(exceptional, 5)
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def fake_raw_print(seed: object, n: int = 240) -> list[int]:
    rng = random.Random(str(seed))
    return [rng.getrandbits(32) for _ in range(n)]


def flip_bits(items: Sequence[int], count: int, seed: object = 0) -> list[int]:
    rng = random.Random(f"flip-{seed}")
    out = list(items)
    for _ in range(count):
        i = rng.randrange(len(out))
        out[i] ^= 1 << rng.randrange(32)
    return out


def fake_fingerprint(seed: object, *, flips: int = 0, n: int = 240) -> str:
    """A compressed fingerprint: same ``seed`` = same recording."""
    raw = fake_raw_print(seed, n)
    if flips:
        raw = flip_bits(raw, flips, seed)
    return compress_chromaprint(raw)


def real_fingerprints() -> dict[str, str]:
    return json.loads(DATA.read_text(encoding="utf-8"))["fingerprints"]


def write_fake_fpcalc(bin_dir: Path) -> Path:
    """Write an executable ``fpcalc`` into ``bin_dir`` (prepend it to PATH).

    It prints ``{"duration": ..., "fingerprint": ...}`` like ``fpcalc -json``.
    The fingerprint is derived from the WAV's length (ffmpeg converts every
    input to a temporary WAV first), so equal-length clips get equal prints.
    """
    if os.name == "nt":  # pragma: no cover - a shebang script is POSIX-only
        raise RuntimeError("write_fake_fpcalc is POSIX-only")
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "fpcalc"
    helper = Path(__file__).resolve()
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(helper.parent.parent)!r})\n"
        "from _helpers.dupes_chromaprint import fake_fingerprint\n"
        "path = [a for a in sys.argv[1:] if not a.startswith('-')][-1]\n"
        "size = os.path.getsize(path)\n"
        "print(json.dumps({'duration': round(size / 88200, 2),"
        " 'fingerprint': fake_fingerprint(size // 4096)}))\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script
