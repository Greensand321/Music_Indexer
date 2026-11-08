"""Decode tag text that arrives as raw bytes (MP4 freeform atoms, APE, odd taggers)."""
from __future__ import annotations

_BOMS = (
    (b"\xef\xbb\xbf", "utf-8"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)


def decode_tag_bytes(value: bytes) -> str:
    """Best reading of ``value`` as text; never raises.

    UTF-8 first, then UTF-16 *only* with a byte-order mark, then Windows-1252
    (the usual encoding of old Western tags), then Latin-1, which accepts any
    bytes. Trying UTF-16 without a BOM — as the readers used to — succeeds on
    almost any even-length input and turned ``b"Caf\\xe9"`` into two CJK
    characters.
    """
    data = bytes(value)
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            try:
                return data[len(bom):].decode(encoding).rstrip("\x00")
            except UnicodeDecodeError:
                break
    try:
        return data.decode("utf-8").rstrip("\x00")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252").rstrip("\x00")
    except UnicodeDecodeError:
        return data.decode("latin-1").rstrip("\x00")
