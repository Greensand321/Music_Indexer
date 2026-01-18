from __future__ import annotations

import re

UNKNOWN_YEAR = "Unknown"
_YEAR_PATTERN = re.compile(r"(1[0-9]{3}|2[0-9]{3})")


def normalize_year(value: object) -> str:
    if value is None:
        return UNKNOWN_YEAR
    if isinstance(value, (int, float)):
        return _format_year(int(value))
    try:
        text = str(value).strip()
    except Exception:
        return UNKNOWN_YEAR
    if not text:
        return UNKNOWN_YEAR
    match = _YEAR_PATTERN.search(text)
    if not match:
        return UNKNOWN_YEAR
    return match.group(1)


def _format_year(year: int) -> str:
    if 1000 <= year <= 2999:
        return f"{year:04d}"
    return UNKNOWN_YEAR
