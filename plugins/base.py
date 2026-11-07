from __future__ import annotations

import re
from dataclasses import dataclass

_QUERY = re.compile(r"\?\S*")


def describe_lookup_error(exc: BaseException) -> str:
    """A short, plain-words reason for a failed lookup.

    Shown in the Tag Fixer and its log, so it must never carry secrets:
    query strings (which hold ``api_key=…`` for some services) are stripped
    and the text is capped.
    """
    name = type(exc).__name__
    text = _QUERY.sub("?…", str(exc).strip())
    lowered = f"{name} {text}".lower()
    if "nobackend" in lowered or "fpcalc" in lowered or "chromaprint" in lowered:
        return "fingerprinting unavailable (fpcalc/Chromaprint not found)"
    if "fingerprintgeneration" in lowered:
        return "could not fingerprint the audio"
    if "invalid api key" in lowered or "invalid client" in lowered or ("key" in lowered and "invalid" in lowered):
        return "the API key was rejected"
    if any(word in lowered for word in ("connection", "timed out", "timeout", "resolve", "network", "unreachable")):
        return "network error (offline?)"
    message = f"{name}: {text}" if text else name
    return message if len(message) <= 120 else message[:117] + "…"


@dataclass
class SourceStatus:
    """Whether a lookup source will be used for a scan, and why (not).

    ``name`` is what the user sees ("AcoustID"); ``detail`` says where its
    configuration came from ("key from Settings"); ``problem`` explains, in
    plain words, why an expected source is off ("fpcalc not found"). A source
    that is simply not selected is inactive with an empty ``problem``.
    """

    name: str
    active: bool
    detail: str = ""
    problem: str = ""


class MetadataPlugin:
    #: Shown in the Tag Fixer's "lookup sources" line; ``None`` = not listed.
    name: str | None = None

    def preflight(self) -> SourceStatus:
        """Check, once per scan, whether this source can be used.

        The default is "always active". Sources that need a tool, a key or
        the network override this so a scan can say *why* nothing matched
        instead of failing silently file after file.
        """
        return SourceStatus(self.name or type(self).__name__, True)

    def identify(self, file_path: str) -> dict:
        """Given a filepath, return a dict with any of:
          - 'artist': str
          - 'title': str
          - 'album': str
          - 'genres': list[str]
          - 'score': float
        or return {} or None if no data. May raise on a lookup *failure*
        (bad key, network error); the scan records the reason per source."""
        raise NotImplementedError
