from __future__ import annotations

from dataclasses import dataclass


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
