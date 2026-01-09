# dry_run_coordinator.py
"""Thread-safe coordinator for dry-run phases."""
from __future__ import annotations

import threading
from typing import List, Any, Dict


class DryRunCoordinator:
    """Aggregate results and HTML from dry-run phases."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.exact_dupes: List[Any] = []
        self.metadata_groups: List[Any] = []
        self.near_dupe_clusters: List[Any] = []
        self._sections: Dict[str, str] = {}

    def add_exact_dupes(self, items: List[Any]) -> None:
        with self._lock:
            self.exact_dupes.extend(items)

    def add_metadata_groups(self, items: List[Any]) -> None:
        with self._lock:
            self.metadata_groups.extend(items)

    def add_near_dupe_clusters(self, clusters: List[Any]) -> None:
        with self._lock:
            self.near_dupe_clusters.extend(clusters)

    def set_html_section(self, phase: str, html: str) -> None:
        with self._lock:
            self._sections[phase] = html or ""

    def assemble_final_report(self) -> str:
        with self._lock:
            parts = []
            for p in ["A", "B", "C"]:
                section = self._sections.get(p)
                if section:
                    parts.append(section)
            return "\n".join(parts)
