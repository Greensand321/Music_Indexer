from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class ApiService(ABC):
    """Abstract base class for external metadata services."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Return a tuple ``(success, message)`` after checking connectivity."""
        raise NotImplementedError

    @abstractmethod
    def query(self, fingerprint: str) -> Dict:
        """Query the service using ``fingerprint`` and return metadata."""
        raise NotImplementedError
