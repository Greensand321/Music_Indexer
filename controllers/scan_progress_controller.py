import threading
from typing import Callable, Any


class ScanProgressController:
    """Helper for coordinating scan progress updates and cancellation."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._callback: Callable[..., Any] | None = None

    def set_callback(self, cb: Callable[..., Any]) -> None:
        """Register a callback for progress events."""
        self._callback = cb

    def update(self, *args, **kwargs) -> None:
        """Forward a progress event to the registered callback."""
        if self._callback:
            self._callback(*args, **kwargs)
