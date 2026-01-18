import threading

cancel_event = threading.Event()

class IndexCancelled(Exception):
    """Raised when indexing is cancelled."""
    pass


def check_cancelled() -> None:
    """Raise IndexCancelled if the shared cancel_event is set."""
    if cancel_event.is_set():
        raise IndexCancelled()
