"""ThreadRecorder.wait_all must not return while a finished thread's signal
is still queued (Windows CI: test_bar_drops_previous_cover_for_a_track_without_art).

On Windows a thread often finished *between* wait_all's processEvents() and
its alive() check: wait_all saw nothing running and returned, and the final
processEvents() then delivered the thread's "art ready" signal, whose slot
started the neighbour-art loaders, which were reported as still running.
The race is reproduced deterministically here by finishing the thread right
after the first processEvents() call.
"""
from __future__ import annotations

import threading
import time

from PySide6 import QtCore, QtWidgets


class _Emitter(QtCore.QThread):
    done = QtCore.Signal()

    def __init__(self, gate: threading.Event) -> None:
        super().__init__()
        self._gate = gate

    def run(self) -> None:
        self._gate.wait(5)
        self.done.emit()


class _Follower(QtCore.QThread):
    def run(self) -> None:
        time.sleep(0.2)


class _Receiver(QtCore.QObject):
    """Lives on the GUI thread, so ``done`` reaches it as a queued signal."""

    def __init__(self) -> None:
        super().__init__()
        self.followers: list[_Follower] = []

    @QtCore.Slot()
    def start_follower(self) -> None:
        t = _Follower()
        self.followers.append(t)
        t.start()


def test_wait_all_delivers_signals_of_threads_that_finish_late(threads, monkeypatch):
    gate = threading.Event()
    first = _Emitter(gate)
    receiver = _Receiver()
    first.done.connect(receiver.start_follower)
    first.start()

    real_process_events = QtWidgets.QApplication.processEvents

    def process_events(*args):
        real_process_events(*args)
        if not gate.is_set():
            # The worker finishes right after this processEvents() returned:
            # its signal is queued, and alive() will no longer list it.
            gate.set()
            assert first.wait(5000)

    monkeypatch.setattr(QtWidgets.QApplication, "processEvents", staticmethod(process_events))

    assert threads.wait_all(5000) == []
    assert len(receiver.followers) == 1
    assert receiver.followers[0].isFinished()
