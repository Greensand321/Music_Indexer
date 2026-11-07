"""Helpers for the Tag Fixer / Genre Normalizer GUI tests (area: meta).

* ``FakeSource`` — a Tag Fixer lookup source answering by track title. It is
  installed in place of ``tag_fixer.PLUGINS`` so no test touches AcoustID,
  MusicBrainz or Last.fm (or needs fpcalc).
* ``table_rows`` — the Tag Fixer proposal grid as plain tuples.
"""
from __future__ import annotations

import threading
import time

from PySide6 import QtCore

from plugins.base import SourceStatus


class FakeSource:
    def __init__(self, answers=None, *, name="FakeID", active=True, problem="", delay=0.0):
        self.name = name
        self.answers = answers or {}
        self.active = active
        self.problem = problem
        self.delay = delay
        self.calls: list[str] = []
        self.release = threading.Event()  # set to stop waiting early

    def preflight(self):
        return SourceStatus(self.name, self.active, "test source", self.problem)

    def identify(self, path):
        self.calls.append(path)
        if self.delay:
            self.release.wait(self.delay)
        from utils.audio_metadata_reader import read_tags

        return dict(self.answers.get(read_tags(path).get("title"), {}))


def install_source(monkeypatch, answers=None, **kwargs) -> FakeSource:
    import tag_fixer

    source = FakeSource(answers, **kwargs)
    monkeypatch.setattr(tag_fixer, "PLUGINS", [source])
    return source


def table_rows(ws, *, visible_only=True):
    """``[(file, field, current, proposed, checked)]`` of the proposal grid."""
    rows = []
    for i in range(ws._table.topLevelItemCount()):
        item = ws._table.topLevelItem(i)
        if visible_only and item.isHidden():
            continue
        rows.append((
            item.text(1), item.text(2), item.text(3), item.text(4),
            item.checkState(0) == QtCore.Qt.CheckState.Checked,
        ))
    return sorted(rows)


def set_row_checked(ws, file_name: str, field_label: str, checked: bool) -> None:
    for i in range(ws._table.topLevelItemCount()):
        item = ws._table.topLevelItem(i)
        if item.text(1) == file_name and item.text(2) == field_label:
            item.setCheckState(
                0, QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
            )
            return
    raise KeyError((file_name, field_label))


def wait_for(qtbot, predicate, timeout=10_000):
    qtbot.waitUntil(predicate, timeout=timeout)


def slow_reader(delay: float):
    """A genre reader that takes ``delay`` seconds per file (for in-flight tests)."""
    from utils.audio_metadata_reader import read_genres

    def reader(path):
        time.sleep(delay)
        return {"genre": read_genres(path)}

    return reader
