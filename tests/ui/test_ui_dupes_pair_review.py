"""Tools → Pair Review (DUP-06 + DUP-08, fixed together).

DUP-06: ``QtWidgets.QShortcut`` (Qt5 location) made the constructor raise, so
the dialog never opened. DUP-08, hidden behind it: Return was bound to
"delete inferior" dialog-wide (Enter in "Jump to" deleted a file), deletion
was an immediate ``os.remove`` with no confirmation, quarantine or log, and
filename pairing spanned the whole library, so a different song that merely
shared a file name could be the one deleted.
"""
from __future__ import annotations

import json
import os

import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

YES = QtWidgets.QMessageBox.StandardButton.Yes


@pytest.fixture
def no_remove(monkeypatch):
    import gui.dialogs.pair_review_dialog as mod

    def forbidden(*_a, **_k):
        raise AssertionError("Pair Review must never delete files")

    monkeypatch.setattr(mod.os, "remove", forbidden)
    monkeypatch.setattr(mod.os, "unlink", forbidden)


def _lib(tmp_path):
    root = tmp_path / "Lib"
    winner = root / "Artist" / "Album" / "01 - Song.flac"
    loser = root / "Artist" / "Album" / "01 - Song.mp3"
    for p, data in ((winner, b"flac"), (loser, b"mp3")):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    playlist = root / "Playlists" / "mix.m3u"
    playlist.parent.mkdir(parents=True)
    playlist.write_text("#EXTM3U\n../Artist/Album/01 - Song.mp3\n../Artist/Album/other.mp3\n", encoding="utf-8")
    docs = root / "Docs"
    docs.mkdir()
    (docs / "duplicate_pair_review.json").write_text(json.dumps({"pairs": [
        {"left_path": str(winner), "right_path": str(loser), "winner_path": str(winner)},
    ]}), encoding="utf-8")
    return root, winner, loser, playlist


def _open(qtbot, root):
    from gui.dialogs.pair_review_dialog import PairReviewDialog

    dlg = PairReviewDialog(str(root))
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    return dlg


def test_dialog_opens(qtbot, tmp_path):
    root, *_ = _lib(tmp_path)
    dlg = _open(qtbot, root)
    assert dlg._progress_lbl.text() == "Pair 1 of 1"
    assert dlg.testAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)


def test_enter_never_acts_on_files(qtbot, tmp_path, dialogs, no_remove):
    root, winner, loser, _ = _lib(tmp_path)
    dlg = _open(qtbot, root)
    dlg._jump_edit.setFocus()
    QTest.keyClicks(dlg._jump_edit, "1")
    QTest.keyClick(dlg._jump_edit, QtCore.Qt.Key.Key_Return)
    QTest.keyClick(dlg, QtCore.Qt.Key.Key_Return)
    QTest.keyClick(dlg, QtCore.Qt.Key.Key_Enter)
    dlg._no_btn.setFocus()
    QTest.keyClick(dlg._no_btn, QtCore.Qt.Key.Key_Return)   # a focused button may only do its own job
    assert winner.exists() and loser.exists()
    assert dialogs.calls_of("question") == []
    assert not any(b.isDefault() or b.autoDefault() for b in dlg.findChildren(QtWidgets.QPushButton))


def test_quarantine_needs_confirmation(qtbot, tmp_path, dialogs, no_remove):
    root, winner, loser, _ = _lib(tmp_path)
    dlg = _open(qtbot, root)
    dialogs.answer_no()
    dlg._yes_btn.click()
    (question,) = dialogs.calls_of("question")
    assert str(loser) in question.text and str(winner) in question.text
    assert loser.exists()
    assert not (root / "Quarantine").exists()


def test_confirmed_quarantine_moves_logs_and_fixes_playlists(qtbot, tmp_path, dialogs, no_remove):
    root, winner, loser, playlist = _lib(tmp_path)
    (root / "Quarantine").mkdir()
    (root / "Quarantine" / loser.name).write_bytes(b"an earlier quarantined file")
    dlg = _open(qtbot, root)
    dialogs.answer_yes()
    dlg._yes_btn.click()

    assert not loser.exists() and winner.exists()
    moved = root / "Quarantine" / f"{loser.stem} (1){loser.suffix}"
    assert moved.read_bytes() == b"mp3"
    assert (root / "Quarantine" / loser.name).read_bytes() == b"an earlier quarantined file"
    log = (root / "Docs" / "duplicate_pair_review_log.txt").read_text(encoding="utf-8")
    assert "\tquarantine\t" in log and str(loser) in log and str(moved) in log and str(winner) in log
    lines = playlist.read_text(encoding="utf-8").splitlines()
    assert "../Artist/Album/01 - Song.flac" in lines
    assert "../Artist/Album/01 - Song.mp3" not in lines
    assert "../Artist/Album/other.mp3" in lines
    assert dlg._progress_lbl.text() == "All pairs reviewed."


def test_filename_pairs_stay_within_one_folder(tmp_path):
    from gui.dialogs.pair_review_dialog import _collect_filename_pairs

    root = tmp_path / "Lib"
    same_a = root / "A" / "Album1" / "Intro.mp3"
    same_b = root / "A" / "Album1" / "Intro (1).mp3"
    other_album = root / "B" / "Album2" / "Intro.mp3"
    quarantined = root / "Quarantine" / "Intro.mp3"
    for p in (same_a, same_b, other_album, quarantined):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    pairs = _collect_filename_pairs(str(root))
    assert [tuple(sorted(p)) for p in pairs] == [tuple(sorted((str(same_a), str(same_b))))]


def test_no_winner_and_no_flac_moves_nothing(qtbot, tmp_path, dialogs, no_remove):
    root = tmp_path / "Lib"
    a, b = root / "A" / "x.mp3", root / "A" / "x (1).mp3"
    for p in (a, b):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    (root / "Docs").mkdir()
    (root / "Docs" / "duplicate_pair_review.json").write_text(json.dumps(
        {"pairs": [{"left_path": str(a), "right_path": str(b)}]}), encoding="utf-8")
    dlg = _open(qtbot, root)
    dlg._yes_btn.click()
    assert a.exists() and b.exists()
    assert dialogs.calls_of("question") == []
    assert "nothing moved" in dlg._status_lbl.text()
