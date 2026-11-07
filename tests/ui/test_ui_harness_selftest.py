"""Tests for the harness itself — if these fail, no other UI result is trustworthy."""
from __future__ import annotations

import os

import pytest
from PySide6 import QtCore, QtWidgets

from ui_harness import QtWarningPolicy, UnexpectedDialog, stuck_opacity


def test_home_is_isolated(_isolated_home):
    import config

    assert os.environ["HOME"] == str(_isolated_home)
    assert config.CONFIG_PATH.startswith(str(_isolated_home))
    assert not os.path.exists(os.path.expanduser("~/.soundvault_config.json"))


def test_unqueued_question_raises(dialogs):
    with pytest.raises(UnexpectedDialog):
        QtWidgets.QMessageBox.question(None, "Delete?", "Really delete everything?")
    assert dialogs.calls_of("question")[0].title == "Delete?"


def test_queued_answers_are_consumed_in_order(dialogs, tmp_path):
    dialogs.answer_yes().answer_no().choose_directory(tmp_path)
    Yes = QtWidgets.QMessageBox.StandardButton.Yes
    No = QtWidgets.QMessageBox.StandardButton.No
    assert QtWidgets.QMessageBox.question(None, "a", "b") == Yes
    assert QtWidgets.QMessageBox.question(None, "a", "b") == No
    assert QtWidgets.QFileDialog.getExistingDirectory(None, "pick", "") == str(tmp_path)
    assert dialogs.pending() == {}


def test_information_boxes_are_auto_acknowledged(dialogs):
    QtWidgets.QMessageBox.information(None, "Done", "All finished")
    QtWidgets.QMessageBox.warning(None, "Careful", "Something odd")
    assert [c.kind for c in dialogs.calls] == ["information", "warning"]


def test_dialog_exec_is_intercepted(dialogs, qtbot):
    dlg = QtWidgets.QDialog()
    qtbot.addWidget(dlg)
    with pytest.raises(UnexpectedDialog):
        dlg.exec()
    seen = []
    dialogs.on_exec(lambda d: (seen.append(d), QtWidgets.QDialog.DialogCode.Accepted)[1])
    assert dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted.value
    assert seen == [dlg]


def test_menu_exec_triggers_chosen_action(dialogs, qtbot):
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    menu = QtWidgets.QMenu(host)
    fired = []
    menu.addAction("Copy").triggered.connect(lambda: fired.append("copy"))
    menu.addAction("Replace").triggered.connect(lambda: fired.append("replace"))
    dialogs.on_menu("Replace")
    action = menu.exec(QtCore.QPoint(0, 0))
    assert action is not None and action.text() == "Replace"
    assert fired == ["replace"]


def test_thread_recorder_waits_for_workers(threads):
    class Worker(QtCore.QThread):
        def run(self):
            self.msleep(50)

    w = Worker()
    w.start()
    assert threads.started == ["Worker"]
    assert threads.wait_all(2000) == []


def test_motion_recorder_compresses_and_settles(motion, qtbot):
    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    w.show()
    anim = QtCore.QPropertyAnimation(w, b"windowOpacity", w)
    anim.setDuration(5000)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.start()
    assert motion.records[0].prop == "windowOpacity"
    assert motion.records[0].duration <= 1  # default time_scale=0 compresses
    assert motion.settle(2000) == []
    assert stuck_opacity(w) == []


@pytest.mark.motion(time_scale=1.0)
def test_motion_marker_keeps_real_durations(motion, qtbot):
    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    anim = QtCore.QVariantAnimation(w)
    anim.setDuration(40)
    anim.setStartValue(0)
    anim.setEndValue(1)
    anim.start()
    assert motion.records[0].duration == 40
    assert motion.settle(2000) == []


def test_motion_recorder_flags_orphans(motion):
    anim = QtCore.QVariantAnimation()  # no parent
    anim.setDuration(10)
    anim.setStartValue(0)
    anim.setEndValue(1)
    anim.start()
    assert len(motion.orphans()) == 1
    motion.settle(1000)


def test_stuck_opacity_detects_half_faded_widget(qtbot):
    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    child = QtWidgets.QLabel("x", w)
    eff = QtWidgets.QGraphicsOpacityEffect(child)
    eff.setOpacity(0.4)
    child.setGraphicsEffect(eff)
    w.show()
    assert any("opacity=0.40" in s for s in stuck_opacity(w))


def test_warning_policy_matches_cross_thread_messages():
    policy = QtWarningPolicy()
    assert policy.violations([
        "QObject::setParent: Cannot set parent, new parent is in a different thread",
        "QBasicTimer::start: Timers cannot be started from another thread",
        "This plugin does not support propagateSizeHints()",
    ]) == [
        "QObject::setParent: Cannot set parent, new parent is in a different thread",
        "QBasicTimer::start: Timers cannot be started from another thread",
    ]


def test_synthetic_library_is_real_and_tagged(library):
    import mutagen

    assert len(library.audio_paths) == 11
    exts = {p.suffix for p in library.audio_paths}
    assert {".mp3", ".flac"} <= exts
    tagged = mutagen.File(library.find("Magma"), easy=True)
    assert tagged.tags["genre"] == ["rock"]
    assert tagged.info.length > 0.5
