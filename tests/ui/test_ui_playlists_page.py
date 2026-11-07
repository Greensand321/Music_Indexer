"""Playlist Generator workspace: worker lifecycle and preview-first writes.

Covers PL-01 (double click), PL-02 (close mid-run), PL-10/PL-11 (repair and
overwrites are previewed, confirmed and backed up), PL-12 (reserved folders),
PL-13 (Auto-DJ seed), PL-14 (range fields and Prefer Opus are read) and
PL-32 (Cancel, neutral cancelled status).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

import playlist_engine
from playlists_helpers import make_library
from ui_harness import process_events


def _ws(make_workspace, tmp_path):
    lib = make_library(tmp_path / "lib")
    ws = make_workspace("playlists", library_path=lib)
    return ws, lib


def _wait_idle(ws, qtbot, timeout=5000):
    qtbot.waitUntil(lambda: not ws.is_busy(), timeout=timeout)
    process_events(20)


@pytest.fixture
def slow_tempo(monkeypatch):
    """bucket_by_tempo_energy that blocks until released (or cancelled)."""
    gate = threading.Event()
    calls = []

    def fake(tracks, root, log_callback=None, cancel_event=None, **kw):
        calls.append({"tracks": list(tracks), **kw})
        while not gate.wait(0.02):
            if cancel_event is not None and cancel_event.is_set():
                return {"buckets": {}, "playlist_paths": {}, "cancelled": True}
        out = os.path.join(root, "Playlists", "slow_low.m3u")
        return {"buckets": {("slow", "low"): list(tracks)}, "playlist_paths": {("slow", "low"): out}}

    monkeypatch.setattr(playlist_engine, "bucket_by_tempo_energy", fake)
    return gate, calls


# ── PL-01 / PL-32 ────────────────────────────────────────────────────────


def test_second_click_while_running_starts_no_second_job(make_workspace, tmp_path, qtbot, dialogs, threads, slow_tempo):
    gate, calls = slow_tempo
    ws, lib = _ws(make_workspace, tmp_path)
    ws._on_run("tempo")
    assert ws.is_busy()
    assert all(not b.isEnabled() for b in ws._run_buttons)
    ws._on_run("tempo")      # a queued second click
    ws._on_run("folder")
    process_events(50)
    assert threads.started.count("PlaylistWorker") == 1

    dialogs.answer_no()
    gate.set()
    _wait_idle(ws, qtbot)
    assert all(b.isEnabled() for b in ws._run_buttons)
    assert dialogs.calls_of("question"), "the plan must be confirmed"
    assert not (lib / "Playlists" / "slow_low.m3u").exists(), "declined plan was written"


def test_cancel_stops_the_job_with_a_neutral_status(make_workspace, tmp_path, qtbot, threads, slow_tempo):
    _gate, _calls = slow_tempo
    ws, lib = _ws(make_workspace, tmp_path)
    statuses = []
    ws.status_changed.connect(lambda text, colour: statuses.append(text))
    ws._on_run("tempo")
    assert ws._cancel_btn.isVisible()
    ws._cancel_btn.click()
    _wait_idle(ws, qtbot)
    assert statuses[-1] == "Cancelled"
    assert not (lib / "Playlists" / "slow_low.m3u").exists()
    assert not ws._cancel_btn.isVisible()


def test_tempo_ranges_reach_the_backend(make_workspace, tmp_path, qtbot, dialogs, slow_tempo):
    gate, calls = slow_tempo
    gate.set()
    ws, lib = _ws(make_workspace, tmp_path)
    ws._tempo_min.setValue(90)
    ws._tempo_max.setValue(140)
    ws._energy_min.setValue(0.1)
    ws._energy_max.setValue(0.6)
    dialogs.answer_no()
    ws._on_run("tempo")
    _wait_idle(ws, qtbot)
    assert calls[0]["tempo_range"] == (90, 140)
    assert calls[0]["energy_range"] == pytest.approx((0.1, 0.6))
    # Reserved folders are not analysed.
    assert not any("Quarantine" in t or "Docs" in t for t in calls[0]["tracks"])


# ── PL-02: closing the app mid-run ───────────────────────────────────────


def test_closing_the_window_mid_run_cancels_cleanly(main_window, tmp_path, qtbot, dialogs, slow_tempo):
    lib = make_library(tmp_path / "lib")
    main_window.set_library(str(lib))
    ws = main_window._workspaces["playlists"]
    ws._on_run("tempo")
    assert ws.is_busy()
    dialogs.answer_yes()  # "Stop them and quit?"
    main_window.close()
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=5000)
    assert not (lib / "Playlists" / "slow_low.m3u").exists()


# ── PL-12 / PL-11: folder playlists ──────────────────────────────────────


def test_folder_playlists_skip_reserved_and_back_up_overwrites(make_workspace, tmp_path, qtbot, dialogs):
    ws, lib = _ws(make_workspace, tmp_path)
    existing = lib / "Playlists" / "Artist A_Album 1.m3u"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("user edits\n", encoding="utf-8")

    dialogs.answer_yes()
    ws._on_run("folder")
    _wait_idle(ws, qtbot)
    _wait_idle(ws, qtbot)  # write job

    question = dialogs.calls_of("question")[0]
    assert "replace: Playlists" in question.text and "new: Playlists" in question.text
    names = sorted(p.name for p in (lib / "Playlists").iterdir())
    assert "Artist A_Album 1.m3u" in names and "Artist A_Album 2.m3u" in names
    assert not any(n.startswith(("Quarantine", "Trash", "Docs", "Not Sorted")) for n in names)
    backups = [n for n in names if n.endswith(".bak")]
    assert len(backups) == 1
    assert (lib / "Playlists" / backups[0]).read_text(encoding="utf-8") == "user edits\n"
    assert "user edits" not in existing.read_text(encoding="utf-8")
    assert (lib / "Docs" / "playlist_writes.log").exists()


def test_declining_writes_nothing(make_workspace, tmp_path, qtbot, dialogs):
    ws, lib = _ws(make_workspace, tmp_path)
    dialogs.answer_no()
    ws._on_run("folder")
    _wait_idle(ws, qtbot)
    assert not list((lib / "Playlists").glob("*.m3u")) if (lib / "Playlists").exists() else True


# ── PL-10 / PL-11: repair ────────────────────────────────────────────────


def _broken_playlist(lib: Path) -> Path:
    pl = lib / "Playlists" / "mine.m3u"
    pl.parent.mkdir(parents=True, exist_ok=True)
    pl.write_text("../Artist A/Old Album/01 a.mp3\n../gone/zzz qqq.mp3\n", encoding="utf-8")
    return pl


def test_repair_of_listed_playlist_is_previewed_then_applied(make_workspace, tmp_path, qtbot, dialogs):
    ws, lib = _ws(make_workspace, tmp_path)
    pl = _broken_playlist(lib)
    ws._repair_list.addItem(str(pl))
    dialogs.answer_yes()
    ws._on_run("repair")
    _wait_idle(ws, qtbot)
    _wait_idle(ws, qtbot)

    q = dialogs.calls_of("question")[0]
    assert "1 entry will be re-pointed" in q.text and "1 still missing" in q.text
    lines = pl.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "../Artist A/Album 1/01 a.mp3"   # the repaired entry keeps its "/" style
    assert lines[1] == "../gone/zzz qqq.mp3"
    assert list(pl.parent.glob("mine.m3u.*.bak"))


def test_repair_declined_leaves_playlist_untouched(make_workspace, tmp_path, qtbot, dialogs):
    ws, lib = _ws(make_workspace, tmp_path)
    pl = _broken_playlist(lib)
    before = pl.read_text(encoding="utf-8")
    dialogs.answer_no()
    ws._on_run("repair")  # empty list → every playlist under Playlists/
    _wait_idle(ws, qtbot)
    assert pl.read_text(encoding="utf-8") == before
    assert not list(pl.parent.glob("*.bak"))


def test_repair_with_nothing_fixable_asks_nothing(make_workspace, tmp_path, qtbot, dialogs):
    ws, lib = _ws(make_workspace, tmp_path)
    pl = lib / "Playlists" / "ok.m3u"
    pl.parent.mkdir(parents=True, exist_ok=True)
    pl.write_text("../Artist A/Album 1/01 a.mp3\n", encoding="utf-8")
    ws._on_run("repair")
    _wait_idle(ws, qtbot)
    assert not dialogs.calls_of("question")


def test_prefer_opus_checkbox_is_read(make_workspace, tmp_path, qtbot, dialogs):
    ws, lib = _ws(make_workspace, tmp_path)
    (lib / "Artist A/Album 1/03 s.opus").write_bytes(b"\0")
    pl = lib / "Playlists" / "codec.m3u"
    pl.parent.mkdir(parents=True, exist_ok=True)
    pl.write_text("../Artist A/Album 1/03 s.flac\n", encoding="utf-8")
    ws._prefer_opus_cb.setChecked(True)
    dialogs.answer_yes()
    ws._on_run("repair")
    _wait_idle(ws, qtbot)
    _wait_idle(ws, qtbot)
    assert pl.read_text(encoding="utf-8").strip().endswith("03 s.opus")


# ── PL-13: Auto-DJ seed ──────────────────────────────────────────────────


def test_autodj_requires_a_seed(make_workspace, tmp_path, threads, dialogs):
    ws, _lib = _ws(make_workspace, tmp_path)
    ws._on_run("autodj")
    assert dialogs.calls_of("warning")
    assert "PlaylistWorker" not in threads.started


def test_autodj_rejects_a_seed_outside_the_library(make_workspace, tmp_path, qtbot, dialogs, monkeypatch):
    monkeypatch.setattr(playlist_engine, "autodj_playlist", lambda *a, **k: pytest.fail("ran"))
    ws, lib = _ws(make_workspace, tmp_path)
    ws._autodj_seed.setText(str(lib / "Docs" / "sample.wav"))  # reserved folder
    logs = []
    ws.log_message.connect(lambda m, lvl: logs.append((lvl, m)))
    ws._on_run("autodj")
    _wait_idle(ws, qtbot)
    assert any(lvl == "error" and "Seed track is not in the library" in m for lvl, m in logs)
    assert not dialogs.calls_of("question")


def test_autodj_uses_the_chosen_seed_and_previews_overwrite(make_workspace, tmp_path, qtbot, dialogs, monkeypatch):
    seen = {}

    def fake_autodj(seed, tracks, n=20, **kw):
        seen["seed"], seen["tracks"] = seed, list(tracks)
        return [seed] + [t for t in tracks if t != seed][: n - 1]

    monkeypatch.setattr(playlist_engine, "autodj_playlist", fake_autodj)
    ws, lib = _ws(make_workspace, tmp_path)
    out = lib / "Playlists" / "autodj.m3u"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("previous mix\n", encoding="utf-8")
    seed = lib / "Artist B" / "Album X" / "01 d.mp3"
    ws._autodj_seed.setText(str(seed))
    dialogs.answer_yes()
    ws._on_run("autodj")
    _wait_idle(ws, qtbot)
    _wait_idle(ws, qtbot)

    assert seen["seed"] == str(seed)
    assert not any("Docs" in t or "Quarantine" in t for t in seen["tracks"])
    assert "replace: Playlists" in dialogs.calls_of("question")[0].text
    assert out.read_text(encoding="utf-8").splitlines()[0] == os.path.join("..", "Artist B", "Album X", "01 d.mp3")
    assert list(out.parent.glob("autodj.m3u.*.bak"))


# ── Library change mid-run ───────────────────────────────────────────────


def test_library_change_mid_run_cancels_and_ignores_stale_result(make_workspace, tmp_path, qtbot, dialogs, threads, slow_tempo):
    gate, _calls = slow_tempo
    ws, lib = _ws(make_workspace, tmp_path)
    ws._on_run("tempo")
    other = make_library(tmp_path / "other")
    ws.set_library_path(str(other))
    assert not ws.is_busy()
    gate.set()
    assert threads.wait_all(5000) == []
    process_events(50)
    assert not dialogs.calls_of("question"), "a plan for the old library was offered"
