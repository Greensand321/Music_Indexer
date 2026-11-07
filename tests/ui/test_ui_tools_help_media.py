"""Help links (TOOL-12) and the M4A / Opus testers (TOOL-08/09/10)."""
from __future__ import annotations

import base64
import shutil
from pathlib import Path

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from library_factory import TrackSpec, _real_mutagen, build_library

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


# ── Help ──────────────────────────────────────────────────────────────────────

def test_every_help_link_resolves_and_the_maintained_docs_are_linked():
    from gui.workspaces.help import DOC_LINKS, doc_path

    paths = [p for _label, p in DOC_LINKS]
    for rel in paths:
        assert doc_path(rel).is_file(), f"Help links to a missing file: {rel}"
    for required in ("docs/README.md", "docs/overview.md"):
        assert required in paths
    features = sorted(p.relative_to(doc_path(".").resolve()).as_posix()
                      for p in doc_path("docs/features").resolve().glob("*.md"))
    assert set(features) <= set(paths), "every docs/features page should be linked"


def test_help_buttons_open_file_uris(make_workspace, dialogs, monkeypatch):
    import gui.workspaces.help as help_mod

    opened: list[str] = []
    monkeypatch.setattr(help_mod.webbrowser, "open", lambda uri, *a, **k: opened.append(uri))
    ws = make_workspace("help")
    for btn in ws._doc_buttons:
        btn.click()
    assert not dialogs.calls_of("information"), "no 'File not found' box"
    assert len(opened) == len(help_mod.DOC_LINKS)
    for uri in opened:
        assert uri.startswith("file://")
        assert Path(QtCore.QUrl(uri).toLocalFile()).is_file()


# ── Media testers ─────────────────────────────────────────────────────────────

def _png_bytes(color: str = "#ff0000") -> bytes:
    img = QtGui.QImage(8, 8, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(color))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


@pytest.fixture
def media(tmp_path):
    m = _real_mutagen()
    lib = build_library(tmp_path / "Media", [
        TrackSpec("Cobalt Choir", "Vespers", "Hymns", 3, "2015", "Classical",
                  albumartist="Cobalt Choir & Friends", ext=".m4a", relpath="full.m4a"),
        TrackSpec("Delta Static", "Low Tide", "Coastal", 1, "2023", "Ambient",
                  ext=".opus", relpath="full.opus"),
        # Like a Compression output: only some keys present.
        TrackSpec(None, "Sparse", None, ext=".opus", relpath="sparse.opus"),
    ])
    root = lib.root
    png = _png_bytes()
    mp4 = m["mp4"].MP4(str(root / "full.m4a"))
    mp4["covr"] = [m["mp4"].MP4Cover(png, imageformat=m["mp4"].MP4Cover.FORMAT_PNG)]
    mp4.save()
    pic = m["flac"].Picture()
    pic.type = 3
    pic.mime = "image/png"
    pic.data = png
    opus = m["oggopus"].OggOpus(str(root / "full.opus"))
    opus["metadata_block_picture"] = [base64.b64encode(pic.write()).decode("ascii")]
    opus.save()
    (root / "garbage.opus").write_bytes(b"not audio at all")
    return root


@needs_ffmpeg
def test_m4a_tester_shows_the_right_fields(media):
    from gui.dialogs.media_tester_dialog import describe_media

    lines, art, error = describe_media(str(media / "full.m4a"))
    assert error is None
    text = "\n".join(lines)
    assert "Title: Vespers" in text
    assert "Artist: Cobalt Choir" in text
    assert "Album: Hymns" in text
    assert "Album artist: Cobalt Choir & Friends" in text   # was showing the album
    assert "Track: 3" in text
    assert "Genre: Classical" in text
    assert art and art.startswith(b"\x89PNG")


@needs_ffmpeg
def test_opus_tester_reads_tags_and_cover(media):
    from gui.dialogs.media_tester_dialog import describe_media

    lines, art, error = describe_media(str(media / "full.opus"))
    assert error is None
    assert "Title: Low Tide" in lines and "Artist: Delta Static" in lines
    assert art and art.startswith(b"\x89PNG"), "Opus METADATA_BLOCK_PICTURE must be found"

    lines, art, error = describe_media(str(media / "sparse.opus"))
    assert lines == ["Title: Sparse"] and art is None and error is None


@needs_ffmpeg
def test_tester_dialog_shows_art_then_clears_it_for_a_bad_file(media, qtbot):
    from gui.dialogs.media_tester_dialog import MediaTesterDialog

    dlg = MediaTesterDialog(codec="opus")
    qtbot.addWidget(dlg)
    dlg._load_file(str(media / "full.opus"))
    assert dlg._art_lbl.pixmap() is not None and not dlg._art_lbl.pixmap().isNull()
    assert "Title: Low Tide" in dlg._meta_text.toPlainText()

    dlg._load_file(str(media / "garbage.opus"))
    assert dlg._art_lbl.pixmap() is None or dlg._art_lbl.pixmap().isNull(), "stale art must be cleared"
    assert "Low Tide" not in dlg._meta_text.toPlainText()
