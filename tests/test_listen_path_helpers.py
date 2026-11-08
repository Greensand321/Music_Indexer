"""``utils.path_helpers`` under Windows semantics: network shares (UNC).

``ensure_long_path`` used to prefix a UNC path like any other, giving
``\\\\?\\\\\\server\\share\\a.flac`` — not a path at all — so on Windows every
open/stat of a library on a NAS share failed. The extended-length form of a
share is ``\\\\?\\UNC\\server\\share\\a.flac``, and ``strip_ext_prefix`` must
turn it back into ``\\\\server\\share\\a.flac``.

Runs on any OS by swapping ``os.path`` for ``ntpath`` (as
``tests/test_windows_paths.py`` does).
"""
from __future__ import annotations

import contextlib
import ntpath
import os

import pytest

from utils import path_helpers


@pytest.fixture
def windows(monkeypatch):
    @contextlib.contextmanager
    def ctx():
        with monkeypatch.context() as m:
            m.setattr(os, "path", ntpath)
            m.setattr(os, "sep", "\\")
            m.setattr(os, "altsep", "/")
            m.setattr(os, "name", "nt")
            yield

    return ctx


UNC = "\\\\nas\\music\\Artist\\a.flac"
UNC_LONG = "\\\\?\\UNC\\nas\\music\\Artist\\a.flac"


def test_unc_path_gets_the_unc_long_form(windows):
    with windows():
        assert path_helpers.ensure_long_path(UNC) == UNC_LONG
        # Forward slashes and dot segments are normalised first.
        assert path_helpers.ensure_long_path("//nas/music/Other/../Artist/a.flac") == UNC_LONG
        # Idempotent.
        assert path_helpers.ensure_long_path(UNC_LONG) == UNC_LONG


def test_unc_long_form_strips_back_to_the_share_path(windows):
    with windows():
        assert path_helpers.strip_ext_prefix(UNC_LONG) == UNC
        assert path_helpers.strip_ext_prefix(path_helpers.ensure_long_path(UNC)) == UNC
        # The UNC marker is case-insensitive on Windows.
        assert path_helpers.strip_ext_prefix("\\\\?\\unc\\nas\\music\\a.flac") == "\\\\nas\\music\\a.flac"


def test_drive_paths_keep_their_long_form(windows):
    with windows():
        assert path_helpers.ensure_long_path("C:/Music/a.flac") == "\\\\?\\C:\\Music\\a.flac"
        assert path_helpers.strip_ext_prefix("\\\\?\\C:\\Music\\a.flac") == "C:\\Music\\a.flac"
        # The device namespace is left alone.
        assert path_helpers.ensure_long_path("\\\\.\\pipe\\x") == "\\\\.\\pipe\\x"


def test_playlist_entry_and_relpath_across_share_and_drive(windows):
    with windows():
        # A share and a drive have no relative path between them.
        assert path_helpers.safe_relpath(UNC, "C:\\Playlists") == UNC
        assert path_helpers.playlist_entry(UNC, "C:\\Playlists", like="../x/a.flac") == "//nas/music/Artist/a.flac"
        # Within one share the entry stays relative.
        assert path_helpers.playlist_entry(
            UNC, "\\\\nas\\music\\Playlists", like="..\\Old\\a.flac"
        ) == "..\\Artist\\a.flac"


def test_off_windows_nothing_changes():
    if os.name == "nt":
        pytest.skip("POSIX semantics")
    assert path_helpers.ensure_long_path("/music/a.flac") == "/music/a.flac"
    assert path_helpers.strip_ext_prefix(UNC_LONG) == UNC_LONG
