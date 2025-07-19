import os
from playlist_generator import generate_playlists


def test_generate_playlists_appends(tmp_path):
    root = tmp_path
    moves1 = {
        str(root / 'A' / 't1.mp3'): str(root / 'Music' / 'a' / 't1.mp3'),
    }
    generate_playlists(moves1, str(root), overwrite=True, log_callback=lambda m: None)

    playlist_dir = root / 'Playlists'
    playlist = playlist_dir / 'A.m3u'
    assert playlist.exists()
    first_line = os.path.relpath(str(root / 'Music' / 'a' / 't1.mp3'), str(playlist_dir))
    assert playlist.read_text().splitlines() == [first_line]

    moves2 = {
        str(root / 'A' / 't2.mp3'): str(root / 'Music' / 'a' / 't2.mp3'),
    }
    generate_playlists(moves2, str(root), overwrite=False, log_callback=lambda m: None)

    files = sorted(os.listdir(playlist_dir))
    assert files == ['A.m3u']
    expected = sorted([
        os.path.relpath(str(root / 'Music' / 'a' / 't1.mp3'), str(playlist_dir)),
        os.path.relpath(str(root / 'Music' / 'a' / 't2.mp3'), str(playlist_dir)),
    ])
    assert playlist.read_text().splitlines() == expected

    import hashlib
    hashed = f"A_{hashlib.md5('A'.encode('utf-8')).hexdigest()[:6]}.m3u"
    assert not (playlist_dir / hashed).exists()
