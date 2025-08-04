import os
from controllers.import_controller import build_import_preview_html


def test_import_preview_html_only_new(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()

    # existing library file that should not appear in preview
    existing = root / "Music" / "Old" / "old.mp3"
    existing.parent.mkdir(parents=True)
    existing.write_text("dummy")

    dest1 = root / "Music" / "Artist" / "song1.mp3"
    dest2 = root / "Music" / "Artist" / "song2.mp3"
    moves = {"a": str(dest1), "b": str(dest2)}
    html = tmp_path / "preview.html"

    build_import_preview_html(str(root), moves, str(html))
    content = html.read_text()

    assert "song1.mp3" in content
    assert "song2.mp3" in content
    assert "old.mp3" not in content
