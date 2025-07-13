import os
import sys
import types
import tempfile
import unittest

# Provide simple stubs for mutagen so the module imports
mutagen_stub = types.ModuleType('mutagen')
class DummyAudio:
    def __init__(self):
        self.tags = None
        self.pictures = []
def File(*a, **k):
    return DummyAudio()
mutagen_stub.File = File

id3_stub = types.ModuleType('id3')
class DummyID3Error(Exception):
    pass
id3_stub.ID3NoHeaderError = DummyID3Error
mutagen_stub.id3 = id3_stub
sys.modules['mutagen'] = mutagen_stub
sys.modules['mutagen.id3'] = id3_stub

from music_indexer_api import build_dry_run_html

class DryRunTest(unittest.TestCase):
    def test_dry_run_does_not_modify_files(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, 'A'), exist_ok=True)
            os.makedirs(os.path.join(root, 'B'), exist_ok=True)
            open(os.path.join(root, 'A', 'dup.mp3'), 'wb').close()
            open(os.path.join(root, 'B', 'dup.mp3'), 'wb').close()

            before_files = {
                os.path.relpath(os.path.join(dp, f), root)
                for dp, _, fs in os.walk(root)
                for f in fs
            }

            html = os.path.join(root, 'index.html')
            build_dry_run_html(root, html)

            after_files = {
                os.path.relpath(os.path.join(dp, f), root)
                for dp, _, fs in os.walk(root)
                for f in fs
            }

            # Original audio files should remain untouched
            self.assertTrue({'A/dup.mp3', 'B/dup.mp3'}.issubset(after_files))
            self.assertEqual(before_files, {'A/dup.mp3', 'B/dup.mp3'})

if __name__ == '__main__':
    unittest.main()
