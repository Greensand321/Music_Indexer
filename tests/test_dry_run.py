import os
import tempfile
import unittest

from _helpers.module_isolation import isolated_modules, make_mutagen_stub


class DryRunTest(unittest.TestCase):
    def test_dry_run_does_not_modify_files(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, 'A'), exist_ok=True)
            os.makedirs(os.path.join(root, 'B'), exist_ok=True)
            open(os.path.join(root, 'A', 'dup.mp3'), 'wb').close()
            open(os.path.join(root, 'B', 'dup.mp3'), 'wb').close()

            # "/"-joined relative paths, so the expectations below hold on
            # Windows too.
            before_files = {
                os.path.relpath(os.path.join(dp, f), root).replace(os.sep, '/')
                for dp, _, fs in os.walk(root)
                for f in fs
            }

            html = os.path.join(root, 'index.html')
            # Stub mutagen only for this call; nothing leaks into later tests.
            with isolated_modules(make_mutagen_stub(), fresh=["music_indexer_api"]) as mods:
                mods["music_indexer_api"].build_dry_run_html(root, html)

            after_files = {
                os.path.relpath(os.path.join(dp, f), root).replace(os.sep, '/')
                for dp, _, fs in os.walk(root)
                for f in fs
            }

            # Original audio files should remain untouched
            self.assertTrue({'A/dup.mp3', 'B/dup.mp3'}.issubset(after_files))
            self.assertEqual(before_files, {'A/dup.mp3', 'B/dup.mp3'})

if __name__ == '__main__':
    unittest.main()
