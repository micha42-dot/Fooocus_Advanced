from pathlib import Path
import tempfile
import unittest

from tools.build_release import write_assets


class TestReleaseAssets(unittest.TestCase):
    def test_conflict_preserves_existing_release_and_allows_retry(self):
        assets = {'source.zip': b'source', 'windows.zip': b'windows', 'RELEASE-NOTES.md': b'notes',
                  'BUILD-INFO.json': b'new info'}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            info = folder / 'BUILD-INFO.json'
            info.write_bytes(b'previous info')
            notes = folder / 'RELEASE-NOTES.md'
            notes.write_bytes(b'notes')
            with self.assertRaises(FileExistsError):
                write_assets(folder, assets)
            self.assertEqual({info, notes}, set(folder.iterdir()))
            self.assertEqual(info.read_bytes(), b'previous info')
            info.unlink()
            write_assets(folder, assets)
            self.assertEqual(assets, {file.name: file.read_bytes() for file in folder.iterdir()})
