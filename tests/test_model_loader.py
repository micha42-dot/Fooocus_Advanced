import os
import tempfile
import unittest

from modules.model_loader import load_file_from_url, set_model_downloads_enabled


class TestModelLoader(unittest.TestCase):
    def tearDown(self):
        set_model_downloads_enabled(True)

    def test_disabled_downloads_still_allow_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            expected = os.path.join(folder, 'model.bin')
            with open(expected, 'wb') as model_file:
                model_file.write(b'model')
            set_model_downloads_enabled(False)

            actual = load_file_from_url('https://example.invalid/model.bin', model_dir=folder)

            self.assertEqual(actual, os.path.abspath(expected))

    def test_disabled_downloads_fail_before_network_access(self):
        with tempfile.TemporaryDirectory() as folder:
            set_model_downloads_enabled(False)

            with self.assertRaisesRegex(FileNotFoundError, 'Automatic model downloads are disabled'):
                load_file_from_url('https://example.invalid/model.bin', model_dir=folder)


if __name__ == '__main__':
    unittest.main()
