import os
import tempfile
import unittest
from unittest import mock

import build_launcher


class TestBuildLauncher(unittest.TestCase):
    def test_creates_no_download_launcher_without_overwriting_existing_launchers(self):
        with tempfile.TemporaryDirectory() as folder:
            existing_launcher = os.path.join(folder, 'run.bat')
            with open(existing_launcher, 'w', encoding='utf-8') as launcher_file:
                launcher_file.write('custom launcher')

            with mock.patch.object(build_launcher, 'is_win32_standalone_build', True), \
                    mock.patch.object(build_launcher, 'win32_root', folder):
                build_launcher.build_launcher()

            with open(existing_launcher, 'r', encoding='utf-8') as launcher_file:
                self.assertEqual(launcher_file.read(), 'custom launcher')

            no_download_launcher = os.path.join(folder, 'run_no_download.bat')
            with open(no_download_launcher, 'r', encoding='utf-8') as launcher_file:
                self.assertIn('--disable-model-download', launcher_file.read())


if __name__ == '__main__':
    unittest.main()
