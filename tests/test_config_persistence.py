import json
import tempfile
import unittest
from pathlib import Path

from modules import config_persistence, flags


class TestConfigPersistence(unittest.TestCase):
    def test_save_sampling_defaults_preserves_other_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.txt'
            config_path.write_text(
                json.dumps({'unrelated_setting': True}),
                encoding='utf-8',
            )

            config_persistence.save_sampling_defaults(
                config_path,
                'euler',
                'normal',
            )

            saved = json.loads(config_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['default_sampler'], 'euler')
            self.assertEqual(saved['default_scheduler'], 'normal')
            self.assertTrue(saved['unrelated_setting'])

    def test_reset_removes_only_sampling_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.txt'
            config_path.write_text(
                json.dumps({
                    'default_sampler': 'euler',
                    'default_scheduler': 'normal',
                    'unrelated_setting': 42,
                }),
                encoding='utf-8',
            )

            was_reset = config_persistence.reset_sampling_defaults(config_path)

            saved = json.loads(config_path.read_text(encoding='utf-8'))
            self.assertTrue(was_reset)
            self.assertNotIn('default_sampler', saved)
            self.assertNotIn('default_scheduler', saved)
            self.assertEqual(saved['unrelated_setting'], 42)

    def test_invalid_choice_does_not_change_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.txt'
            original = '{"unrelated_setting": true}'
            config_path.write_text(original, encoding='utf-8')

            with self.assertRaises(config_persistence.ConfigPersistenceError):
                config_persistence.save_sampling_defaults(
                    config_path,
                    'not-a-sampler',
                    flags.factory_default_scheduler,
                )

            self.assertEqual(config_path.read_text(encoding='utf-8'), original)

    def test_malformed_config_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.txt'
            original = '{this is not json'
            config_path.write_text(original, encoding='utf-8')

            with self.assertRaises(config_persistence.ConfigPersistenceError):
                config_persistence.save_sampling_defaults(
                    config_path,
                    flags.factory_default_sampler,
                    flags.factory_default_scheduler,
                )

            self.assertEqual(config_path.read_text(encoding='utf-8'), original)


if __name__ == '__main__':
    unittest.main()
