import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from modules import external_api
from modules import private_logger


class TestExternalAPILog(unittest.TestCase):
    def test_direct_api_image_is_written_to_daily_html_log(self):
        metadata = [
            ('Prompt', 'prompt', 'a glass city at sunrise'),
            ('Negative Prompt', 'negative_prompt', 'fog'),
        ]
        metadata.extend(external_api.get_generation_metadata(
            external_api.PROVIDER_FAL,
            'fal:nano-banana-2'))
        task = {
            'positive': ['a glass city at sunrise'],
            'negative': ['fog'],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            day_dir = os.path.join(temp_dir, '2026-07-27')
            image_path = os.path.join(day_dir, 'api-image.png')
            with patch.object(
                    private_logger.args_manager.args, 'disable_image_log', False), \
                    patch.object(private_logger.modules.config, 'path_outputs', temp_dir), \
                    patch.object(
                        private_logger,
                        'generate_temp_filename',
                        return_value=('2026-07-27', image_path, 'api-image.png')):
                saved_path = private_logger.log(
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    metadata,
                    output_format='png',
                    task=task)

            with open(os.path.join(day_dir, 'log.html'), encoding='utf-8') as log_file:
                html = log_file.read()

        self.assertEqual(saved_path, image_path)
        self.assertIn('a glass city at sunrise', html)
        self.assertIn('Nano Banana 2', html)
        self.assertIn('Local Refinement', html)
        self.assertIn('Disabled', html)


if __name__ == '__main__':
    unittest.main()
