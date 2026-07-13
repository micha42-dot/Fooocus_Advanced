import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.performance_report import compare_images, summarize, workload_key


class TestPerformanceReport(unittest.TestCase):
    def test_summarizes_completed_runs(self):
        records = [
            {'status': 'complete', 'total_seconds': 4.0, 'peak_vram_mb': 1000},
            {'status': 'complete', 'total_seconds': 2.0, 'peak_vram_mb': 800},
            {'status': 'error', 'total_seconds': 1.0, 'peak_vram_mb': 200},
        ]
        summary = summarize(records)

        self.assertEqual(summary['runs'], 2)
        self.assertEqual(summary['median_seconds'], 3.0)
        self.assertEqual(summary['median_peak_vram_mb'], 900.0)

    def test_workload_key_ignores_optimization_configuration(self):
        workload = {
            'prompt_hash': 'abc', 'seed': 1, 'model': 'm', 'refiner': 'None',
            'performance': 'Speed', 'resolution': '1024x1024', 'images': 1,
            'steps': 30, 'sampler': 'euler', 'scheduler': 'karras', 'cfg': 4.0,
            'clip_skip': 2,
        }
        first = {'workload': workload, 'configuration': {'unet_cache': 'off'}}
        second = {'workload': workload, 'configuration': {'unet_cache': 'balanced'}}
        self.assertEqual(workload_key(first), workload_key(second))

    def test_image_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = os.path.join(directory, 'reference.png')
            candidate_path = os.path.join(directory, 'candidate.png')
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(reference_path)
            Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)).save(candidate_path)

            score = compare_images(reference_path, candidate_path)

        self.assertEqual(score['mae'], 1.0)
        self.assertGreater(score['psnr'], 40.0)


if __name__ == '__main__':
    unittest.main()
