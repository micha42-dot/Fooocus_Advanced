import unittest

from modules.flags import Performance


class TestPerformanceModes(unittest.TestCase):
    def test_quality_10_ays_defaults(self):
        performance = Performance.QUALITY_10

        self.assertEqual(performance.steps(), 10)
        self.assertEqual(performance.steps_uov(), 10)
        self.assertEqual(performance.sampler_name(), 'dpmpp_2m_sde_gpu')
        self.assertEqual(performance.scheduler_name(), 'align_your_steps')
        self.assertIsNone(performance.lora_filename())
        self.assertFalse(Performance.has_restricted_features(performance))
        self.assertEqual(Performance.by_steps(10), performance)


if __name__ == '__main__':
    unittest.main()
