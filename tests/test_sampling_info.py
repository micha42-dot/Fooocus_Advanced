import unittest

from modules import flags


class TestSamplingInfo(unittest.TestCase):
    def test_every_sampler_has_a_description_in_display_order(self):
        self.assertEqual(
            flags.sampler_list,
            list(flags.sampler_descriptions),
        )
        self.assertTrue(all(
            description.strip()
            for description in flags.sampler_descriptions.values()
        ))

    def test_every_scheduler_has_a_description_in_display_order(self):
        self.assertEqual(
            flags.scheduler_list,
            list(flags.scheduler_descriptions),
        )
        self.assertTrue(all(
            description.strip()
            for description in flags.scheduler_descriptions.values()
        ))


if __name__ == '__main__':
    unittest.main()
