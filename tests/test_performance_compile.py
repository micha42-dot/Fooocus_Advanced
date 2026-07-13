import unittest
from unittest.mock import patch

import torch

from modules.performance import ResolutionCompiledForward, compile_profile_key


class TestPerformanceCompile(unittest.TestCase):
    def test_profile_key_tracks_resolution_and_context(self):
        key = compile_profile_key(
            (torch.zeros((2, 4, 128, 96)),),
            {'context': torch.zeros((2, 77, 2048))},
        )

        self.assertEqual(key[0], (2, 4, 128, 96))
        self.assertEqual(key[3], (2, 77, 2048))

    def test_resolution_profiles_are_bounded(self):
        original = lambda x, **kwargs: x + 1
        compiled = ResolutionCompiledForward(original, mode=None, max_profiles=1)

        with patch('modules.performance.torch.compile', side_effect=lambda function, **kwargs: function):
            first = compiled(torch.zeros((1, 4, 8, 8)))
            second = compiled(torch.zeros((1, 4, 16, 16)))

        self.assertTrue(torch.equal(first, torch.ones_like(first)))
        self.assertTrue(torch.equal(second, torch.ones_like(second)))
        self.assertEqual(len(compiled.profiles), 1)
        self.assertEqual(len(compiled.skipped_profiles), 1)


if __name__ == '__main__':
    unittest.main()
