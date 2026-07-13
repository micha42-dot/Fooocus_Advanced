import unittest

import torch

from modules.clip_cache import ClipConditionCache


class DummyClip:
    def __init__(self, layer_idx=-2):
        self.layer_idx = layer_idx


class TestClipConditionCache(unittest.TestCase):
    def test_reuses_conditions_and_returns_clones(self):
        cache = ClipConditionCache(1024)
        clip = DummyClip()
        calls = 0

        def encode():
            nonlocal calls
            calls += 1
            return torch.ones((1, 2)), torch.ones((1, 1))

        first = cache.get_or_encode(clip, 'prompt', encode)
        second = cache.get_or_encode(clip, 'prompt', encode)
        second[0][0, 0] = 9
        third = cache.get_or_encode(clip, 'prompt', encode)

        self.assertEqual(calls, 1)
        self.assertEqual(first[0][0, 0].item(), 1)
        self.assertEqual(third[0][0, 0].item(), 1)
        self.assertEqual(cache.stats()['hits'], 2)

    def test_clip_layer_is_part_of_cache_key(self):
        cache = ClipConditionCache(1024)
        clip = DummyClip(-2)
        calls = 0

        def encode():
            nonlocal calls
            calls += 1
            return torch.tensor([calls], dtype=torch.float32)

        cache.get_or_encode(clip, 'prompt', encode)
        clip.layer_idx = -1
        cache.get_or_encode(clip, 'prompt', encode)

        self.assertEqual(calls, 2)

    def test_evicts_least_recently_used_entries(self):
        cache = ClipConditionCache(8)
        clip = DummyClip()

        cache.get_or_encode(clip, 'a', lambda: torch.ones(2))
        cache.get_or_encode(clip, 'b', lambda: torch.ones(2))

        self.assertEqual(cache.stats()['entries'], 1)
        self.assertLessEqual(cache.stats()['size_bytes'], 8)


if __name__ == '__main__':
    unittest.main()
