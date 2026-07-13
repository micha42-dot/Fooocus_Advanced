import unittest

import torch

from modules.unet_cache import DeepCache


class TestDeepCache(unittest.TestCase):
    @staticmethod
    def begin(cache, timestep, control=None, patches=None, slots=None):
        return cache.begin(
            x=torch.ones((2, 4, 128, 128)),
            timesteps=torch.tensor([timestep]),
            context=torch.ones((2, 77, 2048)),
            y=torch.ones((2, 2816)),
            control=control,
            transformer_options={
                'cond_or_uncond': [0, 1],
                'fooocus_conditioning_slots': slots or [(0, 0), (1, 0)],
                'fooocus_full_shape': [1, 4, 128, 128],
                'patches': patches or {},
            },
            input_blocks=9,
            output_blocks=9,
        )

    def test_balanced_profile_reuses_deep_feature(self):
        cache = DeepCache()
        cache.configure('balanced', steps=30)

        first = self.begin(cache, 700.0)
        self.assertFalse(first.reuse)
        cache.store(first, torch.tensor([700.0]), torch.ones((2, 1280, 32, 32)), 12)

        second = self.begin(cache, 650.0)
        self.assertTrue(second.reuse)
        self.assertEqual(second.output_start, 3)
        self.assertEqual(second.input_count, 6)
        self.assertEqual(second.transformer_index, 12)
        self.assertEqual(cache.stats()['hits'], 1)

    def test_same_timestep_refreshes_feature(self):
        cache = DeepCache()
        cache.configure('balanced', steps=30)
        first = self.begin(cache, 700.0)
        cache.store(first, torch.tensor([700.0]), torch.ones((2, 1280, 32, 32)), 4)

        second = self.begin(cache, 700.0)
        self.assertFalse(second.reuse)
        self.assertEqual(cache.stats()['hits'], 0)
        self.assertEqual(cache.stats()['misses'], 2)

    def test_different_conditioning_slots_do_not_share_features(self):
        cache = DeepCache()
        cache.configure('balanced', steps=30)
        first = self.begin(cache, 700.0)
        cache.store(first, torch.tensor([700.0]), torch.ones((2, 1280, 32, 32)), 4)

        second = self.begin(cache, 650.0, slots=[(0, 1), (1, 0)])
        self.assertFalse(second.reuse)
        self.assertEqual(cache.stats()['hits'], 0)

    def test_controlnet_and_transformer_patches_bypass_cache(self):
        cache = DeepCache()
        cache.configure('aggressive', steps=30)

        self.assertIsNone(self.begin(cache, 700.0, control=object()))
        self.assertIsNone(self.begin(cache, 650.0, patches={'output_block_patch': [object()]}))
        self.assertEqual(cache.stats()['bypasses'], 2)

    def test_short_and_compiled_runs_stay_disabled(self):
        cache = DeepCache()
        cache.configure('balanced', steps=8)
        self.assertFalse(cache.enabled)
        cache.configure('balanced', steps=30, compiled=True)
        self.assertFalse(cache.enabled)


if __name__ == '__main__':
    unittest.main()
