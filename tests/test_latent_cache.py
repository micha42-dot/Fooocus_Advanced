import unittest

import torch

from modules.latent_cache import LatentCache


class FakeVAE:
    vae_dtype = torch.float16


class TestLatentCache(unittest.TestCase):
    def test_reuses_latent_and_returns_a_copy(self):
        cache = LatentCache(1024 * 1024)
        vae = FakeVAE()
        pixels = torch.arange(48, dtype=torch.float32).reshape(1, 4, 4, 3)
        calls = []

        def encode():
            calls.append(True)
            return torch.ones((1, 4, 2, 2), dtype=torch.float32)

        first = cache.get_or_encode(vae, pixels, 'regular', encode)
        second = cache.get_or_encode(vae, pixels, 'regular', encode)
        second.zero_()
        third = cache.get_or_encode(vae, pixels, 'regular', encode)

        self.assertEqual(1, len(calls))
        self.assertTrue(torch.equal(first, third))
        self.assertFalse(torch.equal(second, third))

    def test_separates_vae_and_encode_variant(self):
        cache = LatentCache(1024 * 1024)
        pixels = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        first_vae = FakeVAE()
        second_vae = FakeVAE()
        calls = []

        def encode():
            calls.append(True)
            return torch.zeros((1, 4, 2, 2), dtype=torch.float32)

        cache.get_or_encode(first_vae, pixels, 'regular', encode)
        cache.get_or_encode(first_vae, pixels, 'tiled-512', encode)
        cache.get_or_encode(second_vae, pixels, 'regular', encode)

        self.assertEqual(3, len(calls))
