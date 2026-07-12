import unittest

import numpy as np

from modules import tiled_upscale


class TestTiledUpscale(unittest.TestCase):
    def test_detects_tiled_sdxl_detail_upscale(self):
        method = 'Upscale (Tiled SDXL Detail 2x)'

        self.assertTrue(tiled_upscale.is_tiled_detail_upscale(method))
        self.assertEqual(2.0, tiled_upscale.get_upscale_factor(method))

    def test_tile_positions_cover_end(self):
        positions = tiled_upscale.get_tile_positions(2500, tile_size=1024, overlap=192)

        self.assertEqual(0, positions[0])
        self.assertEqual(1476, positions[-1])
        self.assertEqual(sorted(set(positions)), positions)

    def test_tile_count_uses_both_axes(self):
        count = tiled_upscale.get_tiled_detail_tile_count(1800, 2500, tile_size=1024, overlap=192)

        self.assertEqual(6, count)

    def test_blend_mask_keeps_outer_edges_opaque(self):
        mask = tiled_upscale.make_blend_mask(
            tile_height=1024,
            tile_width=1024,
            y=0,
            x=0,
            image_height=1600,
            image_width=1600,
            overlap=192,
        )

        self.assertEqual(np.float32, mask.dtype)
        self.assertEqual(1.0, mask[0, 0])
        self.assertLess(mask[-1, -1], 0.01)
