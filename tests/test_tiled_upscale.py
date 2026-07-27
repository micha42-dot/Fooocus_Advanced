import unittest

import numpy as np

from modules import tiled_upscale
from modules import flags


class TestTiledUpscale(unittest.TestCase):
    def test_detects_tiled_sdxl_detail_upscale(self):
        method = 'Upscale (Tiled SDXL Detail 2x)'

        self.assertTrue(tiled_upscale.is_tiled_detail_upscale(method))
        self.assertEqual(2.0, tiled_upscale.get_upscale_factor(method))
        self.assertIn('2.0', flags.upscale_tiled_detail)

    def test_tile_positions_cover_end(self):
        positions = tiled_upscale.get_tile_positions(2500, tile_size=1024, overlap=192)

        self.assertEqual(0, positions[0])
        self.assertEqual(1476, positions[-1])
        self.assertEqual(sorted(set(positions)), positions)

    def test_tile_positions_distribute_overlap_evenly(self):
        positions = tiled_upscale.get_tile_positions(2500, tile_size=1024, overlap=192)
        gaps = np.diff(positions)

        self.assertEqual([0, 738, 1476], positions)
        self.assertLessEqual(int(gaps.max() - gaps.min()), 1)

    def test_tile_plans_cover_arbitrary_lengths_without_gaps(self):
        for length in range(1025, 10000, 137):
            positions = tiled_upscale.get_tile_positions(length, tile_size=1024, overlap=256)
            self.assertEqual(0, positions[0])
            self.assertEqual(length - 1024, positions[-1])
            self.assertTrue(np.all(np.diff(positions) <= 1024))

        with self.assertRaises(ValueError):
            tiled_upscale.get_tile_positions(2048, tile_size=1024, overlap=1024)

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

    def test_neighbor_masks_are_complementary_across_real_overlap(self):
        positions = tiled_upscale.get_tile_positions(2500, tile_size=1024, overlap=192)
        left_x, right_x = positions[:2]
        overlap_width = left_x + 1024 - right_x
        left = tiled_upscale.make_blend_mask(
            1024, 1024, 0, left_x, 1024, 2500, 192,
            y_positions=[0], x_positions=positions)
        right = tiled_upscale.make_blend_mask(
            1024, 1024, 0, right_x, 1024, 2500, 192,
            y_positions=[0], x_positions=positions)

        combined = left[0, -overlap_width:] + right[0, :overlap_width]
        self.assertTrue(np.allclose(combined, 1.0, atol=2e-6))

    def test_small_tiles_use_reflection_padding(self):
        image = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
        padded, height, width = tiled_upscale.get_padded_tile(image, 0, 0, tile_size=4)

        self.assertEqual((2, 2), (height, width))
        self.assertEqual((4, 4, 3), padded.shape)
        self.assertTrue(np.array_equal(padded[0, 2], image[0, 0]))

    def test_coordinate_seed_is_stable_and_position_specific(self):
        first = tiled_upscale.get_tile_seed(1234, 0, 738, 2 ** 63 - 1)
        repeated = tiled_upscale.get_tile_seed(1234, 0, 738, 2 ** 63 - 1)
        neighbor = tiled_upscale.get_tile_seed(1234, 0, 1476, 2 ** 63 - 1)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, neighbor)

    def test_low_frequency_anchor_reduces_color_drift_and_keeps_texture(self):
        checker = (np.indices((64, 64)).sum(axis=0) % 2).astype(np.float32)
        checker = (checker * 20.0 - 10.0)[:, :, None]
        reference = np.full((64, 64, 3), 80.0, dtype=np.float32)
        generated = np.clip(reference + 30.0 + checker, 0.0, 255.0)

        corrected = tiled_upscale.anchor_tile_low_frequency(
            generated, reference, strength=0.65, sigma=4.0)

        before_drift = abs(float(generated.mean() - reference.mean()))
        after_drift = abs(float(corrected.mean() - reference.mean()))
        self.assertLess(after_drift, before_drift)
        self.assertGreater(float(corrected.std()), float(generated.std()) * 0.8)

    def test_tile_batch_size_is_conservative_and_bounded(self):
        self.assertEqual(1, tiled_upscale.get_tiled_detail_batch_size(12288, total_tiles=8))
        self.assertEqual(2, tiled_upscale.get_tiled_detail_batch_size(16384, total_tiles=8))
        self.assertEqual(4, tiled_upscale.get_tiled_detail_batch_size(24576, total_tiles=8))
        self.assertEqual(3, tiled_upscale.get_tiled_detail_batch_size(24576, requested=3, total_tiles=8))
        self.assertEqual(2, tiled_upscale.get_tiled_detail_batch_size(24576, requested=8, total_tiles=2))

        with self.assertRaises(ValueError):
            tiled_upscale.get_tiled_detail_batch_size(24576, requested=-1)
