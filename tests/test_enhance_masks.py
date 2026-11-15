import unittest

import numpy as np

from modules.enhance_masks import object_context_field, prepare_enhance_masks


class TestEnhanceMasks(unittest.TestCase):
    def setUp(self):
        self.left = np.zeros((100, 160), dtype=np.uint8)
        self.right = self.left.copy()
        self.left[20:30, 10:20] = 255
        self.right[20:30, 140:150] = 255
        self.union = self.left | self.right

    def test_separate_detections_keep_their_own_crop(self):
        masks = prepare_enhance_masks(self.union, [self.left, self.right], True)
        self.assertEqual(2, len(masks))
        self.assertEqual([100, 100], [np.count_nonzero(mask) for mask in masks])
        self.assertLess(object_context_field(masks[0], 0.618), 0.2)
        self.assertGreater(object_context_field(self.union, 0.618), 0.9)

    def test_disconnected_parts_of_one_detection_stay_together(self):
        masks = prepare_enhance_masks(self.union, [self.union], True)
        self.assertEqual(1, len(masks))
        np.testing.assert_array_equal(self.union, masks[0])

    def test_disabled_and_legacy_worker_use_combined_mask(self):
        for instances, enabled in [([self.left, self.right], False), ([], True)]:
            with self.subTest(enabled=enabled):
                masks = prepare_enhance_masks(self.union, instances, enabled)
                self.assertEqual(1, len(masks))
                np.testing.assert_array_equal(self.union, masks[0])

    def test_inversion_uses_union_once_and_preserves_both_objects(self):
        masks = prepare_enhance_masks(self.union, [self.left, self.right], True, invert=True)
        self.assertEqual(1, len(masks))
        np.testing.assert_array_equal(255 - self.union, masks[0])

    def test_empty_duplicate_and_eroded_away_masks_are_skipped(self):
        empty = np.zeros_like(self.union)
        masks = prepare_enhance_masks(self.union, [empty, self.left, self.left.copy()], True)
        self.assertEqual(1, len(masks))
        self.assertEqual([], prepare_enhance_masks(empty, [], True))
        self.assertEqual([], prepare_enhance_masks(self.union, [self.left], True,
                                                 transform=lambda mask: np.zeros_like(mask)))

    def test_mask_transform_does_not_mutate_detector_results(self):
        original = self.left.copy()

        def transform(mask):
            mask[0, 0] = 255
            return mask

        masks = prepare_enhance_masks(self.union, [self.left], True, transform=transform)
        np.testing.assert_array_equal(original, self.left)
        self.assertEqual(255, masks[0][0, 0])

    def test_invalid_instance_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_enhance_masks(self.union, [np.zeros((2, 2))], True)

    def test_context_increases_crop_but_is_bounded(self):
        self.assertLess(object_context_field(self.left, 0), object_context_field(self.left, 1))
        self.assertEqual(0, object_context_field(np.zeros_like(self.left), 1))
        self.assertEqual(1, object_context_field(np.ones_like(self.left), 1))


if __name__ == '__main__':
    unittest.main()
