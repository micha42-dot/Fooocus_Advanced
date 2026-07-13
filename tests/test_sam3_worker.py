import unittest

import numpy as np

from extras.sam3_worker import combine_masks


class TestSam3Worker(unittest.TestCase):
    def test_combines_selected_masks_by_score(self):
        masks = np.zeros((3, 1, 4, 4), dtype=bool)
        masks[0, 0, :2, :2] = True
        masks[1, 0, 2:, 2:] = True
        masks[2, 0, :, :] = True
        scores = np.array([0.9, 0.8, 0.1], dtype=np.float32)

        result, selected = combine_masks(
            masks, scores, image_shape=(4, 4), confidence_threshold=0.3, max_detections=2)

        self.assertEqual([0, 1], selected.tolist())
        self.assertEqual(8, np.count_nonzero(result))

    def test_returns_empty_mask_without_detection(self):
        masks = np.ones((1, 4, 4), dtype=bool)
        result, selected = combine_masks(
            masks, [0.1], image_shape=(4, 4), confidence_threshold=0.3)

        self.assertEqual(0, len(selected))
        self.assertEqual(0, np.count_nonzero(result))
