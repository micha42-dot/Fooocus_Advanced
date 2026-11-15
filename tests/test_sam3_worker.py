import base64
import contextlib
import io
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from extras.sam3_worker import combine_masks, SAM3Runtime


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

    def test_segment_returns_only_selected_instances_and_legacy_union(self):
        masks = np.zeros((3, 1, 8, 8), dtype=bool)
        masks[0, 0, :2, :2] = True
        masks[1, 0, 6:, 6:] = True
        masks[2, 0, :, :] = True
        mask_tensor, score_tensor = Mock(), Mock()
        mask_tensor.detach.return_value.to.return_value.numpy.return_value = masks
        score_tensor.detach.return_value.float.return_value.to.return_value.numpy.return_value = np.array([.7, .9, .1])
        processor = Mock()
        processor.set_text_prompt.return_value = {'masks': mask_tensor, 'scores': score_tensor}
        processor_module = types.ModuleType('sam3.model.sam3_image_processor')
        processor_module.Sam3Processor = Mock(return_value=processor)
        torch = types.ModuleType('torch')
        torch.inference_mode = contextlib.nullcontext
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        runtime = SAM3Runtime.__new__(SAM3Runtime)
        runtime.lock = threading.Lock()
        runtime.model = Mock()
        runtime.device = types.SimpleNamespace(type='cpu')
        runtime.keep_gpu = False
        runtime.load = Mock()
        image = Image.new('RGB', (8, 8))
        with patch.dict('sys.modules', {'torch': torch, 'sam3.model.sam3_image_processor': processor_module}):
            result = runtime.segment(image, 'face', .3, 1, include_instances=True)
            legacy = runtime.segment(image, 'face', .3, 1)
        self.assertEqual(1, result['detection_count'])
        self.assertEqual(1, len(result['instance_masks']))
        decoded = np.array(Image.open(io.BytesIO(base64.b64decode(result['instance_masks'][0]))))
        np.testing.assert_array_equal(masks[1, 0].astype(np.uint8) * 255, decoded)
        self.assertEqual(result['mask'], legacy['mask'])
        self.assertNotIn('instance_masks', legacy)
