import ast
import base64
import io
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from extras.sam3_worker import encode_mask


class TestSam3Client(unittest.TestCase):
    def setUp(self):
        # Load the real client function without parsing Fooocus command-line arguments
        # or importing its CUDA model manager in this CPU-only protocol test.
        path = Path(__file__).resolve().parents[1] / 'extras' / 'sam3_client.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        self.http = types.SimpleNamespace(get=Mock(), post=Mock())
        namespace = dict(np=np, Image=Image, base64=base64, io=io, httpx=self.http,
                         args_manager=types.SimpleNamespace(args=types.SimpleNamespace()))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        self.generate = namespace['generate_sam3_mask']
        self.error = namespace['SAM3UnavailableError']
        self.image = np.zeros((12, 16, 3), dtype=np.uint8)
        self.left = np.zeros((12, 16), dtype=np.uint8)
        self.right = self.left.copy()
        self.left[2:5, 2:5] = 255
        self.right[2:5, 11:14] = 255
        self.result = dict(mask=encode_mask(self.left | self.right), detection_count=2,
                           instance_masks=[encode_mask(self.left), encode_mask(self.right)])
        self.http.post.return_value.json.side_effect = lambda: self.result
        manager = types.SimpleNamespace(unload_all_models=Mock(), soft_empty_cache=Mock())
        modules = types.ModuleType('ldm_patched.modules')
        modules.model_management = manager
        self.module_patch = patch.dict('sys.modules', {'ldm_patched.modules': modules})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_instances_roundtrip_without_merging(self):
        union, *counts, masks = self.generate(self.image, 'face', return_instances=True)
        self.assertEqual([2, 2, 2], counts)
        np.testing.assert_array_equal(self.left | self.right, union[:, :, 0])
        np.testing.assert_array_equal(self.left, masks[0])
        np.testing.assert_array_equal(self.right, masks[1])
        self.assertTrue(self.http.post.call_args.kwargs['json']['include_instances'])

    def test_legacy_caller_still_receives_four_values(self):
        self.assertEqual(4, len(self.generate(self.image, 'face')))
        self.assertFalse(self.http.post.call_args.kwargs['json']['include_instances'])

    def test_old_worker_falls_back_to_union(self):
        del self.result['instance_masks']
        result = self.generate(self.image, 'face', return_instances=True)
        self.assertEqual([], result[4])
        np.testing.assert_array_equal(self.left | self.right, result[0][:, :, 0])

    def test_malformed_instance_size_is_rejected(self):
        self.result['instance_masks'] = [encode_mask(np.zeros((2, 2), dtype=np.uint8))]
        with self.assertRaises(self.error):
            self.generate(self.image, 'face', return_instances=True)


if __name__ == '__main__':
    unittest.main()
