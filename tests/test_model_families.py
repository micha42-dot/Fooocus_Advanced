import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from modules import flags
from modules import model_families


class TestModelFamilies(unittest.TestCase):
    def test_context_dimensions_identify_supported_families(self):
        self.assertEqual(
            model_families.family_from_context_dimensions({768}),
            model_families.FAMILY_SD1,
        )
        self.assertEqual(
            model_families.family_from_context_dimensions({1024}),
            model_families.FAMILY_SD2,
        )
        self.assertEqual(
            model_families.family_from_context_dimensions({2048}),
            model_families.FAMILY_SDXL,
        )

    def test_ambiguous_dimensions_are_unknown(self):
        self.assertEqual(
            model_families.family_from_context_dimensions({768, 2048}),
            model_families.FAMILY_UNKNOWN,
        )

    def test_safetensors_header_is_detected_without_model_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'sd15.safetensors')
            save_file({
                'model.diffusion_model.input_blocks.1.1.'
                'transformer_blocks.0.attn2.to_k.weight': torch.zeros(1, 768),
            }, checkpoint_path)

            self.assertEqual(
                model_families.detect_checkpoint_family(checkpoint_path),
                model_families.FAMILY_SD1,
            )

    def test_torch_checkpoint_is_detected_on_meta_device(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'sd15.ckpt')
            torch.save({
                'state_dict': {
                    'model.diffusion_model.input_blocks.1.1.'
                    'transformer_blocks.0.attn2.to_k.weight': torch.zeros(1, 768),
                },
            }, checkpoint_path)

            self.assertEqual(
                model_families.detect_checkpoint_family(checkpoint_path),
                model_families.FAMILY_SD1,
            )

    def test_sd15_presets_cover_512_and_768(self):
        self.assertIn('512*512', flags.sd15_aspect_ratios)
        self.assertIn('768*768', flags.sd15_aspect_ratios)
        self.assertTrue(all(
            int(dimension) % 64 == 0
            for ratio in flags.sd15_aspect_ratios
            for dimension in ratio.split('*')
        ))

    def test_resolution_falls_back_to_family_default(self):
        self.assertEqual(
            model_families.choose_resolution(
                '1152x896',
                ['512x512', '768x768'],
                '512x512',
            ),
            '512x512',
        )
        self.assertEqual(
            model_families.choose_resolution(
                '768x768',
                ['512x512', '768x768'],
                '512x512',
            ),
            '768x768',
        )


if __name__ == '__main__':
    unittest.main()
