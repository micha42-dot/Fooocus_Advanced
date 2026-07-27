import unittest

from modules import flags


class TestMaskModelLabels(unittest.TestCase):
    def test_sam3_has_recommended_ui_label(self):
        self.assertIn(flags.inpaint_mask_sam3_label, flags.inpaint_mask_model_choices)
        self.assertNotIn(flags.inpaint_mask_sam3, flags.inpaint_mask_model_choices)

    def test_sam3_ui_label_normalizes_to_internal_value(self):
        self.assertEqual(
            flags.normalize_inpaint_mask_model(flags.inpaint_mask_sam3_label),
            flags.inpaint_mask_sam3,
        )
        self.assertEqual(
            flags.display_inpaint_mask_model(flags.inpaint_mask_sam3),
            flags.inpaint_mask_sam3_label,
        )

    def test_other_mask_models_are_unchanged(self):
        self.assertEqual(flags.normalize_inpaint_mask_model('sam'), 'sam')
        self.assertEqual(
            flags.display_inpaint_mask_model('isnet-general-use'),
            'isnet-general-use',
        )


if __name__ == '__main__':
    unittest.main()
