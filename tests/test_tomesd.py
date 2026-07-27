import unittest

import torch

from ldm_patched.contrib.external_tomesd import get_functions
from modules.tomesd import apply_tomesd_profile, normalize_tomesd_profile, tomesd_ratio


class FakeModel:
    def __init__(self):
        self.attn1_patches = []
        self.attn1_output_patches = []

    def clone(self):
        clone = FakeModel()
        clone.attn1_patches = self.attn1_patches.copy()
        clone.attn1_output_patches = self.attn1_output_patches.copy()
        return clone

    def set_model_attn1_patch(self, patch):
        self.attn1_patches.append(patch)

    def set_model_attn1_output_patch(self, patch):
        self.attn1_output_patches.append(patch)


class TestToMeSD(unittest.TestCase):
    def test_profiles_are_normalized_and_ordered(self):
        self.assertEqual(normalize_tomesd_profile('balanced'), 'balanced')
        self.assertEqual(normalize_tomesd_profile('unknown'), 'off')
        self.assertLess(tomesd_ratio('conservative'), tomesd_ratio('balanced'))
        self.assertLess(tomesd_ratio('balanced'), tomesd_ratio('aggressive'))
        self.assertEqual(tomesd_ratio('off'), 0.0)

    def test_deterministic_merge_repeats_exactly(self):
        tokens = torch.arange(64 * 8, dtype=torch.float32).reshape(1, 64, 8)
        merge_a, unmerge_a = get_functions(
            tokens, ratio=0.3, original_shape=(1, 4, 8, 8), deterministic=True)
        merge_b, unmerge_b = get_functions(
            tokens, ratio=0.3, original_shape=(1, 4, 8, 8), deterministic=True)

        merged_a = merge_a(tokens)
        merged_b = merge_b(tokens)

        self.assertTrue(torch.equal(merged_a, merged_b))
        self.assertTrue(torch.equal(unmerge_a(merged_a), unmerge_b(merged_b)))

    def test_profile_patches_a_clone_and_off_keeps_original(self):
        model = FakeModel()
        patched = apply_tomesd_profile(model, 'balanced')

        self.assertIsNot(patched, model)
        self.assertEqual(len(model.attn1_patches), 0)
        self.assertEqual(len(patched.attn1_patches), 1)
        self.assertEqual(len(patched.attn1_output_patches), 1)
        self.assertIs(apply_tomesd_profile(model, 'off'), model)


if __name__ == '__main__':
    unittest.main()
