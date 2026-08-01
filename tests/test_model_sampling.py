import unittest

import torch

from ldm_patched.modules.model_sampling import ModelSamplingDiscrete


class TestModelSamplingDiscrete(unittest.TestCase):
    def test_default_schedule_uses_torch_tensors(self):
        sampling = ModelSamplingDiscrete()

        self.assertEqual(tuple(sampling.sigmas.shape), (1000,))
        self.assertEqual(tuple(sampling.alphas_cumprod.shape), (1000,))
        self.assertTrue(torch.isfinite(sampling.sigmas).all())
        self.assertTrue(torch.isfinite(sampling.alphas_cumprod).all())
        self.assertGreater(sampling.sigma_max, sampling.sigma_min)


if __name__ == '__main__':
    unittest.main()
