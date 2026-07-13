import unittest

import torch

from ldm_patched.k_diffusion import sampling
from modules.guidance import adaptive_projected_guidance, cfgpp_scale, supports_cfgpp_sampler


class TestGuidance(unittest.TestCase):
    def test_apg_removes_parallel_component(self):
        uncond = torch.tensor([[0.0, 1.0]])
        cond = torch.tensor([[1.0, 1.0]])
        result, running = adaptive_projected_guidance(
            uncond, cond, cfg_scale=2.0, norm_threshold=100.0, eta=0.0, momentum=0.0)

        self.assertTrue(torch.allclose(result, torch.tensor([[1.0, 0.0]]), atol=1e-5))
        self.assertTrue(torch.equal(running, cond - uncond))

    def test_cfgpp_uses_small_lambda_range(self):
        self.assertAlmostEqual(cfgpp_scale(7.0), 0.56)
        self.assertEqual(cfgpp_scale(30.0), 1.0)
        self.assertTrue(supports_cfgpp_sampler('dpmpp_2m_sde_gpu'))
        self.assertFalse(supports_cfgpp_sampler('heun'))

    def test_euler_cfgpp_renoises_with_unconditional_prediction(self):
        original = sampling.cfgpp_get_uncond_denoised
        sampling.cfgpp_get_uncond_denoised = lambda: torch.tensor([2.0])
        try:
            result = sampling.sample_euler(
                lambda x, sigma, **kwargs: torch.tensor([4.0]),
                torch.tensor([10.0]),
                torch.tensor([2.0, 1.0]),
                disable=True,
            )
        finally:
            sampling.cfgpp_get_uncond_denoised = original

        self.assertTrue(torch.equal(result, torch.tensor([8.0])))

    def test_dpmpp_2m_cfgpp_matches_first_order_update(self):
        original = sampling.cfgpp_get_uncond_denoised
        sampling.cfgpp_get_uncond_denoised = lambda: torch.tensor([2.0])
        try:
            result = sampling.sample_dpmpp_2m(
                lambda x, sigma, **kwargs: torch.tensor([4.0]),
                torch.tensor([10.0]),
                torch.tensor([2.0, 1.0]),
                disable=True,
            )
        finally:
            sampling.cfgpp_get_uncond_denoised = original

        self.assertTrue(torch.equal(result, torch.tensor([8.0])))


if __name__ == '__main__':
    unittest.main()
