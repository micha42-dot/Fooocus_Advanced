import unittest

from modules.vram_policy import choose_vram_policy


class TestVramPolicy(unittest.TestCase):
    def test_auto_policy_uses_vram_tiers(self):
        self.assertEqual('conservative', choose_vram_policy(8192))
        self.assertEqual('balanced', choose_vram_policy(12288))
        self.assertEqual('resident', choose_vram_policy(24576))

    def test_explicit_policy_and_force_resident(self):
        self.assertEqual('conservative', choose_vram_policy(24576, requested='conservative'))
        self.assertEqual('resident', choose_vram_policy(4096, force_resident=True))
