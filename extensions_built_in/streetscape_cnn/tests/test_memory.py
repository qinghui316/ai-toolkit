import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from extensions_built_in.streetscape_cnn.memory import parse_profile, configure_decoder, scoring_memory_context, clear_decoder_cache
from extensions_built_in.streetscape_cnn.contract import check_or_create_identity
from tempfile import TemporaryDirectory


class MemoryTests(unittest.TestCase):
    def test_default_and_invalid_profiles(self):
        self.assertEqual(parse_profile(), 'legacy')
        for value in ('turbo', True, {}, 1):
            with self.assertRaises(ValueError):
                parse_profile(value)

    def test_frozen_decoder_flags_and_no_legacy_activation(self):
        vae = Mock(dtype=torch.float32, training=False)
        vae.decoder = SimpleNamespace(gradient_checkpointing=False)
        configure_decoder(vae, 'legacy')
        vae.enable_gradient_checkpointing.assert_not_called()
        vae.enable_gradient_checkpointing.side_effect = lambda: setattr(vae.decoder, 'gradient_checkpointing', True)
        with patch('toolkit.util.qwen_vae_gradient_checkpointing.patch_qwen_vae_gradient_checkpointing'):
            self.assertTrue(configure_decoder(vae, 'vae_checkpoint')['decoder_checkpointing'])
        vae.requires_grad_.assert_called_with(False)
        clear_decoder_cache(vae)
        vae.clear_cache.assert_called_once()

    def test_cpu_saving_preserves_input_gradient(self):
        for profile in ('legacy', 'vae_checkpoint', 'vae_checkpoint_cpu'):
            x = torch.linspace(.1, .9, 32, requires_grad=True)
            with scoring_memory_context(profile):
                loss = x.square().sum()
            self.assertTrue(torch.equal(torch.autograd.grad(loss, x)[0], 2 * x.detach()))

    def test_cross_profile_identity_is_rejected(self):
        with TemporaryDirectory() as root:
            check_or_create_identity(root, {'cnn_memory_profile': 'vae_checkpoint'})
            with self.assertRaises(ValueError):
                check_or_create_identity(root, {'cnn_memory_profile': 'vae_checkpoint_cpu'})
