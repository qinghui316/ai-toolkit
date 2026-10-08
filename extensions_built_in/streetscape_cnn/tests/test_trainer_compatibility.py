import unittest
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions_built_in.streetscape_cnn.contract import GuidanceConfig
from extensions_built_in.streetscape_cnn.trainer import StreetscapeCNNTrainer, assert_optimizer_restored
from test_contract import enabled_config
from test_guidance import toy_scorer


class TrainerCompatibilityTests(unittest.TestCase):
    def test_disabled_returns_identical_native_base_loss_once(self):
        trainer = StreetscapeCNNTrainer.__new__(StreetscapeCNNTrainer)
        trainer.cnn_guidance = GuidanceConfig()
        trainer._decode_for_cnn = Mock(side_effect=AssertionError("disabled"))
        trainer._score_for_cnn = Mock(side_effect=AssertionError("disabled"))
        native_loss = torch.tensor(1.25, requires_grad=True)
        with patch.object(SDTrainer, "calculate_loss", return_value=native_loss) as native:
            result = trainer.calculate_loss(None, None, None, None, None)
        self.assertIs(result, native_loss)
        native.assert_called_once()
        trainer._decode_for_cnn.assert_not_called()
        trainer._score_for_cnn.assert_not_called()

    def test_total_loss_three_components_are_finite_and_weighted_once(self):
        trainer = StreetscapeCNNTrainer.__new__(StreetscapeCNNTrainer)
        trainer.cnn_guidance = enabled_config()
        trainer.step_num = 5
        trainer._cnn_log_sums, trainer._cnn_log_weights, trainer._cnn_log_count = {}, {}, 0
        trainer.sd = SimpleNamespace(noise_scheduler=SimpleNamespace(
            timesteps=torch.tensor([600.]), sigmas=torch.tensor([.2, 0.])))
        trainer._decode_for_cnn = lambda clean: clean[:, :3]
        trainer._score_for_cnn = toy_scorer
        batch = Mock(mask_tensor=None, loss_multiplier_list=[1.])
        batch.get_is_reg_list.return_value = [False]
        prediction = torch.zeros(1, 16, 8, 12, requires_grad=True)
        noisy = torch.full_like(prediction, .1)
        native_loss = torch.tensor(1.25)
        with patch.object(SDTrainer, "calculate_loss", return_value=native_loss), \
                patch.dict(os.environ, {"STREETSCAPE_VERIFY_GRADIENTS": "0"}):
            total = trainer.calculate_loss(prediction, torch.zeros_like(prediction), noisy,
                                           torch.tensor([600.]), batch)
        auxiliary = trainer._cnn_log_sums["loss/cnn"]
        self.assertAlmostEqual(float(total), 1.25 + .1 * auxiliary, places=6)
        self.assertTrue(torch.isfinite(torch.autograd.grad(total, prediction)[0]).all())
        self.assertEqual(trainer._cnn_log_count, 1)

    def test_optimizer_load_failure_cannot_be_swallowed(self):
        expected = {0: {"step": torch.tensor(4.), "moment": torch.ones(2)}}
        assert_optimizer_restored(expected, expected)
        with self.assertRaises(ValueError):
            assert_optimizer_restored({}, expected)
        with self.assertRaises(ValueError):
            assert_optimizer_restored({0: {"step": torch.tensor(3.), "moment": torch.ones(2)}}, expected)
