import unittest

import torch
from torch import nn

from extensions_built_in.streetscape_cnn.scorer import FrozenStreetscapeScorer, decode_scores, reward_components

CODEC = {"type": "shared_delta", "mean": {"mean": [5.] * 6, "std": [1.] * 6},
         "delta": {"mean": [0.] * 6, "std": [1.] * 6}}


class ScorerTests(unittest.TestCase):
    def test_decode_matches_frozen_mean_delta_formula(self):
        values = torch.randn(2, 12, requires_grad=True)
        scores = decode_scores(values, CODEC)
        self.assertTrue(torch.equal(scores[:, :6], values[:, :6] + 5 + values[:, 6:] / 2))
        self.assertTrue(torch.equal(scores[:, 6:], values[:, :6] + 5 - values[:, 6:] / 2))
        gradient = torch.autograd.grad(scores.square().mean(), values)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_reward_matches_existing_scalar_contract(self):
        row = torch.tensor([[8., 2., 3., 7., 9., 6., 7., 3., 4., 8., 6., 5.]])
        youth = [8, 9, 8, 7, 9, 6]
        older = [7, 8, 7, 8, 6, 5]
        expected = sum(.5 * ((a + b) / 2) + .5 * min(a, b) for a, b in zip(youth, older)) / 6
        self.assertAlmostEqual(float(reward_components(row)["inclusive_all6"]), expected, places=6)

    def test_freeze_buffers_and_input_gradient(self):
        model = nn.Sequential(nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 12))
        nn.init.constant_(model[-1].weight, .05)
        nn.init.zeros_(model[-1].bias)
        scorer = FrozenStreetscapeScorer(model, CODEC)
        scorer.train(True)
        self.assertFalse(model.training)
        before = {key: value.clone() for key, value in scorer.named_buffers()}
        rgb = torch.full((1, 3, 32, 48), .5, requires_grad=True)
        result = scorer(rgb)
        gradient = torch.autograd.grad(result["inclusive_all6"].mean(), rgb)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient != 0).any())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in scorer.parameters()))
        self.assertTrue(all(torch.equal(before[key], value) for key, value in scorer.named_buffers()))

    def test_bad_codec_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_scores(torch.zeros(1, 12), {"type": "other"})
