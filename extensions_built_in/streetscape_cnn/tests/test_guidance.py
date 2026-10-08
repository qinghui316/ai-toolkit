import unittest
from unittest.mock import Mock

import torch

from extensions_built_in.streetscape_cnn.guidance import guidance_loss, scheduler_sigmas
from extensions_built_in.streetscape_cnn.scorer import reward_components
from test_contract import enabled_config


def toy_scorer(rgb):
    raw = 5 + rgb.mean((1, 2, 3))[:, None].expand(-1, 12)
    return {"raw_scores": raw, "clipped_fraction": raw.new_zeros(()),
            "raw_inclusive_all6": reward_components(raw)["inclusive_all6"], **reward_components(raw)}


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.pred = torch.nn.Parameter(torch.zeros(3, 16, 8, 12))
        self.noisy = torch.full_like(self.pred, .1)
        self.sigmas = torch.tensor([.2, .4, .8])

    def call(self, **overrides):
        values = dict(prediction=self.pred, noisy_latents=self.noisy, sigmas=self.sigmas,
                      config=enabled_config(), step=5, decode=lambda clean: clean[:, :3], scorer=toy_scorer)
        values.update(overrides)
        return guidance_loss(**values)

    def test_actual_sigma_grid_not_divided_timestep(self):
        scheduler = Mock(timesteps=torch.tensor([1000., 700., 0.]), sigmas=torch.tensor([1., .35, 0., 0.]))
        self.assertTrue(torch.equal(scheduler_sigmas(scheduler, torch.tensor([700., 0.])), torch.tensor([.35, 0.])))
        with self.assertRaises(ValueError):
            scheduler_sigmas(scheduler, torch.tensor([123.]))

    def test_full_microbatch_normalization_and_finite_gradients(self):
        loss, logs = self.call()
        expected = (10 - toy_scorer(torch.full((1, 3, 8, 12), .55))["inclusive_all6"]) / 9 / 3
        self.assertTrue(torch.allclose(loss, expected.squeeze()))
        gradient = torch.autograd.grad(loss, self.pred)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient[0] != 0).any())
        self.assertEqual(int(torch.count_nonzero(gradient[1:])), 0)
        self.assertEqual(float(logs["cnn/eligible_count"]), 1)

    def test_inactive_or_ineligible_skips_decoder_and_scorer(self):
        decode, scorer = Mock(side_effect=AssertionError("must not be called")), Mock(side_effect=AssertionError("must not be called"))
        for kwargs in ({"step": 4}, {"step": 6}, {"sigmas": torch.full((3,), .8)}):
            loss, _ = self.call(decode=decode, scorer=scorer, **kwargs)
            self.assertEqual(float(loss), 0)
        decode.assert_not_called()
        scorer.assert_not_called()

    def test_detached_decode_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.call(decode=lambda clean: clean[:, :3].detach())
        invalid = self.noisy.clone()
        invalid[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.call(noisy_latents=invalid)
