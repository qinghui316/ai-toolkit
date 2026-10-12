"""Tensor-only one-step flow-matching proxy; not a full sampled-image reward."""
from __future__ import annotations

import torch

from .contract import GuidanceConfig


def scheduler_sigmas(scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """Gather the actual scheduler grid; never assume timestep/1000 equals sigma."""
    grid = scheduler.timesteps.to(timesteps.device)
    matches = grid[:, None] == timesteps.reshape(1, -1)
    if not (matches.sum(0) == 1).all():
        raise ValueError("Each training timestep must match exactly one scheduler entry")
    indices = matches.to(torch.int64).argmax(0)
    sigmas = scheduler.sigmas.to(timesteps.device)[indices].float()
    if not torch.isfinite(sigmas).all() or not ((sigmas >= 0) & (sigmas <= 1)).all():
        raise ValueError("Invalid flow-matching sigma grid")
    return sigmas


def guidance_loss(*, prediction: torch.Tensor, noisy_latents: torch.Tensor,
                  sigmas: torch.Tensor, config: GuidanceConfig, step: int,
                  decode, scorer, observer=None) -> tuple[torch.Tensor, dict]:
    zero = prediction.new_zeros((), dtype=torch.float32)
    metrics = {"cnn/eligible_count": zero, "cnn/full_microbatch": zero + prediction.shape[0],
               "loss/cnn": zero}
    if not config.active_at(step):
        return zero, metrics
    if prediction.shape != noisy_latents.shape or prediction.ndim != 4:
        raise ValueError("Krea 2 v1 requires equal BCHW velocity and noisy latent tensors")
    sigmas = sigmas.to(device=prediction.device, dtype=torch.float32)
    if (sigmas.shape != (prediction.shape[0],) or not torch.isfinite(sigmas).all()
            or not torch.isfinite(prediction).all() or not torch.isfinite(noisy_latents).all()):
        raise ValueError("Invalid or non-finite guidance batch")
    eligible = (sigmas >= config.sigma_min) & (sigmas <= config.sigma_max)
    metrics["cnn/eligible_count"] = eligible.sum().float()
    if not eligible.any():
        return zero, metrics
    sigma = sigmas[eligible].to(prediction.device).view(-1, 1, 1, 1)
    clean = noisy_latents[eligible].float() - sigma * prediction[eligible].float()
    decoded = decode(clean)
    if decoded.ndim != 4 or decoded.shape[1] != 3 or not torch.isfinite(decoded).all():
        raise ValueError("VAE must return finite BCHW RGB tensors")
    # Qwen-Image VAE outputs [-1,1]; clipping is a differentiable, tensor-only range policy.
    rgb = ((decoded.float() + 1) / 2).clamp(0, 1)
    result = scorer(rgb)
    reward = result["inclusive_all6"]
    if reward.shape != (int(eligible.sum()),) or not torch.isfinite(reward).all():
        raise ValueError("CNN reward shape/value is invalid")
    loss = ((10 - reward) / 9).sum() / prediction.shape[0]
    if not torch.isfinite(loss):
        raise ValueError("Non-finite auxiliary loss")
    if torch.is_grad_enabled() and prediction.requires_grad and not loss.requires_grad:
        raise RuntimeError("Auxiliary loss lost its graph; do not use no_grad/PIL/detach in the scoring path")
    metrics.update({"loss/cnn": loss, "cnn/inclusive_all6": reward.mean(),
                    "cnn/raw_inclusive_all6": result["raw_inclusive_all6"].mean(),
                    "cnn/clipped_fraction": result["clipped_fraction"],
                    "cnn/youth_quality": result["youth_quality"].mean(),
                    "cnn/older_quality": result["older_quality"].mean(),
                    "cnn/group_gap": result["group_gap"].mean(),
                    "cnn/minimum_safety": result["minimum_safety"].mean(),
                    "cnn/raw_score_mean": result["raw_scores"].mean(),
                    "cnn/raw_score_min": result["raw_scores"].min(),
                    "cnn/raw_score_max": result["raw_scores"].max()})
    if observer is not None:
        observer(eligible, result)
    return loss, metrics
