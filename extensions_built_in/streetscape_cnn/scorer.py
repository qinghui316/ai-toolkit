"""Frozen ResNet18 scoring contract, derived from lunwen/cnn_training.

No training or fitting lives here. The existing checkpoint supplies all statistics.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contract import LABEL_COLUMNS, validate_cnn_files


def build_scoring_model(dropout: float = 0.3, hidden_features: int = 128) -> nn.Module:
    from torchvision.models import resnet18

    class SharedDeltaHead(nn.Module):
        def __init__(self, input_features: int):
            super().__init__()
            self.shared = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_features, hidden_features), nn.ReLU())
            self.mean_head = nn.Linear(hidden_features, 6)
            self.delta_head = nn.Linear(hidden_features, 6)

        def forward(self, features):
            shared = self.shared(features)
            return torch.cat((self.mean_head(shared), self.delta_head(shared)), dim=1)

    model = resnet18(weights=None)
    model.fc = SharedDeltaHead(model.fc.in_features)
    return model


def decode_scores(values: torch.Tensor, codec: dict) -> torch.Tensor:
    if codec.get("type") != "shared_delta" or values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("Expected batch x 12 standardized shared-delta outputs")
    statistics = []
    for head in ("mean", "delta"):
        item = codec[head]
        mean, std = (values.new_tensor(item[key]) for key in ("mean", "std"))
        if mean.shape != (6,) or std.shape != (6,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("Invalid frozen shared-delta statistics")
        statistics.append(values[:, :6] * std + mean if head == "mean" else values[:, 6:] * std + mean)
    means, deltas = statistics
    return torch.cat((means + deltas / 2, means - deltas / 2), dim=1)


def reward_components(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    if scores.ndim != 2 or scores.shape[1] != 12:
        raise ValueError("Reward input must have shape batch x 12")
    direction = scores.new_tensor((1., -1., -1., 1., 1., 1.))
    offset = scores.new_tensor((0., 11., 11., 0., 0., 0.))
    youth, older = (scores[:, start:start + 6] * direction + offset for start in (0, 6))
    lower = torch.minimum(youth, older)
    inclusive = .5 * ((youth + older) / 2) + .5 * lower
    return {"inclusive_all6": inclusive.mean(1), "youth_quality": youth.mean(1),
            "older_quality": older.mean(1), "group_gap": (youth - older).abs().mean(1),
            "minimum_safety": lower[:, 4]}


def cnn_preprocess(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not torch.isfinite(rgb).all():
        raise ValueError("Expected finite RGB images in BCHW layout")
    rgb = F.interpolate(rgb.float(), size=(224, 336), mode="bilinear", align_corners=False, antialias=True)
    mean = rgb.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    return (rgb - mean) / std


class FrozenStreetscapeScorer(nn.Module):
    def __init__(self, model: nn.Module, codec: dict):
        super().__init__()
        self.model = model.float().eval().requires_grad_(False)
        self.codec = codec
        # Validate statistics without modifying/refitting them.
        decode_scores(torch.zeros(1, 12), codec)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, manifest_path):
        identity = validate_cnn_files(checkpoint_path, manifest_path)
        # This is the user's own trusted checkpoint, whose bytes were verified above.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint.get("run_type") != "full-development-final-v1"
                or checkpoint.get("model_variant") != "shared-delta"
                or checkpoint.get("training_objective") != "base"
                or checkpoint.get("reward_version") != "inclusive_reward_v1"
                or checkpoint.get("label_columns") != LABEL_COLUMNS
                or checkpoint.get("config_id") != identity["experiment_identity_sha256"][:20]
                or checkpoint.get("phase") != "finetune" or not checkpoint.get("phase_complete")):
            raise ValueError("Not the compatible, complete frozen CNN checkpoint")
        config = checkpoint["training_config"]
        model = build_scoring_model(config["dropout"], config["hidden_features"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        return cls(model, checkpoint["target_codec"])

    def train(self, mode=True):
        # Freeze dropout and batch-normalization behavior as well as weights.
        super().train(False)
        return self

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            raw = decode_scores(self.model(cnn_preprocess(rgb)), self.codec)
            if not torch.isfinite(raw).all():
                raise ValueError("CNN produced non-finite predictions")
            clipped = raw.clamp(1, 10)
            return {"raw_scores": raw, "clipped_scores": clipped,
                    "clipped_fraction": ((raw < 1) | (raw > 10)).float().mean(),
                    **reward_components(clipped),
                    "raw_inclusive_all6": reward_components(raw)["inclusive_all6"]}
