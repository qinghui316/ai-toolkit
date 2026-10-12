"""Dependency-free configuration, content identities and resume guards."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

UPSTREAM_COMMIT = "ecee894ed2b1f3716d9d7326693061ec1a3105bb"
OBJECTIVE_VERSION = "krea2_streetscape_inclusive_all6_v1"
FROZEN_CNN_SHA256 = "fb4e439e2c473c51bb0bb07b58d8a318076ac245fe4d54428b25754a6e82294e"
FROZEN_CONFIG_SHA256 = "b9e32c76b0d56aa965e2d29312c3b09ebc89366907e73c28fd6b95471c0b6d51"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LABEL_COLUMNS = [f"{group}_{dimension}" for group in ("youth", "older")
                 for dimension in ("beautiful", "boring", "depressing", "lively", "safe", "wealthy")]


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def source_sha256(path: Path | str) -> str:
    """Code identity ignores checkout newline conversion, not code content."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def read_json(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path | str, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


@dataclass(frozen=True)
class GuidanceConfig:
    enabled: bool = False
    checkpoint: str | None = None
    manifest: str | None = None
    weight: float | None = None
    start_step: int | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None
    every_n_steps: int = 1

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> "GuidanceConfig":
        value = dict(value or {})
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown cnn_guidance fields: {sorted(unknown)}")
        result = cls(**value)
        if type(result.enabled) is not bool:
            raise ValueError("cnn_guidance.enabled must be a boolean")
        if type(result.every_n_steps) is not int or result.every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        if not result.enabled:
            return result
        if not result.checkpoint or not result.manifest:
            raise ValueError("Enabled guidance requires checkpoint and manifest")
        for key in ("weight", "sigma_min", "sigma_max"):
            number = getattr(result, key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError(f"Explicit finite cnn_guidance.{key} is required")
        if result.weight <= 0:
            raise ValueError("weight must be positive")
        if type(result.start_step) is not int or result.start_step < 0:
            raise ValueError("Explicit nonnegative integer start_step is required")
        if not 0 < result.sigma_min < result.sigma_max < 1:
            raise ValueError("Require 0 < sigma_min < sigma_max < 1")
        return result

    def active_at(self, step: int) -> bool:
        return (self.enabled and step >= self.start_step
                and (step - self.start_step) % self.every_n_steps == 0)

    def identity(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "objective": OBJECTIVE_VERSION}
        result = asdict(self)
        result.pop("checkpoint")
        result.pop("manifest")
        return {**result, "objective": OBJECTIVE_VERSION,
                "score_policy": "clip_1_10", "minimum_policy": "hard_min",
                "normalization": "(10-inclusive_all6)/9; sum/full_microbatch"}


def validate_process(config: Mapping[str, Any], guidance: GuidanceConfig) -> None:
    if not guidance.enabled:
        return
    model, train, network = (config.get(key, {}) for key in ("model", "train", "network"))
    # Resolve RAW through its canonical repository (a pre-downloaded HF cache works).
    # A filename alone cannot prove that local weights are RAW rather than Turbo.
    if model.get("arch") != "krea2" or model.get("name_or_path") != "krea/Krea-2-Raw":
        raise ValueError("Guidance v1 requires arch=krea2 and canonical krea/Krea-2-Raw")
    kwargs = model.get("model_kwargs", {})
    if kwargs.get("edit") or kwargs.get("kv_cache") or kwargs.get("checkpoint_filename", "raw.safetensors") != "raw.safetensors":
        raise ValueError("Only unmodified RAW text-to-image conditioning is supported")
    if network.get("type", "lora") != "lora" or network.get("pretrained_lora_path"):
        raise ValueError("Guidance v1 requires plain LoRA; warm-up occurs within the same run")
    if train.get("train_text_encoder") or not train.get("train_unet", True):
        raise ValueError("Only diffusion-model LoRA parameters may be trained")
    if train.get("noise_scheduler") != "flowmatch":
        raise ValueError("Krea 2 guidance requires noise_scheduler=flowmatch")
    incompatible = ("train_turbo", "do_guidance_loss", "do_prior_divergence", "match_noise_norm",
                    "correct_pred_norm", "do_signal_amplification", "diff_output_preservation",
                    "blank_prompt_preservation", "learnable_snr_gos", "do_differential_guidance")
    if any(train.get(key) for key in incompatible) or train.get("diffusion_feature_extractor_path"):
        raise ValueError("Additional prediction transformations/objectives are not supported in v1")
    if train.get("pred_scaler", 1.0) != 1.0 or train.get("target_noise_multiplier", 1.0) != 1.0:
        raise ValueError("Velocity scaling must remain unchanged")
    if config.get("adapter") or config.get("embedding") or config.get("decorator"):
        raise ValueError("Additional trainable adapters/embeddings are not supported")
    if config.get("save", {}).get("save_format", "safetensors") != "safetensors":
        raise ValueError("Research resume guard supports safetensors LoRA saves")
    if train.get("merge_network_on_save") or train.get("start_step") is not None:
        raise ValueError("Do not merge or override native resume step")
    for dataset in config.get("datasets", []):
        if dataset.get("is_reg") or dataset.get("mask_path") or dataset.get("loss_multiplier", 1.0) != 1.0:
            raise ValueError("v1 requires unmasked, non-regularization, uniformly weighted image datasets")


def validate_explicit_training_settings(config: Mapping[str, Any], guidance: GuidanceConfig) -> None:
    """Catch template placeholders before any expensive model download/load."""
    train, network = config.get("train", {}), config.get("network", {})
    for container, field in ((train, "steps"), (train, "batch_size"),
                             (network, "linear"), (network, "linear_alpha")):
        value = container.get(field)
        if type(value) is not int or value < 1:
            raise ValueError(f"Explicit positive integer {field} is required; template is not a training recipe")
    rate = train.get("lr")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Explicit positive finite train.lr is required")
    if guidance.enabled and guidance.start_step >= train["steps"]:
        raise ValueError("start_step must fall inside this run's native global-step range")
    if type(config.get("training_seed")) is not int or config["training_seed"] < 0:
        raise ValueError("Explicit nonnegative training_seed is required")
    if train.get("gradient_accumulation_steps", 1) != 1:
        raise ValueError("Use native gradient_accumulation (microbatch list), not legacy cross-iteration accumulation")
    accumulation = train.get("gradient_accumulation", 1)
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    for dataset in config.get("datasets", []):
        if dataset.get("caption_ext", "txt") != "txt":
            raise ValueError("Research datasets require same-stem UTF-8 .txt captions")
        resolution = dataset.get("resolution")
        sizes = resolution if isinstance(resolution, list) else [resolution]
        if not sizes or any(type(size) is not int or size < 16 for size in sizes):
            raise ValueError("Fill explicit dataset.resolution before starting")


def validate_cnn_files(checkpoint: Path | str, manifest: Path | str) -> dict:
    manifest = Path(manifest)
    payload = read_json(manifest)
    if (payload.get("frozen") is not True or payload.get("sample_count") != 400
            or payload.get("run_type") != "full-development-final-v1"):
        raise ValueError("Expected the frozen 400-development final CNN manifest")
    if (payload.get("model_sha256") != FROZEN_CNN_SHA256
            or payload.get("config_sha256") != FROZEN_CONFIG_SHA256):
        raise ValueError("Manifest is not bound to the previously frozen CNN/config")
    if file_sha256(checkpoint) != payload.get("model_sha256"):
        raise ValueError("Frozen CNN checkpoint SHA-256 mismatch")
    # Ignore historical Windows absolute paths; ship the unchanged config beside the manifest.
    filename = PureWindowsPath(payload.get("config_path", "config.json")).name
    config_path = manifest.parent / filename
    if not config_path.is_file() or file_sha256(config_path) != payload.get("config_sha256"):
        raise ValueError("Frozen config SHA-256 mismatch; copy config.json beside the manifest")
    configuration = read_json(config_path)
    if (configuration.get("model_variant") != "shared-delta"
            or configuration.get("training_objective") != "base"
            or configuration.get("label_columns") != LABEL_COLUMNS
            or (configuration.get("image_height"), configuration.get("image_width")) != (224, 336)):
        raise ValueError("Frozen CNN configuration is incompatible")
    return {"checkpoint_sha256": payload["model_sha256"],
            "manifest_sha256": file_sha256(manifest), "config_sha256": payload["config_sha256"],
            "experiment_identity_sha256": payload["experiment_identity_sha256"]}


def image_id(path: Path) -> str:
    match = re.search(r"CS_T0[1-8]_\d{3}$", path.stem)
    return match.group(0) if match else path.stem


def scan_caption_pairs(folder: Path | str, *, allow_unpaired: bool = False) -> dict:
    """Read only. Mixed tagger exports are auditable but never silently trainable."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Dataset folder does not exist: {folder}")
    pairs, unpaired, seen_ids, seen_hashes = [], [], set(), set()
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES or "cache" in path.relative_to(folder).parts:
            continue
        caption = path.with_suffix(".txt")
        if not caption.is_file():
            unpaired.append(path.relative_to(folder).as_posix())
            continue
        content = caption.read_text(encoding="utf-8-sig")
        if not content.strip() or "\ufffd" in content or "\x00" in content:
            raise ValueError(f"Empty or malformed UTF-8 caption: {caption}")
        identifier, digest = image_id(path), file_sha256(path)
        if identifier in seen_ids or digest in seen_hashes:
            raise ValueError(f"Duplicate paired image ID/content: {path}")
        seen_ids.add(identifier)
        seen_hashes.add(digest)
        pairs.append({"id": identifier, "image": path.relative_to(folder).as_posix(),
                      "caption": caption.relative_to(folder).as_posix(),
                      "image_sha256": digest, "caption_sha256": file_sha256(caption)})
    if not pairs or (unpaired and not allow_unpaired):
        raise ValueError(f"Require an isolated image/TXT dataset: {len(pairs)} pairs, {len(unpaired)} unpaired images")
    paired_captions = {row["caption"] for row in pairs}
    orphan = [p.relative_to(folder).as_posix() for p in folder.rglob("*.txt")
              if "cache" not in p.relative_to(folder).parts
              and p.relative_to(folder).as_posix() not in paired_captions]
    if orphan:
        raise ValueError(f"Orphan captions: {orphan[:10]}")
    return {"pairs": pairs, "pair_count": len(pairs), "unpaired_images": unpaired}


def portable_process_config(process: Mapping[str, Any]) -> dict:
    result = json.loads(json.dumps(process))
    for key in ("training_folder", "log_dir", "performance_log_every", "logging", "cnn_guidance"):
        result.pop(key, None)
    result.pop("name", None)
    for index, dataset in enumerate(result.get("datasets", [])):
        dataset["folder_path"] = f"dataset:{index}"
    model = result.get("model", {})
    model["name_or_path"] = "content-bound-model"
    kwargs = model.get("model_kwargs", {})
    for key in ("vae_path", "text_encoder_path"):
        if key in kwargs:
            kwargs[key] = "content-bound-component"
    if result.get("local_model_assets"):
        result["local_model_assets"] = {key: "content-bound-asset" for key in result["local_model_assets"]}
    return result


def check_or_create_identity(root: Path | str, identity: dict) -> bool:
    root = Path(root)
    target = root / "streetscape_identity.json"
    if target.exists():
        if read_json(target) != identity:
            raise ValueError("Research identity mismatch: use a separate experiment directory")
        return True
    if root.exists() and any(root.iterdir()):
        raise ValueError("Unbound existing run directory: do not overwrite or adopt historical saves")
    write_json_atomic(target, identity)
    return False


def validate_resume_pair(root: Path | str, identity_sha256: str) -> dict:
    root = Path(root)
    record = read_json(root / "streetscape_latest.json")
    if record.get("identity_sha256") != identity_sha256:
        raise ValueError("Latest save belongs to a different research identity")
    for key in ("lora", "optimizer", *(["trainer_state"] if "trainer_state" in record else [])):
        item = record[key]
        name = item["file"]
        if Path(name).name != name or file_sha256(root / name) != item["sha256"]:
            raise ValueError(f"Missing, changed or incomplete native {key} save")
    return record
