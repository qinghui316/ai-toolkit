"""Opt-in SDTrainer subclass. Native training/backward/cache/optimizer remain in charge."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from .assets import resolve_krea_assets
from .contract import (GuidanceConfig, OBJECTIVE_VERSION, UPSTREAM_COMMIT, check_or_create_identity,
                       file_sha256, object_sha256, portable_process_config, read_json,
                       scan_caption_pairs, validate_cnn_files, validate_process,
                       validate_resume_pair, validate_explicit_training_settings, write_json_atomic, source_sha256)
from .guidance import guidance_loss, scheduler_sigmas
from .scorer import FrozenStreetscapeScorer


def assert_optimizer_restored(actual, expected, path="state"):
    """Fail closed if upstream swallowed an optimizer-load failure; no replacement optimizer."""
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                or actual.dtype != expected.dtype or not torch.equal(actual.detach().cpu(), expected.cpu())):
            raise ValueError(f"Native optimizer state mismatch at {path}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Native optimizer keys mismatch at {path}")
        for key, value in expected.items():
            assert_optimizer_restored(actual[key], value, f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"Native optimizer sequence mismatch at {path}")
        for index, value in enumerate(expected):
            assert_optimizer_restored(actual[index], value, f"{path}/{index}")
    elif actual != expected:
        raise ValueError(f"Native optimizer value mismatch at {path}")


class StreetscapeCNNTrainer(SDTrainer):
    def __init__(self, process_id, job, config, **kwargs):
        if int(os.getenv("WORLD_SIZE", "1")) != 1:
            raise ValueError("Research extension v1 is single-process; distributed identity/save handling is not accepted")
        self.cnn_guidance = GuidanceConfig.parse(config.get("cnn_guidance"))
        validate_process(config, self.cnn_guidance)
        validate_explicit_training_settings(config, self.cnn_guidance)
        original = copy.deepcopy(config)
        folder = config.get("training_folder", getattr(job, "training_folder", None))
        name = config.get("name", job.name)
        root = Path(folder) / name
        identity_path = root / "streetscape_identity.json"
        previous = read_json(identity_path) if identity_path.exists() else None
        # Check research settings BEFORE BaseTrainProcess overwrites config.yaml.
        configuration = portable_process_config(original)
        if previous and (previous["guidance"] != self.cnn_guidance.identity()
                         or previous["configuration"] != configuration):
            raise ValueError("Changed training/guidance configuration; use a new run name")
        if not previous and root.exists() and any(root.iterdir()):
            raise ValueError("Existing output is not a content-bound streetscape run")
        repo = Path(__file__).resolve().parents[2]
        subprocess.run(["git", "merge-base", "--is-ancestor", UPSTREAM_COMMIT, "HEAD"],
                       cwd=repo, check=True, capture_output=True)
        source_files = {p.name: source_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        protected = ["jobs/process/BaseSDTrainProcess.py", "extensions_built_in/sd_trainer/SDTrainer.py",
                     "extensions_built_in/diffusion_models/krea2/krea2.py", "toolkit/models/v2/vae/qwen_image.py"]
        base_sources = {p: source_sha256(repo / p) for p in protected}
        if subprocess.run(["git", "diff", "--quiet", UPSTREAM_COMMIT, "--", *protected], cwd=repo).returncode:
            raise ValueError("Protected upstream training/Krea/VAE sources changed")
        changed = subprocess.run(["git", "diff", "--name-only", UPSTREAM_COMMIT], cwd=repo,
                                 check=True, capture_output=True, text=True).stdout.splitlines()
        allowed_files = {".gitignore", "docs/streetscape_cnn.md",
                         "config/examples/train_lora_krea2_streetscape_base.yaml",
                         "config/examples/train_lora_krea2_streetscape_cnn.yaml"}
        if any(p not in allowed_files and not p.startswith(("extensions_built_in/streetscape_cnn/",
                                                          "tools/streetscape_cnn/")) for p in changed):
            raise ValueError("Upstream/framework changes outside the audited research extension are not supported")
        datasets = [scan_caption_pairs(item["folder_path"]) for item in original.get("datasets", [])]
        if not datasets:
            raise ValueError("Research run requires an explicit captioned dataset")
        ids = [row["id"] for dataset in datasets for row in dataset["pairs"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated image IDs across dataset folders")
        cnn = validate_cnn_files(self.cnn_guidance.checkpoint, self.cnn_guidance.manifest) if self.cnn_guidance.enabled else None
        materialized, assets = resolve_krea_assets(original, previous.get("model_assets") if previous else None)
        self.research_identity = {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
                                  "objective": OBJECTIVE_VERSION, "configuration": configuration,
                                  "guidance": self.cnn_guidance.identity(), "cnn": cnn,
                                  "datasets": datasets, "model_assets": assets,
                                  "extension_sources": source_files, "protected_upstream_sources": base_sources}
        self.research_identity_sha256 = object_sha256(self.research_identity)
        existed = check_or_create_identity(root, self.research_identity)
        self._resume_record = validate_resume_pair(root, self.research_identity_sha256) if existed else None
        self._saving_lora = None
        self._cnn_scorer = None
        self._gradient_verified = False
        self._cnn_log_sums = {}
        self._cnn_log_weights = {}
        self._cnn_log_count = 0
        super().__init__(process_id, job, materialized, **kwargs)
        self.add_meta({"streetscape_identity_sha256": self.research_identity_sha256,
                       "streetscape_objective": OBJECTIVE_VERSION,
                       "streetscape_cnn_guidance": self.cnn_guidance.identity()})

    def get_latest_save_path(self, name=None, post="", include_pretrained_lora=True):
        # Native "newest ctime" is ambiguous after files are transferred to another host.
        if self._resume_record and name is None and not post and self._saving_lora is None:
            return str(Path(self.save_root) / self._resume_record["lora"]["file"])
        return super().get_latest_save_path(name, post, include_pretrained_lora)

    def hook_before_train_loop(self):
        if self._resume_record:
            if self.network is None or self.network.did_change_weights:
                raise ValueError("Native LoRA restore is incompatible")
            if int(self.step_num) != self._resume_record["step"]:
                raise ValueError("Native restore did not restore the saved global step")
            expected = torch.load(Path(self.save_root) / "optimizer.pt", map_location="cpu", weights_only=True)
            assert_optimizer_restored(self.optimizer.state_dict()["state"], expected["state"])
            # Native code deliberately resets lr from config; all other hyperparameters must agree.
            actual_groups = copy.deepcopy(self.optimizer.state_dict()["param_groups"])
            expected_groups = copy.deepcopy(expected["param_groups"])
            for group in actual_groups + expected_groups:
                group.pop("lr", None)
                group.pop("initial_lr", None)
            assert_optimizer_restored(actual_groups, expected_groups, "param_groups")
            del expected
        super().hook_before_train_loop()
        from importlib.metadata import version, PackageNotFoundError
        versions = {}
        for package in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "peft", "huggingface-hub"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        if self.accelerator.is_main_process:
            write_json_atomic(Path(self.save_root) / "streetscape_environment.json",
                              {"platform": platform.platform(), "python": platform.python_version(),
                               "packages": versions, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(self.device_torch) if self.device_torch.type == "cuda" else None,
                               "status": "server runtime started; acceptance not implied"})

    def _decode_for_cnn(self, clean):
        vae = self.sd.vae
        if vae is None:
            raise RuntimeError("VAE was released; auxiliary scoring requires a frozen decoder")
        # Keep it on the graph's device until the complete native train-loop call returns.
        vae.to(clean.device)
        vae.eval().requires_grad_(False)
        return self.sd.decode_latents(clean, device=clean.device, dtype=vae.dtype)

    def _score_for_cnn(self, rgb):
        if self._cnn_scorer is None:
            self._cnn_scorer = FrozenStreetscapeScorer.from_checkpoint(
                self.cnn_guidance.checkpoint, self.cnn_guidance.manifest).to(rgb.device)
        self._cnn_scorer.to(rgb.device).eval()
        return self._cnn_scorer(rgb)

    def calculate_loss(self, noise_pred, noise, noisy_latents, timesteps, batch,
                       mask_multiplier=1.0, prior_pred=None, **kwargs):
        base = super().calculate_loss(noise_pred=noise_pred, noise=noise, noisy_latents=noisy_latents,
                                      timesteps=timesteps, batch=batch, mask_multiplier=mask_multiplier,
                                      prior_pred=prior_pred, **kwargs)
        if not self.cnn_guidance.enabled:
            return base
        if (prior_pred is not None or batch.mask_tensor is not None or any(batch.get_is_reg_list())
                or any(value != 1.0 for value in batch.loss_multiplier_list)):
            raise ValueError("Unsupported weighted/masked/prior batch in CNN guidance")
        if self.cnn_guidance.active_at(int(self.step_num)):
            sigmas = scheduler_sigmas(self.sd.noise_scheduler, timesteps)
        else:
            sigmas = noise_pred.new_zeros(noise_pred.shape[0])
        auxiliary, metrics = guidance_loss(prediction=noise_pred, noisy_latents=noisy_latents,
                                           sigmas=sigmas, config=self.cnn_guidance, step=int(self.step_num),
                                           decode=self._decode_for_cnn, scorer=self._score_for_cnn)
        weighted = self.cnn_guidance.weight * auxiliary
        total = base + weighted
        if not torch.isfinite(base).all() or not torch.isfinite(total).all():
            raise ValueError("Non-finite base or combined loss")
        if os.getenv("STREETSCAPE_VERIFY_GRADIENTS") == "1" and auxiliary.requires_grad and not self._gradient_verified:
            # Server acceptance only: extra backward traversal, never enabled by production config.
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            gradients = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
            if not any(g is not None and bool((g != 0).any()) for g in gradients):
                raise RuntimeError("CNN auxiliary loss produced no nonzero LoRA gradients")
            if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("CNN auxiliary LoRA gradients are non-finite")
            self._gradient_verified = True
            write_json_atomic(Path(self.save_root) / "streetscape_gradient_check.json",
                              {"identity_sha256": self.research_identity_sha256, "step": int(self.step_num),
                               "finite_nonzero_lora_gradient": True})
        metrics.update({"loss/base": base, "loss/cnn_weighted": weighted,
                        "cnn/weight": base.new_tensor(self.cnn_guidance.weight)})
        # Only logging leaves the graph. Accumulate across native micro-batches, not just the last one.
        eligible_count = float(metrics["cnn/eligible_count"].detach().cpu())
        for key, value in metrics.items():
            number = float(value.detach().cpu())
            if key in ("cnn/raw_score_min", "cnn/raw_score_max"):
                previous = self._cnn_log_sums.get(key, number)
                self._cnn_log_sums[key] = min(previous, number) if key.endswith("min") else max(previous, number)
                self._cnn_log_weights[key] = 1.0
            elif key in ("cnn/eligible_count", "cnn/full_microbatch"):
                self._cnn_log_sums[key] = self._cnn_log_sums.get(key, 0.0) + number
                self._cnn_log_weights[key] = 1.0
            else:
                count = 1.0 if key.startswith("loss/") or key == "cnn/weight" else eligible_count
                self._cnn_log_sums[key] = self._cnn_log_sums.get(key, 0.0) + number * count
                self._cnn_log_weights[key] = self._cnn_log_weights.get(key, 0.0) + count
        self._cnn_log_count += 1
        return total

    def hook_train_loop(self, batch):
        self._cnn_log_sums, self._cnn_log_weights, self._cnn_log_count = {}, {}, 0
        vae_cpu = (self.cnn_guidance.enabled and self.sd.vae is not None
                   and self.sd.vae.device.type == "cpu")
        result = super().hook_train_loop(batch)
        if self._cnn_log_count:
            self.additional_logs.update({key: value / self._cnn_log_weights[key]
                                         for key, value in self._cnn_log_sums.items()})
        # Native synchronous backward has finished for ALL micro-batches at this boundary.
        if vae_cpu:
            self.sd.vae.to("cpu")
        return result

    def save(self, step=None):
        if not self.accelerator.is_main_process:
            return super().save(step)
        if self.is_grad_accumulation_step:
            raise ValueError("Native save is between optimizer updates: align saves/termination with accumulation boundaries")
        suffix = "" if step is None else f"_{step:09d}"
        self._saving_lora = Path(self.save_root) / f"{self.job.name}{suffix}.safetensors"
        try:
            return super().save(step)
        finally:
            self._saving_lora = None

    def post_save_hook(self, save_path):
        super().post_save_hook(save_path)
        lora = self._saving_lora
        optimizer = Path(self.save_root) / "optimizer.pt"
        if lora is None or not lora.is_file() or not optimizer.is_file():
            raise ValueError("Native LoRA/optimizer save incomplete; research resume record not published")
        # The upstream saver catches optimizer exceptions. Reject stale/partially saved optimizer files.
        saved = torch.load(optimizer, map_location="cpu", weights_only=True)
        assert_optimizer_restored(self.optimizer.state_dict()["state"], saved["state"])
        del saved
        from safetensors import safe_open
        with safe_open(str(lora), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("streetscape_identity_sha256") != self.research_identity_sha256:
            raise ValueError("Native LoRA save lacks research identity")
        write_json_atomic(Path(self.save_root) / "streetscape_latest.json",
                          {"identity_sha256": self.research_identity_sha256, "step": int(self.step_num),
                           "lora": {"file": lora.name, "sha256": file_sha256(lora)},
                           "optimizer": {"file": optimizer.name, "sha256": file_sha256(optimizer)}})
        self._resume_record = read_json(Path(self.save_root) / "streetscape_latest.json")
