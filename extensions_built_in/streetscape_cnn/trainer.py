"""Opt-in SDTrainer subclass. Native training/backward/cache/optimizer remain in charge."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import platform
import subprocess
import hashlib
import shutil
import time
import sys
from fnmatch import fnmatch

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from .assets import resolve_krea_assets
from .contract import (GuidanceConfig, OBJECTIVE_VERSION, UPSTREAM_COMMIT, check_or_create_identity,
                       file_sha256, object_sha256, portable_process_config, read_json,
                       scan_caption_pairs, validate_cnn_files, validate_process,
                       validate_resume_pair, validate_explicit_training_settings, write_json_atomic, source_sha256)
from .guidance import guidance_loss, scheduler_sigmas
from .scorer import FrozenStreetscapeScorer
from .runtime import ResearchLRSchedule, ResearchRecorder, ResearchAbort, recording_config, stop_requested
from .contract import image_id


def module_fingerprint(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
        self.research_schedule = ResearchLRSchedule.parse(config.get('research_lr_schedule'), config['train'])
        self.recording_settings = recording_config(config.get('research_recording'))
        self._recorder = None
        self._completed_updates = self._image_presentations = self._save_sequence = 0
        self._last_successful_step = None
        self._pending_samples = []
        self._vae_fingerprint = self._cnn_fingerprint = None
        self._gradient_norm = None
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
            if 'trainer_state' in self._resume_record:
                state = torch.load(Path(self.save_root) / self._resume_record['trainer_state']['file'],
                                   map_location='cpu', weights_only=True)
                if state['native_step'] != int(self.step_num):
                    raise ResearchAbort('Research scheduler/counter progress mismatch')
                self._completed_updates = state['completed_updates']
                self._image_presentations = state['image_presentations']
                self._save_sequence = state['save_sequence']
                self._last_successful_step = int(self.step_num) - 1
                self._gradient_verified = state['gradient_verified']
                self._vae_fingerprint, self._cnn_fingerprint = state['vae_fingerprint'], state['cnn_fingerprint']
                self.lr_scheduler.load_state_dict(state['scheduler'])
        super().hook_before_train_loop()
        if self.recording_settings['enabled']:
            self._recorder = ResearchRecorder(self.save_root, self.research_identity_sha256)
            self._recorder.status('running', native_step=int(self.step_num),
                                  completed_updates=self._completed_updates)
            optimizer = getattr(self.optimizer, 'optimizer', self.optimizer)
            def before_update(opt, args, kwargs):
                square = 0.0
                for group in opt.param_groups:
                    for parameter in group['params']:
                        if parameter.grad is not None:
                            if not torch.isfinite(parameter.grad).all():
                                raise ResearchAbort('Non-finite LoRA gradient; optimizer update rejected')
                            square += float(parameter.grad.detach().float().norm().cpu()) ** 2
                self._gradient_norm = square ** .5
            def after_update(opt, args, kwargs):
                self._completed_updates += 1
            optimizer.register_step_pre_hook(before_update)
            optimizer.register_step_post_hook(after_update)
            if self.cnn_guidance.enabled:
                if self.sd.vae is None:
                    raise ResearchAbort('Frozen decoder was released during caching')
                actual = module_fingerprint(self.sd.vae)
                if self._vae_fingerprint is not None and actual != self._vae_fingerprint:
                    raise ResearchAbort('Frozen VAE changed across resume')
                self._vae_fingerprint = actual
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
            # Building then loading a ResNet must not consume the generator's RNG.
            with torch.random.fork_rng(devices=[]):
                self._cnn_scorer = FrozenStreetscapeScorer.from_checkpoint(
                    self.cnn_guidance.checkpoint, self.cnn_guidance.manifest).to(rgb.device)
        self._cnn_scorer.to(rgb.device).eval()
        if getattr(self, '_recorder', None) and not getattr(self, '_cnn_fingerprint_checked', False):
            actual = module_fingerprint(self._cnn_scorer)
            if self._cnn_fingerprint is not None and actual != self._cnn_fingerprint:
                raise ResearchAbort('Frozen CNN changed across resume')
            self._cnn_fingerprint = actual
            self._cnn_fingerprint_checked = True
        return self._cnn_scorer(rgb)

    def calculate_loss(self, noise_pred, noise, noisy_latents, timesteps, batch,
                       mask_multiplier=1.0, prior_pred=None, **kwargs):
        base = super().calculate_loss(noise_pred=noise_pred, noise=noise, noisy_latents=noisy_latents,
                                      timesteps=timesteps, batch=batch, mask_multiplier=mask_multiplier,
                                      prior_pred=prior_pred, **kwargs)
        if not torch.isfinite(base).all():
            raise ResearchAbort('Non-finite native base loss; no zero-loss substitution')
        if not self.cnn_guidance.enabled:
            if getattr(self, 'recording_settings', {}).get('enabled'):
                self._record_batch(noise_pred, timesteps, batch, {})
                self._accumulate_metrics({'loss/base': base, 'loss/cnn': base.new_zeros(()),
                                          'loss/cnn_weighted': base.new_zeros(()), 'loss/total': base,
                                          'cnn/eligible_count': base.new_zeros(()),
                                          'cnn/full_microbatch': base.new_tensor(noise_pred.shape[0])})
            return base
        if (prior_pred is not None or batch.mask_tensor is not None or any(batch.get_is_reg_list())
                or any(value != 1.0 for value in batch.loss_multiplier_list)):
            raise ValueError("Unsupported weighted/masked/prior batch in CNN guidance")
        if self.cnn_guidance.active_at(int(self.step_num)):
            sigmas = scheduler_sigmas(self.sd.noise_scheduler, timesteps)
        else:
            sigmas = noise_pred.new_zeros(noise_pred.shape[0])
        scored = {}
        observer = None
        if getattr(self, 'recording_settings', {}).get('per_sample_scores'):
            observer = lambda eligible, result: scored.update(self._observe_scores(eligible, result))
        auxiliary, metrics = guidance_loss(prediction=noise_pred, noisy_latents=noisy_latents,
                                           sigmas=sigmas, config=self.cnn_guidance, step=int(self.step_num),
                                           decode=self._decode_for_cnn, scorer=self._score_for_cnn,
                                           observer=observer)
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
        self._record_batch(noise_pred, timesteps, batch, scored)
        metrics.update({"loss/base": base, "loss/cnn_weighted": weighted, 'loss/total': total,
                        "cnn/weight": base.new_tensor(self.cnn_guidance.weight)})
        self._accumulate_metrics(metrics)
        return total

    def _accumulate_metrics(self, metrics):
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

    def hook_train_loop(self, batch):
        self._cnn_log_sums, self._cnn_log_weights, self._cnn_log_count = {}, {}, 0
        vae_cpu = (self.cnn_guidance.enabled and self.sd.vae is not None
                   and self.sd.vae.device.type == "cpu")
        self._pending_samples = []
        self._step_started = time.monotonic()
        rate = getattr(self, 'research_schedule', ResearchLRSchedule()).rate(int(self.step_num))
        if rate is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = rate
        rates = [float(group['lr']) for group in self.optimizer.param_groups]
        self._updates_before_step = getattr(self, '_completed_updates', 0)
        try:
            result = super().hook_train_loop(batch)
            if self._cnn_log_count:
                self.additional_logs.update({key: value / self._cnn_log_weights[key]
                                             for key, value in self._cnn_log_sums.items()
                                             if self._cnn_log_weights[key] > 0})
            if getattr(self, '_recorder', None):
                if self._completed_updates != self._updates_before_step + 1:
                    raise ResearchAbort('Native optimizer did not perform exactly one update')
                self._last_successful_step = int(self.step_num)
                count = len(self._pending_samples)
                self._image_presentations += count
                metrics = {key: value / self._cnn_log_weights[key]
                           for key, value in self._cnn_log_sums.items() if self._cnn_log_weights[key] > 0}
                row = {'native_step': int(self.step_num), 'completed_updates': self._completed_updates,
                       'image_presentations': self._image_presentations, 'learning_rates': rates,
                       'gradient_norm': self._gradient_norm, 'seconds': time.monotonic() - self._step_started,
                       'cuda_peak_allocated': torch.cuda.max_memory_allocated(self.device_torch),
                       'cuda_peak_reserved': torch.cuda.max_memory_reserved(self.device_torch),
                       'native_loss': float(result['loss']), **metrics}
                self._recorder.append('steps', row)
                for sample in self._pending_samples:
                    self._recorder.append('samples', {**sample, 'completed_update': self._completed_updates})
            return result
        except BaseException as exc:
            if getattr(self, '_recorder', None):
                self._recorder.status('failed', native_step=int(self.step_num),
                                      completed_updates=self._completed_updates,
                                      error_type=type(exc).__name__, message=str(exc))
                for sample in self._pending_samples:
                    self._recorder.append('failed_samples', sample)
            if isinstance(exc, torch.cuda.OutOfMemoryError) or (isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()):
                raise ResearchAbort('Research CUDA memory exhaustion; no batch skip or parameter fallback') from exc
            raise
        finally:
            active_exception = sys.exc_info()[0] is not None
            if vae_cpu and self.sd.vae is not None:
                try:
                    self.sd.vae.to('cpu')
                except Exception:
                    if not active_exception:
                        raise

    def hook_after_model_load(self):
        super().hook_after_model_load()
        if not self.recording_settings['enabled']:
            return
        report = {}
        from toolkit.util.quantize import is_quantized_tensor
        for role, module, enabled in [('transformer', self.sd.model, self.model_config.quantize),
                                       ('text_encoder', self.sd.text_encoder, self.model_config.quantize_te)]:
            modules = module if isinstance(module, list) else [module]
            quantized, missing = 0, []
            excludes = self.sd.get_quantization_exclude_modules() if role == 'transformer' else []
            for component in modules:
                if component is None:
                    continue
                for name, layer in component.named_modules():
                    if not isinstance(layer, torch.nn.Linear):
                        continue
                    if any(fnmatch(name, pattern) for pattern in excludes):
                        continue
                    if is_quantized_tensor(layer.weight) or str(layer.weight.dtype).startswith('torch.float8'):
                        quantized += 1
                    else:
                        missing.append(name)
            report[role] = {'quantized_linears': quantized, 'unquantized_linears': missing}
            if enabled and (quantized == 0 or (role == 'transformer' and missing)):
                write_json_atomic(Path(self.save_root) / 'quantization_check.json', report)
                raise ResearchAbort(f'Actual {role} FP8 quantization incomplete')
        write_json_atomic(Path(self.save_root) / 'quantization_check.json', report)

    def _record_batch(self, noise_pred, timesteps, batch, scored):
        if not getattr(self, 'recording_settings', {}).get('enabled'):
            return
        sigmas = scheduler_sigmas(self.sd.noise_scheduler, timesteps).detach().cpu().tolist()
        items = batch.file_items
        if len(items) != len(sigmas):
            raise ResearchAbort('Sample metadata count mismatch')
        for index, item in enumerate(items):
            row = {'native_step': int(self.step_num), 'microbatch_index': self._cnn_log_count,
                   'image_id': image_id(Path(item.path)), 'sigma': sigmas[index],
                   'scored': index in scored, 'raw_scores': None, 'clipped_scores': None,
                   'inclusive_all6': None, 'raw_inclusive_all6': None,
                   'youth_quality': None, 'older_quality': None, 'group_gap': None,
                   'minimum_safety': None, 'clipped_fraction': None}
            row.update(scored.get(index, {}))
            self._pending_samples.append(row)

    def _observe_scores(self, eligible, result):
        indices = eligible.detach().cpu().nonzero().flatten().tolist()
        values = {key: value.detach().cpu().tolist() for key, value in result.items()
                  if key != 'clipped_fraction'}
        rows = {}
        for offset, index in enumerate(indices):
            raw = values['raw_scores'][offset]
            rows[index] = {key: value[offset] for key, value in values.items()}
            rows[index]['clipped_fraction'] = sum(x < 1 or x > 10 for x in raw) / len(raw)
        return rows

    def end_step_hook(self):
        super().end_step_hook()
        if getattr(self, '_recorder', None) and stop_requested(self.save_root, self.research_identity_sha256):
            self.save(step=int(self.step_num))
            requested = Path(self.save_root) / 'research_stop_request.json'
            consumed = Path(self.save_root) / 'records' / f'stop_{self.step_num:09d}_{self._save_sequence:04d}.json'
            if consumed.exists():
                raise ResearchAbort('Stop audit target already exists')
            requested.rename(consumed)
            self._recorder.status('stopped', native_step=int(self.step_num), completed_updates=self._completed_updates)
            raise KeyboardInterrupt('Identity-bound user stop after a complete native update/save')

    def done_hook(self):
        super().done_hook()
        if getattr(self, '_recorder', None):
            # Native BaseSDTrainProcess deletes self.sd before invoking done_hook.
            # Frozen-module checks run during the final save, while modules exist.
            if not getattr(self, '_frozen_checks_complete', False):
                raise ResearchAbort('Final frozen-state check was not completed')
            self._recorder.status('completed', native_step=int(self.step_num),
                                  completed_updates=self._completed_updates,
                                  image_presentations=self._image_presentations)

    def save(self, step=None):
        if not self.accelerator.is_main_process:
            return super().save(step)
        if self.is_grad_accumulation_step:
            raise ValueError("Native save is between optimizer updates: align saves/termination with accumulation boundaries")
        original_step = int(self.step_num)
        if step is None and getattr(self, '_recorder', None):
            checks = {}
            for name, module, expected in [('vae', self.sd.vae, self._vae_fingerprint),
                                           ('cnn', self._cnn_scorer, self._cnn_fingerprint)]:
                if expected:
                    if module is None or module_fingerprint(module) != expected:
                        raise ResearchAbort(f'Frozen {name} parameters/buffers changed')
                    if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                        raise ResearchAbort(f'Frozen {name} acquired gradients')
                    checks[name] = True
            write_json_atomic(Path(self.save_root) / 'frozen_state_check.json', checks)
            self._frozen_checks_complete = True
        if getattr(self, '_recorder', None) and self._last_successful_step is not None:
            # Upstream periodic saves use a zero-based current-step index, while
            # end_step_hook uses next-step progress. Always publish next-step metadata.
            self.step_num = self._last_successful_step + 1
            if step is not None:
                step = int(self.step_num)
        suffix = "" if step is None else f"_{step:09d}"
        self._saving_lora = Path(self.save_root) / f"{self.job.name}{suffix}.safetensors"
        try:
            return super().save(step)
        finally:
            self._saving_lora = None
            self.step_num = original_step

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
        record = {"identity_sha256": self.research_identity_sha256, "step": int(self.step_num),
                  "lora": {"file": lora.name, "sha256": file_sha256(lora)},
                  "optimizer": {"file": optimizer.name, "sha256": file_sha256(optimizer)}}
        if getattr(self, '_recorder', None):
            self._save_sequence += 1
            state = {'native_step': int(self.step_num), 'completed_updates': self._completed_updates,
                     'image_presentations': self._image_presentations, 'save_sequence': self._save_sequence,
                     'scheduler': self.lr_scheduler.state_dict(), 'gradient_verified': self._gradient_verified,
                     'vae_fingerprint': self._vae_fingerprint, 'cnn_fingerprint': self._cnn_fingerprint}
            state_path = Path(self.save_root) / 'research_training_state.pt'
            temporary = state_path.with_suffix('.pending')
            torch.save(state, temporary)
            temporary.replace(state_path)
            record['trainer_state'] = {'file': state_path.name, 'sha256': file_sha256(state_path)}
            if self.recording_settings['archive_checkpoints']:
                archive = Path(self.save_root) / 'checkpoints' / f'step_{self.step_num:09d}_save_{self._save_sequence:04d}'
                archive.mkdir(parents=True, exist_ok=False)
                for item in (lora, optimizer, state_path, Path(self.save_root) / 'streetscape_identity.json'):
                    shutil.copy2(item, archive / item.name)
                write_json_atomic(archive / 'checkpoint_manifest.json', record)
                self._recorder.append('checkpoints', {'native_step': int(self.step_num),
                                                     'archive': archive.relative_to(self.save_root).as_posix(), **record})
        write_json_atomic(Path(self.save_root) / 'streetscape_latest.json', record)
        self._resume_record = read_json(Path(self.save_root) / "streetscape_latest.json")
