"""Stress the actual native LoRA forward/backward without optimizer updates or saves."""
from __future__ import annotations
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class ProbeComplete(ValueError):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--root', type=Path, default=Path('/workspace/streetscape'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new immutable probe result')
    os.environ['STREETSCAPE_VERIFY_GRADIENTS'] = '1'
    import torch
    from toolkit.config import get_config
    from toolkit.job import get_job
    from extensions_built_in.streetscape_cnn.contract import read_json, write_json_atomic, image_id
    from extensions_built_in.streetscape_cnn.guidance import scheduler_sigmas
    from extensions_built_in.streetscape_cnn.trainer import module_fingerprint
    from extensions_built_in.streetscape_cnn.memory import clear_decoder_cache
    raw = copy.deepcopy(get_config(args.config))
    raw['config']['name'] = args.output.stem
    raw['meta']['research_status'] = 'gradient_memory_probe_zero_optimizer_updates'
    process = raw['config']['process'][0]
    # Only fixture conditioning disables dropout; native smoke config remains .05.
    for dataset in process['datasets']:
        dataset['caption_dropout_rate'] = 0.
    job = get_job(raw)
    trainer = job.process[0]
    original_predict, original_loss = trainer.predict_noise, trainer.calculate_loss
    captured, report = {}, {'optimizer_updates':0, 'cases':[]}

    def parameter_hash():
        result = hashlib.sha256()
        for name, param in trainer.network.named_parameters():
            if param.requires_grad:
                result.update(name.encode())
                result.update(param.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return result.hexdigest()

    def predict(*positional, **kwargs):
        if positional:
            raise ValueError('Native probe requires keyword prediction inputs')
        grid = trainer.sd.noise_scheduler
        old_sigma = scheduler_sigmas(grid, kwargs['timesteps']).reshape(-1,1,1,1)
        if (old_sigma == 0).any():
            raise ValueError('Cannot recover fixture noise at zero sigma')
        z = kwargs['noisy_latents']
        clean = kwargs['batch'].latents.to(z.device, z.dtype)
        noise = (z.float()-(1-old_sigma)*clean.float())/old_sigma
        index = int((grid.sigmas[:-1].float()-.25).abs().argmin())
        times = grid.timesteps[index].to(z.device).expand(z.shape[0])
        sigma = scheduler_sigmas(grid, times).reshape(-1,1,1,1)
        noisy = ((1-sigma)*clean.float()+sigma*noise).to(z.dtype)
        captured.update(noisy=noisy, noise=noise.to(z.dtype), times=times, sigma=float(sigma.flatten()[0]))
        kwargs.update(noisy_latents=noisy, timesteps=times)
        return original_predict(**kwargs)

    def loss(noise_pred, noise, noisy_latents, timesteps, batch, **kwargs):
        return original_loss(noise_pred, captured['noise'], captured['noisy'], captured['times'], batch, **kwargs)

    trainer.predict_noise, trainer.calculate_loss = predict, loss

    def probe(_unused):
        counts = read_json(args.root/'acceptance/preflight/token_counts.json')['rows']
        longest = max((row for row in counts if row['role'] == 'train'), key=lambda row:(row['conditioning_tokens'],row['image_id']))
        largest, long_batch, largest_area = None, None, -1
        for batch in trainer.data_loader:
            if len(batch.file_items) != 1:
                raise ValueError('Probe physical batch must be one')
            area = batch.latents.shape[-1]*batch.latents.shape[-2]
            if area > largest_area:
                largest, largest_area = batch, area
            if image_id(Path(batch.file_items[0].path)) == longest['image_id']:
                long_batch = batch
        if long_batch is None or largest is None:
            raise ValueError('Could not locate locked training stress fixtures')
        before = parameter_hash()
        vae_before = module_fingerprint(trainer.sd.vae)
        trainer.step_num = 8
        for label,batch in [('largest_bucket',largest),('longest_caption',long_batch)]:
            trainer._cnn_log_sums,trainer._cnn_log_weights,trainer._cnn_log_count = {},{},0
            trainer._pending_samples=[]
            trainer._cpu_saved_tensor_count=trainer._cpu_saved_tensor_bytes=0
            trainer._gradient_verified=False
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer._memory_monitor.begin_update(8)
            loss_value=trainer.train_single_accumulation(batch,accum_scale=.5)
            from toolkit.memory_management import sync_grad_transfers
            sync_grad_transfers()
            trainer._memory_monitor.check()
            gradients=[param.grad for param in trainer.network.parameters() if param.requires_grad and param.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients) or not any((g!=0).any() for g in gradients):
                raise ValueError('Real LoRA total gradients invalid')
            if trainer.cnn_guidance.enabled and not trainer._gradient_verified:
                raise ValueError('Actual auxiliary LoRA gradient not verified')
            report['cases'].append({'case':label,'image_id':image_id(Path(batch.file_items[0].path)),
                                    'latent_shape':list(batch.latents.shape),'actual_sigma':captured['sigma'],
                                    'loss':float(loss_value),'memory':dict(trainer._memory_monitor.peaks),
                                    'auxiliary_gradient_verified':trainer._gradient_verified,
                                    'cpu_packed_tensor_count':trainer._cpu_saved_tensor_count})
            clear_decoder_cache(trainer.sd.vae)
        if before != parameter_hash() or vae_before != module_fingerprint(trainer.sd.vae):
            raise ValueError('Probe changed LoRA or frozen VAE weights')
        if trainer._cnn_scorer is not None and module_fingerprint(trainer._cnn_scorer) != trainer._cnn_fingerprint:
            raise ValueError('Probe changed frozen CNN state')
        report.update(state='passed',lora_weights_unchanged=True,frozen_state_unchanged=True)
        write_json_atomic(args.output,report)
        raise ProbeComplete('Memory probe passed; zero optimizer updates; not a checkpoint')

    trainer.hook_train_loop = probe
    try:
        job.run()
    except ProbeComplete:
        print('Native maximum-bucket/longest-caption memory probe passed without updates',flush=True)
    except Exception as exc:
        report.update(state='failed',error_type=type(exc).__name__,message=str(exc))
        write_json_atomic(args.output,report)
        raise
    finally:
        if trainer._memory_monitor:
            trainer._memory_monitor.close()
        job.cleanup()

if __name__ == '__main__':
    main()
