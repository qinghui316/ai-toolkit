"""Real small FP32 single-frame VAE/CNN loss and input-gradient equivalence."""
import argparse
import gc
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/workspace/streetscape'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Parity result already exists')
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from toolkit.models.v2.vae.qwen_image import QwenImageVAE, QwenImageVAEHolderMixin
    from toolkit.util.qwen_vae_gradient_checkpointing import patch_qwen_vae_gradient_checkpointing
    from extensions_built_in.streetscape_cnn.memory import configure_decoder, scoring_memory_context, clear_decoder_cache
    from extensions_built_in.streetscape_cnn.scorer import FrozenStreetscapeScorer
    from extensions_built_in.streetscape_cnn.trainer import module_fingerprint
    from extensions_built_in.streetscape_cnn.contract import write_json_atomic
    from types import SimpleNamespace
    patch_qwen_vae_gradient_checkpointing()
    # An FP32 reference must not silently execute cuDNN convolutions in TF32.
    # These flags are confined to this isolated parity process, not training.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(20260909)
    vae = QwenImageVAE.from_pretrained('/model/ModelScope/krea/Krea-2-Raw', subfolder='vae', torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    scorer = FrozenStreetscapeScorer.from_checkpoint(args.root/'cnn/final.pt', args.root/'cnn/model_manifest.json').to('cuda').eval()
    holder = SimpleNamespace(vae=vae, vae_device_torch=torch.device('cuda'), vae_torch_dtype=torch.float32,
                             vae_decode_tiled_on_low_vram=True, model_config=SimpleNamespace(low_vram=True))
    before = (module_fingerprint(vae), module_fingerprint(scorer))
    with Image.open(args.root/'dataset/train/CS_T01_002.png') as image:
        rgb = to_tensor(image.convert('RGB').resize((384,256))).unsqueeze(0).to('cuda')
    # Fixture construction alone is no-grad; every scored decode remains differentiable.
    with torch.no_grad():
        latent = QwenImageVAEHolderMixin.encode_images(holder, [rgb[0]*2-1])
    rows, reference = [], None
    for profile in ('legacy', 'legacy', 'vae_checkpoint', 'vae_checkpoint_cpu', 'vae_checkpoint'):
        clear_decoder_cache(vae)
        configure_decoder(vae, profile)
        x = latent.clone().requires_grad_()
        packed = []
        with scoring_memory_context(profile, lambda device, size: packed.append((device, size))):
            decoded = QwenImageVAEHolderMixin.decode_latents(holder, x)
            loss = ((10-scorer(((decoded+1)/2).clamp(0,1))['inclusive_all6'])/9).mean()
        gradient = torch.autograd.grad(loss, x)[0]
        if not torch.isfinite(gradient).all() or not (gradient != 0).any():
            raise RuntimeError('Reference and optimized latent gradients must be finite/nonzero')
        values = [decoded.detach().cpu(), loss.detach().cpu(), gradient.detach().cpu()]
        errors = [0.,0.,0.] if reference is None else [float((a-b).double().norm()/b.double().norm().clamp_min(1e-12)) for a,b in zip(values, reference)]
        if reference is None:
            reference = values
        if errors[0] > 1e-5 or errors[1] > 1e-5 or errors[2] > 1e-4:
            write_json_atomic(args.output, {'state':'failed','cases':rows,
                                           'failed_profile':profile,'errors':errors,'tf32':False})
            raise RuntimeError(f'FP32 memory-policy parity rejected: {profile}: {errors}')
        if profile == 'vae_checkpoint_cpu' and (not packed or any(device != 'cpu' for device,size in packed)):
            raise RuntimeError('Native saved tensors were not actually stored on CPU')
        rows.append({'profile':profile,'relative_l2_output_loss_gradient':errors,
                     'finite_nonzero_gradient':True,'cpu_packed_tensor_count':len(packed)})
        del decoded, loss, gradient, x, values
        gc.collect()
    if before != (module_fingerprint(vae), module_fingerprint(scorer)) or any(p.grad is not None or p.requires_grad for m in (vae,scorer) for p in m.parameters()):
        raise RuntimeError('Frozen VAE/CNN state changed')
    write_json_atomic(args.output, {'state':'passed','cases':rows,'frozen_state_unchanged':True,
                                   'size':[256,384],'tf32':False,'cudnn_deterministic':True})
    print('Real FP32 VAE/CNN memory-policy parity and repeat-cache checks passed', flush=True)

if __name__ == '__main__':
    main()
