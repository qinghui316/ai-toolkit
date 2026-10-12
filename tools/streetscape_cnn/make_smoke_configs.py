"""Generate only the approved 24-step engineering configurations."""
import argparse
import copy
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import read_json, write_json_atomic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/workspace/streetscape'))
    p.add_argument('--fallback', action='store_true', help='Approved OOM-only batch 1 / accumulation 2 profile')
    args = p.parse_args()
    root = args.root.resolve()
    profile = 'b1ga2' if args.fallback else 'b2ga1'
    common = {
        'type': 'streetscape_cnn_trainer', 'training_folder': str(root / 'runs'),
        'training_seed': 20260909, 'device': 'cuda:0', 'log_dir': str(root / 'tensorboard'),
        'network': {'type': 'lora', 'linear': 32, 'linear_alpha': 32},
        'save': {'dtype': 'float32', 'save_every': 4, 'max_step_saves_to_keep': 100, 'push_to_hub': False},
        'logging': {'log_every': 1, 'use_wandb': False},
        'datasets': [{'folder_path': str(root / 'dataset/train'), 'caption_ext': 'txt',
                      'shuffle_tokens': False, 'caption_dropout_rate': .05, 'num_repeats': 1,
                      'cache_latents_to_disk': True, 'cache_text_embeddings': True,
                      'resolution': [1024], 'num_workers': 2, 'cache_latents_num_workers': 2,
                      'flip_x': False, 'flip_y': False, 'random_crop': False, 'random_scale': False,
                      'square_crop': False}],
        'train': {'batch_size': 1 if args.fallback else 2, 'steps': 24, 'lr': .0001,
                  'optimizer': 'adamw8bit', 'optimizer_params': {'weight_decay': .0001},
                  'lr_scheduler': 'constant', 'train_unet': True, 'train_text_encoder': False,
                  'gradient_checkpointing': True, 'gradient_accumulation': 2 if args.fallback else 1,
                  'gradient_accumulation_steps': 1, 'noise_scheduler': 'flowmatch',
                  'timestep_type': 'linear', 'content_or_style': 'balanced', 'loss_type': 'mse',
                  'dtype': 'bf16', 'cache_text_embeddings': True, 'disable_sampling': True,
                  'ema_config': {'use_ema': False}},
        'model': {'arch': 'krea2', 'name_or_path': 'krea/Krea-2-Raw', 'quantize': True,
                  'quantize_te': True, 'qtype': 'float8', 'qtype_te': 'float8', 'low_vram': True,
                  'layer_offloading': True, 'layer_offloading_transformer_percent': .50,
                  'layer_offloading_text_encoder_percent': 1.0, 'model_kwargs': {'max_text_length': 512}},
        'local_model_assets': {'raw': '/model/ModelScope/krea/Krea-2-Raw/raw.safetensors',
                               'text_encoder': '/model/HuggingFace/Qwen/Qwen3-VL-4B-Instruct',
                               'vae': '/model/ModelScope/krea/Krea-2-Raw',
                               'manifest': str(root / 'public_model_manifest.json')},
        'research_lr_schedule': {'enabled': True, 'stages': [{'start_step': 0, 'lr': .0001},
                                                            {'start_step': 8, 'lr': .00015},
                                                            {'start_step': 16, 'lr': .0001}]},
        'research_recording': {'enabled': True, 'per_sample_scores': True, 'archive_checkpoints': True},
        'sample': {'sampler': 'flowmatch', 'prompts': []}}
    for objective in ['base', 'cnn']:
        process = copy.deepcopy(common)
        process['cnn_guidance'] = {'enabled': False} if objective == 'base' else {
            'enabled': True, 'checkpoint': str(root / 'cnn/final.pt'),
            'manifest': str(root / 'cnn/model_manifest.json'), 'weight': .05,
            'start_step': 8, 'sigma_min': .05, 'sigma_max': .50, 'every_n_steps': 1}
        value = {'job': 'extension', 'config': {'name': f'krea2_streetscape_{objective}_smoke_1024_{profile}_v1',
                                               'process': [process]},
                 'meta': {'research_status': 'engineering_smoke_only_not_production'}}
        path = root / 'configs' / f'{objective}_{profile}.json'
        if path.exists() and read_json(path) != value:
            raise ValueError('Existing engineering configuration differs; no overwrite')
        write_json_atomic(path, value)
        print(path)


if __name__ == '__main__':
    main()
