"""Lock the authorized 1000-update, CNN-disabled early fitting observation."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import (
    file_sha256, read_json, write_json_atomic)


def select_prompts(dataset: Path, manifest: dict) -> list[dict]:
    rows = manifest['pairs']
    ids = [row['image_id'] for row in rows]
    if len(ids) != 480 or len(set(ids)) != 480:
        raise ValueError('Expected the locked 480 unique candidate IDs')
    counts = Counter((row['lora_data_role'], row['archetype_code']) for row in rows)
    expected = {(role, f'T{index:02d}'): count
                for role, count in [('train', 48), ('validation', 6), ('test', 6)]
                for index in range(1, 9)}
    if counts != expected:
        raise ValueError('Do not regenerate or change the 384/48/48 split')
    result = []
    for role in ['train', 'validation']:
        for index in range(1, 9):
            archetype = f'T{index:02d}'
            row = min((r for r in rows if r['lora_data_role'] == role
                       and r['archetype_code'] == archetype), key=lambda r: r['image_id'])
            caption = (dataset / row['caption']).resolve()
            if not caption.is_relative_to(dataset.resolve()):
                raise ValueError('Unsafe caption path')
            if file_sha256(caption) != row['caption_sha256']:
                raise ValueError(f'Caption changed: {row["image_id"]}')
            prompt = caption.read_text(encoding='utf-8-sig').strip()
            if not prompt:
                raise ValueError('Empty observation prompt')
            result.append({'image_id': row['image_id'], 'role': role,
                           'archetype_code': archetype, 'caption_sha256': row['caption_sha256'],
                           'prompt': prompt})
    return result


def build_configs(root: Path, prompts: list[dict], text_encoder: Path) -> dict:
    common = {
        'type': 'streetscape_cnn_trainer', 'training_folder': str(root / 'runs'),
        'training_seed': 20260909, 'device': 'cuda:0', 'log_dir': str(root / 'tensorboard'),
        'network': {'type': 'lora', 'linear': 32, 'linear_alpha': 32},
        'save': {'dtype': 'float32', 'save_every': 1000,
                 'max_step_saves_to_keep': 2, 'push_to_hub': False},
        'logging': {'log_every': 1, 'use_wandb': False},
        'datasets': [{'folder_path': str(root / 'dataset/train'), 'caption_ext': 'txt',
                      'shuffle_tokens': False, 'caption_dropout_rate': .05, 'num_repeats': 1,
                      'cache_latents_to_disk': True, 'cache_text_embeddings': True,
                      'resolution': [1024], 'num_workers': 2, 'cache_latents_num_workers': 2,
                      'flip_x': False, 'flip_y': False, 'random_crop': False,
                      'random_scale': False, 'square_crop': False}],
        'train': {'batch_size': 1, 'steps': 1000, 'lr': .0001, 'optimizer': 'adamw8bit',
                  'optimizer_params': {'weight_decay': .0001}, 'lr_scheduler': 'constant',
                  'train_unet': True, 'train_text_encoder': False, 'gradient_checkpointing': True,
                  'gradient_accumulation': 2, 'gradient_accumulation_steps': 1,
                  'noise_scheduler': 'flowmatch', 'timestep_type': 'linear',
                  'content_or_style': 'balanced', 'loss_type': 'mse', 'dtype': 'bf16',
                  'cache_text_embeddings': True, 'disable_sampling': False,
                  'skip_first_sample': False, 'force_first_sample': False,
                  'ema_config': {'use_ema': False}},
        'model': {'arch': 'krea2', 'name_or_path': 'krea/Krea-2-Raw',
                  'quantize': True, 'quantize_te': True, 'qtype': 'float8', 'qtype_te': 'float8',
                  'low_vram': True, 'layer_offloading': True,
                  'layer_offloading_transformer_percent': .5,
                  'layer_offloading_text_encoder_percent': 1.,
                  'model_kwargs': {'max_text_length': 512}},
        'local_model_assets': {'raw': '/model/ModelScope/krea/Krea-2-Raw/raw.safetensors',
                               'text_encoder': str(text_encoder),
                               'vae': '/model/ModelScope/krea/Krea-2-Raw',
                               'manifest': str(root / 'public_model_manifest.json')},
        'cnn_guidance': {'enabled': False}, 'cnn_memory_profile': 'vae_checkpoint',
        'research_memory_acceptance': True,
        'research_lr_schedule': {'enabled': False},
        'research_recording': {'enabled': True, 'per_sample_scores': True, 'archive_checkpoints': True},
        'sample': {'sampler': 'flowmatch', 'sample_every': 250, 'width': 768, 'height': 512,
                   'sample_steps': 30, 'guidance_scale': 4., 'seed': 42, 'walk_seed': False,
                   'neg': '', 'network_multiplier': 1., 'prompts': [r['prompt'] for r in prompts]}}
    configs = {}
    for name, smoke in [('krea2_streetscape_fit_probe_1000_v1', False),
                        ('krea2_streetscape_fit_probe_smoke4_v1', True)]:
        process = copy.deepcopy(common)
        if smoke:
            process['train']['steps'] = 4
            process['sample']['sample_every'] = 2
            process['sample']['prompts'] = [prompts[0]['prompt']]
        configs['fit_smoke4.json' if smoke else 'fit_probe_1000.json'] = {
            'job': 'extension', 'config': {'name': name, 'process': [process]},
            'meta': {'research_status': 'early_fitting_observation_not_cnn_effect_experiment',
                     'smoke': smoke}}
    return configs


def save_consistent(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Existing observation artifact differs: {path}')
        return
    write_json_atomic(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/streetscape'))
    parser.add_argument('--text-encoder-path', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / 'dataset/dataset_manifest.json'
    prompts = select_prompts(root / 'dataset', read_json(manifest_path))
    save_consistent(root / 'configs/fit_prompts.json', {
        'dataset_manifest_sha256': file_sha256(manifest_path), 'rows': prompts,
        'scope': '16 train/validation prompts; test excluded'})
    for name, config in build_configs(root, prompts, args.text_encoder_path).items():
        path = root / 'configs' / name
        save_consistent(path, config)
        print(path)


if __name__ == '__main__':
    main()
