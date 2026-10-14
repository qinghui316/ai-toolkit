"""Derive the registered CNN comparison from an immutable, completed base config."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, read_json
from tools.streetscape_cnn.make_fit_probe_configs import save_consistent

TIERS = {1: ('vae_checkpoint', .5), 2: ('vae_checkpoint_cpu', .5),
         3: ('vae_checkpoint_cpu', 1.)}


def derive_configs(baseline: dict, root: Path) -> dict[str, dict]:
    if len(baseline['config']['process']) != 1:
        raise ValueError('Expected one base process')
    base = baseline['config']['process'][0]
    expected = {'batch_size': 1, 'gradient_accumulation': 2,
                'gradient_accumulation_steps': 1, 'steps': 1000, 'lr': .0001,
                'dtype': 'bf16', 'optimizer': 'adamw8bit', 'lr_scheduler': 'constant'}
    if (base['cnn_guidance'] != {'enabled': False}
            or any(base['train'].get(k) != v for k, v in expected.items())
            or base['training_seed'] != 20260909
            or base['network'] != {'type': 'lora', 'linear': 32, 'linear_alpha': 32}
            or base['model']['name_or_path'] != 'krea/Krea-2-Raw'
            or base['research_lr_schedule'] != {'enabled': False}
            or len(base['sample']['prompts']) != 16
            or len(base['datasets']) != 1
            or Path(base['datasets'][0]['folder_path']) != root/'dataset/train'
            or base['datasets'][0]['resolution'] != [1024]):
        raise ValueError('Baseline differs from the approved 1000-update observation')
    result = {}
    for tier, (profile, offload) in TIERS.items():
        for smoke in (False, True):
            value = copy.deepcopy(baseline)
            name = f'krea2_streetscape_cnn_probe_{"smoke24" if smoke else "1000"}_p{tier}_v1'
            value['config']['name'] = name
            process = value['config']['process'][0]
            process['cnn_guidance'] = {'enabled': True, 'checkpoint': str(root/'cnn/final.pt'),
                'manifest': str(root/'cnn/model_manifest.json'), 'weight': .05,
                'start_step': 0, 'sigma_min': .05, 'sigma_max': .50, 'every_n_steps': 1}
            process['cnn_memory_profile'] = profile
            process['research_gpu_memory_reserve_gib'] = 2
            process['model']['layer_offloading_transformer_percent'] = offload
            if smoke:
                process['train'].update(steps=24, disable_sampling=True, skip_first_sample=True)
            value['meta'] = {'research_status': 'registered_cnn_comparison_not_quality_claim',
                'tier': tier, 'smoke': smoke, 'baseline_run': baseline['config']['name'],
                'initialization': 'fresh_RAW_no_pretrained_adapter'}
            result[f'cnn_{"smoke24" if smoke else "1000"}_p{tier}_v1.json'] = value
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline-config', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    configs = derive_configs(read_json(args.baseline_config), root)
    for name, value in configs.items():
        save_consistent(root/'configs'/name, value)
    save_consistent(root/'configs/cnn_probe_registration_v1.json', {
        'baseline_config_sha256': file_sha256(args.baseline_config),
        'configs': {name: file_sha256(root/'configs'/name) for name in configs},
        'changes': ['cnn_guidance', 'cnn_memory_profile', 'GPU safety reserve',
                    'registered Transformer offload tier', 'independent name and smoke length'],
        'scientific_parameters': {'updates': 1000, 'weight': .05, 'start_step': 0,
                                 'sigma_range': [.05, .50], 'every_n_steps': 1}})
    print('Registered six independent configs; baseline unchanged')


if __name__ == '__main__':
    main()
