"""Verify the approved GPU stack, native imports, data/token limits, and model identities."""
from __future__ import annotations
import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, read_json, write_json_atomic
from extensions_built_in.streetscape_cnn.assets import resolve_krea_assets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--asset-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if platform.system() != 'Linux' or sys.version_info[:2] != (3, 12):
        raise ValueError('Expected isolated Linux Python 3.12')
    expected = {'torch': '2.9.1', 'torchvision': '0.24.1', 'torchaudio': '2.9.1',
                'torchao': '0.15.0', 'torchcodec': '0.9.0', 'bitsandbytes': '0.50.2'}
    packages = {name: version(name) for name in expected}
    if any(packages[name].split('+')[0] != value for name, value in expected.items()):
        raise ValueError(f'Installed core versions differ: {packages}')
    import torch
    import bitsandbytes as bnb
    from torchao.quantization import quantize_, Float8WeightOnlyConfig
    from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    if not torch.cuda.is_available() or torch.version.cuda != '12.8':
        raise ValueError('Approved CUDA stack is not active')
    layer = torch.nn.Linear(128, 128, bias=False).to('cuda', dtype=torch.bfloat16).requires_grad_(False)
    quantize_(layer, Float8WeightOnlyConfig())
    x = torch.randn(2, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().square().mean().backward()
    if not torch.isfinite(x.grad).all() or not (x.grad != 0).any():
        raise ValueError('FP8 native input backward failed')
    parameter = torch.nn.Parameter(torch.ones(8192, device='cuda'))
    optimizer = bnb.optim.AdamW8bit([parameter], lr=1e-4, eps=1e-6, weight_decay=1e-4)
    parameter.sum().backward()
    optimizer.step()
    if not torch.isfinite(parameter).all() or torch.equal(parameter, torch.ones_like(parameter)):
        raise ValueError('AdamW8Bit GPU update failed')
    root = args.dataset.resolve()
    manifest = read_json(root / 'dataset_manifest.json')
    for row in manifest['pairs']:
        for key, digest in [('image', 'sha256'), ('caption', 'caption_sha256')]:
            if file_sha256(root / row[key]) != row[digest]:
                raise ValueError(f'Transferred data hash mismatch: {row["image_id"]}')
    from transformers import AutoTokenizer
    from extensions_built_in.diffusion_models.krea2.src.text_encoder import (
        PROMPT_TEMPLATE_ENCODE_PREFIX, PROMPT_TEMPLATE_ENCODE_START_IDX,
        PROMPT_TEMPLATE_ENCODE_SUFFIX)
    te_path = '/model/HuggingFace/Qwen/Qwen3-VL-4B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(te_path, local_files_only=True)
    counts, invalid = [], []
    for row in manifest['pairs']:
        text = (root / row['caption']).read_text(encoding='utf-8-sig')
        caption_tokens = len(tokenizer.encode(text))
        native_tokens = len(tokenizer.encode(PROMPT_TEMPLATE_ENCODE_PREFIX + text))
        count = {'image_id': row['image_id'], 'role': row['lora_data_role'],
                 'caption_tokens': caption_tokens, 'native_prefix_and_caption_tokens': native_tokens,
                 'conditioning_tokens': native_tokens + len(tokenizer.encode(PROMPT_TEMPLATE_ENCODE_SUFFIX))
                                        - PROMPT_TEMPLATE_ENCODE_START_IDX}
        counts.append(count)
        if caption_tokens > 512 or native_tokens > 512 + PROMPT_TEMPLATE_ENCODE_START_IDX:
            invalid.append(count)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output / 'token_counts.json', {'rows': counts, 'over_limit': invalid})
    if invalid:
        raise ValueError(f'{len(invalid)} captions exceed the approved token limit; captions unchanged')
    process = {'model': {'arch': 'krea2', 'name_or_path': 'krea/Krea-2-Raw'},
               'local_model_assets': {'raw': '/model/ModelScope/krea/Krea-2-Raw/raw.safetensors',
                                      'text_encoder': te_path,
                                      'vae': '/model/ModelScope/krea/Krea-2-Raw',
                                      'manifest': str(args.asset_manifest)}}
    _, assets = resolve_krea_assets(process)
    write_json_atomic(args.output / 'public_assets_verified.json', assets)
    write_json_atomic(args.output / 'gpu_preflight.json',
                      {'packages': packages, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                       'bf16': torch.cuda.is_bf16_supported(), 'fp8_input_gradient': True,
                       'adamw8bit_update': True, 'native_imports': [Krea2Model.__name__, SDTrainer.__name__],
                       'max_caption_tokens': max(row['caption_tokens'] for row in counts), 'data_count': len(counts)})
    print('GPU/dependency/data/token/official asset preflight passed', flush=True)


if __name__ == '__main__':
    main()
