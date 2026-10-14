"""Score every original preview offline; never update weights or alter image pixels."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import (
    LABEL_COLUMNS, file_sha256, read_json, write_json_atomic)
from extensions_built_in.streetscape_cnn.evaluation import (
    METRICS, preview_inventory, paired_rows, descriptive_summary)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse empty comparison table')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(module):
    import torch
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, action='append', required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists() or len(args.run_dir) not in (1, 2):
        raise ValueError('Use a new output and one baseline or two paired runs')
    prompts = read_json(args.prompts)['rows']
    inventories = [preview_inventory(run, prompts) for run in args.run_dir]
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from extensions_built_in.streetscape_cnn.scorer import FrozenStreetscapeScorer
    scorer = FrozenStreetscapeScorer.from_checkpoint(args.checkpoint, args.manifest).to(args.device)
    before = fingerprint(scorer)
    rows_by_run = []
    for run, inventory in zip(args.run_dir, inventories):
        rows = []
        for item in inventory:
            path = run/item['image']
            with Image.open(path) as image:
                rgb = to_tensor(image.convert('RGB')).unsqueeze(0).to(args.device)
            # Only offline evaluation is no-grad. The training scorer remains differentiable.
            with torch.inference_mode():
                values = scorer(rgb)
            raw = values['raw_scores'][0].cpu().tolist()
            clipped = values['clipped_scores'][0].cpu().tolist()
            row = {**item, **{k: float(values[k].reshape(-1)[0].cpu()) for k in METRICS},
                   **{f'raw_{k}': v for k, v in zip(LABEL_COLUMNS, raw)},
                   **{f'clipped_{k}': v for k, v in zip(LABEL_COLUMNS, clipped)}}
            if file_sha256(path) != item['image_sha256']:
                raise ValueError('Preview changed while scoring')
            rows.append(row)
        rows_by_run.append(rows)
    if fingerprint(scorer) != before or any(p.requires_grad or p.grad is not None for p in scorer.parameters()):
        raise ValueError('Offline scoring changed frozen CNN state')
    args.output.mkdir(parents=True)
    all_rows = [r for rows in rows_by_run for r in rows]
    write_csv(args.output/'preview_scores.csv', all_rows)
    write_json_atomic(args.output/'preview_scores.json', {'rows': all_rows,
        'checkpoint_sha256': file_sha256(args.checkpoint), 'manifest_sha256': file_sha256(args.manifest),
        'prompts_sha256': file_sha256(args.prompts), 'preprocess': 'FP32 bilinear antialias 224x336 ImageNet',
        'original_pixels_modified': False, 'single_seed': 42, 'statistical_or_quality_claim': False})
    summaries = {}
    for rows in rows_by_run:
        name = rows[0]['run']
        paired = paired_rows(rows, rows, 0, 1000)
        write_csv(args.output/f'{name}_zero_to_1000.csv', paired)
        summaries[f'{name}:0_to_1000'] = descriptive_summary(paired)
    if len(rows_by_run) == 2:
        for milestone in (0, 251, 501, 751, 1000):
            paired = paired_rows(*rows_by_run, milestone, milestone)
            write_csv(args.output/f'base_vs_cnn_update_{milestone:04d}.csv', paired)
            summaries[f'base_vs_cnn:{milestone}'] = descriptive_summary(paired)
    duplicates = {}
    for row in all_rows:
        duplicates.setdefault(row['image_sha256'], []).append(
            [row['run'], row['completed_updates'], row['prompt_id']])
    write_json_atomic(args.output/'comparison_summary.json', {
        'comparisons': summaries, 'exact_duplicate_groups': [v for v in duplicates.values() if len(v) > 1],
        'interpretation': 'Descriptive single-seed diagnostic; same training scorer, no significance/generalization claim.',
        'human_structure_prompt_match_diversity_review': 'pending'})
    print(f'Scored {len(all_rows)} original previews; frozen state unchanged', flush=True)


if __name__ == '__main__':
    main()
