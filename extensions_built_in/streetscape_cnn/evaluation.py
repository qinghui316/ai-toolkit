"""Read-only complete-run auditing and descriptive, paired preview evaluation."""
from __future__ import annotations
from collections import defaultdict
import math
from pathlib import Path
import re
import statistics

from .contract import (file_sha256, object_sha256, read_json, validate_resume_pair)

PREVIEW_NAME = re.compile(r'^\d+__(\d{9})_(\d+)\.(jpg|jpeg|png|webp)$', re.I)
METRICS = ('inclusive_all6', 'raw_inclusive_all6', 'youth_quality', 'older_quality',
           'group_gap', 'minimum_safety', 'clipped_fraction')


def journal(path):
    return [read_json_line(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def read_json_line(line):
    import json
    return json.loads(line)


def audit_run(run: Path, updates: int, cnn: bool, limit_bytes=None,
              expected_previews=None, require_resume=False) -> dict:
    status = read_json(run/'research_status.json')
    rows = journal(run/'records/steps.jsonl')
    samples = journal(run/'records/samples.jsonl')
    identity = read_json(run/'streetscape_identity.json')
    identity_hash = object_sha256(identity)
    pair = validate_resume_pair(run, identity_hash)
    if (status.get('state') != 'completed' or status['completed_updates'] != updates
            or status['image_presentations'] != 2*updates or pair['step'] != updates
            or [r['completed_updates'] for r in rows] != list(range(1, updates+1))
            or [r['native_step'] for r in rows] != list(range(updates))
            or len(samples) != 2*updates):
        raise ValueError('Incomplete, repeated or noncanonical successful updates/presentations')
    by_step = defaultdict(list)
    for sample in samples:
        by_step[sample['native_step']].append(sample)
        expected = cnn and .05 <= sample['sigma'] <= .5
        if bool(sample['scored']) != expected:
            raise ValueError('Scoring schedule/sigma gating differs from registration')
        if expected:
            for field in ('raw_scores', 'clipped_scores'):
                scores = sample[field]
                if len(scores) != 12 or not all(math.isfinite(x) for x in scores):
                    raise ValueError('Invalid per-sample scores')
            if not all(1 <= x <= 10 for x in sample['clipped_scores']):
                raise ValueError('Scoring clipping contract differs')
        elif sample['raw_scores'] is not None or sample['clipped_scores'] is not None:
            raise ValueError('Unscored sample contains fabricated scores')
    for index, row in enumerate(rows):
        if row['identity_sha256'] != identity_hash or row['learning_rates'] != [.0001]:
            raise ValueError('Identity or constant learning rate changed')
        if row['image_presentations'] != 2*(index+1):
            raise ValueError('Presentation counters do not match native accumulation')
        if any(not math.isfinite(row[k]) for k in
               ('native_loss', 'loss/base', 'loss/cnn', 'loss/cnn_weighted', 'loss/total', 'gradient_norm')):
            raise ValueError('Non-finite loss/gradient')
        batch = by_step[index]
        if sorted(s['microbatch_index'] for s in batch) != [0, 1]:
            raise ValueError('Native microbatch coverage changed')
        if any(s['completed_update'] != index+1 for s in batch):
            raise ValueError('Per-sample update mapping changed')
        if int(row['cnn/eligible_count']) != sum(s['scored'] for s in batch):
            raise ValueError('Eligible sample counts do not match loss records')
        if cnn:
            if (not math.isclose(row['loss/cnn_weighted'], .05*row['loss/cnn'], abs_tol=1e-7)
                    or not math.isclose(row['loss/total'], row['loss/base']+row['loss/cnn_weighted'], abs_tol=1e-6)):
                raise ValueError('Registered combined objective changed')
        if limit_bytes is not None and max(row['cuda_peak_reserved'], row['sampled_process_gpu_peak']) > limit_bytes:
            raise ValueError('GPU memory acceptance gate exceeded')
        if row.get('host_available_min', 2**63) < 2*2**30:
            raise ValueError('Host memory safety reserve exhausted')
    if cnn:
        if (read_json(run/'frozen_state_check.json') != {'cnn': True, 'vae': True}
                or not any(s['scored'] for s in samples)
                or not read_json(run/'streetscape_gradient_check.json')['finite_nonzero_lora_gradient']
                or not read_json(run/'cnn_memory_actual.json')['decoder_checkpointing']):
            raise ValueError('Frozen-state/auxiliary-gradient/decoder acceptance incomplete')
    if require_resume:
        checkpoints = journal(run/'records/checkpoints.jsonl')
        if not any(c['native_step'] == 8 for c in checkpoints):
            raise ValueError('Missing complete interruption checkpoint at update 8')
    originals = [p for p in (run/'samples').glob('*') if p.is_file()
                 and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
    if expected_previews is not None and len(originals) != expected_previews:
        raise ValueError('Complete preview count differs')
    return {'state': 'passed', 'run': run.name, 'updates': updates,
        'image_presentations': 2*updates, 'scored_samples': sum(s['scored'] for s in samples),
        'main_preview_images': len(originals), 'latest_pair': pair,
        'process_gpu_peak_bytes': max(r['sampled_process_gpu_peak'] for r in rows),
        'reserved_peak_bytes': max(r['cuda_peak_reserved'] for r in rows),
        'first100_mean_loss': statistics.mean(r['loss/base'] for r in rows[:100]),
        'last100_mean_loss': statistics.mean(r['loss/base'] for r in rows[-100:]),
        'cnn_enabled': cnn, 'quality_or_generalization_claim': False}


def preview_inventory(run: Path, prompts: list[dict]) -> list[dict]:
    from PIL import Image
    if (len(prompts) != 16 or len({p['image_id'] for p in prompts}) != 16
            or [(p['role'], p['archetype_code']) for p in prompts]
               != [(role, f'T{i:02d}') for role in ('train', 'validation') for i in range(1, 9)]):
        raise ValueError('Expected the same sixteen locked train/validation prompts')
    identity = read_json(run/'streetscape_identity.json')
    sample = identity['configuration']['sample']
    expected = {'sampler': 'flowmatch', 'width': 768, 'height': 512, 'sample_steps': 30,
                'guidance_scale': 4., 'seed': 42, 'walk_seed': False, 'neg': '', 'network_multiplier': 1.}
    if (any(sample.get(k) != v for k, v in expected.items())
            or sample['prompts'] != [p['prompt'] for p in prompts]):
        raise ValueError('Preview conditioning/sampler identity differs')
    groups = defaultdict(dict)
    for path in sorted((run/'samples').iterdir()):
        if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
            continue
        match = PREVIEW_NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f'Unmapped main preview: {path.name}')
        native, index = int(match[1]), int(match[2])
        if index in groups[native]:
            raise ValueError('Repeated prompt index in a preview batch')
        groups[native][index] = path
    if sorted(groups) != [0, 250, 500, 750, 1000]:
        raise ValueError('Expected all five complete preview milestones')
    updates = {r['native_step']: r['completed_updates'] for r in journal(run/'records/steps.jsonl')}
    result = []
    for native, files in sorted(groups.items()):
        if sorted(files) != list(range(16)):
            raise ValueError('Incomplete preview batch; no selective scoring')
        count = 0 if native == 0 else 1000 if native == 1000 else updates[native]
        for index, path in sorted(files.items()):
            with Image.open(path) as image:
                if image.size != (768, 512):
                    raise ValueError('Unexpected original preview dimensions')
                image.verify()
            result.append({'run': run.name, 'native_step': native, 'completed_updates': count,
                'prompt_index': index, 'prompt_id': prompts[index]['image_id'],
                'role': prompts[index]['role'], 'archetype_code': prompts[index]['archetype_code'],
                'prompt': prompts[index]['prompt'], 'seed': 42,
                'image': path.relative_to(run).as_posix(), 'image_sha256': file_sha256(path)})
    return result


def paired_rows(left, right, left_update, right_update):
    def select(rows, update):
        result = {}
        for row in rows:
            if row['completed_updates'] != update:
                continue
            key = (row['prompt_id'], row['seed'])
            if key in result:
                raise ValueError('Duplicate paired observation')
            result[key] = row
        return result
    a, b = select(left, left_update), select(right, right_update)
    if len(a) != 16 or a.keys() != b.keys():
        raise ValueError('Comparison requires sixteen identical prompt/seed pairs')
    result = []
    for key in a:
        x, y = a[key], b[key]
        if any(x[k] != y[k] for k in ('prompt', 'role', 'archetype_code')):
            raise ValueError('Paired conditioning differs')
        result.append({'prompt_id': key[0], 'seed': key[1], 'role': x['role'],
            'archetype_code': x['archetype_code'], 'left_run': x['run'], 'right_run': y['run'],
            'left_update': left_update, 'right_update': right_update,
            **{f'left_{k}': x[k] for k in METRICS}, **{f'right_{k}': y[k] for k in METRICS},
            **{f'delta_{k}': y[k]-x[k] for k in METRICS}})
    return result


def descriptive_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['role'], 'all')].append(row)
        groups[(row['role'], row['archetype_code'])].append(row)
    return [{'role': role, 'archetype_code': code, 'pairs': len(items),
        **{f'mean_delta_{k}': statistics.mean(r[f'delta_{k}'] for r in items) for k in METRICS}}
        for (role, code), items in sorted(groups.items())]
