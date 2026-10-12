"""Deterministic candidate validation and independent, group-safe LoRA split."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import shutil
from .contract import file_sha256, object_sha256, read_json, write_json_atomic

CELLS = {'joint_supportive', 'youth_leaning', 'older_leaning', 'joint_constrained'}


def split_candidates(rows, seed=20260909):
    ids = [r['image_id'] for r in rows]
    if len(rows) != 480 or len(set(ids)) != 480:
        raise ValueError('Require exactly 480 unique candidate IDs')
    result = []
    rank = lambda value: hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    for code in [f'T{i:02d}' for i in range(1, 9)]:
        subset = [dict(r) for r in rows if r['archetype_code'] == code]
        original = sorted([r for r in subset if r['source_origin'] == 'original160'], key=lambda r: rank(r['image_id']))
        added = [r for r in subset if r['source_origin'].startswith('added')]
        if len(subset) != 60 or len(original) != 20 or len(added) != 40:
            raise ValueError(f'Expected {code}: 20 original and 40 added')
        for index, row in enumerate(original):
            row['lora_data_role'] = 'validation' if index < 2 else 'test' if index < 4 else 'train'
            row['split_group'] = row.get('verified_duplicate_group') or row['image_id']
            result.append(row)
        groups = {}
        for row in added:
            group = f"{code}:{row['source_batch']}"
            groups.setdefault(group, []).append(row)
        if len(groups) != 10:
            raise ValueError(f'Expected ten added contrast groups for {code}')
        for index, group in enumerate(sorted(groups, key=rank)):
            values = groups[group]
            if len(values) != 4 or {r['contrast_cell'] for r in values} != CELLS:
                raise ValueError(f'Incomplete contrast group: {group}')
            for row in values:
                row['lora_data_role'] = 'validation' if index == 0 else 'test' if index == 1 else 'train'
                row['split_group'] = group
                result.append(row)
    memberships = {}
    for row in result:
        group = row.get('verified_duplicate_group') or row['split_group']
        memberships.setdefault(group, set()).add(row['lora_data_role'])
    if any(len(values) != 1 for values in memberships.values()):
        raise ValueError('Confirmed linked/duplicate group crosses splits; do not split silently')
    if {r['image_id'] for r in result} != set(ids):
        raise ValueError('Incomplete candidate coverage')
    return sorted(result, key=lambda r: r['image_id'])


def prepare_dataset(source, candidates, output, seed=20260909):
    from PIL import Image
    source, output = Path(source).resolve(), Path(output).resolve()
    with Path(candidates).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows = split_candidates(rows, seed)
    content_hashes = set()
    for row in rows:
        name = row['image_id']
        image, caption = source / f'{name}.png', source / f'{name}.txt'
        if not image.is_file() or not caption.is_file():
            raise ValueError(f'Missing candidate pair: {name}')
        digest = file_sha256(image)
        reference_digest = row['sha256']
        if digest in content_hashes:
            raise ValueError(f'Duplicate candidate content: {name}')
        provenance = 'original_file_bytes'
        if digest != reference_digest:
            # Tagger exports can losslessly re-encode PNG. Do not equate that to
            # matching bytes: prove every pixel against the untouched frozen source.
            reference = Path(row['source_path'])
            if not reference.is_file() or file_sha256(reference) != reference_digest:
                raise ValueError(f'Unverifiable changed image source: {name}')
            with Image.open(reference) as old, Image.open(image) as current:
                if old.size != current.size or old.mode != current.mode or old.tobytes() != current.tobytes():
                    raise ValueError(f'Candidate pixels changed: {name}')
            provenance = 'tagger_lossless_reencoding_pixel_verified'
        row.update(sha256=digest, original_file_sha256=reference_digest, pixel_provenance=provenance)
        content_hashes.add(digest)
        text = caption.read_text(encoding='utf-8-sig')
        if not text.strip() or '\ufffd' in text or '\x00' in text:
            raise ValueError(f'Invalid UTF-8 caption: {name}')
        with Image.open(image) as decoded:
            width, height = decoded.size
            decoded.verify()
        row.update(image=f"{row['lora_data_role']}/{name}.png",
                   caption=f"{row['lora_data_role']}/{name}.txt", caption_sha256=file_sha256(caption),
                   width=width, height=height)
    identity = {'schema_version': 1, 'seed': seed, 'count': 480,
                'source_candidate_sha256': file_sha256(candidates),
                'group_policy': 'added_batch_archetype_4_cells; existing_reviewed_distinct_originals',
                'pairs': [{key: row.get(key, '') for key in
                           ('image_id', 'archetype_code', 'source_batch', 'source_origin', 'contrast_cell',
                            'cnn_data_role', 'lora_data_role', 'split_group', 'sha256', 'caption_sha256',
                            'image', 'caption', 'width', 'height', 'existing_review_status', 'existing_review_note',
                            'original_file_sha256', 'pixel_provenance')}
                          for row in rows]}
    manifest = output / 'dataset_manifest.json'
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError('Existing dataset split/content differs; no overwrite')
    if output.exists() and any(output.iterdir()) and not manifest.exists():
        raise ValueError('Unbound dataset output; no overwrite')
    # Preflight every target before copying any file.
    for row in rows:
        for key, digest_key in [('image', 'sha256'), ('caption', 'caption_sha256')]:
            target = output / row[key]
            if target.exists() and file_sha256(target) != row[digest_key]:
                raise ValueError(f'Dataset target conflict: {target}')
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for key in ('image', 'caption'):
            target = output / row[key]
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copy2(source / target.name, target)
    write_json_atomic(manifest, identity)
    table = output / 'split.csv'
    with table.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(identity['pairs'][0]))
        writer.writeheader()
        writer.writerows(identity['pairs'])
    for row in identity['pairs']:
        for key, digest_key in [('image', 'sha256'), ('caption', 'caption_sha256')]:
            if file_sha256(output / row[key]) != row[digest_key]:
                raise ValueError('Copied data verification failed')
    return {'identity_sha256': object_sha256(identity), 'counts':
            {role: sum(r['lora_data_role'] == role for r in rows) for role in ['train', 'validation', 'test']}}
