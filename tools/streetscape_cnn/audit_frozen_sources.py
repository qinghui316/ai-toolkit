"""Verify untouched local frozen sources by hashes, without displaying score labels."""
import argparse
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, read_json, write_json_atomic


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.project.resolve()
    candidate_file = root / '建筑街道训练集/caption_candidates_480_v1.csv'
    with candidate_file.open(encoding='utf-8-sig', newline='') as stream:
        candidates = list(csv.DictReader(stream))
    originals = {r['image_id']: r for r in candidates}
    for row in candidates:
        if file_sha256(row['source_path']).lower() != row['sha256'].lower():
            raise ValueError(f'Frozen image changed: {row["image_id"]}')
    final = root / 'cnn_training/results/resnet18_shared_delta_dev400_final_v1'
    frozen = read_json(final / 'model_manifest.json')
    for name, key in [('checkpoints/final.pt', 'model_sha256'), ('config.json', 'config_sha256')]:
        if file_sha256(final / name) != frozen[key]:
            raise ValueError('Frozen CNN asset changed')
    expansion = root / 'dataset_expansion/expansion_v0.1'
    development = read_json(expansion / 'development_400/development_manifest.json')
    for name, key in [('cnn_labels.csv', 'labels_sha256'), ('selection.csv', 'selection_sha256')]:
        if file_sha256(expansion / 'development_400' / name).lower() != development[key].lower():
            raise ValueError('Frozen development labels or selection changed')
    seals = {}
    for file in sorted((expansion / 'batches').glob('batch*/sealed_test/seal_manifest.json')):
        value = read_json(file)
        for name, expected in value['test_image_sha256'].items():
            if file_sha256(originals[name]['source_path']).lower() != expected.lower():
                raise ValueError('Sealed image changed')
        hashes = {file_sha256(f).lower() for f in file.parent.iterdir() if f.is_file() and f != file}
        for key in ['raw_responses_sha256', 'test_labels_sha256']:
            if value[key].lower() not in hashes:
                raise ValueError(f'Sealed payload changed: {value["batch"]}/{key}')
        seals[value['batch']] = file_sha256(file)
    if len(seals) != 10:
        raise ValueError('Expected ten seal manifests')
    write_json_atomic(args.output, {'original_images_verified': len(candidates), 'seals_verified': seals,
                                    'cnn_model_sha256': frozen['model_sha256'],
                                    'cnn_config_sha256': frozen['config_sha256'],
                                    'development_hashes_verified': True,
                                    'scope': 'integrity only; no scoring, release, or label display'})
    print('480 frozen originals, ten seals, CNN and development identities verified')
