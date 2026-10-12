"""Snapshot explicit engineering outputs, export plot-ready CSVs and checksum all files."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, write_json_atomic, LABEL_COLUMNS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/workspace/streetscape'))
    p.add_argument('--output', type=Path, required=True, help='New snapshot folder, outside source outputs')
    args = p.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Snapshot already exists; no overwrite')
    output.mkdir(parents=True)
    # No original images/captions, CNN weights, public model weights, or credentials.
    def snapshot_file(source, target):
        path = Path(source)
        if 'checkpoints' in path.parts and path.suffix in {'.pt', '.safetensors'}:
            # Published checkpoint folders are immutable. Hard-link their large
            # payloads on this filesystem instead of requiring a second full copy.
            os.link(source, target)
            return target
        return shutil.copy2(source, target)
    def exclude_live_binaries(directory, names):
        path = Path(directory)
        if path.parent == root / 'runs':
            return [name for name in names if name.endswith(('.pt', '.safetensors', '.pending'))]
        return []
    for name in ['runs', 'configs', 'acceptance', 'tensorboard']:
        source = root / name
        if source.exists():
            shutil.copytree(source, output / name, copy_function=snapshot_file, ignore=exclude_live_binaries)
    if (root / 'public_model_manifest.json').exists():
        shutil.copy2(root / 'public_model_manifest.json', output / 'public_model_manifest.json')
    manifest = root / 'dataset/dataset_manifest.json'
    if manifest.exists():
        shutil.copy2(manifest, output / 'dataset_manifest.json')
        shutil.copy2(root / 'dataset/split.csv', output / 'split.csv')
    for run in (output / 'runs').glob('*'):
        records = run / 'records'
        for stem in ['steps', 'samples', 'checkpoints', 'failed_samples']:
            source = records / f'{stem}.jsonl'
            if not source.exists():
                continue
            rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
            flattened = []
            for row in rows:
                value = dict(row)
                if stem in ('samples', 'failed_samples'):
                    for key, prefix in [('raw_scores', 'raw'), ('clipped_scores', 'clipped')]:
                        scores = value.get(key)
                        for index, column in enumerate(LABEL_COLUMNS):
                            value[f'{prefix}_{column}'] = scores[index] if scores is not None else None
                flattened.append({key: json.dumps(item, ensure_ascii=False) if isinstance(item, (list, dict)) else item
                                  for key, item in value.items()})
            if flattened:
                with (records / f'{stem}.csv').open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in flattened))))
                    writer.writeheader()
                    writer.writerows(flattened)
    files = {file.relative_to(output).as_posix(): file_sha256(file) for file in sorted(output.rglob('*')) if file.is_file()}
    write_json_atomic(output / 'result_manifest.json', {'schema_version': 1, 'files': files,
                                                       'scope': 'engineering outputs, not final model quality evidence'})
    archive = output.with_suffix('.tar')
    if archive.exists():
        raise ValueError('Snapshot archive already exists')
    with tarfile.open(archive, 'w') as package:
        package.add(output, arcname=output.name)
    print(json.dumps({'archive': str(archive), 'archive_sha256': file_sha256(archive),
                      'file_count': len(files), 'bytes': archive.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
