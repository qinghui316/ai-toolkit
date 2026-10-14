"""Create a positive-allowlist metadata/preview transfer; no model binaries or datasets."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, write_json_atomic

ALLOWED = {'.json', '.jsonl', '.csv', '.yaml', '.yml', '.jpg', '.jpeg', '.png',
           '.svg', '.pdf', '.log', '.html'}


def collect(root: Path, runs: list[str], reports: list[Path]):
    selected = {}
    def add_tree(folder, prefix):
        if not folder.is_dir():
            raise ValueError(f'Missing records directory: {folder}')
        for file in sorted(folder.rglob('*')):
            relative = file.relative_to(folder)
            if (not file.is_file() or file.suffix.lower() not in ALLOWED
                    or any(part.startswith('.') for part in relative.parts)):
                continue
            if file.is_symlink() or not file.resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe metadata source')
            selected[f'{prefix}/{relative.as_posix()}'] = file
    for name in runs:
        if Path(name).name != name or name in ('.', '..') or '\\' in name:
            raise ValueError('Expected explicit run basenames')
        add_tree(root/'runs'/name, f'runs/{name}')
    for report in reports:
        path = report.resolve()
        if not path.is_relative_to(root/'acceptance'):
            raise ValueError('Report must be under the project acceptance directory')
        add_tree(path, path.relative_to(root).as_posix())
    for relative in ('configs/fit_prompts.json', 'configs/fit_probe_1000.json',
                     'configs/cnn_probe_registration_v1.json', 'dataset/dataset_manifest.json',
                     'dataset/split.csv', 'public_model_manifest.json'):
        path = root/relative
        if path.is_file():
            selected[relative] = path
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--report-dir', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True, help='New .tar outside selected directories')
    args = p.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    receipt = output.with_suffix('.json')
    if output.suffix != '.tar' or output.exists() or receipt.exists():
        raise ValueError('Use a new .tar/receipt pair; no overwrite')
    selected = collect(root, args.run, args.report_dir)
    hashes = {name: file_sha256(path) for name, path in selected.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'x') as archive:
        for name, path in selected.items():
            archive.add(path, arcname=name, recursive=False)
    if any(file_sha256(path) != hashes[name] for name, path in selected.items()):
        raise ValueError('Records changed during transfer snapshot; receipt not published')
    write_json_atomic(receipt, {'archive_sha256': file_sha256(output), 'files': hashes,
                               'model_weights_included': False, 'datasets_included': False})
    print(str(output), str(receipt), flush=True)


if __name__ == '__main__':
    main()
