"""Extract an owned result snapshot to a new directory and verify every manifest hash."""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, write_json_atomic

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if file_sha256(args.archive).lower() != args.sha256.lower():
        raise ValueError('Archive content hash differs')
    if args.output.exists():
        raise ValueError('Extract to a new owned directory')
    with tarfile.open(args.archive) as package:
        members = package.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or '\\' in member.name or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe snapshot member')
        roots = {PurePosixPath(m.name).parts[0] for m in members}
        if len(roots) != 1:
            raise ValueError('Snapshot requires exactly one root')
        package.extractall(args.output, filter='data')
    root = args.output / roots.pop()
    manifest = json.loads((root / 'result_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path) != expected:
            raise ValueError(f'Payload hash differs: {name}')
    write_json_atomic(args.output / 'verified.json', {'archive_sha256': args.sha256,
                                                     'payload_files_verified': len(manifest['files'])})
    print(json.dumps({'root': str(root), 'files_verified': len(manifest['files'])}), flush=True)
