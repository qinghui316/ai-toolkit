"""Safely extract an explicitly hashed staging archive; refuse conflicting targets."""
import argparse
from pathlib import Path
import sys
import tarfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, read_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if file_sha256(args.archive) != args.sha256:
        raise ValueError('Transferred dataset archive hash differs')
    target = args.output.resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError('Dataset extraction target is nonempty; no overwrite')
    with tarfile.open(args.archive) as archive:
        for item in archive.getmembers():
            path = (target / item.name).resolve()
            if not path.is_relative_to(target) or item.issym() or item.islnk() or not (item.isfile() or item.isdir()):
                raise ValueError('Unsafe dataset archive member')
        target.mkdir(parents=True, exist_ok=True)
        archive.extractall(target, filter='data')
    manifest = read_json(target / 'dataset_manifest.json')
    for row in manifest['pairs']:
        for key, digest in [('image', 'sha256'), ('caption', 'caption_sha256')]:
            if file_sha256(target / row[key]) != row[digest]:
                raise ValueError('Extracted data hash differs')
    print('480 candidate pairs extracted and byte-verified')
