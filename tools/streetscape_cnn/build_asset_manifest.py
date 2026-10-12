"""Fetch pinned official file identities, never model weights or credentials."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.assets import PINNED_REVISIONS, MODEL_SUFFIXES
from extensions_built_in.streetscape_cnn.contract import write_json_atomic, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from huggingface_hub import HfApi
    components = {}
    for role, (repo, revision) in PINNED_REVISIONS.items():
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError('Official revision mismatch')
        files = {}
        for item in info.siblings:
            name = item.rfilename
            selected = (name == 'raw.safetensors' if role == 'raw' else
                        name.startswith('vae/') if role == 'vae' else '/' not in name)
            if selected and Path(name).suffix in MODEL_SUFFIXES:
                lfs = getattr(item, 'lfs', None)
                files[name] = {'sha256': getattr(lfs, 'sha256', None),
                               'git_blob': getattr(item, 'blob_id', None), 'bytes': item.size}
        components[role] = {'repo': repo, 'revision': revision, 'files': files}
    value = {'schema_version': 1, 'components': components}
    if args.output.exists() and read_json(args.output) != value:
        raise ValueError('Existing asset manifest differs; refusing overwrite')
    write_json_atomic(args.output, value)
    print(args.output)


if __name__ == '__main__':
    main()
