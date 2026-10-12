"""Locally upload a verified owned archive to the explicitly named PRIVATE HF repo."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, write_json_atomic

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--sha256',required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--repo',default='qinghui316/streetscape-krea2-engineering-v1')
    args=p.parse_args()
    if file_sha256(args.archive).lower()!=args.sha256.lower():
        raise ValueError('Local archive does not match verified transfer')
    from huggingface_hub import HfApi
    api=HfApi()
    if api.model_info(args.repo).private is not True:
        raise ValueError('Engineering checkpoint repository must be private')
    target='archives/'+args.archive.name
    existing=api.get_paths_info(args.repo,[target],expand=True)
    if existing:
        info=existing[0]
        if info.lfs is None or info.lfs.sha256!=args.sha256.lower():
            raise ValueError('Existing remote archive differs; no overwrite')
    else:
        api.upload_file(path_or_fileobj=args.archive,path_in_repo=target,repo_id=args.repo,
                        commit_message='Archive immutable engineering outputs, not formal training')
    api.upload_file(path_or_fileobj=args.manifest,path_in_repo='manifests/'+args.archive.stem+'.json',repo_id=args.repo)
    revision=api.model_info(args.repo).sha
    remote=api.get_paths_info(args.repo,[target],expand=True,revision=revision)[0]
    if remote.size!=args.archive.stat().st_size or remote.lfs is None or remote.lfs.sha256!=args.sha256.lower():
        raise ValueError('Uploaded archive size or SHA-256 differs')
    write_json_atomic(args.receipt,{'repo':args.repo,'private':True,'revision':revision,
                                   'path':target,'sha256':args.sha256.lower(),'bytes':remote.size,
                                   'verification':'HF LFS SHA-256 and size matched; no full download roundtrip'})
    print(json.dumps({'repo':args.repo,'revision':revision,'remote_hash_verified':True}),flush=True)
