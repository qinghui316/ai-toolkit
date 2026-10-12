"""Resolve immutable input assets on the server, before native model loading.

Uses Hugging Face's normal cache. Nothing is downloaded merely by importing this module.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .contract import file_sha256, read_json
import hashlib

MODEL_SUFFIXES = {".safetensors", ".bin", ".json", ".txt", ".model", ".tiktoken", ".jinja"}
PINNED_REVISIONS = {'raw': ('krea/Krea-2-Raw', '6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603'),
                    'text_encoder': ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17'),
                    'vae': ('krea/Krea-2-Raw', '6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603')}
PINNED_WEIGHTS = {'raw.safetensors': 'f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7',
                  'model-00001-of-00002.safetensors': '30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9',
                  'model-00002-of-00002.safetensors': '046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6',
                  'vae/diffusion_pytorch_model.safetensors': 'ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08'}


def resolve_local_assets(process):
    result = copy.deepcopy(process)
    local = result.get('local_model_assets')
    if not isinstance(local, dict) or set(local) != {'raw', 'text_encoder', 'vae', 'manifest'}:
        raise ValueError('local_model_assets requires raw, text_encoder, vae and manifest')
    manifest = read_json(local['manifest'])
    if manifest.get('schema_version') != 1 or set(manifest.get('components', {})) != set(PINNED_REVISIONS):
        raise ValueError('Invalid public model asset manifest')
    assets = {}
    for role, (repo, revision) in PINNED_REVISIONS.items():
        component = manifest['components'][role]
        if (component.get('repo'), component.get('revision')) != (repo, revision):
            raise ValueError(f'Unapproved {role} provenance')
        root = Path(local[role])
        if role == 'raw':
            if not root.is_file() or root.name != 'raw.safetensors':
                raise ValueError('Explicit RAW safetensors file required')
            root = root.parent
        files = {}
        for name, expected in component['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe model manifest path')
            path = root / relative
            if not path.is_file():
                raise ValueError(f'Missing {role} asset {name}')
            actual = file_sha256(path)
            pinned = PINNED_WEIGHTS.get(name)
            if pinned and actual != pinned:
                raise ValueError(f'Pinned {role} weight mismatch: {name}')
            if expected.get('sha256'):
                if actual != expected['sha256']:
                    raise ValueError(f'Model SHA-256 mismatch: {name}')
            else:
                content = path.read_bytes()
                git_blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
                if git_blob != expected.get('git_blob'):
                    raise ValueError(f'Model config/tokenizer git-blob mismatch: {name}')
            files[name] = actual
        required = {'raw': {'raw.safetensors'},
                    'text_encoder': {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                                     'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'},
                    'vae': {'vae/config.json', 'vae/diffusion_pytorch_model.safetensors'}}[role]
        if not required <= files.keys():
            raise ValueError(f'Incomplete {role} component manifest')
        assets[role] = {'repo': repo, 'revision': revision, 'files': files}
    result['model']['name_or_path'] = str(Path(local['raw']).absolute())
    kwargs = result['model'].setdefault('model_kwargs', {})
    kwargs['text_encoder_path'] = str(Path(local['text_encoder']).absolute())
    kwargs['vae_path'] = str(Path(local['vae']).absolute())
    return result, assets


def inventory_asset(path: Path | str) -> dict:
    path = Path(path)
    if path.is_file():
        return {path.name: file_sha256(path)}
    if not path.is_dir():
        raise ValueError(f"Missing model asset: {path}")
    files = {p.relative_to(path).as_posix(): file_sha256(p) for p in sorted(path.rglob("*"))
             if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES
             and not any(part.startswith(".") for part in p.relative_to(path).parts)}
    if not files:
        raise ValueError(f"Model asset folder is empty: {path}")
    return files


def resolve_krea_assets(process: dict, previous: dict | None = None) -> tuple[dict, dict]:
    """Pin RAW/TE/VAE contents and force native loaders to use those exact files.

Previous snapshot revisions are reused when resuming, rather than silently fetching new main.
The custom trainer is intentionally a Krea 2 RAW T2I extension, not another generic trainer.
"""
    previous = previous or {}
    result = copy.deepcopy(process)
    model = result["model"]
    if model.get("arch") != "krea2" or model.get("name_or_path") != "krea/Krea-2-Raw":
        raise ValueError("streetscape_cnn_trainer uses canonical Krea 2 RAW; use sd_trainer for other models")
    kwargs = model.setdefault("model_kwargs", {})
    if kwargs.get("checkpoint_filename", "raw.safetensors") != "raw.safetensors":
        raise ValueError("RAW checkpoint_filename must be raw.safetensors")
    if process.get('local_model_assets') is not None:
        return resolve_local_assets(process)
    from huggingface_hub import hf_hub_download, snapshot_download
    raw_revision = previous.get("raw", {}).get("revision")
    raw = Path(hf_hub_download("krea/Krea-2-Raw", "raw.safetensors", revision=raw_revision))
    assets = {"raw": {"repo": "krea/Krea-2-Raw", "revision": raw.parent.name,
                      "files": inventory_asset(raw)}}
    # Keep the .safetensors cache symlink name: resolving it yields an extensionless blob,
    # which the native Krea loader would mistakenly interpret as a Hub repository.
    model["name_or_path"] = str(raw.absolute())
    for label, key, default, patterns in (
        ("text_encoder", "text_encoder_path", "Qwen/Qwen3-VL-4B-Instruct",
         ["*.safetensors", "*.bin", "*.json", "*.txt", "*.model", "*.tiktoken"]),
        ("vae", "vae_path", "Qwen/Qwen-Image", ["vae/*.safetensors", "vae/*.bin", "vae/*.json"]),
    ):
        source = kwargs.get(key, default)
        if Path(source).exists():
            path = Path(source).resolve()
            assets[label] = {"files": inventory_asset(path)}
        else:
            revision = previous.get(label, {}).get("revision")
            path = Path(snapshot_download(source, revision=revision, allow_patterns=patterns))
            assets[label] = {"repo": source, "revision": path.name, "files": inventory_asset(path)}
        kwargs[key] = str(path)
    return result, assets
