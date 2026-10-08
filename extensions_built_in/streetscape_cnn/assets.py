"""Resolve immutable input assets on the server, before native model loading.

Uses Hugging Face's normal cache. Nothing is downloaded merely by importing this module.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .contract import file_sha256

MODEL_SUFFIXES = {".safetensors", ".bin", ".json", ".txt", ".model", ".tiktoken"}


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
    from huggingface_hub import hf_hub_download, snapshot_download

    previous = previous or {}
    result = copy.deepcopy(process)
    model = result["model"]
    if model.get("arch") != "krea2" or model.get("name_or_path") != "krea/Krea-2-Raw":
        raise ValueError("streetscape_cnn_trainer uses canonical Krea 2 RAW; use sd_trainer for other models")
    kwargs = model.setdefault("model_kwargs", {})
    if kwargs.get("checkpoint_filename", "raw.safetensors") != "raw.safetensors":
        raise ValueError("RAW checkpoint_filename must be raw.safetensors")
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
