"""Load a saved adapter and sample through native sd_trainer with ZERO optimizer steps."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Filled smoke YAML supplying base model/network/dataset settings")
    parser.add_argument("export", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New EMPTY acceptance output directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, required=True, help="Inference sampler steps, not training steps")
    parser.add_argument("--cfg", type=float, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--identity", type=Path, help="Training streetscape_identity.json; defaults to export's directory")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Export inference acceptance belongs on the Linux server")
    if (not args.export.is_file() or args.steps < 1 or min(args.width, args.height) < 16
            or args.width % 16 or args.height % 16 or not math.isfinite(args.cfg) or args.cfg <= 0):
        parser.error("Require existing export and positive inference dimensions/steps")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory; do not overwrite acceptance results")
    from toolkit.config import get_config
    from toolkit.job import get_job
    from extensions_built_in.streetscape_cnn.contract import file_sha256, write_json_atomic, read_json, object_sha256
    from extensions_built_in.streetscape_cnn.assets import resolve_krea_assets
    from safetensors import safe_open
    identity_path = args.identity or args.export.parent / "streetscape_identity.json"
    identity = read_json(identity_path)
    with safe_open(str(args.export), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("streetscape_identity_sha256") != object_sha256(identity):
        raise ValueError("Export does not match the supplied training identity")
    raw = copy.deepcopy(get_config(args.config))
    raw["config"]["name"] = args.output.name
    processes = raw["config"]["process"]
    if len(processes) != 1:
        raise ValueError("Use a single-process smoke config")
    process = processes[0]
    resolved, assets = resolve_krea_assets(process, identity["model_assets"])
    if assets != identity["model_assets"]:
        raise ValueError("Export acceptance base/TE/VAE weights changed")
    process["model"] = resolved["model"]
    process["type"] = "sd_trainer"
    process.pop("cnn_guidance", None)
    process["training_folder"] = str(args.output.parent.resolve())
    process.pop("name", None)
    process["network"]["pretrained_lora_path"] = str(args.export.resolve())
    process["train"].update(steps=0, disable_sampling=False, skip_first_sample=False,
                           force_first_sample=True, start_step=None)
    process["save"]["push_to_hub"] = False
    process["sample"] = {"sampler": "flowmatch", "sample_every": 1, "width": args.width,
                         "height": args.height, "sample_steps": args.steps, "guidance_scale": args.cfg,
                         "seed": args.seed, "walk_seed": False, "prompts": [args.prompt]}
    job = get_job(raw)
    try:
        job.run()
    finally:
        job.cleanup()
    images = [p for p in (args.output / "samples").rglob("*")
              if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not images:
        raise RuntimeError("No inference image produced; export loading not accepted")
    write_json_atomic(args.output / "export_inference_acceptance.json",
                      {"export_sha256": file_sha256(args.export), "optimizer_steps": 0,
                       "base_model": process["model"]["name_or_path"],
                       "training_identity_sha256": object_sha256(identity),
                       "image_files": [p.relative_to(args.output).as_posix() for p in images],
                       "status": "native adapter load/sample completed; not a quality benchmark"})


if __name__ == "__main__":
    main()
