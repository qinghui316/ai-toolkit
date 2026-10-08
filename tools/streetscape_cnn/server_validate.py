"""Linux/server-only acceptance entrypoint. No invocation is made by local implementation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", action="store_true")
    parser.add_argument("--cnn-checkpoint", type=Path)
    parser.add_argument("--cnn-manifest", type=Path)
    parser.add_argument("--cnn-image", type=Path, help="Optional approved development image for the real CNN gradient check")
    parser.add_argument("--export", type=Path, help="Native exported LoRA safetensors")
    parser.add_argument("--run-dir", type=Path, help="Check hashes of the native latest save pair")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Runtime acceptance is intentionally reserved for the Linux server")
    if args.unit:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                        "extensions_built_in/streetscape_cnn/tests", "-v"], cwd=ROOT, check=True)
    if args.cnn_checkpoint or args.cnn_manifest:
        if not args.cnn_checkpoint or not args.cnn_manifest:
            parser.error("Supply both --cnn-checkpoint and --cnn-manifest")
        import torch
        from extensions_built_in.streetscape_cnn.scorer import FrozenStreetscapeScorer
        scorer = FrozenStreetscapeScorer.from_checkpoint(args.cnn_checkpoint, args.cnn_manifest).to("cuda")
        if args.cnn_image:
            from PIL import Image
            from torchvision.transforms.functional import to_tensor
            with Image.open(args.cnn_image) as image:
                rgb = to_tensor(image.convert("RGB")).unsqueeze(0).to("cuda").requires_grad_()
        else:
            rgb = torch.linspace(.2, .8, 3 * 224 * 336, device="cuda").reshape(1, 3, 224, 336).requires_grad_()
        before = {key: tensor.clone() for key, tensor in scorer.named_buffers()}
        reward = scorer(rgb)["inclusive_all6"].mean()
        gradient = torch.autograd.grad(reward, rgb)[0]
        if not torch.isfinite(gradient).all() or not (gradient != 0).any():
            raise RuntimeError("Frozen CNN input gradients must be finite and nonzero")
        if any(p.requires_grad or p.grad is not None for p in scorer.parameters()):
            raise RuntimeError("Frozen CNN acquired trainable parameters/gradients")
        if any(not torch.equal(before[key], tensor) for key, tensor in scorer.named_buffers()):
            raise RuntimeError("Frozen CNN buffers changed")
        print("Frozen real CNN CUDA/input-gradient check passed; not a generative-quality metric")
    if args.run_dir:
        from extensions_built_in.streetscape_cnn.contract import read_json, object_sha256, validate_resume_pair
        identity = read_json(args.run_dir / "streetscape_identity.json")
        record = validate_resume_pair(args.run_dir, object_sha256(identity))
        print(json.dumps({"latest_step": record["step"], "pair_hashes_valid": True}))
    if args.export:
        import torch
        from safetensors import safe_open
        with safe_open(str(args.export), framework="pt", device="cpu") as handle:
            keys, metadata = list(handle.keys()), handle.metadata() or {}
            if not keys or not metadata.get("streetscape_identity_sha256"):
                raise ValueError("Export is empty or missing research metadata")
            if any(not torch.isfinite(handle.get_tensor(key)).all() for key in keys):
                raise ValueError("Non-finite exported adapter tensor")
        print("Export structure/finiteness passed. Actual model loading/inference still required.")
    if not any((args.unit, args.cnn_checkpoint, args.run_dir, args.export)):
        parser.error("Select an acceptance check")


if __name__ == "__main__":
    main()
