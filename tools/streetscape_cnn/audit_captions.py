"""Read-only tagger pool audit; does not copy/delete images or assign LoRA splits."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--expected-count", type=int, default=480)
    parser.add_argument("--allow-unpaired", action="store_true",
                        help="Audit mixed tagger exports only; trainer always rejects unpaired images")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    # Keep this read-only audit dependency-free: no torch/toolkit/model initialization.
    path = Path(__file__).resolve().parents[2] / "extensions_built_in/streetscape_cnn/contract.py"
    spec = importlib.util.spec_from_file_location("streetscape_contract_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module.scan_caption_pairs(args.folder, allow_unpaired=args.allow_unpaired)
    if result["pair_count"] != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} pairs, found {result['pair_count']}")
    counts = Counter(row["id"].split("_")[1] if row["id"].startswith("CS_") else "other"
                     for row in result["pairs"])
    print(json.dumps({"pair_count": result["pair_count"], "archetype_counts": dict(sorted(counts.items())),
                      "pairs_identity_sha256": module.object_sha256(result["pairs"]),
                      "unpaired_image_count": len(result["unpaired_images"]),
                      "training_ready_directory": not result["unpaired_images"],
                      "mutations": "none; no copying, deletion or split assignment"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
