"""Prepare byte-preserving 480 candidate pairs and lock a 384/48/48 split."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.data import prepare_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260909)
    args = p.parse_args()
    print(json.dumps(prepare_dataset(args.source, args.candidates, args.output, args.seed)))


if __name__ == '__main__':
    main()
