"""Submit an identity-bound stop request without loading a model."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.runtime import request_stop


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['request-stop'])
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(request_stop(args.run_dir)))
