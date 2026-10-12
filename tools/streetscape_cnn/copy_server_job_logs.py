"""Copy byte-exact logs of explicit completed CompShare jobs into the result archive."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extensions_built_in.streetscape_cnn.contract import file_sha256, write_json_atomic


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {}
    for job in args.job:
        if not job.startswith('job-') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in job):
            raise ValueError('Invalid explicit job ID')
        source = Path.home() / '.local/state/compshare/jobs' / job
        if not (source / 'exit_code').exists():
            raise ValueError('Job is not complete; snapshot only completed logs')
        for stream in ['stdout', 'stderr']:
            original = source / f'{stream}.log'
            target = args.output / f'{job}.{stream}.log'
            digest = file_sha256(original)
            if target.exists() and file_sha256(target) != digest:
                raise ValueError('Existing log archive differs')
            if not target.exists():
                shutil.copy2(original, target)
            result[target.name] = digest
        result[job + '.exit_code'] = (source / 'exit_code').read_text().strip()
    write_json_atomic(args.output / 'job_logs_manifest.json', result)
    print(f'{len(args.job)} completed jobs archived byte-exactly')
