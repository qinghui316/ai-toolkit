"""Install the approved isolated server profile and record every command outcome."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/streetscape'))
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('Deployment belongs on the Linux server')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    reports = root / 'acceptance'
    reports.mkdir(exist_ok=True)
    environment = root / '.venv'
    if not environment.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
    python = str(environment / 'bin/python')
    commands = [
        [python, '-m', 'pip', 'install', 'pip==25.3'],
        [python, '-m', 'pip', 'install', 'torch==2.9.1', 'torchvision==0.24.1',
         'torchaudio==2.9.1', '--index-url', 'https://download.pytorch.org/whl/cu128'],
        [python, '-m', 'pip', 'install', '-r', str(Path(__file__).with_name('requirements_server.txt'))],
        [python, '-m', 'pip', 'check'],
    ]
    for index, command in enumerate(commands):
        started = time.time()
        print(f'bootstrap stage {index + 1}/{len(commands)}', flush=True)
        with (reports / f'install_{index + 1}.log').open('a', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = {'stage': index + 1, 'command': command, 'exit_code': result.returncode,
                  'seconds': time.time() - started}
        (reports / f'install_{index + 1}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        if result.returncode:
            raise SystemExit(f'Installation failed; see install_{index + 1}.log. No fallback performed.')
    with (reports / 'requirements.lock.txt').open('w', encoding='utf-8') as log:
        subprocess.run([python, '-m', 'pip', 'freeze'], stdout=log, check=True)
    print('Isolated installation complete; runtime acceptance still required.', flush=True)


if __name__ == '__main__':
    main()
