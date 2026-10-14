"""Bounded, registered image+CNN experiment using the existing Linux environment."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from extensions_built_in.streetscape_cnn.contract import (
    file_sha256, read_json, write_json_atomic)
from extensions_built_in.streetscape_cnn.evaluation import audit_run, journal, preview_inventory
from tools.streetscape_cnn.make_cnn_probe_configs import derive_configs
from tools.streetscape_cnn.make_fit_probe_configs import save_consistent, select_prompts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--token-counts', type=Path, required=True)
    args = p.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('Runtime validation belongs on Linux')
    root = args.root.resolve()
    reports = root/'acceptance/cnn_probe_v1'
    if reports.exists():
        raise ValueError('Use a fresh registered experiment, not an automatic pipeline restart')
    reports.mkdir(parents=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    baseline_config = root/'configs/fit_probe_1000.json'
    baseline = read_json(baseline_config)
    base_run = root/'runs'/baseline['config']['name']
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONUNBUFFERED': '1', 'STREETSCAPE_VERIFY_GRADIENTS': '1'}
    stage = 'baseline_audit'

    def status(state='running', **extra):
        write_json_atomic(reports/'pipeline_status.json', {'stage': stage, 'state': state,
            'source_revision': revision, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'scope': 'registered image+CNN 1000-update comparison; no extra experiments', **extra})

    def run(name, arguments):
        nonlocal stage
        stage = name
        status()
        with (reports/f'{name}.log').open('w', encoding='utf-8') as stream:
            started = time.monotonic()
            value = subprocess.run([str(x) for x in arguments], cwd=REPO, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        write_json_atomic(reports/f'{name}_command.json', {
            'arguments': [str(x) for x in arguments], 'exit_code': value.returncode,
            'seconds': time.monotonic()-started})
        return value.returncode

    def required(name, arguments):
        if run(name, arguments):
            raise RuntimeError(f'{name} failed; no alternative scientific parameters')

    tools = REPO/'tools/streetscape_cnn'
    try:
        status()
        before_config = file_sha256(baseline_config)
        write_json_atomic(reports/'baseline_audit.json', audit_run(base_run, 1000, False, expected_previews=80))
        prompts = read_json(root/'configs/fit_prompts.json')['rows']
        if select_prompts(root/'dataset', read_json(root/'dataset/dataset_manifest.json')) != prompts:
            raise ValueError('Locked dataset/prompts changed')
        preview_inventory(base_run, prompts)
        required('unit_tests', [sys.executable, tools/'server_validate.py', '--unit'])
        required('make_configs', [sys.executable, tools/'make_cnn_probe_configs.py',
                                 '--root', root, '--baseline-config', baseline_config])
        configs = derive_configs(baseline, root)
        for name, expected in configs.items():
            if read_json(root/'configs'/name) != expected:
                raise ValueError('Registered CNN configuration changed')
        required('baseline_scores', [sys.executable, tools/'score_previews.py', '--run-dir', base_run,
            '--prompts', root/'configs/fit_prompts.json', '--checkpoint', root/'cnn/final.pt',
            '--manifest', root/'cnn/model_manifest.json', '--output', reports/'baseline_scores'])
        required('memory_parity', [sys.executable, tools/'server_memory_parity.py',
                                  '--root', root, '--output', reports/'memory_parity.json'])
        import torch
        total = torch.cuda.get_device_properties(0).total_memory
        limit = total-2*2**30
        write_json_atomic(reports/'hardware_memory_gate.json', {'total_bytes': total, 'limit_bytes': limit,
            'host_min_available_bytes': 2*2**30, 'reserve_bytes': 2*2**30})
        selected = None
        for tier in (1, 2, 3):
            config = root/'configs'/f'cnn_smoke24_p{tier}_v1.json'
            result_path = reports/f'probe_p{tier}.json'
            code = run(f'probe_p{tier}', [sys.executable, tools/'server_memory_probe.py', config,
                '--root', root, '--token-counts', args.token_counts, '--output', result_path])
            result = read_json(result_path) if result_path.exists() else {}
            if code:
                message = str(result.get('message', '')).lower()
                if not any(text in message for text in ('gpu memory acceptance gate exceeded',
                                                       'cuda memory exhaustion', 'cuda out of memory')):
                    raise RuntimeError(f'Non-GPU-capacity failure in P{tier}; stop, do not bypass')
                continue
            if result.get('state') != 'passed' or len(result['cases']) != 2:
                raise ValueError('Native stress probe acceptance incomplete')
            if any(max(c['memory']['reserved'], c['memory']['process_gpu']) > limit
                   for c in result['cases']):
                continue
            selected = tier
            break
        if selected is None:
            raise RuntimeError('All three registered GPU memory tiers failed; no training started')
        write_json_atomic(reports/'selected_tier.json', {'tier': selected, 'limit_bytes': limit,
            'profile': read_json(root/'configs'/f'cnn_1000_p{selected}_v1.json')['config']['process'][0]['cnn_memory_profile']})
        smoke_config = root/'configs'/f'cnn_smoke24_p{selected}_v1.json'
        smoke = root/'runs'/read_json(smoke_config)['config']['name']
        required('smoke_to8', [sys.executable, tools/'server_train_smoke.py', smoke_config,
                              '--stop-after', '8', '--check-gradients'])
        status8 = read_json(smoke/'research_status.json')
        if status8['completed_updates'] != 8 or status8['native_step'] != 8:
            raise ValueError('Interruption was not exactly at update 8')
        save_consistent(reports/'resume8.json', read_json(smoke/'streetscape_latest.json'))
        required('smoke_resume24', [sys.executable, tools/'server_train_smoke.py', smoke_config, '--check-gradients'])
        write_json_atomic(reports/'smoke_audit.json', audit_run(smoke, 24, True, limit, 0, require_resume=True))
        export = smoke/f'{smoke.name}.safetensors'
        required('smoke_export_load', [sys.executable, tools/'server_export_inference.py', smoke_config,
            export, '--output', reports/'smoke_export', '--prompt', prompts[8]['prompt'],
            '--steps', '30', '--cfg', '4', '--width', '768', '--height', '512', '--seed', '42'])
        if file_sha256(baseline_config) != before_config:
            raise ValueError('Immutable baseline configuration changed')
        full_config = root/'configs'/f'cnn_1000_p{selected}_v1.json'
        full = root/'runs'/read_json(full_config)['config']['name']
        if full.exists():
            raise ValueError('Full comparison must start fresh, never inherit smoke/base weights')
        required('cnn1000', [sys.executable, tools/'server_train_smoke.py', full_config, '--check-gradients'])
        write_json_atomic(reports/'cnn1000_audit.json', audit_run(full, 1000, True, limit, 80))
        preview_inventory(full, prompts)
        required('paired_scores', [sys.executable, tools/'score_previews.py', '--run-dir', base_run,
            '--run-dir', full, '--prompts', root/'configs/fit_prompts.json',
            '--checkpoint', root/'cnn/final.pt', '--manifest', root/'cnn/model_manifest.json',
            '--output', reports/'paired_scores'])
        required('full_export_load', [sys.executable, tools/'server_export_inference.py', full_config,
            full/f'{full.name}.safetensors', '--output', reports/'full_export',
            '--prompt', prompts[8]['prompt'], '--steps', '30', '--cfg', '4',
            '--width', '768', '--height', '512', '--seed', '42'])
        stage = 'complete'
        status('completed', updates=1000, image_presentations=2000, main_previews=80,
               tier=selected, run=full.name, next_action='Review paired diagnostics; no automatic extension')
        # Complete metadata package remains on the server until an authorized transfer.
        code = subprocess.run([sys.executable, str(tools/'export_observation_records.py'),
            '--root', str(root), '--run', base_run.name, '--run', smoke.name, '--run', full.name,
            '--report-dir', str(reports), '--output', str(root/'transfers/cnn_probe_v1_records.tar')],
            cwd=REPO, env=env).returncode
        if code:
            raise RuntimeError('Completed training records package failed; preserve all source outputs')
    except BaseException as exc:
        (reports/'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        status('failed', error_type=type(exc).__name__, message=str(exc))
        raise


if __name__ == '__main__':
    main()
