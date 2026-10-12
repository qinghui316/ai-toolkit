"""Run an explicitly configured real Krea smoke job, optionally interrupting after a save."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Filled independent smoke YAML; not the production template")
    parser.add_argument("--stop-after", type=int, help="Stop after a native optimizer-boundary save at this global step")
    parser.add_argument("--check-gradients", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Real Krea validation belongs on the Linux GPU server")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if args.check_gradients:
        os.environ["STREETSCAPE_VERIFY_GRADIENTS"] = "1"
    from toolkit.job import get_job
    job = get_job(args.config)
    from extensions_built_in.streetscape_cnn.trainer import StreetscapeCNNTrainer
    processes = [p for p in job.process if isinstance(p, StreetscapeCNNTrainer)]
    if len(processes) != 1 or len(job.process) != 1:
        raise ValueError("Smoke validation requires exactly one streetscape_cnn_trainer")
    trainer = processes[0]
    from extensions_built_in.streetscape_cnn.runtime import request_stop
    def graceful_signal(signum, frame):
        request_stop(trainer.save_root)
        print('Stop requested; waiting for complete optimizer/save boundary', flush=True)
    signal.signal(signal.SIGTERM, graceful_signal)
    signal.signal(signal.SIGINT, graceful_signal)
    intentional_stop = False
    if args.stop_after is not None:
        if args.stop_after >= trainer.train_config.steps:
            raise ValueError("stop-after must be before the final configured training step")
        original = trainer.end_step_hook

        def stop_hook():
            nonlocal intentional_stop
            original()
            if trainer.step_num >= args.stop_after and not trainer.is_grad_accumulation_step:
                trainer.save(step=trainer.step_num)
                intentional_stop = True
                raise KeyboardInterrupt("Intentional server acceptance interruption after native save")

        trainer.end_step_hook = stop_hook
    try:
        job.run()
    except KeyboardInterrupt:
        if not intentional_stop:
            from extensions_built_in.streetscape_cnn.contract import read_json
            status = Path(trainer.save_root) / 'research_status.json'
            if not status.exists() or read_json(status).get('state') != 'stopped':
                raise
            print('User-controlled stop saved; this is not completed acceptance')
            return
        print("Intentional smoke interruption; rerun the SAME config without --stop-after to verify native resume")
    else:
        if args.check_gradients and not trainer._gradient_verified:
            raise RuntimeError("No eligible batch verified CNN -> LoRA gradients; do not mark acceptance passed")
    finally:
        job.cleanup()


if __name__ == "__main__":
    main()
