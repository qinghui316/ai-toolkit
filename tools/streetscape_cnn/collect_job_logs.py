"""Locally archive complete redacted official CompShare job streams."""
import argparse
import json
from pathlib import Path
import subprocess


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', type=Path, required=True)
    p.add_argument('--instance', required=True)
    p.add_argument('--job', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for job in args.job:
        offsets = {'stdout': 0, 'stderr': 0}
        streams = {'stdout': [], 'stderr': []}
        state = None
        while True:
            command = [str(args.cli), '--json', 'instance', 'job', 'logs', args.instance, job,
                       '--stdout-offset', str(offsets['stdout']), '--stderr-offset', str(offsets['stderr']),
                       '--limit', '65536']
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=45)
            response = json.loads(result.stdout)
            if result.returncode or not response.get('ok'):
                raise RuntimeError(f'Official log retrieval failed for {job}')
            log = response['data']['logs']
            state = log['State']
            progressed = False
            for stream in ['stdout', 'stderr']:
                text = log.get(stream.title(), '')
                if text:
                    streams[stream].append(text)
                    offsets[stream] += len(text.encode('utf-8'))
                    progressed = True
            if not progressed:
                break
        for stream, chunks in streams.items():
            (args.output / f'{job}.{stream}.log').write_text(''.join(chunks), encoding='utf-8')
        (args.output / f'{job}.json').write_text(json.dumps({'job': job, 'state': state,
                                                          'retrieved_byte_offsets': offsets}, indent=2), encoding='utf-8')
        print(json.dumps({'job': job, 'state': state, 'bytes': offsets}))
