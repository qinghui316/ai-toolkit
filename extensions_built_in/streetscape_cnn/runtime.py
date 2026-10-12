"""Opt-in schedules, durable research records, and fail-closed runtime controls."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from .contract import read_json, write_json_atomic


class ResearchAbort(ValueError):
    """Not swallowed by the upstream CUDA RuntimeError/OOM skip handlers."""


@dataclass(frozen=True)
class ResearchLRSchedule:
    stages: tuple[tuple[int, float], ...] = ()

    @classmethod
    def parse(cls, value, train):
        if not value:
            return cls()
        if set(value) - {'enabled', 'stages'} or type(value.get('enabled', False)) is not bool:
            raise ValueError('Invalid research_lr_schedule')
        if not value.get('enabled', False):
            return cls()
        stages = value.get('stages')
        if not isinstance(stages, list) or not stages:
            raise ValueError('An enabled schedule requires explicit stages')
        result = []
        for item in stages:
            if set(item) != {'start_step', 'lr'}:
                raise ValueError('Schedule stages require start_step and lr')
            step, lr = item['start_step'], item['lr']
            if type(step) is not int or not 0 <= step < train['steps']:
                raise ValueError('Schedule step outside the native run')
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
                raise ValueError('Schedule lr must be finite and positive')
            result.append((step, float(lr)))
        if result[0] != (0, float(train['lr'])) or any(b[0] <= a[0] for a, b in zip(result, result[1:])):
            raise ValueError('Schedule must start at native step 0 with train.lr and strictly increase')
        if train.get('lr_scheduler', 'constant') != 'constant' or train.get('lr_scheduler_params', {}).get('factor', 1) != 1:
            raise ValueError('Research stages require the native constant scheduler with factor 1')
        return cls(tuple(result))

    def rate(self, step):
        if not self.stages:
            return None
        return next(lr for start, lr in reversed(self.stages) if step >= start)


def recording_config(value):
    value = dict(value or {})
    allowed = {'enabled', 'per_sample_scores', 'archive_checkpoints'}
    if set(value) - allowed or any(type(v) is not bool for v in value.values()):
        raise ValueError('Invalid research_recording boolean fields')
    enabled = value.get('enabled', False)
    return {'enabled': enabled, 'per_sample_scores': value.get('per_sample_scores', enabled),
            'archive_checkpoints': value.get('archive_checkpoints', enabled)}


class ResearchRecorder:
    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity
        (self.root / 'records').mkdir(exist_ok=True)

    def append(self, name, row):
        payload = {'identity_sha256': self.identity, **row}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with (self.root / 'records' / f'{name}.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded + '\n')
            stream.flush()

    def status(self, state, **details):
        write_json_atomic(self.root / 'research_status.json',
                          {'identity_sha256': self.identity, 'state': state, **details})


def request_stop(root):
    root = Path(root)
    identity = read_json(root / 'streetscape_identity.json')
    from .contract import object_sha256
    value = {'identity_sha256': object_sha256(identity), 'requested': True}
    target = root / 'research_stop_request.json'
    if target.exists() and read_json(target) != value:
        raise ValueError('Existing stop request does not match this run')
    write_json_atomic(target, value)
    return value


def stop_requested(root, identity):
    target = Path(root) / 'research_stop_request.json'
    if not target.exists():
        return False
    value = read_json(target)
    if value.get('identity_sha256') != identity or value.get('requested') is not True:
        raise ResearchAbort('Stop request identity mismatch')
    return True
