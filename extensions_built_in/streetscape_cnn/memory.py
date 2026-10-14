"""Opt-in frozen-decoder memory policies and measured engineering gates."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import subprocess
import threading
import time

import torch

from .runtime import ResearchAbort

PROFILES = ('legacy', 'vae_checkpoint', 'vae_checkpoint_cpu')
MEMORY_LIMIT_BYTES = 28 * 1024 ** 3


def parse_profile(value=None):
    value = 'legacy' if value is None else value
    if value not in PROFILES or not isinstance(value, str):
        raise ValueError(f'cnn_memory_profile must be one of {PROFILES}')
    return value


def configure_decoder(vae, profile):
    """Do not change decoder math, dtype, weights, or the tiling implementation."""
    profile = parse_profile(profile)
    vae.eval()
    vae.requires_grad_(False)
    if profile != 'legacy':
        from toolkit.util.qwen_vae_gradient_checkpointing import patch_qwen_vae_gradient_checkpointing
        patch_qwen_vae_gradient_checkpointing()
        vae.enable_gradient_checkpointing()
        if not getattr(vae.decoder, 'gradient_checkpointing', False):
            raise ResearchAbort('Requested frozen VAE decoder checkpointing is not active')
    return {'profile': profile, 'decoder_checkpointing': bool(getattr(vae.decoder, 'gradient_checkpointing', False)),
            'saved_tensors_cpu': profile == 'vae_checkpoint_cpu', 'pin_memory': False,
            'vae_dtype': str(vae.dtype), 'vae_training': vae.training}


def scoring_memory_context(profile, observer=None):
    if parse_profile(profile) == 'vae_checkpoint_cpu':
        context = torch.autograd.graph.save_on_cpu(pin_memory=False)
        if observer is not None:
            original = context.pack_hook
            def measured_pack(tensor):
                packed = original(tensor)
                observer(str(packed[1].device), tensor.numel() * tensor.element_size())
                return packed
            context.pack_hook = measured_pack
        return context
    return nullcontext()


def clear_decoder_cache(vae):
    if vae is not None and hasattr(vae, 'clear_cache'):
        vae.clear_cache()


def host_memory():
    status = Path('/proc/self/status').read_text().splitlines()
    values = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in status if line.startswith(('VmRSS:', 'VmHWM:'))}
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return {'rss': values['VmRSS'], 'rss_high_water': values['VmHWM'], 'available': available}


class PhaseMemoryMonitor:
    """GPU process memory is sampled; PyTorch peaks are exact allocator counters.

    Preserve the maximum across phase-local resets in each full native update.
    The monitor never reads or retains a scoring-path tensor.
    """
    def __init__(self, device, recorder, limit_bytes=None):
        self.device, self.recorder = device, recorder
        self.limit_bytes = MEMORY_LIMIT_BYTES if limit_bytes is None else int(limit_bytes)
        if self.limit_bytes <= 0:
            raise ValueError('GPU memory acceptance limit must be positive')
        self.phase_name = 'between_updates'
        self.step = 0
        self.peaks = {}
        self.phase_rows = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.error = None
        self.samples = 0
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._sample()
        if self.samples == 0:
            raise ResearchAbort('GPU process-memory measurement unavailable')
        self._thread.start()

    def _sample(self):
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                          '--format=csv,noheader,nounits'], text=True, timeout=5)
        used = None
        for line in output.splitlines():
            fields = [part.strip() for part in line.split(',')]
            if len(fields) == 2 and fields[0] == str(os.getpid()):
                used = int(fields[1]) * 1024 ** 2
        if used is None:
            raise ResearchAbort('Current CUDA process absent from nvidia-smi measurement')
        host = host_memory()
        with self._lock:
            self.samples += 1
            self.peaks['process_gpu'] = max(self.peaks.get('process_gpu', 0), used)
            self.peaks['rss'] = max(self.peaks.get('rss', 0), host['rss'])
            self.peaks['available_min'] = min(self.peaks.get('available_min', host['available']), host['available'])
        if host['available'] < 2 * 1024 ** 3:
            self.error = 'Host available memory below 2 GiB safety reserve'

    def _watch(self):
        while not self._stop.wait(.2):
            try:
                self._sample()
            except Exception as exc:
                self.error = f'Memory monitoring failed: {exc}'
                return

    def begin_update(self, step):
        torch.cuda.synchronize(self.device)
        with self._lock:
            self.step, self.peaks, self.phase_rows = int(step), {}, []
        torch.cuda.reset_peak_memory_stats(self.device)
        self._sample()

    @contextmanager
    def phase(self, name):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        previous = self.phase_name
        self.phase_name = name
        started = time.monotonic()
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            torch.cuda.synchronize(self.device)
            allocated = torch.cuda.max_memory_allocated(self.device)
            reserved = torch.cuda.max_memory_reserved(self.device)
            self._sample()
            with self._lock:
                self.peaks['allocated'] = max(self.peaks.get('allocated', 0), allocated)
                self.peaks['reserved'] = max(self.peaks.get('reserved', 0), reserved)
                row = {'native_step': self.step, 'phase': name, 'seconds': time.monotonic() - started,
                       'cuda_peak_allocated': allocated, 'cuda_peak_reserved': reserved,
                       'sampled_process_gpu_peak': self.peaks['process_gpu'], 'cpu_rss_peak': self.peaks['rss'],
                       'host_available_min': self.peaks['available_min'], 'error_type': error,
                       'process_memory_sampling_seconds': .2}
                self.phase_rows.append(row)
            self.recorder.append('memory_phases', row)
            self.phase_name = previous

    def check(self):
        if self.error:
            raise ResearchAbort(self.error)
        limit = getattr(self, 'limit_bytes', MEMORY_LIMIT_BYTES)
        if max(self.peaks.get('reserved', 0), self.peaks.get('process_gpu', 0)) > limit:
            raise ResearchAbort(f'GPU memory acceptance gate exceeded {limit} bytes; stop this attempt')

    def close(self):
        self._stop.set()
        self._thread.join(timeout=6)
