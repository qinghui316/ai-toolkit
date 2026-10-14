import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.streetscape_cnn.make_fit_probe_configs import build_configs
from tools.streetscape_cnn.make_cnn_probe_configs import derive_configs
from tools.streetscape_cnn.export_observation_records import collect
from extensions_built_in.streetscape_cnn.evaluation import paired_rows, descriptive_summary, METRICS
from extensions_built_in.streetscape_cnn.memory import PhaseMemoryMonitor, MEMORY_LIMIT_BYTES
from extensions_built_in.streetscape_cnn.runtime import ResearchAbort


class CNNProbeTests(unittest.TestCase):
    def baseline(self, root):
        return build_configs(root, [{'prompt': f'street {i}'} for i in range(16)],
                             Path('/model/ModelScope/Qwen/Qwen3-VL-4B-Instruct'))['fit_probe_1000.json']

    def test_independent_fixed_objective_and_no_baseline_mutation(self):
        root = Path('/workspace/streetscape')
        base = self.baseline(root)
        before = copy.deepcopy(base)
        configs = derive_configs(base, root)
        self.assertEqual(base, before)
        self.assertEqual(len(configs), 6)
        for name, value in configs.items():
            process = value['config']['process'][0]
            self.assertEqual(process['cnn_guidance']['weight'], .05)
            self.assertEqual(process['cnn_guidance']['start_step'], 0)
            self.assertEqual(process['cnn_guidance']['every_n_steps'], 1)
            self.assertEqual(process['research_gpu_memory_reserve_gib'], 2)
            self.assertEqual(process['train']['lr'], .0001)
            self.assertEqual(process['train']['gradient_accumulation'], 2)
            self.assertNotIn('pretrained_lora_path', process['network'])
            self.assertNotEqual(value['config']['name'], base['config']['name'])
            self.assertEqual(process['save']['save_every'], 1000)
            self.assertEqual(process['train']['steps'], 24 if 'smoke24' in name else 1000)
        self.assertEqual(configs['cnn_1000_p3_v1.json']['config']['process'][0]
                         ['model']['layer_offloading_transformer_percent'], 1.)

    def test_baseline_contract_refuses_changed_steps_or_adapter(self):
        root = Path('/workspace/streetscape')
        for mutation in ('steps', 'adapter'):
            base = self.baseline(root)
            if mutation == 'steps':
                base['config']['process'][0]['train']['steps'] = 2000
            else:
                base['config']['process'][0]['network']['pretrained_lora_path'] = 'old.safetensors'
            with self.assertRaises(ValueError):
                derive_configs(base, root)

    def rows(self, run, score, update=1000):
        return [{'run': run, 'completed_updates': update, 'prompt_id': f'p{i}',
            'seed': 42, 'role': 'train' if i < 8 else 'validation',
            'archetype_code': f'T{i%8+1:02d}', 'prompt': f'caption {i}',
            **{k: score for k in METRICS}} for i in range(16)]

    def test_paired_descriptive_summary_and_roles(self):
        pairs = paired_rows(self.rows('base', 5), self.rows('cnn', 5.2), 1000, 1000)
        summary = descriptive_summary(pairs)
        overall = [row for row in summary if row['archetype_code'] == 'all']
        self.assertEqual([row['pairs'] for row in overall], [8, 8])
        self.assertAlmostEqual(overall[0]['mean_delta_inclusive_all6'], .2)
        self.assertEqual(len(summary), 18)

    def test_refuse_missing_duplicate_or_changed_pair(self):
        for reason in ('missing', 'duplicate', 'caption'):
            right = self.rows('cnn', 6)
            if reason == 'missing':
                right.pop()
            elif reason == 'duplicate':
                right[1]['prompt_id'] = 'p0'
            else:
                right[0]['prompt'] = 'different conditioning'
            with self.assertRaises(ValueError):
                paired_rows(self.rows('base', 5), right, 1000, 1000)

    def test_dynamic_gpu_reserve_and_legacy_gate(self):
        monitor = PhaseMemoryMonitor.__new__(PhaseMemoryMonitor)
        monitor.error = None
        monitor.limit_bytes = 21*2**30
        monitor.peaks = {'reserved': 22*2**30, 'process_gpu': 20*2**30}
        with self.assertRaises(ResearchAbort):
            monitor.check()
        del monitor.limit_bytes
        monitor.check()
        monitor.peaks['reserved'] = MEMORY_LIMIT_BYTES+1
        with self.assertRaises(ResearchAbort):
            monitor.check()

    def test_metadata_export_excludes_all_weights_and_derived_thumbnails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'runs/test'
            (run/'checkpoints').mkdir(parents=True)
            (run/'samples/.thumbs').mkdir(parents=True)
            for name in ('optimizer.pt', 'model.safetensors', 'checkpoint.bin',
                         'research_status.json', 'records.csv', 'samples/main.jpg',
                         'samples/.thumbs/derived.jpg', 'checkpoints/state.pt',
                         'checkpoints/checkpoint_manifest.json'):
                (run/name).write_bytes(b'fixture')
            files = collect(root, ['test'], [])
            self.assertEqual(len(files), 4)
            self.assertTrue(all(not name.endswith(('.pt', '.bin', '.safetensors')) for name in files))
            self.assertFalse(any('.thumbs' in name for name in files))


if __name__ == '__main__':
    unittest.main()
