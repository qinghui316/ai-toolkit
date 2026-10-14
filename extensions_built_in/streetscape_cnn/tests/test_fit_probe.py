from pathlib import Path
import unittest

from tools.streetscape_cnn.make_fit_probe_configs import build_configs, save_consistent
from tempfile import TemporaryDirectory


class FitProbeTests(unittest.TestCase):
    def configs(self):
        return build_configs(Path('/workspace/streetscape'),
                             [{'prompt': f'street {n}'} for n in range(16)],
                             Path('/model/ModelScope/Qwen/Qwen3-VL-4B-Instruct'))

    def test_fixed_base_objective_updates_and_native_accumulation(self):
        config = self.configs()['fit_probe_1000.json']['config']['process'][0]
        self.assertEqual(config['train']['steps'], 1000)
        self.assertEqual(config['train']['lr'], .0001)
        self.assertEqual(config['train']['gradient_accumulation'], 2)
        self.assertEqual(config['train']['gradient_accumulation_steps'], 1)
        self.assertEqual(config['cnn_guidance'], {'enabled': False})
        self.assertFalse(config['research_lr_schedule']['enabled'])
        self.assertNotIn('pretrained_lora_path', config['network'])

    def test_preview_does_not_increase_checkpoint_frequency(self):
        config = self.configs()['fit_probe_1000.json']['config']['process'][0]
        self.assertEqual(config['sample']['sample_every'], 250)
        self.assertEqual(config['save']['save_every'], 1000)
        self.assertEqual(len(config['sample']['prompts']), 16)
        self.assertEqual(config['sample']['neg'], '')
        self.assertEqual(config['sample']['seed'], 42)
        self.assertFalse(config['sample']['walk_seed'])
        self.assertFalse(config['train']['disable_sampling'])
        self.assertFalse(config['save']['push_to_hub'])

    def test_smoke_is_an_independent_four_update_run(self):
        configs = self.configs()
        smoke = configs['fit_smoke4.json']['config']
        self.assertNotEqual(smoke['name'], configs['fit_probe_1000.json']['config']['name'])
        self.assertEqual(smoke['process'][0]['train']['steps'], 4)
        self.assertEqual(smoke['process'][0]['sample']['sample_every'], 2)
        self.assertEqual(len(smoke['process'][0]['sample']['prompts']), 1)
        self.assertEqual(smoke['process'][0]['save']['save_every'], 1000)

    def test_asset_paths_are_explicit_and_only_train_folder_is_loaded(self):
        config = self.configs()['fit_probe_1000.json']['config']['process'][0]
        self.assertIn('/ModelScope/Qwen/', config['local_model_assets']['text_encoder'])
        self.assertEqual([d['folder_path'] for d in config['datasets']],
                         ['/workspace/streetscape/dataset/train'])

    def test_repeated_generation_reuses_only_identical_artifacts(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'config.json'
            save_consistent(path, {'steps': 1000})
            save_consistent(path, {'steps': 1000})
            with self.assertRaises(ValueError):
                save_consistent(path, {'steps': 2000})
