from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from extensions_built_in.streetscape_cnn.assets import resolve_krea_assets, PINNED_REVISIONS
from extensions_built_in.streetscape_cnn.contract import file_sha256, write_json_atomic, portable_process_config


class LocalAssetsTests(unittest.TestCase):
    def fixture(self, root):
        paths = {'raw': root / 'raw', 'text_encoder': root / 'text', 'vae': root / 'vae-root'}
        names = {'raw': ['raw.safetensors'],
                 'text_encoder': ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                                  'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'],
                 'vae': ['vae/config.json', 'vae/diffusion_pytorch_model.safetensors']}
        components = {}
        for role, files in names.items():
            entries = {}
            for name in files:
                path = paths[role] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                entries[name] = {'sha256': file_sha256(path)}
            repo, revision = PINNED_REVISIONS[role]
            components[role] = {'repo': repo, 'revision': revision, 'files': entries}
        manifest = root / 'manifest.json'
        write_json_atomic(manifest, {'schema_version': 1, 'components': components})
        return {'model': {'arch': 'krea2', 'name_or_path': 'krea/Krea-2-Raw'},
                'local_model_assets': {'raw': str(paths['raw'] / 'raw.safetensors'),
                                       'text_encoder': str(paths['text_encoder']), 'vae': str(paths['vae']),
                                       'manifest': str(manifest)}}

    def test_local_mode_never_calls_hub_and_records_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory))
            with patch('extensions_built_in.streetscape_cnn.assets.PINNED_WEIGHTS', {}):
                resolved, assets = resolve_krea_assets(config)
            self.assertTrue(resolved['model']['name_or_path'].endswith('raw.safetensors'))
            self.assertEqual(set(assets), {'raw', 'text_encoder', 'vae'})
            moved = {**config, 'local_model_assets': {k: f'/other/{k}' for k in config['local_model_assets']}}
            self.assertEqual(portable_process_config(config), portable_process_config(moved))

    def test_changed_content_and_unapproved_raw_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory))
            with self.assertRaises(ValueError):
                resolve_krea_assets(config)  # toy RAW is not the pinned official weight
            Path(config['local_model_assets']['text_encoder'], 'config.json').write_bytes(b'changed')
            with patch('extensions_built_in.streetscape_cnn.assets.PINNED_WEIGHTS', {}):
                with self.assertRaises(ValueError):
                    resolve_krea_assets(config)

    def test_unsafe_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory))
            from extensions_built_in.streetscape_cnn.contract import read_json
            manifest = read_json(config['local_model_assets']['manifest'])
            manifest['components']['raw']['files']['../outside'] = {'sha256': 'x'}
            write_json_atomic(config['local_model_assets']['manifest'], manifest)
            with patch('extensions_built_in.streetscape_cnn.assets.PINNED_WEIGHTS', {}):
                with self.assertRaises(ValueError):
                    resolve_krea_assets(config)
