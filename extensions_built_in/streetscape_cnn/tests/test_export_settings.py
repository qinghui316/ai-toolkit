import unittest
from types import SimpleNamespace
from toolkit.config_modules import SampleConfig
from tools.streetscape_cnn.server_export_inference import sampling_settings, zero_update_settings, native_sample_images
from tempfile import TemporaryDirectory
from pathlib import Path


class ExportSettingsTests(unittest.TestCase):
    def test_native_negative_prompt_is_text_and_three_prompts_preserved(self):
        args=SimpleNamespace(prompt=['street one','street two','street three'],width=768,height=512,steps=30,cfg=4.,seed=42)
        settings=sampling_settings(args)
        native=SampleConfig(**settings)
        self.assertEqual(native.prompts,args.prompt)
        self.assertEqual([sample.neg for sample in native.samples],['','',''])
        self.assertEqual(settings['seed'],42)
        self.assertFalse(settings['walk_seed'])

    def test_boolean_or_empty_prompt_rejected(self):
        for prompts in ([False],[''],[]):
            with self.assertRaises(ValueError):sampling_settings(SimpleNamespace(prompt=prompts))

    def test_native_zero_step_only_samples_once_at_end(self):
        value=zero_update_settings()
        self.assertEqual(value['steps'],0)
        self.assertTrue(value['skip_first_sample'])
        self.assertFalse(value['force_first_sample'])
        self.assertFalse(value['disable_sampling'])

    def test_native_thumbnails_are_not_extra_generated_images(self):
        with TemporaryDirectory() as root:
            folder=Path(root)/'samples'
            (folder/'.thumbs').mkdir(parents=True)
            for index in range(3):
                (folder/f'{index}.jpg').touch()
                (folder/'.thumbs'/f'{index}.jpg.jpg').touch()
            self.assertEqual(len(native_sample_images(root)),3)
