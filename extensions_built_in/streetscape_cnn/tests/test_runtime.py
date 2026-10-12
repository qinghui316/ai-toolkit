import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from extensions_built_in.streetscape_cnn.runtime import (
    ResearchLRSchedule, ResearchAbort, ResearchRecorder, recording_config, request_stop, stop_requested)
from extensions_built_in.streetscape_cnn.contract import object_sha256, write_json_atomic
from extensions_built_in.streetscape_cnn.trainer import StreetscapeCNNTrainer
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions_built_in.streetscape_cnn.contract import GuidanceConfig


class RuntimeTests(unittest.TestCase):
    def test_schedule_boundaries_and_disabled(self):
        cfg = {'enabled': True, 'stages': [{'start_step': 0, 'lr': .0001},
                                          {'start_step': 8, 'lr': .00015}, {'start_step': 16, 'lr': .0001}]}
        schedule = ResearchLRSchedule.parse(cfg, {'steps': 24, 'lr': .0001})
        self.assertEqual([schedule.rate(i) for i in [0, 7, 8, 15, 16, 23]],
                         [.0001, .0001, .00015, .00015, .0001, .0001])
        self.assertIsNone(ResearchLRSchedule.parse(None, {}).rate(0))

    def test_bad_schedule_and_recording(self):
        for stages in [[], [{'start_step': 1, 'lr': .0001}], [{'start_step': 0, 'lr': float('nan')}],
                       [{'start_step': 0, 'lr': .0001}, {'start_step': 0, 'lr': .00015}]]:
            with self.assertRaises(ValueError):
                ResearchLRSchedule.parse({'enabled': True, 'stages': stages}, {'steps': 24, 'lr': .0001})
        with self.assertRaises(ValueError):
            recording_config({'enabled': 'true'})

    def test_stop_is_identity_bound_and_loads_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = {'input': 'frozen'}
            write_json_atomic(Path(directory) / 'streetscape_identity.json', identity)
            request_stop(directory)
            self.assertTrue(stop_requested(directory, object_sha256(identity)))
            with self.assertRaises(ResearchAbort):
                stop_requested(directory, 'different')

    def test_records_distinguish_unscored_and_reject_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ResearchRecorder(directory, 'identity')
            recorder.append('samples', {'scored': False, 'raw_scores': None})
            value = json.loads((Path(directory) / 'records/samples.jsonl').read_text())
            self.assertIsNone(value['raw_scores'])
            with self.assertRaises(ValueError):
                recorder.append('steps', {'loss': float('nan')})

    def test_first_oom_is_not_a_runtimeerror_skip(self):
        trainer = StreetscapeCNNTrainer.__new__(StreetscapeCNNTrainer)
        trainer.cnn_guidance = GuidanceConfig()
        trainer.sd = SimpleNamespace(vae=None)
        trainer.optimizer = SimpleNamespace(param_groups=[])
        trainer.step_num = 0
        with patch.object(SDTrainer, 'hook_train_loop', side_effect=torch.cuda.OutOfMemoryError('test')):
            with self.assertRaises(ResearchAbort) as caught:
                trainer.hook_train_loop([])
        self.assertNotIsInstance(caught.exception, RuntimeError)

    def test_disabled_nonfinite_loss_is_not_substituted(self):
        trainer = StreetscapeCNNTrainer.__new__(StreetscapeCNNTrainer)
        trainer.cnn_guidance = GuidanceConfig()
        with patch.object(SDTrainer, 'calculate_loss', return_value=torch.tensor(float('nan'))):
            with self.assertRaises(ResearchAbort):
                trainer.calculate_loss(None, None, None, None, None)

    def test_periodic_save_uses_next_native_step_and_restores_live_index(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = StreetscapeCNNTrainer.__new__(StreetscapeCNNTrainer)
            trainer.accelerator = SimpleNamespace(is_main_process=True)
            trainer.is_grad_accumulation_step = False
            trainer.step_num = 4
            trainer._last_successful_step = 4
            trainer._recorder = object()
            trainer.save_root = directory
            trainer.job = SimpleNamespace(name='test')
            observations = []
            def native_save(step):
                observations.append((trainer.step_num, step, trainer._saving_lora.name))
            with patch.object(SDTrainer, 'save', side_effect=native_save):
                trainer.save(4)
            self.assertEqual(observations, [(5, 5, 'test_000000005.safetensors')])
            self.assertEqual(trainer.step_num, 4)
