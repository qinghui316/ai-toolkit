"""Server regression checks for low-frequency, record-preserving templates."""
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = [ROOT / "config/examples" / f"train_lora_krea2_streetscape_{kind}.yaml"
             for kind in ("base", "cnn")]


class SavePolicyTests(unittest.TestCase):
    def processes(self):
        for path in TEMPLATES:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            yield path.name, value["config"]["process"][0]

    def test_periodic_weights_are_infrequent_and_bounded(self):
        for name, process in self.processes():
            with self.subTest(template=name):
                self.assertEqual(process["save"]["save_every"], 1000)
                self.assertEqual(process["save"]["max_step_saves_to_keep"], 2)
                self.assertIs(process["save"]["push_to_hub"], False)

    def test_dense_records_and_complete_save_snapshots_remain_enabled(self):
        for name, process in self.processes():
            with self.subTest(template=name):
                self.assertEqual(process["logging"]["log_every"], 1)
                self.assertEqual(process["research_recording"], {
                    "enabled": True, "per_sample_scores": True, "archive_checkpoints": True})

    def test_templates_do_not_lock_unapproved_formal_parameters(self):
        for name, process in self.processes():
            with self.subTest(template=name):
                self.assertIsNone(process["train"]["steps"])
                self.assertIsNone(process["train"]["lr"])
                if process["cnn_guidance"]["enabled"]:
                    self.assertIsNone(process["cnn_guidance"]["weight"])
                    self.assertIsNone(process["cnn_guidance"]["start_step"])


if __name__ == "__main__":
    unittest.main()
