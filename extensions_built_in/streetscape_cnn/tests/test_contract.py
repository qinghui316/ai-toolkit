"""Server tests; written locally but intentionally not executed on Windows."""
import json
from pathlib import Path
import tempfile
import unittest

from extensions_built_in.streetscape_cnn.contract import (
    GuidanceConfig, check_or_create_identity, file_sha256, object_sha256,
    portable_process_config, scan_caption_pairs, validate_process, validate_resume_pair)


def enabled_config(**overrides):
    value = dict(enabled=True, checkpoint="final.pt", manifest="model_manifest.json", weight=.1,
                 start_step=5, sigma_min=.1, sigma_max=.35, every_n_steps=2)
    value.update(overrides)
    return GuidanceConfig.parse(value)


class ContractTests(unittest.TestCase):
    def test_off_requires_no_assets_or_experiment_values(self):
        self.assertFalse(GuidanceConfig.parse({}).active_at(100))

    def test_missing_invalid_and_unknown_configuration(self):
        for value in ({"enabled": True}, {"enabled": "false"}, {"every_n_steps": 0}, {"typo": 1}):
            with self.assertRaises(ValueError):
                GuidanceConfig.parse(value)
        for key, value in (("weight", 0), ("weight", float("nan")), ("start_step", -1),
                           ("sigma_max", 1), ("sigma_min", .5), ("every_n_steps", True)):
            with self.assertRaises(ValueError):
                enabled_config(**{key: value})

    def test_step_gate_uses_start_relative_interval(self):
        config = enabled_config()
        self.assertFalse(config.active_at(4))
        self.assertTrue(config.active_at(5))
        self.assertFalse(config.active_at(6))
        self.assertTrue(config.active_at(7))

    def test_wrong_model_and_other_objective_are_rejected(self):
        for model in ({"arch": "flux", "name_or_path": "krea/Krea-2-Raw"},
                      {"arch": "krea2", "name_or_path": "krea/Krea-2-Turbo"},
                      {"arch": "krea2", "name_or_path": "raw.safetensors"}):
            with self.assertRaises(ValueError):
                validate_process({"model": model}, enabled_config())
        validate_process({}, GuidanceConfig())

    def test_pairs_duplicates_missing_and_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CS_T01_001.png").write_bytes(b"image-one")
            (root / "CS_T01_001.txt").write_text("A lane.", encoding="utf-8")
            self.assertEqual(scan_caption_pairs(root)["pair_count"], 1)
            (root / "unpaired.png").write_bytes(b"another-image")
            with self.assertRaises(ValueError):
                scan_caption_pairs(root)
            self.assertEqual(len(scan_caption_pairs(root, allow_unpaired=True)["unpaired_images"]), 1)
            (root / "unpaired.txt").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                scan_caption_pairs(root)
            (root / "unpaired.txt").write_text("Another lane.", encoding="utf-8")
            (root / "unpaired.png").write_bytes(b"image-one")
            with self.assertRaises(ValueError):
                scan_caption_pairs(root)

    def test_duplicate_normalized_ids_even_with_different_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stem, payload in (("CS_T01_001", b"one"), ("prefix_CS_T01_001", b"two")):
                (root / (stem + ".png")).write_bytes(payload)
                (root / (stem + ".txt")).write_text("A street.", encoding="utf-8")
            with self.assertRaises(ValueError):
                scan_caption_pairs(root)

    def test_portable_identity_keeps_training_and_discards_host_paths(self):
        left = {"training_folder": "E:/runs", "model": {"name_or_path": "E:/raw"},
                "datasets": [{"folder_path": "E:/data", "resolution": 512}], "train": {"lr": .1}}
        right = json.loads(json.dumps(left))
        right["training_folder"] = "/workspace/runs"
        right["model"]["name_or_path"] = "/workspace/raw"
        right["datasets"][0]["folder_path"] = "/workspace/data"
        self.assertEqual(portable_process_config(left), portable_process_config(right))
        right["train"]["lr"] = .2
        self.assertNotEqual(portable_process_config(left), portable_process_config(right))

    def test_identity_and_latest_pair_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            identity = {"guidance": {"enabled": False}}
            self.assertFalse(check_or_create_identity(root, identity))
            self.assertTrue(check_or_create_identity(root, identity))
            with self.assertRaises(ValueError):
                check_or_create_identity(root, {"guidance": {"enabled": True}})
            with self.assertRaises(FileNotFoundError):
                validate_resume_pair(root, object_sha256(identity))
            (root / "run.safetensors").write_bytes(b"lora")
            (root / "optimizer.pt").write_bytes(b"optimizer")
            record = {"identity_sha256": object_sha256(identity), "step": 6,
                      "lora": {"file": "run.safetensors", "sha256": file_sha256(root / "run.safetensors")},
                      "optimizer": {"file": "optimizer.pt", "sha256": file_sha256(root / "optimizer.pt")}}
            (root / "streetscape_latest.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(validate_resume_pair(root, object_sha256(identity))["step"], 6)
            (root / "optimizer.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_resume_pair(root, object_sha256(identity))
