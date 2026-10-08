import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extensions_built_in.streetscape_cnn.contract import LABEL_COLUMNS, file_sha256, validate_cnn_files


class FrozenFileTests(unittest.TestCase):
    def make_files(self, root):
        checkpoint = root / "final.pt"
        checkpoint.write_bytes(b"trusted-model-fixture")
        config = {"model_variant": "shared-delta", "training_objective": "base",
                  "label_columns": LABEL_COLUMNS, "image_height": 224, "image_width": 336}
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        manifest = {"frozen": True, "sample_count": 400, "run_type": "full-development-final-v1",
                    "model_sha256": file_sha256(checkpoint), "config_sha256": file_sha256(root / "config.json"),
                    "config_path": "E:\\historical\\frozen\\config.json",
                    "experiment_identity_sha256": "a" * 64}
        (root / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # Only test fixtures override the two immutable production model/config hashes.
        for key, value in (("FROZEN_CNN_SHA256", manifest["model_sha256"]),
                           ("FROZEN_CONFIG_SHA256", manifest["config_sha256"])):
            override = patch(f"extensions_built_in.streetscape_cnn.contract.{key}", value)
            override.start()
            self.addCleanup(override.stop)
        return checkpoint, root / "model_manifest.json"

    def test_config_path_is_portable_and_content_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest = self.make_files(root)
            self.assertEqual(validate_cnn_files(checkpoint, manifest)["checkpoint_sha256"], file_sha256(checkpoint))
            (root / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_cnn_files(checkpoint, manifest)

    def test_changed_model_or_wrong_frozen_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest = self.make_files(root)
            checkpoint.write_bytes(b"other-weights")
            with self.assertRaises(ValueError):
                validate_cnn_files(checkpoint, manifest)
            checkpoint, manifest = self.make_files(root)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["sample_count"] = 160
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_cnn_files(checkpoint, manifest)

    def test_a_rewritten_manifest_cannot_substitute_another_cnn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest = self.make_files(root)
            checkpoint.write_bytes(b"replacement-cnn")
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["model_sha256"] = file_sha256(checkpoint)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_cnn_files(checkpoint, manifest)
