"""Integration Tests for VoxGuard Training & Evaluation CLI Tools (scripts/train_model.py, scripts/evaluate_model.py).

Validates CLI argument parsing, dry-run flags, output report creation, and threshold search.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class TestTrainingCLI(unittest.TestCase):
    """Test suite for CLI training and evaluation scripts."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.features_dir = os.path.join(self.test_dir, "features")
        self.output_dir = os.path.join(self.test_dir, "models")
        self.report_dir = os.path.join(self.test_dir, "reports")
        self.manifest_path = os.path.join(self.test_dir, "manifest.csv")

        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

        rows = []
        rows.extend(self._create_synthetic_features("train", num_human=3, num_synth=3))
        rows.extend(self._create_synthetic_features("validation", num_human=2, num_synth=2))

        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"]
            )
            writer.writeheader()
            writer.writerows(rows)

        self.backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.project_root = os.path.abspath(os.path.join(self.backend_dir, ".."))
        self.train_script = os.path.join(self.project_root, "scripts", "train_model.py")
        self.evaluate_script = os.path.join(self.project_root, "scripts", "evaluate_model.py")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_features(self, split: str, num_human: int, num_synth: int):
        split_dir = os.path.join(self.features_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        rows = []
        idx = 0
        for label, count in [(0, num_human), (1, num_synth)]:
            label_str = "human" if label == 0 else "synthetic"
            for i in range(count):
                idx += 1
                fname = f"sample_{idx}_{label_str}.npz"
                rel_path = os.path.join(split, fname)
                abs_path = os.path.join(self.features_dir, rel_path)

                log_mel = np.random.randn(80, 300).astype(np.float32)
                mfcc = np.random.randn(20, 300).astype(np.float32)
                centroid = np.random.randn(1, 300).astype(np.float32)
                rolloff = np.random.randn(1, 300).astype(np.float32)

                np.savez_compressed(
                    abs_path,
                    log_mel=log_mel,
                    mfcc=mfcc,
                    spectral_centroid=centroid,
                    spectral_rolloff=rolloff,
                    label=np.array(label, dtype=np.int64),
                    speaker_id=f"spk_{idx}_{split}",
                    sample_rate=16000
                )

                rows.append({
                    "feature_path": rel_path,
                    "label": label_str,
                    "speaker_id": f"spk_{idx}_{split}",
                    "source": "synthetic_test",
                    "source_id": f"src_{idx}",
                    "split": split
                })
        return rows

    def test_train_model_cli_dry_run(self):
        """Verify train_model.py --dry-run returns exit code 0."""
        cmd = [
            sys.executable,
            self.train_script,
            "--manifest", self.manifest_path,
            "--feature-dir", self.features_dir,
            "--output-dir", self.output_dir,
            "--dry-run"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"train_model.py dry run failed: {res.stderr}")
        self.assertIn("DRY-RUN", res.stdout)

    def test_train_and_evaluate_cli_e2e(self):
        """Verify end-to-end execution of train_model.py and evaluate_model.py."""
        # Step 1: Run training for 1 epoch
        train_cmd = [
            sys.executable,
            self.train_script,
            "--manifest", self.manifest_path,
            "--feature-dir", self.features_dir,
            "--output-dir", self.output_dir,
            "--epochs", "1",
            "--batch-size", "4"
        ]
        res_train = subprocess.run(train_cmd, capture_output=True, text=True)
        self.assertEqual(res_train.returncode, 0, f"train_model.py execution failed: {res_train.stderr}")

        best_checkpoint = os.path.join(self.output_dir, "best_model.pt")
        self.assertTrue(os.path.exists(best_checkpoint))

        # Step 2: Run evaluation with threshold search and output report
        report_file = os.path.join(self.report_dir, "eval_report.json")
        eval_cmd = [
            sys.executable,
            self.evaluate_script,
            "--model-path", best_checkpoint,
            "--manifest", self.manifest_path,
            "--feature-dir", self.features_dir,
            "--splits", "validation",
            "--threshold-search",
            "--output-report", report_file
        ]
        res_eval = subprocess.run(eval_cmd, capture_output=True, text=True)
        self.assertEqual(res_eval.returncode, 0, f"evaluate_model.py execution failed: {res_eval.stderr}")

        # Step 3: Validate generated evaluation report JSON
        self.assertTrue(os.path.exists(report_file))
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metrics", data)
        self.assertIn("validation", data["metrics"])
        self.assertIn("applied_threshold", data)


if __name__ == "__main__":
    unittest.main()
