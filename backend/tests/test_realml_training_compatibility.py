"""Integration Tests for VoxGuard Trained Checkpoint & RealMLClassifier Compatibility.

Verifies end-to-end integration:
1. Model trained via Trainer saving best_model.pt and manifest.json.
2. Loading trained checkpoint into RealMLClassifier.
3. Successful inference execution yielding READY status and valid probability summation.
"""

import csv
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import AudioFeatureExtractor, AudioFeatures
from app.models.base_classifier import ModelStatus, PredictionLabel
from app.models.ml_classifier import RealMLClassifier
from app.training.config import TrainingConfig
from app.training.dataset import FeatureDataset
from app.training.sampler import TrainingDataLoader
from app.training.trainer import Trainer


class TestRealMLTrainingCompatibility(unittest.TestCase):
    """Integration test suite for RealMLClassifier loading models trained via Trainer."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.features_dir = os.path.join(self.test_dir, "features")
        self.output_dir = os.path.join(self.test_dir, "models")
        self.manifest_path = os.path.join(self.test_dir, "manifest.csv")

        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        rows = self._create_synthetic_features("train", num_human=4, num_synth=4)
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"]
            )
            writer.writeheader()
            writer.writerows(rows)

        self.config = TrainingConfig(BATCH_SIZE=4, RANDOM_SEED=42)
        self.train_dataset = FeatureDataset(self.manifest_path, feature_dir=self.features_dir, split="train")
        self.train_loader = TrainingDataLoader(self.train_dataset, config=self.config, shuffle=True)

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
                    speaker_id=f"spk_{idx}",
                    sample_rate=16000
                )

                rows.append({
                    "feature_path": rel_path,
                    "label": label_str,
                    "speaker_id": f"spk_{idx}",
                    "source": "synthetic_test",
                    "source_id": f"src_{idx}",
                    "split": split
                })
        return rows

    def test_trained_checkpoint_loads_and_predicts_in_realml(self):
        """Verify model trained via Trainer loads in RealMLClassifier and executes inference."""
        trainer = Trainer(output_dir=self.output_dir)
        summary = trainer.train(train_loader=self.train_loader, epochs=1)

        checkpoint_path = summary.best_checkpoint_path
        self.assertTrue(os.path.exists(checkpoint_path))

        # Instantiate RealMLClassifier using trained checkpoint
        classifier = RealMLClassifier(config_override={
            "MODEL_ENABLED": True,
            "MODEL_PATH": checkpoint_path
        })

        self.assertTrue(classifier.is_ready())
        self.assertEqual(classifier.status, ModelStatus.READY)

        # Generate AudioFeatures
        extractor = AudioFeatureExtractor()
        t = np.linspace(0, 1.0, 16000, dtype=np.float32)
        sine_pcm = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        preprocessed = PreprocessedAudio(
            pcm_data=sine_pcm,
            sample_rate=16000,
            channels=1,
            num_frames=16000,
            duration_seconds=1.0,
            metadata={"dtype": "float32"}
        )
        features = extractor.extract_all(preprocessed)

        # Execute prediction
        result = classifier.predict(features)

        self.assertEqual(result.status, ModelStatus.READY)
        self.assertIn(result.label, [PredictionLabel.HUMAN, PredictionLabel.SYNTHETIC, PredictionLabel.UNCERTAIN])
        self.assertIsNotNone(result.probabilities["human"])
        self.assertIsNotNone(result.probabilities["synthetic"])

        prob_sum = result.probabilities["human"] + result.probabilities["synthetic"]
        self.assertAlmostEqual(prob_sum, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
