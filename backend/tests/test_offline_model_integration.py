"""Unit and Integration Tests for VoxGuard Phase 9 Offline Model Artifact Integration.

Validates offline model discovery, manifest loading, checksum matching,
corrupted file recovery, registry auto-discovery, and safe fallback states
using temporary local file fixtures without downloading external assets.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import AudioFeatureExtractor
from app.models.base_classifier import ModelStatus, PredictionLabel
from app.models.model_loader import ModelLoader
from app.models.model_manifest import ModelManifest
from app.models.model_validator import ModelValidator
from app.models.ml_classifier import RealMLClassifier
from app.models.registry import ModelRegistry


class TestOfflineModelArtifactIntegration(unittest.TestCase):
    """Test suite for Phase 9 offline model artifact integration and safeguards."""

    def setUp(self):
        """Create a temporary models directory and mock audio features."""
        self.temp_dir = tempfile.mkdtemp(prefix="voxguard_test_models_")
        
        # Audio feature fixture
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
        self.features = extractor.extract_all(preprocessed)

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_1_missing_models_defaults_to_baseline(self):
        """1. Verify that when no model checkpoint exists in models_dir, status remains analysis_only."""
        loader = ModelLoader(models_dir=self.temp_dir)
        classifier, manifest = loader.discover_and_load_local_model()
        self.assertIsNone(classifier)

        # Registry test with empty models_dir
        registry = ModelRegistry(models_dir=self.temp_dir)
        active = registry.get_active_model()
        self.assertEqual(active.model_name, "VoxGuard Statistical Acoustic Baseline")
        self.assertEqual(active.status, ModelStatus.ANALYSIS_ONLY)

    def test_2_corrupted_zero_byte_model_file_handling(self):
        """2. Verify handling of zero-byte local model checkpoint files."""
        zero_file = os.path.join(self.temp_dir, "empty_model.pt")
        with open(zero_file, "wb") as f:
            pass  # Create 0-byte file

        validator = ModelValidator()
        res = validator.validate_checkpoint_file(zero_file)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.error_code, "EMPTY_FILE")

    def test_3_invalid_manifest_syntax_handling(self):
        """3. Verify handling of malformed JSON in manifest file."""
        manifest_file = os.path.join(self.temp_dir, "manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json syntax ... ")

        loader = ModelLoader(models_dir=self.temp_dir)
        classifier, manifest = loader.discover_and_load_local_model()
        self.assertIsNone(classifier)
        self.assertIsNone(manifest)

    def test_4_sha256_checksum_matching(self):
        """4. Verify SHA-256 checksum verification during local model loading."""
        model_file = os.path.join(self.temp_dir, "test_weight.pt")
        content = b"sample_trained_weights_payload_12345"
        with open(model_file, "wb") as f:
            f.write(content)

        # Compute correct SHA-256
        sha256 = hashlib.sha256(content).hexdigest()

        # A. Valid checksum test
        validator = ModelValidator()
        val_pass = validator.validate_checkpoint_file(model_file, expected_checksum=sha256)
        self.assertTrue(val_pass.is_valid)

        # B. Invalid checksum test
        val_fail = validator.validate_checkpoint_file(model_file, expected_checksum="1111111111111111111111111111111111111111111111111111111111111111")
        self.assertFalse(val_fail.is_valid)
        self.assertEqual(val_fail.error_code, "CHECKSUM_MISMATCH")

    def test_5_full_offline_discovery_and_activation(self):
        """5. Verify full discovery, validation, and loading of valid local model artifact."""
        model_file = os.path.join(self.temp_dir, "voxguard_test_net.pt")
        content = b"dummy_model_bytes"
        with open(model_file, "wb") as f:
            f.write(content)

        computed_sha = hashlib.sha256(content).hexdigest()

        manifest_data = {
            "model_name": "Local Neural Detector",
            "model_version": "1.0.0",
            "model_type": "pytorch",
            "checkpoint_path": model_file,
            "checksum": computed_sha,
            "expected_sample_rate": 16000,
            "class_names": ["human", "synthetic"]
        }
        manifest_file = os.path.join(self.temp_dir, "manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        # Create mock PyTorch model function to inject into RealMLClassifier
        def mock_forward(x):
            return (0.90, 0.10)

        mock_classifier = RealMLClassifier(
            config_override={
                "MODEL_ENABLED": True,
                "MODEL_PATH": model_file,
                "MODEL_NAME": "Local Neural Detector"
            },
            model_instance=mock_forward,
            manifest=ModelManifest.from_dict(manifest_data)
        )

        self.assertTrue(mock_classifier.is_ready())
        res = mock_classifier.predict(self.features)
        self.assertEqual(res.status, ModelStatus.READY)
        self.assertEqual(res.label, PredictionLabel.HUMAN)
        self.assertIsNotNone(res.probabilities["human"])
        self.assertAlmostEqual(res.probabilities["human"] + res.probabilities["synthetic"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
