"""Unit and Integration Tests for VoxGuard Real ML Classifier & Registry.

Validates model adapter configuration, fallback states, model registry integration,
probability summation, metadata extraction, provenance tags, and API safety rules
without requiring external downloaded weights.
"""

import unittest
import numpy as np

from app.config import settings
from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import AudioFeatureExtractor, AudioFeatures
from app.forensics.report_builder import ForensicReportBuilder
from app.models.base_classifier import (
    ClassificationResult,
    ModelMetadata,
    ModelStatus,
    PredictionLabel,
)
from app.models.baseline_classifier import BaselineClassifier
from app.models.ml_classifier import RealMLClassifier
from app.models.registry import ModelRegistry


class TestRealMLClassifierAdapter(unittest.TestCase):
    """Test suite for RealMLClassifier adapter behavior and safeguards."""

    def setUp(self):
        """Build mock canonical features for testing."""
        self.extractor = AudioFeatureExtractor()
        # Generate 1 second 16kHz mono sine wave PCM float32 array
        t = np.linspace(0, 1.0, 16000, dtype=np.float32)
        sine_pcm = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        self.preprocessed = PreprocessedAudio(
            pcm_data=sine_pcm,
            sample_rate=16000,
            channels=1,
            num_frames=16000,
            duration_seconds=1.0,
            metadata={"dtype": "float32"}
        )
        self.features = self.extractor.extract_all(self.preprocessed)

    def test_model_disabled_defaults_to_not_ready(self):
        """1. Verify that when MODEL_ENABLED=False, status returns NOT_READY."""
        classifier = RealMLClassifier(config_override={"MODEL_ENABLED": False, "MODEL_PATH": ""})
        self.assertFalse(classifier.is_ready())
        self.assertEqual(classifier.status, ModelStatus.NOT_READY)

    def test_model_unavailable_path_defaults_to_not_ready(self):
        """2. Verify that when MODEL_ENABLED=True but path does not exist, status returns NOT_READY."""
        classifier = RealMLClassifier(config_override={
            "MODEL_ENABLED": True,
            "MODEL_PATH": "/non_existent_path/fake_model.pt"
        })
        self.assertFalse(classifier.is_ready())
        self.assertEqual(classifier.status, ModelStatus.NOT_READY)

    def test_invalid_model_path_handling(self):
        """3. Verify load_model returns False gracefully on non-existent path without throwing."""
        classifier = RealMLClassifier(config_override={
            "MODEL_ENABLED": True,
            "MODEL_PATH": "invalid_file_path.bin"
        })
        success = classifier.load_model()
        self.assertFalse(success)
        self.assertEqual(classifier.status, ModelStatus.NOT_READY)

    def test_model_metadata_structure(self):
        """4. Verify model metadata structure and provenance fields."""
        classifier = RealMLClassifier(config_override={
            "MODEL_NAME": "Test Net",
            "MODEL_VERSION": "2.0.0",
            "MODEL_TYPE": "pytorch"
        })
        meta = classifier.get_model_metadata()
        self.assertIsInstance(meta, ModelMetadata)
        self.assertEqual(meta.model_name, "Test Net")
        self.assertEqual(meta.model_version, "2.0.0")
        self.assertEqual(meta.framework, "pytorch")
        self.assertEqual(meta.input_sample_rate, 16000)
        self.assertEqual(meta.input_channels, 1)

    def test_predict_returns_none_probabilities_when_not_ready(self):
        """8 & 9. Verify prediction when NOT_READY outputs None probabilities."""
        classifier = RealMLClassifier(config_override={"MODEL_ENABLED": False})
        res = classifier.predict(self.features)
        self.assertEqual(res.status, ModelStatus.NOT_READY)
        self.assertEqual(res.label, PredictionLabel.UNCERTAIN)
        self.assertIsNone(res.confidence_score)
        self.assertEqual(res.probabilities, {"human": None, "synthetic": None})

    def test_deterministic_mock_ready_model(self):
        """6, 7 & 8. Verify READY model inference, schema, and probability summation."""
        # Create a mock model function returning (0.85 human, 0.15 synthetic)
        def mock_forward(x):
            return (0.85, 0.15)

        classifier = RealMLClassifier(
            config_override={
                "MODEL_ENABLED": True,
                "MODEL_NAME": "Mock Deepfake Model",
                "MODEL_VERSION": "1.0.0"
            },
            model_instance=mock_forward
        )

        self.assertTrue(classifier.is_ready())
        self.assertEqual(classifier.status, ModelStatus.READY)

        res = classifier.predict(self.features)
        self.assertEqual(res.status, ModelStatus.READY)
        self.assertEqual(res.label, PredictionLabel.HUMAN)
        self.assertIsNotNone(res.probabilities)
        self.assertIsNotNone(res.confidence_score)

        # Check probability sum approximately equals 1.0
        p_human = res.probabilities["human"]
        p_synthetic = res.probabilities["synthetic"]
        self.assertAlmostEqual(p_human + p_synthetic, 1.0, places=3)
        self.assertGreaterEqual(p_human, 0.80)

    def test_model_registry_integration_and_selection(self):
        """5 & 13. Verify ModelRegistry registration, active model selection, and fallback."""
        registry = ModelRegistry()
        models = registry.list_models()
        self.assertIn("baseline", models)
        self.assertIn("ml_classifier", models)

        # Default fallback when no weights exist should be baseline
        active = registry.get_active_model()
        self.assertEqual(active.model_name, "VoxGuard Statistical Acoustic Baseline")

        # Explicitly switch active model to mock ready model
        mock_classifier = RealMLClassifier(
            config_override={"MODEL_NAME": "Active Mock Net"},
            model_instance=lambda x: (0.10, 0.90)
        )
        registry.register_model("mock_ml", mock_classifier)
        registry.set_active_model("mock_ml")

        new_active = registry.get_active_model()
        self.assertEqual(new_active.model_name, "Active Mock Net")
        self.assertEqual(new_active.status, ModelStatus.READY)

    def test_forensic_report_provenance_integration(self):
        """10, 11 & 12. Verify forensic report tags and model assessment presence."""
        builder = ForensicReportBuilder()
        
        # A. Test Baseline (ANALYSIS_ONLY)
        baseline = BaselineClassifier()
        base_res = baseline.predict(self.features)
        base_report = builder.build_report(base_res, self.features, self.preprocessed)
        
        self.assertIsNone(base_report.model_assessment)
        provenances = [ind.provenance for ind in base_report.evidence_indicators]
        self.assertIn("metadata", provenances)
        self.assertIn("signal_analysis", provenances)
        self.assertIn("heuristic", provenances)
        self.assertNotIn("ml_model", provenances)

        # B. Test READY ML Model
        mock_classifier = RealMLClassifier(
            config_override={"MODEL_NAME": "Evaluated Neural Detector"},
            model_instance=lambda x: (0.05, 0.95)
        )
        ml_res = mock_classifier.predict(self.features)
        ml_report = builder.build_report(ml_res, self.features, self.preprocessed)

        self.assertIsNotNone(ml_report.model_assessment)
        self.assertEqual(ml_report.model_assessment["verdict"], "synthetic")
        ml_provenances = [ind.provenance for ind in ml_report.evidence_indicators]
        self.assertIn("ml_model", ml_provenances)

        # Check explicit ML indicator properties
        ml_indicators = [ind for ind in ml_report.evidence_indicators if ind.provenance == "ml_model"]
        self.assertTrue(len(ml_indicators) > 0)
        self.assertTrue(ml_indicators[0].is_model_derived)


if __name__ == "__main__":
    unittest.main()
