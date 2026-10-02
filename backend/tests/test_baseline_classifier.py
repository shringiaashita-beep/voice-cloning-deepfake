"""Unit tests for VoxGuard Statistical Acoustic Baseline Classifier.

Explicitly verifies that the baseline classifier reports status ANALYSIS_ONLY,
CANNOT return a synthetic probability, output is deterministic, and handles
empty, short, silent, and malformed inputs cleanly.
Inherits from unittest.TestCase for standalone and pytest compatibility.
"""

import unittest

import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import AudioFeatureExtractor
from app.models.base_classifier import ModelStatus, PredictionLabel
from app.models.baseline_classifier import BaselineClassifier


class TestBaselineClassifier(unittest.TestCase):
    """Test suite for BaselineClassifier service."""

    def setUp(self):
        self.classifier = BaselineClassifier()
        self.extractor = AudioFeatureExtractor()

    def _extract_features_from_pcm(self, pcm: np.ndarray, duration_sec: float) -> PreprocessedAudio:
        preprocessed = PreprocessedAudio(
            pcm_data=pcm, sample_rate=16000, channels=1, num_frames=len(pcm), duration_seconds=duration_sec
        )
        return self.extractor.extract_all(preprocessed)

    def test_baseline_properties(self):
        self.assertEqual(self.classifier.model_name, "VoxGuard Statistical Acoustic Baseline")
        self.assertEqual(self.classifier.model_version, "0.1.0-baseline")
        self.assertEqual(self.classifier.status, ModelStatus.ANALYSIS_ONLY)

    def test_explicit_cannot_return_synthetic_probability(self):
        """EXPLICIT TEST: Verifies baseline NEVER outputs a synthetic probability score."""
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        features = self._extract_features_from_pcm(pcm, duration_sec=1.0)

        result = self.classifier.predict(features)

        # Assertion: Baseline status MUST be ANALYSIS_ONLY
        self.assertEqual(result.status, ModelStatus.ANALYSIS_ONLY)

        # Assertion: Verdict MUST be UNCERTAIN (never SYNTHETIC or HUMAN)
        self.assertEqual(result.label, PredictionLabel.UNCERTAIN)
        self.assertNotEqual(result.label, PredictionLabel.SYNTHETIC)
        self.assertNotEqual(result.label, PredictionLabel.HUMAN)

        # Assertion: Probabilities MUST be None or None-values (never float numbers like 0.87)
        self.assertEqual(result.probabilities, {"human": None, "synthetic": None})
        self.assertIsNone(result.probabilities["synthetic"])
        self.assertIsNone(result.probabilities["human"])
        self.assertIsNone(result.confidence_score)

    def test_silent_audio_input(self):
        # 1 sec of pure silence (zeros)
        silent_pcm = np.zeros(16000, dtype=np.float32)
        features = self._extract_features_from_pcm(silent_pcm, duration_sec=1.0)

        result = self.classifier.predict(features)

        self.assertEqual(result.status, ModelStatus.ANALYSIS_ONLY)
        self.assertEqual(result.label, PredictionLabel.UNCERTAIN)
        self.assertIn("acoustic_statistics", result.metadata)

    def test_short_audio_input(self):
        # 0.05 seconds of audio (800 samples at 16kHz)
        short_pcm = (0.3 * np.ones(800)).astype(np.float32)
        features = self._extract_features_from_pcm(short_pcm, duration_sec=0.05)

        result = self.classifier.predict(features)

        self.assertEqual(result.status, ModelStatus.ANALYSIS_ONLY)
        self.assertIn("spectral_centroid_mean_hz", result.metadata["acoustic_statistics"])

    def test_malformed_feature_input(self):
        # Passing None as features
        result = self.classifier.predict(None)

        self.assertEqual(result.status, ModelStatus.NOT_READY)
        self.assertEqual(result.label, PredictionLabel.UNCERTAIN)
        self.assertIsNone(result.probabilities)

    def test_deterministic_output(self):
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        pcm = (0.7 * np.cos(2 * np.pi * 880 * t)).astype(np.float32)
        features = self._extract_features_from_pcm(pcm, duration_sec=1.0)

        res1 = self.classifier.predict(features)
        res2 = self.classifier.predict(features)

        self.assertEqual(res1.label, res2.label)
        self.assertEqual(res1.probabilities, res2.probabilities)
        self.assertEqual(res1.status, res2.status)
        self.assertEqual(
            res1.metadata["acoustic_statistics"],
            res2.metadata["acoustic_statistics"]
        )


if __name__ == "__main__":
    unittest.main()
