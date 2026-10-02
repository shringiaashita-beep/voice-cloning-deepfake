"""Unit tests for VoxGuard Forensic Report Builder Module.

Verifies report categories (audio metadata, acoustic observations, model assessment),
evidence indicators, explicit provenance tags ("metadata", "signal_analysis", "heuristic", "ml_model"),
and JSON serialization safety.
Inherits from unittest.TestCase for standalone and pytest compatibility.
"""

import json
import unittest

import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import AudioFeatureExtractor
from app.forensics.report_builder import ForensicReport, ForensicReportBuilder
from app.models.base_classifier import ClassificationResult, ModelStatus, PredictionLabel
from app.models.baseline_classifier import BaselineClassifier


class TestReportBuilder(unittest.TestCase):
    """Test suite for ForensicReportBuilder service."""

    def setUp(self):
        self.classifier = BaselineClassifier()
        self.extractor = AudioFeatureExtractor()
        self.report_builder = ForensicReportBuilder()

    def _generate_test_data(self):
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        preprocessed = PreprocessedAudio(
            pcm_data=pcm, sample_rate=16000, channels=1, num_frames=16000, duration_seconds=1.0,
            metadata={"format": "wav"}
        )
        features = self.extractor.extract_all(preprocessed)
        classification = self.classifier.predict(features)
        return preprocessed, features, classification

    def test_report_schema_and_categories(self):
        preprocessed, features, classification = self._generate_test_data()
        report: ForensicReport = self.report_builder.build_report(
            classification=classification, features=features, preprocessed=preprocessed
        )

        # 1. Audio Metadata Category
        self.assertEqual(report.audio_metadata["sample_rate_hz"], 16000)
        self.assertEqual(report.audio_metadata["channels"], 1)
        self.assertEqual(report.audio_metadata["duration_seconds"], 1.0)

        # 2. Acoustic Observations Category
        self.assertIn("spectral_centroid", report.acoustic_observations)
        self.assertIn("spectral_rolloff_85_percent", report.acoustic_observations)
        self.assertIn("mfcc_statistics", report.acoustic_observations)
        self.assertIn("signal_characteristics", report.acoustic_observations)

        # 3. Model Assessment Category MUST BE NONE when status is "analysis_only"
        self.assertIsNone(report.model_assessment)

    def test_model_assessment_present_only_when_ready(self):
        preprocessed, features, _ = self._generate_test_data()

        # Simulate ready ML model classification result
        ready_classification = ClassificationResult(
            label=PredictionLabel.HUMAN,
            probabilities={"human": 0.95, "synthetic": 0.05},
            confidence_score=0.92,
            model_name="Trained ResNet Audio Classifier",
            model_version="1.0.0",
            status=ModelStatus.READY,
            metadata={"device": "cuda"}
        )

        report = self.report_builder.build_report(
            classification=ready_classification, features=features, preprocessed=preprocessed
        )

        # Model assessment MUST be present when status is "ready"
        self.assertIsNotNone(report.model_assessment)
        self.assertEqual(report.model_assessment["verdict"], "human")
        self.assertEqual(report.model_assessment["confidence_score"], 0.92)

    def test_evidence_indicators_and_provenance_tags(self):
        preprocessed, features, classification = self._generate_test_data()
        report = self.report_builder.build_report(
            classification=classification, features=features, preprocessed=preprocessed
        )

        indicators = report.evidence_indicators
        self.assertGreater(len(indicators), 0)

        # Verify every indicator has a valid provenance field from approved list
        valid_provenances = {"metadata", "signal_analysis", "heuristic", "ml_model"}
        for ind in indicators:
            self.assertIn(ind.provenance, valid_provenances)
            self.assertIsInstance(ind.is_model_derived, bool)

        # Find specific provenance items
        meta_inds = [ind for ind in indicators if ind.provenance == "metadata"]
        signal_inds = [ind for ind in indicators if ind.provenance == "signal_analysis"]
        heuristic_inds = [ind for ind in indicators if ind.provenance == "heuristic"]

        self.assertGreater(len(meta_inds), 0)
        self.assertGreater(len(signal_inds), 0)
        self.assertGreater(len(heuristic_inds), 0)

    def test_json_serialization_safety(self):
        preprocessed, features, classification = self._generate_test_data()
        report = self.report_builder.build_report(
            classification=classification, features=features, preprocessed=preprocessed
        )

        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        self.assertIsInstance(json_str, str)
        self.assertIn("audio_metadata", report_dict)
        self.assertIn("evidence_indicators", report_dict)


if __name__ == "__main__":
    unittest.main()
