"""Unit tests for VoxGuard Classifier Interface, Statuses, and Model Registry.

Tests ModelStatus values ("ready", "analysis_only", "not_ready"),
ClassificationResult serialization, and ModelRegistry functionality.
Inherits from unittest.TestCase for standalone and pytest compatibility.
"""

import unittest

from app.features.extractor import AudioFeatures
from app.models.base_classifier import (
    AbstractDeepfakeClassifier,
    ClassificationResult,
    ModelStatus,
    PredictionLabel,
)
from app.models.registry import ModelRegistry, model_registry


class MockNotReadyClassifier(AbstractDeepfakeClassifier):
    """Mock classifier simulating model-unavailable state."""

    @property
    def model_name(self) -> str:
        return "Mock Uninitialized Model"

    @property
    def model_version(self) -> str:
        return "0.0.0-uninitialized"

    @property
    def status(self) -> ModelStatus:
        return ModelStatus.NOT_READY

    def get_model_metadata(self):
        from app.models.base_classifier import ModelMetadata
        return ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            framework="mock"
        )

    def predict(self, features: AudioFeatures) -> ClassificationResult:
        return ClassificationResult(
            label=PredictionLabel.UNCERTAIN,
            probabilities=None,
            confidence_score=None,
            model_name=self.model_name,
            model_version=self.model_version,
            status=self.status,
            metadata={"reason": "Model weights uninitialized."}
        )


class TestClassifierInterface(unittest.TestCase):
    """Test suite for classifier interface and model registry."""

    def test_model_not_ready_state(self):
        mock_model = MockNotReadyClassifier()
        self.assertEqual(mock_model.status, ModelStatus.NOT_READY)

        result = mock_model.predict(None)
        self.assertEqual(result.status, ModelStatus.NOT_READY)
        self.assertEqual(result.label, PredictionLabel.UNCERTAIN)
        self.assertIsNone(result.probabilities)
        self.assertIsNone(result.confidence_score)

    def test_classification_result_to_dict(self):
        result = ClassificationResult(
            label=PredictionLabel.UNCERTAIN,
            probabilities={"human": None, "synthetic": None},
            confidence_score=None,
            model_name="Baseline",
            model_version="0.1.0",
            status=ModelStatus.ANALYSIS_ONLY,
            metadata={"device": "cpu"}
        )

        res_dict = result.to_dict()
        self.assertEqual(res_dict["label"], "uncertain")
        self.assertEqual(res_dict["status"], "analysis_only")
        self.assertIsNone(res_dict["probabilities"]["human"])
        self.assertIsNone(res_dict["probabilities"]["synthetic"])

    def test_model_registry_registration_and_switching(self):
        reg = ModelRegistry()
        mock_model = MockNotReadyClassifier()

        reg.register_model("mock_nr", mock_model)
        models_list = reg.list_models()

        self.assertIn("baseline", models_list)
        self.assertIn("mock_nr", models_list)
        self.assertTrue(models_list["baseline"]["is_active"])

        # Switch active model
        reg.set_active_model("mock_nr")
        active_instance = reg.get_active_model()
        self.assertEqual(active_instance.model_name, "Mock Uninitialized Model")


if __name__ == "__main__":
    unittest.main()
