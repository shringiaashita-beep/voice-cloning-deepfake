"""Model Registry Module for VoxGuard.

Provides a pluggable model registry allowing seamless switching between:
- Statistical Acoustic Baseline (default)
- Classical ML Classifiers (SVM / Random Forest)
- Deep Neural Networks (CNN / ResNet / RawNet)
- Pretrained Speech Foundation Models (Wav2Vec2 / HuBERT)
without requiring changes to FastAPI route handlers or frontend clients.
"""

from typing import Dict, Optional

from app.config import settings
from app.models.base_classifier import AbstractDeepfakeClassifier, ModelStatus
from app.models.baseline_classifier import BaselineClassifier
from app.models.ml_classifier import RealMLClassifier
from app.models.model_loader import ModelLoader


class ModelRegistry:
    """Model Registry managing classifier registration and active model selection."""

    def __init__(self, models_dir: str = "models"):
        self._models: Dict[str, AbstractDeepfakeClassifier] = {}
        
        # 1. Register default baseline model
        baseline = BaselineClassifier()
        self.register_model("baseline", baseline)

        # 2. Discover and load local ML classifier artifact using ModelLoader
        loader = ModelLoader(models_dir=models_dir)
        discovered_classifier, manifest = loader.discover_and_load_local_model()

        if discovered_classifier and discovered_classifier.is_ready():
            self.register_model("ml_classifier", discovered_classifier)
        else:
            ml_classifier = RealMLClassifier(manifest=manifest)
            self.register_model("ml_classifier", ml_classifier)
        self._active_model_id: str = "baseline"

        # 3. Register Voice Clone Deepfake Detector
        from app.models.voice_clone_detector import VoiceCloneDetector
        voice_clone_detector = VoiceCloneDetector()
        self.register_model("voice_clone_detector", voice_clone_detector)

    def register_model(self, model_id: str, classifier: AbstractDeepfakeClassifier) -> None:
        """Registers a classifier instance under a unique identifier string."""
        self._models[model_id] = classifier

    def get_model(self, model_id: str) -> AbstractDeepfakeClassifier:
        """Returns the classifier instance registered under the given ID."""
        if model_id not in self._models:
            raise KeyError(f"Model ID '{model_id}' is not registered in ModelRegistry.")
        return self._models[model_id]

    def set_active_model(self, model_id: str) -> None:
        """Sets the active classifier model to be invoked for predictions."""
        if model_id not in self._models:
            raise KeyError(f"Model ID '{model_id}' is not registered in ModelRegistry.")
        self._active_model_id = model_id

    def get_active_model(self) -> AbstractDeepfakeClassifier:
        """Returns the currently active classifier instance."""
        return self._models[self._active_model_id]

    def list_models(self) -> Dict[str, Dict[str, str]]:
        """Returns metadata summary of all registered models and active status."""
        return {
            m_id: {
                "name": instance.model_name,
                "version": instance.model_version,
                "status": instance.status.value,
                "is_active": (m_id == self._active_model_id)
            }
            for m_id, instance in self._models.items()
        }


# Instantiate global model registry
model_registry = ModelRegistry()

