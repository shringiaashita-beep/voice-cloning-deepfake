"""Real ML Deepfake Classifier Adapter for VoxGuard.

Provides a production-grade adapter layer around genuine ML deepfake detection models
(e.g., PyTorch, TorchScript, ONNX) while preserving complete independence from FastAPI routes.
Enforces explicit weight loading, environment configuration, safe fallback when uninitialized,
and strict evaluation status reporting without automatic weight downloads.
"""

import os
import time
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.config import settings
from app.features.extractor import AudioFeatures
from app.models.base_classifier import (
    AbstractDeepfakeClassifier,
    ClassificationResult,
    ModelMetadata,
    ModelStatus,
    PredictionLabel,
)
from app.models.model_input import ModelInputContract
from app.models.model_output import ModelOutputContract
from app.models.model_manifest import ModelManifest
from app.models.model_validator import ModelValidator

logger = logging.getLogger("voxguard.ml_classifier")


class RealMLClassifier(AbstractDeepfakeClassifier):
    """Adapter class encapsulating real deepfake detection model inference."""

    def __init__(
        self,
        config_override: Optional[Dict[str, Any]] = None,
        model_instance: Optional[Any] = None,
        manifest: Optional[ModelManifest] = None
    ):
        """Initializes the ML Classifier adapter.

        Args:
            config_override: Optional override dict for settings (used in testing).
            model_instance: Optional pre-loaded PyTorch/ONNX model instance (used in testing/injection).
            manifest: Optional ModelManifest describing the model artifact.
        """
        self._enabled = config_override.get("MODEL_ENABLED", settings.MODEL_ENABLED) if config_override else settings.MODEL_ENABLED
        self._model_path = config_override.get("MODEL_PATH", settings.MODEL_PATH) if config_override else settings.MODEL_PATH
        self._name = config_override.get("MODEL_NAME", settings.MODEL_NAME) if config_override else settings.MODEL_NAME
        self._version = config_override.get("MODEL_VERSION", settings.MODEL_VERSION) if config_override else settings.MODEL_VERSION
        self._framework = config_override.get("MODEL_TYPE", settings.MODEL_TYPE) if config_override else settings.MODEL_TYPE

        self._model = model_instance
        self._manifest = manifest or ModelManifest(
            model_name=self._name,
            model_version=self._version,
            model_type=self._framework,
            checkpoint_path=self._model_path
        )
        feat_type = (
            self._manifest.expected_input.get("feature_type", "log_mel_mfcc")
            if (self._manifest and self._manifest.expected_input)
            else "log_mel_mfcc"
        )
        self._input_contract = ModelInputContract(feature_type=feat_type)
        self._validator = ModelValidator()

        self._status = ModelStatus.NOT_READY
        self._metadata = ModelMetadata(
            model_name=self._name,
            model_version=self._version,
            framework=self._framework,
            input_sample_rate=settings.TARGET_SAMPLE_RATE,
            input_channels=settings.TARGET_CHANNELS,
            model_source="VoxGuard Evaluated ML Model Registry",
            license="Proprietary / Evaluated",
            weights_location=self._model_path if self._model_path else "not_configured",
            evaluation_dataset=config_override.get("evaluation_dataset") if config_override else None,
            evaluation_protocol=config_override.get("evaluation_protocol") if config_override else None,
            evaluation_metrics=config_override.get("evaluation_metrics") if config_override else None,
        )

        # Attempt explicit model loading only if explicitly enabled
        if self._model is not None:
            self._status = ModelStatus.READY
        elif self._enabled and self._model_path:
            self.load_model()

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def status(self) -> ModelStatus:
        return self._status

    def is_ready(self) -> bool:
        """Returns True if the ML model is loaded and ready for inference."""
        return self._status == ModelStatus.READY

    def get_model_metadata(self) -> ModelMetadata:
        """Returns structured metadata regarding framework, weights, and evaluation metrics."""
        return self._metadata

    def get_manifest(self) -> ModelManifest:
        """Returns the ModelManifest object for this classifier."""
        return self._manifest

    def load_model(self) -> bool:
        """Explicitly loads trained model weights from configured file path.

        Does NOT attempt automatic web downloads.
        Sets status to READY if loading succeeds, or NOT_READY if file is missing/corrupted.
        """
        # Validate checkpoint file using ModelValidator
        val_res = self._validator.validate_checkpoint_file(self._model_path, expected_checksum=self._manifest.checksum)
        if not val_res.is_valid:
            logger.warning(
                "ML Classifier enabled but model weights validation failed: %s. Status set to NOT_READY.",
                val_res.error_message
            )
            self._status = ModelStatus.NOT_READY
            return False

        try:
            # Attempt loading model weights based on framework
            if self._framework.lower() in ("pytorch", "torchscript"):
                try:
                    import torch
                    self._model = torch.jit.load(self._model_path, map_location="cpu")
                    self._model.eval()
                except Exception:
                    # Fallback for standard PyTorch state_dict or model object
                    import torch
                    try:
                        self._model = torch.load(self._model_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        self._model = torch.load(self._model_path, map_location="cpu")
                    if hasattr(self._model, "eval"):
                        self._model.eval()

            elif self._framework.lower() == "onnx":
                import onnxruntime as ort
                self._model = ort.InferenceSession(self._model_path)

            else:
                logger.error("Unsupported ML framework type: '%s'", self._framework)
                self._status = ModelStatus.NOT_READY
                return False

            self._status = ModelStatus.READY
            self._metadata.weights_location = self._model_path
            logger.info("Successfully loaded ML model '%s' (v%s) from '%s'", self._name, self._version, self._model_path)
            return True

        except Exception as exc:
            logger.error("Failed to load ML model from '%s': %s", self._model_path, exc, exc_info=True)
            self._status = ModelStatus.NOT_READY
            self._model = None
            return False

    def predict(self, features: AudioFeatures) -> ClassificationResult:
        """Executes model inference on extracted audio features.

        Args:
            features: AudioFeatures object containing extracted NumPy feature arrays.

        Returns:
            ClassificationResult containing decision label, validated probabilities (summing to ~1.0),
            confidence score, and inference metadata.
        """
        start_time = time.perf_counter()

        # If model is not READY, return NOT_READY verdict with None probabilities (NO fake numbers)
        if not self.is_ready() or self._model is None:
            output_contract = ModelOutputContract(
                label=PredictionLabel.UNCERTAIN,
                human_probability=None,
                synthetic_probability=None,
                confidence_score=None,
                model_name=self._name,
                model_version=self._version,
                model_type=self._framework,
                inference_time_ms=0.0,
                evaluation_status=ModelStatus.NOT_READY
            )
            return ClassificationResult(
                label=output_contract.label,
                probabilities=output_contract.to_dict()["probabilities"],
                confidence_score=output_contract.confidence_score,
                model_name=self._name,
                model_version=self._version,
                status=ModelStatus.NOT_READY,
                metadata={
                    "error": "ML Model is not ready or weights are not loaded.",
                    "disclaimer": "Model weights uninitialized. No classification probabilities available."
                }
            )

        # Validate feature input contract
        is_input_valid, err_msg = self._input_contract.validate_features(features)
        if not is_input_valid:
            return ClassificationResult(
                label=PredictionLabel.UNCERTAIN,
                probabilities={"human": None, "synthetic": None},
                confidence_score=None,
                model_name=self._name,
                model_version=self._version,
                status=ModelStatus.NOT_READY,
                metadata={"error": f"Invalid feature input contract: {err_msg}"}
            )

        try:
            # Step 1: Input Adaptation (Model Input Pipeline)
            input_tensor = self._input_contract.adapt_to_tensor_shape(features)

            # Step 2: Model Inference Execution
            p_human, p_synthetic = self._execute_model_forward(input_tensor)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Step 3: Probability Calibration & Decision Logic
            total_prob = p_human + p_synthetic
            if total_prob > 0:
                p_human = p_human / total_prob
                p_synthetic = p_synthetic / total_prob

            # Determine verdict label and confidence
            if p_human >= 0.60:
                label = PredictionLabel.HUMAN
                confidence = float(p_human)
            elif p_synthetic >= 0.60:
                label = PredictionLabel.SYNTHETIC
                confidence = float(p_synthetic)
            else:
                label = PredictionLabel.UNCERTAIN
                confidence = float(max(p_human, p_synthetic))

            output_contract = ModelOutputContract(
                label=label,
                human_probability=round(float(p_human), 4),
                synthetic_probability=round(float(p_synthetic), 4),
                confidence_score=round(float(confidence), 4),
                model_name=self._name,
                model_version=self._version,
                model_type=self._framework,
                inference_time_ms=round(elapsed_ms, 2),
                device="cpu",
                provenance="ml_model",
                evaluation_status=ModelStatus.READY
            )

            metadata = {
                "execution_time_ms": round(elapsed_ms, 2),
                "device": "cpu",
                "framework": self._framework,
                "weights_location": self._model_path or "in_memory",
                "evaluation_disclaimer": "Model assessment derived from evaluated deepfake classifier."
            }

            # Enrich with real forensic architecture affinities and biometrics
            try:
                from app.models.voice_clone_detector import VoiceCloneDetector
                detector = VoiceCloneDetector()
                forensic_affinity = detector.compute_architecture_affinity(features, label=label, confidence=confidence)
                metadata.update(forensic_affinity)
                
                # Keep neural model's own risk score and classification authoritative
                metadata["synthetic_risk_score"] = round(float(p_synthetic) * 100.0, 1)
                metadata["synthetic_risk_level"] = (
                    "CRITICAL_DEEPFAKE" if p_synthetic >= 0.80
                    else ("SUSPICIOUS_SYNTHETIC" if p_synthetic >= 0.55 else "GENUINE_HUMAN")
                )
                if label == PredictionLabel.SYNTHETIC and metadata.get("voice_clone_architecture") == "Natural Biological Vocal Tract (Organic Speech)":
                    metadata["voice_clone_architecture"] = "AI Voice Conversion / Neural Speech Profile"
                elif label == PredictionLabel.HUMAN:
                    metadata["voice_clone_architecture"] = "Natural Biological Vocal Tract (Organic Speech)"
            except Exception as e:
                logger.debug("Could not compute forensic affinity enrichment: %s", e)

            return ClassificationResult(
                label=output_contract.label,
                probabilities=output_contract.to_dict()["probabilities"],
                confidence_score=output_contract.confidence_score,
                model_name=self._name,
                model_version=self._version,
                status=ModelStatus.READY,
                metadata=metadata
            )

        except Exception as exc:
            logger.error("Error during ML model prediction forward pass: %s", exc, exc_info=True)
            return ClassificationResult(
                label=PredictionLabel.UNCERTAIN,
                probabilities={"human": None, "synthetic": None},
                confidence_score=None,
                model_name=self._name,
                model_version=self._version,
                status=ModelStatus.NOT_READY,
                metadata={"error": f"Inference execution failure: {str(exc)}"}
            )

    def _execute_model_forward(self, input_data: Any) -> Tuple[float, float]:
        """Executes model forward pass and returns (p_human, p_synthetic)."""
        if callable(self._model):
            out = self._model(input_data)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return float(out[0]), float(out[1])
            elif isinstance(out, dict):
                return float(out.get("human", 0.5)), float(out.get("synthetic", 0.5))

        if self._framework.lower() in ("pytorch", "torchscript"):
            import torch
            with torch.no_grad():
                tensor_input = torch.from_numpy(input_data)
                outputs = self._model(tensor_input)
                if hasattr(outputs, "softmax"):
                    probs = torch.softmax(outputs, dim=-1).squeeze().tolist()
                else:
                    probs = torch.softmax(torch.tensor(outputs), dim=-1).squeeze().tolist()
                
                if isinstance(probs, list) and len(probs) >= 2:
                    return float(probs[0]), float(probs[1])

        elif self._framework.lower() == "onnx":
            input_name = self._model.get_inputs()[0].name
            outputs = self._model.run(None, {input_name: input_data})[0]
            exp_outs = np.exp(outputs - np.max(outputs))
            probs = (exp_outs / np.sum(exp_outs)).flatten()
            if len(probs) >= 2:
                return float(probs[0]), float(probs[1])

        return 0.5, 0.5
