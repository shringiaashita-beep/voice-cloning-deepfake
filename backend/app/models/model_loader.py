"""Offline Model Artifact Discovery & Loader Service for VoxGuard.

Provides safe, offline model artifact discovery from local directories (e.g., models/),
manifest parsing, checksum verification, and RealMLClassifier adapter initialization
without downloading external network assets.
"""

import json
import os
import logging
from typing import Any, Dict, Optional, Tuple

from app.config import settings
from app.models.model_manifest import ModelManifest
from app.models.model_validator import ModelValidator
from app.models.ml_classifier import RealMLClassifier

logger = logging.getLogger("voxguard.model_loader")


class ModelLoader:
    """Service discovering, validating, and instantiating local ML model artifacts."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.validator = ModelValidator()

    def discover_and_load_local_model(
        self,
        config_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[RealMLClassifier], Optional[ModelManifest]]:
        """Discovers and attempts loading a local model artifact and manifest.

        Args:
            config_override: Optional configuration dictionary.

        Returns:
            Tuple of (classifier_instance_or_none, manifest_or_none).
        """
        model_enabled = config_override.get("MODEL_ENABLED", settings.MODEL_ENABLED) if config_override else settings.MODEL_ENABLED
        model_path = config_override.get("MODEL_PATH", settings.MODEL_PATH) if config_override else settings.MODEL_PATH

        if not model_enabled:
            logger.info("ML Classifier disabled by configuration (MODEL_ENABLED=False).")
            return None, None

        # If a custom models_dir is specified (e.g. isolated test directory), search exclusively within it
        if self.models_dir and self.models_dir != "models":
            manifest_path = os.path.join(self.models_dir, "manifest.json")
            model_path = os.path.join(self.models_dir, os.path.basename(model_path) if model_path else "best_model.pt")
        else:
            manifest_path = None
            if os.path.exists(os.path.join(self.models_dir, "manifest.json")):
                manifest_path = os.path.join(self.models_dir, "manifest.json")
            elif model_path and os.path.exists(os.path.join(os.path.dirname(model_path), "manifest.json")):
                manifest_path = os.path.join(os.path.dirname(model_path), "manifest.json")

        manifest: Optional[ModelManifest] = None
        if manifest_path and os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    manifest = ModelManifest.from_dict(manifest_data)
                    logger.info("Loaded model manifest from '%s'", manifest_path)
            except Exception as exc:
                logger.warning("Failed to parse model manifest at '%s': %s", manifest_path, exc)

        # Resolve checkpoint file path portably across runtimes
        if not model_path or not os.path.exists(model_path):
            candidates = []
            if manifest and manifest.checkpoint_path:
                candidates.append(manifest.checkpoint_path)
                if manifest_path:
                    candidates.append(os.path.join(os.path.dirname(manifest_path), os.path.basename(manifest.checkpoint_path)))
                    candidates.append(os.path.join(os.path.dirname(manifest_path), manifest.checkpoint_path))
            if self.models_dir and self.models_dir != "models":
                candidates.append(os.path.join(self.models_dir, "best_model.pt"))
            else:
                candidates.extend([
                    os.path.join(self.models_dir, "best_model.pt"),
                    os.path.join("models", "best_model.pt"),
                    os.path.join("backend", "models", "best_model.pt"),
                    os.path.join("..", "models", "best_model.pt")
                ])
            for cand in candidates:
                if cand and os.path.exists(cand):
                    model_path = cand
                    break

        # Validate checkpoint file
        if not model_path or not os.path.exists(model_path):
            logger.warning("No valid model checkpoint exists at '%s'. Cannot load local model weights.", model_path)
            return None, manifest

        val_checkpoint = self.validator.validate_checkpoint_file(
            model_path,
            expected_checksum=manifest.checksum if manifest else None
        )
        if not val_checkpoint.is_valid:
            logger.warning("Local model checkpoint validation failed: %s", val_checkpoint.error_message)
            return None, manifest

        # If manifest exists, validate manifest integrity
        if manifest:
            val_manifest = self.validator.validate_manifest(manifest)
            if not val_manifest.is_valid:
                logger.warning("Local model manifest validation failed: %s", val_manifest.error_message)
                return None, manifest

        # Construct RealMLClassifier instance
        try:
            classifier = RealMLClassifier(
                config_override=config_override,
                manifest=manifest
            )
            if classifier.is_ready():
                logger.info("Successfully loaded and activated local ML classifier '%s'.", classifier.model_name)
                return classifier, manifest
            else:
                logger.warning("RealMLClassifier initialization resulted in status NOT_READY.")
                return None, manifest

        except Exception as exc:
            logger.error("Error creating RealMLClassifier instance: %s", exc, exc_info=True)
            return None, manifest
