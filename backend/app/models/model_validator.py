"""Model Validation Service for VoxGuard.

Provides safe, passive model validation checking file existence, format support,
manifest integrity, tensor shape compatibility, class label specifications, and
SHA-256 checksums without executing untrusted code or auto-downloading external assets.
"""

import hashlib
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.models.model_input import ModelInputContract
from app.models.model_manifest import ModelManifest

logger = logging.getLogger("voxguard.model_validator")


@dataclass
class ValidationResult:
    """Strongly typed validation output result."""
    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ModelValidator:
    """Service performing safe model checkpoint, manifest, and tensor contract validation."""

    SUPPORTED_FORMATS: List[str] = [".pt", ".pth", ".onnx", ".bin", ".tflite"]

    def validate_checkpoint_file(
        self,
        filepath: str,
        expected_checksum: Optional[str] = None
    ) -> ValidationResult:
        """Validates that a model weight file exists, is readable, has a supported format, and matches SHA-256.

        Args:
            filepath: Absolute or relative path to model weights file.
            expected_checksum: Optional SHA-256 hex digest string.

        Returns:
            ValidationResult indicating pass/fail.
        """
        if not filepath or not isinstance(filepath, str):
            return ValidationResult(
                is_valid=False,
                error_code="MISSING_FILEPATH",
                error_message="Model file path is empty or invalid."
            )

        if not os.path.exists(filepath):
            return ValidationResult(
                is_valid=False,
                error_code="FILE_NOT_FOUND",
                error_message=f"Model weight file not found at path: '{filepath}'."
            )

        if not os.path.isfile(filepath):
            return ValidationResult(
                is_valid=False,
                error_code="NOT_A_FILE",
                error_message=f"Model path exists but is not a file: '{filepath}'."
            )

        # Extension check
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            return ValidationResult(
                is_valid=False,
                error_code="UNSUPPORTED_FORMAT",
                error_message=f"Model file format '{ext}' is unsupported. Supported: {self.SUPPORTED_FORMATS}"
            )

        # Check file readability
        try:
            with open(filepath, "rb") as f:
                header = f.read(1024)
                if len(header) == 0:
                    return ValidationResult(
                        is_valid=False,
                        error_code="EMPTY_FILE",
                        error_message="Model weight file is 0 bytes."
                    )
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                error_code="FILE_UNREADABLE",
                error_message=f"Cannot read model weight file: {str(exc)}"
            )

        # SHA-256 Checksum Verification
        if expected_checksum:
            try:
                sha256 = hashlib.sha256()
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256.update(chunk)
                computed_hash = sha256.hexdigest().lower()

                if computed_hash != expected_checksum.lower():
                    return ValidationResult(
                        is_valid=False,
                        error_code="CHECKSUM_MISMATCH",
                        error_message=f"Model checksum mismatch! Expected '{expected_checksum}', computed '{computed_hash}'."
                    )
            except Exception as exc:
                return ValidationResult(
                    is_valid=False,
                    error_code="CHECKSUM_COMPUTATION_FAILED",
                    error_message=f"Failed to calculate checksum: {str(exc)}"
                )

        return ValidationResult(is_valid=True)

    def validate_manifest(self, manifest: ModelManifest) -> ValidationResult:
        """Validates manifest completeness and compatibility.

        Args:
            manifest: ModelManifest object.

        Returns:
            ValidationResult object.
        """
        if manifest is None:
            return ValidationResult(
                is_valid=False,
                error_code="NULL_MANIFEST",
                error_message="ModelManifest object is None."
            )

        if not manifest.model_name or not manifest.model_version:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_MANIFEST_METADATA",
                error_message="ModelManifest missing required model_name or model_version."
            )

        # Validate class labels
        if not manifest.class_names or "human" not in manifest.class_names or "synthetic" not in manifest.class_names:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_CLASS_LABELS",
                error_message="ModelManifest class_names must contain 'human' and 'synthetic'."
            )

        # Validate expected audio parameters
        if manifest.expected_sample_rate != 16000:
            return ValidationResult(
                is_valid=False,
                error_code="INCOMPATIBLE_SAMPLE_RATE",
                error_message=f"Model expected sample rate {manifest.expected_sample_rate} Hz does not match target 16000 Hz."
            )

        return ValidationResult(is_valid=True)

    def validate_input_compatibility(
        self,
        input_contract: ModelInputContract,
        manifest: ModelManifest
    ) -> ValidationResult:
        """Validates that a model manifest matches the backend's ModelInputContract.

        Args:
            input_contract: ModelInputContract instance.
            manifest: ModelManifest instance.

        Returns:
            ValidationResult indicating whether input contracts align.
        """
        val_m = self.validate_manifest(manifest)
        if not val_m.is_valid:
            return val_m

        if input_contract.sample_rate != manifest.expected_sample_rate:
            return ValidationResult(
                is_valid=False,
                error_code="SAMPLE_RATE_MISMATCH",
                error_message=f"Input contract SR ({input_contract.sample_rate}) != manifest SR ({manifest.expected_sample_rate})."
            )

        if input_contract.channels != manifest.expected_channels:
            return ValidationResult(
                is_valid=False,
                error_code="CHANNELS_MISMATCH",
                error_message=f"Input contract channels ({input_contract.channels}) != manifest channels ({manifest.expected_channels})."
            )

        return ValidationResult(is_valid=True)
