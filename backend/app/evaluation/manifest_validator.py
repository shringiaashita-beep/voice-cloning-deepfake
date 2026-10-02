"""CSV Manifest Validation Service for VoxGuard.

Validates dataset CSV manifests checking column presence, label enums, speaker IDs,
source IDs, split designations, relative path sanitization, duplicate checks, and
path traversal prevention without exposing raw server filesystem paths.
"""

import csv
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("voxguard.manifest_validator")

REQUIRED_COLUMNS: List[str] = [
    "file_path",
    "label",
    "speaker_id",
    "source",
    "source_id",
    "split"
]

VALID_LABELS: Set[str] = {"human", "synthetic"}
VALID_SPLITS: Set[str] = {"train", "validation", "test", "ood"}


@dataclass
class ManifestRow:
    """Strongly typed representation of a single dataset manifest entry."""
    file_path: str
    label: str
    speaker_id: str
    source: str
    source_id: str
    split: str
    line_number: int = 0


@dataclass
class ManifestValidationResult:
    """Validation output summary object."""
    is_valid: bool
    total_samples: int = 0
    valid_rows: List[ManifestRow] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.is_valid,
            "total_samples": self.total_samples,
            "errors": self.errors,
            "warnings": self.warnings
        }


class ManifestValidator:
    """Validator inspecting dataset manifest CSV structure and row attributes."""

    def validate_csv_manifest(
        self,
        manifest_path: str,
        base_dir: Optional[str] = None,
        check_file_existence: bool = False
    ) -> ManifestValidationResult:
        """Validates a dataset CSV manifest file.

        Args:
            manifest_path: Path to dataset manifest CSV file.
            base_dir: Optional base directory to resolve relative file_path values against.
            check_file_existence: If True, verifies audio files exist on local disk.

        Returns:
            ManifestValidationResult containing errors, warnings, and valid rows.
        """
        errors: List[str] = []
        warnings: List[str] = []
        valid_rows: List[ManifestRow] = []

        if not manifest_path or not os.path.exists(manifest_path):
            return ManifestValidationResult(
                is_valid=False,
                errors=[f"Manifest CSV file not found: '{os.path.basename(manifest_path)}'"]
            )

        try:
            with open(manifest_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                
                # 1. Header Validation
                if not reader.fieldnames:
                    return ManifestValidationResult(
                        is_valid=False,
                        errors=["Manifest CSV file is empty or missing headers."]
                    )

                fieldnames_set = {col.strip().lower() for col in reader.fieldnames}
                for req_col in REQUIRED_COLUMNS:
                    if req_col not in fieldnames_set:
                        errors.append(f"Missing required column in CSV header: '{req_col}'")

                if errors:
                    return ManifestValidationResult(is_valid=False, errors=errors)

                seen_paths: Set[str] = set()

                # 2. Row-by-Row Validation
                for line_idx, row in enumerate(reader, start=2):
                    # Strip whitespace from keys and values
                    cleaned_row = {k.strip().lower(): v.strip() for k, v in row.items() if k}
                    
                    file_path = cleaned_row.get("file_path", "")
                    label = cleaned_row.get("label", "").lower()
                    speaker_id = cleaned_row.get("speaker_id", "")
                    source = cleaned_row.get("source", "")
                    source_id = cleaned_row.get("source_id", "")
                    split = cleaned_row.get("split", "").lower()

                    # Empty field check
                    missing_fields = []
                    if not file_path: missing_fields.append("file_path")
                    if not label: missing_fields.append("label")
                    if not speaker_id: missing_fields.append("speaker_id")
                    if not source: missing_fields.append("source")
                    if not source_id: missing_fields.append("source_id")
                    if not split: missing_fields.append("split")

                    if missing_fields:
                        errors.append(f"Line {line_idx}: Missing required values for {missing_fields}.")
                        continue

                    # Path Traversal & Absolute Path Checks
                    if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\") or ":" in file_path:
                        errors.append(f"Line {line_idx}: Unsafe or absolute file path format in 'file_path': '{file_path}'. Must be dataset-relative.")
                        continue

                    # Duplicate file_path check
                    norm_path = os.path.normpath(file_path).lower()
                    if norm_path in seen_paths:
                        errors.append(f"Line {line_idx}: Duplicate 'file_path' entry: '{file_path}'.")
                        continue
                    seen_paths.add(norm_path)

                    # Enum validation
                    if label not in VALID_LABELS:
                        errors.append(f"Line {line_idx}: Invalid label '{label}'. Must be one of {VALID_LABELS}.")
                        continue

                    if split not in VALID_SPLITS:
                        errors.append(f"Line {line_idx}: Invalid split '{split}'. Must be one of {VALID_SPLITS}.")
                        continue

                    # Local File Existence Check (Optional)
                    if check_file_existence:
                        resolved_path = os.path.join(base_dir, file_path) if base_dir else file_path
                        if not os.path.exists(resolved_path):
                            errors.append(f"Line {line_idx}: Audio file does not exist on disk: '{file_path}'.")
                            continue

                    valid_rows.append(ManifestRow(
                        file_path=file_path,
                        label=label,
                        speaker_id=speaker_id,
                        source=source,
                        source_id=source_id,
                        split=split,
                        line_number=line_idx
                    ))

        except Exception as exc:
            logger.error("Error reading manifest CSV file: %s", exc, exc_info=True)
            return ManifestValidationResult(
                is_valid=False,
                errors=[f"Failed to parse CSV manifest file: {str(exc)}"]
            )

        is_valid = len(errors) == 0 and len(valid_rows) > 0
        return ManifestValidationResult(
            is_valid=is_valid,
            total_samples=len(valid_rows),
            valid_rows=valid_rows,
            errors=errors,
            warnings=warnings
        )
