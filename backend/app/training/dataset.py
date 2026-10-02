"""Training Feature Dataset Loader for VoxGuard.

Loads pre-extracted .npz feature artifacts, validates array dtypes, ranks, and finiteness,
applies deterministic windowing to 256-frame segments, and yields strongly typed TrainingSample records.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.evaluation.labels import encode_label, decode_label
from app.training.config import TrainingConfig
from app.training.types import TrainingSample
from app.training.validation import TrainingBatchValidator, TrainingValidationError
from app.training.windowing import FeatureWindowing

try:
    import torch
    from torch.utils.data import Dataset as BaseDataset
    HAS_TORCH = True
except ImportError:
    BaseDataset = object
    HAS_TORCH = False


class DatasetLoaderError(Exception):
    """Raised when dataset loading, manifest parsing, or feature artifact validation fails."""
    pass


class FeatureDataset(BaseDataset):
    """Dataset loader reading variable-length .npz feature artifacts and outputting 256-frame TrainingSample items."""

    def __init__(
        self,
        manifest_path: str,
        feature_dir: Optional[str] = None,
        split: str = "train",
        config: Optional[TrainingConfig] = None,
        validate_artifacts: bool = True
    ):
        self.manifest_path = os.path.abspath(manifest_path)
        self.feature_dir = os.path.abspath(feature_dir) if feature_dir else os.path.dirname(self.manifest_path)
        self.split = split.strip().lower()
        self.config = config or TrainingConfig()
        self.validate_artifacts = validate_artifacts

        self.windower = FeatureWindowing(
            target_frames=self.config.TARGET_FRAMES,
            stride=self.config.WINDOW_STRIDE,
            padding_mode=self.config.PADDING_MODE
        )
        self.validator = TrainingBatchValidator(target_frames=self.config.TARGET_FRAMES)

        self.samples: List[TrainingSample] = []
        self._load_and_window_dataset()

    def _load_and_window_dataset(self) -> None:
        """Parses manifest, loads .npz feature files, applies windowing, and stores TrainingSample records."""
        if not os.path.exists(self.manifest_path):
            raise DatasetLoaderError(f"Manifest file not found: '{self.manifest_path}'")

        # Read CSV manifest
        rows: List[Dict[str, str]] = []
        try:
            with open(self.manifest_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise DatasetLoaderError("Manifest file is empty.")
                for row in reader:
                    rows.append(row)
        except Exception as exc:
            raise DatasetLoaderError(f"Failed to parse manifest CSV: {str(exc)}")

        for row_idx, row in enumerate(rows, start=1):
            row_split = (row.get("split") or "").strip().lower()
            if row_split != self.split:
                continue

            rel_path = (row.get("feature_path") or row.get("file_path") or "").strip().replace("\\", "/")
            if not rel_path:
                raise DatasetLoaderError(f"Row {row_idx} in manifest missing feature path.")

            # Security path traversal checks
            if rel_path.startswith("/") or ":" in rel_path or ".." in rel_path.split("/"):
                raise DatasetLoaderError(f"Path traversal or absolute path rejected: '{rel_path}'")

            abs_path = os.path.normpath(os.path.join(self.feature_dir, rel_path))
            if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
                raise DatasetLoaderError(f"Feature artifact file not found: '{rel_path}'")

            # Load .npz artifact safely without unpickling arbitrary objects
            try:
                npz_data = np.load(abs_path, allow_pickle=False)
            except Exception as exc:
                raise DatasetLoaderError(f"Failed to load .npz artifact '{rel_path}': {str(exc)}")

            # Extract arrays
            try:
                log_mel = npz_data["log_mel"]
                mfcc = npz_data["mfcc"]
                spectral_centroid = npz_data["spectral_centroid"]
                spectral_rolloff = npz_data["spectral_rolloff"]
            except KeyError as exc:
                raise DatasetLoaderError(f"Missing required feature array {str(exc)} in '{rel_path}'")

            # Extract label
            label_str = (row.get("label") or "").strip().lower()
            if not label_str and "label_code" in npz_data:
                label_code_val = int(npz_data["label_code"])
                label_str = decode_label(label_code_val)

            try:
                label_code = encode_label(label_str)
            except ValueError as exc:
                raise DatasetLoaderError(f"Invalid label '{label_str}' in '{rel_path}': {str(exc)}")

            speaker_id = (row.get("speaker_id") or "").strip()
            source = (row.get("source") or "").strip()
            source_id = (row.get("source_id") or "").strip()
            sample_id = (row.get("sample_id") or Path(rel_path).stem).strip()

            if not speaker_id:
                raise DatasetLoaderError(f"Missing speaker_id in row {row_idx} ({rel_path}).")

            # Validate raw tensors
            if self.validate_artifacts:
                self._validate_raw_tensors(rel_path, log_mel, mfcc, spectral_centroid, spectral_rolloff)

            # Extract 256-frame windows
            window_dicts = self.windower.process_features(log_mel, mfcc, spectral_centroid, spectral_rolloff)

            for win_idx, win_data in enumerate(window_dicts):
                t_sample = TrainingSample(
                    log_mel=win_data["log_mel"],
                    mfcc=win_data["mfcc"],
                    spectral_centroid=win_data["spectral_centroid"],
                    spectral_rolloff=win_data["spectral_rolloff"],
                    label=label_code,
                    label_str=label_str,
                    speaker_id=speaker_id,
                    source=source,
                    source_id=source_id,
                    sample_id=sample_id,
                    window_index=win_idx,
                    split=self.split
                )

                # Validate windowed sample
                if self.validate_artifacts:
                    self.validator.validate_sample(t_sample)

                self.samples.append(t_sample)

    def _validate_raw_tensors(
        self,
        rel_path: str,
        log_mel: np.ndarray,
        mfcc: np.ndarray,
        spectral_centroid: np.ndarray,
        spectral_rolloff: np.ndarray
    ) -> None:
        """Audits raw loaded feature tensors prior to windowing."""
        tensors = {
            "log_mel": (log_mel, 80),
            "mfcc": (mfcc, 20),
            "spectral_centroid": (spectral_centroid, 1),
            "spectral_rolloff": (spectral_rolloff, 1)
        }

        t_len = None
        for name, (arr, expected_axis0) in tensors.items():
            if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 2:
                raise DatasetLoaderError(f"Feature '{name}' in '{rel_path}' is not a 2D NumPy array.")

            if str(arr.dtype) != "float32":
                raise DatasetLoaderError(f"Feature '{name}' in '{rel_path}' has invalid dtype '{arr.dtype}'. Expected float32.")

            if arr.shape[0] != expected_axis0:
                raise DatasetLoaderError(
                    f"Feature '{name}' in '{rel_path}' has axis-0 shape {arr.shape[0]}. Expected {expected_axis0}."
                )

            if np.isnan(arr).any():
                raise DatasetLoaderError(f"Feature '{name}' in '{rel_path}' contains NaN values.")

            if np.isinf(arr).any():
                raise DatasetLoaderError(f"Feature '{name}' in '{rel_path}' contains Infinity values.")

            if t_len is None:
                t_len = arr.shape[1]
            elif arr.shape[1] != t_len:
                raise DatasetLoaderError(
                    f"Feature '{name}' in '{rel_path}' has inconsistent frame count {arr.shape[1]}. Expected {t_len}."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrainingSample:
        return self.samples[idx]
