"""Dataset Ingestion Service for VoxGuard.

Sequentially ingests validated dataset audio manifest entries, decodes audio streams,
preprocesses to canonical 16kHz mono float32 PCM, extracts acoustic features,
validates feature tensors, and yields structured DatasetSample records without loading
the entire dataset into RAM.
"""

from dataclasses import dataclass, asdict
import os
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

from app.audio.decoder import AudioDecoder, AudioDecodingError, DecodedAudio
from app.audio.preprocessor import AudioPreprocessor, AudioPreprocessingError, PreprocessedAudio
from app.audio.validator import AudioValidator
from app.evaluation.feature_validator import FeatureValidator, FeatureValidationError
from app.evaluation.labels import encode_label
from app.evaluation.manifest_validator import ManifestRow, ManifestValidator, ManifestValidationResult
from app.features.extractor import AudioFeatureExtractor, AudioFeatures


@dataclass
class DatasetSample:
    """Typed representation of a processed dataset audio sample."""
    sample_id: str
    file_path: str
    label: str
    speaker_id: str
    source: str
    source_id: str
    split: str
    audio_metadata: Dict[str, Any]
    canonical_audio: Dict[str, Any]
    features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Converts sample record to a JSON-serializable dictionary."""
        return asdict(self)


class DatasetIngestionError(Exception):
    """Raised when dataset sample ingestion or processing fails."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class DatasetIngestor:
    """Stream processing service converting dataset audio manifests into ML-ready feature records."""

    def __init__(
        self,
        validator: Optional[AudioValidator] = None,
        decoder: Optional[AudioDecoder] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        feature_validator: Optional[FeatureValidator] = None,
        manifest_validator: Optional[ManifestValidator] = None
    ):
        self.validator = validator or AudioValidator()
        self.decoder = decoder or AudioDecoder(validator=self.validator)
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.feature_validator = feature_validator or FeatureValidator()
        self.manifest_validator = manifest_validator or ManifestValidator()

    def process_sample_row(
        self,
        row: ManifestRow,
        base_dir: Optional[str] = None
    ) -> Tuple[DatasetSample, AudioFeatures]:
        """Ingests and processes a single ManifestRow entry.

        Args:
            row: Validated ManifestRow.
            base_dir: Optional base directory to resolve relative audio path.

        Returns:
            Tuple of (DatasetSample record, AudioFeatures object containing numpy feature arrays).

        Raises:
            DatasetIngestionError: If file is missing, invalid, undecodable, or features are malformed.
        """
        # 1. Label Validation
        try:
            encode_label(row.label)
        except ValueError as exc:
            raise DatasetIngestionError("INVALID_LABEL", f"Invalid label '{row.label}': {str(exc)}")

        # 2. Path Traversal & Existence Check
        rel_path = row.file_path.strip().replace("\\", "/")
        if rel_path.startswith("/") or rel_path.startswith("\\") or ".." in rel_path.split("/"):
            raise DatasetIngestionError("PATH_TRAVERSAL", "Path traversal or absolute path detected.")

        if base_dir:
            full_path = os.path.normpath(os.path.join(base_dir, rel_path))
        else:
            full_path = os.path.normpath(rel_path)

        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            raise DatasetIngestionError("FILE_NOT_FOUND", "Audio file not found on disk.")

        # Read file bytes cleanly
        try:
            with open(full_path, "rb") as f:
                file_bytes = f.read()
        except Exception as exc:
            raise DatasetIngestionError("FILE_READ_ERROR", f"Failed to read audio file: {str(exc)}")

        # 3. Audio Decoding
        filename = os.path.basename(rel_path)
        try:
            decoded: DecodedAudio = self.decoder.decode(file_bytes, filename=filename)
        except AudioDecodingError as exc:
            raise DatasetIngestionError("DECODING_FAILED", f"Audio decoding failed: {str(exc)}")
        except Exception as exc:
            raise DatasetIngestionError("DECODING_ERROR", f"Unexpected decoding error: {str(exc)}")

        # 4. Canonical Preprocessing (16kHz Mono float32)
        try:
            preprocessed: PreprocessedAudio = self.preprocessor.process(decoded)
        except AudioPreprocessingError as exc:
            raise DatasetIngestionError("PREPROCESSING_FAILED", f"Preprocessing failed: {str(exc)}")

        # 5. Feature Extraction
        try:
            features: AudioFeatures = self.feature_extractor.extract_all(preprocessed)
        except Exception as exc:
            raise DatasetIngestionError("EXTRACTION_FAILED", f"Feature extraction failed: {str(exc)}")

        # 6. Feature Validation
        try:
            self.feature_validator.validate_features(
                log_mel=features.log_mel_spectrogram.data,
                mfcc=features.mfcc.data,
                spectral_centroid=features.spectral_centroid.data,
                spectral_rolloff=features.spectral_rolloff.data
            )
        except FeatureValidationError as exc:
            raise DatasetIngestionError("INVALID_FEATURES", f"Feature validation failed: {str(exc)}")

        # 7. Construct Sample ID & Metadata
        base_name = Path(filename).stem
        sample_id = f"{row.split}_{row.speaker_id}_{base_name}"

        audio_metadata = {
            "duration_seconds": round(decoded.duration_seconds, 4),
            "original_sample_rate": decoded.sample_rate,
            "original_channels": decoded.channels,
            "format": decoded.format
        }

        canonical_audio = {
            "sample_rate": preprocessed.sample_rate,
            "channels": preprocessed.channels,
            "num_frames": preprocessed.num_frames,
            "duration_seconds": round(preprocessed.duration_seconds, 4)
        }

        num_feature_frames = features.log_mel_spectrogram.data.shape[1]
        feature_metadata = {
            "log_mel_shape": list(features.log_mel_spectrogram.data.shape),
            "mfcc_shape": list(features.mfcc.data.shape),
            "spectral_centroid_shape": list(features.spectral_centroid.data.shape),
            "spectral_rolloff_shape": list(features.spectral_rolloff.data.shape),
            "num_feature_frames": num_feature_frames,
            "dtype": "float32"
        }

        sample = DatasetSample(
            sample_id=sample_id,
            file_path=rel_path,
            label=row.label.strip().lower(),
            speaker_id=row.speaker_id.strip(),
            source=row.source.strip(),
            source_id=row.source_id.strip(),
            split=row.split.strip().lower(),
            audio_metadata=audio_metadata,
            canonical_audio=canonical_audio,
            features=feature_metadata
        )

        return sample, features
