"""Statistical Acoustic Baseline Classifier for VoxGuard.

Implements an uncalibrated baseline runner with status ModelStatus.ANALYSIS_ONLY.
Calculates objective acoustic descriptive statistics (spectral centroid, rolloff,
MFCC, log-mel distribution, temporal variation) without fabricating deepfake detection
probabilities or claiming heuristics constitute synthetic classification.
"""

import time
from typing import Any, Dict, Optional

import numpy as np

from app.features.extractor import AudioFeatures
from app.models.base_classifier import (
    AbstractDeepfakeClassifier,
    ClassificationResult,
    ModelStatus,
    PredictionLabel,
)


class BaselineClassifier(AbstractDeepfakeClassifier):
    """Statistical Acoustic Baseline Classifier.
    
    Establishes a reproducible baseline for feature statistics and future experiments.
    Output explicitly states that all measurements are acoustic indicators, not a validated deepfake classification.
    """

    @property
    def model_name(self) -> str:
        return "VoxGuard Statistical Acoustic Baseline"

    @property
    def model_version(self) -> str:
        return "0.1.0-baseline"

    @property
    def status(self) -> ModelStatus:
        return ModelStatus.ANALYSIS_ONLY

    def get_model_metadata(self):
        from app.models.base_classifier import ModelMetadata
        return ModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            framework="statistical_baseline",
            input_sample_rate=16000,
            input_channels=1,
            model_source="VoxGuard Core Acoustic Engine",
            license="MIT",
            weights_location="not_applicable",
            evaluation_dataset=None,
            evaluation_protocol=None,
            evaluation_metrics=None
        )

    def predict(self, features: AudioFeatures) -> ClassificationResult:
        """Processes extracted audio features and computes acoustic statistical metrics.

        Args:
            features: AudioFeatures object containing extracted NumPy feature arrays.

        Returns:
            ClassificationResult with ModelStatus.ANALYSIS_ONLY, PredictionLabel.UNCERTAIN,
            and None probabilities.
        """
        start_time = time.perf_counter()

        # Handle empty/malformed feature input safely
        if features is None or features.log_mel_spectrogram is None:
            return ClassificationResult(
                label=PredictionLabel.UNCERTAIN,
                probabilities=None,
                confidence_score=None,
                model_name=self.model_name,
                model_version=self.model_version,
                status=ModelStatus.NOT_READY,
                metadata={"error": "Malformed or empty feature input provided."}
            )

        # Compute descriptive acoustic indicators
        acoustic_stats = self._compute_descriptive_indicators(features)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = {
            "execution_time_ms": round(elapsed_ms, 2),
            "device": "cpu",
            "disclaimer": "All values are descriptive acoustic indicators, not a validated deepfake classification.",
            "evaluation_notice": "No trained model weights are loaded. Probabilities are uncalibrated and set to None.",
            "acoustic_statistics": acoustic_stats
        }

        # Return ANALYSIS_ONLY verdict with None probabilities (strictly NO synthetic probability output)
        return ClassificationResult(
            label=PredictionLabel.UNCERTAIN,
            probabilities={"human": None, "synthetic": None},
            confidence_score=None,
            model_name=self.model_name,
            model_version=self.model_version,
            status=self.status,
            metadata=metadata
        )

    def _compute_descriptive_indicators(self, features: AudioFeatures) -> Dict[str, Any]:
        """Calculates descriptive acoustic indicators across spectral and temporal domains."""
        stats: Dict[str, Any] = {}

        # 1. Spectral Centroid mean/std & temporal variation
        if features.spectral_centroid and features.spectral_centroid.data.size > 0:
            c_data = features.spectral_centroid.data
            stats["spectral_centroid_mean_hz"] = round(float(np.mean(c_data)), 2)
            stats["spectral_centroid_std_hz"] = round(float(np.std(c_data)), 2)
            stats["spectral_centroid_min_hz"] = round(float(np.min(c_data)), 2)
            stats["spectral_centroid_max_hz"] = round(float(np.max(c_data)), 2)

            # Temporal variation (frame-to-frame delta mean)
            if c_data.shape[1] > 1:
                centroid_delta = np.abs(np.diff(c_data, axis=1))
                stats["temporal_variation_centroid_delta_mean"] = round(float(np.mean(centroid_delta)), 2)
            else:
                stats["temporal_variation_centroid_delta_mean"] = 0.0
        else:
            stats["spectral_centroid_mean_hz"] = 0.0
            stats["spectral_centroid_std_hz"] = 0.0
            stats["temporal_variation_centroid_delta_mean"] = 0.0

        # 2. Spectral Rolloff mean/std (85% energy threshold)
        if features.spectral_rolloff and features.spectral_rolloff.data.size > 0:
            r_data = features.spectral_rolloff.data
            stats["spectral_rolloff_mean_hz"] = round(float(np.mean(r_data)), 2)
            stats["spectral_rolloff_std_hz"] = round(float(np.std(r_data)), 2)
        else:
            stats["spectral_rolloff_mean_hz"] = 0.0
            stats["spectral_rolloff_std_hz"] = 0.0

        # 3. Log-Mel Spectrogram Statistics
        if features.log_mel_spectrogram and features.log_mel_spectrogram.data.size > 0:
            log_mel = features.log_mel_spectrogram.data
            stats["log_mel_mean_db"] = round(float(np.mean(log_mel)), 2)
            stats["log_mel_std_db"] = round(float(np.std(log_mel)), 2)

            # Frequency band energy distribution (lower half vs upper half mel bins)
            n_mels = log_mel.shape[0]
            half_mels = n_mels // 2
            stats["low_band_mel_mean_db"] = round(float(np.mean(log_mel[:half_mels, :])), 2)
            stats["high_band_mel_mean_db"] = round(float(np.mean(log_mel[half_mels:, :])), 2)
        else:
            stats["log_mel_mean_db"] = 0.0
            stats["log_mel_std_db"] = 0.0

        # 4. MFCC mean/std/variance
        if features.mfcc and features.mfcc.data.size > 0:
            mfcc_data = features.mfcc.data
            stats["mfcc_mean"] = round(float(np.mean(mfcc_data)), 2)
            stats["mfcc_std"] = round(float(np.std(mfcc_data)), 2)
            stats["mfcc_variance"] = round(float(np.var(mfcc_data)), 2)
        else:
            stats["mfcc_mean"] = 0.0
            stats["mfcc_std"] = 0.0
            stats["mfcc_variance"] = 0.0

        return stats
