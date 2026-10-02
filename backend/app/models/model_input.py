"""Model Input Contract Specification for VoxGuard.

Defines a strongly typed input representation specifying canonical sample rate,
channel constraints, tensor shapes, normalization assumptions, and padding policies
for future trained audio deepfake detection models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.features.extractor import AudioFeatures


@dataclass
class ModelInputContract:
    """Strongly typed input specification contract for ML audio deepfake classifiers."""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    feature_type: str = "log_mel_spectrogram"    # "log_mel_spectrogram", "mfcc", "raw_pcm"
    expected_freq_bins: int = 80
    expected_channels_dim: int = 1
    normalization_policy: str = "peak_normalized"  # "peak_normalized", "zero_mean_unit_var"
    padding_truncation_policy: str = "dynamic_sequence"  # "dynamic_sequence", "fixed_window_pad"
    max_duration_seconds: float = 300.0

    def validate_features(self, features: AudioFeatures) -> Tuple[bool, Optional[str]]:
        """Validates an extracted AudioFeatures instance against this input contract.

        Args:
            features: Extracted AudioFeatures object.

        Returns:
            Tuple of (is_valid, error_message_if_invalid).
        """
        if features is None:
            return False, "AudioFeatures input is None."

        if self.feature_type == "log_mel_spectrogram":
            if features.log_mel_spectrogram is None or features.log_mel_spectrogram.data is None:
                return False, "Log-Mel Spectrogram feature payload is missing or empty."
            
            grid = features.log_mel_spectrogram.data
            if grid.size == 0:
                return False, "Log-Mel Spectrogram data array has 0 size."

            # Check frequency bin count match
            n_bins = grid.shape[0] if grid.ndim >= 2 else 0
            if n_bins != self.expected_freq_bins:
                return False, f"Frequency bins mismatch: expected {self.expected_freq_bins}, got {n_bins}."

        elif self.feature_type == "mfcc":
            if features.mfcc is None or features.mfcc.data is None:
                return False, "MFCC feature payload is missing or empty."
            
            grid = features.mfcc.data
            if grid.size == 0:
                return False, "MFCC data array has 0 size."

        elif self.feature_type == "raw_pcm":
            # For raw PCM classifiers expecting 1D float32 arrays
            pass

        return True, None

    def adapt_to_tensor_shape(self, features: AudioFeatures) -> np.ndarray:
        """Adapts AudioFeatures payload into standardized NumPy array shape [Batch, Channels, Freq/Channels, Time].

        Args:
            features: AudioFeatures object containing extracted NumPy arrays.

        Returns:
            Formatted 4D float32 NumPy array ready for model forward pass.
        """
        mel = features.log_mel_spectrogram.data.astype(np.float32) if (features and features.log_mel_spectrogram and features.log_mel_spectrogram.data is not None) else None
        mfcc = features.mfcc.data.astype(np.float32) if (features and features.mfcc and features.mfcc.data is not None) else None

        if self.feature_type == "log_mel_spectrogram" and mel is not None:
            if mel.ndim == 2:
                return np.expand_dims(mel, axis=(0, 1))  # (1, 1, 80, T)
            return mel

        if self.feature_type == "mfcc" and mfcc is not None:
            if mfcc.ndim == 2:
                return np.expand_dims(mfcc, axis=(0, 1))  # (1, 1, 20, T)
            return mfcc

        # Default / log_mel_mfcc mode
        if mel is not None and mfcc is not None:
            cat = np.concatenate([mel, mfcc], axis=0)  # (100, T)
            if cat.ndim == 2:
                return np.expand_dims(cat, axis=(0, 1))  # (1, 1, 100, T)
            return cat

        if mel is not None:
            if mel.ndim == 2:
                return np.expand_dims(mel, axis=(0, 1))
            return mel

        if mfcc is not None:
            if mfcc.ndim == 2:
                return np.expand_dims(mfcc, axis=(0, 1))
            return mfcc

        return np.array([], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Converts contract parameters to dictionary representation."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "dtype": self.dtype,
            "feature_type": self.feature_type,
            "expected_freq_bins": self.expected_freq_bins,
            "expected_channels_dim": self.expected_channels_dim,
            "normalization_policy": self.normalization_policy,
            "padding_truncation_policy": self.padding_truncation_policy,
            "max_duration_seconds": self.max_duration_seconds,
        }
