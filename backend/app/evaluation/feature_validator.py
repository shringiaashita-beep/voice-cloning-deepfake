"""Feature Validation Engine for VoxGuard.

Validates extracted acoustic and spectral feature tensors against expected shapes,
dtypes, non-emptiness, numerical finiteness (no NaN or Inf), and temporal frame alignment.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class FeatureValidationError(Exception):
    """Raised when feature tensor validation fails."""
    pass


class FeatureValidator:
    """Validator auditing feature tensor shape, dtype, finiteness, and dimension alignment."""

    def __init__(
        self,
        expected_n_mels: int = 80,
        expected_n_mfcc: int = 20,
        expected_dtype: str = "float32"
    ):
        self.expected_n_mels = expected_n_mels
        self.expected_n_mfcc = expected_n_mfcc
        self.expected_dtype = expected_dtype

    def validate_features(
        self,
        log_mel: np.ndarray,
        mfcc: np.ndarray,
        spectral_centroid: np.ndarray,
        spectral_rolloff: np.ndarray
    ) -> bool:
        """Validates a complete set of feature arrays.

        Args:
            log_mel: Log-Mel spectrogram array, expected shape (80, T)
            mfcc: MFCC array, expected shape (20, T)
            spectral_centroid: Spectral centroid array, expected shape (1, T)
            spectral_rolloff: Spectral rolloff array, expected shape (1, T)

        Returns:
            True if all feature tensors are valid.

        Raises:
            FeatureValidationError: If any array is malformed, non-finite, empty, or mismatched.
        """
        features = {
            "log_mel": log_mel,
            "mfcc": mfcc,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff
        }

        # 1. Existence and Array Type Checks
        for name, arr in features.items():
            if arr is None:
                raise FeatureValidationError(f"Feature '{name}' is None.")
            if not isinstance(arr, np.ndarray):
                raise FeatureValidationError(f"Feature '{name}' must be a numpy.ndarray, got {type(arr).__name__}.")
            if arr.size == 0:
                raise FeatureValidationError(f"Feature '{name}' is empty (size 0).")

        # 2. Datatype Validation
        for name, arr in features.items():
            if str(arr.dtype) != self.expected_dtype:
                raise FeatureValidationError(
                    f"Feature '{name}' has invalid dtype '{arr.dtype}'. Expected '{self.expected_dtype}'."
                )

        # 3. Finite Values Validation (no NaN, no Inf)
        for name, arr in features.items():
            if np.isnan(arr).any():
                raise FeatureValidationError(f"Feature '{name}' contains NaN values.")
            if np.isinf(arr).any():
                raise FeatureValidationError(f"Feature '{name}' contains Infinity values.")

        # 4. Dimension & Rank Validation
        for name, arr in features.items():
            if arr.ndim != 2:
                raise FeatureValidationError(
                    f"Feature '{name}' must be a 2D array, got shape {arr.shape} (ndim={arr.ndim})."
                )

        # Specific feature axis-0 dimensions
        if log_mel.shape[0] != self.expected_n_mels:
            raise FeatureValidationError(
                f"Log-Mel dimension mismatch: expected {self.expected_n_mels} frequency bins, got {log_mel.shape[0]}."
            )

        if mfcc.shape[0] != self.expected_n_mfcc:
            raise FeatureValidationError(
                f"MFCC dimension mismatch: expected {self.expected_n_mfcc} coefficients, got {mfcc.shape[0]}."
            )

        if spectral_centroid.shape[0] != 1:
            raise FeatureValidationError(
                f"Spectral centroid dimension mismatch: expected 1 row, got {spectral_centroid.shape[0]}."
            )

        if spectral_rolloff.shape[0] != 1:
            raise FeatureValidationError(
                f"Spectral rolloff dimension mismatch: expected 1 row, got {spectral_rolloff.shape[0]}."
            )

        # 5. Temporal Frame Alignment across all features (axis 1)
        t_log_mel = log_mel.shape[1]
        t_mfcc = mfcc.shape[1]
        t_centroid = spectral_centroid.shape[1]
        t_rolloff = spectral_rolloff.shape[1]

        if t_log_mel <= 0:
            raise FeatureValidationError(f"Feature temporal frames must be > 0, got {t_log_mel}.")

        if not (t_log_mel == t_mfcc == t_centroid == t_rolloff):
            raise FeatureValidationError(
                f"Incompatible feature temporal frame dimensions: log_mel={t_log_mel}, mfcc={t_mfcc}, "
                f"spectral_centroid={t_centroid}, spectral_rolloff={t_rolloff}."
            )

        return True
