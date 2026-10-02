"""Audio Feature Extractor Module for VoxGuard.

Extracts modular acoustic and spectral features from canonical 16kHz mono float32 PCM arrays:
- Log-Mel Spectrogram (80 bins)
- MFCC (20 coefficients)
- Spectral Centroid
- Spectral Rolloff (85% energy threshold)
Retains all feature outputs as pure typed NumPy arrays for downstream ML models.
"""

from dataclasses import dataclass, asdict
import math
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.fftpack
import scipy.signal

from app.audio.preprocessor import PreprocessedAudio
from app.config import Settings, settings as global_settings


class FeatureExtractionError(Exception):
    """Raised when feature extraction fails or input is invalid."""
    pass


@dataclass
class FeatureConfig:
    """Configurable parameters for DSP feature extraction."""
    n_fft: int = 1024           # FFT window size (~64ms at 16kHz)
    hop_length: int = 512       # Hop step size (~32ms at 16kHz, 50% overlap)
    win_length: int = 1024      # Window length
    n_mels: int = 80            # Number of Mel frequency bands
    n_mfcc: int = 20            # Number of MFCC coefficients to extract
    f_min: float = 0.0          # Minimum frequency in Hz
    f_max: float = 8000.0       # Maximum frequency in Hz (Nyquist limit at 16kHz)
    roll_percent: float = 0.85  # Spectral rolloff energy threshold (85%)
    window_type: str = "hann"   # Window function type


@dataclass
class FeatureResult:
    """Typed result container for an individual extracted numerical feature."""
    feature_name: str           # Unique feature identifier
    data: np.ndarray            # Raw NumPy array (float32)
    shape: Tuple[int, ...]      # Array shape dimensions
    dtype: str                  # Data type ("float32")
    parameters: Dict            # Extraction parameters used

    def to_dict(self) -> Dict:
        """Returns metadata summary excluding raw numpy array buffer."""
        return {
            "feature_name": self.feature_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "parameters": self.parameters
        }


@dataclass
class AudioFeatures:
    """Aggregate container holding all extracted ML feature sets for an audio payload."""
    log_mel_spectrogram: FeatureResult
    mfcc: FeatureResult
    spectral_centroid: FeatureResult
    spectral_rolloff: FeatureResult
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    metadata: Optional[Dict] = None


class AudioFeatureExtractor:
    """Modular feature extraction service for VoxGuard ML pipeline."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def extract_all(self, preprocessed: PreprocessedAudio) -> AudioFeatures:
        """Extracts complete suite of speech ML features from canonical PCM input.

        Args:
            preprocessed: PreprocessedAudio object (16kHz mono float32 array).

        Returns:
            AudioFeatures object containing all extracted FeatureResult sets.
        """
        pcm = preprocessed.pcm_data
        if pcm is None or pcm.size == 0:
            raise FeatureExtractionError("Cannot extract features from empty PCM array.")

        # Compute STFT Power Spectrogram base representation
        power_spec, freqs = self._compute_stft_power(pcm, preprocessed.sample_rate)

        # 1. Compute Log-Mel Spectrogram
        log_mel_res = self.compute_log_mel_spectrogram(power_spec, freqs, preprocessed.sample_rate)

        # 2. Compute MFCCs from Log-Mel Spectrogram
        mfcc_res = self.compute_mfcc(log_mel_res.data)

        # 3. Compute Spectral Centroid
        centroid_res = self.compute_spectral_centroid(power_spec, freqs)

        # 4. Compute Spectral Rolloff
        rolloff_res = self.compute_spectral_rolloff(power_spec, freqs)

        return AudioFeatures(
            log_mel_spectrogram=log_mel_res,
            mfcc=mfcc_res,
            spectral_centroid=centroid_res,
            spectral_rolloff=rolloff_res,
            sample_rate=preprocessed.sample_rate,
            duration_seconds=preprocessed.duration_seconds,
            metadata=preprocessed.metadata
        )

    def _compute_stft_power(self, pcm: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Short-Time Fourier Transform power spectrogram S = |STFT|^2."""
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length

        min_required_samples = n_fft * 2
        if len(pcm) < min_required_samples:
            # Pad short audio to at least 2 full FFT frame windows
            pcm = np.pad(pcm, (0, min_required_samples - len(pcm)), mode="constant")

        # Frame audio using STFT (scipy.signal.stft returns (freqs, times, Zxx))
        freqs, times, stft_matrix = scipy.signal.stft(
            pcm,
            fs=sample_rate,
            window=self.config.window_type,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary="zeros",
            padded=True
        )

        # Power spectrogram S = |STFT|^2
        power_spec = np.abs(stft_matrix) ** 2
        return power_spec.astype(np.float32), freqs.astype(np.float32)

    def compute_log_mel_spectrogram(
        self, power_spec: np.ndarray, freqs: np.ndarray, sample_rate: int
    ) -> FeatureResult:
        """Computes Log-Mel Spectrogram (dB scale) from power spectrogram."""
        n_mels = self.config.n_mels
        n_fft_bins = len(freqs)

        # Build triangular Mel filterbank matrix M of shape (n_mels, n_fft_bins)
        mel_filterbank = self._create_mel_filterbank(
            sample_rate=sample_rate,
            n_fft_bins=n_fft_bins,
            n_mels=n_mels,
            f_min=self.config.f_min,
            f_max=min(self.config.f_max, sample_rate / 2.0)
        )

        # Matrix multiply: (n_mels, n_fft_bins) * (n_fft_bins, num_frames) -> (n_mels, num_frames)
        mel_spec = np.dot(mel_filterbank, power_spec)

        # Convert to Log-Mel (dB scale) with zero-division guard
        log_mel_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
        log_mel_spec = log_mel_spec.astype(np.float32)

        return FeatureResult(
            feature_name="log_mel_spectrogram",
            data=log_mel_spec,
            shape=log_mel_spec.shape,
            dtype=str(log_mel_spec.dtype),
            parameters={
                "n_fft": self.config.n_fft,
                "hop_length": self.config.hop_length,
                "n_mels": n_mels,
                "f_min": self.config.f_min,
                "f_max": self.config.f_max
            }
        )

    def compute_mfcc(self, log_mel_data: np.ndarray) -> FeatureResult:
        """Computes Mel-Frequency Cepstral Coefficients (MFCCs) via Discrete Cosine Transform."""
        n_mfcc = self.config.n_mfcc
        
        # Apply DCT-II along frequency axis (axis 0)
        mfcc_full = scipy.fftpack.dct(log_mel_data, type=2, axis=0, norm="ortho")
        
        # Retain first n_mfcc coefficients -> shape (n_mfcc, num_frames)
        mfcc_data = mfcc_full[:n_mfcc, :].astype(np.float32)

        return FeatureResult(
            feature_name="mfcc",
            data=mfcc_data,
            shape=mfcc_data.shape,
            dtype=str(mfcc_data.dtype),
            parameters={
                "n_mfcc": n_mfcc,
                "n_mels": self.config.n_mels,
                "dct_type": 2
            }
        )

    def compute_spectral_centroid(self, power_spec: np.ndarray, freqs: np.ndarray) -> FeatureResult:
        """Computes Spectral Centroid (frequency center of mass) for each time frame."""
        freqs_col = freqs[:, np.newaxis]

        total_energy = np.sum(power_spec, axis=0, keepdims=True) + 1e-8
        centroid = np.sum(freqs_col * power_spec, axis=0, keepdims=True) / total_energy
        centroid_data = centroid.astype(np.float32)  # Shape (1, num_frames)

        return FeatureResult(
            feature_name="spectral_centroid",
            data=centroid_data,
            shape=centroid_data.shape,
            dtype=str(centroid_data.dtype),
            parameters={"n_fft": self.config.n_fft, "hop_length": self.config.hop_length}
        )

    def compute_spectral_rolloff(self, power_spec: np.ndarray, freqs: np.ndarray) -> FeatureResult:
        """Computes Spectral Rolloff (frequency below which 85% of total spectral energy lies)."""
        roll_percent = self.config.roll_percent
        total_energy = np.sum(power_spec, axis=0)  # Shape (num_frames,)
        cumulative_energy = np.cumsum(power_spec, axis=0)  # Shape (n_fft_bins, num_frames)

        threshold = roll_percent * total_energy  # Shape (num_frames,)
        
        rolloff_bins = np.zeros(power_spec.shape[1], dtype=np.int32)
        for t_idx in range(power_spec.shape[1]):
            bin_idx = np.where(cumulative_energy[:, t_idx] >= threshold[t_idx])[0]
            if len(bin_idx) > 0:
                rolloff_bins[t_idx] = bin_idx[0]
            else:
                rolloff_bins[t_idx] = len(freqs) - 1

        rolloff_freqs = freqs[rolloff_bins][np.newaxis, :].astype(np.float32)  # Shape (1, num_frames)

        return FeatureResult(
            feature_name="spectral_rolloff",
            data=rolloff_freqs,
            shape=rolloff_freqs.shape,
            dtype=str(rolloff_freqs.dtype),
            parameters={"roll_percent": roll_percent, "n_fft": self.config.n_fft}
        )

    def _create_mel_filterbank(
        self, sample_rate: int, n_fft_bins: int, n_mels: int, f_min: float, f_max: float
    ) -> np.ndarray:
        """Creates triangular Mel filterbank matrix using standard Hertz-to-Mel conversions."""
        hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
        mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(f_min)
        max_mel = hz_to_mel(f_max)

        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_indices = np.floor((n_fft_bins - 1) * hz_points / (sample_rate / 2.0)).astype(int)

        filterbank = np.zeros((n_mels, n_fft_bins), dtype=np.float32)
        for i in range(1, n_mels + 1):
            left = bin_indices[i - 1]
            center = bin_indices[i]
            right = bin_indices[i + 1]

            if center > left:
                filterbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                filterbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

        return filterbank
