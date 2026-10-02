"""Visualization Builder Module for VoxGuard.

Generates compact, frontend-safe JSON payload representations for waveform peak envelopes
(~300 points) and normalized spectrogram visual grids (~40x120 bins).
Designed strictly for UI display and network transport; NOT used for ML model training.
"""

from dataclasses import dataclass, asdict
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import FeatureResult


@dataclass
class WaveformPayload:
    """Compact waveform peak envelope for UI rendering."""
    target_points: int
    min_peaks: List[float]
    max_peaks: List[float]
    peak_envelope: List[float]
    duration_seconds: float


@dataclass
class SpectrogramPayload:
    """Compact, normalized spectrogram grid for UI heatmap rendering."""
    freq_bins: int
    time_bins: int
    data_grid: List[List[float]]  # 2D list normalized strictly to range [0.0, 1.0]
    min_db: float
    max_db: float


@dataclass
class VisualizationPayload:
    """Aggregate visualization payload container ready for JSON API serialization."""
    waveform: WaveformPayload
    spectrogram: SpectrogramPayload

    def to_dict(self) -> Dict:
        """Converts payload to standard dictionary for REST API serialization."""
        return asdict(self)


class VisualizationBuilder:
    """Builder service producing compact frontend-safe visual artifacts."""

    def __init__(
        self,
        waveform_points: int = 300,
        spectrogram_freq_bins: int = 40,
        spectrogram_time_bins: int = 120
    ):
        self.waveform_points = waveform_points
        self.spectrogram_freq_bins = spectrogram_freq_bins
        self.spectrogram_time_bins = spectrogram_time_bins

    def build_all(
        self,
        preprocessed: PreprocessedAudio,
        log_mel_result: FeatureResult
    ) -> VisualizationPayload:
        """Builds both waveform envelope and compact spectrogram grid payloads.

        Args:
            preprocessed: Canonical PreprocessedAudio object.
            log_mel_result: Raw Log-Mel Spectrogram FeatureResult object.

        Returns:
            VisualizationPayload object containing frontend UI render payloads.
        """
        waveform_payload = self.build_waveform(preprocessed.pcm_data, preprocessed.duration_seconds)
        spectrogram_payload = self.build_spectrogram(log_mel_result.data)

        return VisualizationPayload(
            waveform=waveform_payload,
            spectrogram=spectrogram_payload
        )

    def build_waveform(self, pcm: np.ndarray, duration_seconds: float) -> WaveformPayload:
        """Computes deterministic ~300-point min/max peak envelope from PCM audio.

        Args:
            pcm: 1D PCM audio array.
            duration_seconds: Audio duration in seconds.

        Returns:
            WaveformPayload with min_peaks, max_peaks, and normalized peak_envelope lists.
        """
        if pcm is None or len(pcm) == 0:
            return WaveformPayload(
                target_points=self.waveform_points,
                min_peaks=[0.0] * self.waveform_points,
                max_peaks=[0.0] * self.waveform_points,
                peak_envelope=[0.0] * self.waveform_points,
                duration_seconds=0.0
            )

        n_samples = len(pcm)
        target_pts = min(self.waveform_points, n_samples)
        chunk_size = n_samples / float(target_pts)

        min_peaks = []
        max_peaks = []
        peak_envelope = []

        for i in range(target_pts):
            start_idx = int(math.floor(i * chunk_size))
            end_idx = int(math.ceil((i + 1) * chunk_size))
            end_idx = max(start_idx + 1, min(end_idx, n_samples))

            chunk = pcm[start_idx:end_idx]
            if len(chunk) > 0:
                min_val = float(np.min(chunk))
                max_val = float(np.max(chunk))
                peak_val = max(abs(min_val), abs(max_val))
            else:
                min_val = 0.0
                max_val = 0.0
                peak_val = 0.0

            min_peaks.append(round(min_val, 4))
            max_peaks.append(round(max_val, 4))
            peak_envelope.append(round(min(1.0, peak_val), 4))

        return WaveformPayload(
            target_points=len(peak_envelope),
            min_peaks=min_peaks,
            max_peaks=max_peaks,
            peak_envelope=peak_envelope,
            duration_seconds=round(duration_seconds, 4)
        )

    def build_spectrogram(self, log_mel_matrix: np.ndarray) -> SpectrogramPayload:
        """Downsamples and normalizes log-mel matrix into a compact 40x120 [0.0, 1.0] grid.

        Args:
            log_mel_matrix: Raw Log-Mel Spectrogram 2D array of shape (n_mels, num_frames).

        Returns:
            SpectrogramPayload ready for lightweight JSON transport.
        """
        if log_mel_matrix is None or log_mel_matrix.size == 0:
            empty_grid = [[0.0] * self.spectrogram_time_bins for _ in range(self.spectrogram_freq_bins)]
            return SpectrogramPayload(
                freq_bins=self.spectrogram_freq_bins,
                time_bins=self.spectrogram_time_bins,
                data_grid=empty_grid,
                min_db=0.0,
                max_db=0.0
            )

        n_mels, num_frames = log_mel_matrix.shape
        target_f = min(self.spectrogram_freq_bins, n_mels)
        target_t = min(self.spectrogram_time_bins, num_frames)

        # 2D Block Averaging / Downsampling
        f_chunk_size = n_mels / float(target_f)
        t_chunk_size = num_frames / float(target_t)

        grid = np.zeros((target_f, target_t), dtype=np.float32)

        for f_idx in range(target_f):
            f_start = int(math.floor(f_idx * f_chunk_size))
            f_end = int(math.ceil((f_idx + 1) * f_chunk_size))
            f_end = max(f_start + 1, min(f_end, n_mels))

            for t_idx in range(target_t):
                t_start = int(math.floor(t_idx * t_chunk_size))
                t_end = int(math.ceil((t_idx + 1) * t_chunk_size))
                t_end = max(t_start + 1, min(t_end, num_frames))

                sub_block = log_mel_matrix[f_start:f_end, t_start:t_end]
                grid[f_idx, t_idx] = np.mean(sub_block)

        # Compute original dB min/max before normalization
        min_db = float(np.min(grid))
        max_db = float(np.max(grid))

        # Normalize grid values strictly to [0.0, 1.0] range
        range_db = max_db - min_db
        if range_db > 1e-6:
            norm_grid = (grid - min_db) / range_db
        else:
            norm_grid = np.zeros_like(grid)

        # Convert to nested Python float list (rounded to 4 decimals for compact JSON payload size)
        grid_list = [
            [round(float(val), 4) for val in row]
            for row in norm_grid
        ]

        return SpectrogramPayload(
            freq_bins=target_f,
            time_bins=target_t,
            data_grid=grid_list,
            min_db=round(min_db, 2),
            max_db=round(max_db, 2)
        )
