"""Deterministic Feature Windowing Module for VoxGuard.

Converts variable-length feature arrays of temporal length T into fixed-size 256-frame
windows using deterministic sliding windowing with stride 128 and right-side zero-padding.
"""

from typing import Dict, List, Tuple
import numpy as np


class FeatureWindowing:
    """Service generating fixed 256-frame feature windows from variable-length sequences."""

    def __init__(
        self,
        target_frames: int = 256,
        stride: int = 128,
        padding_mode: str = "zero"
    ):
        self.target_frames = target_frames
        self.stride = stride
        self.padding_mode = padding_mode

    def get_window_slices(self, total_frames: int) -> List[Tuple[int, int]]:
        """Calculates deterministic (start, end) index tuples for temporal sequence length T.

        Args:
            total_frames: Total temporal frames T in feature sequence.

        Returns:
            List of (start, end) frame index bounds.
        """
        if total_frames <= 0:
            return [(0, self.target_frames)]

        if total_frames <= self.target_frames:
            # Short sequence: single window 0..total_frames (padded downstream to target_frames)
            return [(0, total_frames)]

        slices: List[Tuple[int, int]] = []
        curr_start = 0

        while curr_start + self.target_frames <= total_frames:
            slices.append((curr_start, curr_start + self.target_frames))
            curr_start += self.stride

        # Final tail window handling: if remaining frames exist, append right-aligned tail window
        last_end = slices[-1][1]
        if last_end < total_frames:
            tail_start = total_frames - self.target_frames
            if tail_start > slices[-1][0]:
                slices.append((tail_start, total_frames))

        return slices

    def apply_window(
        self,
        arr: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> np.ndarray:
        """Extracts and pads/truncates a 2D feature array along axis 1 to target_frames.

        Args:
            arr: 2D NumPy array of shape (C, T), float32.
            start_idx: Slice start frame.
            end_idx: Slice end frame.

        Returns:
            2D NumPy array of shape (C, target_frames), float32.
        """
        if arr is None or arr.ndim != 2:
            raise ValueError(f"Input array must be a 2D NumPy array, got {type(arr)}.")

        c_dim, t_dim = arr.shape
        windowed = arr[:, start_idx:end_idx]

        curr_t = windowed.shape[1]
        if curr_t < self.target_frames:
            # Right-side zero padding
            pad_width = self.target_frames - curr_t
            padding = np.zeros((c_dim, pad_width), dtype=np.float32)
            windowed = np.hstack((windowed, padding))

        elif curr_t > self.target_frames:
            # Truncate to target_frames
            windowed = windowed[:, :self.target_frames]

        return windowed.astype(np.float32)

    def process_features(
        self,
        log_mel: np.ndarray,
        mfcc: np.ndarray,
        spectral_centroid: np.ndarray,
        spectral_rolloff: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """Processes complete feature set into a list of windowed feature dictionaries.

        Args:
            log_mel: (80, T) float32 array
            mfcc: (20, T) float32 array
            spectral_centroid: (1, T) float32 array
            spectral_rolloff: (1, T) float32 array

        Returns:
            List of dicts, each holding windowed 'log_mel', 'mfcc', 'spectral_centroid',
            'spectral_rolloff' arrays of exact shape (C, 256).
        """
        t_frames = log_mel.shape[1]
        slices = self.get_window_slices(t_frames)

        windows: List[Dict[str, np.ndarray]] = []
        for start, end in slices:
            win_dict = {
                "log_mel": self.apply_window(log_mel, start, end),
                "mfcc": self.apply_window(mfcc, start, end),
                "spectral_centroid": self.apply_window(spectral_centroid, start, end),
                "spectral_rolloff": self.apply_window(spectral_rolloff, start, end),
            }
            windows.append(win_dict)

        return windows
