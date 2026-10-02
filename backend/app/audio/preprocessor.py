"""Audio Preprocessor Module for VoxGuard.

Transforms decoded raw PCM audio into VoxGuard's canonical internal representation:
- Mono (1 channel)
- 16,000 Hz sample rate
- 32-bit floating-point (np.float32)
- Peak normalized [-1.0, 1.0]
- Deterministic output
"""

from dataclasses import dataclass, asdict
import math
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.signal

from app.audio.decoder import DecodedAudio
from app.config import Settings, settings as global_settings


class AudioPreprocessingError(Exception):
    """Raised when audio preprocessing or canonical conversion fails."""
    pass


@dataclass
class PreprocessedAudio:
    """Canonical internal audio representation prepared for feature extraction & ML inference."""
    pcm_data: np.ndarray              # 1D float32 numpy array, shape (num_samples,)
    sample_rate: int = 16000          # Always 16000 Hz
    channels: int = 1                 # Always 1 (mono)
    num_frames: int = 0               # Total resampled sample frames
    duration_seconds: float = 0.0     # Final duration in seconds
    metadata: Dict = None             # Transformation audit metadata

    def to_dict(self) -> Dict:
        """Returns metadata dictionary (excluding raw PCM array for clean logging)."""
        res = asdict(self)
        res.pop("pcm_data", None)
        return res


class AudioPreprocessor:
    """Canonical Audio Preprocessing Service.

    Converts any arbitrary decoded PCM stream into a standardized 16kHz mono float32 array
    using deterministic polyphase resampling and peak normalization.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or global_settings

    def process(self, decoded: DecodedAudio) -> PreprocessedAudio:
        """Transforms a DecodedAudio object into the canonical ML representation.

        Args:
            decoded: DecodedAudio object containing raw PCM array and source metadata.

        Returns:
            PreprocessedAudio object with canonical 16kHz mono float32 PCM array.

        Raises:
            AudioPreprocessingError: If input array is empty or non-finite values occur.
        """
        if decoded.pcm_data is None or decoded.pcm_data.size == 0:
            raise AudioPreprocessingError("Input PCM audio array is empty (0 samples).")

        pcm = np.asarray(decoded.pcm_data, dtype=np.float64)  # Use float64 precision for intermediate DSP
        orig_sr = decoded.sample_rate
        orig_channels = decoded.channels
        orig_frames = len(pcm) if pcm.ndim == 1 else pcm.shape[0]

        target_sr = self.config.TARGET_SAMPLE_RATE  # 16000 Hz
        target_channels = self.config.TARGET_CHANNELS  # 1 (Mono)

        # Step 1: Channel Conversion (Stereo/Multi-channel -> Mono)
        was_channel_converted = False
        if pcm.ndim > 1 and pcm.shape[1] > 1:
            # Average across channels axis
            pcm = np.mean(pcm, axis=1)
            was_channel_converted = True
        elif pcm.ndim > 1:
            pcm = pcm.squeeze()

        # Step 2: Resampling to 16,000 Hz
        was_resampled = False
        if orig_sr != target_sr:
            pcm = self._resample_polyphase(pcm, orig_sr, target_sr)
            was_resampled = True

        # Step 3: Type Conversion to 32-bit floating point (np.float32)
        pcm_float32 = pcm.astype(np.float32)

        # Step 4: Peak Loudness Normalization to range [-1.0, 1.0]
        peak = float(np.max(np.abs(pcm_float32))) if pcm_float32.size > 0 else 0.0
        if peak > 1e-8:
            pcm_float32 = pcm_float32 / peak
        else:
            # Handle near-silent audio without division by zero
            peak = 0.0

        # Step 5: Numerical Sanity Verification
        if not np.isfinite(pcm_float32).all():
            raise AudioPreprocessingError("Preprocessing produced non-finite numerical values (NaN or Infinity).")

        num_frames = len(pcm_float32)
        duration_seconds = float(num_frames) / float(target_sr) if target_sr > 0 else 0.0

        transformation_metadata = {
            "original_sample_rate": orig_sr,
            "original_channels": orig_channels,
            "original_frames": orig_frames,
            "resampled": was_resampled,
            "channel_converted": was_channel_converted,
            "peak_scaling_factor": round(peak, 6),
            "target_sample_rate": target_sr,
            "target_channels": target_channels,
            "dtype": str(pcm_float32.dtype)
        }

        return PreprocessedAudio(
            pcm_data=pcm_float32,
            sample_rate=target_sr,
            channels=target_channels,
            num_frames=num_frames,
            duration_seconds=round(duration_seconds, 4),
            metadata=transformation_metadata
        )

    def _resample_polyphase(self, pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resamples 1D audio array using polyphase FIR filtering for high signal fidelity."""
        if orig_sr <= 0 or target_sr <= 0:
            raise AudioPreprocessingError(f"Invalid sample rate during resampling: orig={orig_sr}, target={target_sr}")

        # Compute greatest common divisor for polyphase factors
        common_gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // common_gcd
        down = orig_sr // common_gcd

        try:
            resampled = scipy.signal.resample_poly(pcm, up=up, down=down)
            return resampled
        except Exception as exc:
            # Fallback linear interpolation if polyphase resample encounters edge cases
            num_output_samples = int(round(len(pcm) * (float(target_sr) / float(orig_sr))))
            orig_indices = np.linspace(0, len(pcm) - 1, num=len(pcm))
            target_indices = np.linspace(0, len(pcm) - 1, num=num_output_samples)
            resampled = np.interp(target_indices, orig_indices, pcm)
            return resampled
