"""Audio Quality Assessment Module for VoxGuard.

Performs non-destructive quality evaluation on canonical preprocessed audio streams:
- Duration check
- RMS amplitude level check
- Clipping ratio inspection
- Silence vs speech ratio estimation
- Signal-to-Noise Ratio (SNR) estimation

Guarantees that silent, near-silent, clipped, or extremely short recordings are identified
as unsuitable for high-confidence classification, preventing false-positive synthetic verdicts.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class AudioQualityAssessment:
    """Structured quality evaluation result for preprocessed audio."""
    duration_seconds: float
    rms_level: float
    peak_amplitude: float
    clipping_ratio: float
    silence_ratio: float
    snr_db: float
    usable_speech_duration: float
    is_usable: bool
    unusable_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Converts assessment object to clean dictionary for metadata inclusion."""
        return asdict(self)


class AudioQualityEvaluator:
    """Evaluates physical audio signal quality and speech usability."""

    def evaluate(
        self,
        pcm_data: np.ndarray,
        sample_rate: int = 16000,
        min_duration_seconds: float = 0.5,
        min_speech_duration: float = 0.3,
        min_rms_level: float = 0.003,
        max_silence_ratio: float = 0.88,
        max_clipping_ratio: float = 0.15
    ) -> AudioQualityAssessment:
        """Runs multi-metric physical signal evaluation on 1D float32 PCM array.

        Args:
            pcm_data: 1D NumPy array of normalized float32 audio samples.
            sample_rate: Sampling frequency in Hz (default: 16000).
            min_duration_seconds: Minimum total duration in seconds.
            min_speech_duration: Minimum usable active speech duration in seconds.
            min_rms_level: Minimum overall RMS signal amplitude.
            max_silence_ratio: Maximum allowable proportion of silence.
            max_clipping_ratio: Maximum allowable clipping sample proportion.

        Returns:
            AudioQualityAssessment object detailing usability state and physical metrics.
        """
        if pcm_data is None or pcm_data.size == 0:
            return AudioQualityAssessment(
                duration_seconds=0.0,
                rms_level=0.0,
                peak_amplitude=0.0,
                clipping_ratio=0.0,
                silence_ratio=1.0,
                snr_db=0.0,
                usable_speech_duration=0.0,
                is_usable=False,
                unusable_reason="Audio stream is empty (0 samples)."
            )

        pcm = np.asarray(pcm_data, dtype=np.float32)
        total_samples = len(pcm)
        duration_sec = float(total_samples) / float(sample_rate) if sample_rate > 0 else 0.0

        # 1. Peak & RMS Level
        peak_amp = float(np.max(np.abs(pcm))) if total_samples > 0 else 0.0
        rms_level = float(np.sqrt(np.mean(pcm ** 2))) if total_samples > 0 else 0.0

        # 2. Clipping Ratio (|x| >= 0.99)
        clipped_samples = int(np.sum(np.abs(pcm) >= 0.99))
        clipping_ratio = float(clipped_samples / total_samples) if total_samples > 0 else 0.0

        # 3. Frame-based Silence & Active Speech Analysis (20ms frames)
        frame_len = int(sample_rate * 0.020)  # 20 ms frame
        if frame_len > 0 and total_samples >= frame_len:
            num_frames = total_samples // frame_len
            frames = pcm[:num_frames * frame_len].reshape(num_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

            # Active speech threshold: frame RMS >= 0.005 or peak - 35dB
            max_frame_rms = float(np.max(frame_rms)) if len(frame_rms) > 0 else 0.0
            speech_thresh = max(0.005, max_frame_rms * 0.08)

            active_mask = frame_rms >= speech_thresh
            silent_frames = int(np.sum(~active_mask))
            silence_ratio = float(silent_frames / num_frames) if num_frames > 0 else 0.0
            usable_speech_duration = float(np.sum(active_mask) * 0.020)

            # 4. SNR Estimation (speech frame energy vs silence frame energy)
            speech_energy = float(np.mean(frame_rms[active_mask] ** 2)) if np.any(active_mask) else 1e-8
            noise_energy = float(np.mean(frame_rms[~active_mask] ** 2)) if np.any(~active_mask) else 1e-8
            snr_db = float(10.0 * np.log10((speech_energy + 1e-8) / (noise_energy + 1e-8)))
        else:
            silence_ratio = 0.0 if rms_level >= min_rms_level else 1.0
            usable_speech_duration = duration_sec if rms_level >= min_rms_level else 0.0
            snr_db = 15.0 if rms_level >= min_rms_level else 0.0

        # Determine Usability & Unusable Reason
        is_usable = True
        unusable_reason = None

        if duration_sec < min_duration_seconds:
            is_usable = False
            unusable_reason = f"Audio duration ({duration_sec:.2f}s) is too short for reliable analysis (minimum {min_duration_seconds:.1f}s required)."
        elif rms_level < min_rms_level or peak_amp < 0.005:
            is_usable = False
            unusable_reason = "Audio signal is near-silent or contains insufficient volume energy."
        elif usable_speech_duration < min_speech_duration:
            is_usable = False
            unusable_reason = f"Insufficient active speech detected ({usable_speech_duration:.2f}s usable speech, minimum {min_speech_duration:.1f}s required)."
        elif silence_ratio > max_silence_ratio:
            is_usable = False
            unusable_reason = f"Excessive background silence or dead air ({silence_ratio * 100:.1f}% silence ratio)."
        elif clipping_ratio > max_clipping_ratio:
            is_usable = False
            unusable_reason = f"Severe audio signal distortion/clipping ({clipping_ratio * 100:.1f}% clipped samples)."

        return AudioQualityAssessment(
            duration_seconds=round(duration_sec, 3),
            rms_level=round(rms_level, 5),
            peak_amplitude=round(peak_amp, 5),
            clipping_ratio=round(clipping_ratio, 4),
            silence_ratio=round(silence_ratio, 4),
            snr_db=round(snr_db, 2),
            usable_speech_duration=round(usable_speech_duration, 3),
            is_usable=is_usable,
            unusable_reason=unusable_reason
        )
