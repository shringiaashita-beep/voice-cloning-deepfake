"""Speaker Biometrics & Reference Voice Comparator for VoxGuard.

Enables forensic A/B comparison between Questioned (Sample A) and Reference (Sample B) voices.
Evaluates:
- Cepstral vocal tract geometry similarity (MFCC cosine distance)
- Fundamental frequency / spectral centroid alignment
- Cross-sample impersonation risk assessment
"""

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np

from app.features.extractor import AudioFeatures
from app.models.base_classifier import ClassificationResult, PredictionLabel


@dataclass
class SpeakerComparisonResult:
    """Structured result of reference voice A/B biometric match."""
    speaker_similarity_percent: float
    impersonation_risk: str
    verdict: str
    findings: List[str]
    sample_a: Dict[str, Any]
    sample_b: Dict[str, Any]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpeakerComparator:
    """Forensic comparator for speaker biometrics and clone impersonation."""

    def compare(
        self,
        features_a: AudioFeatures,
        classification_a: ClassificationResult,
        features_b: AudioFeatures,
        classification_b: ClassificationResult,
        filename_a: str = "questioned_sample.wav",
        filename_b: str = "reference_sample.wav"
    ) -> SpeakerComparisonResult:
        """Compares two audio recordings for voice identity and synthetic impersonation.

        Args:
            features_a: Extracted features of questioned sample A.
            classification_a: Deepfake classification of sample A.
            features_b: Extracted features of reference sample B.
            classification_b: Deepfake classification of sample B.
            filename_a: Label for sample A.
            filename_b: Label for sample B.

        Returns:
            SpeakerComparisonResult containing similarity score and forensic findings.
        """
        # 1. Extract MFCC mean embeddings
        mfcc_a = features_a.mfcc.data if features_a.mfcc else np.zeros((20, 10))
        mfcc_b = features_b.mfcc.data if features_b.mfcc else np.zeros((20, 10))

        vec_a = np.mean(mfcc_a, axis=1)
        vec_b = np.mean(mfcc_b, axis=1)

        norm_a = np.linalg.norm(vec_a) + 1e-7
        norm_b = np.linalg.norm(vec_b) + 1e-7
        cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Normalize cosine similarity [-1, 1] to [0, 1]
        mfcc_sim = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

        # 2. Spectral Centroid / Pitch Alignment
        cent_a = features_a.spectral_centroid.data if features_a.spectral_centroid else np.array([[1500.0]])
        cent_b = features_b.spectral_centroid.data if features_b.spectral_centroid else np.array([[1500.0]])
        mean_c_a = float(np.mean(cent_a))
        mean_c_b = float(np.mean(cent_b))
        centroid_delta = abs(mean_c_a - mean_c_b)
        pitch_sim = float(np.clip(1.0 - (centroid_delta / max(mean_c_a, mean_c_b, 1000.0)), 0.0, 1.0))

        # 3. Rolloff & Spectral Envelope Balance
        roll_a = features_a.spectral_rolloff.data if features_a.spectral_rolloff else np.array([[5000.0]])
        roll_b = features_b.spectral_rolloff.data if features_b.spectral_rolloff else np.array([[5000.0]])
        mean_r_a = float(np.mean(roll_a))
        mean_r_b = float(np.mean(roll_b))
        rolloff_sim = float(np.clip(1.0 - (abs(mean_r_a - mean_r_b) / max(mean_r_a, mean_r_b, 1000.0)), 0.0, 1.0))

        # 4. Overall Weighted Speaker Biometric Similarity
        overall_similarity = float(np.clip(
            (0.60 * mfcc_sim + 0.25 * pitch_sim + 0.15 * rolloff_sim) * 100.0,
            2.0, 99.5
        ))

        # 5. Impersonation & Forensic Findings Evaluation
        is_a_synth = classification_a.label == PredictionLabel.SYNTHETIC
        is_b_synth = classification_b.label == PredictionLabel.SYNTHETIC
        findings: List[str] = []

        if is_a_synth and overall_similarity >= 70.0:
            impersonation_risk = "CRITICAL_IMPERSONATION"
            verdict = "Elevated Impersonation Risk (Likely Voice Clone Impersonation)"
            findings.append(
                f"Questioned sample '{filename_a}' contains synthetic speech that closely mimics reference speaker '{filename_b}' ({overall_similarity:.1f}% acoustic similarity match)."
            )
            findings.append("Acoustic indicators suggest targeted neural voice cloning directed at the reference identity.")

        elif is_a_synth and overall_similarity < 70.0:
            impersonation_risk = "SYNTHETIC_UNRELATED"
            verdict = "Synthetic Voice (Acoustic Identity Dissimilar)"
            findings.append(
                f"Questioned sample '{filename_a}' is an AI synthetic voice, but its acoustic profile does not match reference speaker '{filename_b}' ({overall_similarity:.1f}% match)."
            )

        elif not is_a_synth and not is_b_synth and overall_similarity >= 75.0:
            impersonation_risk = "AUTHENTIC_MATCH"
            verdict = "High Acoustic Similarity (Verified Same Human Speaker Match)"
            findings.append(
                f"Both samples exhibit natural biological vocal tract acoustics with high acoustic similarity and consistent vocal tract resonance ({overall_similarity:.1f}% match)."
            )
            findings.append("No synthetic vocoder or voice conversion artifacts were detected in either recording.")

        elif not is_a_synth and not is_b_synth and overall_similarity < 75.0:
            impersonation_risk = "AUTHENTIC_DIFFERENT"
            verdict = "Acoustic Dissimilarity (Different Human Speakers Profile)"
            findings.append(
                f"Both recordings contain authentic human voices, but their vocal tract acoustics indicate distinct acoustic profiles ({overall_similarity:.1f}% match)."
            )

        else:
            impersonation_risk = "INCONCLUSIVE_COMPARISON"
            verdict = "Inconclusive Comparison"
            findings.append("Mixed acoustic characteristics or ambiguous confidence scores prevent definitive voice identification.")

        # Add metric findings and scientific provenance note
        findings.append(f"Cepstral envelope alignment: {mfcc_sim * 100:.1f}%.")
        findings.append(f"Pitch center delta: {centroid_delta:.1f} Hz (Sample A: {mean_c_a:.0f} Hz vs Sample B: {mean_c_b:.0f} Hz).")
        findings.append("Forensic Notice: Measurements evaluate acoustic and cepstral similarity, not legally certified biometric identity.")

        return SpeakerComparisonResult(
            speaker_similarity_percent=round(overall_similarity, 1),
            impersonation_risk=impersonation_risk,
            verdict=verdict,
            findings=findings,
            sample_a={
                "filename": filename_a,
                "label": classification_a.label.value,
                "confidence_score": classification_a.confidence_score,
                "synthetic_probability": classification_a.probabilities.get("synthetic") if classification_a.probabilities else None,
                "model_name": classification_a.model_name,
                "duration_seconds": features_a.duration_seconds
            },
            sample_b={
                "filename": filename_b,
                "label": classification_b.label.value,
                "confidence_score": classification_b.confidence_score,
                "synthetic_probability": classification_b.probabilities.get("synthetic") if classification_b.probabilities else None,
                "model_name": classification_b.model_name,
                "duration_seconds": features_b.duration_seconds
            },
            metrics={
                "mfcc_cosine_similarity": round(mfcc_sim, 4),
                "pitch_alignment_score": round(pitch_sim, 4),
                "spectral_rolloff_similarity": round(rolloff_sim, 4),
                "centroid_delta_hz": round(centroid_delta, 1),
            }
        )
