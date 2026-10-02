"""Forensic Report Builder Module for VoxGuard.

Compiles audio metadata, signal analysis findings, baseline acoustic statistics,
and evidence indicators into an explainable forensic report.
Enforces explicit provenance tags ("metadata", "signal_analysis", "heuristic", "ml_model")
and strictly avoids stating that acoustic anomalies prove synthetic speech.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import AudioFeatures
from app.models.base_classifier import ClassificationResult, ModelStatus, PredictionLabel


@dataclass
class EvidenceIndicator:
    """Individual forensic evidence indicator with explicit provenance and non-overreaching interpretation."""
    name: str
    measured_value: str
    interpretation: str
    severity: str                     # "LOW", "MEDIUM", "HIGH", "INFO"
    confidence_strength: str          # E.g., "Low", "Moderate", "High", "N/A"
    is_model_derived: bool            # True if derived from ML model, False if signal/rule derived
    provenance: str                   # "metadata", "signal_analysis", "heuristic", "ml_model"


@dataclass
class ForensicReport:
    """Explainable forensic report structure returned in API responses."""
    audio_metadata: Dict[str, Any]
    acoustic_observations: Dict[str, Any]
    model_assessment: Optional[Dict[str, Any]]  # ONLY present when status == "ready"
    evidence_indicators: List[EvidenceIndicator]
    disclaimer: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts report object to clean dictionary for REST API serialization."""
        res = asdict(self)
        res["evidence_indicators"] = [asdict(ind) for ind in self.evidence_indicators]
        return res


class ForensicReportBuilder:
    """Service compiling metadata, acoustic indicators, and model outputs into forensic reports."""

    def build_report(
        self,
        classification: ClassificationResult,
        features: AudioFeatures,
        preprocessed: PreprocessedAudio
    ) -> ForensicReport:
        """Compiles a complete ForensicReport object.

        Args:
            classification: ClassificationResult output from classifier.
            features: AudioFeatures extracted from canonical audio payload.
            preprocessed: PreprocessedAudio canonical audio metadata.

        Returns:
            ForensicReport containing explainable findings, provenance, and disclaimers.
        """
        meta_dict = preprocessed.metadata or {}
        # 1. Audio Metadata Category
        audio_meta = {
            "duration_seconds": preprocessed.duration_seconds,
            "sample_rate_hz": preprocessed.sample_rate,
            "channels": preprocessed.channels,
            "num_frames": preprocessed.num_frames,
            "format": meta_dict.get("format", "pcm")
        }

        # 2. Acoustic Observations Category
        acoustic_obs = self._compile_acoustic_observations(features)

        # 3. Model Assessment Category (ONLY present when status == "ready")
        model_assessment: Optional[Dict[str, Any]] = None
        if classification.status == ModelStatus.READY:
            model_assessment = {
                "verdict": classification.label.value,
                "probabilities": classification.probabilities,
                "confidence_score": classification.confidence_score,
                "model_name": classification.model_name,
                "model_version": classification.model_version
            }

        # 4. Evidence / Indicators List with Provenance
        indicators = self._build_evidence_indicators(classification, features, preprocessed)

        # 5. Mandatory Provenance & Evaluation Disclaimer
        disclaimer = (
            "VoxGuard forensic reports provide acoustic feature indicators and model assessments. "
            "Acoustic indicators reflect observed signal characteristics and do not constitute "
            "proof of AI-generated speech. Model probability outputs (when available) represent "
            "evaluated statistical predictions. Refer to docs/EVALUATION_METRICS.md for benchmarking protocols."
        )

        iso_timestamp = datetime.now(timezone.utc).isoformat()

        return ForensicReport(
            audio_metadata=audio_meta,
            acoustic_observations=acoustic_obs,
            model_assessment=model_assessment,
            evidence_indicators=indicators,
            disclaimer=disclaimer,
            timestamp=iso_timestamp
        )

    def _compile_acoustic_observations(self, features: AudioFeatures) -> Dict[str, Any]:
        """Compiles objective acoustic statistics across features."""
        obs: Dict[str, Any] = {}

        if features is None or features.log_mel_spectrogram is None:
            return {"note": "No features available for acoustic observation."}

        # Spectral Centroid
        if features.spectral_centroid and features.spectral_centroid.data.size > 0:
            c_data = features.spectral_centroid.data
            obs["spectral_centroid"] = {
                "mean_hz": round(float(np.mean(c_data)), 2),
                "std_hz": round(float(np.std(c_data)), 2),
                "min_hz": round(float(np.min(c_data)), 2),
                "max_hz": round(float(np.max(c_data)), 2)
            }

        # Spectral Rolloff
        if features.spectral_rolloff and features.spectral_rolloff.data.size > 0:
            r_data = features.spectral_rolloff.data
            obs["spectral_rolloff_85_percent"] = {
                "mean_hz": round(float(np.mean(r_data)), 2),
                "std_hz": round(float(np.std(r_data)), 2)
            }

        # MFCC Statistics
        if features.mfcc and features.mfcc.data.size > 0:
            mfcc_data = features.mfcc.data
            obs["mfcc_statistics"] = {
                "num_coefficients": mfcc_data.shape[0],
                "mean": round(float(np.mean(mfcc_data)), 2),
                "variance": round(float(np.var(mfcc_data)), 2)
            }

        # Signal Characteristics
        if features.log_mel_spectrogram and features.log_mel_spectrogram.data.size > 0:
            log_mel = features.log_mel_spectrogram.data
            obs["signal_characteristics"] = {
                "average_energy_db": round(float(np.mean(log_mel)), 2),
                "dynamic_range_db": round(float(np.max(log_mel) - np.min(log_mel)), 2)
            }

        return obs

    def _build_evidence_indicators(
        self,
        classification: ClassificationResult,
        features: AudioFeatures,
        preprocessed: PreprocessedAudio
    ) -> List[EvidenceIndicator]:
        """Builds evidence indicators with explicit provenance tags."""
        indicators: List[EvidenceIndicator] = []

        # 1. Metadata Provenance Indicator
        indicators.append(EvidenceIndicator(
            name="Sample Rate Standardization",
            measured_value=f"{preprocessed.sample_rate} Hz Mono",
            interpretation="Audio stream decoded and standardized to canonical 16 kHz mono format.",
            severity="INFO",
            confidence_strength="N/A",
            is_model_derived=False,
            provenance="metadata"
        ))

        # 2. Signal Analysis Provenance Indicator
        if features and features.spectral_centroid and features.spectral_centroid.data.size > 0:
            mean_c = float(np.mean(features.spectral_centroid.data))
            indicators.append(EvidenceIndicator(
                name="Spectral Centroid Distribution",
                measured_value=f"{mean_c:.1f} Hz",
                interpretation="Observed acoustic characteristic representing the brightness center of spectral mass.",
                severity="INFO",
                confidence_strength="N/A",
                is_model_derived=False,
                provenance="signal_analysis"
            ))

        # 3. Heuristic / System State Provenance Indicator
        if classification.status == ModelStatus.ANALYSIS_ONLY:
            indicators.append(EvidenceIndicator(
                name="System Operational Mode",
                measured_value="Analysis Only (Baseline)",
                interpretation="System operating on uncalibrated statistical baseline. No ML detection probabilities computed.",
                severity="INFO",
                confidence_strength="N/A",
                is_model_derived=False,
                provenance="heuristic"
            ))
        elif classification.status == ModelStatus.NOT_READY:
            indicators.append(EvidenceIndicator(
                name="Model Readiness Status",
                measured_value="Model Not Ready",
                interpretation="Model weights are uninitialized.",
                severity="HIGH",
                confidence_strength="N/A",
                is_model_derived=False,
                provenance="heuristic"
            ))

        # 4. ML Model Provenance Indicator (ONLY if status == "ready")
        if classification.status == ModelStatus.READY:
            verdict_text = classification.label.value
            conf_val = f"{classification.confidence_score:.2f}" if classification.confidence_score else "N/A"
            indicators.append(EvidenceIndicator(
                name="ML Classifier Decision",
                measured_value=f"Label: {verdict_text}",
                interpretation=f"Model-derived statistical prediction from {classification.model_name}.",
                severity="HIGH" if verdict_text == "synthetic" else "LOW",
                confidence_strength=conf_val,
                is_model_derived=True,
                provenance="ml_model"
            ))

            # Include detailed Voice Clone Evidence Indicators if available in classification metadata
            vc_indicators = (classification.metadata or {}).get("voice_clone_indicators", [])
            for ind in vc_indicators:
                indicators.append(EvidenceIndicator(
                    name=ind.get("name", "Voice Clone Indicator"),
                    measured_value=ind.get("measured_value", ""),
                    interpretation=ind.get("interpretation", ""),
                    severity=ind.get("severity", "INFO"),
                    confidence_strength=ind.get("confidence_strength", "N/A"),
                    is_model_derived=ind.get("is_model_derived", True),
                    provenance=ind.get("provenance", "ml_model")
                ))

        return indicators
