"""Training Dataset Statistics & Risk Assessment Module for VoxGuard.

Calculates comprehensive statistics (window counts, speaker counts, class balance,
sequence length statistics, speaker/source distributions) and flags potential dataset risks
(class imbalance, dominant speaker/source, suspicious split distributions) without modifying data.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import numpy as np

from app.training.dataset import FeatureDataset


@dataclass
class DatasetRiskAssessment:
    """Strongly typed report of potential dataset risks & anomalies."""
    has_class_imbalance: bool = False
    imbalance_warning: Optional[str] = None
    has_dominant_speaker: bool = False
    dominant_speaker_warning: Optional[str] = None
    has_dominant_source: bool = False
    dominant_source_warning: Optional[str] = None
    has_suspicious_split: bool = False
    suspicious_split_warning: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class DatasetStatisticsReport:
    """Strongly typed summary of training feature dataset statistics."""
    split: str
    total_samples: int
    total_windows: int
    total_speakers: int
    human_windows: int
    synthetic_windows: int
    human_percentage: float
    synthetic_percentage: float
    windows_per_speaker: Dict[str, int]
    windows_per_source: Dict[str, int]
    min_sequence_length: int
    max_sequence_length: int
    mean_sequence_length: float
    median_sequence_length: float
    risk_assessment: DatasetRiskAssessment

    def to_dict(self) -> Dict[str, Any]:
        """Converts statistics report to JSON-serializable dictionary."""
        return asdict(self)


class DatasetStatisticsCalculator:
    """Calculates dataset statistics and audits for data distribution anomalies."""

    def calculate(self, dataset: FeatureDataset) -> DatasetStatisticsReport:
        """Calculates complete statistics report from FeatureDataset."""
        split = dataset.split
        samples = dataset.samples
        total_windows = len(samples)

        if total_windows == 0:
            return DatasetStatisticsReport(
                split=split,
                total_samples=0,
                total_windows=0,
                total_speakers=0,
                human_windows=0,
                synthetic_windows=0,
                human_percentage=0.0,
                synthetic_percentage=0.0,
                windows_per_speaker={},
                windows_per_source={},
                min_sequence_length=0,
                max_sequence_length=0,
                mean_sequence_length=0.0,
                median_sequence_length=0.0,
                risk_assessment=DatasetRiskAssessment(
                    has_suspicious_split=True,
                    suspicious_split_warning=f"Split '{split}' contains 0 windowed samples.",
                    warnings=[f"Split '{split}' contains 0 windowed samples."]
                )
            )

        sample_ids = {s.sample_id for s in samples}
        total_samples = len(sample_ids)

        human_windows = sum(1 for s in samples if s.label == 0)
        synthetic_windows = sum(1 for s in samples if s.label == 1)

        human_pct = round((human_windows / total_windows) * 100.0, 2)
        synth_pct = round((synthetic_windows / total_windows) * 100.0, 2)

        spk_counts: Dict[str, int] = {}
        src_counts: Dict[str, int] = {}
        sample_win_counts: Dict[str, int] = {}

        for s in samples:
            spk_counts[s.speaker_id] = spk_counts.get(s.speaker_id, 0) + 1
            src_counts[s.source] = src_counts.get(s.source, 0) + 1
            sample_win_counts[s.sample_id] = sample_win_counts.get(s.sample_id, 0) + 1

        total_speakers = len(spk_counts)

        # Derived original sequence lengths prior to 256-frame windowing
        seq_lengths = [count * 128 + 128 for count in sample_win_counts.values()]
        min_len = min(seq_lengths) if seq_lengths else 256
        max_len = max(seq_lengths) if seq_lengths else 256
        mean_len = float(np.mean(seq_lengths)) if seq_lengths else 256.0
        median_len = float(np.median(seq_lengths)) if seq_lengths else 256.0

        warnings: List[str] = []
        has_imbalance = False
        imbalance_warn = None
        has_dom_spk = False
        dom_spk_warn = None
        has_dom_src = False
        dom_src_warn = None
        has_susp = False
        susp_warn = None

        if human_windows == 0 or synthetic_windows == 0:
            has_imbalance = True
            imbalance_warn = f"Severe class imbalance: split '{split}' contains zero samples for one class."
            warnings.append(imbalance_warn)
        elif abs(human_pct - synth_pct) > 20.0:
            has_imbalance = True
            imbalance_warn = f"Class imbalance detected: human={human_pct}%, synthetic={synth_pct}%."
            warnings.append(imbalance_warn)

        for spk, count in spk_counts.items():
            ratio = count / total_windows
            if ratio > 0.30 and total_speakers > 1:
                has_dom_spk = True
                dom_spk_warn = f"Unusually dominant speaker detected: '{spk}' accounts for {ratio*100:.1f}% of total windows."
                warnings.append(dom_spk_warn)
                break

        for src, count in src_counts.items():
            ratio = count / total_windows
            if ratio > 0.50 and len(src_counts) > 1:
                has_dom_src = True
                dom_src_warn = f"Unusually dominant source detected: '{src}' accounts for {ratio*100:.1f}% of total windows."
                warnings.append(dom_src_warn)
                break

        if total_windows < 5:
            has_susp = True
            susp_warn = f"Suspicious split size: split '{split}' contains only {total_windows} windowed samples."
            warnings.append(susp_warn)

        risk_assessment = DatasetRiskAssessment(
            has_class_imbalance=has_imbalance,
            imbalance_warning=imbalance_warn,
            has_dominant_speaker=has_dom_spk,
            dominant_speaker_warning=dom_spk_warn,
            has_dominant_source=has_dom_src,
            dominant_source_warning=dom_src_warn,
            has_suspicious_split=has_susp,
            suspicious_split_warning=susp_warn,
            warnings=warnings
        )

        return DatasetStatisticsReport(
            split=split,
            total_samples=total_samples,
            total_windows=total_windows,
            total_speakers=total_speakers,
            human_windows=human_windows,
            synthetic_windows=synthetic_windows,
            human_percentage=human_pct,
            synthetic_percentage=synth_pct,
            windows_per_speaker=spk_counts,
            windows_per_source=src_counts,
            min_sequence_length=int(min_len),
            max_sequence_length=int(max_len),
            mean_sequence_length=round(mean_len, 1),
            median_sequence_length=round(median_len, 1),
            risk_assessment=risk_assessment
        )
