"""Dataset Quality Report Builder for VoxGuard.

Consolidates manifest validation, audio stream auditing, speaker-disjoint split verification,
duration statistics (min, max, mean, median), format distributions, and class balance
into a unified dataset readiness report.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.audio.decoder import AudioDecoder
from app.audio.validator import AudioValidator
from app.config import settings
from app.evaluation.class_balance import ClassBalanceAnalyzer, ClassBalanceReport
from app.evaluation.manifest_validator import ManifestRow, ManifestValidator, ManifestValidationResult
from app.evaluation.speaker_split_validator import SpeakerSplitReport, SpeakerSplitValidator

logger = logging.getLogger("voxguard.dataset_report")


@dataclass
class DatasetQualityReport:
    """Comprehensive dataset readiness and quality audit report."""
    manifest_status: str                   # "VALID" or "INVALID"
    audio_audit_status: str                # "VALID" or "INVALID"
    speaker_disjoint_status: str           # "PASS" or "FAIL"
    overall_status: str                    # "READY" or "NOT_READY"
    total_samples: int
    valid_audio_count: int
    invalid_audio_count: int
    duration_stats: Dict[str, float]       # min, max, mean, median
    format_distribution: Dict[str, int]
    sample_rate_distribution: Dict[str, int]
    channel_distribution: Dict[str, int]
    speaker_split_report: SpeakerSplitReport
    class_balance_report: ClassBalanceReport
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_status": self.manifest_status,
            "audio_audit_status": self.audio_audit_status,
            "speaker_disjoint_status": self.speaker_disjoint_status,
            "overall_status": self.overall_status,
            "total_samples": self.total_samples,
            "valid_audio_count": self.valid_audio_count,
            "invalid_audio_count": self.invalid_audio_count,
            "duration_stats": self.duration_stats,
            "format_distribution": self.format_distribution,
            "sample_rate_distribution": self.sample_rate_distribution,
            "channel_distribution": self.channel_distribution,
            "speaker_split_report": self.speaker_split_report.to_dict(),
            "class_balance_report": self.class_balance_report.to_dict(),
            "errors": self.errors,
            "warnings": self.warnings
        }


class DatasetReportBuilder:
    """Service building complete dataset quality and readiness reports."""

    def __init__(self):
        self.manifest_validator = ManifestValidator()
        self.speaker_validator = SpeakerSplitValidator()
        self.balance_analyzer = ClassBalanceAnalyzer()
        self.audio_validator = AudioValidator(config=settings)
        self.audio_decoder = AudioDecoder(validator=self.audio_validator)

    def build_report(
        self,
        manifest_path: str,
        base_dir: Optional[str] = None,
        audit_audio: bool = True
    ) -> DatasetQualityReport:
        """Builds a complete DatasetQualityReport for a given CSV manifest.

        Args:
            manifest_path: Path to dataset manifest CSV file.
            base_dir: Optional directory resolving relative file_path locations.
            audit_audio: If True, reads and decodes audio files to compute duration/format stats.

        Returns:
            DatasetQualityReport detailing readiness status, duration stats, and split integrity.
        """
        # 1. Manifest Validation (CSV syntax and required fields)
        manifest_res = self.manifest_validator.validate_csv_manifest(
            manifest_path,
            base_dir=base_dir,
            check_file_existence=False
        )

        errors = list(manifest_res.errors)
        warnings = list(manifest_res.warnings)

        if not manifest_res.is_valid or not manifest_res.valid_rows:
            spk_report = self.speaker_validator.validate_speaker_splits([])
            bal_report = self.balance_analyzer.analyze([])
            return DatasetQualityReport(
                manifest_status="INVALID",
                audio_audit_status="NOT_RUN",
                speaker_disjoint_status="NOT_RUN",
                overall_status="NOT_READY",
                total_samples=0,
                valid_audio_count=0,
                invalid_audio_count=0,
                duration_stats={"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0},
                format_distribution={},
                sample_rate_distribution={},
                channel_distribution={},
                speaker_split_report=spk_report,
                class_balance_report=bal_report,
                errors=errors,
                warnings=warnings
            )

        rows = manifest_res.valid_rows

        # 2. Speaker Split Verification
        spk_report = self.speaker_validator.validate_speaker_splits(rows)
        errors.extend(spk_report.errors)
        warnings.extend(spk_report.warnings)

        # 3. Class Balance Analysis
        bal_report = self.balance_analyzer.analyze(rows)
        warnings.extend(bal_report.warnings)

        # 4. Audio Stream Auditing (Duration, Format, Channels)
        valid_audio_count = 0
        invalid_audio_count = 0
        durations: List[float] = []
        formats: Dict[str, int] = {}
        sample_rates: Dict[str, int] = {}
        channels_map: Dict[str, int] = {}

        if audit_audio:
            for r in rows:
                full_path = os.path.join(base_dir, r.file_path) if base_dir else r.file_path
                ext = os.path.splitext(r.file_path)[1].lower()
                formats[ext] = formats.get(ext, 0) + 1

                if not os.path.exists(full_path):
                    invalid_audio_count += 1
                    errors.append(f"Audio file missing on disk: '{r.file_path}'")
                    continue

                try:
                    with open(full_path, "rb") as f:
                        file_bytes = f.read()

                    # Validate bytes using AudioValidator
                    val_res = self.audio_validator.validate(file_bytes, filename=r.file_path)
                    if not val_res.is_valid:
                        invalid_audio_count += 1
                        errors.append(f"Audio validation failed for '{r.file_path}': {val_res.error_message}")
                        continue

                    # Decode stream to extract duration/sample rate
                    decoded = self.audio_decoder.decode(file_bytes, filename=r.file_path)
                    valid_audio_count += 1
                    durations.append(decoded.duration_seconds)
                    
                    sr_str = f"{decoded.sample_rate} Hz"
                    sample_rates[sr_str] = sample_rates.get(sr_str, 0) + 1

                    ch_str = f"{decoded.channels} ch"
                    channels_map[ch_str] = channels_map.get(ch_str, 0) + 1

                except Exception as exc:
                    invalid_audio_count += 1
                    errors.append(f"Error reading audio stream '{r.file_path}': {str(exc)}")
        else:
            valid_audio_count = len(rows)

        # Duration Statistics Calculation (min, max, mean, median)
        duration_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        if durations:
            duration_stats = {
                "min": round(float(np.min(durations)), 2),
                "max": round(float(np.max(durations)), 2),
                "mean": round(float(np.mean(durations)), 2),
                "median": round(float(np.median(durations)), 2)
            }

        # Overall Readiness Determination
        manifest_status = "VALID" if manifest_res.is_valid else "INVALID"
        audio_status = "VALID" if invalid_audio_count == 0 else "INVALID"
        spk_status = "PASS" if spk_report.is_speaker_disjoint else "FAIL"

        overall_ready = (
            manifest_status == "VALID" and
            audio_status == "VALID" and
            spk_status == "PASS" and
            len(errors) == 0
        )
        overall_status = "READY" if overall_ready else "NOT_READY"

        return DatasetQualityReport(
            manifest_status=manifest_status,
            audio_audit_status=audio_status,
            speaker_disjoint_status=spk_status,
            overall_status=overall_status,
            total_samples=len(rows),
            valid_audio_count=valid_audio_count,
            invalid_audio_count=invalid_audio_count,
            duration_stats=duration_stats,
            format_distribution=formats,
            sample_rate_distribution=sample_rates,
            channel_distribution=channels_map,
            speaker_split_report=spk_report,
            class_balance_report=bal_report,
            errors=errors,
            warnings=warnings
        )
