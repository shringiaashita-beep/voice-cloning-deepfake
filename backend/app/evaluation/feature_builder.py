"""Feature Dataset Builder Service for VoxGuard.

Orchestrates complete dataset ingestion, audio validation, feature extraction,
feature validation, split isolation verification, artifact generation (.npz),
and metadata manifest creation without training models or downloading external assets.
"""

import csv
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import statistics as stats_lib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.evaluation.dataset_ingestor import DatasetIngestor, DatasetIngestionError, DatasetSample
from app.evaluation.dataset_integrity import DatasetIntegrity
from app.evaluation.labels import encode_label
from app.evaluation.manifest_validator import ManifestRow, ManifestValidator, ManifestValidationResult
from app.evaluation.speaker_split_validator import SpeakerSplitValidator, SpeakerSplitReport

logger = logging.getLogger("voxguard.feature_builder")


class FeatureDatasetBuilder:
    """Orchestrates end-to-end feature extraction dataset generation."""

    def __init__(self, ingestor: Optional[DatasetIngestor] = None):
        self.ingestor = ingestor or DatasetIngestor()
        self.manifest_validator = ManifestValidator()
        self.speaker_validator = SpeakerSplitValidator()

    def build_features(
        self,
        manifest_path: str,
        output_dir: str,
        base_dir: Optional[str] = None,
        overwrite: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Runs feature dataset generation pipeline.

        Args:
            manifest_path: Path to dataset manifest CSV file.
            output_dir: Output directory for generated feature artifacts.
            base_dir: Optional base directory to resolve relative audio paths.
            overwrite: If True, permits overwriting existing feature artifact files.
            dry_run: If True, validates manifest, paths, and splits without writing files.

        Returns:
            Dict containing pipeline run summary and report statuses.

        Raises:
            ValueError: If manifest invalid or speaker leakage is detected.
            FileExistsError: If output directory contains files and overwrite=False.
        """
        # 1. Manifest CSV Validation
        val_result: ManifestValidationResult = self.manifest_validator.validate_csv_manifest(
            manifest_path=manifest_path,
            base_dir=base_dir,
            check_file_existence=False
        )

        if not val_result.is_valid:
            raise ValueError(f"Manifest CSV validation failed: {val_result.errors}")

        # 2. Speaker Leakage Audit (Zero-Speaker-Overlap Verification)
        speaker_report: SpeakerSplitReport = self.speaker_validator.validate_speaker_splits(val_result.valid_rows)
        if not speaker_report.is_speaker_disjoint:
            raise ValueError(f"Speaker leakage detected across splits: {speaker_report.errors}")

        total_samples = len(val_result.valid_rows)

        # 3. Dry Run Early Return
        if dry_run:
            print("\n[DRY-RUN] Manifest CSV is valid.")
            print(f"[DRY-RUN] Speaker disjointness verified across {speaker_report.total_speakers} speakers.")
            print(f"[DRY-RUN] Ready to process {total_samples} audio samples.")
            print("[DRY-RUN] No feature files generated.\n")
            return {
                "status": "DRY_RUN_SUCCESS",
                "total_samples": total_samples,
                "total_speakers": speaker_report.total_speakers,
                "is_speaker_disjoint": True
            }

        # 4. Output Directory Preparation & Overwrite Protection
        output_dir_abs = os.path.abspath(output_dir)
        if os.path.exists(output_dir_abs) and os.listdir(output_dir_abs):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory '{output_dir}' already exists and is not empty. Use --overwrite to replace."
                )

        os.makedirs(output_dir_abs, exist_ok=True)
        split_dirs = ["train", "validation", "test", "ood"]
        for sdir in split_dirs:
            os.makedirs(os.path.join(output_dir_abs, sdir), exist_ok=True)

        # Trackers for outputs
        successful_samples: List[Dict[str, Any]] = []
        failed_samples: List[Dict[str, Any]] = []
        artifact_hashes: Dict[str, str] = {}
        manifest_csv_rows: List[Dict[str, Any]] = []

        # Split statistics tracking
        split_stats: Dict[str, Dict[str, Any]] = {
            s: {
                "sample_count": 0,
                "class_counts": {"human": 0, "synthetic": 0},
                "durations": [],
                "feature_frames": []
            }
            for s in split_dirs
        }

        # 5. Sequential Ingestion & Feature Artifact Generation
        for row in val_result.valid_rows:
            try:
                sample_record, features_obj = self.ingestor.process_sample_row(row, base_dir=base_dir)

                split_name = sample_record.split
                if split_name not in split_dirs:
                    split_name = "train"

                # Generate relative artifact filename: split/sample_id.npz
                rel_artifact_path = f"{split_name}/{sample_record.sample_id}.npz"
                abs_artifact_path = os.path.join(output_dir_abs, rel_artifact_path)

                # Save .npz feature artifact
                label_code = encode_label(sample_record.label)
                np.savez_compressed(
                    abs_artifact_path,
                    log_mel=features_obj.log_mel_spectrogram.data,
                    mfcc=features_obj.mfcc.data,
                    spectral_centroid=features_obj.spectral_centroid.data,
                    spectral_rolloff=features_obj.spectral_rolloff.data,
                    label_code=np.int64(label_code),
                    sample_id=np.array(sample_record.sample_id),
                    speaker_id=np.array(sample_record.speaker_id),
                    source=np.array(sample_record.source),
                    source_id=np.array(sample_record.source_id),
                    split=np.array(sample_record.split)
                )

                # Compute SHA-256 hash of generated .npz
                digest = DatasetIntegrity.compute_file_sha256(abs_artifact_path)
                artifact_hashes[rel_artifact_path] = digest

                num_frames = features_obj.log_mel_spectrogram.data.shape[1]
                duration_sec = sample_record.audio_metadata["duration_seconds"]

                # Record manifest CSV row
                csv_entry = {
                    "feature_path": rel_artifact_path,
                    "sample_id": sample_record.sample_id,
                    "label": sample_record.label,
                    "speaker_id": sample_record.speaker_id,
                    "source": sample_record.source,
                    "source_id": sample_record.source_id,
                    "split": sample_record.split,
                    "duration_seconds": duration_sec,
                    "num_feature_frames": num_frames
                }
                manifest_csv_rows.append(csv_entry)
                successful_samples.append(sample_record.to_dict())

                # Update per-split statistics
                split_stats[split_name]["sample_count"] += 1
                if sample_record.label in split_stats[split_name]["class_counts"]:
                    split_stats[split_name]["class_counts"][sample_record.label] += 1
                split_stats[split_name]["durations"].append(duration_sec)
                split_stats[split_name]["feature_frames"].append(num_frames)

            except DatasetIngestionError as exc:
                failed_samples.append({
                    "sample_id": f"{row.split}_{row.speaker_id}_{Path(row.file_path).stem}",
                    "error_code": exc.code,
                    "safe_message": exc.message
                })
            except Exception as exc:
                failed_samples.append({
                    "sample_id": f"{row.split}_{row.speaker_id}_{Path(row.file_path).stem}",
                    "error_code": "PROCESSING_ERROR",
                    "safe_message": "An error occurred during sample processing."
                })

        # 6. Write features_manifest.csv
        csv_manifest_path = os.path.join(output_dir_abs, "features_manifest.csv")
        csv_fieldnames = [
            "feature_path", "sample_id", "label", "speaker_id",
            "source", "source_id", "split", "duration_seconds", "num_feature_frames"
        ]
        with open(csv_manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(manifest_csv_rows)

        # 7. Write features_manifest.json
        manifest_json_data = DatasetIntegrity.build_features_manifest_json(
            artifact_hashes=artifact_hashes,
            dataset_version="1.0.0",
            feature_pipeline_version="voxguard-features-0.1.0"
        )
        json_manifest_path = os.path.join(output_dir_abs, "features_manifest.json")
        with open(json_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_json_data, f, indent=2)

        # 8. Write processing_report.json
        processing_report = {
            "total_samples": total_samples,
            "successful_samples": len(successful_samples),
            "failed_samples": len(failed_samples),
            "failures": failed_samples
        }
        report_path = os.path.join(output_dir_abs, "processing_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(processing_report, f, indent=2)

        # 9. Write statistics.json
        formatted_stats: Dict[str, Any] = {"splits": {}, "total": {}}
        all_durations: List[float] = []
        all_frames: List[int] = []
        total_human = 0
        total_synthetic = 0

        for split_key, data in split_stats.items():
            cnt = data["sample_count"]
            durs = data["durations"]
            frms = data["feature_frames"]
            all_durations.extend(durs)
            all_frames.extend(frms)
            total_human += data["class_counts"]["human"]
            total_synthetic += data["class_counts"]["synthetic"]

            formatted_stats["splits"][split_key] = {
                "sample_count": cnt,
                "class_counts": data["class_counts"],
                "duration_seconds": self._calc_summary_stats(durs),
                "feature_frames": self._calc_summary_stats(frms)
            }

        formatted_stats["total"] = {
            "sample_count": len(successful_samples),
            "class_counts": {"human": total_human, "synthetic": total_synthetic},
            "duration_seconds": self._calc_summary_stats(all_durations),
            "feature_frames": self._calc_summary_stats(all_frames)
        }

        stats_path = os.path.join(output_dir_abs, "statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(formatted_stats, f, indent=2)

        return {
            "status": "SUCCESS",
            "output_dir": output_dir_abs,
            "total_samples": total_samples,
            "successful_samples": len(successful_samples),
            "failed_samples": len(failed_samples)
        }

    @staticmethod
    def _calc_summary_stats(values: List[Any]) -> Dict[str, Optional[float]]:
        """Helper computing min, max, mean, median for numeric lists."""
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

        float_vals = [float(v) for v in values]
        return {
            "min": round(min(float_vals), 4),
            "max": round(max(float_vals), 4),
            "mean": round(float(stats_lib.mean(float_vals)), 4),
            "median": round(float(stats_lib.median(float_vals)), 4)
        }
