#!/usr/bin/env python3
"""VoxGuard Dataset Validation CLI Script.

Audits dataset manifest CSV files, checks audio stream integrity, verifies speaker-disjoint splits,
and reports overall dataset readiness status without training models or downloading data.
"""

import argparse
import sys
import os

# Add backend directory to sys.path so app imports work seamlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.evaluation.dataset_report import DatasetReportBuilder


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Dataset Readiness & Speaker Leakage Validation Tool")
    parser.add_argument("--manifest", required=True, help="Path to dataset manifest CSV file")
    parser.add_argument("--base-dir", default=None, help="Base directory to resolve relative file_path locations")
    parser.add_argument("--no-audio-audit", action="store_true", help="Skip decoding/checking local audio file streams")

    args = parser.parse_args()

    builder = DatasetReportBuilder()
    report = builder.build_report(
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        audit_audio=not args.no_audio_audit
    )

    print("\n==================================================")
    print("VoxGuard Dataset Validation Audit Report")
    print("==================================================")
    print(f"Manifest CSV Status   : {report.manifest_status}")
    print(f"Audio Audit Status   : {report.audio_audit_status}")
    print(f"Speaker Disjointness : {report.speaker_disjoint_status}")
    print(f"Overall Dataset Status: {report.overall_status}")
    print("--------------------------------------------------")
    print(f"Total Samples        : {report.total_samples}")
    print(f"Valid Audio Files    : {report.valid_audio_count}")
    print(f"Invalid/Missing Files: {report.invalid_audio_count}")
    print(f"Total Speakers       : {report.speaker_split_report.total_speakers}")
    print("--------------------------------------------------")
    print("Class Balance per Split:")
    for split_name, stats in report.class_balance_report.splits.items():
        if stats.total > 0:
            print(f"  [{split_name.upper():<10}] Total: {stats.total:<4} | Human: {stats.human:<4} ({stats.human_pct:.1f}%) | Synthetic: {stats.synthetic:<4} ({stats.synthetic_pct:.1f}%)")
    print("--------------------------------------------------")
    if report.duration_stats and report.duration_stats.get("mean", 0) > 0:
        ds = report.duration_stats
        print(f"Audio Duration Stats : Min: {ds['min']}s | Max: {ds['max']}s | Mean: {ds['mean']}s | Median: {ds['median']}s")
        print("--------------------------------------------------")

    if report.errors:
        print("\nERRORS DETECTED:")
        for err in report.errors:
            print(f"  [!] {err}")

    if report.warnings:
        print("\nWARNINGS:")
        for warn in report.warnings:
            print(f"  [*] {warn}")

    print("\n==================================================\n")

    if report.overall_status != "READY":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
