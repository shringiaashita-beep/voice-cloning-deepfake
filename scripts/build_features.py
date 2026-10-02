#!/usr/bin/env python3
"""VoxGuard Feature Dataset Generation CLI Script.

Converts validated audio manifest entries into ML-ready feature artifacts (.npz),
verifying speaker-disjoint splits and calculating SHA-256 digests without training models.
"""

import argparse
import os
import sys

# Add backend directory to sys.path so app imports work seamlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.evaluation.feature_builder import FeatureDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Feature Dataset Generator Tool")
    parser.add_argument("--manifest", required=True, help="Path to input dataset manifest CSV file")
    parser.add_argument("--output", default="datasets/prepared/features", help="Output directory for feature artifacts")
    parser.add_argument("--base-dir", default=None, help="Base directory to resolve relative audio file_path locations")
    parser.add_argument("--overwrite", action="store_true", help="Permit overwriting existing feature artifact files")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest, audio paths, and splits without creating feature files")

    args = parser.parse_args()

    print("\n==================================================")
    print("VoxGuard Feature Dataset Generator")
    print("==================================================")
    print(f"Manifest CSV : {args.manifest}")
    print(f"Output Dir   : {args.output}")
    print(f"Base Dir     : {args.base_dir or '.'}")
    print(f"Overwrite    : {args.overwrite}")
    print(f"Dry Run      : {args.dry_run}")
    print("--------------------------------------------------")

    try:
        builder = FeatureDatasetBuilder()
        res = builder.build_features(
            manifest_path=args.manifest,
            output_dir=args.output,
            base_dir=args.base_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run
        )

        if args.dry_run:
            print(f"DRY RUN SUCCESSFUL: Validated {res['total_samples']} samples across {res['total_speakers']} speakers.")
            print("==================================================\n")
            sys.exit(0)

        print(f"FEATURE GENERATION SUCCESSFUL!")
        print(f"Total Samples Processed: {res['total_samples']}")
        print(f"Successful Artifacts   : {res['successful_samples']}")
        print(f"Failed Samples         : {res['failed_samples']}")
        print(f"Artifacts Output Dir   : '{res['output_dir']}'")
        print("==================================================\n")
        sys.exit(0)

    except Exception as exc:
        print(f"\nERROR: Feature dataset generation failed: {str(exc)}")
        print("==================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
