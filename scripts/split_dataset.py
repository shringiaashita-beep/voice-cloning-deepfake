#!/usr/bin/env python3
"""VoxGuard Deterministic Speaker Dataset Splitter CLI Script.

Partitions dataset CSV manifests strictly by Speaker ID (not individual audio frames or files)
using deterministic pseudo-random seed shuffling (default seed=42) and writes a prepared manifest.
"""

import argparse
import sys
import os

# Add backend directory to sys.path so app imports work seamlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.evaluation.splitter import DeterministicSpeakerSplitter


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Deterministic Speaker Dataset Splitter")
    parser.add_argument("--manifest", required=True, help="Path to input manifest CSV file")
    parser.add_argument("--output", required=True, help="Path to write output prepared manifest CSV file")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Target ratio for TRAIN split (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Target ratio for VALIDATION split (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Target ratio for TEST split (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Pseudo-random seed integer (default: 42)")

    args = parser.parse_args()

    # Safety check: Prevent overwriting source manifest unless user specifies identical paths explicitly
    norm_in = os.path.normpath(os.path.abspath(args.manifest))
    norm_out = os.path.normpath(os.path.abspath(args.output))
    if norm_in == norm_out:
        print("ERROR: Output path matches input path. Refusing to overwrite input manifest directly.")
        sys.exit(1)

    print("\n==================================================")
    print("VoxGuard Deterministic Speaker Dataset Splitter")
    print("==================================================")
    print(f"Input Manifest  : {args.manifest}")
    print(f"Output Manifest : {args.output}")
    print(f"Ratios          : Train: {args.train_ratio:.2f} | Val: {args.val_ratio:.2f} | Test: {args.test_ratio:.2f}")
    print(f"Random Seed     : {args.seed}")
    print("--------------------------------------------------")

    try:
        splitter = DeterministicSpeakerSplitter()
        updated_rows = splitter.split_manifest(
            manifest_path=args.manifest,
            output_path=args.output,
            train_ratio=args.train_ratio,
            validation_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

        print(f"SUCCESS: Successfully processed {len(updated_rows)} dataset samples.")
        print(f"Saved prepared manifest to: '{args.output}'")
        print("==================================================\n")
        sys.exit(0)

    except Exception as exc:
        print(f"\nERROR: Splitting failed: {str(exc)}")
        print("==================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
