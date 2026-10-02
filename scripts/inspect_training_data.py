#!/usr/bin/env python3
"""VoxGuard Training Data Inspection CLI Script.

Inspects feature datasets, validates array shapes, counts windowed samples, class distributions,
and speaker isolation statistics without executing model training.
"""

import argparse
import os
import sys

# Add backend directory to sys.path so app imports work seamlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.training.config import TrainingConfig
from app.training.dataset import FeatureDataset, DatasetLoaderError
from app.training.sampler import TrainingDataLoader


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Training Data Inspector Tool")
    parser.add_argument("--manifest", required=True, help="Path to input features_manifest.csv or manifest.csv")
    parser.add_argument("--feature-dir", default=None, help="Root directory containing feature artifact files")
    parser.add_argument("--split", default="train", help="Dataset split to inspect (train, validation, test, ood)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for batch shape inspection")

    args = parser.parse_args()

    print("\n==================================================")
    print("VoxGuard Training Data Inspector")
    print("==================================================")
    print(f"Manifest CSV : {args.manifest}")
    print(f"Feature Dir  : {args.feature_dir or os.path.dirname(os.path.abspath(args.manifest))}")
    print(f"Split        : {args.split.upper()}")
    print(f"Batch Size   : {args.batch_size}")
    print("--------------------------------------------------")

    try:
        config = TrainingConfig(BATCH_SIZE=args.batch_size)
        dataset = FeatureDataset(
            manifest_path=args.manifest,
            feature_dir=args.feature_dir,
            split=args.split,
            config=config,
            validate_artifacts=True
        )

        total_windows = len(dataset)
        if total_windows == 0:
            print(f"No samples found for split '{args.split}'.")
            print("==================================================\n")
            sys.exit(0)

        # Class counts, Speaker counts, Source counts
        class_counts = {"human": 0, "synthetic": 0}
        speakers = set()
        sources = set()
        sample_ids = set()

        for sample in dataset:
            class_counts[sample.label_str] = class_counts.get(sample.label_str, 0) + 1
            speakers.add(sample.speaker_id)
            sources.add(sample.source)
            sample_ids.add(sample.sample_id)

        # Batch inspection using TrainingDataLoader
        loader = TrainingDataLoader(dataset=dataset, config=config)
        num_batches = len(loader)

        first_batch = next(iter(loader)) if num_batches > 0 else None

        print(f"Total Unique Audio Samples: {len(sample_ids)}")
        print(f"Total 256-Frame Windows   : {total_windows}")
        print(f"Unique Speakers           : {len(speakers)}")
        print(f"Unique Sources            : {len(sources)}")
        print("--------------------------------------------------")
        print(f"Class Counts (Windows)    : Human: {class_counts.get('human', 0)} | Synthetic: {class_counts.get('synthetic', 0)}")
        print(f"Total Batches (b={args.batch_size})    : {num_batches}")
        print("--------------------------------------------------")

        if first_batch:
            print("Tensor Shapes in First Batch:")
            print(f"  log_mel          : {first_batch.log_mel.shape} (dtype: {first_batch.log_mel.dtype})")
            print(f"  mfcc             : {first_batch.mfcc.shape} (dtype: {first_batch.mfcc.dtype})")
            print(f"  spectral_centroid: {first_batch.spectral_centroid.shape} (dtype: {first_batch.spectral_centroid.dtype})")
            print(f"  spectral_rolloff : {first_batch.spectral_rolloff.shape} (dtype: {first_batch.spectral_rolloff.dtype})")
            print(f"  labels           : {first_batch.labels.shape} (dtype: {first_batch.labels.dtype})")

        print("==================================================")
        print("Training infrastructure ready.")
        print("==================================================\n")
        sys.exit(0)

    except Exception as exc:
        print(f"\nERROR: Training data inspection failed: {str(exc)}")
        print("==================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
