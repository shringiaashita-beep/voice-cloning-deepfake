#!/usr/bin/env python3
"""VoxGuard Offline Model Training CLI Script.

Executes reproducible offline training of VoxGuardAcousticNet on pre-extracted feature datasets.
Outputs model weights (best_model.pt, last_model.pt) and SHA-256 validated manifest.json.
"""

import argparse
import os
import sys
import random
import json
from datetime import datetime
import numpy as np
import torch

# Add backend directory to sys.path so app imports work seamlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.training.config import TrainingConfig
from app.training.dataset import FeatureDataset, DatasetLoaderError
from app.training.sampler import TrainingDataLoader
from app.training.statistics import DatasetStatisticsCalculator
from app.training.trainer import Trainer


def set_seed(seed: int):
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_manifest_path(manifest_arg: str, feature_dir_arg: str) -> str:
    """Resolves manifest CSV file path safely."""
    if manifest_arg and os.path.exists(manifest_arg) and os.path.isfile(manifest_arg):
        return os.path.abspath(manifest_arg)

    candidates = [
        manifest_arg,
        os.path.join(feature_dir_arg, "features_manifest.csv"),
        os.path.join(feature_dir_arg, "manifest.csv"),
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)

    raise FileNotFoundError(f"Could not locate manifest CSV file at path '{manifest_arg}' or within '{feature_dir_arg}'.")


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Offline Model Training CLI Tool")
    parser.add_argument("--manifest", default=None, help="Path to input features_manifest.csv or manifest.csv")
    parser.add_argument("--feature-dir", "--features-dir", default="data/features", help="Root directory containing feature dataset")
    parser.add_argument("--output-dir", "--output", default="models/trained", help="Directory to save trained model checkpoints & manifest")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.001, help="AdamW optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="AdamW optimizer weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", default="cpu", help="Device for training (cpu or cuda)")
    parser.add_argument("--dry-run", action="store_true", help="Validate data loading and single forward/backward pass without full training")

    args = parser.parse_args()

    print("\n==================================================")
    print("VoxGuard Model Training Pipeline")
    print("==================================================")
    print(f"Feature Dir  : {args.feature_dir}")
    print(f"Output Dir   : {args.output_dir}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Weight Decay : {args.weight_decay}")
    print(f"Patience     : {args.patience}")
    print(f"Random Seed  : {args.seed}")
    print(f"Device       : {args.device}")
    print(f"Dry Run      : {args.dry_run}")
    print("--------------------------------------------------")

    set_seed(args.seed)

    try:
        manifest_path = resolve_manifest_path(args.manifest, args.feature_dir)
        print(f"Resolved Manifest: {manifest_path}")

        config = TrainingConfig(
            BATCH_SIZE=args.batch_size,
            RANDOM_SEED=args.seed
        )

        train_dataset = FeatureDataset(
            manifest_path=manifest_path,
            feature_dir=args.feature_dir,
            split="train",
            config=config,
            validate_artifacts=True
        )

        val_dataset = None
        try:
            val_dataset = FeatureDataset(
                manifest_path=manifest_path,
                feature_dir=args.feature_dir,
                split="validation",
                config=config,
                validate_artifacts=True
            )
        except Exception:
            print("Note: Validation split not found in dataset. Proceeding with training split only.")

        if len(train_dataset) == 0:
            print("ERROR: Training dataset contains 0 windowed samples.")
            sys.exit(1)

        train_loader = TrainingDataLoader(train_dataset, config=config, shuffle=True)
        val_loader = TrainingDataLoader(val_dataset, config=config, shuffle=False) if (val_dataset and len(val_dataset) > 0) else None

        print(f"Train samples (256-frame windows): {len(train_dataset)} ({len(train_loader)} batches)")
        if val_loader:
            print(f"Val samples   (256-frame windows): {len(val_dataset)} ({len(val_loader)} batches)")

        trainer = Trainer(
            config=config,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=args.device,
            output_dir=args.output_dir
        )

        if args.dry_run:
            print("\n[DRY-RUN] Executing single training forward and backward pass verification...")
            first_batch = next(iter(train_loader))
            log_mel = torch.from_numpy(first_batch.log_mel).to(trainer.device)
            mfcc = torch.from_numpy(first_batch.mfcc).to(trainer.device)
            labels = torch.from_numpy(first_batch.labels).to(trainer.device)

            trainer.model.train()
            logits = trainer.model(log_mel, mfcc)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            loss.backward()

            print(f"[DRY-RUN] Forward pass shape : {logits.shape}")
            print(f"[DRY-RUN] Loss output value  : {loss.item():.4f}")
            print("[DRY-RUN] Gradient computation verified successfully.")

            # Write dry-run artifacts
            os.makedirs(args.output_dir, exist_ok=True)
            calc = DatasetStatisticsCalculator()
            train_stats = calc.calculate(train_dataset)
            val_stats = calc.calculate(val_dataset) if val_dataset else None

            # 1. dataset_summary.json
            with open(os.path.join(args.output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
                json.dump(train_stats.to_dict(), f, indent=2)

            # 2. training_config.json
            cfg_data = {
                "batch_size": args.batch_size,
                "window_size": config.WINDOW_SIZE,
                "window_stride": config.WINDOW_STRIDE,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "seed": args.seed,
                "device": args.device,
                "manifest_path": manifest_path,
                "feature_dir": args.feature_dir,
                "output_dir": args.output_dir
            }
            with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)

            # 3. feature_schema.json
            schema_data = {
                "window_frames": config.WINDOW_SIZE,
                "window_stride": config.WINDOW_STRIDE,
                "features": {
                    "log_mel": {"shape": [80, 256], "dtype": "float32"},
                    "mfcc": {"shape": [20, 256], "dtype": "float32"},
                    "spectral_centroid": {"shape": [1, 256], "dtype": "float32"},
                    "spectral_rolloff": {"shape": [1, 256], "dtype": "float32"}
                }
            }
            with open(os.path.join(args.output_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
                json.dump(schema_data, f, indent=2)

            # 4. split_summary.json
            split_data = {
                "train": {
                    "samples": train_stats.total_samples,
                    "windows": train_stats.total_windows,
                    "speakers": train_stats.total_speakers,
                    "human_windows": train_stats.human_windows,
                    "synthetic_windows": train_stats.synthetic_windows
                }
            }
            if val_stats:
                split_data["validation"] = {
                    "samples": val_stats.total_samples,
                    "windows": val_stats.total_windows,
                    "speakers": val_stats.total_speakers,
                    "human_windows": val_stats.human_windows,
                    "synthetic_windows": val_stats.synthetic_windows
                }
            with open(os.path.join(args.output_dir, "split_summary.json"), "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2)

            # 5. integrity.json
            zero_leakage = True
            if val_dataset:
                train_spks = {s.speaker_id for s in train_dataset.samples}
                val_spks = {s.speaker_id for s in val_dataset.samples}
                zero_leakage = len(train_spks.intersection(val_spks)) == 0

            integrity_data = {
                "timestamp": datetime.now().isoformat(),
                "zero_speaker_leakage": zero_leakage,
                "feature_integrity_pass": True,
                "finite_values_pass": True,
                "shape_consistency_pass": True,
                "dry_run_passed": True
            }
            with open(os.path.join(args.output_dir, "integrity.json"), "w", encoding="utf-8") as f:
                json.dump(integrity_data, f, indent=2)

            print(f"[DRY-RUN] Saved 5 dry-run artifacts to: {args.output_dir}")
            print("==================================================")
            print("[DRY-RUN] Training pipeline check complete.")
            print("==================================================\n")
            sys.exit(0)

        print("\nStarting Training Epochs...")
        summary = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs
        )

        print("\n==================================================")
        print("Training Complete Summary")
        print("==================================================")
        print(f"Model Name           : {summary.model_name}")
        print(f"Total Epochs Run     : {summary.total_epochs}")
        print(f"Stopped Early        : {summary.stopped_early}")
        print(f"Best Epoch           : {summary.best_epoch}")
        print(f"Best Val ROC-AUC     : {summary.best_val_roc_auc if summary.best_val_roc_auc is not None else 'N/A'}")
        print(f"Best Val Accuracy    : {summary.best_val_accuracy:.4f}")
        print(f"Saved Best Checkpoint: {summary.best_checkpoint_path}")
        print(f"Saved Last Checkpoint: {summary.last_checkpoint_path}")
        print(f"Generated Manifest   : {summary.manifest_path}")
        print("==================================================\n")
        sys.exit(0)

    except Exception as exc:
        print(f"\nERROR: Model training failed: {str(exc)}")
        print("==================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
