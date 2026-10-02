#!/usr/bin/env python3
"""VoxGuard Offline Model Evaluation & OOD Benchmark CLI Script.

Evaluates a trained model checkpoint across validation, test, and OOD splits.
Supports threshold search strictly on validation split (never test or OOD data),
computing Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1-Score,
ROC-AUC, PR-AUC, and Confusion Matrices. Writes JSON evaluation reports.
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple
import numpy as np
import torch

# Add backend directory to sys.path so app imports work seamlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.training.config import TrainingConfig
from app.training.dataset import FeatureDataset, DatasetLoaderError
from app.training.sampler import TrainingDataLoader
from app.evaluation.model_evaluator import ModelEvaluator, EvaluationMetrics
from app.models.model_loader import ModelLoader


def resolve_model_path(model_arg: str) -> str:
    """Resolves path to trained model PyTorch checkpoint file."""
    if os.path.isfile(model_arg) and model_arg.endswith(".pt"):
        return os.path.abspath(model_arg)

    if os.path.isdir(model_arg):
        candidates = [
            os.path.join(model_arg, "best_model.pt"),
            os.path.join(model_arg, "last_model.pt"),
            os.path.join(model_arg, "model.pt")
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)

    raise FileNotFoundError(f"Could not locate model checkpoint .pt file at '{model_arg}'.")


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


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, metric: str = "f1_score") -> Tuple[float, float]:
    """Finds optimal decision threshold strictly using validation predictions."""
    evaluator = ModelEvaluator()
    best_threshold = 0.5
    best_score = -1.0

    for th in np.arange(0.05, 0.96, 0.01):
        m = evaluator.evaluate(y_true, y_probs, split="validation", threshold=float(th))
        score = getattr(m, metric, m.f1_score)
        if score > best_score:
            best_score = score
            best_threshold = float(th)

    return round(best_threshold, 4), round(best_score, 4)


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Offline Model Evaluator Tool")
    parser.add_argument("--model-path", "--checkpoint", "--model-dir", required=True, help="Path to trained model .pt checkpoint or model directory")
    parser.add_argument("--manifest", default=None, help="Path to input features_manifest.csv or manifest.csv")
    parser.add_argument("--feature-dir", "--features-dir", default="data/features", help="Root directory containing feature dataset")
    parser.add_argument("--splits", default="validation,test,ood", help="Comma-separated list of splits to evaluate")
    parser.add_argument("--threshold-search", action="store_true", help="Tune decision threshold on validation split")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed decision threshold for positive (synthetic) class")
    parser.add_argument("--output-report", "--output", default=None, help="Output path for JSON evaluation report")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate model loading and split availability without full evaluation")

    args = parser.parse_args()

    print("\n==================================================")
    print("VoxGuard Model Evaluation Pipeline")
    print("==================================================")
    print(f"Model Path   : {args.model_path}")
    print(f"Feature Dir  : {args.feature_dir}")
    print(f"Target Splits: {args.splits}")
    print(f"Threshold    : {args.threshold}")
    print(f"Tune Thresh  : {args.threshold_search}")
    print(f"Dry Run      : {args.dry_run}")
    print("--------------------------------------------------")

    try:
        model_file = resolve_model_path(args.model_path)
        manifest_path = resolve_manifest_path(args.manifest, args.feature_dir)
        print(f"Loaded Checkpoint: {model_file}")
        print(f"Loaded Manifest  : {manifest_path}")

        # Load PyTorch model
        try:
            model = torch.load(model_file, map_location="cpu", weights_only=False)
        except TypeError:
            model = torch.load(model_file, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()

        config = TrainingConfig(BATCH_SIZE=args.batch_size)
        evaluator = ModelEvaluator()
        split_list = [s.strip().lower() for s in args.splits.split(",") if s.strip()]

        if args.dry_run:
            print("\n[DRY-RUN] Validated checkpoint loading and dataset paths successfully.")
            print("==================================================")
            print("[DRY-RUN] Model evaluation check complete.")
            print("==================================================\n")
            sys.exit(0)

        # Collect predictions per split
        split_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for split_name in split_list:
            try:
                ds = FeatureDataset(
                    manifest_path=manifest_path,
                    feature_dir=args.feature_dir,
                    split=split_name,
                    config=config,
                    validate_artifacts=True
                )
                if len(ds) == 0:
                    print(f"Split '{split_name}': 0 samples found. Skipping.")
                    continue

                loader = TrainingDataLoader(ds, config=config, shuffle=False)
                y_true = []
                y_probs = []

                with torch.no_grad():
                    for batch in loader:
                        if batch.batch_size == 0:
                            continue
                        log_mel = torch.from_numpy(batch.log_mel)
                        mfcc = torch.from_numpy(batch.mfcc)

                        logits = model(log_mel, mfcc)
                        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(synthetic)

                        y_true.extend(batch.labels.tolist())
                        y_probs.extend(probs.numpy().tolist())

                split_predictions[split_name] = (np.array(y_true), np.array(y_probs))
                print(f"Evaluated split '{split_name}': {len(y_true)} windowed samples.")

            except Exception as exc:
                print(f"Warning: Split '{split_name}' could not be evaluated: {str(exc)}")

        if not split_predictions:
            print("ERROR: No valid split predictions collected.")
            sys.exit(1)

        optimal_threshold = args.threshold
        tuning_note = None

        if args.threshold_search and "validation" in split_predictions:
            val_true, val_probs = split_predictions["validation"]
            if len(val_true) > 0 and len(np.unique(val_true)) > 1:
                optimal_threshold, best_f1 = find_optimal_threshold(val_true, val_probs, metric="f1_score")
                tuning_note = f"Threshold tuned on validation split to {optimal_threshold:.4f} (Validation F1={best_f1:.4f})."
                print(f"\n[THRESHOLD TUNING] {tuning_note}")

        # Compute metrics for each split using final decision threshold
        report_metrics: Dict[str, Dict] = {}

        print("\n==================================================")
        print("Evaluation Results Summary")
        print("==================================================")

        for split_name, (y_true, y_probs) in split_predictions.items():
            metrics = evaluator.evaluate(y_true, y_probs, split=split_name, threshold=optimal_threshold)
            report_metrics[split_name] = metrics.to_dict()

            print(f"\n--- Split: {split_name.upper()} (n={metrics.sample_count}, threshold={metrics.threshold}) ---")
            print(f"Accuracy         : {metrics.accuracy:.4f}")
            print(f"Balanced Accuracy: {metrics.balanced_accuracy:.4f}")
            print(f"Precision        : {metrics.precision:.4f}")
            print(f"Recall (Sens.)   : {metrics.recall:.4f}")
            print(f"Specificity      : {metrics.specificity:.4f}")
            print(f"F1-Score         : {metrics.f1_score:.4f}")
            print(f"ROC-AUC          : {metrics.roc_auc if metrics.roc_auc is not None else 'N/A'}")
            print(f"PR-AUC           : {metrics.pr_auc if metrics.pr_auc is not None else 'N/A'}")
            print(f"Confusion Matrix : TN={metrics.confusion_matrix[0][0]}, FP={metrics.confusion_matrix[0][1]}, FN={metrics.confusion_matrix[1][0]}, TP={metrics.confusion_matrix[1][1]}")

        # Output JSON report if path specified
        if args.output_report:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
            report_data = {
                "checkpoint_path": model_file,
                "manifest_path": manifest_path,
                "applied_threshold": optimal_threshold,
                "threshold_tuned": args.threshold_search,
                "tuning_note": tuning_note,
                "metrics": report_metrics
            }
            with open(args.output_report, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"\nSaved evaluation report to '{args.output_report}'")

        print("==================================================\n")
        sys.exit(0)

    except Exception as exc:
        print(f"\nERROR: Model evaluation failed: {str(exc)}")
        print("==================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
