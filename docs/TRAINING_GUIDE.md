# VoxGuard Model Training & Evaluation Guide

This guide details the offline training, validation, threshold tuning, and evaluation procedures for VoxGuard audio deepfake detection models using pre-extracted feature artifacts (`.npz`).

---

## Overview

VoxGuard uses a deterministic 1D Convolutional Neural Network (`VoxGuardAcousticNet`) operating on 256-frame Log-Mel Spectrogram and MFCC feature windows. The training infrastructure provides reproducible offline training, early stopping, class-weighted cross-entropy loss, decision threshold tuning, and SHA-256 manifest creation.

> [!IMPORTANT]
> The production FastAPI pipeline remains in `ANALYSIS_ONLY` mode by default. Model activation requires explicit model configuration with verified SHA-256 checksums in `manifest.json`.

---

## Training Infrastructure Architecture

1. **Feature Windowing & DataLoader** (`app.training.windowing`, `app.training.sampler`):
   - Slices variable-length audio features into fixed 256-frame segments with stride 128 and zero-padding.
   - Preserves speaker isolation across train, validation, test, and OOD splits without cross-split leakage.

2. **Neural Network Model** (`app.training.model`):
   - `VoxGuardAcousticNet`: 1D CNN with 100 input channels (80 Log-Mel + 20 MFCC), 3 Conv1D blocks (64, 128, 128 channels), BatchNorm1D, ReLU, Dropout, AdaptiveAvgPool1D, and a 2-class linear output head.

3. **Trainer Service** (`app.training.trainer`):
   - Executes multi-epoch training with class-weighted loss, AdamW optimizer, gradient clipping, early stopping on validation ROC-AUC / Accuracy, saving `best_model.pt` and `last_model.pt`.
   - Generates `manifest.json` containing SHA-256 hash checksums.

4. **Model Evaluator** (`app.evaluation.model_evaluator`):
   - Computes Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrices `[[TN, FP], [FN, TP]]` adhering to the standard convention (`human=0`, `synthetic=1`).

---

## CLI Command Usage

### 1. Inspect Feature Datasets
Inspect window count, class distribution, speaker counts, and tensor shapes before training:
```bash
python scripts/inspect_training_data.py --manifest data/features/features_manifest.csv --split train
```

### 2. Train Model
Train a new model checkpoint and generate manifest JSON:
```bash
python scripts/train_model.py \
  --manifest data/features/features_manifest.csv \
  --feature-dir data/features \
  --output-dir models/trained \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --patience 5 \
  --seed 42
```

To perform a quick pipeline validation without training:
```bash
python scripts/train_model.py --manifest data/features/features_manifest.csv --dry-run
```

### 3. Evaluate Model & Tune Threshold
Evaluate a trained model across validation, test, and OOD splits, search for optimal decision threshold on validation set, and output a JSON evaluation report:
```bash
python scripts/evaluate_model.py \
  --model-path models/trained/best_model.pt \
  --manifest data/features/features_manifest.csv \
  --feature-dir data/features \
  --splits validation,test,ood \
  --threshold-search \
  --output-report reports/evaluation_summary.json
```

---

## Security & Verification

- **SHA-256 Verification**: Models loaded into `RealMLClassifier` are audited against `manifest.json` SHA-256 checksums before inference.
- **Zero Web Downloads**: All training operates entirely offline on local feature artifacts without external network requests.
