# VoxGuard Training Pipeline Infrastructure & Dataset Guide

## 1. Architecture Overview

VoxGuard features an offline-safe, deterministic voice-forensics training pipeline designed to consume pre-extracted `.npz` acoustic feature files and produce reproducible model checkpoints (`best_model.pt`, `last_model.pt`, and `manifest.json`).

```
+-----------------------------------------------------------------------------------+
|                            VoxGuard Training Pipeline                             |
+-----------------------------------------------------------------------------------+
|  [Manifest CSV] -> [FeatureDataset] -> [FeatureWindowing] -> [TrainingDataLoader] |
|                                                                    |              |
|                                                                    v              |
|  [Integrity Audit] <- [Dry-Run Artifacts] <- [Trainer Engine] <- [VoxGuardNet]    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Components & Detailed Cases Explained

### Case 1: Dataset Manifest & Feature Loading (`app.training.dataset.FeatureDataset`)
- **Purpose**: Safely loads pre-extracted feature artifacts referenced in a CSV manifest.
- **Rules & Constraints**:
  - Rejects directory traversal paths containing `..`.
  - Verifies presence of mandatory feature arrays: `log_mel`, `mfcc`, `spectral_centroid`, `spectral_rolloff`.
  - Validates numerical finiteness (`NaN` or `Inf` checks).
  - Enforces `float32` tensor data types.

### Case 2: Feature Windowing (`app.training.windowing.FeatureWindowing`)
- **Purpose**: Converts variable-length audio feature matrices into uniform `256`-frame feature windows with stride `128`.
- **Handling Short & Long Sequences**:
  - **Sequences < 256 frames**: Padded with zeros on the right up to 256 frames.
  - **Sequences = 256 frames**: Preserved as a single window.
  - **Sequences > 256 frames**: Windowed using a sliding window with stride `128` (50% overlap).
- **Metadata Preservation**: Each window preserves original `sample_id`, `speaker_id`, `source`, `source_id`, `split`, and integer `label` (0 = Human, 1 = Synthetic).

### Case 3: Dataset Risk Assessment Cases (`app.training.statistics.DatasetStatisticsCalculator`)
The dataset statistics engine audits split data distributions and flags potential operational risks:

1. **Severe / Class Imbalance Case**:
   - Triggered if a split contains zero samples for one class OR the percentage gap between human and synthetic samples exceeds `20%`.
   - *Forensic Rationale*: Imbalanced datasets bias neural classifiers toward the dominant class.

2. **Dominant Speaker Risk Case**:
   - Triggered if any single `speaker_id` accounts for more than `30%` of all windowed samples in a multi-speaker dataset.
   - *Forensic Rationale*: Prevents the model from learning speaker identity instead of synthetic acoustic artifacts.

3. **Dominant Source / Generator Risk Case**:
   - Triggered if a single generator source (e.g. ElevenLabs, Bark) accounts for more than `50%` of total windows across multi-source datasets.
   - *Forensic Rationale*: Prevents overfitting to a specific vocoder or voice synthesis engine.

4. **Suspicious Split Size Case**:
   - Triggered if a dataset split contains fewer than `5` total windowed samples.
   - *Forensic Rationale*: Flags insufficient evaluation data.

### Case 4: Deterministic Batching & Dictionary Interface (`app.training.sampler.TrainingDataLoader`)
- **Reproducibility**: Guarantees bit-exact batch ordering using deterministic seed configuration (`SEED = 42`).
- **Collated Batch Structure**:
  - `log_mel`: `(B, 80, 256)` float32 array
  - `mfcc`: `(B, 20, 256)` float32 array
  - `spectral_centroid`: `(B, 1, 256)` float32 array
  - `spectral_rolloff`: `(B, 1, 256)` float32 array
  - `labels`: `(B,)` int64 array
- **Dual Access Interface**: Supports both attribute access (`batch.log_mel`) and dictionary subscription (`batch["log_mel"]`, `batch["speaker_ids"]`, `batch["sample_ids"]`).

### Case 5: Speaker Disjointness & Zero Leakage Validation (`app.training.splitter`)
- Ensures strict separation of speaker identities across `train`, `validation`, `test`, and `ood` (Out-of-Domain) splits.
- Zero speaker overlap prevents data contamination and over-optimistic evaluation metrics.

---

## 3. CLI Usage & Verification

### Running Feature Extraction / Generation
```bash
python scripts/build_features.py --manifest datasets/manifest.example.csv --output data/features
```

### Running Pipeline Verification (`--dry-run`)
```bash
python scripts/train_model.py --manifest data/features/manifest.csv --feature-dir data/features --output models/training --dry-run
```

Executing `--dry-run` performs data loading verification, single-step forward/backward gradient checks, and outputs 5 structural JSON audit artifacts to `--output`:
1. `dataset_summary.json`: Detailed class balance, window counts, sequence length statistics, and risk flags.
2. `training_config.json`: Hyperparameters, seeds, window sizes, batch sizes, and learning rates.
3. `feature_schema.json`: Expected tensor shapes and data types for log-mel, MFCC, and spectral metrics.
4. `split_summary.json`: Window and sample breakdowns across train, validation, and test splits.
5. `integrity.json`: Speaker disjointness status and numerical finiteness audit pass/fail checks.

---

## 4. Production Safety Safeguards

1. **ANALYSIS_ONLY Default Mode**:
   - Unless an evaluated, SHA-256 verified model checkpoint is explicitly registered in production, the VoxGuard API runs in `ANALYSIS_ONLY` safety mode.
   - In `ANALYSIS_ONLY` mode, classification probabilities remain `null`, and forensic evaluation relies strictly on acoustic metric cards and transparent evidence indicators.

2. **No Downloads Guarantee**:
   - The training pipeline operate offline using local feature datasets without external network requests or automated weight downloads.
