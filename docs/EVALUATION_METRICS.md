# VoxGuard Model Evaluation Protocols & Metrics Standard

This document establishes the mandatory standards for evaluating deepfake audio detection models in VoxGuard.

> [!IMPORTANT]
> Model raw probability outputs MUST NOT be presented as verified real-world accuracy without empirical evaluation data. A model CANNOT be considered READY for real-world claims merely because inference code runs without errors. Unverified or fabricated accuracy numbers are strictly prohibited across the VoxGuard codebase and platform UI.

---

## 1. Required Evaluation Metrics

Every model integrated into VoxGuard must be evaluated against standard classification and forensic metrics prior to `READY` status deployment:

| Metric | Definition & Purpose | Formula / Standard |
| :--- | :--- | :--- |
| **Accuracy** | Overall proportion of correct predictions across all classes. | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **Precision** | Ratio of correctly identified synthetic samples to total predicted synthetic samples. | $\frac{TP}{TP + FP}$ |
| **Recall (TPR)** | Ratio of correctly identified synthetic samples to actual synthetic samples. | $\frac{TP}{TP + FN}$ |
| **F1-Score** | Harmonic mean of Precision and Recall. | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **ROC-AUC** | Area Under Receiver Operating Characteristic Curve across variable decision thresholds. | Threshold-agnostic discrimination capability |
| **EER (Equal Error Rate)** | Point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR). | Standard biometrics & audio deepfake metric |
| **FPR (False Positive Rate)** | Genuine human speech misclassified as synthetic. | $\frac{FP}{FP + TN}$ |
| **FNR (False Negative Rate)** | Synthetic/AI speech misclassified as genuine human speech. | $\frac{FN}{TP + FN}$ |
| **Confusion Matrix** | Full 2x2 matrix of TP, FP, TN, FN counts. | Detailed breakdown per dataset split |

---

## 2. Model Readiness & Evaluation Protocol Requirements

A model adapter in VoxGuard reaches `status = "ready"` ONLY when the following empirical protocols are satisfied and documented:

1. **Speaker-Disjoint Dataset Partitioning**:
   - Zero speaker overlap between training, validation, and test sets.
   - Partitioning MUST be executed strictly by **Speaker ID**, not random utterance sampling.
2. **Dataset Provenance & Class Balance**:
   - Document source datasets (e.g. ASVspoof 2019/2021 LA/DF, WaveFake, LibriSpeech).
   - Report exact class balance ratios ($N_{\text{genuine}}$ vs. $N_{\text{synthetic}}$).
3. **Out-of-Distribution (OOD) & Robustness Evaluation**:
   - Benchmark performance on unseen neural speech generators (e.g. ElevenLabs, VALL-E, Bark) not present during training.
   - Benchmark under audio compression artifacts (MP3 bitrates, Opus/OGG codecs, cell network channel mismatch).
4. **No Metric Fabrication**:
   - Unknown or unmeasured metrics must be represented as `null` or `"not_configured"`. Inventing arbitrary performance percentages is strictly prohibited.

---

## 3. Distinction: Acoustic Indicators vs. ML Predictions

VoxGuard strictly enforces a conceptual and architectural boundary between signal-level acoustic indicators and machine-learning predictions:

1. **Acoustic Indicators (`status = "analysis_only"`)**:
   - Descriptive measurements (spectral centroid mean/std, spectral rolloff 85% cutoff, MFCC variance, log-mel energy distribution, temporal variation).
   - Produced by `BaselineClassifier` without trained ML model weights.
   - **Policy**: Probabilities are strictly set to `None`. No heuristic measurement may claim to output a synthetic probability (e.g. *"87% AI generated"* is forbidden from heuristics).

2. **ML Model Predictions (`status = "ready"`)**:
   - Probabilities $P(\text{synthetic})$ and $P(\text{human})$ generated strictly by trained classifiers.
   - Requires documented dataset provenance, zero-speaker-overlap splits, and benchmark results before deployment.

---

## 4. Supplying a Future Model Checkpoint & Manifest Manually

To integrate an evaluated deepfake detection model in the future:

1. **Place Model Weights**: Copy trained `.pt` or `.onnx` weights file to the `models/` directory (e.g. `models/voxguard_resnet18_v1.pt`).
2. **Create Model Manifest**: Create a `models/manifest.json` file specifying model metadata, SHA-256 checksum, sample rate, class labels, and evaluated benchmark metrics:
   ```json
   {
     "model_name": "VoxGuard Neural Deepfake Classifier",
     "model_version": "1.0.0",
     "model_type": "pytorch",
     "architecture": "resnet18_logmel",
     "expected_sample_rate": 16000,
     "class_names": ["human", "synthetic"],
     "checkpoint_path": "models/voxguard_resnet18_v1.pt",
     "checksum": "<sha256_hex_digest>",
     "training_dataset": "ASVspoof 2019 LA",
     "speaker_overlap_policy": "speaker_disjoint_verified"
   }
   ```
3. **Set Environment Configuration**: Configure `.env`:
   ```env
   MODEL_ENABLED=True
   MODEL_PATH=models/voxguard_resnet18_v1.pt
   ```
4. **Automatic Validation**: Upon startup, `ModelValidator` will passively check manifest integrity, SHA-256 checksum, and tensor contract compatibility before activating `status = "ready"`.

