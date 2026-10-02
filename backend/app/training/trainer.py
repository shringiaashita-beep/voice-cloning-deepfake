"""Trainer Module for VoxGuard Model Training Infrastructure.

Implements reproducible PyTorch training loop for VoxGuardAcousticNet, supporting:
- Class-weighted CrossEntropyLoss for dataset balance
- AdamW optimizer with customizable learning rate and weight decay
- Gradient clipping for training stability
- Early stopping based on validation ROC-AUC / Accuracy
- Model checkpoint saving (best_model.pt, last_model.pt)
- Automatic SHA-256 manifest generation matching VoxGuard ModelManifest schema
"""

from dataclasses import dataclass, field, asdict
import os
import time
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from app.training.config import TrainingConfig
from app.training.model import VoxGuardAcousticNet
from app.training.sampler import TrainingDataLoader
from app.evaluation.model_evaluator import ModelEvaluator, EvaluationMetrics
from app.models.model_manifest import ModelManifest

logger = logging.getLogger("voxguard.training.trainer")


@dataclass
class EpochMetrics:
    """Metrics recorded for a single epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_roc_auc: Optional[float]
    val_f1_score: float
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts epoch metrics to dict."""
        return asdict(self)


@dataclass
class TrainingSummary:
    """Summary output after completing training execution."""
    model_name: str
    total_epochs: int
    stopped_early: bool
    best_epoch: int
    best_val_roc_auc: Optional[float]
    best_val_accuracy: float
    history: List[Dict[str, Any]]
    checkpoint_dir: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    manifest_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts summary to dictionary."""
        return asdict(self)


class Trainer:
    """Engine for training and evaluating VoxGuard neural network models."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[TrainingConfig] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        max_grad_norm: float = 1.0,
        patience: int = 5,
        device: str = "cpu",
        output_dir: str = "models/trained",
        class_weighting: bool = True
    ):
        """Initializes VoxGuard Trainer.

        Args:
            model: PyTorch module (defaults to VoxGuardAcousticNet).
            config: TrainingConfig object.
            learning_rate: Optimizer learning rate.
            weight_decay: Weight decay L2 regularization factor.
            max_grad_norm: Gradient clipping max norm.
            patience: Early stopping patience (number of non-improving epochs).
            device: Computation device ("cpu" or "cuda").
            output_dir: Path to directory where checkpoints and manifests are saved.
            class_weighting: Whether to apply inverse class frequency loss weighting.
        """
        self.config = config or TrainingConfig()
        use_cuda = torch.cuda.is_available() and device.startswith("cuda")
        self.device = torch.device(device if use_cuda else "cpu")

        self.model = (model or VoxGuardAcousticNet()).to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.output_dir = output_dir
        self.class_weighting = class_weighting

        self.evaluator = ModelEvaluator()

    def train(
        self,
        train_loader: TrainingDataLoader,
        val_loader: Optional[TrainingDataLoader] = None,
        epochs: int = 10
    ) -> TrainingSummary:
        """Executes full training loop over specified number of epochs.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            epochs: Maximum number of epochs to train.

        Returns:
            TrainingSummary containing complete history and saved artifact paths.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Compute class weights if requested
        weights_tensor = None
        if self.class_weighting and len(train_loader.dataset) > 0:
            human_count = sum(1 for s in train_loader.dataset if s.label == 0)
            synth_count = sum(1 for s in train_loader.dataset if s.label == 1)
            total = human_count + synth_count
            if human_count > 0 and synth_count > 0:
                w_human = total / (2.0 * human_count)
                w_synth = total / (2.0 * synth_count)
                weights_tensor = torch.tensor([w_human, w_synth], dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        history: List[EpochMetrics] = []
        best_val_score = -1.0
        best_epoch = 0
        patience_counter = 0
        stopped_early = False

        best_checkpoint_path = os.path.join(self.output_dir, "best_model.pt")
        last_checkpoint_path = os.path.join(self.output_dir, "last_model.pt")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # --- Training Pass ---
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if batch.batch_size == 0:
                    continue

                log_mel = torch.from_numpy(batch.log_mel).to(self.device)
                mfcc = torch.from_numpy(batch.mfcc).to(self.device)
                labels = torch.from_numpy(batch.labels).to(self.device)

                optimizer.zero_grad()
                logits = self.model(log_mel, mfcc)
                loss = criterion(logits, labels)

                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

                train_loss_sum += loss.item() * batch.batch_size
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += batch.batch_size

            train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
            train_acc = train_correct / train_total if train_total > 0 else 0.0

            # --- Validation Pass ---
            val_loss = 0.0
            val_acc = 0.0
            val_roc_auc = None
            val_f1 = 0.0

            if val_loader is not None and len(val_loader.dataset) > 0:
                self.model.eval()
                val_loss_sum = 0.0
                all_val_labels = []
                all_val_probs = []
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        if batch.batch_size == 0:
                            continue

                        log_mel = torch.from_numpy(batch.log_mel).to(self.device)
                        mfcc = torch.from_numpy(batch.mfcc).to(self.device)
                        labels = torch.from_numpy(batch.labels).to(self.device)

                        logits = self.model(log_mel, mfcc)
                        loss = criterion(logits, labels)

                        val_loss_sum += loss.item() * batch.batch_size
                        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(synthetic)

                        all_val_labels.extend(batch.labels.tolist())
                        all_val_probs.extend(probs.cpu().numpy().tolist())
                        val_total += batch.batch_size

                val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
                val_metrics = self.evaluator.evaluate(
                    np.array(all_val_labels),
                    np.array(all_val_probs),
                    split="validation"
                )

                val_acc = val_metrics.accuracy
                val_roc_auc = val_metrics.roc_auc
                val_f1 = val_metrics.f1_score

                # Early stopping score: prefer ROC-AUC, fallback to accuracy
                current_score = val_roc_auc if val_roc_auc is not None else val_acc

                if current_score > best_val_score:
                    best_val_score = current_score
                    best_epoch = epoch
                    self._save_checkpoint(best_checkpoint_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        stopped_early = True

            else:
                # No validation set provided: track train accuracy for best check
                if train_acc > best_val_score:
                    best_val_score = train_acc
                    best_epoch = epoch
                    self._save_checkpoint(best_checkpoint_path)

            elapsed = time.time() - start_time
            epoch_metric = EpochMetrics(
                epoch=epoch,
                train_loss=round(train_loss, 4),
                train_accuracy=round(train_acc, 4),
                val_loss=round(val_loss, 4),
                val_accuracy=round(val_acc, 4),
                val_roc_auc=round(val_roc_auc, 4) if val_roc_auc is not None else None,
                val_f1_score=round(val_f1, 4),
                duration_seconds=round(elapsed, 2)
            )
            history.append(epoch_metric)

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f, Val AUC: %s (%.2fs)",
                epoch, epochs, train_loss, train_acc, val_loss, val_acc,
                str(round(val_roc_auc, 4)) if val_roc_auc is not None else "N/A", elapsed
            )

            if stopped_early:
                logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, self.patience)
                break

        # Always save last checkpoint
        self._save_checkpoint(last_checkpoint_path)

        # Ensure best_model.pt exists (copy from last if no validation upgrade occurred)
        if not os.path.exists(best_checkpoint_path) and os.path.exists(last_checkpoint_path):
            self._save_checkpoint(best_checkpoint_path)
            best_epoch = len(history)

        # Build & save ModelManifest JSON
        manifest_path = self._generate_manifest(
            checkpoint_path=best_checkpoint_path,
            best_epoch=best_epoch,
            best_score=best_val_score,
            total_epochs=len(history)
        )

        return TrainingSummary(
            model_name="VoxGuardAcousticNet",
            total_epochs=len(history),
            stopped_early=stopped_early,
            best_epoch=best_epoch,
            best_val_roc_auc=history[best_epoch - 1].val_roc_auc if best_epoch > 0 and best_epoch <= len(history) else None,
            best_val_accuracy=history[best_epoch - 1].val_accuracy if best_epoch > 0 and best_epoch <= len(history) else (history[-1].train_accuracy if history else 0.0),
            history=[h.to_dict() for h in history],
            checkpoint_dir=self.output_dir,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            manifest_path=manifest_path
        )

    def _save_checkpoint(self, filepath: str) -> None:
        """Saves model PyTorch object to specified file."""
        torch.save(self.model, filepath)

    def _generate_manifest(
        self,
        checkpoint_path: str,
        best_epoch: int,
        best_score: float,
        total_epochs: int
    ) -> str:
        """Computes SHA-256 digest and writes ModelManifest JSON to output directory."""
        sha256_hash = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        checksum = sha256_hash.hexdigest()

        weights_filename = os.path.basename(checkpoint_path)

        manifest = ModelManifest(
            model_name="VoxGuardAcousticNet",
            model_version="1.0.0",
            model_type="pytorch",
            architecture="VoxGuardAcousticNet",
            framework="pytorch",
            checkpoint_path=checkpoint_path,
            checksum=checksum,
            class_names=["human", "synthetic"],
            expected_input={
                "sample_rate": 16000,
                "channels": 1,
                "feature_type": "log_mel_mfcc",
                "freq_bins": 80,
                "mfcc_bins": 20,
                "input_shape": [100, 256]
            },
            metrics={
                "training_metadata": {
                    "total_epochs": total_epochs,
                    "best_epoch": best_epoch,
                    "best_score": round(best_score, 4),
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.config.BATCH_SIZE,
                    "random_seed": self.config.RANDOM_SEED
                }
            }
        )

        manifest_dict = manifest.to_dict()
        manifest_dict["sha256"] = checksum
        manifest_dict["labels"] = ["human", "synthetic"]

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2)

        return manifest_path
