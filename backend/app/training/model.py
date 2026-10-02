"""Neural Network Architecture Module for VoxGuard.

Defines VoxGuardAcousticNet: A lightweight, deterministic 1D Convolutional Neural Network
for binary audio classification (human=0, synthetic=1) operating on 256-frame Log-Mel & MFCC features.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxGuardAcousticNet(nn.Module):
    """VoxGuard 1D Convolutional Neural Network Classifier."""

    def __init__(
        self,
        n_mels: int = 80,
        n_mfcc: int = 20,
        feature_mode: str = "log_mel_mfcc",
        dropout: float = 0.2
    ):
        super().__init__()
        self.feature_mode = feature_mode.lower()
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        if self.feature_mode == "log_mel":
            in_channels = n_mels
        elif self.feature_mode == "mfcc":
            in_channels = n_mfcc
        else:
            in_channels = n_mels + n_mfcc  # Default 100 channels

        # 1D Convolutional Feature Extractor Blocks
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Classification Head
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 logits: [human_logit, synthetic_logit]

    def forward(
        self,
        log_mel: torch.Tensor,
        mfcc: Optional[torch.Tensor] = None,
        spectral_centroid: Optional[torch.Tensor] = None,
        spectral_rolloff: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            log_mel: Tensor of shape (B, 80, 256) or (B, 100, 256)
            mfcc: Optional Tensor of shape (B, 20, 256)

        Returns:
            Logits tensor of shape (B, 2).
        """
        if isinstance(log_mel, np.ndarray):
            log_mel = torch.from_numpy(log_mel)
        if mfcc is not None and isinstance(mfcc, np.ndarray):
            mfcc = torch.from_numpy(mfcc)

        if log_mel.dim() == 4 and log_mel.shape[1] == 1:
            log_mel = log_mel.squeeze(1)
        if mfcc is not None and mfcc.dim() == 4 and mfcc.shape[1] == 1:
            mfcc = mfcc.squeeze(1)

        if log_mel.dim() == 3 and log_mel.shape[1] == (self.n_mels + self.n_mfcc):
            x = log_mel
        elif self.feature_mode == "log_mel":
            x = log_mel
        elif self.feature_mode == "mfcc":
            x = mfcc if mfcc is not None else log_mel
        else:
            if mfcc is not None:
                x = torch.cat([log_mel, mfcc], dim=1)  # (B, 100, 256)
            else:
                x = log_mel

        if x.dim() == 3 and x.shape[1] < self.conv1.in_channels:
            pad_channels = self.conv1.in_channels - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad_channels))

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = self.pool(x).squeeze(-1)  # (B, 128)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        logits = self.fc2(x)  # (B, 2)
        return logits
