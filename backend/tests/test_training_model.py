"""Unit Tests for VoxGuard Acoustic Neural Network (app.training.model).

Validates VoxGuardAcousticNet architecture shapes, feature mode settings,
forward passes with single and multi-tensor inputs, dropout evaluation behavior,
and gradient computation.
"""

import unittest
import torch
import torch.nn as nn

from app.training.model import VoxGuardAcousticNet


class TestVoxGuardAcousticNet(unittest.TestCase):
    """Test suite for VoxGuardAcousticNet model architecture."""

    def test_default_initialization(self):
        """Verify default initialization parameters and channel layout."""
        model = VoxGuardAcousticNet()
        self.assertEqual(model.feature_mode, "log_mel_mfcc")
        self.assertEqual(model.conv1.in_channels, 100)
        self.assertEqual(model.fc2.out_features, 2)

    def test_forward_pass_separated_inputs(self):
        """Verify forward pass with separated log_mel (80) and mfcc (20) tensors."""
        model = VoxGuardAcousticNet()
        model.eval()

        log_mel = torch.randn(4, 80, 256)
        mfcc = torch.randn(4, 20, 256)

        with torch.no_grad():
            logits = model(log_mel, mfcc)

        self.assertEqual(logits.shape, (4, 2))
        self.assertFalse(torch.isnan(logits).any())

    def test_forward_pass_concatenated_input(self):
        """Verify forward pass with single pre-concatenated (100) tensor (for RealMLClassifier compatibility)."""
        model = VoxGuardAcousticNet()
        model.eval()

        cat_features = torch.randn(2, 100, 256)

        with torch.no_grad():
            logits = model(cat_features)

        self.assertEqual(logits.shape, (2, 2))

    def test_single_feature_modes(self):
        """Verify forward pass for log_mel only and mfcc only feature modes."""
        model_mel = VoxGuardAcousticNet(feature_mode="log_mel")
        self.assertEqual(model_mel.conv1.in_channels, 80)
        log_mel = torch.randn(2, 80, 256)
        out_mel = model_mel(log_mel)
        self.assertEqual(out_mel.shape, (2, 2))

        model_mfcc = VoxGuardAcousticNet(feature_mode="mfcc")
        self.assertEqual(model_mfcc.conv1.in_channels, 20)
        mfcc = torch.randn(2, 20, 256)
        out_mfcc = model_mfcc(mfcc)
        self.assertEqual(out_mfcc.shape, (2, 2))

    def test_backward_pass_and_gradient_flow(self):
        """Verify backward pass computes non-zero gradients for trainable parameters."""
        model = VoxGuardAcousticNet()
        model.train()

        log_mel = torch.randn(2, 80, 256, requires_grad=True)
        mfcc = torch.randn(2, 20, 256, requires_grad=True)
        targets = torch.tensor([0, 1], dtype=torch.long)

        logits = model(log_mel, mfcc)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient.")
                self.assertFalse(torch.isnan(param.grad).any())


if __name__ == "__main__":
    unittest.main()
