"""Unit Tests for VoxGuard Model Evaluator (app.evaluation.model_evaluator).

Validates binary evaluation metrics computation (Accuracy, Balanced Accuracy,
Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix)
strictly enforcing human=0 (negative) and synthetic=1 (positive) convention.
"""

import unittest
import numpy as np

from app.evaluation.model_evaluator import ModelEvaluator, EvaluationMetrics


class TestModelEvaluator(unittest.TestCase):
    """Test suite for ModelEvaluator component."""

    def setUp(self):
        self.evaluator = ModelEvaluator()

    def test_perfect_classification(self):
        """Verify metric outputs for 100% correct binary predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_probs = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])

        metrics = self.evaluator.evaluate(y_true, y_probs, split="validation", threshold=0.5)

        self.assertEqual(metrics.sample_count, 6)
        self.assertEqual(metrics.human_count, 3)
        self.assertEqual(metrics.synthetic_count, 3)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.balanced_accuracy, 1.0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.specificity, 1.0)
        self.assertEqual(metrics.f1_score, 1.0)
        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertEqual(metrics.pr_auc, 1.0)
        self.assertEqual(metrics.confusion_matrix, [[3, 0], [0, 3]])

    def test_known_imperfect_classification(self):
        """Verify metric outputs and confusion matrix for mixed predictions.

        y_true: [0, 0, 0, 0, 1, 1, 1, 1] (4 human, 4 synthetic)
        y_probs: [0.1, 0.2, 0.7, 0.3, 0.8, 0.9, 0.4, 0.6]
        At threshold 0.5:
        human (0): 0.1(TN), 0.2(TN), 0.7(FP), 0.3(TN) -> TN=3, FP=1
        synth (1): 0.8(TP), 0.9(TP), 0.4(FN), 0.6(TP) -> TP=3, FN=1
        """
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.9, 0.4, 0.6])

        metrics = self.evaluator.evaluate(y_true, y_probs, split="test", threshold=0.5)

        self.assertEqual(metrics.confusion_matrix, [[3, 1], [1, 3]])
        self.assertEqual(metrics.accuracy, 6 / 8)  # 0.75
        self.assertEqual(metrics.recall, 3 / 4)    # 0.75
        self.assertEqual(metrics.specificity, 3 / 4) # 0.75
        self.assertEqual(metrics.precision, 3 / 4) # 0.75
        self.assertEqual(metrics.f1_score, 0.75)

    def test_empty_split_handling(self):
        """Verify graceful handling when evaluating empty prediction arrays."""
        y_true = np.array([])
        y_probs = np.array([])

        metrics = self.evaluator.evaluate(y_true, y_probs, split="ood")

        self.assertEqual(metrics.sample_count, 0)
        self.assertEqual(metrics.accuracy, 0.0)
        self.assertIsNone(metrics.roc_auc)
        self.assertIsNone(metrics.pr_auc)
        self.assertIn("0 samples", metrics.notes)

    def test_single_class_split_handling(self):
        """Verify ROC-AUC and PR-AUC are set to None when split contains only one class."""
        y_true = np.array([0, 0, 0, 0])
        y_probs = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = self.evaluator.evaluate(y_true, y_probs, split="validation")

        self.assertEqual(metrics.human_count, 4)
        self.assertEqual(metrics.synthetic_count, 0)
        self.assertIsNone(metrics.roc_auc)
        self.assertIsNone(metrics.pr_auc)
        self.assertIn("single class", metrics.notes)

    def test_custom_threshold(self):
        """Verify custom threshold alters binary classification boundaries."""
        y_true = np.array([0, 1])
        y_probs = np.array([0.6, 0.8])

        # At default threshold 0.5: both predicted 1 (synthetic) -> TN=0, FP=1, FN=0, TP=1 -> Acc 0.5
        m_default = self.evaluator.evaluate(y_true, y_probs, threshold=0.5)
        self.assertEqual(m_default.confusion_matrix, [[0, 1], [0, 1]])

        # At threshold 0.7: 0.6 is 0, 0.8 is 1 -> TN=1, FP=0, FN=0, TP=1 -> Acc 1.0
        m_custom = self.evaluator.evaluate(y_true, y_probs, threshold=0.7)
        self.assertEqual(m_custom.confusion_matrix, [[1, 0], [0, 1]])
        self.assertEqual(m_custom.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
